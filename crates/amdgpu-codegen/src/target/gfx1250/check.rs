//! gfx1250 exact-mode verification (docs/gfx1250-plan.md): the author's waits and padding are checked, never changed. A forward dataflow
//! over the CFG tracks every outstanding operation per counter with the minimum number of same-counter operations issued after it on any
//! path (`younger`); a wait for `n` completes the operations with `younger >= n` on an in-order counter, every one only with `n = 0` on an
//! out-of-order one (or when more than one event class is pending on it). An instruction that depends on an operation still outstanding
//! is a violation.
//!
//! Counters (target-gfx1250.md section 4; LLVM SIInsertWaitcnts): LOADcnt (vector loads, atomics with return), STOREcnt, DScnt (LDS loads),
//! KMcnt (scalar loads, out of order), XCNT (translations of vector (in order) and scalar (out of order) memory operations: their address /
//! data registers may not be overwritten before), and, only while the wave's scheduling mode is the relaxed one (`SetSchedMode 2`; LLVM's
//! "expert scheduling mode 2"), the two dependencies the hardware then leaves to the program: `va_vdst` (a non-VALU instruction reading or
//! writing a VGPR an outstanding VALU writes, or writing one it reads; four VALU classes - XDL, TRANS, DP-MACC, other - complete out of order
//! with each other; a VOP3PX2 WMMA counts twice) and `vm_vsrc` (any instruction overwriting a VGPR an outstanding vector-memory / LDS
//! operation still reads; FLAT, LDS and other VMEM classes out of order with each other), waited with `s_wait_alu` (depctr va_vdst [15:12],
//! vm_vsrc [4:2]). A LOADcnt / STOREcnt / DScnt completion also completes that operation's translation and source reads.
//! The software hazards (hazards.rs: TRANS / WMMA co-execution, TDM drain) must already be satisfied by the author's instructions.

use crate::mir::*;
use crate::regalloc::Alloc;
use crate::sched::effects::{effects, Space, SCC, VCC};
use std::collections::BTreeMap;

#[derive(Clone, Copy, PartialEq, Eq, Debug, PartialOrd, Ord)]
enum Ctr {
    Load,
    Store,
    Ds,
    Km,
    X,
    VaVdst,
    VmVsrc,
}

/// One outstanding operation: (counter, the instruction that issued it). Value: (younger, event class, registers it writes, registers it
/// reads).
type Key = (Ctr, (usize, usize), u8);
#[derive(Clone, Debug, PartialEq)]
struct Item {
    younger: u32,
    class: u8,
    dst: Vec<u32>,
    src: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Default)]
struct State {
    items: BTreeMap<Key, Item>,
    /// the wave's scheduling mode (None: unknown, treated as relaxed)
    mode: Option<u8>,
}

const VGPR_LIMIT: u32 = 0x1_0000;

fn keys(m: &MFunction, a: &Alloc, slots: &[u32]) -> Vec<u32> {
    let mut out = Vec::new();
    for &s in slots {
        if s == VCC {
            out.push(0x3_0000);
            continue;
        }
        if s == SCC {
            continue;
        }
        let (t, off) = (s / 16, s % 16);
        let Some(&p) = a.phys.get(t as usize) else { continue };
        let c = m.vreg_class(t);
        out.push(if c.is_scalar() { 0x2_0000 + p + off } else { p + off });
    }
    out
}

fn is_valu(op: MOp) -> bool {
    crate::op_attrs::attrs(op).mnemonic.starts_with("v_") && !matches!(op, MOp::VReadPhys)
}

/// The VALU event class (va_vdst ordering): 0 other, 1 TRANS, 2 XDL (WMMA).
fn valu_class(op: MOp) -> u8 {
    if super::hazards::wmma_windows(op).is_some() {
        2
    } else if crate::op_attrs::attrs(op).transcendental {
        1
    } else {
        0
    }
}

fn ctr_name(c: Ctr) -> &'static str {
    match c {
        Ctr::Load => "s_wait_loadcnt",
        Ctr::Store => "s_wait_storecnt",
        Ctr::Ds => "s_wait_dscnt",
        Ctr::Km => "s_wait_kmcnt",
        Ctr::X => "s_wait_xcnt",
        Ctr::VaVdst => "s_wait_alu va_vdst",
        Ctr::VmVsrc => "s_wait_alu vm_vsrc",
    }
}

/// A wait for `n` on counter `c`.
fn wait(st: &mut State, c: Ctr, n: u32) {
    let classes: std::collections::BTreeSet<u8> = st.items.iter().filter(|(k, _)| k.0 == c).map(|(_, v)| v.class).collect();
    let in_order = !matches!(c, Ctr::Km) && classes.len() <= 1 && !(c == Ctr::X && st.items.iter().any(|(k, _)| k.0 == Ctr::X && k.2 == 1));
    let done: Vec<Key> = st.items.iter().filter(|(k, v)| k.0 == c && (n == 0 || (in_order && v.younger >= n))).map(|(k, _)| *k).collect();
    for k in done {
        st.items.remove(&k);
        // a completed load / store / LDS operation has also finished its translation and its source reads
        if matches!(c, Ctr::Load | Ctr::Store | Ctr::Ds) {
            st.items.retain(|kk, _| !(kk.1 == k.1 && matches!(kk.0, Ctr::X | Ctr::VmVsrc)));
        }
    }
}

fn issue(st: &mut State, c: Ctr, at: (usize, usize), class: u8, dst: Vec<u32>, src: Vec<u32>, weight: u32) {
    for (k, v) in st.items.iter_mut() {
        if k.0 == c {
            v.younger = (v.younger + weight).min(64);
        }
    }
    let k = (c, at, class);
    let it = Item { younger: 0, class, dst, src };
    st.items.insert(k, it);
}

fn merge(a: &State, b: &State) -> State {
    let mut r = a.clone();
    for (k, v) in &b.items {
        match r.items.get_mut(k) {
            Some(x) => x.younger = x.younger.min(v.younger),
            None => {
                r.items.insert(*k, v.clone());
            }
        }
    }
    r.mode = match (a.mode, b.mode) {
        (Some(x), Some(y)) if x == y => Some(x),
        _ => None,
    };
    r
}

/// The violations of the author's waits (one message per missing wait: block, instruction, counter, the operation it depends on).
pub fn verify(m: &MFunction, a: &Alloc) -> Vec<(u32, String)> {
    let n = m.blocks.len();
    let idx: std::collections::HashMap<u32, usize> = m.blocks.iter().enumerate().map(|(i, b)| (b.label, i)).collect();
    let succ: Vec<Vec<usize>> = m
        .blocks
        .iter()
        .map(|b| match &b.term {
            Term::Goto(t) => vec![idx[t]],
            Term::CondGoto { target, fall, .. } => vec![idx[target], idx[fall]],
            Term::Endpgm => vec![],
        })
        .collect();
    let mut entry: Vec<Option<State>> = vec![None; n];
    entry[0] = Some(State { items: BTreeMap::new(), mode: Some(0) });
    let mut errs: BTreeMap<(usize, usize, Ctr), (u32, String)> = BTreeMap::new();
    for _round in 0..200 {
        let mut changed = false;
        for bi in 0..n {
            let Some(mut st) = entry[bi].clone() else { continue };
            for (ii, inst) in m.blocks[bi].insts.iter().enumerate() {
                // the author's waits
                match inst.op {
                    MOp::SWaitCtr => {
                        let c = match inst.imm >> 8 {
                            0 => Some(Ctr::Load),
                            1 => Some(Ctr::Store),
                            2 => Some(Ctr::Ds),
                            3 => Some(Ctr::Km),
                            4 => Some(Ctr::X),
                            _ => None,
                        };
                        if let Some(c) = c {
                            wait(&mut st, c, (inst.imm & 0xff) as u32);
                        }
                        continue;
                    }
                    MOp::SWaitAlu => {
                        let d = inst.imm as u32;
                        wait(&mut st, Ctr::VaVdst, (d >> 12) & 0xf);
                        wait(&mut st, Ctr::VmVsrc, (d >> 2) & 0x7);
                        continue;
                    }
                    MOp::SSetSchedMode => {
                        st.mode = Some(inst.imm as u8);
                        continue;
                    }
                    _ => {}
                }
                let e = effects(m, inst);
                let reads = keys(m, a, &e.reads);
                let writes = keys(m, a, &e.writes);
                let valu = is_valu(inst.op);
                let relaxed = st.mode != Some(0);
                // dependencies on outstanding operations
                for (k, v) in &st.items {
                    let hit = |xs: &[u32], ys: &[u32]| xs.iter().any(|x| ys.contains(x));
                    let dep = match k.0 {
                        Ctr::Load | Ctr::Ds | Ctr::Km => hit(&reads, &v.dst) || hit(&writes, &v.dst),
                        Ctr::Store => false,
                        Ctr::X => hit(&writes, &v.src),
                        Ctr::VaVdst => relaxed && !valu && (hit(&reads, &v.dst) || hit(&writes, &v.dst) || hit(&writes, &v.src)),
                        Ctr::VmVsrc => relaxed && hit(&writes, &v.src),
                    };
                    if dep {
                        let (pb, pi) = k.1;
                        errs.entry((bi, ii, k.0)).or_insert((
                            inst.stmt,
                            format!(
                                "block {} instruction {ii} (`{}`): needs {} for the operation at block {} instruction {pi} (`{}`){}",
                                m.blocks[bi].label,
                                crate::op_attrs::attrs(inst.op).mnemonic,
                                ctr_name(k.0),
                                m.blocks[pb].label,
                                crate::op_attrs::attrs(m.blocks[pb].insts[pi].op).mnemonic,
                                if matches!(k.0, Ctr::VaVdst | Ctr::VmVsrc) { " (relaxed scheduling mode)" } else { "" }
                            ),
                        ));
                    }
                }
                // the new operation
                let at = (bi, ii);
                if e.smem {
                    issue(&mut st, Ctr::Km, at, 0, writes.clone(), reads.clone(), 1);
                    issue(&mut st, Ctr::X, at, 1, vec![], reads.clone(), 1);
                } else if let Some(ma) = &e.mem {
                    let vsrc: Vec<u32> = reads.iter().copied().filter(|r| *r < VGPR_LIMIT).collect();
                    match ma.space {
                        Space::Global => {
                            let c = if !e.writes.is_empty() { Ctr::Load } else { Ctr::Store };
                            issue(&mut st, c, at, 0, writes.clone(), reads.clone(), 1);
                            issue(&mut st, Ctr::X, at, 0, vec![], reads.clone(), 1);
                            let class = if crate::op_attrs::attrs(inst.op).mnemonic.starts_with("buffer") { 2 } else { 0 };
                            issue(&mut st, Ctr::VmVsrc, at, class, vec![], vsrc, 1);
                        }
                        Space::Lds => {
                            if !e.writes.is_empty() {
                                issue(&mut st, Ctr::Ds, at, 0, writes.clone(), reads.clone(), 1);
                            }
                            issue(&mut st, Ctr::VmVsrc, at, 1, vec![], vsrc, 1);
                        }
                    }
                } else if valu {
                    let w = if matches!(inst.op, MOp::VWmmaScaleF32_32x16x128F4 | MOp::VWmmaScaleF32_32x16x128F4Z) { 2 } else { 1 };
                    let vd: Vec<u32> = writes.iter().copied().filter(|r| *r < VGPR_LIMIT).collect();
                    let vs: Vec<u32> = reads.iter().copied().filter(|r| *r < VGPR_LIMIT).collect();
                    if !vd.is_empty() || !vs.is_empty() {
                        issue(&mut st, Ctr::VaVdst, at, valu_class(inst.op), vd, vs, w);
                    }
                }
            }
            for &s in &succ[bi] {
                let merged = match &entry[s] {
                    None => st.clone(),
                    Some(x) => merge(x, &st),
                };
                if entry[s].as_ref() != Some(&merged) {
                    entry[s] = Some(merged);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    let mut out: Vec<(u32, String)> = errs.into_values().collect();
    // the software hazards: the author's instructions must already satisfy them
    let hz = super::hazards::plan(m, a);
    let mut hk: Vec<_> = hz.before.iter().collect();
    hk.sort_by_key(|x| x.0);
    for ((bi, ii), lines) in hk {
        let inst = &m.blocks[*bi].insts[*ii];
        out.push((inst.stmt, format!("block {} instruction {ii} (`{}`): missing `{}` (gfx1250 software hazard, docs/gfx1250-hazards.md)", m.blocks[*bi].label, crate::op_attrs::attrs(inst.op).mnemonic, lines.join("; "))));
    }
    out
}
