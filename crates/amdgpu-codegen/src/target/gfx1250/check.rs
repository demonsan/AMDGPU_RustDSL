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
    /// younger operations of the same counter AND event class (in order with this one)
    same: u32,
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
    // an event class completes in order, the classes out of order with each other: while an operation is outstanding so are the `same`
    // younger operations of its class, so the counter is at least same + 1; a wait for n <= same therefore completes it (with one class
    // pending, same = younger). The scalar loads (and the scalar translations) complete out of order: only n = 0.
    let ooo = matches!(c, Ctr::Km) || (c == Ctr::X && st.items.iter().any(|(k, _)| k.0 == Ctr::X && k.2 == 1));
    let done: Vec<Key> = st.items.iter().filter(|(k, v)| k.0 == c && (n == 0 || (!ooo && v.same >= n))).map(|(k, _)| *k).collect();
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
            if v.class == class {
                v.same = (v.same + weight).min(64);
            }
        }
    }
    // counter saturation: the hardware counter holds at most `max` outstanding operations, and an event class completes in order, so an
    // operation followed by `max` younger operations of its own class cannot still be outstanding (va_vdst 4 bits / vm_vsrc 3 bits: LLVM
    // AMDGPUWaitcntUtils VaVdstMax 15, VmVsrcMax 7; the memory counters 6 bits). Not for the out-of-order scalar loads.
    let max = match c {
        Ctr::VaVdst => 15,
        Ctr::VmVsrc => 7,
        Ctr::Load | Ctr::Store | Ctr::Ds => 63,
        Ctr::X if class == 0 => 63,
        _ => u32::MAX,
    };
    st.items.retain(|kk, v| !(kk.0 == c && v.class == class && v.same >= max));
    let k = (c, at, class);
    let it = Item { younger: 0, same: 0, class, dst, src };
    st.items.insert(k, it);
}

fn merge(a: &State, b: &State) -> State {
    let mut r = a.clone();
    for (k, v) in &b.items {
        match r.items.get_mut(k) {
            Some(x) => {
                x.younger = x.younger.min(v.younger);
                x.same = x.same.min(v.same);
            }
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
/// (statement, is a software-hazard finding, message).
pub fn verify(m: &MFunction, a: &Alloc) -> Vec<(u32, bool, String)> {
    verify_mode(m, a, 0)
}

/// `verify` with the wave's scheduling mode at entry `mode0` (a kernel that sets it itself overrides it).
pub fn verify_mode(m: &MFunction, a: &Alloc, mode0: u8) -> Vec<(u32, bool, String)> {
    verify_mode_pairs(m, a, mode0, &std::collections::HashSet::new())
}

/// `verify_mode` where the instructions at `pair_second` (block, index) are the second component of a VOPD pair: the pair issues as ONE
/// VALU instruction, so it adds one (not two) to the younger-operation counts.
pub fn verify_mode_pairs(m: &MFunction, a: &Alloc, mode0: u8, pair_second: &std::collections::HashSet<(usize, usize)>) -> Vec<(u32, bool, String)> {
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
    entry[0] = Some(State { items: BTreeMap::new(), mode: Some(mode0) });
    let mut errs: BTreeMap<(usize, usize, Ctr), (u32, bool, String)> = BTreeMap::new();
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
                // a load overwriting the destination of an older load of the SAME in-order counter needs no wait: LOADcnt / DScnt
                // operations return in issue order, so the younger load's data lands last (LLVM SIInsertWaitcnts waits on a WAW only
                // across different VMEM types; `hazards::pad_inflight_waw` makes the same exception)
                let my_load = if e.smem || e.writes.is_empty() {
                    None
                } else {
                    e.mem.as_ref().map(|ma| match ma.space {
                        crate::sched::effects::Space::Global => Ctr::Load,
                        crate::sched::effects::Space::Lds => Ctr::Ds,
                    })
                };
                // dependencies on outstanding operations
                for (k, v) in &st.items {
                    let hit = |xs: &[u32], ys: &[u32]| xs.iter().any(|x| ys.contains(x));
                    let dep = match k.0 {
                        Ctr::Load | Ctr::Ds => hit(&reads, &v.dst) || (hit(&writes, &v.dst) && my_load != Some(k.0)),
                        Ctr::Km => hit(&reads, &v.dst) || hit(&writes, &v.dst),
                        Ctr::Store => false,
                        Ctr::X => hit(&writes, &v.src),
                        Ctr::VaVdst => relaxed && !valu && (hit(&reads, &v.dst) || hit(&writes, &v.dst) || hit(&writes, &v.src)),
                        Ctr::VmVsrc => relaxed && hit(&writes, &v.src),
                    };
                    if dep {
                        let (pb, pi) = k.1;
                        errs.entry((bi, ii, k.0)).or_insert((
                            inst.stmt,
                            false,
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
                    if super::xnack() {
                        issue(&mut st, Ctr::X, at, 1, vec![], reads.clone(), 1);
                    }
                } else if let Some(ma) = &e.mem {
                    let vsrc: Vec<u32> = reads.iter().copied().filter(|r| *r < VGPR_LIMIT).collect();
                    match ma.space {
                        Space::Global => {
                            let c = if !e.writes.is_empty() { Ctr::Load } else { Ctr::Store };
                            issue(&mut st, c, at, 0, writes.clone(), reads.clone(), 1);
                            if super::xnack() {
                                issue(&mut st, Ctr::X, at, 0, vec![], reads.clone(), 1);
                            }
                            let class = if crate::op_attrs::attrs(inst.op).mnemonic.starts_with("buffer") { 2 } else { 0 };
                            issue(&mut st, Ctr::VmVsrc, at, class, vec![], vsrc, 1);
                        }
                        Space::Lds => {
                            // every LDS operation (loads, stores, the transposing loads) counts on DScnt, in order (LLVM: one event,
                            // LDS_ACCESS): a store younger than a load is one of the "n youngest" a wait for n leaves outstanding
                            issue(&mut st, Ctr::Ds, at, 0, writes.clone(), reads.clone(), 1);
                            issue(&mut st, Ctr::VmVsrc, at, 1, vec![], vsrc, 1);
                        }
                    }
                } else if valu {
                    let w = if pair_second.contains(&(bi, ii)) { 0 } else if matches!(inst.op, MOp::VWmmaScaleF32_32x16x128F4 | MOp::VWmmaScaleF32_32x16x128F4Z) { 2 } else { 1 };
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
    let mut out: Vec<(u32, bool, String)> = errs.into_values().collect();
    // the software hazards: the author's instructions must already satisfy them
    let hz = super::hazards::plan(m, a);
    let mut hk: Vec<_> = hz.before.iter().collect();
    hk.sort_by_key(|x| x.0);
    for ((bi, ii), lines) in hk {
        let inst = &m.blocks[*bi].insts[*ii];
        out.push((inst.stmt, true, format!("block {} instruction {ii} (`{}`): missing `{}` (gfx1250 software hazard, docs/gfx1250-hazards.md)", m.blocks[*bi].label, crate::op_attrs::attrs(inst.op).mnemonic, lines.join("; "))));
    }
    out
}

/// The default `s_wait_alu` operand: every depctr field "no wait" (llvm-mc: `s_wait_alu depctr_va_vdst(3)` = 0x3f9f).
pub const DEPCTR_DEFAULT: u32 = 0xff9f;

/// The depctr operand waiting for va_vdst <= `vd` and vm_vsrc <= `vm` (15 / 7: no wait on that field).
pub fn depctr(vd: u32, vm: u32) -> u32 {
    (DEPCTR_DEFAULT & !(0xf << 12) & !(0x7 << 2)) | (vd.min(15) << 12) | (vm.min(7) << 2)
}

thread_local! {
    /// test hook: drop the k-th planned `s_wait_alu` (mutation tests: the independent check must then reject the kernel)
    static MODE2_DROP: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// Runs `g` with the `k`-th planned mode-2 wait dropped (a mutation test hook).
#[doc(hidden)]
pub fn with_mode2_drop<R>(k: usize, g: impl FnOnce() -> R) -> R {
    MODE2_DROP.with(|c| c.set(Some(k)));
    let r = g();
    MODE2_DROP.with(|c| c.set(None));
    r
}

/// The MIR with the gfx1250 wait plan's lines in place (as SWaitCtr instructions) and, per instruction of the new list, the original
/// (block, index) it came from (None: an inserted wait).
fn with_plan(m: &MFunction, plan: &super::waits::Plan) -> (MFunction, Vec<Vec<Option<usize>>>) {
    let wait_of = |line: &str| -> Option<MInst> {
        let (mn, n) = line.trim().split_once(' ')?;
        let n = u32::from_str_radix(n.trim().trim_start_matches("0x"), 16).ok()? as i64;
        let c = match mn {
            "s_wait_loadcnt" => 0,
            "s_wait_storecnt" => 1,
            "s_wait_dscnt" => 2,
            "s_wait_kmcnt" => 3,
            "s_wait_xcnt" => 4,
            _ => return None,
        };
        Some(MInst::new(MOp::SWaitCtr).imm((c << 8) | n))
    };
    let mut m2 = m.clone();
    let mut orig: Vec<Vec<Option<usize>>> = Vec::new();
    for (bi, b) in m2.blocks.iter_mut().enumerate() {
        let mut out = Vec::with_capacity(b.insts.len());
        let mut o = Vec::with_capacity(b.insts.len());
        for (ii, inst) in b.insts.iter().enumerate() {
            for l in plan.before.get(&(bi, ii)).into_iter().flatten() {
                if let Some(w) = wait_of(l) {
                    out.push(w);
                    o.push(None);
                }
            }
            match plan.replace.get(&(bi, ii)) {
                Some(ls) => {
                    for l in ls {
                        if let Some(w) = wait_of(l) {
                            out.push(w);
                            o.push(None);
                        }
                    }
                }
                None => {
                    out.push(inst.clone());
                    o.push(Some(ii));
                }
            }
        }
        for l in plan.end.get(&bi).into_iter().flatten() {
            if let Some(w) = wait_of(l) {
                out.push(w);
                o.push(None);
            }
        }
        b.insts = out;
        orig.push(o);
    }
    (m2, orig)
}

/// MODE 2 (the relaxed scheduling mode, LLVM's expert scheduling mode 2) for the compiler's own code: the `s_wait_alu` waits the program
/// then owns (va_vdst: a non-VALU instruction touching a VGPR an outstanding VALU writes, or writing one it reads; vm_vsrc: overwriting a
/// VGPR an outstanding vector-memory / LDS operation reads), with the smallest counts that complete the needed operations (va_vdst <= the
/// number of younger VALU operations when one event class is pending, else 0; the counters' saturation as in `issue`), from the same
/// dataflow over the CFG as the verifier, iterated to a fixpoint over loops. Returns, per original (block, instruction), the depctr operand
/// of the `s_wait_alu` to emit before it; the result is then CHECKED by `verify_mode(.., 2)` on the code with every wait in place (an
/// unsound plan is an error, never emitted).
pub fn mode2_plan(m: &MFunction, a: &Alloc, plan: &super::waits::Plan) -> Result<std::collections::HashMap<(usize, usize), u32>, Vec<String>> {
    let (m2, orig) = with_plan(m, plan);
    let n = m2.blocks.len();
    let idx: std::collections::HashMap<u32, usize> = m2.blocks.iter().enumerate().map(|(i, b)| (b.label, i)).collect();
    let succ: Vec<Vec<usize>> = m2
        .blocks
        .iter()
        .map(|b| match &b.term {
            Term::Goto(t) => vec![idx[t]],
            Term::CondGoto { target, fall, .. } => vec![idx[target], idx[fall]],
            Term::Endpgm => vec![],
        })
        .collect();
    // the waits (block, index in m2) -> (va_vdst, vm_vsrc); only ever lowered
    let mut need: std::collections::BTreeMap<(usize, usize), (u32, u32)> = std::collections::BTreeMap::new();
    let mut entry: Vec<Option<State>> = vec![None; n];
    entry[0] = Some(State { items: BTreeMap::new(), mode: Some(2) });
    for _round in 0..400 {
        let mut changed = false;
        for bi in 0..n {
            let Some(mut st) = entry[bi].clone() else { continue };
            for (ii, inst) in m2.blocks[bi].insts.iter().enumerate() {
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
                let e = effects(&m2, inst);
                let reads = keys(&m2, a, &e.reads);
                let writes = keys(&m2, a, &e.writes);
                let valu = is_valu(inst.op);
                let relaxed = st.mode != Some(0);
                // the waits this instruction needs (the planned one applied first, so a later round only lowers it)
                let mut vd = 15u32;
                let mut vm = 7u32;
                if relaxed {
                    let hit = |xs: &[u32], ys: &[u32]| xs.iter().any(|x| ys.contains(x));
                    for c in [Ctr::VaVdst, Ctr::VmVsrc] {
                        for (k, v) in st.items.iter().filter(|(k, _)| k.0 == c) {
                            let dep = match c {
                                Ctr::VaVdst => !valu && (hit(&reads, &v.dst) || hit(&writes, &v.dst) || hit(&writes, &v.src)),
                                _ => hit(&writes, &v.src),
                            };
                            if dep {
                                let _ = k;
                                // (the younger operations of its own event class: see `wait`)
                                let cnt = v.same;
                                if c == Ctr::VaVdst {
                                    vd = vd.min(cnt);
                                } else {
                                    vm = vm.min(cnt);
                                }
                            }
                        }
                    }
                }
                if vd < 15 || vm < 7 {
                    let e2 = need.entry((bi, ii)).or_insert((15, 7));
                    if vd < e2.0 || vm < e2.1 {
                        e2.0 = e2.0.min(vd);
                        e2.1 = e2.1.min(vm);
                        changed = true;
                    }
                }
                if let Some(&(pv, pm)) = need.get(&(bi, ii)) {
                    wait(&mut st, Ctr::VaVdst, pv);
                    wait(&mut st, Ctr::VmVsrc, pm);
                }
                // the new operation (as in verify_mode)
                let at = (bi, ii);
                if e.smem {
                    issue(&mut st, Ctr::Km, at, 0, writes.clone(), reads.clone(), 1);
                    if super::xnack() {
                        issue(&mut st, Ctr::X, at, 1, vec![], reads.clone(), 1);
                    }
                } else if let Some(ma) = &e.mem {
                    let vsrc: Vec<u32> = reads.iter().copied().filter(|r| *r < VGPR_LIMIT).collect();
                    match ma.space {
                        Space::Global => {
                            let c = if !e.writes.is_empty() { Ctr::Load } else { Ctr::Store };
                            issue(&mut st, c, at, 0, writes.clone(), reads.clone(), 1);
                            if super::xnack() {
                                issue(&mut st, Ctr::X, at, 0, vec![], reads.clone(), 1);
                            }
                            let class = if crate::op_attrs::attrs(inst.op).mnemonic.starts_with("buffer") { 2 } else { 0 };
                            issue(&mut st, Ctr::VmVsrc, at, class, vec![], vsrc, 1);
                        }
                        Space::Lds => {
                            issue(&mut st, Ctr::Ds, at, 0, writes.clone(), reads.clone(), 1);
                            issue(&mut st, Ctr::VmVsrc, at, 1, vec![], vsrc, 1);
                        }
                    }
                } else if valu {
                    let second = orig[bi].get(ii).copied().flatten().map_or(false, |o| super::vopd::is_second(bi, o));
                    let w = if second { 0 } else if matches!(inst.op, MOp::VWmmaScaleF32_32x16x128F4 | MOp::VWmmaScaleF32_32x16x128F4Z) { 2 } else { 1 };
                    let vdst: Vec<u32> = writes.iter().copied().filter(|r| *r < VGPR_LIMIT).collect();
                    let vs: Vec<u32> = reads.iter().copied().filter(|r| *r < VGPR_LIMIT).collect();
                    if !vdst.is_empty() || !vs.is_empty() {
                        issue(&mut st, Ctr::VaVdst, at, valu_class(inst.op), vdst, vs, w);
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
    // the waits back on the original instructions (an inserted memory wait is never a consumer: no entry there)
    let drop = MODE2_DROP.with(|c| c.get());
    let mut out: std::collections::HashMap<(usize, usize), u32> = std::collections::HashMap::new();
    for (k, ((bi, ii), (vd, vm))) in need.iter().enumerate() {
        if drop == Some(k) {
            continue;
        }
        if let Some(Some(o)) = orig.get(*bi).and_then(|v| v.get(*ii)) {
            out.insert((*bi, *o), depctr(*vd, *vm));
        }
    }
    // the independent check: the code with every wait in place, in mode 2 from the entry
    let mut m3 = m2.clone();
    let mut seconds: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
    for (bi, b) in m3.blocks.iter_mut().enumerate() {
        let mut v = Vec::with_capacity(b.insts.len() + 8);
        for (ii, inst) in b.insts.iter().enumerate() {
            if let Some(Some(o)) = orig[bi].get(ii) {
                if let Some(&d) = out.get(&(bi, *o)) {
                    let mut w = MInst::new(MOp::SWaitAlu).imm(d as i64);
                    w.stmt = inst.stmt;
                    v.push(w);
                }
                if super::vopd::is_second(bi, *o) {
                    seconds.insert((bi, v.len()));
                }
            }
            v.push(inst.clone());
        }
        b.insts = v;
    }
    let errs: Vec<String> = verify_mode_pairs(&m3, a, 2, &seconds).into_iter().filter(|(_, hz, _)| !hz).map(|(_, _, msg)| msg).collect();
    if errs.is_empty() {
        Ok(out)
    } else {
        Err(errs)
    }
}

/// The compiler's own waits checked independently (non-exact gfx1250): the MIR with the gfx1250 wait plan's lines (waits.rs: the gfx950
/// pass's waits converted to split counters, the XCNT waits) put in place of / before the instructions, then the same dataflow as for an
/// exact kernel. The shared wait pass and its verifier share one counter model; this check uses the gfx1250 counters directly, so a wait the
/// shared model drops (e.g. an operation it believes complete by a counter range the target does not have) is an error, not wrong data.
pub fn verify_planned(m: &MFunction, a: &Alloc, plan: &super::waits::Plan) -> Vec<String> {
    let wait_of = |line: &str| -> Option<MInst> {
        let (mn, n) = line.trim().split_once(' ')?;
        let n = u32::from_str_radix(n.trim().trim_start_matches("0x"), 16).ok()? as i64;
        let c = match mn {
            "s_wait_loadcnt" => 0,
            "s_wait_storecnt" => 1,
            "s_wait_dscnt" => 2,
            "s_wait_kmcnt" => 3,
            "s_wait_xcnt" => 4,
            _ => return None,
        };
        Some(MInst::new(MOp::SWaitCtr).imm((c << 8) | n))
    };
    let mut m2 = m.clone();
    for (bi, b) in m2.blocks.iter_mut().enumerate() {
        let mut out = Vec::with_capacity(b.insts.len());
        for (ii, inst) in b.insts.iter().enumerate() {
            for l in plan.before.get(&(bi, ii)).into_iter().flatten() {
                out.extend(wait_of(l).map(|mut w| {
                    w.stmt = inst.stmt;
                    w
                }));
            }
            match plan.replace.get(&(bi, ii)) {
                Some(ls) => {
                    for l in ls {
                        out.extend(wait_of(l).map(|mut w| {
                            w.stmt = inst.stmt;
                            w
                        }));
                    }
                }
                None => out.push(inst.clone()),
            }
        }
        for l in plan.end.get(&bi).into_iter().flatten() {
            out.extend(wait_of(l));
        }
        b.insts = out;
    }
    // the instructions moved; the allocation is per value, so it still applies
    // (debug, `AMDC_MODE2_SITES`: the dependencies the relaxed scheduling mode would leave to software in this code, for the mode-2 probe)
    if std::env::var("AMDC_MODE2_SITES").map_or(false, |v| !v.is_empty()) {
        let mut hist: BTreeMap<String, usize> = BTreeMap::new();
        for (_, hz, msg) in verify_mode(&m2, a, 2) {
            if hz || !msg.contains("relaxed") {
                continue;
            }
            // "... (`consumer`): needs <ctr> for the operation at ... (`producer`) ..."
            let q: Vec<&str> = msg.split('`').collect();
            let ctr = if msg.contains("va_vdst") { "va_vdst" } else { "vm_vsrc" };
            if q.len() >= 4 {
                *hist.entry(format!("{ctr}: {} -> {}", q[3], q[1])).or_insert(0) += 1;
            }
        }
        for (k, n) in hist {
            eprintln!("MODE2-SITE {n:5} {k}");
        }
    }
    verify(&m2, a).into_iter().filter(|(_, hazard, _)| !hazard).map(|(_, _, msg)| msg).collect()
}
