//! gfx1250 waits (docs/gfx1250-plan.md G2): the shared wait pass decides WHERE the program must wait and for how many of the youngest
//! memory operations (gfx950's combined `vmcnt` / `lgkmcnt` semantics); this pass turns each such wait into the gfx1250 split counters by
//! simulating the outstanding operations, and adds the XCNT waits.
//!
//! Counters (target-gfx1250.md section 4): LOADcnt (vector-memory loads, atomics with return), STOREcnt (stores, atomics without return),
//! DScnt (LDS), KMcnt (scalar memory, out of order: only a wait for 0 identifies an operation), XCNT (the address translation of every
//! vector / scalar memory operation: its address and data registers may be overwritten only after it; vector operations in order, scalar
//! ones out of order). TENSORcnt / ASYNCcnt are never waited automatically (explicit `WaitTensor` / `WaitCounter`).
//!
//! A gfx950 `vmcnt(n)` means: every vector-memory operation except the `n` youngest has completed. With the youngest operations known
//! (the `known` suffix of the queue; older ones possibly unknown at a block entry), the loads among the youngest `n` bound LOADcnt and
//! the stores bound STOREcnt; youngest positions beyond the known suffix may be either. `lgkmcnt(n)` likewise for DScnt, and KMcnt 0
//! whenever a scalar load may be among the older operations. Every block that ends with translations pending waits XCNT 0 there, so
//! every block starts XCNT-clean.

use crate::mir::*;
use crate::regalloc::Alloc;
use crate::sched::effects::{effects, Fence, Space, SCC, VCC};
use std::collections::HashSet;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Load,
    Store,
    Ds,
    Km,
}

#[derive(Clone, Debug)]
struct Pending {
    kind: Kind,
    /// physical registers it reads after issue (address, data): XCNT
    srcs: Vec<u32>,
    xlate: bool,
}

#[derive(Clone, Debug, Default)]
struct State {
    /// vector-memory queue (Load / Store), oldest first, and whether older unknown operations may precede it
    vm: Vec<Pending>,
    vm_unknown: bool,
    /// LDS + scalar queue (Ds / Km)
    lg: Vec<Pending>,
    lg_unknown: bool,
}

const LOAD_MAX: u32 = 63;
const STORE_MAX: u32 = 63;
const DS_MAX: u32 = 63;
const KM_MAX: u32 = 31;
const X_MAX: u32 = 63;

/// Physical register keys: VGPR n -> n, AGPR -> 0x1_0000 + n, SGPR n -> 0x2_0000 + n, VCC / SCC.
fn phys_keys(m: &MFunction, a: &Alloc, slots: &[u32]) -> Vec<u32> {
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
        out.push(if c.is_agpr() { 0x1_0000 + (p - AGPR_TAG) + off } else if c.is_scalar() { 0x2_0000 + p + off } else { p + off });
    }
    out
}

/// The per-instruction plan: lines before instruction `i` of block `b`, replacement lines for the gfx950 waits, lines at a block's end.
#[derive(Default)]
pub struct Plan {
    pub before: std::collections::HashMap<(usize, usize), Vec<String>>,
    pub replace: std::collections::HashMap<(usize, usize), Vec<String>>,
    pub end: std::collections::HashMap<usize, Vec<String>>,
}

/// Bound for class `k` among the youngest `n` operations of `q` (positions beyond the known suffix count fully when `unknown`).
fn bound(q: &[Pending], unknown: bool, n: usize, k: Kind) -> u32 {
    let known = q.len().min(n);
    let mut c = q[q.len() - known..].iter().filter(|p| p.kind == k).count();
    if unknown && n > q.len() {
        c += n - q.len();
    }
    c as u32
}

/// Completes the operations of kind `k` except the youngest `keep` of that kind (in-order counters).
fn complete(q: &mut Vec<Pending>, k: Kind, keep: u32) {
    let total = q.iter().filter(|p| p.kind == k).count() as u32;
    let mut drop = total.saturating_sub(keep);
    q.retain(|p| {
        if p.kind == k && drop > 0 {
            drop -= 1;
            false
        } else {
            true
        }
    });
}

pub fn plan(m: &MFunction, a: &Alloc) -> Plan {
    let mut plan = Plan::default();
    // the state flows into the next block only when that block's sole predecessor is this one (falling through or by its branch)
    let mut preds = vec![0usize; m.blocks.len()];
    let idx: std::collections::HashMap<u32, usize> = m.blocks.iter().enumerate().map(|(i, b)| (b.label, i)).collect();
    let mut sole_pred: Vec<Option<usize>> = vec![None; m.blocks.len()];
    for (bi, b) in m.blocks.iter().enumerate() {
        let succ: Vec<usize> = match &b.term {
            Term::Goto(t) => vec![idx[t]],
            Term::CondGoto { target, fall, .. } => vec![idx[target], idx[fall]],
            Term::Endpgm => vec![],
        };
        for s in succ {
            preds[s] += 1;
            sole_pred[s] = Some(bi);
        }
    }
    let mut out_state: Vec<Option<State>> = vec![None; m.blocks.len()];
    for (bi, b) in m.blocks.iter().enumerate() {
        let mut st = match (bi, preds[bi], sole_pred[bi]) {
            (0, 0, _) => State::default(),
            (_, 1, Some(p)) if p < bi => out_state[p].clone().unwrap_or(State { vm_unknown: true, lg_unknown: true, ..Default::default() }),
            _ => State { vm_unknown: true, lg_unknown: true, ..Default::default() },
        };
        for (ii, inst) in b.insts.iter().enumerate() {
            let e = effects(m, inst);
            // the gfx950 waits -> split counters
            let (vm, lg) = match e.fence {
                Fence::WaitVm(n) => (Some(n as usize), None),
                Fence::WaitLgkm(n) => (None, Some(n as usize)),
                Fence::WaitBoth(v, l) => (Some(v as usize), Some(l as usize)),
                _ => (None, None),
            };
            if matches!(inst.op, MOp::SWaitVm | MOp::SWaitLgkm | MOp::SWaitCnt) {
                let mut lines = Vec::new();
                if let Some(n) = vm {
                    if st.vm.len() > n || st.vm_unknown {
                        let k = st.vm.len();
                        let l = bound(&st.vm, st.vm_unknown, n, Kind::Load).min(LOAD_MAX);
                        let s = bound(&st.vm, st.vm_unknown, n, Kind::Store).min(STORE_MAX);
                        let (ql, qs) = (st.vm.iter().filter(|p| p.kind == Kind::Load).count() as u32, st.vm.iter().filter(|p| p.kind == Kind::Store).count() as u32);
                        if st.vm_unknown || ql > l {
                            lines.push(format!("s_wait_loadcnt {l:#x}"));
                        }
                        if st.vm_unknown || qs > s {
                            lines.push(format!("s_wait_storecnt {s:#x}"));
                        }
                        complete(&mut st.vm, Kind::Load, l);
                        complete(&mut st.vm, Kind::Store, s);
                        // the unknown older operations are older than the youngest n: complete when n <= the known suffix
                        if n <= k {
                            st.vm_unknown = false;
                        }
                    }
                }
                if let Some(n) = lg {
                    if st.lg.len() > n || st.lg_unknown {
                        let k = st.lg.len();
                        let d = bound(&st.lg, st.lg_unknown, n, Kind::Ds).min(DS_MAX);
                        let qd = st.lg.iter().filter(|p| p.kind == Kind::Ds).count() as u32;
                        if st.lg_unknown || qd > d {
                            lines.push(format!("s_wait_dscnt {d:#x}"));
                        }
                        // scalar loads complete out of order: any among the older operations -> wait for all of them
                        let older = st.lg.len().saturating_sub(n);
                        let km_older = st.lg_unknown || st.lg[..older].iter().any(|p| p.kind == Kind::Km);
                        if km_older {
                            lines.push("s_wait_kmcnt 0x0".to_string());
                            st.lg.retain(|p| p.kind != Kind::Km);
                        }
                        complete(&mut st.lg, Kind::Ds, d);
                        if km_older && n <= k {
                            st.lg_unknown = false;
                        }
                    }
                }
                plan.replace.insert((bi, ii), lines);
                continue;
            }
            // explicit split waits
            if inst.op == MOp::SWaitCtr {
                let (c, n) = ((inst.imm >> 8) as usize, (inst.imm & 0xff) as u32);
                match c {
                    0 => complete(&mut st.vm, Kind::Load, n),
                    1 => complete(&mut st.vm, Kind::Store, n),
                    2 => complete(&mut st.lg, Kind::Ds, n),
                    3 if n == 0 => st.lg.retain(|p| p.kind != Kind::Km),
                    _ => {}
                }
                continue;
            }
            // XCNT: an operation overwriting a register a pending memory operation still reads waits for its translation
            let writes: HashSet<u32> = phys_keys(m, a, &e.writes).into_iter().collect();
            if !writes.is_empty() {
                let all: Vec<&Pending> = st.vm.iter().chain(st.lg.iter()).filter(|p| p.xlate).collect();
                // the vector ones are in order (the queue order); a scalar one anywhere forces 0
                let hit_vm = st.vm.iter().rposition(|p| p.xlate && p.srcs.iter().any(|r| writes.contains(r)));
                let hit_km = st.lg.iter().any(|p| p.xlate && p.srcs.iter().any(|r| writes.contains(r)));
                if hit_vm.is_some() || hit_km {
                    let km_pending = st.lg.iter().any(|p| p.xlate);
                    let x = match hit_vm {
                        Some(i) if !hit_km && !km_pending => st.vm[i + 1..].iter().filter(|p| p.xlate).count() as u32,
                        _ => 0,
                    }
                    .min(X_MAX);
                    let _ = all;
                    plan.before.entry((bi, ii)).or_default().push(format!("s_wait_xcnt {x:#x}"));
                    let mut keep = x;
                    for p in st.vm.iter_mut().rev() {
                        if p.xlate {
                            if keep > 0 {
                                keep -= 1;
                            } else {
                                p.xlate = false;
                            }
                        }
                    }
                    if x == 0 {
                        for p in st.lg.iter_mut() {
                            p.xlate = false;
                        }
                    }
                }
            }
            // the device-scope cache maintenance of the fences: `global_wb` completes on STOREcnt, `global_inv` on LOADcnt (LLVM
            // SIGfx12CacheControl waits for them on those counters); they are fences, not memory accesses, in the effects model, so without
            // this the explicit wait that follows them would see nothing outstanding and be dropped
            if inst.op == MOp::BufferWbl2 {
                st.vm.push(Pending { kind: Kind::Store, srcs: Vec::new(), xlate: false });
            } else if inst.op == MOp::BufferInv {
                st.vm.push(Pending { kind: Kind::Load, srcs: Vec::new(), xlate: false });
            }
            // the new memory operation
            if e.smem {
                st.lg.push(Pending { kind: Kind::Km, srcs: phys_keys(m, a, &e.reads), xlate: super::xnack() });
            } else if let Some(ma) = &e.mem {
                let srcs = phys_keys(m, a, &e.reads);
                match ma.space {
                    Space::Global => {
                        let kind = if !e.writes.is_empty() { Kind::Load } else { Kind::Store };
                        st.vm.push(Pending { kind, srcs, xlate: super::xnack() });
                    }
                    Space::Lds => st.lg.push(Pending { kind: Kind::Ds, srcs, xlate: false }),
                }
            }
        }
        let pending_x = st.vm.iter().chain(st.lg.iter()).any(|p| p.xlate);
        if pending_x && !matches!(b.term, Term::Endpgm) {
            plan.end.insert(bi, vec!["s_wait_xcnt 0x0".to_string()]);
            for p in st.vm.iter_mut().chain(st.lg.iter_mut()) {
                p.xlate = false;
            }
        }
        out_state[bi] = Some(st);
    }
    plan
}
