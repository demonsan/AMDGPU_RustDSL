//! Slot-filling scheduler (`slotfill`): "issue one, subtract one", with the per-MFMA budgets derived by the compiler instead of authored (Tensile-style knobs
//! such as `dma_gap`).
//!
//! The MFMA sequence of a region is the fixed skeleton (author order kept). Slot `k` is the gap in front of MFMA `k` (slot `M` = after the last one); one slot is one
//! measured MFMA issue interval (32 cycles, tools/mfma_ceiling.py: 32.25 on gfx950) minus the MFMA's own issue, `SLOT_BUDGET` cycles for everything else.
//! Every other instruction (a *floater*) gets a window `[lo, dl]` of slots:
//!   * `lo`  - the earliest slot that respects its producers (an MFMA reader of the old value, a producer's latency in slots, at most `HOIST_SLOTS` before its source
//!             position: the register-pressure cap),
//!   * `dl`  - the DEADLINE, the latest slot that still meets its consumers: an MFMA consumer at slot k minus the producer latency in slots (a `ds_read` feeds its MFMA
//!             about 4 slots later), a floater consumer's own deadline, a DMA with no consumer in the region the region end minus the DMA latency.
//! The memory operations (DMA, `ds_read`, `ds_write`, global loads / stores) are the anchors. Each class is spread evenly over the span of its windows (the k-th of n at
//! `lo_min + (k + 1/2) (dl_max - lo_min + 1) / n`, clamped to its own window) and a slot whose budget is used up passes the instruction to the next slot with room
//! (deadline first, then earlier). Whatever an anchor needs (address chains, `s_mov`s) is placed in the anchor's slot just before it; everything else keeps its authored
//! slot clamped to its window. A final pass makes producers precede consumers slot-wise. The result is checked by `verify::check_perm` (with the LDS-DMA / `ds_read`
//! relaxation below) and, when anything is inconsistent, the region is left alone.
//!
//! Regions are the stretches between hard fences (`s_barrier`, explicit waits, `s_setprio`, cache maintenance); blocks with explicit partial waits are skipped (they count
//! operations by program order). M0 and SCC / VCC stay exclusive resources through the dependence graph (registers / slots); the M0 reuse pass runs after scheduling and
//! keeps working across other instructions. In-order vm / lgkm queues: DMAs and global accesses keep their relative order (fence ordering), `ds_read`s may reorder among
//! themselves but the wait pass recomputes every `lgkmcnt` / `vmcnt` from the final order.
//!
//! LEGALITY CONTRACT (opt-in): inside one barrier interval an LDS-DMA never writes bytes that a `ds_read` of the same interval reads (the 2-slot / ring protocol of
//! `tile::Pipeline`, proven on the IR by the protocol verifier). `order::set_relax_dma_lds_read` (thread-local) drops the DMA <-> `ds_read` ordering edges for the duration of this pass.

use super::depgraph;
use super::effects::*;
use super::model;
use super::order::set_relax_dma_lds_read;
use super::{verify, Options};
use crate::mir::*;

pub const SLOT_CYCLES: u32 = 32;
pub const SLOT_BUDGET: u32 = 28;
/// Register-pressure cap: a floater never moves more than this many slots (MFMAs) above its source position.
pub const HOIST_SLOTS: usize = 40;
/// A DMA with no consumer in the region must be issued this many slots before the region end (about the DMA latency).
pub const DMA_SLOTS: usize = 24;

fn is_matrix(op: MOp) -> bool {
    crate::agpr::is_mfma(op) || op == MOp::VSmfmac16x16x128Fp8
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Class {
    Dma,
    DsRead,
    DsWrite,
    GLoad,
    GStore,
}

fn class_of(e: &Effects) -> Option<Class> {
    if e.fence == Fence::Barrier && e.mem.is_some() {
        return Some(Class::Dma);
    }
    if e.fence != Fence::None {
        return None;
    }
    match &e.mem {
        Some(a) => Some(match (a.space, a.write) {
            (Space::Lds, false) => Class::DsRead,
            (Space::Lds, true) => Class::DsWrite,
            (Space::Global, false) => Class::GLoad,
            (Space::Global, true) => Class::GStore,
        }),
        None => None,
    }
}

fn hard_fence(e: &Effects) -> bool {
    e.fence != Fence::None && e.mem.is_none()
}

/// Re-order one region. `None` = left alone (too few MFMAs, or the result could not be made consistent).
fn fill_region(m: &MFunction, seg: &[MInst], term: &Term, o: &Options, log: &mut Vec<String>) -> Result<Option<Vec<usize>>, String> {
    let n = seg.len();
    let mf: Vec<usize> = (0..n).filter(|&i| is_matrix(seg[i].op)).collect();
    let mm = mf.len();
    if mm < 8 {
        return Ok(None);
    }
    let g = depgraph::build(m, seg, term, true);
    let mut mslot: Vec<Option<usize>> = vec![None; n];
    for (k, &i) in mf.iter().enumerate() {
        mslot[i] = Some(k);
    }
    // source slot: the number of MFMAs before the instruction
    let mut orig = vec![0usize; n];
    let mut cnt = 0;
    for i in 0..n {
        orig[i] = cnt;
        if mslot[i].is_some() {
            cnt += 1;
        }
    }
    let cost: Vec<u32> = seg.iter().map(|i| model::cost(i.op).issue).collect();
    let fl = |lat: u32| (lat / SLOT_CYCLES) as usize;
    let cls: Vec<Option<Class>> = (0..n).map(|i| class_of(&g.eff[i])).collect();
    // ---- windows ----
    let mut lo = vec![0usize; n];
    for i in 0..n {
        if let Some(k) = mslot[i] {
            lo[i] = k;
            continue;
        }
        let mut l = orig[i].saturating_sub(HOIST_SLOTS);
        for &(p, lat) in &g.preds[i] {
            let c = match mslot[p] {
                Some(k) => k + 1 + fl(lat),
                None => lo[p] + fl(lat),
            };
            l = l.max(c);
        }
        lo[i] = l.min(mm);
    }
    let mut dl = vec![mm; n];
    for i in (0..n).rev() {
        if let Some(k) = mslot[i] {
            dl[i] = k;
            continue;
        }
        // a consumer outside the region (next iteration / next tile) sees the value at the region end
        let mut d = match cls[i] {
            Some(Class::Dma) => mm.saturating_sub(DMA_SLOTS),
            Some(_) => mm.saturating_sub(fl(model::cost(seg[i].op).latency)),
            None => mm,
        };
        for &(s, lat) in &g.succs[i] {
            let ds = match mslot[s] {
                Some(k) => k,
                None => dl[s],
            };
            d = d.min(ds.saturating_sub(fl(lat)));
        }
        dl[i] = d.max(lo[i]).min(mm);
    }
    // ---- anchors, class by class ----
    let mut slot: Vec<Option<usize>> = vec![None; n];
    let mut used = vec![0u32; mm + 1];
    for (k, &i) in mf.iter().enumerate() {
        slot[i] = Some(k);
    }
    for class in [Class::Dma, Class::DsRead, Class::DsWrite, Class::GLoad, Class::GStore] {
        let members: Vec<usize> = (0..n).filter(|&i| cls[i] == Some(class)).collect();
        let nm = members.len();
        if nm == 0 {
            continue;
        }
        let wlo = members.iter().map(|&i| lo[i]).min().unwrap();
        let whi = members.iter().map(|&i| dl[i]).max().unwrap();
        for (k, &i) in members.iter().enumerate() {
            let target = wlo + ((2 * k + 1) * (whi - wlo + 1)) / (2 * nm);
            let s0 = target.clamp(lo[i], dl[i]);
            let mut pick = s0;
            let fits = |s: usize, used: &Vec<u32>| used[s] + cost[i] <= SLOT_BUDGET;
            if !fits(s0, &used) {
                if let Some(s) = (s0 + 1..=dl[i]).find(|&s| fits(s, &used)) {
                    pick = s;
                } else if let Some(s) = (lo[i]..s0).rev().find(|&s| fits(s, &used)) {
                    pick = s;
                }
            }
            slot[i] = Some(pick);
            used[pick] += cost[i];
        }
    }
    // chains: the unplaced producers of an anchor join its slot (anchors in slot order, so the first consumer wins)
    let mut anchors: Vec<usize> = (0..n).filter(|&i| cls[i].is_some() && mslot[i].is_none()).collect();
    anchors.sort_by_key(|&i| (slot[i].unwrap(), i));
    for &a in &anchors {
        let s = slot[a].unwrap();
        let mut stack: Vec<usize> = g.preds[a].iter().map(|p| p.0).collect();
        while let Some(p) = stack.pop() {
            if slot[p].is_some() || mslot[p].is_some() || hard_fence(&g.eff[p]) {
                continue;
            }
            slot[p] = Some(s.max(lo[p]));
            used[s] += cost[p];
            stack.extend(g.preds[p].iter().map(|x| x.0));
        }
    }
    for i in 0..n {
        if slot[i].is_none() {
            slot[i] = Some(orig[i].clamp(lo[i], dl[i]));
        }
    }
    let mut sl: Vec<usize> = slot.iter().map(|s| s.unwrap()).collect();
    // ---- consistency: producers never later than consumers (slot-wise), MFMA edges exact ----
    for _ in 0..4 {
        let mut changed = false;
        for i in (0..n).rev() {
            if mslot[i].is_some() {
                continue;
            }
            for &(s, lat) in &g.succs[i] {
                let bound = match mslot[s] {
                    Some(k) => k.saturating_sub(fl(lat)),
                    None => sl[s].saturating_sub(fl(lat)),
                };
                if sl[i] > bound {
                    sl[i] = bound.max(lo[i]);
                    changed = true;
                }
            }
        }
        for i in 0..n {
            if mslot[i].is_some() {
                continue;
            }
            for &(p, lat) in &g.preds[i] {
                let bound = match mslot[p] {
                    Some(k) => k + 1 + fl(lat),
                    None => sl[p] + fl(lat),
                };
                if sl[i] < bound {
                    sl[i] = bound;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    // final check of every edge
    for i in 0..n {
        for &(s, lat) in &g.succs[i] {
            let ok = match (mslot[i], mslot[s]) {
                (Some(a), Some(b)) => a < b,
                (Some(a), None) => sl[s] >= a + 1 + fl(lat),
                (None, Some(b)) => sl[i] <= b.saturating_sub(fl(lat)),
                (None, None) => sl[i] + fl(lat) <= sl[s],
            };
            if !ok {
                log.push(format!("region of {n} instructions, {mm} MFMAs: no consistent slot assignment (edge {i}->{s}), left alone"));
                return Ok(None);
            }
        }
    }
    // ---- emit: slot s = the floaters assigned to it (source order), then MFMA s ----
    let mut order: Vec<usize> = Vec::with_capacity(n);
    let mut by_slot: Vec<Vec<usize>> = vec![Vec::new(); mm + 1];
    for i in 0..n {
        if mslot[i].is_none() {
            by_slot[sl[i]].push(i);
        }
    }
    for s in 0..=mm {
        order.extend(by_slot[s].iter().copied());
        if s < mm {
            order.push(mf[s]);
        }
    }
    if o.stats {
        let dm: Vec<String> = (0..n).filter(|&i| cls[i] == Some(Class::Dma)).map(|i| format!("{}:{}-{}@{}", orig[i], lo[i], dl[i], sl[i])).collect();
        log.push(format!("{mm} MFMAs, DMA src:lo-dl@slot = {}", dm.join(" ")));
        let rd: Vec<String> = (0..n).filter(|&i| cls[i] == Some(Class::DsRead)).map(|i| format!("{}:{}-{}@{}", orig[i], lo[i], dl[i], sl[i])).collect();
        log.push(format!("ds_read src:lo-dl@slot = {}", rd.join(" ")));
    }
    if o.verify {
        let r = verify::check_perm(m, seg, &order, term, true);
        r.map_err(|e| format!("slotfill verification failed: {e}"))?;
    }
    Ok(Some(order))
}

/// Apply `slotfill` to every block with enough MFMAs. Returns the number of instructions that changed position.
pub fn run(m: &mut MFunction, o: &Options) -> Result<usize, String> {
    set_relax_dma_lds_read(true);
    let r = run_inner(m, o);
    set_relax_dma_lds_read(false);
    r
}

fn run_inner(m: &mut MFunction, o: &Options) -> Result<usize, String> {
    let mut moved_total = 0;
    for bi in 0..m.blocks.len() {
        let insts = m.blocks[bi].insts.clone();
        if insts.iter().filter(|i| is_matrix(i.op)).count() < 16 {
            continue;
        }
        if insts.iter().any(|i| matches!(i.op, MOp::SWaitLgkm | MOp::SWaitVm) && i.imm != 0 && !i.auto) {
            if o.stats {
                eprintln!("slotfill: block {bi} has explicit partial waits: skipped");
            }
            continue;
        }
        let term = m.blocks[bi].term.clone();
        let eff: Vec<Effects> = insts.iter().map(|i| effects(m, i)).collect();
        // regions between hard fences; the fences keep their positions
        let mut out: Vec<MInst> = Vec::with_capacity(insts.len());
        let mut start = 0;
        let mut log: Vec<String> = Vec::new();
        let flush = |from: usize, to: usize, out: &mut Vec<MInst>, moved: &mut usize, log: &mut Vec<String>| -> Result<(), String> {
            let seg = &insts[from..to];
            match fill_region(m, seg, &term, o, log)? {
                Some(order) => {
                    *moved += order.iter().enumerate().filter(|(k, p)| *k != **p).count();
                    out.extend(order.into_iter().map(|k| seg[k].clone()));
                }
                None => out.extend(seg.iter().cloned()),
            }
            Ok(())
        };
        let mut moved = 0usize;
        for i in 0..insts.len() {
            if hard_fence(&eff[i]) {
                flush(start, i, &mut out, &mut moved, &mut log)?;
                out.push(insts[i].clone());
                start = i + 1;
            }
        }
        flush(start, insts.len(), &mut out, &mut moved, &mut log)?;
        if o.stats {
            for l in &log {
                eprintln!("slotfill: block {bi}: {l}");
            }
            eprintln!("slotfill: block {bi}: {moved} instructions moved");
        }
        moved_total += moved;
        m.blocks[bi].insts = out;
    }
    Ok(moved_total)
}