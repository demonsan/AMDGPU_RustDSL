//! `ldahead`: deeper LDS-read look-ahead as a post-pass on the final order of each block (docs/lgkm-lookahead.md).
//!
//! The vendor's FMHA loop keeps its `ds_read`s one fragment group AHEAD of the MFMAs that consume them (it waits `lgkmcnt(4)` / `(2)`); the list scheduler issues a read
//! only about one latency before its consumer. This pass walks every `ds_read*` of the block in program order and moves it up, one instruction at a time, until `D` MFMAs
//! sit between the read and its first consumer (`ldahead=D`), a true dependence / fence stops it, or the live VGPR dwords would exceed the block's original peak plus `ldslack`
//! (`ldslack=S`, default 0: the register budget is never raised unless asked). Waits are re-derived from the final order afterwards (count-based, partial `lgkmcnt(n)`).
//! The result is re-checked with `verify::check_perm`; the block is left alone when that fails.

use super::{depgraph, liveness, slotfill, verify, Options};
use crate::mir::*;

fn is_lds_read(op: MOp) -> bool {
    crate::op_attrs::attrs(op).is_lds_read()
}
fn is_matrix(op: MOp) -> bool {
    crate::agpr::is_mfma(op) || op == MOp::VSmfmac16x16x128Fp8
}

/// Number of blocks changed.
pub fn run(m: &mut MFunction, o: &Options) -> usize {
    if o.ldahead == 0 && o.ldahead32 == 0 && o.ldsink == 0 {
        return 0;
    }
    let lv = liveness::compute(m);
    let mut changed = 0;
    for bi in 0..m.blocks.len() {
        let insts = m.blocks[bi].insts.clone();
        let n = insts.len();
        if n < 8 || super::phase::has_markers(&insts) || insts.iter().filter(|i| is_matrix(i.op)).count() < 4 {
            continue;
        }
        let term = m.blocks[bi].term.clone();
        let g = depgraph::build(m, &insts, &term, o.alu_across_barrier);
        let live_out = &lv.live_out[bi];
        let mut order: Vec<usize> = (0..n).collect();
        let mut pos: Vec<usize> = (0..n).collect();
        let limit = slotfill::peak_live(m, &insts, &order, live_out) + o.ldslack;
        let mut moved = 0usize;
        // `ldsink`: sink reads (last first) while more than D MFMAs separate them from their first consumer
        if o.ldsink > 0 {
            for r in (0..n).rev() {
                if !is_lds_read(insts[r].op) {
                    continue;
                }
                let Some(d) = insts[r].dst.map(|d| m.canon(d)) else { continue };
                loop {
                    let p = pos[r];
                    if p + 1 >= n {
                        break;
                    }
                    let mut mf = 0usize;
                    let mut found = false;
                    for q in p + 1..n {
                        let x = &insts[order[q]];
                        if x.uses().iter().any(|u| m.canon(*u) == d) {
                            found = true;
                            break;
                        }
                        if is_matrix(x.op) {
                            mf += 1;
                        }
                    }
                    if !found || mf <= o.ldsink {
                        break;
                    }
                    let next = order[p + 1];
                    if g.succs[r].iter().any(|&(x, _)| x == next) {
                        break;
                    }
                    order.swap(p, p + 1);
                    pos[order[p]] = p;
                    pos[order[p + 1]] = p + 1;
                    moved += 1;
                }
            }
        }
        for r in 0..n {
            let dist = if insts[r].op == MOp::LdsReadB32 { o.ldahead.max(o.ldahead32) } else { o.ldahead };
            if dist == 0 || !is_lds_read(insts[r].op) {
                continue;
            }
            let Some(d) = insts[r].dst.map(|d| m.canon(d)) else { continue };
            loop {
                let p = pos[r];
                if p == 0 {
                    break;
                }
                // MFMAs between the read and its first consumer
                let mut mf = 0usize;
                let mut found = false;
                for q in p + 1..n {
                    let x = &insts[order[q]];
                    if x.uses().iter().any(|u| m.canon(*u) == d) {
                        found = true;
                        break;
                    }
                    if is_matrix(x.op) {
                        mf += 1;
                    }
                }
                if !found || mf >= dist {
                    break;
                }
                let prev = order[p - 1];
                if g.preds[r].iter().any(|&(x, _)| x == prev) {
                    break;
                }
                order.swap(p - 1, p);
                pos[order[p - 1]] = p - 1;
                pos[order[p]] = p;
                if slotfill::peak_live(m, &insts, &order, live_out) > limit {
                    order.swap(p - 1, p);
                    pos[order[p - 1]] = p - 1;
                    pos[order[p]] = p;
                    break;
                }
                moved += 1;
            }
        }
        if moved == 0 {
            continue;
        }
        if o.verify {
            if let Err(e) = verify::check_perm(m, &insts, &order, &term, o.alu_across_barrier) {
                if o.stats {
                    eprintln!("ldahead: block {bi}: {e}: left alone");
                }
                continue;
            }
        }
        if o.stats {
            eprintln!("ldahead: block {bi}: {moved} single-instruction hops");
        }
        m.blocks[bi].insts = order.into_iter().map(|k| insts[k].clone()).collect();
        changed += 1;
    }
    changed
}

/// `wsink`: move every EXPLICIT `s_waitcnt lgkmcnt(..)` (the kernel's `wait_lds`, not an auto wait) down past instructions that neither touch memory nor are fences, until the next fence / memory
/// operation / terminator (the closing `wait_lds(0)` before a barrier then sits after the last MFMAs, whose operand waits become the scoreboard's counted partial waits). A move is legal
/// when no dependence edge links the wait to the instruction it passes; the result is re-checked with `verify::check_perm`. Returns the number of waits moved.
pub fn sink_explicit_waits(m: &mut MFunction, o: &Options) -> usize {
    let mut moved_total = 0;
    for bi in 0..m.blocks.len() {
        let insts = m.blocks[bi].insts.clone();
        let n = insts.len();
        if n < 4 || super::phase::has_markers(&insts) {
            continue;
        }
        let term = m.blocks[bi].term.clone();
        let g = depgraph::build(m, &insts, &term, o.alu_across_barrier);
        let mut order: Vec<usize> = (0..n).collect();
        let mut pos: Vec<usize> = (0..n).collect();
        let mut moved = 0usize;
        for w in 0..n {
            if !matches!(insts[w].op, MOp::SWaitLgkm) || insts[w].auto {
                continue;
            }
            loop {
                let p = pos[w];
                if p + 1 >= n {
                    break;
                }
                let next = order[p + 1];
                let e = &g.eff[next];
                if e.fence != super::effects::Fence::None || e.mem.is_some() || e.smem || e.pinned {
                    break;
                }
                if g.succs[w].iter().any(|&(x, _)| x == next) {
                    break;
                }
                order.swap(p, p + 1);
                pos[order[p]] = p;
                pos[order[p + 1]] = p + 1;
                moved += 1;
            }
        }
        if moved == 0 {
            continue;
        }
        if o.verify {
            if let Err(e) = verify::check_perm(m, &insts, &order, &term, o.alu_across_barrier) {
                if o.stats {
                    eprintln!("wsink: block {bi}: {e}: left alone");
                }
                continue;
            }
        }
        m.blocks[bi].insts = order.into_iter().map(|k| insts[k].clone()).collect();
        moved_total += moved;
    }
    moved_total
}

/// `ovl=N`: spread independent VALU / SALU work evenly over the MFMAs of a barrier-free stretch (the single-wave-per-SIMD design: one wave must overlap its OWN MFMA stream with
/// its softmax VALU, docs/fmha-4wave-design.md). A stretch is a maximal run of instructions without `s_barrier`; with `Ms >= 2` MFMAs and `Nb >= N` movable non-MFMA
/// compute instructions (no memory access, no fence, not pinned, not the terminator's condition writer), the j-th of them (program order) is hoisted to directly behind MFMA
/// `floor(j Ms / Nb)` of the stretch, as far as the dependence graph (registers, memory, fences) allows; nothing is ever moved down or past an instruction it depends on.
/// Re-checked with `verify::check_perm`. Returns the number of hoisted instructions.
pub fn spread_valu(m: &mut MFunction, o: &Options) -> usize {
    use super::effects::Fence;
    let mut total = 0;
    for bi in 0..m.blocks.len() {
        let insts = m.blocks[bi].insts.clone();
        let n = insts.len();
        if n < 6 || super::phase::has_markers(&insts) || insts.iter().filter(|i| is_matrix(i.op)).count() < 2 {
            continue;
        }
        let term = m.blocks[bi].term.clone();
        let g = depgraph::build(m, &insts, &term, o.alu_across_barrier);
        let mut order: Vec<usize> = (0..n).collect();
        let mut pos: Vec<usize> = (0..n).collect();
        let movable: Vec<bool> = (0..n)
            .map(|i| {
                let e = &g.eff[i];
                e.mem.is_none() && e.fence == Fence::None && !e.smem && !e.pinned && !is_matrix(insts[i].op) && insts[i].op != MOp::SchedGroup && g.term_vcc_writer != Some(i)
            })
            .collect();
        let mut moved = 0usize;
        // stretches between barriers
        let mut start = 0usize;
        let mut stretches: Vec<(usize, usize)> = Vec::new();
        for p in 0..n {
            if insts[order[p]].op == MOp::SBarrier {
                stretches.push((start, p));
                start = p + 1;
            }
        }
        stretches.push((start, n));
        for (a, b) in stretches {
            let mfmas: Vec<usize> = (a..b).filter(|&p| is_matrix(insts[order[p]].op)).map(|p| order[p]).collect();
            let bulk: Vec<usize> = (a..b).filter(|&p| movable[order[p]]).map(|p| order[p]).collect();
            if mfmas.len() < 2 || bulk.len() < o.ovl {
                continue;
            }
            for (j, &x) in bulk.iter().enumerate() {
                let tm = mfmas[j * mfmas.len() / bulk.len()];
                // hop up until x sits directly behind MFMA tm (or behind an earlier-placed bulk instruction that sits there)
                loop {
                    let p = pos[x];
                    if p == 0 || p <= pos[tm] + 1 {
                        break;
                    }
                    let prev = order[p - 1];
                    if g.preds[x].iter().any(|&(q, _)| q == prev) {
                        break;
                    }
                    order.swap(p - 1, p);
                    pos[order[p - 1]] = p - 1;
                    pos[order[p]] = p;
                    moved += 1;
                }
            }
        }
        if moved == 0 {
            continue;
        }
        if o.verify {
            if let Err(e) = verify::check_perm(m, &insts, &order, &term, o.alu_across_barrier) {
                if o.stats {
                    eprintln!("ovl: block {bi}: {e}: left alone");
                }
                continue;
            }
        }
        m.blocks[bi].insts = order.into_iter().map(|k| insts[k].clone()).collect();
        total += moved;
    }
    total
}
