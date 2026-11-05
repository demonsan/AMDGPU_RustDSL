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
    if o.ldahead == 0 && o.ldsink == 0 {
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
            if o.ldahead == 0 || !is_lds_read(insts[r].op) {
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
                if !found || mf >= o.ldahead {
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