//! `mfmapair`: dependent same-accumulator MFMA pairs issued back to back (docs/mfma-dep-probe.md section 3).
//!
//! User direction: pairs of MFMAs that accumulate into the SAME accumulator (the second one's srcC is the first one's vDst: the hardware forwards the result,
//! the vendor FMHA's two K halves) are kept ADJACENT - nothing between the two - and every VALU / SALU / memory instruction is placed only after the second MFMA
//! of the pair (its "shadow"), never between the two.
//!
//! Runs on the final order of each block (after the list scheduler / slot filler, before the wait pass). For every MFMA `i` (current order) with a later
//! MFMA `j` whose srcC is `i`'s vDst, the instructions between them are split: those `j` depends on (its other operands: `ds_read`s, address chains, ...) are HOISTED
//! in front of `i`, everything else (including independent MFMAs of other accumulators) SINKS behind `j`. The pair is left alone when
//!   * an instruction `j` needs depends on `i` (it consumes `i`'s result: the two cannot be adjacent), or
//!   * the non-MFMA work that would follow `j` before the next MFMA exceeds the shadow budget (`budget` x the MFMA's pipe occupancy: more than the second MFMA can
//!     hide would stall the next MFMA - measured: with nothing between the pair only ONE MFMA shadow is available for the VALU / SALU, not two), or
//!   * it would hoist more than `MAX_HOIST` instructions.
//! Every dependence is the `depgraph` one (registers, memory, fences, M0 / SCC / VCC); the whole result is re-checked with `verify::check_perm` and the block is left
//! as it was when that fails. Waits are recomputed afterwards (the wait pass runs after this one) and hazards are inserted by the hazard pass as usual.

use super::depgraph;
use super::model;
use super::{verify, Options};
use crate::mir::*;

/// Most instructions one pair may hoist in front of its first MFMA.
pub const MAX_HOIST: usize = 12;

fn matrix(op: MOp) -> bool {
    crate::agpr::is_mfma(op)
}

fn src_c(i: &MInst, m: &MFunction) -> Option<VReg> {
    match i.srcs.get(2) {
        Some(Src::Reg(r)) => Some(m.canon(*r)),
        _ => None,
    }
}

#[derive(Default, Debug, Clone)]
pub struct PairStats {
    pub pairs: usize,
    pub already_adjacent: usize,
    pub skipped_dep: usize,
    pub skipped_budget: usize,
    pub skipped_hoist: usize,
}

/// Re-order one block; returns the new order (indices into `insts`) and the statistics.
pub fn pair_block(m: &MFunction, insts: &[MInst], term: &Term, o: &Options) -> (Vec<usize>, PairStats) {
    let n = insts.len();
    let g = depgraph::build(m, insts, term, o.alu_across_barrier);
    let mut st = PairStats::default();
    let mut order: Vec<usize> = (0..n).collect();
    let mut pos: Vec<usize> = (0..n).collect();
    let is_m: Vec<bool> = insts.iter().map(|i| matrix(i.op)).collect();
    let budget_of = |j: usize| -> u32 { (model::cost(insts[j].op).occ as f64 * o.mfmapair_budget) as u32 };
    let mut k = 0usize;
    while k < n {
        let i = order[k];
        if !is_m[i] {
            k += 1;
            continue;
        }
        let Some(d) = insts[i].dst.map(|d| m.canon(d)) else { k += 1; continue };
        // the nearest later MFMA of the current order that accumulates into d
        let Some(pj) = (k + 1..n).find(|&p| is_m[order[p]] && src_c(&insts[order[p]], m) == Some(d)) else { k += 1; continue };
        let j = order[pj];
        if pj == k + 1 {
            st.already_adjacent += 1;
            k = pj; // continue the chain at j
            continue;
        }
        // between = positions k+1 .. pj-1
        let in_between = |x: usize| pos[x] > k && pos[x] < pj;
        // need: everything between that j depends on (transitively)
        let mut need: Vec<usize> = Vec::new();
        let mut stack: Vec<usize> = g.preds[j].iter().map(|p| p.0).filter(|&p| in_between(p)).collect();
        let mut seen = vec![false; n];
        let mut dep_on_i = false;
        while let Some(x) = stack.pop() {
            if seen[x] {
                continue;
            }
            seen[x] = true;
            need.push(x);
            for &(p, _) in &g.preds[x] {
                if p == i {
                    dep_on_i = true;
                } else if in_between(p) {
                    stack.push(p);
                }
            }
        }
        if dep_on_i {
            st.skipped_dep += 1;
            k += 1;
            continue;
        }
        if need.len() > MAX_HOIST {
            st.skipped_hoist += 1;
            k += 1;
            continue;
        }
        need.sort_by_key(|&x| pos[x]);
        let mut sink: Vec<usize> = (k + 1..pj).map(|p| order[p]).filter(|&x| !seen[x]).collect();
        // vendor style: up to `mfmapair_gap` non-MFMA instructions may stay BETWEEN the two MFMAs (the first MFMA's shadow is not left empty); an instruction can stay only
        // when everything it depends on inside the gap region is already placed before the pair (hoisted) or stays too (never an MFMA that sinks)
        let mut keep: Vec<usize> = Vec::new();
        if o.mfmapair_gap > 0 {
            for &x in sink.clone().iter() {
                if keep.len() >= o.mfmapair_gap {
                    break;
                }
                if is_m[x] {
                    continue;
                }
                let ok = g.preds[x].iter().all(|&(p, _)| !in_between(p) || seen[p] || keep.contains(&p));
                if ok {
                    keep.push(x);
                }
            }
            sink.retain(|x| !keep.contains(x));
        }
        // budget: the non-MFMA work that will follow j before the next MFMA
        let mut next_pos = pj + 1;
        let mut cost: u32 = sink.iter().filter(|&&x| !is_m[x]).map(|&x| model::cost(insts[x].op).issue).sum();
        // (a sunk MFMA ends the shadow: only the non-MFMAs before the first sunk / following MFMA count)
        if let Some(first_m) = sink.iter().position(|&x| is_m[x]) {
            cost = sink[..first_m].iter().map(|&x| model::cost(insts[x].op).issue).sum();
        } else {
            while next_pos < n && !is_m[order[next_pos]] {
                cost += model::cost(insts[order[next_pos]].op).issue;
                next_pos += 1;
            }
        }
        if cost > budget_of(j) {
            st.skipped_budget += 1;
            k += 1;
            continue;
        }
        // new order: [... need..., i, j, sink..., rest]
        let mut new: Vec<usize> = Vec::with_capacity(n);
        new.extend(order[..k].iter().copied());
        new.extend(need.iter().copied());
        new.push(i);
        new.extend(keep.iter().copied());
        new.push(j);
        new.append(&mut sink);
        new.extend(order[pj + 1..].iter().copied());
        order = new;
        for (p, &x) in order.iter().enumerate() {
            pos[x] = p;
        }
        st.pairs += 1;
        k = pos[j]; // continue the chain at j
    }
    (order, st)
}

/// Apply `mfmapair` to every block with at least two MFMAs. Returns the number of pairs made adjacent.
pub fn run(m: &mut MFunction, o: &Options) -> Result<usize, String> {
    let mut total = 0;
    for bi in 0..m.blocks.len() {
        let insts = m.blocks[bi].insts.clone();
        if insts.iter().filter(|i| matrix(i.op)).count() < 2 {
            continue;
        }
        let term = m.blocks[bi].term.clone();
        let (order, st) = pair_block(m, &insts, &term, o);
        if st.pairs == 0 {
            if o.stats {
                eprintln!("mfmapair: block {bi}: {st:?}");
            }
            continue;
        }
        if o.verify {
            if let Err(e) = verify::check_perm(m, &insts, &order, &term, o.alu_across_barrier) {
                if o.stats {
                    eprintln!("mfmapair: block {bi}: {e}: left alone");
                }
                continue;
            }
        }
        if o.stats {
            eprintln!("mfmapair: block {bi}: {st:?}");
        }
        total += st.pairs;
        m.blocks[bi].insts = order.into_iter().map(|k| insts[k].clone()).collect();
    }
    Ok(total)
}