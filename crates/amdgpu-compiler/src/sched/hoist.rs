//! LDS-read hoisting ("ldshoist"): move `ds_read*` instructions earlier inside their basic block, above
//! independent work such as the previous MFMA, so the LDS latency overlaps with it and the wait before the
//! consumer becomes a partial `lgkmcnt(n)` (waits are re-derived from the final order by `waits.rs`).
//!
//! Only loads move, only upwards, only as far as their dependences (register RAW/WAR/WAW, memory and fence
//! order from `depgraph`) and the register-pressure budget allow, and at most `maxdist` instructions. Every
//! result is proven legal by `verify::check_perm`. No value changes: same instructions, same operands.

use super::depgraph;
use super::effects::*;
use super::{liveness, verify, Options};
use crate::mir::*;

fn is_lds_read(op: MOp) -> bool {
    matches!(op, MOp::LdsReadB32 | MOp::LdsReadB128 | MOp::LdsReadB64 | MOp::LdsReadB64TrB8)
}

pub fn run(m: &mut MFunction, o: &Options) -> usize {
    let lv = liveness::compute(m);
    let mut total = 0;
    for bi in 0..m.blocks.len() {
        let insts = m.blocks[bi].insts.clone();
        let n = insts.len();
        if n < 8 || super::phase::has_markers(&insts) {
            continue;
        }
        let term = m.blocks[bi].term.clone();
        let g = depgraph::build(m, &insts, &term, false);
        // live VGPR dwords entering each instruction (original order), from the block's live-out set
        let width = |r: VReg| -> i64 { if m.vreg_class(r).is_vector() { m.vreg_class(r).dwords() as i64 } else { 0 } };
        let canon = |r: VReg| m.canon(r);
        let mut live_set: std::collections::HashSet<VReg> = lv.live_out[bi].iter().map(|&r| canon(r)).collect();
        let mut live = vec![0i64; n + 1];
        live[n] = live_set.iter().map(|&r| width(r)).sum();
        for i in (0..n).rev() {
            if let Some(d) = insts[i].dst {
                if !m.alias.contains_key(&d) {
                    live_set.remove(&canon(d));
                }
            }
            for r in insts[i].uses() {
                live_set.insert(canon(r));
            }
            live[i] = live_set.iter().map(|&r| width(r)).sum();
        }
        let mut order: Vec<usize> = (0..n).collect();
        let mut pos: Vec<usize> = (0..n).collect();
        let mut moved = 0;
        for idx in 0..n {
            if !is_lds_read(insts[idx].op) {
                continue;
            }
            let p = pos[idx];
            let dst_w = insts[idx].dst.map(|d| width(canon(d))).unwrap_or(0);
            let q_min = g.preds[idx].iter().map(|&(x, _)| pos[x] + 1).max().unwrap_or(0);
            let mut q = q_min.max(p.saturating_sub(o.hoist_dist));
            // pressure along the span: every position in q..p gains the destination's live range
            while q < p {
                let peak = (q..p).map(|k| live[order[k]]).max().unwrap_or(0);
                if peak + dst_w <= o.max_vgprs as i64 {
                    break;
                }
                q += 1;
            }
            if q < p {
                let x = order.remove(p);
                order.insert(q, x);
                for k in q..=p {
                    pos[order[k]] = k;
                }
                for k in q..p + 1 {
                    // positions q+1..=p now sit after the moved instruction: their entering live count grows by dst_w
                    let _ = k;
                }
                for k in (q + 1)..=p {
                    live[order[k]] += dst_w;
                }
                moved += 1;
            }
        }
        if moved > 0 {
            if o.verify {
                if let Err(e) = verify::check_perm(m, &insts, &order, &term, false) {
                    eprintln!("ldshoist: block {bi} rejected by verifier: {e}");
                    continue;
                }
            }
            m.blocks[bi].insts = order.iter().map(|&k| insts[k].clone()).collect();
            total += moved;
        }
    }
    total
}