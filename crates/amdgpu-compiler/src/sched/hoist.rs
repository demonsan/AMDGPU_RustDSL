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
    crate::op_attrs::attrs(op).is_lds_read()
}

/// Plain ALU / scalar instructions that may travel with a hoisted load: no memory access, no fence, not pinned, not a matrix op.
fn chain_movable(i: &MInst, e: &Effects) -> bool {
    e.mem.is_none() && e.fence == Fence::None && !e.smem && !e.pinned && i.op != MOp::SchedGroup && crate::op_attrs::attrs(i.op).mfma.is_none()
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
            // `hoistchain`: pure ALU producers of the load's address inside the hoist window move with it
            let floor = p.saturating_sub(o.hoist_dist);
            let mut members: Vec<usize> = vec![idx];
            if o.hoist_chain {
                let mut stack = vec![idx];
                while let Some(y) = stack.pop() {
                    for &(x, _) in &g.preds[y] {
                        if !members.contains(&x) && pos[x] >= floor && chain_movable(&insts[x], &g.eff[x]) {
                            members.push(x);
                            stack.push(x);
                        }
                    }
                }
            }
            let q_min = members.iter().flat_map(|&y| g.preds[y].iter().filter(|&&(x, _)| !members.contains(&x)).map(|&(x, _)| pos[x] + 1)).max().unwrap_or(0);
            let mut q = q_min.max(floor);
            // pressure along the span: every position in q..p gains the destination's live range
            while q < p {
                let peak = (q..p).map(|k| live[order[k]]).max().unwrap_or(0);
                if peak + dst_w <= o.max_vgprs as i64 {
                    break;
                }
                q += 1;
            }
            if q < p {
                // movers: the members positioned at or after q, in program order (members before q stay: their moved successors land after them)
                let mut movers: Vec<usize> = members.iter().copied().filter(|&y| pos[y] >= q).collect();
                movers.sort_by_key(|&y| pos[y]);
                if movers.is_empty() {
                    continue;
                }
                let extra: i64 = movers.iter().filter(|&&y| y != idx).map(|&y| insts[y].dst.map(|d| width(canon(d))).unwrap_or(0)).sum();
                if std::env::var("AMDC_HOIST_DEBUG").map_or(false, |v| !v.is_empty() && v != "0") {
                    eprintln!("hoist block {bi}: load {idx} (pos {p}) q={q} members={:?} movers={:?} ops={:?}", members.iter().map(|&y| (y, pos[y])).collect::<Vec<_>>(), movers.iter().map(|&y| (y, pos[y])).collect::<Vec<_>>(), movers.iter().map(|&y| insts[y].op).collect::<Vec<_>>());
                }
                let first_old = pos[movers[0]];
                let lo = q.min(first_old);
                order.retain(|k| !movers.contains(k));
                for (n, &y) in movers.iter().enumerate() {
                    order.insert(q + n, y);
                }
                for k in lo..=p {
                    pos[order[k]] = k;
                }
                for k in (q + movers.len())..=p {
                    live[order[k]] += dst_w + extra;
                }
                moved += 1;
            }
        }        if moved > 0 {
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