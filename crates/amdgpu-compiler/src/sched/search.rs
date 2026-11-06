//! `search=N`: simulator-guided local search over the instruction order of the HOT PATH of every loop.
//!
//! The machine-model simulator (`sim.rs`, validated against hardware: it predicts the 1-wave FMHA's hjoin regressions and the `ovl` gain) is the objective. A move picks one
//! instruction of a hot-path block, draws a new position inside its dependence window (between the latest predecessor and the earliest successor of the dependence graph, so every
//! visited order is a legal permutation by construction) and keeps it when the simulated steady-state cycles per iteration do not grow (a small, decaying temperature accepts a few
//! worse ones). The allocator's own pressure (`regalloc::pressure_by_block`) of every touched block must stay within what the incoming order needed (+ `pslack`): checked every 64
//! accepted moves, a violation restores the last good checkpoint. The result is re-verified with `verify::check_perm`.

use super::depgraph;
use super::effects::Fence;
use super::sim;
use super::{verify, Options};
use crate::mir::*;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Returns the number of accepted moves that survived into the result.
pub fn run(m: &mut MFunction, o: &Options) -> usize {
    let mut total = 0;
    let loops = m.loops.clone();
    let label_to_block: std::collections::HashMap<u32, usize> = m.blocks.iter().enumerate().map(|(i, b)| (b.label, i)).collect();
    for (h, l) in loops {
        let (Some(&hi), Some(&li)) = (label_to_block.get(&h), label_to_block.get(&l)) else { continue };
        let path = sim::hot_path(m, hi, li);
        if path.is_empty() {
            continue;
        }
        let parts = sim::prog_parts(m, &path, hi);
        let graphs: Vec<Option<depgraph::Graph>> = path
            .iter()
            .map(|&b| {
                let insts = &m.blocks[b].insts;
                if insts.len() < 8 || super::phase::has_markers(insts) {
                    None
                } else {
                    Some(depgraph::build(m, insts, &m.blocks[b].term, o.alu_across_barrier))
                }
            })
            .collect();
        if graphs.iter().all(|g| g.is_none()) {
            continue;
        }
        let mut orders: Vec<Vec<usize>> = parts.insts.iter().map(|v| (0..v.len()).collect()).collect();
        let mut pos: Vec<Vec<usize>> = orders.clone();
        let cost = |orders: &[Vec<usize>]| sim::simulate_prog(&sim::assemble(&parts, orders), 1, 10);
        let c0 = cost(&orders);
        let mut cur = c0;
        let mut best = c0;
        let mut best_orders = orders.clone();
        let real0 = crate::regalloc::pressure_by_block(m);
        let pressure_ok = |m: &MFunction, orders: &[Vec<usize>]| -> bool {
            let mut mm = m.clone();
            for (k, &b) in path.iter().enumerate() {
                if graphs[k].is_some() {
                    mm.blocks[b].insts = orders[k].iter().map(|&i| m.blocks[b].insts[i].clone()).collect();
                }
            }
            let real = crate::regalloc::pressure_by_block(&mm);
            path.iter().all(|&b| real[b] <= real0[b] + o.pslack)
        };
        let weights: Vec<usize> = graphs.iter().enumerate().map(|(k, g)| if g.is_some() { orders[k].len() } else { 0 }).collect();
        let wsum: usize = weights.iter().sum();
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ (h as u64 + 1));
        let mut checkpoint = (orders.clone(), cur);
        let mut accepted_since = 0usize;
        let iters = o.search;
        for it in 0..iters {
            // block by length, instruction uniformly
            let mut r = rng.below(wsum);
            let mut k = 0;
            while r >= weights[k] {
                r -= weights[k];
                k += 1;
            }
            let g = graphs[k].as_ref().unwrap();
            let n = orders[k].len();
            let x = rng.below(n);
            let e = &g.eff[x];
            if e.fence != Fence::None || e.pinned {
                continue;
            }
            let p = pos[k][x];
            let lo = g.preds[x].iter().map(|&(q, _)| pos[k][q] + 1).max().unwrap_or(0);
            let hi_ = g.succs[x].iter().map(|&(q, _)| pos[k][q]).min().unwrap_or(n); // exclusive
            if hi_ <= lo + 1 {
                continue;
            }
            let np = if rng.below(2) == 0 {
                let d = 1 + rng.below(12);
                if rng.below(2) == 0 { p.saturating_sub(d).max(lo) } else { (p + d).min(hi_ - 1) }
            } else {
                lo + rng.below(hi_ - lo)
            };
            if np == p || np < lo || np >= hi_ {
                continue;
            }
            // apply: remove x at p, insert at np (positions shift by one for what lies in between)
            let apply = |orders: &mut Vec<usize>, pos: &mut Vec<usize>| {
                let v = orders.remove(p);
                orders.insert(np, v);
                let (a, b) = (p.min(np), p.max(np));
                for q in a..=b {
                    pos[orders[q]] = q;
                }
            };
            apply(&mut orders[k], &mut pos[k]);
            let c = cost(&orders);
            let temp = 0.004 * c0 * (1.0 - it as f64 / iters.max(1) as f64);
            let ok = c <= cur || (temp > 0.0 && rng.unit() < (-(c - cur) / temp).exp());
            if ok {
                cur = c;
                accepted_since += 1;
                if c < best {
                    if accepted_since >= 64 && !pressure_ok(m, &orders) {
                        orders = checkpoint.0.clone();
                        for kk in 0..orders.len() {
                            for (q, &i) in orders[kk].iter().enumerate() {
                                pos[kk][i] = q;
                            }
                        }
                        cur = checkpoint.1;
                        accepted_since = 0;
                        continue;
                    }
                    best = c;
                    best_orders = orders.clone();
                    if accepted_since >= 64 {
                        checkpoint = (orders.clone(), c);
                        accepted_since = 0;
                    }
                }
            } else {
                // revert: move x back from np to p
                let v = orders[k].remove(np);
                orders[k].insert(p, v);
                let (a, b) = (p.min(np), p.max(np));
                for q in a..=b {
                    pos[k][orders[k][q]] = q;
                }
            }
        }
        if best >= c0 || !pressure_ok(m, &best_orders) {
            if o.stats {
                eprintln!("search: loop@{h}: no usable improvement ({c0} -> best {best}, pressure ok {})", best < c0);
            }
            continue;
        }
        // apply and verify per block
        let mut changed = 0;
        let mut staged: Vec<(usize, Vec<MInst>)> = Vec::new();
        let mut legal = true;
        for (k, &b) in path.iter().enumerate() {
            if graphs[k].is_none() || best_orders[k].iter().enumerate().all(|(q, &i)| q == i) {
                continue;
            }
            let insts = m.blocks[b].insts.clone();
            if o.verify {
                if let Err(e) = verify::check_perm(m, &insts, &best_orders[k], &m.blocks[b].term, o.alu_across_barrier) {
                    if o.stats {
                        eprintln!("search: loop@{h}: block {b}: {e}: search result rejected");
                    }
                    legal = false;
                    break;
                }
            }
            staged.push((b, best_orders[k].iter().map(|&i| insts[i].clone()).collect()));
            changed += best_orders[k].iter().enumerate().filter(|(q, &i)| *q != i).count();
        }
        if !legal {
            continue;
        }
        for (b, v) in staged {
            m.blocks[b].insts = v;
        }
        if o.stats {
            eprintln!("search: loop@{h}: simulated cycles/iteration {c0:.0} -> {best:.0} ({:.1}%), {changed} instructions moved", (best / c0 - 1.0) * 100.0);
        }
        total += changed;
    }
    total
}