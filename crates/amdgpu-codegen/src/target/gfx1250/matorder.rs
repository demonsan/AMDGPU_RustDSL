//! gfx1250 (optimisation item 4 of the MXFP4 round): the operands of the matrix ops first. Inside a block with two or more matrix ops
//! (WMMA), a dependence-respecting reorder issues each matrix op's operand loads (DS reads, their address arithmetic) right before it,
//! in the matrix ops' order: an instruction's key is the original position of the first matrix op that (transitively) needs it, or its
//! own position when none does, ties in the original order. The first matrix op then waits only for its own operands (the wait pass
//! counts `s_wait_dscnt` per matrix op) while the later loads land under it. Every dependence edge of `sched::depgraph` (registers,
//! memory, fences) is kept; exact kernels are never touched (the author's order).

use crate::mir::*;
use crate::sched::model::Pipe;
use std::collections::BTreeSet;

/// Reorders the blocks of `m`; true if any instruction moved.
pub fn order_for_matrix(m: &mut MFunction) -> bool {
    if m.exact {
        return false;
    }
    let mut changed = false;
    for bi in 0..m.blocks.len() {
        let n = m.blocks[bi].insts.len();
        let mats: Vec<usize> = (0..n).filter(|&i| crate::op_attrs::attrs(m.blocks[bi].insts[i].op).cost.pipe == Pipe::Mfma).collect();
        if mats.len() < 2 {
            continue;
        }
        let g = crate::sched::depgraph::build(m, &m.blocks[bi].insts, &m.blocks[bi].term, true);
        let mut key: Vec<usize> = (0..n).collect();
        for &mi in &mats {
            key[mi] = mi;
        }
        // a producer takes the smallest key of what it feeds, if that is a matrix op's (transitively)
        let mut feeds = vec![usize::MAX; n];
        for &mi in &mats {
            feeds[mi] = mi;
        }
        for i in (0..n).rev() {
            for &(j, _) in &g.succs[i] {
                feeds[i] = feeds[i].min(feeds[j]);
            }
        }
        for i in 0..n {
            if feeds[i] != usize::MAX && feeds[i] > i {
                key[i] = feeds[i];
            }
        }
        let mut indeg: Vec<usize> = g.preds.iter().map(|p| p.len()).collect();
        let mut ready: BTreeSet<(usize, usize)> = (0..n).filter(|&i| indeg[i] == 0).map(|i| (key[i], i)).collect();
        let mut order = Vec::with_capacity(n);
        while let Some(&(k, i)) = ready.iter().next() {
            ready.remove(&(k, i));
            order.push(i);
            for &(j, _) in &g.succs[i] {
                indeg[j] -= 1;
                if indeg[j] == 0 {
                    ready.insert((key[j], j));
                }
            }
        }
        if order.len() != n || order.iter().enumerate().all(|(p, &i)| p == i) {
            continue;
        }
        let old = std::mem::take(&mut m.blocks[bi].insts);
        let mut slots: Vec<Option<MInst>> = old.into_iter().map(Some).collect();
        m.blocks[bi].insts = order.iter().map(|&i| slots[i].take().expect("each once")).collect();
        changed = true;
    }
    changed
}