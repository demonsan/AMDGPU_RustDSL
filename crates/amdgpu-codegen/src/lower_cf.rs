//! E5 (docs/exact-mode.md section 5): the layout of out-of-line `if` branches placed with `Stmt::BranchPlace` at a `Stmt::LayoutAnchor`.
//!
//! Lowering tags every block of such a then-branch with its anchor (`Block::place`) and starts a block at each anchor. This pass moves the
//! tagged blocks, in their order, in front of the anchor's block. Every terminator names its targets, so only the fall-through edges change:
//! the condition's taken edge enters the out-of-line branch and its end jumps to the join. A kernel without anchors is untouched.
//! Jumps to an empty block that only jumps on are retargeted (no instruction changes; the exits of a loop reach their join in one branch).

use crate::mir::{MFunction, Term};
use std::collections::HashMap;

pub fn place_layout(m: &mut MFunction) -> usize {
    let anchors: Vec<(u32, u32)> = m.blocks.iter().filter_map(|b| b.anchor.map(|a| (a, b.label))).collect();
    if anchors.is_empty() {
        return 0;
    }
    let mut moved = 0;
    for &(id, label) in &anchors {
        let blocks = std::mem::take(&mut m.blocks);
        let (placed, rest): (Vec<_>, Vec<_>) = blocks.into_iter().partition(|b| b.place == Some(id));
        moved += placed.len();
        let at = rest.iter().position(|b| b.label == label).unwrap_or(rest.len());
        let mut out = Vec::with_capacity(rest.len() + placed.len());
        let mut rest = rest.into_iter();
        out.extend(rest.by_ref().take(at));
        out.extend(placed);
        out.extend(rest);
        m.blocks = out;
    }
    forward_empty_jumps(m);
    moved
}

/// Retargets edges into blocks without instructions whose terminator is an unconditional jump.
fn forward_empty_jumps(m: &mut MFunction) {
    let fwd: HashMap<u32, u32> = m.blocks.iter().filter(|b| b.insts.is_empty()).filter_map(|b| if let Term::Goto(t) = b.term { Some((b.label, t)) } else { None }).collect();
    let resolve = |mut l: u32| {
        let mut n = 0;
        while let Some(&t) = fwd.get(&l) {
            if t == l || n > 64 {
                break;
            }
            l = t;
            n += 1;
        }
        l
    };
    for b in m.blocks.iter_mut() {
        b.term = match b.term {
            Term::Goto(t) => Term::Goto(resolve(t)),
            Term::CondGoto { on, target, fall } => Term::CondGoto { on, target: resolve(target), fall: resolve(fall) },
            Term::Endpgm => Term::Endpgm,
        };
    }
    // the forwarding blocks (and any block no edge reaches any more) are gone from the layout
    let idx: HashMap<u32, usize> = m.blocks.iter().enumerate().map(|(i, b)| (b.label, i)).collect();
    let mut seen = vec![false; m.blocks.len()];
    let mut stack = vec![0usize];
    while let Some(i) = stack.pop() {
        if i >= seen.len() || seen[i] {
            continue;
        }
        seen[i] = true;
        let succ: Vec<u32> = match m.blocks[i].term {
            Term::Goto(t) => vec![t],
            Term::CondGoto { target, fall, .. } => vec![target, fall],
            Term::Endpgm => vec![],
        };
        stack.extend(succ.iter().filter_map(|l| idx.get(l).copied()));
    }
    let mut k = 0;
    m.blocks.retain(|_| {
        let keep = seen[k];
        k += 1;
        keep
    });
}