//! CFG cleanup run by the scheduler pipeline: removes the empty latch block the lowering creates
//! for every `break_if` (a loop then costs one taken branch per iteration instead of two) and
//! turns on fall-through elision in the emitter.

use crate::mir::*;

fn invert(p: Pred) -> Pred {
    match p {
        Pred::Scc0 => Pred::Scc1,
        Pred::Scc1 => Pred::Scc0,
        Pred::Vccnz => Pred::Vccz,
        Pred::Vccz => Pred::Vccnz,
    }
}

pub fn cleanup(m: &mut MFunction) -> usize {
    let mut removed = 0;
    loop {
        let mut target: Option<(usize, usize)> = None; // (branching block, empty block)
        for (ai, a) in m.blocks.iter().enumerate() {
            if let Term::CondGoto { fall, target: x, .. } = &a.term {
                let Some(fi) = m.blocks.iter().position(|b| b.label == *fall) else { continue };
                let f = &m.blocks[fi];
                // fall block empty, jumps back (to a block that is not itself), laid out right after `a`,
                // and the taken target is the block laid out after it.
                if fi == ai + 1 && f.insts.is_empty() {
                    if let Term::Goto(h) = f.term {
                        if h != f.label && m.blocks.get(fi + 1).map(|n| n.label) == Some(*x) {
                            target = Some((ai, fi));
                            break;
                        }
                    }
                }
            }
        }
        let Some((ai, fi)) = target else { break };
        let flabel = m.blocks[fi].label;
        let Term::Goto(h) = m.blocks[fi].term.clone() else { break };
        let Term::CondGoto { on, target: x, .. } = m.blocks[ai].term.clone() else { break };
        m.blocks[ai].term = Term::CondGoto { on: invert(on), target: h, fall: x };
        let alabel = m.blocks[ai].label;
        for l in m.loops.iter_mut() {
            if l.1 == flabel {
                l.1 = alabel;
            }
        }
        m.blocks.remove(fi);
        removed += 1;
    }
    m.elide_fallthrough = true;
    removed
}
/// Remove `v_cndmask` results nobody reads (the 0/1 value the lowering builds for a loop-exit
/// condition) together with the compare feeding it when nothing else reads that vcc.
pub fn dce_dead_selects(m: &mut MFunction) -> usize {
    let mut removed = 0;
    loop {
        let mut used: std::collections::HashSet<VReg> = std::collections::HashSet::new();
        for b in &m.blocks {
            for i in &b.insts {
                for r in i.uses() {
                    used.insert(m.canon(r));
                }
            }
        }
        let mut changed = false;
        for bi in 0..m.blocks.len() {
            let mut j = 0;
            while j < m.blocks[bi].insts.len() {
                let dead = {
                    let i = &m.blocks[bi].insts[j];
                    i.op == MOp::VCndmask && i.dst.map_or(false, |d| !used.contains(&m.canon(d)))
                };
                if !dead {
                    j += 1;
                    continue;
                }
                let writer = |i: &MInst| i.op.defines_vcc() || i.op == MOp::VPtrAdd;
                let wi = m.blocks[bi].insts[..j].iter().rposition(|i| writer(i));
                m.blocks[bi].insts.remove(j);
                removed += 1;
                changed = true;
                if let Some(w) = wi {
                    if m.blocks[bi].insts[w].op.defines_vcc() && !m.blocks[bi].insts[w].op.consumes_vcc() && m.blocks[bi].insts[w].dst.is_none() {
                        // is that vcc still read? by a later consumer before the next writer, or by the terminator
                        let rest = &m.blocks[bi].insts[w + 1..];
                        let next_w = rest.iter().position(|i| writer(i));
                        let upto = next_w.unwrap_or(rest.len());
                        let read = rest[..upto].iter().any(|i| i.op.consumes_vcc())
                            || (next_w.is_none() && matches!(m.blocks[bi].term, Term::CondGoto { on: Pred::Vccnz | Pred::Vccz, .. }));
                        if !read {
                            m.blocks[bi].insts.remove(w);
                            removed += 1;
                            j = j.saturating_sub(1);
                        }
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    removed
}