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