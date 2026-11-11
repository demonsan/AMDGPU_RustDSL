//! Jump threading on flags known per incoming edge (lowering token `jthread`; docs/two-weavers.md section 5.1, the vendor-FMHA two-tile
//! loop): a block whose branch depends only on values that are constants on a given incoming edge is cloned for that edge with the branch
//! replaced by a jump to the known successor. The typical case: a loop with two exits that set a flag (`idx = 1` / `idx = 0`) right
//! before each exit, followed by `if (idx == 1) { drain 1 } else { drain 0 }` - each exit then goes straight to its drain.
//!
//! The clone keeps every instruction of the block (the address work both drains use stays on both paths); only the terminator changes.
//! The evaluation is per wave: every value involved must be the same in every lane (constants), so the branch outcome is exact.

use crate::mir::*;
use std::collections::HashMap;

/// A known value: a 32-bit constant (every lane) or a lane mask that is all-ones / all-zero (vcc / scc as a boolean).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum K {
    Int(i64),
    Bool(bool),
}

const VCC: VReg = u32::MAX;
const SCC: VReg = u32::MAX - 1;

/// Vregs with exactly one definition, a `v_mov_b32` / `s_mov_b32` of an immediate: constants everywhere.
fn global_consts(m: &MFunction) -> HashMap<VReg, i64> {
    let mut defs: HashMap<VReg, (usize, Option<i64>)> = HashMap::new();
    for b in &m.blocks {
        for i in &b.insts {
            if let Some(d) = i.dst {
                let c = match (i.op, i.srcs.as_slice()) {
                    (MOp::VMovB32 | MOp::SMovB32, []) => Some(i.imm),
                    (MOp::VMovB32 | MOp::SMovB32, [Src::Imm(k)]) => Some(*k),
                    _ => None,
                };
                let e = defs.entry(m.canon(d)).or_insert((0, c));
                e.0 += 1;
                if e.0 > 1 {
                    e.1 = None;
                }
            }
        }
    }
    defs.into_iter().filter_map(|(r, (n, c))| if n == 1 { c.map(|c| (r, c)) } else { None }).collect()
}

/// Forward constant evaluation of `insts` from `env`; unknown results remove the destination from `env`.
fn eval(m: &MFunction, insts: &[MInst], env: &mut HashMap<VReg, K>, gc: &HashMap<VReg, i64>) {
    for i in insts {
        let val = |env: &HashMap<VReg, K>, s: &Src| -> Option<i64> {
            match s {
                Src::Imm(k) => Some(*k),
                Src::Reg(r) => {
                    let r = m.canon(*r);
                    match env.get(&r) {
                        Some(K::Int(v)) => Some(*v),
                        Some(K::Bool(_)) => None,
                        None => gc.get(&r).copied(),
                    }
                }
            }
        };
        let s = |k: usize| i.srcs.get(k);
        let cmp = |env: &HashMap<VReg, K>, f: &dyn Fn(i64, i64) -> bool, unsigned: bool| -> Option<K> {
            let (a, b) = (val(env, s(0)?)?, val(env, s(1)?)?);
            Some(K::Bool(if unsigned { f(a as u32 as i64, b as u32 as i64) } else { f(a as i32 as i64, b as i32 as i64) }))
        };
        let (dst, out): (Option<VReg>, Option<K>) = match i.op {
            MOp::VMovB32 | MOp::SMovB32 => (i.dst, if i.srcs.is_empty() { Some(K::Int(i.imm)) } else { s(0).and_then(|x| val(env, x)).map(K::Int) }),
            // SGPR -> VGPR copy (a scalarized flag read by a vector compare)
            MOp::VMovFromS => (i.dst, s(0).and_then(|x| val(env, x)).map(K::Int)),
            // scc ? srcs[0] : srcs[1]
            MOp::SCselectB32 => {
                let o = match env.get(&SCC) {
                    Some(K::Bool(c)) => s(if *c { 0 } else { 1 }).and_then(|x| val(env, x)).map(K::Int),
                    _ => None,
                };
                (i.dst, o)
            }
            MOp::VCmpEqI32 | MOp::VCmpEqU32 => (Some(VCC), cmp(env, &|a, b| a == b, false)),
            MOp::VCmpNeI32 => (Some(VCC), cmp(env, &|a, b| a != b, false)),
            MOp::VCmpLtI32 => (Some(VCC), cmp(env, &|a, b| a < b, false)),
            MOp::VCmpGeI32 => (Some(VCC), cmp(env, &|a, b| a >= b, false)),
            MOp::VCmpGtI32 => (Some(VCC), cmp(env, &|a, b| a > b, false)),
            MOp::VCmpLeI32 => (Some(VCC), cmp(env, &|a, b| a <= b, false)),
            MOp::VCmpLtU32 => (Some(VCC), cmp(env, &|a, b| a < b, true)),
            MOp::VCmpGeU32 => (Some(VCC), cmp(env, &|a, b| a >= b, true)),
            MOp::SCmpLtI32 => (Some(SCC), cmp(env, &|a, b| a < b, false)),
            MOp::VCndmask => {
                // srcs[0] when vcc is false, srcs[1] when true
                let o = match env.get(&VCC) {
                    Some(K::Bool(c)) => s(if *c { 1 } else { 0 }).and_then(|x| val(env, x)).map(K::Int),
                    _ => None,
                };
                (i.dst, o)
            }
            _ => {
                // anything else: its destination (and vcc / scc when it writes them) becomes unknown
                if crate::op_attrs::attrs(i.op).defines_vcc || matches!(i.op, MOp::VPtrAdd) {
                    env.remove(&VCC);
                }
                if crate::op_attrs::attrs(i.op).cost.pipe == crate::sched::model::Pipe::Salu {
                    env.remove(&SCC);
                }
                (i.dst, None)
            }
        };
        if let Some(d) = dst {
            let d = if d == VCC || d == SCC { d } else { m.canon(d) };
            match out {
                Some(k) => {
                    env.insert(d, k);
                }
                None => {
                    env.remove(&d);
                }
            }
        }
    }
}

fn taken(on: Pred, env: &HashMap<VReg, K>) -> Option<bool> {
    match on {
        Pred::Vccnz => match env.get(&VCC) {
            Some(K::Bool(b)) => Some(*b),
            _ => None,
        },
        Pred::Vccz => match env.get(&VCC) {
            Some(K::Bool(b)) => Some(!*b),
            _ => None,
        },
        Pred::Scc1 => match env.get(&SCC) {
            Some(K::Bool(b)) => Some(*b),
            _ => None,
        },
        Pred::Scc0 => match env.get(&SCC) {
            Some(K::Bool(b)) => Some(!*b),
            _ => None,
        },
    }
}

/// Threads every edge whose branch outcome is known. Returns the number of edges threaded.
pub fn run(m: &mut MFunction) -> usize {
    let gc = global_consts(m);
    let idx: HashMap<u32, usize> = m.blocks.iter().enumerate().map(|(i, b)| (b.label, i)).collect();
    let mut next_label = m.blocks.iter().map(|b| b.label).max().unwrap_or(0) + 1;
    // (predecessor index, which edge: 0 = Goto / CondGoto target, 1 = CondGoto fall, block to clone, known successor)
    let mut edits: Vec<(usize, u8, usize, u32)> = Vec::new();
    for (pi, p) in m.blocks.iter().enumerate() {
        let succs: Vec<(u8, u32)> = match &p.term {
            Term::Goto(t) => vec![(0, *t)],
            Term::CondGoto { target, fall, .. } => vec![(0, *target), (1, *fall)],
            Term::Endpgm => vec![],
        };
        for (which, s) in succs {
            let Some(&ei) = idx.get(&s) else { continue };
            if ei == pi {
                continue;
            }
            let e = &m.blocks[ei];
            let Term::CondGoto { on, target, fall } = e.term else { continue };
            // the facts on this edge: the predecessor's own definitions (copies of constants)
            let mut env: HashMap<VReg, K> = HashMap::new();
            eval(m, &p.insts, &mut env, &gc);
            // a conditional predecessor's vcc / scc is its branch condition, not a value on the edge
            env.remove(&VCC);
            env.remove(&SCC);
            eval(m, &e.insts, &mut env, &gc);
            if let Some(t) = taken(on, &env) {
                edits.push((pi, which, ei, if t { target } else { fall }));
            }
        }
    }
    let n = edits.len();
    for (pi, which, ei, succ) in edits {
        let mut clone = m.blocks[ei].clone();
        clone.label = next_label;
        next_label += 1;
        clone.term = Term::Goto(succ);
        let old = m.blocks[ei].label;
        match &mut m.blocks[pi].term {
            Term::Goto(t) if *t == old => *t = clone.label,
            Term::CondGoto { target, fall, .. } => {
                if which == 0 && *target == old {
                    *target = clone.label;
                } else if which == 1 && *fall == old {
                    *fall = clone.label;
                }
            }
            _ => {}
        }
        m.blocks.push(clone);
    }
    if n > 0 {
        remove_unreachable(m);
    }
    n
}

/// Drops blocks no path from the entry reaches.
fn remove_unreachable(m: &mut MFunction) {
    let idx: HashMap<u32, usize> = m.blocks.iter().enumerate().map(|(i, b)| (b.label, i)).collect();
    let mut seen = vec![false; m.blocks.len()];
    let mut stack = vec![0usize];
    while let Some(i) = stack.pop() {
        if seen[i] {
            continue;
        }
        seen[i] = true;
        let succ: Vec<u32> = match &m.blocks[i].term {
            Term::Goto(t) => vec![*t],
            Term::CondGoto { target, fall, .. } => vec![*target, *fall],
            Term::Endpgm => vec![],
        };
        for s in succ {
            if let Some(&j) = idx.get(&s) {
                stack.push(j);
            }
        }
    }
    let mut k = 0;
    m.blocks.retain(|_| {
        let keep = seen[k];
        k += 1;
        keep
    });
}

/// `coldlayout`: the blocks of branches hinted unlikely (`Block::cold`) move after every hot block (stable order on both sides). Every
/// terminator names its targets, so the order changes only which edges fall through; the cold path ends in a jump back to its join.
pub fn cold_layout(m: &mut MFunction) -> usize {
    if m.blocks.first().map_or(true, |b| b.cold) {
        return 0;
    }
    let n = m.blocks.iter().filter(|b| b.cold).count();
    if n > 0 {
        let blocks = std::mem::take(&mut m.blocks);
        let (hot, cold): (Vec<Block>, Vec<Block>) = blocks.into_iter().partition(|b| !b.cold);
        m.blocks = hot;
        m.blocks.extend(cold);
    }
    n
}
