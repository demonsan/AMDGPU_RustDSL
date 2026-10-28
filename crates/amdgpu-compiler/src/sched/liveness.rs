//! Block-level liveness on canonical virtual registers (same rules as `regalloc`: any
//! destination write kills, aliases map to their tuple).

use std::collections::{HashMap, HashSet};

use crate::mir::*;

pub struct Liveness {
    pub live_in: Vec<HashSet<VReg>>,
    pub live_out: Vec<HashSet<VReg>>,
    pub succs: Vec<Vec<usize>>,
}

pub fn successors(m: &MFunction) -> Vec<Vec<usize>> {
    let label_to_block: HashMap<u32, usize> = m.blocks.iter().enumerate().map(|(i, b)| (b.label, i)).collect();
    let mut succs: Vec<Vec<usize>> = vec![Vec::new(); m.blocks.len()];
    for (i, b) in m.blocks.iter().enumerate() {
        match &b.term {
            Term::Goto(t) => {
                if let Some(j) = label_to_block.get(t) {
                    succs[i].push(*j);
                }
            }
            Term::CondGoto { target, fall, .. } => {
                if let Some(j) = label_to_block.get(target) {
                    succs[i].push(*j);
                }
                if let Some(j) = label_to_block.get(fall) {
                    if !succs[i].contains(j) {
                        succs[i].push(*j);
                    }
                }
            }
            Term::Endpgm => {}
        }
    }
    succs
}

pub fn compute(m: &MFunction) -> Liveness {
    let nb = m.blocks.len();
    let succs = successors(m);
    let mut use_set: Vec<HashSet<VReg>> = vec![HashSet::new(); nb];
    let mut def_set: Vec<HashSet<VReg>> = vec![HashSet::new(); nb];
    for (i, b) in m.blocks.iter().enumerate() {
        for inst in &b.insts {
            for r in inst.uses() {
                let r = m.canon(r);
                if !def_set[i].contains(&r) {
                    use_set[i].insert(r);
                }
            }
            if let Some(d) = inst.dst {
                def_set[i].insert(m.canon(d));
            }
        }
    }
    let mut live_in: Vec<HashSet<VReg>> = vec![HashSet::new(); nb];
    let mut live_out: Vec<HashSet<VReg>> = vec![HashSet::new(); nb];
    let mut changed = true;
    let mut guard = 0;
    while changed && guard < 10_000 {
        changed = false;
        guard += 1;
        for i in (0..nb).rev() {
            let mut out: HashSet<VReg> = HashSet::new();
            for s in &succs[i] {
                for r in &live_in[*s] {
                    out.insert(*r);
                }
            }
            let mut inn = use_set[i].clone();
            for r in &out {
                if !def_set[i].contains(r) {
                    inn.insert(*r);
                }
            }
            if inn != live_in[i] || out != live_out[i] {
                live_in[i] = inn;
                live_out[i] = out;
                changed = true;
            }
        }
    }
    Liveness { live_in, live_out, succs }
}