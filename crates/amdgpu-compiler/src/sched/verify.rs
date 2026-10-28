//! Independent schedule verifier.
//!
//! Proves that a permutation of a straight-line region is a legal reordering of the original:
//! (1) value numbering: every read of every register dword (including `vcc`/`scc`) sees the same
//! reaching definition as in the original order, and the final definition of every dword is the
//! same (so everything live out is unchanged); the block terminator reads the same flags;
//! (2) memory / fence constraints: every pair that `order::mem_must_order` orders keeps its order.
//! It does not use the dependence graph, only the per-instruction effects.

use std::collections::HashMap;

use super::effects::*;
use super::order::mem_must_order;
use crate::mir::*;

type Def = Option<usize>;

struct Sim {
    /// For every instruction (by original index): the reaching def of each slot it reads.
    reads: Vec<Vec<(Slot, Def)>>,
    term: Vec<(Slot, Def)>,
    fin: HashMap<Slot, usize>,
}

fn simulate(eff: &[Effects], order: &[usize], term: &[Slot]) -> (Sim, Vec<Def>) {
    let mut defs: HashMap<Slot, usize> = HashMap::new();
    let mut sim = Sim { reads: vec![Vec::new(); eff.len()], term: Vec::new(), fin: HashMap::new() };
    let mut ver: Vec<Def> = vec![None; eff.len()];
    for &i in order {
        let e = &eff[i];
        for &s in &e.reads {
            sim.reads[i].push((s, defs.get(&s).copied()));
        }
        ver[i] = e.mem.as_ref().and_then(|a| a.base).and_then(|b| defs.get(&b).copied());
        for &s in &e.writes {
            defs.insert(s, i);
        }
    }
    for &s in term {
        sim.term.push((s, defs.get(&s).copied()));
    }
    sim.fin = defs;
    (sim, ver)
}

/// `perm[k]` = original index of the instruction at new position `k`.
pub fn check_perm(m: &MFunction, insts: &[MInst], perm: &[usize], term: &Term, alu_across: bool) -> Result<(), String> {
    let n = insts.len();
    if perm.len() != n {
        return Err(format!("permutation length {} != {}", perm.len(), n));
    }
    let mut pos = vec![usize::MAX; n];
    for (k, &p) in perm.iter().enumerate() {
        if p >= n || pos[p] != usize::MAX {
            return Err(format!("not a permutation (position {k} -> {p})"));
        }
        pos[p] = k;
    }
    let eff: Vec<Effects> = insts.iter().map(|i| effects(m, i)).collect();
    let tr = term_reads(term);
    let ident: Vec<usize> = (0..n).collect();
    let (orig, ver) = simulate(&eff, &ident, &tr);
    let (new, _) = simulate(&eff, perm, &tr);
    for i in 0..n {
        if orig.reads[i] != new.reads[i] {
            let bad = orig.reads[i].iter().zip(new.reads[i].iter()).find(|(a, b)| a != b);
            return Err(format!(
                "instruction {i} ({:?}) reads a different definition after scheduling: {:?}",
                insts[i].op, bad
            ));
        }
    }
    if orig.term != new.term {
        return Err("terminator reads a different flag definition".to_string());
    }
    if orig.fin != new.fin {
        let bad = orig.fin.iter().find(|(s, d)| new.fin.get(s) != Some(d));
        return Err(format!("final definition of a register changed: {:?}", bad));
    }
    for i in 0..n {
        for j in 0..i {
            if pos[j] > pos[i] && mem_must_order(&eff[j], ver[j], &eff[i], ver[i], alu_across) {
                return Err(format!(
                    "memory/fence order violated: {j} ({:?}) must stay before {i} ({:?})",
                    insts[j].op, insts[i].op
                ));
            }
        }
    }
    Ok(())
}