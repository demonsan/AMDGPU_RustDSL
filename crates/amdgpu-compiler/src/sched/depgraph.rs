//! Dependence graph over the instructions of one straight-line region.
//!
//! Edges (all point forward in program order): register RAW (latency of the producer), WAR
//! (0), WAW (1), memory / fence ordering (`order::mem_must_order`, 1). `vcc` and `scc` are
//! ordinary register slots, so a compare and its `v_cndmask` stay paired automatically (no other
//! `vcc` writer can slip in between) while independent work may fill the gap between them.

use std::collections::HashMap;

use super::effects::*;
use super::model;
use super::order::mem_must_order;
use crate::mir::*;

pub struct Graph {
    pub n: usize,
    pub succs: Vec<Vec<(usize, u32)>>,
    pub preds: Vec<Vec<(usize, u32)>>,
    pub eff: Vec<Effects>,
    /// Value version (defining instruction index) of each memory access's base register.
    pub ver: Vec<Option<usize>>,
    /// Index of the last vcc writer when the terminator reads vcc (its result feeds the branch).
    pub term_vcc_writer: Option<usize>,
}

fn add_edge(g: &mut Graph, from: usize, to: usize, lat: u32) {
    if from == to {
        return;
    }
    for e in g.succs[from].iter_mut() {
        if e.0 == to {
            if lat > e.1 {
                e.1 = lat;
                for p in g.preds[to].iter_mut() {
                    if p.0 == from {
                        p.1 = lat;
                    }
                }
            }
            return;
        }
    }
    g.succs[from].push((to, lat));
    g.preds[to].push((from, lat));
}

pub fn build(m: &MFunction, insts: &[MInst], term: &Term, alu_across: bool) -> Graph {
    let n = insts.len();
    let eff: Vec<Effects> = insts.iter().map(|i| effects(m, i)).collect();
    let cost: Vec<model::OpCost> = insts.iter().map(|i| model::cost(i.op)).collect();
    let mut g = Graph {
        n,
        succs: vec![Vec::new(); n],
        preds: vec![Vec::new(); n],
        eff: Vec::new(),
        ver: vec![None; n],
        term_vcc_writer: None,
    };
    let mut last_def: HashMap<Slot, usize> = HashMap::new();
    let mut readers: HashMap<Slot, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let e = &eff[i];
        g.ver[i] = e.mem.as_ref().and_then(|a| a.base).and_then(|b| last_def.get(&b).copied());
        for &s in &e.reads {
            if let Some(&d) = last_def.get(&s) {
                // vcc producer -> consumer needs 2 wait states (hazards strict mode)
                let lat = if s == VCC { cost[d].latency.max(12) } else { cost[d].latency };
                add_edge(&mut g, d, i, lat);
            }
            readers.entry(s).or_default().push(i);
        }
        for &s in &e.writes {
            if let Some(&d) = last_def.get(&s) {
                add_edge(&mut g, d, i, 1);
            }
            if let Some(rs) = readers.get_mut(&s) {
                for &r in rs.iter() {
                    if r != i {
                        add_edge(&mut g, r, i, 0);
                    }
                }
                rs.clear();
            }
            last_def.insert(s, i);
        }
    }
    if term_reads(term).contains(&VCC) {
        g.term_vcc_writer = (0..n).rev().find(|&i| eff[i].writes.contains(&VCC));
    }
    let is_fence = |i: usize| eff[i].fence != Fence::None;
    let fences: Vec<usize> = (0..n).filter(|&i| is_fence(i)).collect();
    let mems: Vec<usize> = (0..n).filter(|&i| !is_fence(i) && eff[i].mem.is_some()).collect();
    for i in 0..n {
        if is_fence(i) {
            for j in 0..i {
                if mem_must_order(&eff[j], g.ver[j], &eff[i], g.ver[i], alu_across) {
                    add_edge(&mut g, j, i, 1);
                }
            }
        } else {
            for &f in &fences {
                if f >= i {
                    break;
                }
                if mem_must_order(&eff[f], g.ver[f], &eff[i], g.ver[i], alu_across) {
                    add_edge(&mut g, f, i, 1);
                }
            }
            if eff[i].mem.is_some() {
                for &j in &mems {
                    if j >= i {
                        break;
                    }
                    if mem_must_order(&eff[j], g.ver[j], &eff[i], g.ver[i], alu_across) {
                        add_edge(&mut g, j, i, 1);
                    }
                }
            }
        }
    }
    g.eff = eff;
    g
}