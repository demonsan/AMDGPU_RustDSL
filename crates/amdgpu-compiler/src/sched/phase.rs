//! `sched_group_barrier`-style phase scheduling (the vendor's interleave tool).
//!
//! A basic block that contains `MOp::SchedGroup` markers (DSL `FnBuilder::sched_group_barrier` / `sched_pattern`) is re-ordered so that its instructions follow the
//! pipeline the markers describe: group 1 = the next `size` instructions of class `mask` (program order among the not yet placed), group 2, ... Whatever a chosen
//! instruction depends on (address chains, `s_mov m0`, the loads it consumes, ...) is placed immediately before it, in program order; every register, memory and fence
//! dependence of `depgraph` is respected (the result is checked with `verify::check_perm`). Instructions no group selects follow in program order.
//! `s_setprio` instructions are anchored to the MFMA stream: each one is re-emitted right after the k-th MFMA, k = the number of MFMAs before it in the source.
//! Markers are always removed (`strip`), also when no scheduler runs.

use super::depgraph;
use super::effects::*;
use super::{verify, Options};
use crate::mir::*;

pub const ALU: u32 = 0x1;
pub const VALU: u32 = 0x2;
pub const SALU: u32 = 0x4;
pub const MFMA: u32 = 0x8;
pub const VMEM: u32 = 0x10;
pub const VMEM_READ: u32 = 0x20;
pub const VMEM_WRITE: u32 = 0x40;
pub const DS: u32 = 0x80;
pub const DS_READ: u32 = 0x100;
pub const DS_WRITE: u32 = 0x200;

fn is_matrix(op: MOp) -> bool {
    crate::agpr::is_mfma(op) || op == MOp::VSmfmac16x16x128Fp8
}

/// LLVM-style class mask of an instruction.
pub fn class_of(m: &MFunction, inst: &MInst) -> u32 {
    let e = effects(m, inst);
    if is_matrix(inst.op) {
        return ALU | MFMA;
    }
    if let Some(a) = &e.mem {
        return match (a.space, a.write) {
            (Space::Lds, false) => DS | DS_READ,
            (Space::Lds, true) => DS | DS_WRITE,
            (Space::Global, false) => VMEM | VMEM_READ,
            (Space::Global, true) => VMEM | VMEM_WRITE,
        };
    }
    if e.fence != Fence::None || inst.op == MOp::SchedGroup {
        return 0;
    }
    let name = format!("{:?}", inst.op);
    if name.starts_with('S') && !name.starts_with("SchedGroup") {
        return ALU | SALU;
    }
    ALU | VALU
}

/// Remove the directives from every block (no-op when there are none).
pub fn strip(m: &mut MFunction) {
    for b in &mut m.blocks {
        b.insts.retain(|i| i.op != MOp::SchedGroup);
    }
}

pub fn has_markers(insts: &[MInst]) -> bool {
    insts.iter().any(|i| i.op == MOp::SchedGroup)
}

/// Re-order one block according to its markers. Returns the number of instructions that changed position, or `None` when the block was left alone.
pub fn schedule_block(m: &MFunction, insts: &[MInst], term: &Term, o: &Options) -> Result<Option<(Vec<MInst>, usize)>, String> {
    let groups: Vec<(u32, u32)> = insts
        .iter()
        .filter(|i| i.op == MOp::SchedGroup)
        .map(|i| ((i.imm & 0xffff) as u32, ((i.imm >> 16) & 0xffff) as u32))
        .collect();
    // marker-free instruction list; s_setprio pulled out (anchored to the MFMA count)
    let mut body: Vec<MInst> = Vec::new();
    let mut prio: Vec<(usize, MInst)> = Vec::new(); // (MFMAs before it, instruction)
    let mut mfmas = 0usize;
    for i in insts {
        match i.op {
            MOp::SchedGroup => {}
            MOp::SSetPrio => prio.push((mfmas, i.clone())),
            _ => {
                if is_matrix(i.op) {
                    mfmas += 1;
                }
                body.push(i.clone());
            }
        }
    }
    // explicit partial waits count outstanding operations by program order: leave such a block alone
    if body.iter().any(|i| matches!(i.op, MOp::SWaitLgkm | MOp::SWaitVm) && i.imm != 0 && !i.auto) {
        return Ok(None);
    }
    let n = body.len();
    if n < 2 {
        return Ok(None);
    }
    let g = depgraph::build(m, &body, term, true);
    let class: Vec<u32> = body.iter().map(|i| class_of(m, i)).collect();
    let mut placed = vec![false; n];
    let mut order: Vec<usize> = Vec::with_capacity(n);
    // place `x` after everything it transitively needs (ascending program order = a topological order: all edges point forward)
    fn place(x: usize, g: &depgraph::Graph, placed: &mut Vec<bool>, order: &mut Vec<usize>) {
        let mut need: Vec<usize> = Vec::new();
        let mut stack = vec![x];
        let mut seen = vec![false; g.n];
        while let Some(y) = stack.pop() {
            if seen[y] || placed[y] {
                continue;
            }
            seen[y] = true;
            need.push(y);
            for &(p, _) in &g.preds[y] {
                stack.push(p);
            }
        }
        need.sort_unstable();
        for y in need {
            placed[y] = true;
            order.push(y);
        }
    }
    let mut first_unplaced = 0usize;
    for &(mask, size) in &groups {
        for _ in 0..size {
            while first_unplaced < n && placed[first_unplaced] {
                first_unplaced += 1;
            }
            let pick = (first_unplaced..n).find(|&i| !placed[i] && class[i] & mask != 0 && class[i] != 0);
            match pick {
                Some(i) => place(i, &g, &mut placed, &mut order),
                None => break,
            }
        }
    }
    for i in 0..n {
        if !placed[i] {
            place(i, &g, &mut placed, &mut order);
        }
    }
    if o.verify {
        verify::check_perm(m, &body, &order, term, true).map_err(|e| format!("phase schedule verification failed: {e}"))?;
    }
    let moved = order.iter().enumerate().filter(|(k, p)| *k != **p).count();
    // re-insert s_setprio after the k-th MFMA of the new order
    let mut out: Vec<MInst> = Vec::with_capacity(insts.len());
    let mut seen_m = 0usize;
    let mut pi = 0usize;
    let emit_prio = |out: &mut Vec<MInst>, pi: &mut usize, seen_m: usize| {
        while *pi < prio.len() && prio[*pi].0 <= seen_m {
            out.push(prio[*pi].1.clone());
            *pi += 1;
        }
    };
    emit_prio(&mut out, &mut pi, 0);
    for &k in &order {
        out.push(body[k].clone());
        if is_matrix(body[k].op) {
            seen_m += 1;
            emit_prio(&mut out, &mut pi, seen_m);
        }
    }
    while pi < prio.len() {
        out.push(prio[pi].1.clone());
        pi += 1;
    }
    Ok(Some((out, moved)))
}

/// Apply `schedule_block` to every block with markers; strips the markers everywhere.
pub fn run(m: &mut MFunction, o: &Options) -> Result<usize, String> {
    let mut moved_total = 0;
    for bi in 0..m.blocks.len() {
        if !has_markers(&m.blocks[bi].insts) {
            continue;
        }
        let insts = m.blocks[bi].insts.clone();
        let term = m.blocks[bi].term.clone();
        match schedule_block(m, &insts, &term, o)? {
            Some((out, moved)) => {
                moved_total += moved;
                m.blocks[bi].insts = out;
            }
            None => {}
        }
    }
    strip(m);
    Ok(moved_total)
}