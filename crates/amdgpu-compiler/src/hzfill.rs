//! `hzfill` (opt-in scheduler token): post-RA hazard-window filling. After register allocation and the hazard passes, every `s_nop` wait
//! window is offered independent instructions from BELOW it (the same block, up to the next fence): an instruction moves up to the nop's
//! position (and the nop shrinks by one wait state) when
//!  * it is an ordinary instruction (no fence / wait / barrier / nop / priority / pinned safe-asm / LDS-DMA, which carries a fence and an
//!    implicit M0), and no fence lies between the nop and it;
//!  * it has no PHYSICAL register dependence (RAW / WAR / WAW on the allocated VGPR / AGPR / SGPR ranges, `vcc`, `scc`) on any instruction it
//!    hops over;
//!  * it is not a memory operation hopping over another memory operation (the vmcnt / lgkmcnt counters are positional: a counted wait
//!    after both would then retire the wrong one), and no memory-order rule (sched/order.rs) is violated.
//! The hazard model is the authority on the rest: after the moves of a block the value-hazard and physical-WAR passes are re-run on a copy;
//! when they would change ANYTHING (a moved instruction created a new hazard, e.g. it reads an MFMA result inside the window) the block's
//! moves are retried one by one and only the moves after which the copy is unchanged are kept. Each block is finally checked as a permutation
//! of the original (`sched::verify::check_perm`, value level).

use crate::mir::*;
use crate::regalloc::Alloc;
use crate::sched::effects::{effects, Effects, Fence, SCC, VCC};

#[derive(Debug, Default, Clone, Copy)]
pub struct FillStats {
    pub nops_before: usize,
    pub states_before: usize,
    pub states_after: usize,
    pub moved: usize,
    pub rejected: usize,
}

/// (file, lo, hi) physical ranges; file 0 = VGPR/AGPR (AGPRs are tagged in `Alloc::phys`), 1 = SGPR, 2 = vcc, 3 = scc.
type Rng = (u8, u32, u32);

fn ranges(m: &MFunction, a: &Alloc, r: VReg) -> Rng {
    let p = a.phys[r as usize];
    let w = m.vreg_class(r).dwords();
    let scalar = m.vreg_class(r).is_scalar() || a.scalar.get(r as usize).copied().unwrap_or(false);
    (if scalar { 1 } else { 0 }, p, p + w)
}

struct Fx {
    reads: Vec<Rng>,
    writes: Vec<Rng>,
    eff: Effects,
}

fn fx(m: &MFunction, a: &Alloc, i: &MInst) -> Fx {
    let eff = effects(m, i);
    let mut reads: Vec<Rng> = i.srcs.iter().filter_map(|s| if let Src::Reg(r) = s { Some(ranges(m, a, *r)) } else { None }).collect();
    let mut writes: Vec<Rng> = i.dst.iter().map(|d| ranges(m, a, *d)).collect();
    // partial writers / in-place forms read their destination too (effects lists those dwords as reads)
    if let Some(d) = i.dst {
        let ds = crate::sched::effects::reg_slots(m, d);
        if eff.reads.iter().any(|s| ds.contains(s)) {
            reads.push(ranges(m, a, d));
        }
    }
    // swap forms write their sources
    if eff.writes.len() > i.dst.map_or(0, |d| m.vreg_class(d).dwords() as usize) {
        for s in &i.srcs {
            if let Src::Reg(r) = s {
                writes.push(ranges(m, a, *r));
            }
        }
    }
    if eff.reads.contains(&VCC) {
        reads.push((2, 0, 1));
    }
    if eff.writes.contains(&VCC) {
        writes.push((2, 0, 1));
    }
    if eff.reads.contains(&SCC) {
        reads.push((3, 0, 1));
    }
    if eff.writes.contains(&SCC) {
        writes.push((3, 0, 1));
    }
    Fx { reads, writes, eff }
}

fn ov(x: &Rng, y: &Rng) -> bool {
    x.0 == y.0 && x.1 < y.2 && y.1 < x.2
}

fn conflict(a: &Fx, b: &Fx) -> bool {
    let any = |p: &[Rng], q: &[Rng]| p.iter().any(|x| q.iter().any(|y| ov(x, y)));
    any(&a.writes, &b.reads) || any(&a.reads, &b.writes) || any(&a.writes, &b.writes)
}

fn is_mem(e: &Effects) -> bool {
    e.mem.is_some() || e.smem
}

/// May the instruction take part in filling at all (as a mover or as an instruction hopped over)? Fences stop the scan.
fn is_stop(i: &MInst, e: &Effects) -> bool {
    i.pin || e.fence != Fence::None || e.pinned || matches!(i.op, MOp::SNop | MOp::SchedGroup | MOp::SEndpgm)
}

fn states(insts: &[MInst]) -> usize {
    insts.iter().filter(|i| i.op == MOp::SNop).map(|i| i.imm.max(0) as usize + 1).sum()
}

fn same(a: &MFunction, b: &MFunction) -> bool {
    a.blocks.len() == b.blocks.len()
        && a.blocks.iter().zip(&b.blocks).all(|(x, y)| {
            x.insts.len() == y.insts.len() && x.insts.iter().zip(&y.insts).all(|(p, q)| p.op == q.op && p.dst == q.dst && p.srcs == q.srcs && p.imm == q.imm)
        })
}

/// The hazard passes find nothing to change on `m`.
fn hazard_clean(m: &MFunction, a: &Alloc, strict: bool) -> bool {
    let mut c = m.clone();
    if strict {
        crate::hazards::fix_value_hazards_opts(&mut c, true);
    } else {
        crate::hazards::fix_value_hazards(&mut c);
    }
    crate::hazards::pad_physical_war(&mut c, a);
    same(&c, m)
}

/// One fill step at nop index `i` of block `bi`: the first legal candidate below it moves up (returns the candidate's old index).
fn candidate(m: &MFunction, a: &Alloc, bi: usize, i: usize, window: usize) -> Option<usize> {
    let insts = &m.blocks[bi].insts;
    let fxs: Vec<Fx> = insts.iter().skip(i + 1).take(window).map(|x| fx(m, a, x)).collect();
    for (k, y) in fxs.iter().enumerate() {
        let j = i + 1 + k;
        if is_stop(&insts[j], &y.eff) {
            return None;
        }
        let hops = &fxs[..k];
        let blocked = hops.iter().any(|z| conflict(z, y) || (is_mem(&z.eff) && is_mem(&y.eff)) || crate::sched::order::mem_must_order(&z.eff, None, &y.eff, None, false));
        if !blocked {
            return Some(j);
        }
    }
    None
}

fn apply(m: &mut MFunction, bi: usize, i: usize, j: usize) {
    let b = &mut m.blocks[bi].insts;
    let y = b.remove(j);
    b.insert(i, y);
    // the nop is now at i + 1
    if b[i + 1].imm > 0 {
        b[i + 1].imm -= 1;
    } else {
        b.remove(i + 1);
    }
}

/// Fill the wait windows of every block (see the module doc). `strict`: the value-hazard pass of the scheduled path (block-end padding).
pub fn fill(m: &mut MFunction, a: &Alloc, strict: bool) -> FillStats {
    const WINDOW: usize = 24;
    let mut st = FillStats::default();
    st.nops_before = m.blocks.iter().map(|b| b.insts.iter().filter(|i| i.op == MOp::SNop).count()).sum();
    st.states_before = m.blocks.iter().map(|b| states(&b.insts)).sum();
    // the model must be a fixed point on the input, else nothing can be judged
    if !hazard_clean(m, a, strict) {
        st.states_after = st.states_before;
        return st;
    }
    for bi in 0..m.blocks.len() {
        let orig = m.blocks[bi].insts.clone();
        // greedy pass over the block with the dependence rules only
        let mut trial = m.clone();
        let mut moves: Vec<(usize, usize)> = Vec::new();
        let mut i = 0usize;
        while i < trial.blocks[bi].insts.len() {
            let x = &trial.blocks[bi].insts[i];
            if x.op == MOp::SNop && !x.pin && !(i > 0 && trial.blocks[bi].insts[i - 1].pin) {
                if let Some(j) = candidate(&trial, a, bi, i, WINDOW) {
                    apply(&mut trial, bi, i, j);
                    moves.push((i, j));
                    i += 1; // the nop (if any) is now at i + 1: offer it the next candidate
                    continue;
                }
            }
            i += 1;
        }
        if moves.is_empty() {
            continue;
        }
        if hazard_clean(&trial, a, strict) {
            st.moved += moves.len();
            *m = trial;
        } else {
            // one move at a time, each checked
            let mut i = 0usize;
            while i < m.blocks[bi].insts.len() {
                let x = &m.blocks[bi].insts[i];
                if x.op == MOp::SNop && !x.pin && !(i > 0 && m.blocks[bi].insts[i - 1].pin) {
                    if let Some(j) = candidate(m, a, bi, i, WINDOW) {
                        let mut t = m.clone();
                        apply(&mut t, bi, i, j);
                        if hazard_clean(&t, a, strict) {
                            *m = t;
                            st.moved += 1;
                            i += 1;
                            continue;
                        }
                        st.rejected += 1;
                    }
                }
                i += 1;
            }
        }
        // the block must still be a dependence-preserving permutation of the original (nops aside)
        let now: Vec<MInst> = m.blocks[bi].insts.iter().filter(|x| x.op != MOp::SNop).cloned().collect();
        let was: Vec<MInst> = orig.iter().filter(|x| x.op != MOp::SNop).cloned().collect();
        let mut used = vec![false; was.len()];
        let perm: Option<Vec<usize>> = now
            .iter()
            .map(|x| {
                let k = (0..was.len()).find(|&k| !used[k] && was[k].op == x.op && was[k].dst == x.dst && was[k].srcs == x.srcs && was[k].imm == x.imm)?;
                used[k] = true;
                Some(k)
            })
            .collect();
        let ok = match perm {
            Some(p) if p.len() == was.len() => crate::sched::verify::check_perm(m, &was, &p, &m.blocks[bi].term, false).is_ok(),
            _ => false,
        };
        if !ok {
            m.blocks[bi].insts = orig;
        }
    }
    st.states_after = m.blocks.iter().map(|b| states(&b.insts)).sum();
    st
}
