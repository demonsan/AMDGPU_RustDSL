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

/// The fill candidates for the nop at index `i` of block `bi`, nearest first: instructions below it that may move up to it. Besides the
/// dependence rules above, a candidate must be independent of the HAZARD PAIR: it touches no register written or read by the instructions
/// in front of the nop (the producer side, up to `window` back) - so it is neither the consumer the nop protects nor a new consumer / WAR
/// victim of the producer.
fn candidates(m: &MFunction, a: &Alloc, bi: usize, i: usize, window: usize) -> Vec<usize> {
    let insts = &m.blocks[bi].insts;
    let pre: Vec<Fx> = insts[i.saturating_sub(window)..i].iter().map(|x| fx(m, a, x)).collect();
    let fxs: Vec<Fx> = insts.iter().skip(i + 1).take(window).map(|x| fx(m, a, x)).collect();
    let touches = |p: &Fx, y: &Fx| {
        let any = |u: &[Rng], v: &[Rng]| u.iter().any(|x| v.iter().any(|z| ov(x, z)));
        any(&p.writes, &y.reads) || any(&p.writes, &y.writes) || any(&p.reads, &y.writes)
    };
    let mut out = Vec::new();
    for (k, y) in fxs.iter().enumerate() {
        let j = i + 1 + k;
        if is_stop(&insts[j], &y.eff) {
            break;
        }
        let hops = &fxs[..k];
        let blocked = hops.iter().any(|z| conflict(z, y) || (is_mem(&z.eff) && is_mem(&y.eff)) || crate::sched::order::mem_must_order(&z.eff, None, &y.eff, None, false))
            || pre.iter().any(|p| touches(p, y));
        if !blocked {
            out.push(j);
        }
    }
    out
}

/// Candidates from ABOVE the nop (nearest first): an instruction in front of the hazard producer that may sink to the nop's position (it
/// hops over everything between it and the nop, the producer included, with the same dependence rules). The hazard model then confirms
/// that sinking it did not shorten another window.
fn candidates_above(m: &MFunction, a: &Alloc, bi: usize, i: usize, window: usize) -> Vec<usize> {
    let insts = &m.blocks[bi].insts;
    let lo = i.saturating_sub(window);
    let fxs: Vec<Fx> = insts[lo..i].iter().map(|x| fx(m, a, x)).collect();
    let mut out = Vec::new();
    for j in (lo..i).rev() {
        let y = &fxs[j - lo];
        if is_stop(&insts[j], &y.eff) {
            break;
        }
        let hops = &fxs[j - lo + 1..];
        let blocked = hops.iter().any(|z| conflict(y, z) || (is_mem(&z.eff) && is_mem(&y.eff)) || crate::sched::order::mem_must_order(&y.eff, None, &z.eff, None, false));
        if !blocked {
            out.push(j);
        }
    }
    out
}

/// Sink the instruction at `j` (< nop index `i`) to just before the nop, shrinking the nop by one state.
fn apply_above(m: &mut MFunction, bi: usize, j: usize, i: usize) {
    let b = &mut m.blocks[bi].insts;
    let y = b.remove(j);
    // the nop moved to i - 1
    b.insert(i - 1, y);
    if b[i].imm > 0 {
        b[i].imm -= 1;
    } else {
        b.remove(i);
    }
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
    fill_with(m, a, strict, None)
}

/// May the nop window at `i` be filled with the instruction at `j` under the weave-region strategies `o`? The nop belongs to the instruction it makes wait (the next real one); the mover and every
/// instruction it hops over must share that instruction's strategy, and the strategy must enable `hzfill`. Without regions (`None` / an empty table) everything is allowed.
fn region_ok(o: Option<&crate::sched::Options>, m: &MFunction, bi: usize, i: usize, j: usize) -> bool {
    let Some(o) = o.filter(|o| !o.regions.is_empty()) else { return true };
    let insts = &m.blocks[bi].insts;
    let strat = |k: usize| crate::sched::region::strategy_of(o, insts[k].region);
    // a trailing nop (before the terminator) belongs to the last real instruction before it
    let Some(w) = (i + 1..insts.len()).find(|&k| insts[k].op != MOp::SNop).or_else(|| (0..i).rev().find(|&k| insts[k].op != MOp::SNop)) else { return o.for_region(0).hzfill };
    let want = strat(w);
    if !o.for_region(want).hzfill {
        return false;
    }
    let (lo, hi) = (i.min(j), i.max(j));
    (lo..=hi).all(|k| insts[k].op == MOp::SNop || strat(k) == want)
}

/// `fill` with the per-region strategies of the scheduler options (`hzfill` as a region option).
pub fn fill_with(m: &mut MFunction, a: &Alloc, strict: bool, o: Option<&crate::sched::Options>) -> FillStats {
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
                if let Some(j) = candidates(&trial, a, bi, i, WINDOW).into_iter().find(|&j| region_ok(o, &trial, bi, i, j)) {
                    apply(&mut trial, bi, i, j);
                    moves.push((i, j));
                    i += 1; // the nop (if any) is now at i + 1: offer it the next candidate
                    continue;
                }
            }
            i += 1;
        }
        let greedy_ok = !moves.is_empty() && hazard_clean(&trial, a, strict);
        if greedy_ok {
            st.moved += moves.len();
            *m = trial;
        }
        {
            // one move at a time, each checked
            let mut i = 0usize;
            while i < m.blocks[bi].insts.len() {
                let x = &m.blocks[bi].insts[i];
                if x.op == MOp::SNop && !x.pin && !(i > 0 && m.blocks[bi].insts[i - 1].pin) {
                    let mut done = false;
                    for j in candidates(m, a, bi, i, WINDOW).into_iter().filter(|&j| region_ok(o, m, bi, i, j)).take(4) {
                        let mut t = m.clone();
                        apply(&mut t, bi, i, j);
                        if hazard_clean(&t, a, strict) {
                            *m = t;
                            st.moved += 1;
                            done = true;
                            break;
                        }
                        st.rejected += 1;
                    }
                    if !done {
                        for j in candidates_above(m, a, bi, i, WINDOW).into_iter().filter(|&j| region_ok(o, m, bi, i, j)).take(4) {
                            let mut t = m.clone();
                            apply_above(&mut t, bi, j, i);
                            if hazard_clean(&t, a, strict) {
                                *m = t;
                                st.moved += 1;
                                done = true;
                                break;
                            }
                            st.rejected += 1;
                        }
                        if done {
                            // the nop is now at i (or gone): offer it again
                            continue;
                        }
                    }
                    if done {
                        i += 1;
                        continue;
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
