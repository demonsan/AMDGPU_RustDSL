//! VGPR-window grouping (gfx1250, opt-in: lowering token `msbgroup` or `AMDC_MSB_GROUP`): with more than 256 VGPRs every instruction's
//! operands select their 256-register window through `s_set_vgpr_msb` (msb.rs); an epilogue that converts accumulators spread over several
//! windows in their natural order switches the windows almost every instruction. Per block, a list schedule over the dependence graph (the
//! shared one, plus the PHYSICAL register dependences of the allocation: this runs after register allocation) that keeps issuing ready
//! instructions whose windows match the current state, and only when none does takes the earliest ready instruction in program order. The
//! new order is applied only when it emits fewer window switches than the old one.

use super::msb::Msb;
use crate::mir::*;
use crate::regalloc::Alloc;

pub fn enabled(m: &MFunction) -> bool {
    std::env::var("AMDC_NO_MSB_GROUP").is_err() && (m.msb_group || std::env::var("AMDC_MSB_GROUP").is_ok())
}

/// The window needs of one instruction (per slot), from its emitted lines; None: it has no VGPR operand above v255 and does not care.
fn needs(a: &Alloc, inst: &MInst) -> [Option<u8>; 4] {
    let mut need = [None; 4];
    let Ok(lines) = super::emit::inst_lines(a, inst) else { return need };
    for l in lines {
        // the switch window() would emit from an all-unknown state tells the needed slots: run it from two different states and keep the
        // slots it sets in both
        let mut s0 = Msb([0; 4]);
        let mut s1 = Msb([3; 4]);
        let (Ok(_), Ok(_)) = (super::msb::window(&l, &mut s0, false), super::msb::window(&l, &mut s1, false)) else { continue };
        for k in 0..4 {
            if s0.0[k] == s1.0[k] {
                need[k] = Some(s0.0[k]);
            }
        }
    }
    need
}

fn switches(needs: &[[Option<u8>; 4]], order: &[usize]) -> usize {
    let mut cur = [0u8; 4];
    let mut n = 0;
    for &i in order {
        let mut want = cur;
        for k in 0..4 {
            if let Some(m) = needs[i][k] {
                want[k] = m;
            }
        }
        if want != cur {
            n += 1;
            cur = want;
        }
    }
    n
}

fn phys_regs(m: &MFunction, a: &Alloc, r: VReg) -> Vec<u32> {
    let Some(&p) = a.phys.get(r as usize) else { return Vec::new() };
    if p == u32::MAX {
        return Vec::new();
    }
    let s = a.scalar.get(r as usize).copied().unwrap_or(false);
    let w = m.vreg_class(r).dwords();
    // (SGPRs and VGPRs in disjoint key ranges)
    (p..p + w).map(|x| if s { x | 0x8000_0000 } else { x }).collect()
}

/// Reorders the blocks of `m`; returns (blocks reordered, window switches saved).
pub fn group(m: &mut MFunction, a: &Alloc) -> (usize, usize) {
    if !enabled(m) || m.exact {
        return (0, 0);
    }
    let (mut blocks, mut saved) = (0usize, 0usize);
    for bi in 0..m.blocks.len() {
        let insts = m.blocks[bi].insts.clone();
        let n = insts.len();
        if n < 3 {
            continue;
        }
        let nd: Vec<[Option<u8>; 4]> = insts.iter().map(|i| needs(a, i)).collect();
        let orig: Vec<usize> = (0..n).collect();
        let before = switches(&nd, &orig);
        if before < 2 {
            continue;
        }
        let g = crate::sched::depgraph::build(m, &insts, &m.blocks[bi].term, false);
        let mut preds: Vec<Vec<usize>> = (0..n).map(|j| g.preds[j].iter().map(|&(i, _)| i).collect()).collect();
        // physical register dependences (RAW / WAR / WAW on the allocated registers)
        {
            let mut last_w: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
            let mut reads_since: std::collections::HashMap<u32, Vec<usize>> = std::collections::HashMap::new();
            for (j, inst) in insts.iter().enumerate() {
                let e = &g.eff[j];
                let rd: Vec<u32> = inst.srcs.iter().filter_map(|s| if let Src::Reg(r) = s { Some(*r) } else { None }).flat_map(|r| phys_regs(m, a, r)).collect();
                let wr: Vec<u32> = inst.dst.iter().flat_map(|&r| phys_regs(m, a, r)).collect();
                let _ = e;
                for x in &rd {
                    if let Some(&w) = last_w.get(x) {
                        preds[j].push(w);
                    }
                }
                for x in &wr {
                    if let Some(&w) = last_w.get(x) {
                        preds[j].push(w);
                    }
                    if let Some(rs) = reads_since.get(x) {
                        preds[j].extend(rs.iter().copied().filter(|&r| r != j));
                    }
                }
                for x in &rd {
                    reads_since.entry(*x).or_default().push(j);
                }
                for x in &wr {
                    last_w.insert(*x, j);
                    reads_since.remove(x);
                }
            }
        }
        // pinned instructions and anything that is not a plain ALU / memory op keep their order relative to everything (a barrier: every
        // earlier instruction precedes it, it precedes every later one)
        let fence: Vec<bool> = insts.iter().enumerate().map(|(j, i)| i.pin || i.words.is_some() || !matches!(g.eff[j].fence, crate::sched::effects::Fence::None)).collect();
        let mut last_fence: Option<usize> = None;
        for j in 0..n {
            if let Some(f) = last_fence {
                preds[j].push(f);
            }
            if fence[j] {
                preds[j].extend(0..j);
                last_fence = Some(j);
            }
        }
        for p in preds.iter_mut() {
            p.sort_unstable();
            p.dedup();
        }
        let mut npred: Vec<usize> = preds.iter().map(|p| p.len()).collect();
        let mut succs: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (j, p) in preds.iter().enumerate() {
            for &i in p {
                succs[i].push(j);
            }
        }
        let mut ready: std::collections::BTreeSet<usize> = (0..n).filter(|&j| npred[j] == 0).collect();
        let mut order: Vec<usize> = Vec::with_capacity(n);
        let mut cur = [0u8; 4];
        while !ready.is_empty() {
            let fits = |j: usize| (0..4).all(|k| nd[j][k].map_or(true, |m| m == cur[k]));
            // a ready instruction that needs no switch (and does need some window: the window-free ones go in program order) ...
            let pick = ready.iter().copied().find(|&j| fits(j) && nd[j].iter().any(|x| x.is_some()))
                .or_else(|| ready.iter().copied().find(|&j| nd[j].iter().all(|x| x.is_none())))
                .unwrap_or_else(|| *ready.iter().next().unwrap());
            ready.remove(&pick);
            for k in 0..4 {
                if let Some(mm) = nd[pick][k] {
                    cur[k] = mm;
                }
            }
            order.push(pick);
            for &s in &succs[pick] {
                npred[s] -= 1;
                if npred[s] == 0 {
                    ready.insert(s);
                }
            }
        }
        if order.len() != n {
            continue;
        }
        let after = switches(&nd, &order);
        if after < before {
            m.blocks[bi].insts = order.iter().map(|&i| insts[i].clone()).collect();
            blocks += 1;
            saved += before - after;
        }
    }
    (blocks, saved)
}
