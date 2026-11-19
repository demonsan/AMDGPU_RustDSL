//! Post-allocation dead-write elimination (gfx1250, block-local): an instruction without side effects whose every written PHYSICAL register
//! (dword) is overwritten later in the same block before anything reads it - and whose SCC / VCC write, if any, is overwritten the same
//! way - is removed. The value-level DCE before allocation cannot see these: a tensor descriptor's group words are first assembled from one
//! template, then partly rewritten (the kernel prologue of the MXFP4 GEMM wrote 4 x 8 SGPR words twice, and computed an address into them
//! that the second write discarded). Repeats until nothing changes (removing a write can make its operands' producers dead).
//! Opt-in (`AMDC_DSE=1`, and `AMDC_TDM_FIRST=1` for `tdm_first`): measured neutral on the MXFP4 GEMM and the MLA (e08-01, 2026-11-19;
//! the prologue 159 -> 106 instructions before the first tensor load).

use crate::mir::*;
use crate::regalloc::Alloc;
use crate::sched::effects::{effects, SCC, VCC};

/// A physical location: VGPR n -> n, SGPR n -> (1 << 20) + n, VCC -> u32::MAX.
fn keys(m: &MFunction, a: &Alloc, slots: &[u32]) -> Vec<u32> {
    let mut out = Vec::new();
    for &s in slots {
        if s == VCC {
            out.push(u32::MAX);
            continue;
        }
        if s == SCC {
            // SCC: tracked by the op attributes (scc_read / scc_write) below
            continue;
        }
        if s >= u32::MAX - 1 {
            // another special slot: not tracked, treated as a read / write of everything (conservative)
            out.push(u32::MAX - 1);
            continue;
        }
        let (t, off) = (s / 16, s % 16);
        let Some(&p) = a.phys.get(t as usize) else { out.push(u32::MAX - 1); continue };
        let scalar = a.scalar.get(t as usize).copied().unwrap_or(false);
        let _ = m;
        out.push(if scalar { (1 << 20) + p + off } else { p + off });
    }
    out
}

pub fn dse(m: &mut MFunction, a: &Alloc) -> usize {
    if m.exact || std::env::var("AMDC_DSE").is_err() {
        return 0;
    }
    let mut removed = 0;
    for bi in 0..m.blocks.len() {
        loop {
            let n = m.blocks[bi].insts.len();
            let info: Vec<(Vec<u32>, Vec<u32>, bool, bool)> = m.blocks[bi]
                .insts
                .iter()
                .map(|i| {
                    let e = effects(m, i);
                    let at = crate::op_attrs::attrs(i.op);
                    (keys(m, a, &e.reads), keys(m, a, &e.writes), at.scc_read, at.scc_write)
                })
                .collect();
            let mut kill = None;
            for i in 0..n {
                let inst = &m.blocks[bi].insts[i];
                let at = crate::op_attrs::attrs(inst.op);
                let pure = matches!(at.dce, crate::op_attrs::Dce::Pure) && at.mem.is_none() && !at.smem && !inst.pin && inst.dst.is_some();
                let (_, w, _, sw) = &info[i];
                if !pure || w.is_empty() || w.contains(&(u32::MAX - 1)) {
                    continue;
                }
                // every written location is overwritten before any read, inside the block
                let dead_loc = |k: u32| -> bool {
                    for j in i + 1..n {
                        let (r, wj, _, _) = &info[j];
                        if r.contains(&k) || r.contains(&(u32::MAX - 1)) {
                            return false;
                        }
                        if wj.contains(&k) {
                            return true;
                        }
                    }
                    false
                };
                let scc_dead = || -> bool {
                    for j in i + 1..n {
                        let (_, _, sr, swj) = &info[j];
                        if *sr {
                            return false;
                        }
                        if *swj {
                            return true;
                        }
                    }
                    false
                };
                if w.iter().all(|&k| dead_loc(k)) && (!sw || scc_dead()) {
                    kill = Some(i);
                    break;
                }
            }
            match kill {
                Some(i) => {
                    m.blocks[bi].insts.remove(i);
                    removed += 1;
                }
                None => break,
            }
        }
    }
    removed
}

/// The kernel prologue issues its first tensor loads as early as possible: in the entry block, every instruction before the first group of
/// `tensor_load_to_lds` that has no side effect and no dependence (RAW / WAR / WAW on a physical register, SCC, VCC) with any instruction it
/// would jump over is moved behind the group (keeping its order). The lane-offset VALU math of the later LDS reads goes there; the
/// descriptor words stay in front. Opt-in: `AMDC_TDM_FIRST=1`.
pub fn tdm_first(m: &mut MFunction, a: &Alloc) -> usize {
    if m.exact || std::env::var("AMDC_TDM_FIRST").is_err() || m.blocks.is_empty() {
        return 0;
    }
    let insts = &m.blocks[0].insts;
    let Some(first) = insts.iter().position(|i| i.op == MOp::TensorLoadToLds) else { return 0 };
    let mut last = first;
    while last + 1 < insts.len() && insts[last + 1].op == MOp::TensorLoadToLds {
        last += 1;
    }
    let info: Vec<(Vec<u32>, Vec<u32>, bool, bool)> = insts
        .iter()
        .map(|i| {
            let e = effects(m, i);
            let at = crate::op_attrs::attrs(i.op);
            (keys(m, a, &e.reads), keys(m, a, &e.writes), at.scc_read, at.scc_write)
        })
        .collect();
    let movable = |i: usize| -> bool {
        let inst = &insts[i];
        let at = crate::op_attrs::attrs(inst.op);
        matches!(at.dce, crate::op_attrs::Dce::Pure) && at.mem.is_none() && !at.smem && !inst.pin && inst.dst.is_some() && !info[i].0.contains(&(u32::MAX - 1)) && !info[i].1.contains(&(u32::MAX - 1))
    };
    let conflict = |r: usize, s: usize| -> bool {
        let (rr, rw, rsr, rsw) = &info[r];
        let (sr, sw, ssr, ssw) = &info[s];
        let hit = |x: &[u32], y: &[u32]| x.iter().any(|k| y.contains(k));
        hit(rw, sr) || hit(rr, sw) || hit(rw, sw) || (*rsw && (*ssr || *ssw)) || (*rsr && *ssw)
    };
    let mut moved = vec![false; last + 1];
    for r in (0..first).rev() {
        if !movable(r) {
            continue;
        }
        if (r + 1..=last).filter(|&s| !moved[s]).all(|s| !conflict(r, s)) {
            moved[r] = true;
        }
    }
    let n = moved.iter().filter(|x| **x).count();
    if n == 0 {
        return 0;
    }
    let old = std::mem::take(&mut m.blocks[0].insts);
    let (mut keep, mut tail) = (Vec::new(), Vec::new());
    for (i, inst) in old.into_iter().enumerate() {
        if i <= last && moved[i] {
            tail.push(inst);
        } else {
            keep.push(inst);
            if i == last {
                keep.append(&mut tail);
            }
        }
    }
    m.blocks[0].insts = keep;
    n
}
