//! Post-allocation dead-write elimination (gfx1250, block-local): an instruction without side effects whose every written PHYSICAL register
//! (dword) is overwritten later in the same block before anything reads it - and whose SCC / VCC write, if any, is overwritten the same
//! way - is removed. The value-level DCE before allocation cannot see these: a tensor descriptor's group words are first assembled from one
//! template, then partly rewritten (the kernel prologue of the MXFP4 GEMM wrote 4 x 8 SGPR words twice, and computed an address into them
//! that the second write discarded). Repeats until nothing changes (removing a write can make its operands' producers dead).

use crate::mir::*;
use crate::regalloc::Alloc;
use crate::sched::effects::{effects, VCC};

/// A physical location: VGPR n -> n, SGPR n -> (1 << 20) + n, VCC -> u32::MAX.
fn keys(m: &MFunction, a: &Alloc, slots: &[u32]) -> Vec<u32> {
    let mut out = Vec::new();
    for &s in slots {
        if s == VCC {
            out.push(u32::MAX);
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
    if m.exact || std::env::var("AMDC_NO_DSE").is_ok() {
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
