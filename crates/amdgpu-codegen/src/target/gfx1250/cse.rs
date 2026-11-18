//! gfx1250 (instructions per WMMA, the MXFP4 round): value numbering of the per-load address arithmetic. The templates build every LDS
//! read address from the slot base expression again (`slot * stride + offset`, then `+ lane base`), so a window recomputes the same
//! scalar product and vector sum per load. Within a block, a pure instruction (no SCC write, `Dce::Pure`) whose operands are single-
//! definition registers and which repeats an earlier instruction (same op, operands, immediate) is dropped, its result renamed to the
//! earlier one; `x + 0` becomes the copy of `x` and a copy of a single-definition register is renamed away. Every renamed register is
//! defined once (SSA), so the earlier definition dominates the later uses. Exact kernels are never touched.

use crate::mir::*;
use std::collections::HashMap;

fn canon_of(alias: &HashMap<VReg, (VReg, u8)>, mut r: VReg) -> VReg {
    while let Some(&(u, _)) = alias.get(&r) {
        r = u;
    }
    r
}

fn defs(m: &MFunction) -> HashMap<VReg, u32> {
    let mut d: HashMap<VReg, u32> = HashMap::new();
    for b in &m.blocks {
        for i in &b.insts {
            if let Some(r) = i.dst {
                *d.entry(m.canon(r)).or_insert(0) += 1;
            }
        }
    }
    d
}

/// Returns the number of instructions removed.
pub fn cse(m: &mut MFunction) -> usize {
    if m.exact {
        return 0;
    }
    let mut removed = 0;
    loop {
        let ndefs = defs(m);
        let aliased: std::collections::HashSet<VReg> = m.alias.keys().copied().chain(m.alias.values().map(|x| x.0)).collect();
        let plain = |r: VReg| ndefs.get(&r).copied() == Some(1) && !aliased.contains(&r);
        let mut rename: HashMap<VReg, VReg> = HashMap::new();
        // every register each instruction writes (the destination and the second results of swaps / permutes), by the effects model
        let writes: Vec<Vec<Vec<u32>>> = m.blocks.iter().map(|b| b.insts.iter().map(|i| crate::sched::effects::effects(m, i).writes).collect()).collect();
        for (bi, b) in m.blocks.iter_mut().enumerate() {
            let mut seen: HashMap<(String, i64), VReg> = HashMap::new();
            // the version of every register in this block (bumped at each write of it or of its tuple): equal operands in equal versions
            // hold equal values, whether or not the register is defined once (loop variables are not)
            let mut ver: HashMap<VReg, u32> = HashMap::new();
            let mut keep = Vec::with_capacity(b.insts.len());
            let insts: Vec<MInst> = b.insts.drain(..).collect();
            // SCC dead after instruction k: no SCC reader (nor a conditional terminator) before the next SCC writer
            let scc_dead: Vec<bool> = (0..insts.len())
                .map(|k| {
                    for j in &insts[k + 1..] {
                        let a = crate::op_attrs::attrs(j.op);
                        if a.scc_read {
                            return false;
                        }
                        if a.scc_write {
                            return true;
                        }
                    }
                    !matches!(b.term, Term::CondGoto { .. })
                })
                .collect();
            for (k, inst) in insts.into_iter().enumerate() {
                let at = crate::op_attrs::attrs(inst.op);
                let pure = matches!(at.dce, crate::op_attrs::Dce::Pure) && (!at.scc_write || scc_dead[k]) && !at.scc_read && at.mem.is_none() && !at.smem && !inst.pin;
                if let (true, Some(d)) = (pure, inst.dst) {
                    let srcs_ok = true;
                    if plain(d) && srcs_ok && !rename.contains_key(&d) {
                        // a copy of a single-definition register of the same class: the register itself (the source must be defined once
                        // as well: a copy of a loop variable or of a register written in place keeps the value of the moment of the copy)
                        if matches!(inst.op, MOp::SMovB32 | MOp::VMovB32) && inst.srcs.len() == 1 {
                            if let Src::Reg(x) = inst.srcs[0] {
                                if plain(x) && m.vregs.get(x as usize) == m.vregs.get(d as usize) {
                                    rename.insert(d, x);
                                    removed += 1;
                                    continue;
                                }
                            }
                        }
                        let vs: Vec<(Src, u32)> = inst.srcs.iter().map(|s| (*s, match s { Src::Reg(r) => ver.get(&canon_of(&m.alias, *r)).copied().unwrap_or(0), _ => 0 })).collect();
                        let key = (format!("{:?} {:?}", inst.op, vs), inst.imm);
                        if let Some(&e) = seen.get(&key) {
                            if m.vregs.get(e as usize) == m.vregs.get(d as usize) {
                                rename.insert(d, e);
                                removed += 1;
                                continue;
                            }
                        }
                        seen.insert(key, d);
                    }
                }
                // every register the instruction writes (its destination, and the second results of swaps / permutes) gets a new version
                if let Some(d) = inst.dst {
                    *ver.entry(canon_of(&m.alias, d)).or_insert(0) += 1;
                }
                for &s in &writes[bi][k] {
                    if s < u32::MAX - 1 {
                        *ver.entry(canon_of(&m.alias, s / 16)).or_insert(0) += 1;
                    }
                }
                keep.push(inst);
            }
            b.insts = keep;
        }
        if rename.is_empty() {
            break;
        }
        let res = |mut r: VReg| {
            while let Some(&x) = rename.get(&r) {
                r = x;
            }
            r
        };
        for b in m.blocks.iter_mut() {
            for i in b.insts.iter_mut() {
                for s in i.srcs.iter_mut() {
                    if let Src::Reg(r) = s {
                        *r = res(*r);
                    }
                }
            }
        }
    }
    removed
}

/// `s_add d, x, 0` (and `v_add_u32 d, x, 0`) as the copy `mov d, x` (no flag the add would have set is read: checked by the caller's
/// SCC rule in `cse`, which keeps every SCC writer).
pub fn fold_zero_adds(m: &mut MFunction) -> usize {
    if m.exact {
        return 0;
    }
    let mut n = 0;
    for b in m.blocks.iter_mut() {
        for k in 0..b.insts.len() {
            let i = &b.insts[k];
            let zero = |s: &Src| matches!(s, Src::Imm(0));
            let (op, mv) = (i.op, match i.op {
                MOp::SAddU32 => MOp::SMovB32,
                MOp::VAddU32 => MOp::VMovB32,
                _ => continue,
            });
            if i.srcs.len() != 2 {
                continue;
            }
            // SCC of an s_add of 0 is 0: keep the add when the next SCC reader before another SCC writer could read it
            if op == MOp::SAddU32 {
                let mut scc_read = false;
                for j in &b.insts[k + 1..] {
                    let a = crate::op_attrs::attrs(j.op);
                    if a.scc_read {
                        scc_read = true;
                        break;
                    }
                    if a.scc_write {
                        break;
                    }
                }
                if scc_read || !matches!(b.term, Term::Goto(_) | Term::Endpgm | Term::CondGoto { .. }) {
                    continue;
                }
            }
            let x = if zero(&i.srcs[1]) { i.srcs[0] } else if zero(&i.srcs[0]) { i.srcs[1] } else { continue };
            if let Src::Reg(_) = x {
                let d = i.dst;
                let mut ni = MInst::new(mv).src(x);
                ni.dst = d;
                ni.stmt = i.stmt;
                ni.unit = i.unit;
                b.insts[k] = ni;
                n += 1;
            }
        }
    }
    n
}