//! M0 reuse for LDS-DMA loads (`CompileProfile::Gemm`): consecutive DMAs of a basic block that take the same `M0` source register (the LDS base; only the instruction
//! offset differs) need the `s_mov_b32 m0, sX` + the mandatory `s_nop 0` only once. `MInst::skip_m0` marks the later ones and the emitter drops the two lines.
//! The state resets at every block start and at every pinned (hand-written) instruction; a redefinition of the source register in between invalidates it.

use crate::mir::*;

fn m0_src(inst: &MInst) -> Option<VReg> {
    // the source operand holding M0 (op_attrs: `m0_src`)
    let k = crate::op_attrs::attrs(inst.op).m0_src?;
    match inst.srcs.get(k) {
        Some(Src::Reg(r)) => Some(*r),
        _ => None,
    }
}

/// Mark the redundant M0 writes; returns how many were saved.
pub fn reuse(m: &mut MFunction) -> usize {
    let alias = m.alias.clone();
    // (base register, dword offset): two views of one tuple are the same source only at the same offset
    let canon = |r: VReg| alias.get(&r).map(|a| (a.0, a.1 as u32)).unwrap_or((r, 0));
    let mut saved = 0;
    for b in &mut m.blocks {
        let mut last: Option<(VReg, u32)> = None;
        for inst in b.insts.iter_mut() {
            // a pinned (hand-written) instruction or another M0 writer (`set_m0`, an M0 add) changes M0
            if inst.pin || is_m0_writer(inst.op) {
                last = None;
                continue;
            }
            if let Some(r) = m0_src(inst) {
                let c = canon(r);
                if last == Some(c) {
                    inst.skip_m0 = true;
                    saved += 1;
                } else {
                    inst.skip_m0 = false;
                    last = Some(c);
                }
                continue;
            }
            // a write to any part of the source's tuple invalidates it
            if let (Some(d), Some(l)) = (inst.dst, last) {
                if canon(d).0 == l.0 {
                    last = None;
                }
            }
        }
    }
    saved
}

fn is_m0_writer(op: MOp) -> bool {
    matches!(op, MOp::SMovM0 | MOp::SAddM0)
}

/// Make the M0 write of every LDS-DMA load explicit and move it as early as it can go, so the mandatory wait state between the M0 write and the
/// DMA is filled by the instructions in between instead of an `s_nop 0` (`pad` adds the nop only where nothing ended up in between):
///  * a DMA that still writes its own M0 (`!skip_m0`) gets `s_mov_b32 m0, src` in front of it and becomes `skip_m0`;
///  * when `src` is `s_add_u32 src, x, y` used only here (and SCC is dead at the DMA), the add writes M0 directly (`s_add_u32 m0, x, y`);
///  * the M0 write is hoisted above every instruction that neither touches M0 (another M0 write / DMA), nor defines its sources, nor reads SCC
///    (for the add form), nor is pinned - i.e. to just after the previous DMA (or the definition of its base).
/// Returns (explicit writes, folded adds).
pub fn explicit(m: &mut MFunction) -> (usize, usize) {
    let env_off = |n: &str| std::env::var(n).map_or(false, |v| v == "0");
    if env_off("AMDC_M0_EXPLICIT") {
        return (0, 0);
    }
    let (no_fold, no_hoist) = (env_off("AMDC_M0_FOLD"), env_off("AMDC_M0_HOIST"));
    let alias = m.alias.clone();
    let canon = |r: VReg| alias.get(&r).map(|a| a.0).unwrap_or(r);
    let mut uses: std::collections::HashMap<VReg, u32> = std::collections::HashMap::new();
    for b in &m.blocks {
        for i in &b.insts {
            for r in i.uses() {
                *uses.entry(canon(r)).or_insert(0) += 1;
            }
        }
        if let Term::CondGoto { .. } = b.term {}
    }
    let (mut made, mut folded) = (0, 0);
    for b in &mut m.blocks {
        // 1. explicit writers
        let mut i = 0;
        while i < b.insts.len() {
            let inst = &b.insts[i];
            if !inst.pin && !inst.skip_m0 {
                if let Some(r) = m0_src(inst) {
                    let mut w = MInst::new(MOp::SMovM0).srcreg(r);
                    w.region = inst.region;
                    w.origin = inst.origin;
                    b.insts[i].skip_m0 = true;
                    // the DMA no longer reads its M0 source register (the explicit write does): drop the operand, or the register would stay
                    // live up to the DMA - and become an undefined use (live from the kernel entry) once the write folds its defining add
                    let k = crate::op_attrs::attrs(b.insts[i].op).m0_src.expect("m0 operand");
                    b.insts[i].srcs[k] = Src::Imm(0);
                    b.insts.insert(i, w);
                    made += 1;
                    i += 2;
                    continue;
                }
            }
            i += 1;
        }
        // 2. fold the add, 3. hoist
        let scc_live_at = |insts: &[MInst], term: &Term, p: usize| -> bool {
            for z in insts.iter().skip(p) {
                let at = crate::op_attrs::attrs(z.op);
                if at.scc_read {
                    return true;
                }
                if at.scc_write {
                    return false;
                }
            }
            matches!(term, Term::CondGoto { on, .. } if !matches!(on, Pred::Vccnz | Pred::Vccz))
        };
        let mut p = 0;
        while p < b.insts.len() {
            if b.insts[p].op != MOp::SMovM0 {
                p += 1;
                continue;
            }
            // fold: the source defined by a single-use s_add_u32 in this block
            if let (false, Some(Src::Reg(r))) = (no_fold, b.insts[p].srcs.first().cloned()) {
                let rc = canon(r);
                if uses.get(&rc).copied() == Some(1) && !alias.contains_key(&r) {
                    if let Some(q) = (0..p).rev().find(|&q| b.insts[q].dst.map_or(false, |d| canon(d) == rc)) {
                        let add = b.insts[q].clone();
                        let srcs_ok = add.op == MOp::SAddU32 && add.srcs.len() == 2 && !add.pin;
                        // the add's sources unchanged between q and p, and SCC dead at the DMA position
                        let src_regs: Vec<VReg> = add.uses().into_iter().map(|x| canon(x)).collect();
                        let unchanged = b.insts[q + 1..p].iter().all(|z| z.dst.map_or(true, |d| !src_regs.contains(&canon(d))));
                        if srcs_ok && unchanged && !scc_live_at(&b.insts, &b.term, p + 1) {
                            let mut w = MInst::new(MOp::SAddM0);
                            w.srcs = add.srcs.clone();
                            w.region = b.insts[p].region;
                            w.origin = b.insts[p].origin;
                            b.insts[p] = w;
                            b.insts.remove(q);
                            p -= 1;
                            folded += 1;
                        }
                    }
                }
            }
            // chained constant adds: s_add_u32 m0, s, k  with  s = s_add_u32 x, k2 (single use)  ->  s_add_u32 m0, x, k + k2
            loop {
                if no_fold || b.insts[p].op != MOp::SAddM0 {
                    break;
                }
                let (Some(Src::Reg(s)), Some(Src::Imm(k))) = (b.insts[p].srcs.first().cloned(), b.insts[p].srcs.get(1).cloned()) else { break };
                let sc = canon(s);
                if uses.get(&sc).copied() != Some(1) || alias.contains_key(&s) {
                    break;
                }
                let Some(q) = (0..p).rev().find(|&q| b.insts[q].dst.map_or(false, |d| canon(d) == sc)) else { break };
                let add = b.insts[q].clone();
                let (Some(Src::Reg(x)), Some(Src::Imm(k2))) = (add.srcs.first().cloned(), add.srcs.get(1).cloned()) else { break };
                if add.op != MOp::SAddU32 || add.pin || b.insts[q + 1..p].iter().any(|z| z.dst.map_or(false, |d| canon(d) == canon(x))) {
                    break;
                }
                let sum = (k as i64 + k2 as i64) as i32 as i64;
                b.insts[p].srcs = vec![Src::Reg(x), Src::Imm(sum)];
                b.insts.remove(q);
                p -= 1;
                folded += 1;
            }
            // hoist: the M0 write, together with the single-use SALU instruction right above it that computes its source (e.g. a recomputed
            // `s_mul_i32 base, wave, size`), up to the previous M0 user / the definition of their inputs / an SCC reader
            let mut start = p;
            if p > 0 {
                let z = &b.insts[p - 1];
                let feeds = z.dst.map_or(false, |d| b.insts[p].uses().iter().any(|u| canon(*u) == canon(d)) && uses.get(&canon(d)).copied() == Some(1) && !alias.contains_key(&d));
                let salu = matches!(z.op, MOp::SMulI32 | MOp::SAddU32 | MOp::SMovB32 | MOp::SAndB32 | MOp::SLshrB32);
                if feeds && salu && !z.pin && !crate::op_attrs::attrs(z.op).scc_read {
                    start = p - 1;
                }
            }
            let group: Vec<MInst> = b.insts[start..=p].to_vec();
            let gdefs: Vec<VReg> = group.iter().filter_map(|z| z.dst.map(|d| canon(d))).collect();
            let gsrcs: Vec<VReg> = group.iter().flat_map(|z| z.uses()).map(|x| canon(x)).filter(|x| !gdefs.contains(x)).collect();
            let writes_scc = group.iter().any(|z| crate::op_attrs::attrs(z.op).scc_write);
            let mut j = start;
            while j > 0 {
                let z = &b.insts[j - 1];
                let at = crate::op_attrs::attrs(z.op);
                let blocks = z.pin
                    || is_m0_writer(z.op)
                    || at.m0_src.is_some()
                    || z.dst.map_or(false, |d| gsrcs.contains(&canon(d)) || gdefs.contains(&canon(d)))
                    || z.uses().iter().any(|u| gdefs.contains(&canon(*u)))
                    || (writes_scc && (at.scc_read || at.scc_write))
                    || matches!(z.op, MOp::SchedGroup);
                if blocks {
                    break;
                }
                j -= 1;
            }
            if j < start && !no_hoist {
                let moved: Vec<MInst> = b.insts.drain(start..=p).collect();
                for (k, z) in moved.into_iter().enumerate() {
                    b.insts.insert(j + k, z);
                }
            }
            p += 1;
        }
    }
    (made, folded)
}

/// The wait state an M0 write needs before the LDS-DMA that reads it: an `s_nop 0` where the explicit write ended up right before its DMA.
pub fn pad(m: &mut MFunction) -> usize {
    let mut n = 0;
    for b in &mut m.blocks {
        let mut i = 1;
        while i < b.insts.len() {
            if crate::op_attrs::attrs(b.insts[i].op).m0_src.is_some() && b.insts[i].skip_m0 && is_m0_writer(b.insts[i - 1].op) {
                b.insts.insert(i, MInst::new(MOp::SNop).imm(crate::target::T.hazard("m0-lds-dma").n() as i64 - 1));
                n += 1;
                i += 1;
            }
            i += 1;
        }
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dma(m0: VReg) -> MInst {
        // buffer_load_dword ... lds: srcs rsrc, voff, soff, M0 source
        MInst::new(MOp::BufferLoadDwordX1Lds).srcreg(1).srcreg(2).srcreg(3).srcreg(m0)
    }

    fn func(alias: Vec<(VReg, (VReg, u8))>, insts: Vec<MInst>) -> MFunction {
        let mut b = amdgpu_dsl::FnBuilder::new("pp_m0", 64);
        let _ = b.let_(amdgpu_dsl::Op::WorkitemId);
        let mut m = crate::lower::lower(&b.finish()).expect("template");
        m.vregs = vec![RClass::S4, RClass::S4, RClass::V1, RClass::S1, RClass::S1, RClass::S1, RClass::S1];
        m.alias = alias.into_iter().collect();
        let mut blk = Block::new(0);
        blk.insts = insts;
        m.blocks = vec![blk];
        m
    }

    #[test]
    fn two_words_of_one_tuple_are_two_m0_sources() {
        // r5, r6: words 2 and 3 of the kernarg batch r0 (kabatch): different LDS bases
        let mut m = func(vec![(5, (0, 2)), (6, (0, 3))], vec![dma(5), dma(6)]);
        reuse(&mut m);
        assert!(!m.blocks[0].insts[1].skip_m0);
        let mut same = func(vec![(5, (0, 2)), (6, (0, 2))], vec![dma(5), dma(6)]);
        reuse(&mut same);
        assert!(same.blocks[0].insts[1].skip_m0, "control: the same word is reused");
    }

    #[test]
    fn an_m0_write_in_between_resets_the_reuse() {
        let mut m = func(vec![], vec![dma(4), MInst::new(MOp::SMovM0).srcreg(5), dma(4)]);
        reuse(&mut m);
        assert!(!m.blocks[0].insts[2].skip_m0);
    }
}
