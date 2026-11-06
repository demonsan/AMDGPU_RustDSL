//! Value-latency hazards and the local scheduler that hides them.
//!
//! Most VALU dependencies are interlocked by the hardware, but transcendental
//! results on gfx9 (`v_exp_f32`, `v_rcp_f32`, ...) and MFMA results are not: a
//! consumer issued too early reads the previous value of the register. This pass
//! models that latency. When producer and consumer are too close in program
//! order it first tries to fill the gap with later, independent ALU instructions
//! of the same block (a small in-block scheduler) and only pads the remainder
//! with `s_nop`.
//!
//! It is deliberately conservative: padding costs a few cycles, reading a stale
//! value costs correctness.

use std::collections::HashMap;

use crate::mir::*;

/// Cycles that must separate a transcendental write from its first reader.
const TRANS_LATENCY: usize = 5;

/// How many instructions after the stalled consumer are examined as fill candidates.
const FILL_WINDOW: usize = 48;

fn is_transcendental(op: MOp) -> bool {
    crate::op_attrs::attrs(op).transcendental
}

/// Independent instructions that must follow an MFMA before its result may be read (CDNA4 ISA section 7.6; `op_attrs`: `mfma.latency`).
fn mfma_latency(op: MOp) -> Option<usize> {
    crate::op_attrs::attrs(op).mfma.map(|m| m.latency)
}

/// In-place accumulating matrix ops whose dependent same-opcode chain is interlocked by the hardware (`mfma.chain`).
fn chains_in_place(op: MOp) -> bool {
    crate::op_attrs::attrs(op).mfma.map_or(false, |m| m.chain)
}

fn is_scaled_mfma(op: MOp) -> bool {
    crate::op_attrs::attrs(op).mfma.map_or(false, |m| m.scaled)
}

/// Plain ALU instructions that may be moved ahead of a stalled consumer (`movable`).
fn movable(op: MOp) -> bool {
    crate::op_attrs::attrs(op).movable == crate::op_attrs::Movable::Yes
}

/// VALU instructions whose VGPR result feeding an MFMA operand, a permlane, readfirstlane or DPP needs wait states first
/// (LLVM GCNHazardRecognizer: 2 states; 4 after carry-out adds; `valu_writer_delay`).
fn valu_writer_delay(op: MOp) -> Option<usize> {
    crate::op_attrs::attrs(op).valu_writer_delay
}

fn reads_valu_hazard_operands(op: MOp) -> bool {
    crate::op_attrs::attrs(op).reads_valu_hazard
}

/// Instructions the scanner must never look past (waits, barriers, nops).
fn is_fence(op: MOp) -> bool {
    matches!(op, MOp::SWaitVm | MOp::SWaitLgkm | MOp::SBarrier | MOp::SNop | MOp::SEndpgm)
}

/// Wait states from the start of block `s` (plus `d0`) to the first touch of `reg` (usize::MAX: an in-place accumulation of the same MFMA,
/// which never needs the wait); blocks that do not touch it are followed into their successors (min over paths) for up to `hops` blocks or
/// until the distance covers `need`. The terminator's own issue slot is not counted (conservative).
#[allow(clippy::too_many_arguments)]
fn first_touch(s: usize, d0: usize, hops: usize, need: usize, reg: VReg, from_mfma: bool, snap: &[Vec<MInst>], succs: &[Vec<usize>], canon: &dyn Fn(VReg) -> VReg) -> usize {
    let mut d = d0;
    for x in &snap[s] {
        let touches = x.uses().iter().any(|u| canon(*u) == reg) || x.dst.map(|y| canon(y) == reg).unwrap_or(false);
        if touches {
            let chain = from_mfma
                && crate::op_attrs::attrs(x.op).mfma.map_or(false, |m| m.chain && !m.sparse)
                && matches!(x.srcs.get(2), Some(Src::Reg(c)) if canon(*c) == reg);
            return if chain { usize::MAX } else { d };
        }
        d += if x.op == MOp::SNop { x.imm as usize + 1 } else { 1 };
    }
    if d >= need || hops == 0 || succs[s].is_empty() {
        return d;
    }
    succs[s].iter().map(|&t| first_touch(t, d, hops - 1, need, reg, from_mfma, snap, succs, canon)).min().unwrap_or(d)
}

pub fn fix_value_hazards(m: &mut MFunction) {
    fix_value_hazards_opts(m, false)
}

/// pad_block_end: also pad before the terminator until every in-flight MFMA/transcendental result is
/// readable (the per-block scan cannot see readers in successor blocks; a scheduler that sinks
/// MFMAs to the end of a block needs this).
pub fn fix_value_hazards_opts(m: &mut MFunction, strict: bool) {
    let pad_block_end = strict;
    let alias = m.alias.clone();
    let canon = |r: VReg| alias.get(&r).map(|a| a.0).unwrap_or(r);
    let succs = crate::sched::liveness::successors(m);
    let snap: Vec<Vec<MInst>> = m.blocks.iter().map(|b| b.insts.clone()).collect();
    let labels: Vec<u32> = m.blocks.iter().map(|b| b.label).collect();
    // `edgepad` (opt-in profile option): follow the successors through blocks that do not touch a register (4 hops) and pad only the edges
    // that need it
    let edge = m.edge_pad;
    let hops = if edge { 4 } else { 0 };
    let mut edge_pads: Vec<(usize, u32, usize)> = Vec::new();
    for (bi, b) in m.blocks.iter_mut().enumerate() {
        let mut insts: Vec<MInst> = std::mem::take(&mut b.insts);
        let mut out: Vec<MInst> = Vec::with_capacity(insts.len() + 8);
        // Virtual register -> (earliest cycle at which its in-flight result may be read, whether
        // the producer is a scaled MFMA).
        let mut ready: HashMap<VReg, (usize, bool)> = HashMap::new();
        // Virtual register -> earliest cycle at which a plain VALU result may feed an MFMA operand
        // or a permlane (a different, shorter hazard than the in-flight latencies above).
        let mut vw: HashMap<VReg, usize> = HashMap::new();
        // strict: v_pk_add/mul_f32 result -> v_exp/v_rcp needs one wait state (LLVM oracle).
        let mut pkw: HashMap<VReg, usize> = HashMap::new();
        // v_accvgpr_read result -> v_add_co/v_addc pointer arithmetic needs 2 wait states (LLVM oracle, AGPR rows of the hazard matrix)
        let mut aw: HashMap<VReg, usize> = HashMap::new();
        // strict: VALU write of vcc -> VALU read of vcc needs 2 wait states (LLVM oracle: _cmp -> v_cndmask, _add_co -> v_addc).
        let vccw = std::cell::Cell::new(0usize);
        let mut cycle = 0usize;

        // The earliest cycle at which `inst` may issue given the in-flight results.
        let earliest_of = |inst: &MInst, ready: &HashMap<VReg, (usize, bool)>, vw: &HashMap<VReg, usize>, pkw: &HashMap<VReg, usize>, aw: &HashMap<VReg, usize>, cycle: usize| -> usize {
            let mut earliest = cycle;
            if inst.op == MOp::VPtrAdd {
                for s in &inst.srcs {
                    if let Src::Reg(r) = s {
                        if let Some(rd) = aw.get(&canon(*r)) {
                            earliest = earliest.max(*rd);
                        }
                    }
                }
            }
            if strict && inst.op.consumes_vcc() {
                earliest = earliest.max(vccw.get());
            }
            if strict && is_transcendental(inst.op) {
                for s in &inst.srcs {
                    if let Src::Reg(r) = s {
                        if let Some(rd) = pkw.get(&canon(*r)) {
                            earliest = earliest.max(*rd);
                        }
                    }
                }
            }
            // the HIGH-half conversion (`v_cvt_scalef32_pk_fp8_* ... op_sel:[0,0,1]`, a partial-dword write) feeding a v_accvgpr_write needs one wait state (hardware: a dependent adjacent pair read a stale value, one
            // s_nop fixes it); the map is `pkw`, which holds the conversion results (and, strict, the packed-math ones)
            if inst.op == MOp::VAccvgprWrite {
                for s in &inst.srcs {
                    if let Src::Reg(r) = s {
                        if let Some(rd) = pkw.get(&canon(*r)) {
                            earliest = earliest.max(*rd);
                        }
                    }
                }
            }
            if reads_valu_hazard_operands(inst.op) {
                for s in &inst.srcs {
                    if let Src::Reg(r) = s {
                        if let Some(rd) = vw.get(&canon(*r)) {
                            earliest = earliest.max(*rd);
                        }
                    }
                }
            }
            let c_reg = match inst.op {
                MOp::VMfmaScale32x32x64Fp8 | MOp::VMfma32x32x64Fp8 | MOp::VMfmaScale16x16x128Fp8 if inst.srcs.len() > 2 => match inst.srcs[2] {
                    Src::Reg(r) => Some(canon(r)),
                    _ => None,
                },
                MOp::VSmfmac16x16x128Fp8 if inst.srcs.len() > 3 => match inst.srcs[3] {
                    Src::Reg(r) => Some(canon(r)),
                    _ => None,
                },
                _ => None,
            };
            for s in &inst.srcs {
                if let Src::Reg(r) = s {
                    let r = canon(*r);
                    if let Some((rd, from_mfma)) = ready.get(&r) {
                        // Same-opcode dependent MFMA accumulating in place needs no extra wait.
                        if *from_mfma && c_reg == Some(r) {
                            continue;
                        }
                        earliest = earliest.max(*rd);
                    }
                }
            }
            if let Some(d) = inst.dst {
                let d = canon(d);
                if let Some((rd, from_mfma)) = ready.get(&d) {
                    if !(*from_mfma && c_reg == Some(d)) {
                        earliest = earliest.max(*rd);
                    }
                }
            }
            earliest
        };

        let mut i = 0usize;
        while i < insts.len() {
            let mut earliest = earliest_of(&insts[i], &ready, &vw, &pkw, &aw, cycle);

            // Fill the stall with independent ALU work from further down the block.
            if earliest > cycle && !insts[i].op.consumes_vcc() {
                let mut j = i + 1;
                let limit = (i + 1 + FILL_WINDOW).min(insts.len());
                while earliest > cycle && j < limit.min(insts.len()) {
                    if is_fence(insts[j].op) {
                        break;
                    }
                    let cand = &insts[j];
                    // pinned (hand-authored) instructions are never moved, and nothing is moved across them
                    let mut ok = !cand.pin
                        && !insts[i..j].iter().any(|x| x.pin)
                        && movable(cand.op)
                        && cand.dst.is_some()
                        && earliest_of(cand, &ready, &vw, &pkw, &aw, cycle) <= cycle;
                    if ok {
                        let cd = canon(cand.dst.unwrap());
                        let cs: Vec<VReg> = cand.uses().into_iter().map(canon).collect();
                        for k in i..j {
                            let other = &insts[k];
                            let od = other.dst.map(canon);
                            let os: Vec<VReg> = other.uses().into_iter().map(canon).collect();
                            // RAW, WAR, WAW against everything the candidate would jump over.
                            if od.map_or(false, |d| cs.contains(&d) || d == cd) || os.contains(&cd) {
                                ok = false;
                                break;
                            }
                        }
                    }
                    if ok {
                        let x = insts.remove(j);
                        if let Some(d) = x.dst {
                            aw.remove(&canon(d));
                            if x.op == MOp::VAccvgprRead {
                                aw.insert(canon(d), cycle + 3);
                            }
                            pkw.remove(&canon(d));
                            if strict && matches!(x.op, MOp::VPkAddF32 | MOp::VPkMulF32) {
                                pkw.insert(canon(d), cycle + 2);
                            }
                            if matches!(x.op, MOp::VCvtPkFp8Hi | MOp::VCvtPkFp8F32Hi) {
                                pkw.insert(canon(d), cycle + 2);
                            }
                            ready.remove(&canon(d));
                            vw.remove(&canon(d));
                            if let Some(dly) = valu_writer_delay(x.op) {
                                vw.insert(canon(d), cycle + 1 + dly);
                            }
                        }
                        out.push(x);
                        cycle += 1;
                        // `j` now points at the next candidate; the consumer moved no further.
                    } else {
                        j += 1;
                    }
                    // The consumer may have become ready.
                    earliest = earliest_of(&insts[i], &ready, &vw, &pkw, &aw, cycle);
                }
            }

            let inst = insts[i].clone();
            let earliest = earliest_of(&inst, &ready, &vw, &pkw, &aw, cycle);
            if earliest > cycle {
                let mut missing = earliest - cycle;
                while missing > 0 {
                    let n = missing.min(7) as i64;
                    out.push(MInst::new(MOp::SNop).imm(n));
                    cycle += (n as usize) + 1;
                    missing = missing.saturating_sub(n as usize + 1);
                }
            }
            if let Some(d) = inst.dst {
                ready.remove(&canon(d));
                vw.remove(&canon(d));
                aw.remove(&canon(d));
                if inst.op == MOp::VAccvgprRead {
                    aw.insert(canon(d), cycle + 3);
                }
                if let Some(dly) = valu_writer_delay(inst.op) {
                    vw.insert(canon(d), cycle + 1 + dly);
                }
                if inst.op == MOp::VPermlane32Swap {
                    // the swap rewrites its second operand too
                    if let Some(Src::Reg(r2)) = inst.srcs.get(1) {
                        vw.insert(canon(*r2), cycle + 1 + 2);
                    }
                }
            }
            if strict {
                if let Some(d) = inst.dst {
                    pkw.remove(&canon(d));
                    if matches!(inst.op, MOp::VPkAddF32 | MOp::VPkMulF32) {
                        pkw.insert(canon(d), cycle + 2);
                    }
                }
            }
            if matches!(inst.op, MOp::VCvtPkFp8Hi | MOp::VCvtPkFp8F32Hi) {
                if let Some(d) = inst.dst {
                    pkw.insert(canon(d), cycle + 2);
                }
            }
            if is_transcendental(inst.op) {
                if let Some(d) = inst.dst {
                    ready.insert(canon(d), (cycle + TRANS_LATENCY, false));
                }
            }
            if let Some(latency) = mfma_latency(inst.op) {
                if let Some(d) = inst.dst {
                    // strict: latency wait states must lie *between* the MFMA and the reader (hardware: 19 is still stale)
                    // The scaled MFMA fix is hardware-confirmed (microbenchmark: 19 states between MFMA and reader is still
                // stale, 20 are needed), so it also applies to the default path; the other opcodes keep the table value
                // outside strict mode.
                let lat = if strict || is_scaled_mfma(inst.op) { latency + 1 } else { latency };
                    ready.insert(canon(d), (cycle + lat, chains_in_place(inst.op)));
                }
            }
            if strict && inst.op.defines_vcc() {
                vccw.set(cycle + 3);
            }
            if strict && matches!(inst.op, MOp::VPtrAdd) {
                vccw.set(cycle + 3);
            }
            // an explicit `s_nop N` already in the stream (hand-authored blocks) covers N+1 wait states
            let spent = if inst.op == MOp::SNop { inst.imm.max(0) as usize + 1 } else { 1 };
            out.push(inst);
            cycle += spent;
            i += 1;
        }
        if pad_block_end {
            // Only results that a successor reads soon need the padding; the in-place accumulation chain
            // of one MFMA opcode never does. Per successor: the wait states still missing on that edge.
            let ss = &succs[bi];
            let mut miss = vec![0usize; ss.len()];
            for (reg, (rd, from_mfma)) in ready.iter() {
                let remaining = rd.saturating_sub(cycle);
                if remaining == 0 {
                    continue;
                }
                for (k, &s) in ss.iter().enumerate() {
                    let dist = first_touch(s, 0, hops, remaining, *reg, *from_mfma, &snap, &succs, &canon);
                    if dist != usize::MAX {
                        miss[k] = miss[k].max(remaining.saturating_sub(dist));
                    }
                }
            }
            // `edgepad`: the padding every path needs stays at the block end, the rest goes on the edges that need it (a loop exit pays it once
            // instead of every iteration); otherwise the block end covers every successor
            let mut missing = if edge { miss.iter().copied().min().unwrap_or(0) } else { miss.iter().copied().max().unwrap_or(0) };
            if edge {
                for (k, &s) in ss.iter().enumerate() {
                    if miss[k] > missing {
                        edge_pads.push((bi, labels[s], miss[k] - missing));
                    }
                }
            }
            while missing > 0 {
                let n = missing.min(7) as i64;
                out.push(MInst::new(MOp::SNop).imm(n));
                missing = missing.saturating_sub(n as usize + 1);
            }
        }        b.insts = out;
    }
    // split the padded edges: a block of `s_nop`s between the branch and its successor (after the source block for the fall-through edge, so it
    // still falls through; at the end for a taken edge)
    let mut next_label = m.blocks.iter().map(|b| b.label).max().unwrap_or(0) + 1;
    for (bi, to, mut n) in edge_pads.into_iter().rev() {
        let mut insts = Vec::new();
        while n > 0 {
            let k = n.min(8);
            insts.push(MInst::new(MOp::SNop).imm(k as i64 - 1));
            n -= k;
        }
        let l = next_label;
        next_label += 1;
        let pad = Block { label: l, insts, term: Term::Goto(to), pred: None };
        let is_fall = match &mut m.blocks[bi].term {
            Term::CondGoto { target, fall, .. } => {
                if *fall == to {
                    *fall = l;
                    true
                } else {
                    if *target == to {
                        *target = l;
                    }
                    false
                }
            }
            Term::Goto(t) => {
                *t = l;
                true
            }
            Term::Endpgm => continue,
        };
        if is_fall {
            m.blocks.insert(bi + 1, pad);
        } else {
            m.blocks.push(pad);
        }
    }
}
/// Wait slots (issue distance) after a VMEM store of more than 8 bytes before a VALU may overwrite its data registers: 2 wait states on gfx940+ (so the
/// next write may issue 3 slots after the store; an `s_waitcnt` between counts as one).
const STORE_DATA_WAIT: usize = 3;

/// Global stores of more than 64 bits of data (`global_store_dwordx3/x4`): the data VGPRs are read over several cycles after issue.
fn is_wide_vmem_store(op: MOp) -> bool {
    crate::op_attrs::attrs(op).wide_store_data.is_some()
}

/// Index of the data register among the sources of a wide store (global stores: [ptr, data]; buffer stores: [data, rsrc, voff, soff]).
fn wide_store_data_idx(op: MOp) -> usize {
    crate::op_attrs::attrs(op).wide_store_data.unwrap_or(1)
}

/// VGPR or AGPR (the two files of the unified register file): the hazards below apply to both. Physical ranges are tagged per file (AGPR_TAG), so a\n/// VGPR range never overlaps an AGPR range.
fn vl(c: RClass) -> bool {
    c.is_vector() || c.is_agpr()
}

/// A VALU (non-memory, non-matrix) instruction that writes a vector register.
fn is_valu_vgpr_write(m: &MFunction, inst: &MInst) -> bool {
    inst.dst.map_or(false, |d| vl(m.vreg_class(d))) && mfma_latency(inst.op).is_none() && crate::sched::effects::effects(m, inst).mem.is_none() && !matches!(inst.op, MOp::VSmfmac16x16x128Fp8)
}

/// Wait states until an MFMA has finished WRITING its destination (not the value-level RAW latency): 16-pass ops need 18 + 1 (CDNA4 ISA 7.6 "XDL write
/// VGPR -> VALU write of the same VGPR"), the 32x32x64 scaled and sparse forms 21.
fn mfma_dst_wait(lat: usize) -> usize {
    if lat == 20 {
        21
    } else {
        19
    }
}

/// Post-register-allocation hazard: a VALU (or other) write to a physical VGPR range that a still
/// running MFMA reads (SrcA/B/C) can corrupt the MFMA. Allocation reuses registers freely and the
/// value-level pass above cannot see it; a scheduler that moves definitions closer to MFMAs makes
/// it real (found by hardware: scheduled kernels that are bit-exact through the LLVM tail but wrong
/// through the own tail). Pads `s_nop` until the MFMA's wait-state count elapsed, and before the
/// end of a block for successors. Only run for scheduled code (default output unchanged).
pub fn pad_physical_war(m: &mut MFunction, a: &crate::regalloc::Alloc) {
    let rng = |m: &MFunction, r: VReg| -> (u32, u32) {
        let b = a.phys[r as usize];
        (b, b + m.vreg_class(r).dwords())
    };
    let overlaps = |x: (u32, u32), y: (u32, u32)| x.0 < y.1 && y.0 < x.1;
    // MFMA DESTINATIONS still being written when the block is left (per block: canonical vreg, physical range, remaining wait states). A register reuse
    // by a different value in the successor must wait too. Computed from the unpadded block (padding only shortens what remains).
    let succs = crate::sched::liveness::successors(m);
    let exit_dst: Vec<Vec<(VReg, (u32, u32), usize)>> = m
        .blocks
        .iter()
        .map(|b| {
            let mut cycle = 0usize;
            let mut live: Vec<(usize, usize, VReg, (u32, u32))> = Vec::new();
            for inst in &b.insts {
                if let (Some(l), Some(d)) = (mfma_latency(inst.op), inst.dst) {
                    live.push((cycle, mfma_dst_wait(l), m.canon(d), rng(m, d)));
                }
                if matches!(inst.op, MOp::SNop) {
                    cycle += inst.imm as usize;
                }
                cycle += 1;
                live.retain(|(i, n, _, _)| i + n > cycle);
            }
            live.iter().map(|(i, n, d, r)| (*d, *r, i + n - cycle)).collect()
        })
        .collect();
    let mut entry_dst: Vec<Vec<(VReg, (u32, u32), usize)>> = vec![Vec::new(); m.blocks.len()];
    for (bi, ss) in succs.iter().enumerate() {
        for &s in ss {
            if s < entry_dst.len() {
                for e in &exit_dst[bi] {
                    entry_dst[s].push(*e);
                }
            }
        }
    }
    // `edgepad`: the SrcC WAR windows of MFMAs still running at a block end are carried into the successors instead of padding every block
    // end, and both those and the in-flight destinations are carried THROUGH short blocks (fixpoint over the CFG: max remaining per key)
    let war_of = |m: &MFunction, inst: &MInst| -> Option<(usize, Vec<(u32, u32)>)> {
        let need = mfma_latency(inst.op).map(|l| match l {
            20 => 21,
            _ => 12,
        })?;
        let mut srcs = Vec::new();
        if inst.op == MOp::VSmfmac16x16x128Fp8 {
            for s in inst.srcs.iter().take(3) {
                if let Src::Reg(r) = s {
                    if vl(m.vreg_class(*r)) {
                        srcs.push(rng(m, *r));
                    }
                }
            }
        } else if !crate::op_attrs::attrs(inst.op).mfma.map_or(false, |m| m.zero_c) {
            if let Some(Src::Reg(r)) = inst.srcs.get(2) {
                if vl(m.vreg_class(*r)) {
                    srcs.push(rng(m, *r));
                }
            }
        }
        Some((need, srcs))
    };
    let edge = m.edge_pad;
    let nb = m.blocks.len();
    let mut entry_war: Vec<Vec<(MOp, Vec<(u32, u32)>, usize)>> = vec![Vec::new(); nb];
    if edge {
        loop {
            let mut changed = false;
            for bi in 0..nb {
                let mut cycle = 0usize;
                let mut w: Vec<(usize, usize, MOp, Vec<(u32, u32)>)> = entry_war[bi].iter().map(|(o, s, r)| (0, *r, *o, s.clone())).collect();
                let mut dd: Vec<(usize, usize, VReg, (u32, u32))> = entry_dst[bi].iter().map(|(d, r, rem)| (0, *rem, *d, *r)).collect();
                for inst in &m.blocks[bi].insts {
                    if let Some((need, srcs)) = war_of(m, inst) {
                        w.push((cycle, need, inst.op, srcs));
                    }
                    if let (Some(l), Some(d)) = (mfma_latency(inst.op), inst.dst) {
                        dd.push((cycle, mfma_dst_wait(l), m.canon(d), rng(m, d)));
                    }
                    if matches!(inst.op, MOp::SNop) {
                        cycle += inst.imm as usize;
                    }
                    cycle += 1;
                    w.retain(|(i, n, _, _)| i + n > cycle);
                    dd.retain(|(i, n, _, _)| i + n > cycle);
                }
                for &s in &succs[bi] {
                    for (i, n, o, sr) in &w {
                        let rem = i + n - cycle;
                        match entry_war[s].iter_mut().find(|(o2, s2, _)| o2 == o && s2 == sr) {
                            Some(e) => {
                                if e.2 < rem {
                                    e.2 = rem;
                                    changed = true;
                                }
                            }
                            None => {
                                entry_war[s].push((*o, sr.clone(), rem));
                                changed = true;
                            }
                        }
                    }
                    for (i, n, d, r) in &dd {
                        let rem = i + n - cycle;
                        match entry_dst[s].iter_mut().find(|(d2, r2, _)| d2 == d && r2 == r) {
                            Some(e) => {
                                if e.2 < rem {
                                    e.2 = rem;
                                    changed = true;
                                }
                            }
                            None => {
                                entry_dst[s].push((*d, *r, rem));
                                changed = true;
                            }
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
    }
    let blocks = std::mem::take(&mut m.blocks);
    let mut out_blocks = Vec::with_capacity(blocks.len());
    for (bi, mut b) in blocks.into_iter().enumerate() {
        let mut out: Vec<MInst> = Vec::with_capacity(b.insts.len() + 8);
        // in-flight MFMA destinations: (issue cycle, wait states, canonical dst vreg, physical range). Entries inherited from predecessors are issued
        // at cycle 0 with only their remaining wait states left.
        let mut store_data_inflight: Vec<(usize, usize, Vec<(u32, u32)>)> = Vec::new();
        let mut dst_inflight: Vec<(usize, usize, VReg, (u32, u32))> = entry_dst[bi].iter().map(|(d, r, rem)| (0usize, *rem, *d, *r)).collect();
        // (issue cycle, wait states needed, op, source ranges)
        let mut inflight: Vec<(usize, usize, MOp, Vec<(u32, u32)>)> = entry_war[bi].iter().map(|(o, s, r)| (0usize, *r, *o, s.clone())).collect();
        let mut cycle = 0usize;
        for inst in b.insts.drain(..) {
            // destination ranges written by this instruction
            let mut wr: Vec<(u32, u32)> = Vec::new();
            match inst.op {
                MOp::VWriteSub => {
                    if let Some(d) = inst.dst {
                        let base = a.phys[d as usize] + inst.imm as u32;
                        wr.push((base, base + 1));
                    }
                }
                MOp::VPermlane32Swap => {
                    for s in inst.srcs.iter().take(2) {
                        if let Src::Reg(r) = s {
                            wr.push(rng(m, *r));
                        }
                    }
                }
                _ => {
                    if let Some(d) = inst.dst {
                        if vl(m.vreg_class(d)) {
                            wr.push(rng(m, d));
                        }
                    }
                }
            }
            if inst.op == MOp::VMovFromSPair || inst.op == MOp::VPtrAdd {
                // two-instruction expansions write the whole pair (already covered by dst range)
            }
            let mut earliest = cycle;
            // VMEM store data hazard (CDNA3/4: LLVM `has12DWordStoreHazard`, `VALUWaitStates` = 2 on gfx940+): a store of MORE than 8 bytes may still be reading
            // its data VGPRs when the next instructions issue; a VALU write of those registers needs 2 wait states first. Found on hardware as intermittent
            // corruption of the first dword of every `global_store_dwordx4` data (an LDS read-back into v[4:7] followed by the store, then the VALU chain that
            // rebuilds v4..v7 one `s_waitcnt` later): only VALU writers are affected (a nop between the store and a following LDS read changed nothing).
            if !store_data_inflight.is_empty() && is_valu_vgpr_write(m, &inst) {
                for (issue, need, srcs) in &store_data_inflight {
                    if wr.iter().any(|w| srcs.iter().any(|s| overlaps(*w, *s))) {
                        earliest = earliest.max(issue + need);
                    }
                }
            }
            // physical WAW/RAW through a DIFFERENT value: another vreg whose register range overlaps the destination of an MFMA that is still writing
            // it (typically the MFMA result was dead - e.g. the last tile of an unrolled loop copy - so allocation handed its registers to a new value).
            // The late MFMA write would overwrite that value. The MFMA's own vreg is the value-level pass's business.
            if !dst_inflight.is_empty() {
                let mut touched: Vec<VReg> = inst.uses();
                if mfma_latency(inst.op).is_none() {
                    if let Some(d) = inst.dst {
                        touched.push(d);
                    }
                    if inst.op == MOp::VPermlane32Swap {
                        for s in inst.srcs.iter().take(2) {
                            if let Src::Reg(r) = s {
                                touched.push(*r);
                            }
                        }
                    }
                }
                for (issue, need, dv, drange) in &dst_inflight {
                    for r in &touched {
                        if !vl(m.vreg_class(*r)) || m.canon(*r) == *dv {
                            continue;
                        }
                        if overlaps(rng(m, *r), *drange) {
                            earliest = earliest.max(issue + need);
                        }
                    }
                }
            }
            for (issue, need, op, srcs) in &inflight {
                if inst.op == *op && chains_in_place(inst.op) {
                    continue; // in-place accumulation chain of the same opcode
                }
                if wr.iter().any(|w| srcs.iter().any(|s| overlaps(*w, *s))) {
                    earliest = earliest.max(issue + need);
                }
            }
            while earliest > cycle {
                let n = (earliest - cycle).min(8);
                out.push(MInst::new(MOp::SNop).imm(n as i64 - 1));
                cycle += n;
            }
            if let Some(need) = mfma_latency(inst.op).map(|l| match l {
                20 => 21,
                _ => 12,
            }) {
                // Only the accumulator (SrcC) is a WAR hazard: the LLVM tail inserts nothing for a VALU
                // write to SrcA/SrcB/scale registers right after the MFMA (docs/scheduling.md).
                let mut srcs = Vec::new();
                if inst.op == MOp::VSmfmac16x16x128Fp8 {
                    // SMFMAC reads A, B and SRC2 over several passes; the accumulator C is D itself (value-level pass)
                    for s in inst.srcs.iter().take(3) {
                        if let Src::Reg(r) = s {
                            if vl(m.vreg_class(*r)) {
                                srcs.push(rng(m, *r));
                            }
                        }
                    }
                } else if !crate::op_attrs::attrs(inst.op).mfma.map_or(false, |m| m.zero_c) {
                    if let Some(Src::Reg(r)) = inst.srcs.get(2) {
                        if vl(m.vreg_class(*r)) {
                            srcs.push(rng(m, *r));
                        }
                    }
                }
                inflight.push((cycle, need, inst.op, srcs));
            }
            if is_wide_vmem_store(inst.op) {
                if let Some(Src::Reg(d)) = inst.srcs.get(wide_store_data_idx(inst.op)) {
                    if m.vreg_class(*d).is_vector() {
                        store_data_inflight.push((cycle, STORE_DATA_WAIT, vec![rng(m, *d)]));
                    }
                }
            }
            if let (Some(l), Some(d)) = (mfma_latency(inst.op), inst.dst) {
                dst_inflight.push((cycle, mfma_dst_wait(l), m.canon(d), rng(m, d)));
            }
            if matches!(inst.op, MOp::SNop) {
                cycle += inst.imm as usize;
            }
            out.push(inst);
            cycle += 1;
            inflight.retain(|(issue, need, _, _)| issue + need > cycle);
            dst_inflight.retain(|(issue, need, _, _)| issue + need > cycle);
            store_data_inflight.retain(|(issue, need, _)| issue + need > cycle);
        }
        // (edgepad: the successors inherit what is still running instead)
        let mut missing = if edge { 0 } else { inflight.iter().map(|(i, n, _, _)| (i + n).saturating_sub(cycle)).max().unwrap_or(0) };
        while missing > 0 {
            let n = missing.min(8);
            out.push(MInst::new(MOp::SNop).imm(n as i64 - 1));
            missing -= n;
        }
        b.insts = out;
        out_blocks.push(b);
    }
    m.blocks = out_blocks;
}