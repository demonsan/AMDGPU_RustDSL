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

/// `AMDC_NO_FILL_VCC=1`: the old behaviour (no filling in front of a vcc consumer) for A/B measurements
fn opt_fill_vcc() -> bool {
    std::env::var("AMDC_NO_FILL_VCC").map_or(true, |v| v.is_empty())
}

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

/// A conversion writing a selected part of its destination (op_sel dst select: the high half / a byte) - the next consumer of the
/// register needs one wait state (gfx950 cvt-scale forwarding; LLVM inserts `s_nop 0`, measured with probe_fp4_chain).
fn cvt_dst_sel_write(op: MOp, imm: i64) -> bool {
    match op {
        MOp::VCvtPkFp8Hi | MOp::VCvtPkFp8F32Hi => true,
        MOp::VCvtScalePkFp8F32 | MOp::VCvtScalePkBf8F32 | MOp::VCvtScalePkFp4F32 | MOp::VCvtPkBf8F32 => imm != 0,
        _ => false,
    }
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
        // high-half scalef32 conversion result -> any consumer needs one wait state
        let mut hiw: HashMap<VReg, usize> = HashMap::new();
        // strict: VALU write of vcc -> VALU read of vcc needs 2 wait states (LLVM oracle: _cmp -> v_cndmask, _add_co -> v_addc).
        let vccw = std::cell::Cell::new(0usize);
        let mut cycle = 0usize;

        // The earliest cycle at which `inst` may issue given the in-flight results.
        let earliest_of = |inst: &MInst, ready: &HashMap<VReg, (usize, bool)>, vw: &HashMap<VReg, usize>, pkw: &HashMap<VReg, usize>, aw: &HashMap<VReg, usize>, hiw: &HashMap<VReg, usize>, cycle: usize| -> usize {
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
            // the HIGH-half conversion (`v_cvt_scalef32_pk_fp8_* ... op_sel:[0,0,1]`, a partial-dword write) needs one wait state before ANY consumer of its result (hardware: a dependent
            // adjacent `v_accvgpr_write` and a `v_mov_b32` copy read a stale value; one `s_nop` fixes it) - except the partner conversion of the other half, which merges into the same
            // dword and is interlocked (the Lo / Hi pairs of the shipped kernels run unpadded)
            if !matches!(inst.op, MOp::VCvtPkFp8Lo | MOp::VCvtPkFp8Hi | MOp::VCvtPkFp8F32 | MOp::VCvtPkFp8F32Hi) {
                for s in &inst.srcs {
                    if let Src::Reg(r) = s {
                        if let Some(rd) = hiw.get(&canon(*r)) {
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
            // the accumulator operand (srcC) of a matrix op: a dependent same-opcode MFMA accumulating in place is interlocked (back-to-back)
            let c_reg = match crate::op_attrs::attrs(inst.op).mfma {
                Some(ma) if ma.chain => {
                    let idx = if ma.sparse { 3 } else { 2 };
                    match inst.srcs.get(idx) {
                        Some(Src::Reg(r)) => Some(canon(*r)),
                        _ => None,
                    }
                }
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
            let mut earliest = earliest_of(&insts[i], &ready, &vw, &pkw, &aw, &hiw, cycle);

            // Fill the stall with independent ALU work from further down the block.
            // (a vcc consumer may be filled too, with instructions that neither read nor write vcc: they cannot disturb the compare it waits for)
            let vcc_cons = insts[i].op.consumes_vcc();
            if earliest > cycle && (!vcc_cons || opt_fill_vcc()) {
                let mut j = i + 1;
                let limit = (i + 1 + FILL_WINDOW).min(insts.len());
                while earliest > cycle && j < limit.min(insts.len()) {
                    if is_fence(insts[j].op) {
                        break;
                    }
                    let cand = &insts[j];
                    // pinned (hand-authored) instructions are never moved, and nothing is moved across them
                    let touches_vcc = |op: MOp| {
                        let at = crate::op_attrs::attrs(op);
                        at.vcc_read || at.vcc_write || at.defines_vcc || at.consumes_vcc || op == MOp::VPtrAdd
                    };
                    let mut ok = !cand.pin
                        && !insts[i..j].iter().any(|x| x.pin)
                        && !(vcc_cons && touches_vcc(cand.op))
                        && movable(cand.op)
                        && cand.dst.is_some()
                        && earliest_of(cand, &ready, &vw, &pkw, &aw, &hiw, cycle) <= cycle;
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
                            hiw.remove(&canon(d));
                            if cvt_dst_sel_write(x.op, x.imm) {
                                hiw.insert(canon(d), cycle + 2);
                            }
                            if x.op == MOp::VAccvgprRead {
                                aw.insert(canon(d), cycle + 3);
                            }
                            pkw.remove(&canon(d));
                            if strict && matches!(x.op, MOp::VPkAddF32 | MOp::VPkMulF32) {
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
                    earliest = earliest_of(&insts[i], &ready, &vw, &pkw, &aw, &hiw, cycle);
                }
            }

            let inst = insts[i].clone();
            let earliest = earliest_of(&inst, &ready, &vw, &pkw, &aw, &hiw, cycle);
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
                hiw.remove(&canon(d));
                if cvt_dst_sel_write(inst.op, inst.imm) {
                    hiw.insert(canon(d), cycle + 2);
                }
                if inst.op == MOp::VAccvgprRead {
                    aw.insert(canon(d), cycle + 3);
                }
                if let Some(dly) = valu_writer_delay(inst.op) {
                    vw.insert(canon(d), cycle + 1 + dly);
                }
                if matches!(inst.op, MOp::VPermlane32Swap | MOp::VPermlane16Swap) {
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
                    // a zero-C MFMA (the first of an accumulation chain) chains like its accumulating form: the same instruction with srcC = 0
                    let zero_c = crate::op_attrs::attrs(inst.op).mfma.map_or(false, |m| m.zero_c);
                    ready.insert(canon(d), (cycle + lat, chains_in_place(inst.op) || zero_c));
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
                MOp::VPermlane32Swap | MOp::VPermlane16Swap => {
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
                    if matches!(inst.op, MOp::VPermlane32Swap | MOp::VPermlane16Swap) {
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

/// One in-flight memory load at the PHYSICAL register level: its counter (vmcnt / lgkmcnt), how many same-counter operations were issued
/// after it (min over paths), whether it is a scalar load (returns out of order), and its destination registers (file-tagged).
#[derive(Clone, Debug, PartialEq, Eq)]
struct Flight {
    vm: bool,
    after: u32,
    smem: bool,
    regs: (u64, u64),
}

/// Post-RA guard: an instruction that WRITES a physical register still targeted by an in-flight load of a DIFFERENT instruction gets a
/// wait first. The pre-RA wait accounting is per value, so it cannot see the allocator handing a load's registers to another value on a
/// path where the load's result is not used (found in the MLA decode: Q-fragment loads issued before an if / else were in flight into the
/// else arm, whose temporaries reused those registers; the late loads clobbered the else arm's store offsets). Same for loads whose result
/// is never read. Returns the number of waits inserted.
pub fn pad_inflight_waw(m: &mut MFunction, a: &crate::regalloc::Alloc) -> usize {
    let key = |m: &MFunction, r: VReg| -> (u64, u64) {
        let b = a.phys[r as usize] as u64 + if a.scalar.get(r as usize).copied().unwrap_or(false) { 1 << 32 } else { 0 };
        (b, b + m.vreg_class(r).dwords() as u64)
    };
    let overl = |x: (u64, u64), y: (u64, u64)| x.0 < y.1 && y.0 < x.1;
    let succs = crate::sched::liveness::successors(m);
    let n = m.blocks.len();
    let merge = |into: &mut Vec<Flight>, from: &[Flight]| -> bool {
        let mut ch = false;
        for f in from {
            match into.iter_mut().find(|g| g.vm == f.vm && g.regs == f.regs && g.smem == f.smem) {
                Some(g) => {
                    if f.after < g.after {
                        g.after = f.after;
                        ch = true;
                    }
                }
                None => {
                    into.push(f.clone());
                    ch = true;
                }
            }
        }
        ch
    };
    // simulate one block from `st`; with `emit`, return the rewritten instruction list
    let run = |m: &MFunction, bi: usize, mut st: Vec<Flight>, emit: bool| -> (Vec<Flight>, Vec<MInst>, usize) {
        let mut out = Vec::new();
        let mut added = 0usize;
        for inst in &m.blocks[bi].insts {
            let at = crate::op_attrs::attrs(inst.op);
            match inst.op {
                MOp::SWaitVm => st.retain(|f| !(f.vm && !f.smem && f.after >= inst.imm as u32)),
                MOp::SWaitLgkm => {
                    let k = inst.imm as u32;
                    // lgkmcnt(k) retires the LDS entries with >= k younger operations, and every scalar load only at k = 0
                    st.retain(|f| !(!f.vm && (if f.smem { k == 0 } else { f.after >= k })));
                }
                _ => {}
            }
            // registers this instruction writes
            let mut writes: Vec<(u64, u64)> = inst.dst.iter().map(|d| key(m, *d)).collect();
            if matches!(inst.op, MOp::VPermlane32Swap | MOp::VPermlane16Swap) {
                for s in inst.srcs.iter().take(2) {
                    if let Src::Reg(r) = s {
                        writes.push(key(m, *r));
                    }
                }
            }
            let is_mem = at.mem.is_some() || at.smem;
            let my_vm = at.mem.map_or(false, |mm| matches!(mm.space, crate::sched::effects::Space::Global));
            let mut need_vm: Option<u32> = None;
            let mut need_lg: Option<u32> = None;
            for w in &writes {
                for f in &st {
                    if !overl(*w, f.regs) {
                        continue;
                    }
                    // a load overwriting the destination of an older load of the same in-order counter is ordered by the hardware
                    if is_mem && at.mem.map_or(false, |mm| !mm.write) && f.vm == my_vm && !f.smem && !at.smem {
                        continue;
                    }
                    if f.vm {
                        need_vm = Some(need_vm.map_or(f.after, |x: u32| x.min(f.after)));
                    } else {
                        let k = if f.smem || st.iter().any(|g| !g.vm && g.smem) { 0 } else { f.after };
                        need_lg = Some(need_lg.map_or(k, |x: u32| x.min(k)));
                    }
                }
            }
            if emit && std::env::var("AMDC_WAW_DEBUG").map_or(false, |v| !v.is_empty()) && (need_vm.is_some() || need_lg.is_some()) {
                eprintln!("inflight-waw:   wait vm {need_vm:?} lg {need_lg:?} before {:?} writes {writes:?}", inst.op);
            }
            if let Some(k) = need_vm {
                if emit {
                    out.push(MInst::new(MOp::SWaitVm).imm(k as i64));
                }
                added += 1;
                st.retain(|f| !(f.vm && f.after >= k));
            }
            if let Some(k) = need_lg {
                if emit {
                    out.push(MInst::new(MOp::SWaitLgkm).imm(k as i64));
                }
                added += 1;
                st.retain(|f| !(!f.vm && (if f.smem { k == 0 } else { f.after >= k })));
            }
            if is_mem {
                let vm = my_vm;
                for f in st.iter_mut() {
                    if f.vm == vm {
                        f.after += 1;
                    }
                }
                let lim = if vm { 63 } else { 15 };
                // a counter saturates: an entry with that many younger SAME-counter operations has completed
                st.retain(|f| f.smem || f.vm != vm || f.after < lim);
                if at.mem.map_or(true, |mm| !mm.write) || at.smem {
                    for w in &writes {
                        st.push(Flight { vm, after: 0, smem: at.smem, regs: *w });
                    }
                }
            }
            if emit {
                out.push(inst.clone());
            }
        }
        (st, out, added)
    };
    let mut entry: Vec<Vec<Flight>> = vec![Vec::new(); n];
    let mut changed = true;
    let mut guard = 0;
    while changed && guard < 64 {
        changed = false;
        guard += 1;
        for bi in 0..n {
            let (exit, _, _) = run(m, bi, entry[bi].clone(), false);
            for &s in &succs[bi] {
                if s < n {
                    let mut e = entry[s].clone();
                    if merge(&mut e, &exit) {
                        entry[s] = e;
                        changed = true;
                    }
                }
            }
        }
    }
    let mut total = 0;
    let dbg = std::env::var("AMDC_WAW_DEBUG").map_or(false, |v| !v.is_empty());
    for bi in 0..n {
        if dbg {
            eprintln!("inflight-waw: block {bi} ({}) enters with {} in-flight loads: {:?}", m.blocks[bi].label, entry[bi].len(), entry[bi].iter().map(|f| (f.vm, f.after, f.regs)).collect::<Vec<_>>());
        }
        let (_, out, added) = run(m, bi, entry[bi].clone(), true);
        if added > 0 {
            m.blocks[bi].insts = out;
            total += added;
        }
    }
    total
}

/// One `s_nop` of the final code, classified (`amdc nops`): the hazard rule that most plausibly required it, the producer and the consumer.
#[derive(Clone, Debug)]
pub struct NopSite {
    pub block: usize,
    pub index: usize,
    pub states: usize,
    pub rule: &'static str,
    pub producer: Option<MOp>,
    pub consumer: Option<MOp>,
    /// the IR value the consumer was lowered from (diagnostics origin; u32::MAX = none)
    pub origin: u32,
}

/// Post-hoc classification of every `s_nop` of an allocated function: the next real instruction is the consumer; the closest earlier
/// instruction (within the block, up to 32 back) whose result it reads (or whose vcc / whose register it overwrites) is the producer, and the
/// producer's class names the rule: mfma-result, transcendental, valu-to-special-operand (MFMA operand / permlane / DPP / readfirstlane),
/// vcc, pk-to-transcendental, accvgpr-read-to-ptr-add, cvt-dst-select, M0 (LDS DMA), else block-end / phys-WAR / other.
pub fn classify_nops(m: &MFunction) -> Vec<NopSite> {
    let alias = m.alias.clone();
    let canon = |r: VReg| alias.get(&r).map(|a| a.0).unwrap_or(r);
    let mut out = Vec::new();
    for (bi, b) in m.blocks.iter().enumerate() {
        for (ii, inst) in b.insts.iter().enumerate() {
            if inst.op != MOp::SNop || inst.pin {
                continue;
            }
            let states = inst.imm.max(0) as usize + 1;
            // the consumer: the next non-nop instruction of the block
            let cons = b.insts[ii + 1..].iter().find(|x| x.op != MOp::SNop);
            let Some(c) = cons else {
                out.push(NopSite { block: bi, index: ii, states, rule: "block-end", producer: None, consumer: None, origin: u32::MAX });
                continue;
            };
            let reads: Vec<VReg> = c.uses().into_iter().map(canon).collect();
            let cd = c.dst.map(canon);
            let mut rule = "other";
            let mut prod = None;
            let at = crate::op_attrs::attrs(c.op);
            if at.m0_src.is_some() && ii > 0 && matches!(b.insts[ii - 1].op, MOp::SMovM0 | MOp::SAddM0) {
                rule = "m0-write-to-lds-dma";
                prod = Some(b.insts[ii - 1].op);
            } else {
                // two passes: first a producer of a hazard class (an MFMA / transcendental / special VALU result the consumer reads), then any
                let window: Vec<&MInst> = b.insts[..ii].iter().rev().filter(|x| x.op != MOp::SNop).take(32).collect();
                let hazardous = |p: &MInst| {
                    let pa = crate::op_attrs::attrs(p.op);
                    pa.mfma.is_some() || pa.transcendental || cvt_dst_sel_write(p.op, p.imm) || (pa.valu_writer_delay.is_some() && at.reads_valu_hazard)
                        || p.op == MOp::VAccvgprRead
                };
                let order: Vec<&MInst> = window.iter().copied().filter(|p| hazardous(p)).chain(window.iter().copied()).collect();
                for p in order {
                    let pa = crate::op_attrs::attrs(p.op);
                    let pd = p.dst.map(canon);
                    let reads_it = pd.map_or(false, |d| reads.contains(&d));
                    if c.op.consumes_vcc() && (pa.defines_vcc || p.op == MOp::VPtrAdd) {
                        rule = "vcc-write-to-vcc-read";
                        prod = Some(p.op);
                        break;
                    }
                    if reads_it || (pd.is_some() && pd == cd && pa.mfma.is_some()) {
                        prod = Some(p.op);
                        rule = if pa.mfma.is_some() {
                            "mfma-result"
                        } else if pa.transcendental {
                            "transcendental-result"
                        } else if cvt_dst_sel_write(p.op, p.imm) {
                            "cvt-dst-select"
                        } else if p.op == MOp::VAccvgprRead && c.op == MOp::VPtrAdd {
                            "accvgpr-read-to-ptr-add"
                        } else if matches!(p.op, MOp::VPkAddF32 | MOp::VPkMulF32) && at.transcendental {
                            "pk-f32-to-transcendental"
                        } else if pa.valu_writer_delay.is_some() && at.reads_valu_hazard {
                            "valu-to-special-operand"
                        } else {
                            "other-dependence"
                        };
                        break;
                    }
                }
            }
            if prod.is_none() && rule == "other" {
                rule = "no-local-producer (block edge / phys WAR)";
            }
            out.push(NopSite { block: bi, index: ii, states, rule, producer: prod, consumer: Some(c.op), origin: c.origin });
        }
    }
    out
}

/// Post-RA: a VALU instruction that WRITES an SGPR (`v_readfirstlane` / `v_readlane`, a VOP3 compare into an SGPR pair) and a later reader of
/// that physical SGPR (LLVM GCNHazardRecognizer, gfx950 = GFX9 with the gfx940 instructions):
///   * a VMEM instruction reading it (soffset, resource, saddr): 5 wait states (`checkVMEMHazards`, hasVMEMReadSGPRVALUDefHazard). LLVM counts
///     from the VALU write even when an SALU rewrote the register in between, so this pass does too.
///   * a VALU instruction reading it as an explicit operand (incl. a v_cndmask lane mask): 2 wait states (`checkVALUHazards`,
///     hasVDecCoExecHazard, "VALUWriteSGPRVALURead").
/// Writes still pending at a block end are carried into the successors (to a fixpoint). Returns the wait states added.
/// `AMDC_NO_VALU_SGPR=1`: off (A/B).
pub fn pad_valu_sgpr(m: &mut MFunction, a: &crate::regalloc::Alloc) -> usize {
    use crate::sched::model::Pipe;
    if std::env::var("AMDC_NO_VALU_SGPR").map_or(false, |v| v == "1") {
        return 0;
    }
    const VMEM_WAIT: usize = 5;
    const VALU_WAIT: usize = 2;
    let pipe = |op: MOp| crate::op_attrs::attrs(op).cost.pipe;
    let is_valu = |op: MOp| matches!(pipe(op), Pipe::Valu | Pipe::Trans | Pipe::Mfma);
    let srange = |m: &MFunction, r: VReg| -> Option<(u32, u32)> {
        let c = m.vreg_class(r);
        c.is_scalar().then(|| (a.phys[r as usize], a.phys[r as usize] + c.dwords()))
    };
    let states = |i: &MInst| if i.op == MOp::SNop { i.imm.max(0) as usize + 1 } else { 1 };
    // the VALU-written SGPR ranges still inside the longest window when a block ends: (lo, hi, wait states elapsed at the end)
    let succs = crate::sched::liveness::successors(m);
    let nb = m.blocks.len();
    let mut entry: Vec<Vec<(u32, u32, usize)>> = vec![Vec::new(); nb];
    let block_exit = |m: &MFunction, bi: usize, ent: &[(u32, u32, usize)]| -> Vec<(u32, u32, usize)> {
        // pending: (lo, hi, states since the write)
        let mut pend: Vec<(u32, u32, usize)> = ent.to_vec();
        for inst in &m.blocks[bi].insts {
            let s = states(inst);
            for p in pend.iter_mut() {
                p.2 += s;
            }
            if is_valu(inst.op) {
                if let Some(d) = inst.dst {
                    if let Some((lo, hi)) = srange(m, d) {
                        pend.push((lo, hi, 0));
                    }
                }
            }
            pend.retain(|p| p.2 < VMEM_WAIT);
        }
        pend
    };
    for _ in 0..8 {
        let mut changed = false;
        for bi in 0..nb {
            let ex = block_exit(m, bi, &entry[bi]);
            for &s in &succs[bi] {
                for e in &ex {
                    match entry[s].iter_mut().find(|x| x.0 == e.0 && x.1 == e.1) {
                        Some(x) if x.2 <= e.2 => {}
                        Some(x) => {
                            x.2 = e.2;
                            changed = true;
                        }
                        None => {
                            entry[s].push(*e);
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
    let mut added = 0usize;
    for bi in 0..nb {
        let insts = std::mem::take(&mut m.blocks[bi].insts);
        let mut out: Vec<MInst> = Vec::with_capacity(insts.len() + 4);
        // (lo, hi, states elapsed since the write, counted up to the CURRENT instruction's issue)
        let mut pend: Vec<(u32, u32, usize)> = entry[bi].clone();
        for inst in insts {
            let p = pipe(inst.op);
            let need_win = if p == Pipe::Vmem { VMEM_WAIT } else if is_valu(inst.op) { VALU_WAIT } else { 0 };
            if need_win > 0 {
                let mut need = 0usize;
                for s in &inst.srcs {
                    if let Src::Reg(r) = s {
                        if let Some((lo, hi)) = srange(m, *r) {
                            for w in &pend {
                                if w.0 < hi && lo < w.1 && w.2 < need_win {
                                    need = need.max(need_win - w.2);
                                }
                            }
                        }
                    }
                }
                let mut n = need;
                while n > 0 {
                    let k = n.min(8);
                    out.push(MInst::new(MOp::SNop).imm(k as i64 - 1));
                    for w in pend.iter_mut() {
                        w.2 += k;
                    }
                    n -= k;
                    added += k;
                }
            }
            let s = states(&inst);
            let def = if is_valu(inst.op) { inst.dst.and_then(|d| srange(m, d)) } else { None };
            out.push(inst);
            for w in pend.iter_mut() {
                w.2 += s;
            }
            if let Some((lo, hi)) = def {
                pend.push((lo, hi, 0));
            }
            pend.retain(|w| w.2 < VMEM_WAIT);
        }
        m.blocks[bi].insts = out;
    }
    added
}
