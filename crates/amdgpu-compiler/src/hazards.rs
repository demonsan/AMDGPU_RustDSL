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

thread_local! {
    static VCC_FILL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Run `f` with stall filling for `vcc` consumers enabled: the movable candidates never read or write `vcc`, so an independent
/// instruction may be placed between a `v_cmp` and its `v_cndmask` to cover the 2 wait states instead of an `s_nop`.
pub fn with_vcc_fill<R>(on: bool, f: impl FnOnce() -> R) -> R {
    let old = VCC_FILL.with(|c| c.replace(on));
    let r = f();
    VCC_FILL.with(|c| c.set(old));
    r
}

fn is_transcendental(op: MOp) -> bool {
    matches!(op, MOp::VExpF32 | MOp::VLogF32 | MOp::VRcpF32)
}

/// Independent instructions that must follow an MFMA before its result may be
/// read (CDNA4 ISA section 7.6, "Table: XDL write VGPR" -> "VALU read/write of
/// the destination VGPR"). The table lists 5/8/12/20 depending on the opcode
/// size; the largest value is used for the F8F6F4 shapes, which take 32 to 64
/// cycles.
fn mfma_latency(op: MOp) -> Option<usize> {
    match op {
        MOp::VMfma16x16x16F16 => Some(12),
        MOp::VMfma32x32x8F16 | MOp::VMfma32x32x8F16Z => Some(12),
        MOp::VMfma16x16x32Fp8 => Some(12),
        MOp::VMfma32x32x16Fp8 => Some(12),
        MOp::VMfmaScale32x32x64Fp8 | MOp::VMfmaScale32x32x64Fp8Z => Some(20),
        MOp::VSmfmac16x16x128Fp8 => Some(20),
        _ => None,
    }
}

/// In-place accumulating matrix ops whose dependent same-opcode chain is interlocked by the hardware.
fn chains_in_place(op: MOp) -> bool {
    matches!(op, MOp::VMfmaScale32x32x64Fp8 | MOp::VSmfmac16x16x128Fp8)
}

fn is_scaled_mfma(op: MOp) -> bool {
    matches!(op, MOp::VMfmaScale32x32x64Fp8 | MOp::VMfmaScale32x32x64Fp8Z)
}

/// Plain ALU instructions that may be moved ahead of a stalled consumer: no memory, no scalar
/// state, no `vcc`, no transcendental latency of their own.
fn movable(op: MOp) -> bool {
    matches!(
        op,
        MOp::VMovB32
            | MOp::VAddU32
            | MOp::VMulLoU32
            | MOp::VAndB32
            | MOp::VMinI32
            | MOp::VOrB32
            | MOp::VXorB32
            | MOp::VLshrB32
            | MOp::VLshlB32
            | MOp::VAddF32
            | MOp::VSubF32
            | MOp::VMulF32
            | MOp::VMaxF32
            | MOp::VMax3F32
            | MOp::VPermB32
            | MOp::VPkSubU16
            | MOp::VFmaF32
            | MOp::VCvtPkFp8F32
            | MOp::VCvtPkFp8F32Hi
            | MOp::VPkAddF32
            | MOp::VPkMulF32
            | MOp::VCvtPkBf16F32
    )
}

/// Extra fill candidates when `vcc` filling is on (none touches vcc; partial writers are guarded by the RAW/WAR/WAW check).
fn movable_wide(op: MOp) -> bool {
    matches!(op, MOp::VCvtPknormU16 | MOp::VCvtPkFp8Lo | MOp::VCvtPkFp8Hi | MOp::VCvtF32F16 | MOp::VCvtF16F32 | MOp::VMovFromS)
}

/// VALU instructions whose VGPR result feeding an MFMA operand or a `v_permlane*_swap` needs wait
/// states first (LLVM GCNHazardRecognizer: 2 states; 4 after carry-out adds). Found by the LLVM-tail
/// oracle (docs/llvm-tail-integration.md); the exact count is taken from LLVM, not measured.
fn valu_writer_delay(op: MOp) -> Option<usize> {
    match op {
        MOp::VAddCoU32 | MOp::VAddcCoU32 | MOp::VPtrAdd => Some(4),
        MOp::VExpF32 | MOp::VLogF32 | MOp::VRcpF32 => None,
        o if movable(o) => Some(2),
        // lane write of a fragment (expands to v_mov): the tuple may feed a matrix op next
        // mask compares write an SGPR pair that the cndmask reads: same 2 wait states as the vcc form (conservative)
        MOp::VCmpMaskLtI32 | MOp::VCmpMaskGeI32 | MOp::VCmpMaskGtI32 | MOp::VCmpMaskLtF32 => Some(2),
        MOp::VWriteSub | MOp::VMovFromS | MOp::VMovFromSPair | MOp::VCndmask | MOp::VCvtF32F16 | MOp::VCvtF16F32 | MOp::VCvtPknormU16 => Some(2),
        // audit (safe-asm review): VALU writers that were missing from the class - the scalef32 half-write converts (P words feeding the PV MFMA),
        // the permlane swap, and the register reads that expand to v_mov
        MOp::VCvtPkFp8Lo | MOp::VCvtPkFp8Hi | MOp::VPermlane32Swap | MOp::VReadPhys | MOp::VReadSub => Some(2),
        _ => None,
    }
}

fn reads_valu_hazard_operands(op: MOp) -> bool {
    // v_readfirstlane reading a VGPR the previous VALU instruction wrote needs the same 2 wait states (found on hardware:
    // v_add d; v_readfirstlane s, d; s_mov m0, s; global_load_lds read a stale lane value)
    // SMFMAC: a VALU write of A, B, SRC2 or the accumulator C needs 2 wait states (8 clocks) before the matrix op (CDNA4 ISA 7.6,
    // docs/vendor-spec.md 4.5.2); C is srcs[3] of the MIR op so it is covered like any other source.
    is_scaled_mfma(op) || matches!(op, MOp::VCndmaskM | MOp::VSmfmac16x16x128Fp8 | MOp::VReadFirstLane | MOp::VPermlane32Swap | MOp::VMfma16x16x16F16 | MOp::VMfma32x32x8F16 | MOp::VMfma32x32x8F16Z | MOp::VMfma16x16x32Fp8 | MOp::VMfma32x32x16Fp8)
}

/// Instructions the scanner must never look past (waits, barriers, nops).
fn is_fence(op: MOp) -> bool {
    matches!(op, MOp::SWaitVm | MOp::SWaitLgkm | MOp::SBarrier | MOp::SNop | MOp::SEndpgm)
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
        // strict: VALU write of vcc -> VALU read of vcc needs 2 wait states (LLVM oracle: _cmp -> v_cndmask, _add_co -> v_addc).
        let vccw = std::cell::Cell::new(0usize);
        let mut cycle = 0usize;

        // The earliest cycle at which `inst` may issue given the in-flight results.
        let earliest_of = |inst: &MInst, ready: &HashMap<VReg, (usize, bool)>, vw: &HashMap<VReg, usize>, pkw: &HashMap<VReg, usize>, cycle: usize| -> usize {
            let mut earliest = cycle;
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
                MOp::VMfmaScale32x32x64Fp8 if inst.srcs.len() > 2 => match inst.srcs[2] {
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
            let mut earliest = earliest_of(&insts[i], &ready, &vw, &pkw, cycle);

            // Fill the stall with independent ALU work from further down the block.
            if earliest > cycle && (!insts[i].op.consumes_vcc() || VCC_FILL.with(|c| c.get())) {
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
                        && (movable(cand.op) || (VCC_FILL.with(|c| c.get()) && movable_wide(cand.op)))
                        && cand.dst.is_some()
                        && earliest_of(cand, &ready, &vw, &pkw, cycle) <= cycle;
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
                    earliest = earliest_of(&insts[i], &ready, &vw, &pkw, cycle);
                }
            }

            let inst = insts[i].clone();
            let earliest = earliest_of(&inst, &ready, &vw, &pkw, cycle);
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
            out.push(inst);
            cycle += 1;
            i += 1;
        }
        if pad_block_end {
            // Only results that a successor reads soon need the padding; the in-place accumulation chain
            // of one MFMA opcode never does.
            let mut missing = 0usize;
            for (reg, (rd, from_mfma)) in ready.iter() {
                let remaining = rd.saturating_sub(cycle);
                if remaining == 0 {
                    continue;
                }
                let mut dist = usize::MAX;
                for &s in &succs[bi] {
                    let mut d = 0usize;
                    let mut found = None;
                    for x in &snap[s] {
                        let touches = x.uses().iter().any(|u| canon(*u) == *reg) || x.dst.map(|y| canon(y) == *reg).unwrap_or(false);
                        if touches {
                            let chain = *from_mfma
                                && x.op == MOp::VMfmaScale32x32x64Fp8
                                && matches!(x.srcs.get(2), Some(Src::Reg(c)) if canon(*c) == *reg);
                            found = Some(if chain { usize::MAX } else { d });
                            break;
                        }
                        d += if x.op == MOp::SNop { x.imm as usize + 1 } else { 1 };
                    }
                    dist = dist.min(found.unwrap_or(d));
                }
                if dist != usize::MAX {
                    missing = missing.max(remaining.saturating_sub(dist));
                }
            }
            while missing > 0 {
                let n = missing.min(7) as i64;
                out.push(MInst::new(MOp::SNop).imm(n));
                missing = missing.saturating_sub(n as usize + 1);
            }
        }
        b.insts = out;
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
    let blocks = std::mem::take(&mut m.blocks);
    let mut out_blocks = Vec::with_capacity(blocks.len());
    for mut b in blocks {
        let mut out: Vec<MInst> = Vec::with_capacity(b.insts.len() + 8);
        // (issue cycle, wait states needed, op, source ranges)
        let mut inflight: Vec<(usize, usize, MOp, Vec<(u32, u32)>)> = Vec::new();
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
                        if m.vreg_class(d).is_vector() {
                            wr.push(rng(m, d));
                        }
                    }
                }
            }
            if inst.op == MOp::VMovFromSPair || inst.op == MOp::VPtrAdd {
                // two-instruction expansions write the whole pair (already covered by dst range)
            }
            let mut earliest = cycle;
            for (issue, need, op, srcs) in &inflight {
                if inst.op == *op && matches!(inst.op, MOp::VMfmaScale32x32x64Fp8 | MOp::VSmfmac16x16x128Fp8) {
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
                            if m.vreg_class(*r).is_vector() {
                                srcs.push(rng(m, *r));
                            }
                        }
                    }
                } else if !matches!(inst.op, MOp::VMfmaScale32x32x64Fp8Z | MOp::VMfma32x32x8F16Z) {
                    if let Some(Src::Reg(r)) = inst.srcs.get(2) {
                        if m.vreg_class(*r).is_vector() {
                            srcs.push(rng(m, *r));
                        }
                    }
                }
                inflight.push((cycle, need, inst.op, srcs));
            }
            if matches!(inst.op, MOp::SNop) {
                cycle += inst.imm as usize;
            }
            out.push(inst);
            cycle += 1;
            inflight.retain(|(issue, need, _, _)| issue + need > cycle);
        }
        let mut missing = inflight.iter().map(|(i, n, _, _)| (i + n).saturating_sub(cycle)).max().unwrap_or(0);
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