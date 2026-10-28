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
    matches!(op, MOp::VExpF32 | MOp::VRcpF32)
}

/// Independent instructions that must follow an MFMA before its result may be
/// read (CDNA4 ISA section 7.6, "Table: XDL write VGPR" -> "VALU read/write of
/// the destination VGPR"). The table lists 5/8/12/20 depending on the opcode
/// size; the largest value is used for the F8F6F4 shapes, which take 32 to 64
/// cycles.
fn mfma_latency(op: MOp) -> Option<usize> {
    match op {
        MOp::VMfma16x16x16F16 => Some(12),
        MOp::VMfma32x32x8F16 => Some(12),
        MOp::VMfma16x16x32Fp8 => Some(12),
        MOp::VMfma32x32x16Fp8 => Some(12),
        MOp::VMfmaScale32x32x64Fp8 | MOp::VMfmaScale32x32x64Fp8Z => Some(20),
        _ => None,
    }
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
            | MOp::VOrB32
            | MOp::VXorB32
            | MOp::VLshrB32
            | MOp::VLshlB32
            | MOp::VAddF32
            | MOp::VSubF32
            | MOp::VMulF32
            | MOp::VMaxF32
            | MOp::VMax3F32
            | MOp::VFmaF32
            | MOp::VCvtPkFp8F32
            | MOp::VCvtPkFp8F32Hi
            | MOp::VPkAddF32
            | MOp::VPkMulF32
            | MOp::VCvtPkBf16F32
    )
}

/// VALU instructions whose VGPR result feeding an MFMA operand or a `v_permlane*_swap` needs wait
/// states first (LLVM GCNHazardRecognizer: 2 states; 4 after carry-out adds). Found by the LLVM-tail
/// oracle (docs/llvm-tail-integration.md); the exact count is taken from LLVM, not measured.
fn valu_writer_delay(op: MOp) -> Option<usize> {
    match op {
        MOp::VAddCoU32 | MOp::VAddcCoU32 | MOp::VPtrAdd => Some(4),
        MOp::VExpF32 | MOp::VRcpF32 => None,
        o if movable(o) => Some(2),
        MOp::VMovFromS | MOp::VMovFromSPair | MOp::VCndmask | MOp::VCvtF32F16 | MOp::VCvtF16F32 | MOp::VCvtPknormU16 => Some(2),
        _ => None,
    }
}

fn reads_valu_hazard_operands(op: MOp) -> bool {
    is_scaled_mfma(op) || matches!(op, MOp::VPermlane32Swap | MOp::VMfma16x16x16F16 | MOp::VMfma32x32x8F16 | MOp::VMfma16x16x32Fp8 | MOp::VMfma32x32x16Fp8)
}

/// Instructions the scanner must never look past (waits, barriers, nops).
fn is_fence(op: MOp) -> bool {
    matches!(op, MOp::SWaitVm | MOp::SWaitLgkm | MOp::SBarrier | MOp::SNop | MOp::SEndpgm)
}

pub fn fix_value_hazards(m: &mut MFunction) {
    let alias = m.alias.clone();
    let canon = |r: VReg| alias.get(&r).map(|a| a.0).unwrap_or(r);
    for b in &mut m.blocks {
        let mut insts: Vec<MInst> = std::mem::take(&mut b.insts);
        let mut out: Vec<MInst> = Vec::with_capacity(insts.len() + 8);
        // Virtual register -> (earliest cycle at which its in-flight result may be read, whether
        // the producer is a scaled MFMA).
        let mut ready: HashMap<VReg, (usize, bool)> = HashMap::new();
        // Virtual register -> earliest cycle at which a plain VALU result may feed an MFMA operand
        // or a permlane (a different, shorter hazard than the in-flight latencies above).
        let mut vw: HashMap<VReg, usize> = HashMap::new();
        let mut cycle = 0usize;

        // The earliest cycle at which `inst` may issue given the in-flight results.
        let earliest_of = |inst: &MInst, ready: &HashMap<VReg, (usize, bool)>, vw: &HashMap<VReg, usize>, cycle: usize| -> usize {
            let mut earliest = cycle;
            if reads_valu_hazard_operands(inst.op) {
                for s in &inst.srcs {
                    if let Src::Reg(r) = s {
                        if let Some(rd) = vw.get(&canon(*r)) {
                            earliest = earliest.max(*rd);
                        }
                    }
                }
            }
            let chained = is_scaled_mfma(inst.op);
            let c_reg = if chained && inst.srcs.len() > 2 && inst.op == MOp::VMfmaScale32x32x64Fp8 {
                match inst.srcs[2] {
                    Src::Reg(r) => Some(canon(r)),
                    _ => None,
                }
            } else {
                None
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
            let mut earliest = earliest_of(&insts[i], &ready, &vw, cycle);

            // Fill the stall with independent ALU work from further down the block.
            if earliest > cycle && !insts[i].op.consumes_vcc() {
                let mut j = i + 1;
                let limit = (i + 1 + FILL_WINDOW).min(insts.len());
                while earliest > cycle && j < limit.min(insts.len()) {
                    if is_fence(insts[j].op) {
                        break;
                    }
                    let cand = &insts[j];
                    let mut ok = movable(cand.op)
                        && cand.dst.is_some()
                        && earliest_of(cand, &ready, &vw, cycle) <= cycle;
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
                    earliest = earliest_of(&insts[i], &ready, &vw, cycle);
                }
            }

            let inst = insts[i].clone();
            let earliest = earliest_of(&inst, &ready, &vw, cycle);
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
            }
            if is_transcendental(inst.op) {
                if let Some(d) = inst.dst {
                    ready.insert(canon(d), (cycle + TRANS_LATENCY, false));
                }
            }
            if let Some(latency) = mfma_latency(inst.op) {
                if let Some(d) = inst.dst {
                    ready.insert(canon(d), (cycle + latency, is_scaled_mfma(inst.op)));
                }
            }
            out.push(inst);
            cycle += 1;
            i += 1;
        }
        b.insts = out;
    }
}