//! Value-latency hazards.
//!
//! Most VALU dependencies are interlocked by the hardware, but transcendental
//! results on gfx9 (`v_exp_f32`, `v_rcp_f32`, ...) are not: a consumer issued
//! too early reads the previous value of the register. This pass models that
//! latency and pads with `s_nop` when producer and consumer are too close in
//! program order.
//!
//! It is deliberately conservative: padding costs a few cycles, reading a stale
//! value costs correctness.

use std::collections::HashMap;

use crate::mir::*;

/// Cycles that must separate a transcendental write from its first reader.
const TRANS_LATENCY: usize = 5;

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

pub fn fix_value_hazards(m: &mut MFunction) {
    let alias = m.alias.clone();
    let canon = |r: VReg| alias.get(&r).map(|a| a.0).unwrap_or(r);
    for b in &mut m.blocks {
        let mut out: Vec<MInst> = Vec::with_capacity(b.insts.len() + 8);
        // Virtual register -> earliest cycle at which its transcendental result
        // may be read.
        let mut ready: HashMap<VReg, usize> = HashMap::new();
        let mut cycle = 0usize;

        for inst in &b.insts {
            let mut earliest = cycle;
            for s in &inst.srcs {
                if let Src::Reg(r) = s {
                    if let Some(rd) = ready.get(&canon(*r)) {
                        earliest = earliest.max(*rd);
                    }
                }
            }
            if let Some(d) = inst.dst {
                if let Some(rd) = ready.get(&canon(d)) {
                    earliest = earliest.max(*rd);
                }
            }
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
            }
            if is_transcendental(inst.op) {
                if let Some(d) = inst.dst {
                    ready.insert(canon(d), cycle + TRANS_LATENCY);
                }
            }
            if let Some(latency) = mfma_latency(inst.op) {
                if let Some(d) = inst.dst {
                    ready.insert(canon(d), cycle + latency);
                }
            }
            out.push(inst.clone());
            cycle += 1;
        }
        b.insts = out;
    }
}
