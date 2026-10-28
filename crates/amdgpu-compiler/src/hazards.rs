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

pub fn fix_value_hazards(m: &mut MFunction) {
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
                    if let Some(rd) = ready.get(r) {
                        earliest = earliest.max(*rd);
                    }
                }
            }
            if earliest > cycle {
                let missing = earliest - cycle;
                let n = missing.saturating_sub(1).min(7) as i64;
                out.push(MInst::new(MOp::SNop).imm(n));
                cycle += (n as usize) + 1;
            }
            if let Some(d) = inst.dst {
                ready.remove(&d);
            }
            if is_transcendental(inst.op) {
                if let Some(d) = inst.dst {
                    ready.insert(d, cycle + TRANS_LATENCY);
                }
            }
            out.push(inst.clone());
            cycle += 1;
        }
        b.insts = out;
    }
}
