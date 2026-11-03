//! M0 reuse for LDS-DMA loads (`CompileProfile::Gemm`): consecutive DMAs of a basic block that take the same `M0` source register (the LDS base; only the instruction
//! offset differs) need the `s_mov_b32 m0, sX` + the mandatory `s_nop 0` only once. `MInst::skip_m0` marks the later ones and the emitter drops the two lines.
//! The state resets at every block start and at every pinned (hand-written) instruction; a redefinition of the source register in between invalidates it.

use crate::mir::*;

fn m0_src(inst: &MInst) -> Option<VReg> {
    let k = match inst.op {
        MOp::BufferLoadDwordX4Lds | MOp::BufferLoadDwordX1Lds => 3,
        MOp::GlobalLoadLdsB128S | MOp::GlobalLoadLdsB32S => 2,
        MOp::GlobalLoadLdsB128 | MOp::GlobalLoadLdsB32 => 1,
        _ => return None,
    };
    match inst.srcs.get(k) {
        Some(Src::Reg(r)) => Some(*r),
        _ => None,
    }
}

/// Mark the redundant M0 writes; returns how many were saved.
pub fn reuse(m: &mut MFunction) -> usize {
    let alias = m.alias.clone();
    let canon = |r: VReg| alias.get(&r).map(|a| a.0).unwrap_or(r);
    let mut saved = 0;
    for b in &mut m.blocks {
        let mut last: Option<VReg> = None;
        for inst in b.insts.iter_mut() {
            if inst.pin {
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
            if let (Some(d), Some(l)) = (inst.dst, last) {
                if canon(d) == l {
                    last = None;
                }
            }
        }
    }
    saved
}