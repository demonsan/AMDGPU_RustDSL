//! Migration check (op attribute table): the attribute-driven predicates, cost model and effects are equal, for every MOp, to the
//! per-pass implementations they replaced (verbatim copies below). Kept as a regression test of the table.
#![allow(dead_code, unused_imports, clippy::all)]
use crate::mir::*;
use crate::sched::effects::*;
use crate::sched::model::{Grade, OpCost, Pipe};
// ---- verbatim copies of the replaced predicates
/// Cost of one opcode. Grades: see docs/scheduling.md (tables 1-2).
pub fn cost_legacy(op: MOp) -> OpCost {
    use crate::sched::model::{c, co};
    use Grade::*;
    use MOp::*;
    use Pipe::*;
    // Numbers: tools/sched_mb/RESULTS.md (s_memtime microbenchmarks on MI355X, 2.39 GHz shader clock);
    // grade V = 3 repetitions within 5%, E = experimental. See docs/scheduling.md.
    match op {
        // VALU: one wave issues one wave64 VALU every 4 cycles, dependent or not (V).
        VMovB32 | VAddF32 | VSubF32 | VMulF32 | VFmaF32 => co(Valu, 4, 2, 4, Verified),
        VMovFromS | VReadPhys | VAddU32 | VMaxF32 | VMax3F32 | VPermB32 | VPkSubU16
        | VAndB32 | VMinI32 | VOrB32 | VXorB32 | VLshrB32 | VLshlB32 | VCvtPknormU16 | VCvtPkBf16F32 | VCvtF32F16
        | VCvtF16F32 | VCndmask | VReadSub | VWriteSub | VCmpLtI32 | VCmpGeI32 | VCmpGtI32 | VCmpNeI32 | VCmpLtF32 | VCmpMaskLtI32 | VCmpMaskGeI32 | VCmpMaskGtI32 | VCmpMaskLtF32 | VCndmaskM
        | VAddCoU32 | VAddcCoU32 | VPkAddF32 | VPkMulF32 => c(Valu, 4, 4, Verified),
        // 8 cycles (V): fp8 conversion, permlane32_swap, exp/rcp (no separate trans pipe: 8 + 4k for k adds).
        VCvtPkFp8F32 | VCvtPkFp8Lo | VCvtPkFp8Hi | VCvtPkFp8F32Hi | VPermlane32Swap | VExpF32 | VLogF32 | VRcpF32 => {
            c(Valu, 8, 8, Verified)
        }
        // accumulation-register moves are ordinary VALU issue slots (Unknown: not microbenchmarked yet)
        VAccvgprRead | VAccvgprWrite | VAccvgprMov => c(Valu, 4, 8, Unknown),
        VMulLoU32 => c(Valu, 16, 16, Unknown),
        VMovFromSPair | VPtrAdd => c(Valu, 8, 8, Experimental),
        VReadFirstLane => c(Valu, 4, 12, Unknown),
        VMovDpp => c(Valu, 4, 8, Unknown),
        // MFMA issue spacing 64/32/16 (V), same for dependent chains; latency = wait states until the
        // last D register is readable (20/11/7 s_nop slots of 4 cycles; V) -> hazards pass pads the rest.
        VMfmaScale32x32x64Fp8 | VMfmaScale32x32x64Fp8Z | VMfma32x32x64Fp8 | VMfma32x32x64Fp8Z => c(Mfma, 64, 84, Verified),
        VMfma32x32x16Fp8 | VMfma32x32x8F16 | VMfma32x32x8F16Z => c(Mfma, 32, 48, Verified),
        VMfma16x16x32Fp8 | VMfma16x16x16F16 => c(Mfma, 16, 32, Verified),
        // Sparse fp8 16x16x128: arch table says 32 cycles (MI350X, F8 involved); latency not measured.
        VSmfmac16x16x128Fp8 => c(Mfma, 32, 32, Experimental),
        // scaled 16x16x128 f8f6f4: 8 passes (32 cycles issue); latency from the 16-pass table until measured
        VMfmaScale16x16x128Fp8 => c(Mfma, 32, 48, Experimental),
        // LDS (E): b128 issue 8, b32 4, tr_b8 ~6, write b128 ~26; latency ~52 (reads) / ~75 (write).
        LdsReadB128 => c(Lds, 8, 52, Experimental),
        LdsReadB64 => c(Lds, 4, 48, Experimental),
        LdsWriteB64 => c(Lds, 8, 4, Experimental),
        LdsReadB32 => c(Lds, 4, 52, Experimental),
        LdsReadB64TrB8 => c(Lds, 6, 52, Experimental),
        LdsWriteB128 => c(Lds, 26, 4, Experimental),
        LdsWriteB32 => c(Lds, 8, 4, Unknown),
        // Global: L1 hit ~100-108, L2 ~210-250 (E); streaming dwordx4 issue 14.
        GlobalLoadU16 | GlobalLoadF32 | GlobalLoadF32S | BufferLoadDwordX1 => c(Vmem, 4, 220, Experimental),
        GlobalLoadB128 | GlobalLoadB128S | BufferLoadDwordX2 | BufferLoadDwordX4 => c(Vmem, 14, 240, Experimental),
        GlobalLoadF32Sc1 => c(Vmem, 4, 900, Unknown),
        GlobalLoadB128Sc1 => c(Vmem, 14, 900, Unknown),
        GlobalStoreF32Sc1 | GlobalStoreF32x4Sc1 => c(Vmem, 8, 4, Unknown),
        GlobalAtomicAddRet => c(Vmem, 8, 900, Unknown),
        BufferWbl2 | BufferInv => c(Ctrl, 4, 4, Unknown),
        GlobalLoadLdsB128 | GlobalLoadLdsB32 | GlobalLoadLdsB128S | GlobalLoadLdsB32S => c(Vmem, 14, 240, Unknown),
        GlobalStoreU16 | GlobalStoreF32 | GlobalStoreF32x4 | GlobalStoreF32x2 | BufferStoreDwordX1 | BufferStoreDwordX2 | BufferStoreDwordX4 => c(Vmem, 8, 4, Unknown),
        BufferLoadDwordX4Lds | BufferLoadDwordX1Lds => c(Vmem, 14, 240, Unknown),
        SBufferRsrc | SBufferRsrcDyn => c(Salu, 4, 4, Unknown),
        SLoadDword | SLoadDwordx2 | SLoadDwordx4 | SMemTime => c(Smem, 4, 200, Unknown),
        SLoPart => c(Salu, 4, 4, Verified),
        SReadPhys | SMovB32 | SAddU32 | SAddcU32 | SMulI32 | SShlB32 | SLshrB32 | SAndB32 | SXorB32 | SCmpLtI32 | SMinI32 | SCselectB32 | SCselectB64 | SCselectB128 => c(Salu, 4, 4, Verified),
        SBarrier | SSetPrio | SNop | SWaitVm | SWaitLgkm | SEndpgm | SchedGroup => c(Ctrl, 4, 4, Verified),    }
}
fn consumes_vcc_legacy(op: MOp) -> bool {
    matches!(op, MOp::VAddcCoU32 | MOp::VCndmask)
}
fn defines_vcc_legacy(op: MOp) -> bool {
    matches!(op, MOp::VAddCoU32 | MOp::VAddcCoU32 | MOp::VCmpLtI32 | MOp::VCmpGeI32 | MOp::VCmpGtI32 | MOp::VCmpNeI32 | MOp::VCmpLtF32)
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
        MOp::VMfmaScale32x32x64Fp8 | MOp::VMfmaScale32x32x64Fp8Z | MOp::VMfma32x32x64Fp8 | MOp::VMfma32x32x64Fp8Z => Some(20),
        MOp::VSmfmac16x16x128Fp8 => Some(20),
        // 8-pass XDL op: conservatively the 16-pass wait count until the gfx950 table entry is pinned
        MOp::VMfmaScale16x16x128Fp8 => Some(20),
        _ => None,
    }
}

/// In-place accumulating matrix ops whose dependent same-opcode chain is interlocked by the hardware.
fn chains_in_place(op: MOp) -> bool {
    matches!(op, MOp::VMfmaScale32x32x64Fp8 | MOp::VMfma32x32x64Fp8 | MOp::VSmfmac16x16x128Fp8 | MOp::VMfmaScale16x16x128Fp8)
}

fn is_scaled_mfma(op: MOp) -> bool {
    matches!(op, MOp::VMfmaScale32x32x64Fp8 | MOp::VMfmaScale32x32x64Fp8Z | MOp::VMfma32x32x64Fp8 | MOp::VMfma32x32x64Fp8Z | MOp::VMfmaScale16x16x128Fp8)
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
        // v_accvgpr_read/write/mov are VALU writes of a VGPR/AGPR: VALU write -> MFMA read = 2 wait states (CDNA4 ISA 7.6 table 38; the ISA table does not separate the files)
        MOp::VAccvgprRead | MOp::VAccvgprWrite | MOp::VAccvgprMov => Some(2),
        // a DPP move is a VALU write like any other
        MOp::VMovDpp => Some(2),
        _ => None,
    }
}

fn reads_valu_hazard_operands(op: MOp) -> bool {
    // v_readfirstlane reading a VGPR the previous VALU instruction wrote needs the same 2 wait states (found on hardware:
    // v_add d; v_readfirstlane s, d; s_mov m0, s; global_load_lds read a stale lane value)
    // SMFMAC: a VALU write of A, B, SRC2 or the accumulator C needs 2 wait states (8 clocks) before the matrix op (CDNA4 ISA 7.6,
    // docs/vendor-spec.md 4.5.2); C is srcs[3] of the MIR op so it is covered like any other source.
    // DPP: a VALU write of a VGPR read by a DPP instruction needs 2 wait states (gfx9 `VALUWriteVGPR -> DPP read`, LLVM checkDPPHazards)
    is_scaled_mfma(op) || matches!(op, MOp::VCndmaskM | MOp::VSmfmac16x16x128Fp8 | MOp::VReadFirstLane | MOp::VMovDpp | MOp::VPermlane32Swap | MOp::VMfma16x16x16F16 | MOp::VMfma32x32x8F16 | MOp::VMfma32x32x8F16Z | MOp::VMfma16x16x32Fp8 | MOp::VMfma32x32x16Fp8)
}


fn is_wide_vmem_store(op: MOp) -> bool {
    matches!(op, MOp::GlobalStoreF32x4 | MOp::GlobalStoreF32x4Sc1 | MOp::BufferStoreDwordX4)
}

}

fn wide_store_data_idx(op: MOp) -> usize {
    if op == MOp::BufferStoreDwordX4 {
        0
    } else {
        1
    }
}

}

pub fn legal_src_legacy(op: MOp, k: usize) -> bool {
    use MOp::*;
    match op {
        LdsWriteB32 | LdsWriteB128 => k == 1,
        GlobalStoreU16 | GlobalStoreF32 | GlobalStoreF32x4 | GlobalStoreF32x2 | GlobalStoreF32Sc1 | GlobalStoreF32x4Sc1 => k == 1,
        BufferStoreDwordX1 | BufferStoreDwordX2 | BufferStoreDwordX4 => k == 0,
        VMfma16x16x16F16 | VMfma32x32x8F16 | VMfma16x16x32Fp8 | VMfma32x32x16Fp8 | VMfmaScale32x32x64Fp8 | VMfma32x32x64Fp8 | VMfmaScale16x16x128Fp8 => k <= 2,
        VMfma32x32x8F16Z | VMfmaScale32x32x64Fp8Z | VMfma32x32x64Fp8Z => k <= 1,
        VAccvgprRead | VAccvgprMov => k == 0,
        _ => false,
    }
}

pub fn is_mfma_legacy(op: MOp) -> bool {
    use MOp::*;
    matches!(
        op,
        VMfma16x16x16F16 | VMfma32x32x8F16 | VMfma32x32x8F16Z | VMfma16x16x32Fp8 | VMfma32x32x16Fp8 | VMfmaScale32x32x64Fp8 | VMfmaScale32x32x64Fp8Z | VMfma32x32x64Fp8 | VMfma32x32x64Fp8Z | VMfmaScale16x16x128Fp8
    )
}

pub fn legal_dst_legacy(op: MOp) -> bool {
    use MOp::*;
    is_mfma_legacy(op)
        || matches!(
            op,
            LdsReadB32
                | LdsReadB128
                | LdsReadB64TrB8
                | GlobalLoadU16
                | GlobalLoadF32
                | GlobalLoadB128
                | GlobalLoadB128S
                | GlobalLoadF32S
                | GlobalLoadF32Sc1
                | GlobalLoadB128Sc1
                | BufferLoadDwordX1
                | BufferLoadDwordX2
                | BufferLoadDwordX4
                | VAccvgprWrite
                | VAccvgprMov
        )
}


fn rmw_dst_legacy(op: MOp) -> bool {
    matches!(op, MOp::VCvtPkFp8F32 | MOp::VCvtPkFp8Lo | MOp::VCvtPkFp8Hi | MOp::VCvtPkFp8F32Hi | MOp::VPermlane32Swap)
}

}

fn dce_class_legacy(op: MOp) -> (bool, bool) {
    fn load_op(op: MOp) -> bool {
        matches!(
            op,
            MOp::SLoadDword | MOp::SLoadDwordx2 | MOp::SLoadDwordx4 | MOp::LdsReadB32 | MOp::LdsReadB128 | MOp::LdsReadB64 | MOp::LdsReadB64TrB8 | MOp::GlobalLoadU16
                | MOp::GlobalLoadF32 | MOp::GlobalLoadB128 | MOp::GlobalLoadB128S | MOp::GlobalLoadF32S
        )
    }
    let removable = |op: MOp| -> bool {
        if load_op(op) {
            return true;
        }
        
        matches!(
            op,
            MOp::VMovB32 | MOp::VMovFromS | MOp::VMovFromSPair | MOp::VPtrAdd | MOp::VAddU32 | MOp::VAddCoU32 | MOp::VAddcCoU32
                | MOp::VAddF32 | MOp::VSubF32 | MOp::VMulF32 | MOp::VMaxF32 | MOp::VMax3F32 | MOp::VFmaF32 | MOp::VExpF32 | MOp::VLogF32
                | MOp::VRcpF32 | MOp::VAndB32 | MOp::VMinI32 | MOp::VOrB32 | MOp::VXorB32 | MOp::VLshrB32 | MOp::VLshlB32 | MOp::VMulLoU32
                | MOp::VCndmask | MOp::VCvtF32F16 | MOp::VCvtF16F32 | MOp::VReadPhys | MOp::VCmpLtI32 | MOp::VCmpGtI32 | MOp::VCmpGeI32
                | MOp::VCmpNeI32 | MOp::VCmpLtF32
        )
    };

    (load_op(op), removable(op))
}

fn own_only_legacy(op: MOp, rows: bool) -> bool {
    use crate::mir;
    let i = crate::mir::MInst::new(op);
    matches!(i.op, mir::MOp::GlobalLoadLdsB128 | mir::MOp::GlobalLoadLdsB32 | mir::MOp::GlobalLoadLdsB128S | mir::MOp::GlobalLoadLdsB32S | mir::MOp::BufferLoadDwordX4Lds | mir::MOp::BufferLoadDwordX1Lds | mir::MOp::SMinI32 | mir::MOp::SCselectB32 | mir::MOp::SCselectB64 | mir::MOp::SCselectB128 | mir::MOp::VMovDpp | mir::MOp::SLoadDwordx4) || (!rows && matches!(i.op, mir::MOp::BufferWbl2 | mir::MOp::BufferInv | mir::MOp::GlobalAtomicAddRet))
}

fn push_unique(v: &mut Vec<Slot>, s: Slot) {
    if !v.contains(&s) {
        v.push(s);
    }
}
// ---- verbatim copy of the replaced effects
pub fn effects_core_legacy(m: &MFunction, inst: &MInst) -> Effects {
    let mut e = Effects::default();
    match inst.op {
        MOp::SNop | MOp::SEndpgm => {
            e.fence = Fence::Nop;
            return e;
        }
        MOp::SWaitVm => {
            e.fence = Fence::WaitVm(inst.imm);
            return e;
        }
        MOp::SWaitLgkm => {
            e.fence = Fence::WaitLgkm(inst.imm);
            return e;
        }
        MOp::SBarrier => {
            e.fence = Fence::Barrier;
            return e;
        }
        MOp::SSetPrio => {
            e.fence = Fence::SetPrio;
            return e;
        }
        _ => {}
    }
    for s in &inst.srcs {
        if let Src::Reg(r) = s {
            for x in reg_slots(m, *r) {
                push_unique(&mut e.reads, x);
            }
        }
    }
    if let Some(d) = inst.dst {
        for x in reg_slots(m, d) {
            push_unique(&mut e.writes, x);
        }
    }
    let src_reg = |k: usize| -> Option<VReg> {
        match inst.srcs.get(k) {
            Some(Src::Reg(r)) => Some(*r),
            _ => None,
        }
    };
    let base_slot = |k: usize| -> Option<Slot> { src_reg(k).map(|r| reg_slots(m, r)[0]) };
    match inst.op {
        MOp::VWriteSub => {
            if let Some(d) = inst.dst {
                let sl = reg_slots(m, d);
                e.writes = vec![sl[(inst.imm as usize).min(sl.len() - 1)]];
            }
        }
        MOp::VReadSub => {
            if let Some(r) = src_reg(0) {
                let sl = reg_slots(m, r);
                e.reads = vec![sl[(inst.imm as usize).min(sl.len() - 1)]];
            }
        }
        // Partial (16-bit) writers: the other half of the destination survives, so the old value is read.
        MOp::VCvtPkFp8F32 | MOp::VCvtPkFp8Lo | MOp::VCvtPkFp8Hi | MOp::VCvtPkFp8F32Hi => {
            if let Some(d) = inst.dst {
                for x in reg_slots(m, d) {
                    push_unique(&mut e.reads, x);
                }
            }
        }
        MOp::VPermlane32Swap => {
            for k in 0..2 {
                if let Some(r) = src_reg(k) {
                    for x in reg_slots(m, r) {
                        push_unique(&mut e.writes, x);
                    }
                }
            }
        }
        MOp::VPtrAdd | MOp::VAddCoU32 => push_unique(&mut e.writes, VCC),
        MOp::VAddcCoU32 => {
            push_unique(&mut e.reads, VCC);
            push_unique(&mut e.writes, VCC);
        }
        MOp::VCmpLtI32 | MOp::VCmpGeI32 | MOp::VCmpGtI32 | MOp::VCmpNeI32 | MOp::VCmpLtF32 => push_unique(&mut e.writes, VCC),
        MOp::VCndmask => push_unique(&mut e.reads, VCC),
        MOp::SAddU32 | MOp::SShlB32 | MOp::SLshrB32 | MOp::SAndB32 | MOp::SXorB32 | MOp::SCmpLtI32 | MOp::SMinI32 => push_unique(&mut e.writes, SCC),
        MOp::SCselectB32 | MOp::SCselectB64 | MOp::SCselectB128 => push_unique(&mut e.reads, SCC),
        MOp::SAddcU32 => {
            push_unique(&mut e.reads, SCC);
            push_unique(&mut e.writes, SCC);
        }
        MOp::LdsReadB32 => {
            e.mem = Some(MemAccess { space: Space::Lds, write: false, base: base_slot(0), off: inst.imm, size: 4 })
        }
        MOp::LdsReadB128 => {
            e.mem = Some(MemAccess { space: Space::Lds, write: false, base: base_slot(0), off: inst.imm, size: 16 })
        }
        MOp::LdsReadB64 => {
            e.mem = Some(MemAccess { space: Space::Lds, write: false, base: base_slot(0), off: inst.imm, size: 8 })
        }
        MOp::LdsWriteB64 => {
            e.mem = Some(MemAccess { space: Space::Lds, write: true, base: base_slot(0), off: inst.imm, size: 8 })
        }
        MOp::LdsReadB64TrB8 => {
            e.mem = Some(MemAccess { space: Space::Lds, write: false, base: base_slot(0), off: inst.imm, size: 8 })
        }
        MOp::LdsWriteB32 => {
            e.mem = Some(MemAccess { space: Space::Lds, write: true, base: base_slot(0), off: inst.imm, size: 4 })
        }
        MOp::LdsWriteB128 => {
            e.mem = Some(MemAccess { space: Space::Lds, write: true, base: base_slot(0), off: inst.imm, size: 16 })
        }
        MOp::GlobalLoadU16 => {
            e.mem = Some(MemAccess { space: Space::Global, write: false, base: base_slot(0), off: inst.imm, size: 2 })
        }
        MOp::GlobalLoadF32Sc1 => {
            e.mem = Some(MemAccess { space: Space::Global, write: false, base: base_slot(0), off: inst.imm, size: 4 })
        }
        MOp::GlobalLoadB128Sc1 => {
            e.mem = Some(MemAccess { space: Space::Global, write: false, base: base_slot(0), off: inst.imm, size: 16 })
        }
        MOp::GlobalStoreF32Sc1 => {
            e.mem = Some(MemAccess { space: Space::Global, write: true, base: base_slot(0), off: inst.imm, size: 4 })
        }
        MOp::GlobalStoreF32x4Sc1 => {
            e.mem = Some(MemAccess { space: Space::Global, write: true, base: base_slot(0), off: inst.imm, size: 16 })
        }
        // read-modify-write with a result: a global access (vmcnt queue entry whose destination is the old value)
        MOp::GlobalAtomicAddRet => {
            e.mem = Some(MemAccess { space: Space::Global, write: true, base: base_slot(0), off: inst.imm, size: 4 })
        }
        // cache maintenance: ordered against every memory access, never reordered, no register effects
        MOp::BufferWbl2 | MOp::BufferInv => {
            e.fence = Fence::Barrier;
            return e;
        }
        MOp::GlobalLoadF32 => {
            e.mem = Some(MemAccess { space: Space::Global, write: false, base: base_slot(0), off: inst.imm, size: 4 })
        }
        MOp::GlobalLoadB128S => {
            e.mem = Some(MemAccess { space: Space::Global, write: false, base: base_slot(1), off: inst.imm, size: 16 })
        }
        MOp::GlobalLoadF32S => {
            e.mem = Some(MemAccess { space: Space::Global, write: false, base: base_slot(1), off: inst.imm, size: 4 })
        }
        MOp::GlobalLoadB128 => {
            e.mem = Some(MemAccess { space: Space::Global, write: false, base: base_slot(0), off: inst.imm, size: 16 })
        }
        // MUBUF: the per-lane address is not an affine function of one base register, so the access is modelled with an unknown base (always may-alias)
        MOp::BufferLoadDwordX1 => e.mem = Some(MemAccess { space: Space::Global, write: false, base: None, off: inst.imm, size: 4 }),
        MOp::BufferLoadDwordX2 => e.mem = Some(MemAccess { space: Space::Global, write: false, base: None, off: inst.imm, size: 8 }),
        MOp::BufferLoadDwordX4 => e.mem = Some(MemAccess { space: Space::Global, write: false, base: None, off: inst.imm, size: 16 }),
        MOp::BufferStoreDwordX1 => e.mem = Some(MemAccess { space: Space::Global, write: true, base: None, off: inst.imm, size: 4 }),
        MOp::BufferStoreDwordX2 => e.mem = Some(MemAccess { space: Space::Global, write: true, base: None, off: inst.imm, size: 8 }),
        MOp::BufferStoreDwordX4 => e.mem = Some(MemAccess { space: Space::Global, write: true, base: None, off: inst.imm, size: 16 }),
        MOp::BufferLoadDwordX4Lds | MOp::BufferLoadDwordX1Lds => {
            // reads global memory and writes LDS through M0 (no VGPR destination): same model as the global_load_lds DMA
            e.mem = Some(MemAccess { space: Space::Global, write: false, base: None, off: inst.imm, size: 16 });
            e.fence = Fence::Barrier;
        }
        MOp::GlobalLoadLdsB128S | MOp::GlobalLoadLdsB32S => {
            e.mem = Some(MemAccess { space: Space::Global, write: false, base: base_slot(1), off: inst.imm, size: 16 });
            e.fence = Fence::Barrier;
        }
        MOp::GlobalLoadLdsB128 | MOp::GlobalLoadLdsB32 => {
            // reads global memory and writes LDS through M0: modelled as a global read with a fence-like LDS write
            e.mem = Some(MemAccess { space: Space::Global, write: false, base: base_slot(0), off: inst.imm, size: 16 });
            e.fence = Fence::Barrier;
        }
        MOp::GlobalStoreU16 => {
            e.mem = Some(MemAccess { space: Space::Global, write: true, base: base_slot(0), off: inst.imm, size: 2 })
        }
        MOp::GlobalStoreF32 => {
            e.mem = Some(MemAccess { space: Space::Global, write: true, base: base_slot(0), off: inst.imm, size: 4 })
        }
        MOp::GlobalStoreF32x2 => {
            e.mem = Some(MemAccess { space: Space::Global, write: true, base: base_slot(0), off: inst.imm, size: 8 })
        }
        MOp::GlobalStoreF32x4 => {
            e.mem = Some(MemAccess { space: Space::Global, write: true, base: base_slot(0), off: inst.imm, size: 16 })
        }
        MOp::SLoadDword | MOp::SLoadDwordx2 | MOp::SLoadDwordx4 => e.smem = true,
        // a timing stamp: SMEM result, and pinned in program order against every other memory op so it measures what it brackets
        MOp::SMemTime => {
            e.smem = true;
            e.fence = Fence::Barrier;
        }
        _ => {}
    }
    e
}


fn all_ops() -> Vec<MOp> {
    crate::backend::ALL_MOPS.to_vec()
}

#[test]
fn every_variant_is_listed() {
    // ALL_MOPS must list every variant of the enum (the attribute match is exhaustive at compile time)
    let src = include_str!("mir.rs");
    let body = &src[src.find("pub enum MOp {").unwrap()..];
    let body = &body[..body.find("\n}").unwrap()];
    let names: Vec<&str> = body.lines().map(|l| l.trim()).filter(|l| l.ends_with(',') && l.chars().next().map_or(false, |c| c.is_ascii_uppercase())).map(|l| l.trim_end_matches(',')).collect();
    let listed: Vec<String> = all_ops().iter().map(|o| format!("{o:?}")).collect();
    for n in &names {
        assert!(listed.iter().any(|l| l == n), "MOp::{n} is missing from backend::ALL_MOPS");
    }
    assert_eq!(names.len(), listed.len(), "ALL_MOPS has duplicates or unknown entries");
}

#[test]
fn attributes_equal_the_replaced_predicates() {
    for op in all_ops() {
        let a = crate::op_attrs::attrs(op);
        let (l, n) = (cost_legacy(op), a.cost);
        assert_eq!(format!("{l:?}"), format!("{n:?}"), "{op:?} cost");
        assert_eq!(consumes_vcc_legacy(op), op.consumes_vcc(), "{op:?} consumes_vcc");
        assert_eq!(defines_vcc_legacy(op), op.defines_vcc(), "{op:?} defines_vcc");
        assert_eq!(is_transcendental(op), a.transcendental, "{op:?} trans");
        assert_eq!(mfma_latency(op), a.mfma.map(|m| m.latency), "{op:?} mfma latency");
        assert_eq!(chains_in_place(op), a.mfma.map_or(false, |m| m.chain), "{op:?} chain");
        assert_eq!(is_scaled_mfma(op), a.mfma.map_or(false, |m| m.scaled), "{op:?} scaled");
        assert_eq!(movable(op), a.movable == crate::op_attrs::Movable::Yes, "{op:?} movable");
        assert_eq!(movable_wide(op), a.movable == crate::op_attrs::Movable::VccFill, "{op:?} movable_wide");
        assert_eq!(valu_writer_delay(op), a.valu_writer_delay, "{op:?} valu_writer_delay");
        assert_eq!(reads_valu_hazard_operands(op), a.reads_valu_hazard, "{op:?} reads_valu_hazard");
        assert_eq!(is_wide_vmem_store(op), a.wide_store_data.is_some(), "{op:?} wide store");
        if is_wide_vmem_store(op) {
            assert_eq!(wide_store_data_idx(op), a.wide_store_data.unwrap(), "{op:?} wide store data");
        }
        for k in 0..6 {
            assert_eq!(legal_src_legacy(op, k), crate::agpr::legal_src(op, k), "{op:?} agpr src {k}");
        }
        assert_eq!(is_mfma_legacy(op), crate::agpr::is_mfma(op), "{op:?} is_mfma");
        assert_eq!(legal_dst_legacy(op), crate::agpr::legal_dst(op), "{op:?} agpr dst");
        assert_eq!(rmw_dst_legacy(op), matches!(a.regfx, crate::op_attrs::RegFx::PartialWrite | crate::op_attrs::RegFx::SwapSrc01), "{op:?} rmw");
        let (ld, rm) = dce_class_legacy(op);
        assert_eq!(ld, a.dce == crate::op_attrs::Dce::Load, "{op:?} dce load");
        assert_eq!(rm, a.dce != crate::op_attrs::Dce::Keep, "{op:?} dce removable");
        for rows in [true, false] {
            let new = match a.backend {
                crate::op_attrs::Backend::OwnOnly => true,
                crate::op_attrs::Backend::OwnOnlyIfNoRows => !rows,
                _ => false,
            };
            assert_eq!(own_only_legacy(op, rows), new, "{op:?} own-only (rows {rows})");
        }
    }
}

#[test]
fn effects_equal_the_replaced_implementation() {
    // registers 0..7 of several classes; every op with up to 4 register sources, a destination and an immediate
    let classes = [RClass::V4, RClass::V2, RClass::V1, RClass::S1, RClass::S2, RClass::V8, RClass::S4, RClass::V1];
    let m = MFunction {
        name: "e".into(),
        args: Vec::new(),
        kernarg_size: 0,
        workgroup_size: 64,
        lds_bytes: 0,
        blocks: Vec::new(),
        vregs: classes.to_vec(),
        loops: Vec::new(),
        alias: Default::default(),
        elide_fallthrough: false,
        edge_pad: false,
        agpr_pin: Default::default(),
    };
    for op in all_ops() {
        for (dst, nsrc, imm) in [(Some(0u32), 4usize, 3i64), (None, 2, 0), (Some(5), 1, 1), (Some(2), 0, 7)] {
            let mut i = MInst::new(op).imm(imm);
            i.dst = dst;
            for k in 0..nsrc {
                i = i.srcreg((k as u32 + 1) % 8);
            }
            let l = effects_core_legacy(&m, &i);
            let n = effects(&m, &i);
            assert_eq!(format!("{:?} {:?} {:?} {:?} {}", l.reads, l.writes, l.mem, l.fence, l.smem), format!("{:?} {:?} {:?} {:?} {}", n.reads, n.writes, n.mem, n.fence, n.smem), "{op:?} effects");
        }
    }
}