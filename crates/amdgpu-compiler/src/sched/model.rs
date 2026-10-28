//! Machine model: per-opcode pipe, issue occupancy and result latency (cycles of one wave on a
//! gfx950 SIMD). This is a DATA table; every row carries an evidence grade. The numbers are
//! documented in docs/scheduling.md together with the microbenchmarks that produced them.
//!
//! Semantics used by the list scheduler (see `list.rs`): a wave issues at most one instruction per
//! cycle, in order; an instruction occupies its pipe for `issue` cycles (the next instruction of
//! the same pipe cannot start earlier); a dependent instruction may start `latency` cycles after
//! its producer started.

use crate::mir::MOp;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Pipe {
    Valu,
    Trans,
    Mfma,
    Lds,
    Vmem,
    Salu,
    Smem,
    Ctrl,
}

pub const NPIPES: usize = 8;

impl Pipe {
    pub fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Grade {
    Verified,
    Experimental,
    Unknown,
}

#[derive(Clone, Copy, Debug)]
pub struct OpCost {
    pub pipe: Pipe,
    pub issue: u32,
    pub latency: u32,
    pub grade: Grade,
}

const fn c(pipe: Pipe, issue: u32, latency: u32, grade: Grade) -> OpCost {
    OpCost { pipe, issue, latency, grade }
}

/// Global model knobs (also data): whether `v_exp`/`v_rcp` run on their own pipe next to the VALU.
pub struct ModelParams {
    pub trans_separate_pipe: bool,
}

pub const PARAMS: ModelParams = ModelParams { trans_separate_pipe: false };

/// Cost of one opcode. Grades: see docs/scheduling.md (tables 1-2).
pub fn cost(op: MOp) -> OpCost {
    use Grade::*;
    use MOp::*;
    use Pipe::*;
    // Numbers: tools/sched_mb/RESULTS.md (s_memtime microbenchmarks on MI355X, 2.39 GHz shader clock);
    // grade V = 3 repetitions within 5%, E = experimental. See docs/scheduling.md.
    match op {
        // VALU: one wave issues one wave64 VALU every 4 cycles, dependent or not (V).
        VMovB32 | VMovFromS | VReadPhys | VAddU32 | VAddF32 | VSubF32 | VMulF32 | VMaxF32 | VMax3F32 | VFmaF32
        | VAndB32 | VOrB32 | VXorB32 | VLshrB32 | VLshlB32 | VCvtPknormU16 | VCvtPkBf16F32 | VCvtF32F16
        | VCvtF16F32 | VCndmask | VReadSub | VWriteSub | VCmpLtI32 | VCmpGeI32 | VCmpNeI32 | VCmpLtF32
        | VAddCoU32 | VAddcCoU32 | VPkAddF32 | VPkMulF32 => c(Valu, 4, 4, Verified),
        // 8 cycles (V): fp8 conversion, permlane32_swap, exp/rcp (no separate trans pipe: 8 + 4k for k adds).
        VCvtPkFp8F32 | VCvtPkFp8Lo | VCvtPkFp8Hi | VCvtPkFp8F32Hi | VPermlane32Swap | VExpF32 | VRcpF32 => {
            c(Valu, 8, 8, Verified)
        }
        VMulLoU32 => c(Valu, 16, 16, Unknown),
        VMovFromSPair | VPtrAdd => c(Valu, 8, 8, Experimental),
        VReadFirstLane => c(Valu, 4, 12, Unknown),
        // MFMA issue spacing 64/32/16 (V), same for dependent chains; latency = wait states until the
        // last D register is readable (20/11/7 s_nop slots of 4 cycles; V) -> hazards pass pads the rest.
        VMfmaScale32x32x64Fp8 | VMfmaScale32x32x64Fp8Z => c(Mfma, 64, 84, Verified),
        VMfma32x32x16Fp8 | VMfma32x32x8F16 => c(Mfma, 32, 48, Verified),
        VMfma16x16x32Fp8 | VMfma16x16x16F16 => c(Mfma, 16, 32, Verified),
        // LDS (E): b128 issue 8, b32 4, tr_b8 ~6, write b128 ~26; latency ~52 (reads) / ~75 (write).
        LdsReadB128 => c(Lds, 8, 52, Experimental),
        LdsReadB32 => c(Lds, 4, 52, Experimental),
        LdsReadB64TrB8 => c(Lds, 6, 52, Experimental),
        LdsWriteB128 => c(Lds, 26, 4, Experimental),
        LdsWriteB32 => c(Lds, 8, 4, Unknown),
        // Global: L1 hit ~100-108, L2 ~210-250 (E); streaming dwordx4 issue 14.
        GlobalLoadU16 | GlobalLoadF32 => c(Vmem, 4, 220, Experimental),
        GlobalLoadB128 => c(Vmem, 14, 240, Experimental),
        GlobalStoreU16 | GlobalStoreF32 => c(Vmem, 8, 4, Unknown),
        SLoadDword | SLoadDwordx2 => c(Smem, 4, 200, Unknown),
        SReadPhys | SMovB32 | SAddU32 | SAddcU32 | SMulI32 | SShlB32 | SCmpLtI32 => c(Salu, 4, 4, Verified),
        SBarrier | SSetPrio | SNop | SWaitVm | SWaitLgkm | SEndpgm => c(Ctrl, 4, 4, Verified),    }
}