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
    let trans = if PARAMS.trans_separate_pipe { Trans } else { Valu };
    match op {
        // VALU: wave64 on a SIMD16-class pipe -> 4 cycles per instruction.
        VMovB32 | VMovFromS | VReadPhys | VAddU32 | VAddF32 | VSubF32 | VMulF32 | VMaxF32 | VMax3F32 | VFmaF32
        | VAndB32 | VOrB32 | VXorB32 | VLshrB32 | VLshlB32 | VCvtPknormU16 | VCvtPkFp8F32 | VCvtPkBf16F32
        | VCvtPkFp8Lo | VCvtPkFp8Hi | VCvtPkFp8F32Hi | VCvtF32F16 | VCvtF16F32 | VCndmask | VReadSub | VWriteSub
        | VCmpLtI32 | VCmpGeI32 | VCmpNeI32 | VCmpLtF32 | VAddCoU32 | VAddcCoU32 => c(Valu, 4, 8, Experimental),
        VMulLoU32 => c(Valu, 16, 20, Unknown),
        VMovFromSPair | VPtrAdd => c(Valu, 8, 12, Experimental),
        VPkAddF32 | VPkMulF32 => c(Valu, 8, 12, Unknown),
        VPermlane32Swap => c(Valu, 8, 12, Unknown),
        VReadFirstLane => c(Valu, 4, 12, Unknown),
        VExpF32 | VRcpF32 => c(trans, 16, 20, Unknown),
        // MFMA: scaled fp8 32x32x64 (16 passes on the matrix pipe); latency until D readable.
        VMfmaScale32x32x64Fp8 | VMfmaScale32x32x64Fp8Z => c(Mfma, 32, 40, Unknown),
        VMfma32x32x16Fp8 | VMfma32x32x8F16 => c(Mfma, 32, 40, Unknown),
        VMfma16x16x32Fp8 | VMfma16x16x16F16 => c(Mfma, 16, 24, Unknown),
        LdsReadB32 | LdsReadB128 | LdsReadB64TrB8 => c(Lds, 8, 64, Unknown),
        LdsWriteB32 | LdsWriteB128 => c(Lds, 8, 4, Unknown),
        GlobalLoadU16 | GlobalLoadF32 | GlobalLoadB128 => c(Vmem, 8, 500, Unknown),
        GlobalStoreU16 | GlobalStoreF32 => c(Vmem, 8, 4, Unknown),
        SLoadDword | SLoadDwordx2 => c(Smem, 1, 200, Unknown),
        SReadPhys | SMovB32 | SAddU32 | SAddcU32 | SMulI32 | SShlB32 | SCmpLtI32 => c(Salu, 1, 2, Experimental),
        SBarrier | SSetPrio | SNop | SWaitVm | SWaitLgkm | SEndpgm => c(Ctrl, 1, 1, Unknown),
    }
}