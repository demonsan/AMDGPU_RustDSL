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
    /// Cycles the *wave* cannot issue another instruction of this pipe.
    pub issue: u32,
    /// Cycles the SIMD-shared unit is busy (what a second wave on the SIMD competes for).
    pub occ: u32,
    pub latency: u32,
    pub grade: Grade,
}

pub(crate) const fn c(pipe: Pipe, issue: u32, latency: u32, grade: Grade) -> OpCost {
    OpCost { pipe, issue, occ: issue, latency, grade }
}

pub(crate) const fn co(pipe: Pipe, issue: u32, occ: u32, latency: u32, grade: Grade) -> OpCost {
    OpCost { pipe, issue, occ, latency, grade }
}

/// Taken branch (s_branch / taken s_cbranch) and not-taken s_cbranch cost in cycles (calibrated in docs/scheduling.md).
/// Measured (tools/sched_mb/RESULTS2.md, 1 wave, V): taken backward s_cbranch ~24, uncond s_branch (even to the next label) ~18, not taken ~6,
/// extra ~20 cycles from a VALU v_cmp issue to the s_cbranch_vccnz that reads it (about 5 independent VALU hide it).
/// With >=4 waves per CU some launches show +96 per taken branch (cause unknown, not modelled).
pub const BRANCH_TAKEN: u32 = crate::target::T.latency.branch_taken;
pub const BRANCH_UNCOND: u32 = crate::target::T.latency.branch_uncond;
pub const BRANCH_NOT_TAKEN: u32 = crate::target::T.latency.branch_not_taken;
pub const VCC_TO_BRANCH: u32 = crate::target::T.latency.vcc_to_branch;
pub const BARRIER_COST: u32 = crate::target::T.latency.barrier;

/// Global model knobs (also data): whether `v_exp`/`v_rcp` run on their own pipe next to the VALU.
pub struct ModelParams {
    pub trans_separate_pipe: bool,
}

pub const PARAMS: ModelParams = ModelParams { trans_separate_pipe: crate::target::T.latency.trans_separate_pipe };

/// Cost of one opcode (the `cost` attribute of `crate::op_attrs`). Grades: see docs/scheduling.md (tables 1-2).
pub fn cost(op: MOp) -> OpCost {
    let c = crate::op_attrs::attrs(op).cost;
    if crate::target::cur().name == "gfx1250" && std::env::var("AMDC_SCHED_MODEL950").is_err() {
        gfx1250_cost(op, c)
    } else {
        c
    }
}

/// The gfx1250 machine model (one wave32 on a SIMD), from LLVM's GFX1250SpeedModel (SISchedule.td, `GFX125xCommonWriteRes`): VALU 5 cycles
/// (64-bit 6, integer multiply 8), TRANS 8 on its own pipe, LDS 20, SMEM 20, SALU 2, VMEM 320; XDL WMMA 16x16x64 FP8/BF8 one pass (4
/// cycles, the XDL unit busy 4), 16x16x128 FP8/BF8 and the scaled 32x16x128 F4 two passes (8 / 8). The pipes stay the shared table's;
/// the WMMA co-execution windows are the hazard pass's business. `AMDC_SCHED_MODEL950`: the gfx950 numbers (A/B).
fn gfx1250_cost(op: MOp, c: OpCost) -> OpCost {
    let g = Grade::Experimental;
    match c.pipe {
        Pipe::Mfma => match op {
            MOp::VWmmaF32_16x16x64Fp8 | MOp::VWmmaF32_16x16x64Fp8Z => co(Pipe::Mfma, 4, 4, 4, g),
            MOp::VWmmaF32_16x16x128Fp8 | MOp::VWmmaF32_16x16x128Fp8Z | MOp::VWmmaScaleF32_32x16x128F4 | MOp::VWmmaScaleF32_32x16x128F4Z => co(Pipe::Mfma, 8, 8, 8, g),
            _ => c,
        },
        Pipe::Valu => match op {
            MOp::VMulLoU32 | MOp::VMulHiU32 => c_(Pipe::Valu, 1, 8, g),
            _ => c_(Pipe::Valu, 1, 5, g),
        },
        Pipe::Trans => c_(Pipe::Trans, 1, 8, g),
        Pipe::Lds => c_(Pipe::Lds, c.issue.min(4).max(1), 20, g),
        Pipe::Smem => c_(Pipe::Smem, 1, 20, g),
        Pipe::Salu => c_(Pipe::Salu, 1, 2, g),
        Pipe::Vmem => c_(Pipe::Vmem, c.issue.min(4).max(1), 320, g),
        Pipe::Ctrl => c,
    }
}

const fn c_(pipe: Pipe, issue: u32, latency: u32, grade: Grade) -> OpCost {
    c(pipe, issue, latency, grade)
}
