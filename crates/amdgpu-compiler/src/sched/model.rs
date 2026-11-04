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
pub const BRANCH_TAKEN: u32 = 24;
pub const BRANCH_UNCOND: u32 = 18;
pub const BRANCH_NOT_TAKEN: u32 = 6;
pub const VCC_TO_BRANCH: u32 = 16;
pub const BARRIER_COST: u32 = 12;

/// Global model knobs (also data): whether `v_exp`/`v_rcp` run on their own pipe next to the VALU.
pub struct ModelParams {
    pub trans_separate_pipe: bool,
}

pub const PARAMS: ModelParams = ModelParams { trans_separate_pipe: false };

/// Cost of one opcode (the `cost` attribute of `crate::op_attrs`). Grades: see docs/scheduling.md (tables 1-2).
pub fn cost(op: MOp) -> OpCost {
    crate::op_attrs::attrs(op).cost
}
