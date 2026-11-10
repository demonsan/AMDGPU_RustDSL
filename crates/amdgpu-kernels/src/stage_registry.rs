//! The stage registry of the shipped kernel families (docs/fusion-stage-library-design.md, docs/crate-split.md D10): one `StageDesc` per stage kind the kernels instantiate. The op graph (`amdgpu-graph`) refers
//! to these keys; its skeleton generator receives this slice as an argument instead of reading a table of the L2 crate.

use amdgpu_stage::stage::{Prim, StageDesc};

/// The stage kinds instantiated by the kernels (keys are stable: the op graph refers to them). The registry is owned by the kernel crate (docs/crate-split.md D10): the L2 stage crate holds
/// `StageDesc` and the composition API, the families register their stage kinds.
pub fn registry() -> &'static [StageDesc] {
    &REGISTRY
}

static REGISTRY: [StageDesc; 17] = [
    StageDesc { key: "move.row_load", prim: Prim::M, params: &[("n", "row length"), ("dtype", "element type")], doc: "one row per wave into registers (16-byte buffer loads, kept packed)" },
    StageDesc { key: "reduce.row_sumsq", prim: Prim::R, params: &[("op", "reduction")], doc: "sum of squares over a wave-held row -> rsqrt scalar (RMSNorm)" },
    StageDesc { key: "quant.fp8_row", prim: Prim::Q, params: &[("format", "fp8 format"), ("granularity", "scale granularity"), ("scale", "scale type"), ("map", "elementwise map before quantization")], doc: "row amax -> scale -> fp8 pack" },
    StageDesc { key: "move.row_store", prim: Prim::M, params: &[("dtype", "stored types")], doc: "packed row + its scale to global memory" },
    StageDesc { key: "merged", prim: Prim::A, params: &[], doc: "several instances in one region (stage::merge); members recorded as parameters" },
    StageDesc { key: "move.ab_stage", prim: Prim::M, params: &[("slots", "ring slots")], doc: "GEMM A / B tile copies into an LDS ring slot (one unit per DMA)" },
    StageDesc { key: "mma.blockscale_tile", prim: Prim::A, params: &[("atom", "MMA atom"), ("wave_tile", "rows x cols per wave"), ("staged", "reload the next tile")], doc: "the register tile of a blockscale GEMM with in-place fragment reloads" },
    StageDesc { key: "move.lds_image", prim: Prim::M, params: &[("bytes", "image size"), ("pitch", "row pitch")], doc: "a whole global image into an LDS tile by LDS-direct DMA (weight-stationary prologue)" },
    StageDesc { key: "move.row_frags", prim: Prim::M, params: &[("k", "row length (bytes)")], doc: "row-operand MMA fragments straight from global memory (buffer loads)" },
    StageDesc { key: "mma.rows_x_lds", prim: Prim::A, params: &[("atom", "MMA atom"), ("n", "output columns"), ("k", "reduction length")], doc: "C^T += W (LDS) x rows^T (registers), accumulators in registers" },
    StageDesc { key: "map.relu_pack_fp8", prim: Prim::E, params: &[("alpha", "scale")], doc: "fp8(relu(alpha x)) packed in accumulator order (the zero-cost B-operand hand-off)" },
    StageDesc { key: "map.store_bf16", prim: Prim::E, params: &[("n", "columns")], doc: "accumulators -> bf16 -> global stores" },
    StageDesc { key: "move.kv_stage", prim: Prim::M, params: &[("slots", "ring slots")], doc: "K / K-scale / V tile copies into an LDS ring slot (buffer-form DMA units)" },
    StageDesc { key: "mma.qk", prim: Prim::A, params: &[("atom", "MMA atom"), ("scale", "K-scale form")], doc: "S^T = K Q^T into the S accumulators (FMHA)" },
    StageDesc { key: "online.softmax", prim: Prim::O, params: &[("recipe", "reference recipe")], doc: "lazy online softmax: decide (rollback branch) + P conversion" },
    StageDesc { key: "mma.pv", prim: Prim::A, params: &[("atom", "MMA atom")], doc: "O^T += V^T P^T with the row sums (FMHA)" },
    StageDesc { key: "map.epilogue_chain", prim: Prim::E, params: &[("ops", "scale / bias / act / quant / store")], doc: "accumulator epilogue chain (P2)" },
];
