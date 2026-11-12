//! MXFP8 blockscale GEMM on the route-B layers (P4; target = aiter PR #6696 4-wave pin-AGPR kernel, docs/mxfp8-gemm-4wave-analysis.md), same 96-byte ABI:
//!   C[m, n] (bf16) = sum_k A[m, k] 2^(sa[k/128][m] - 127) * B[n, k] 2^(sb[n/128][k/128] - 127)
//!   A [M, K] fp8 e4m3 row-major; B [N, K] fp8 preshuffled (aiter shuffle_weight 16x16); sa E8M0 [K/128][M]; sb E8M0 [N/128][K/128].
//!
//!
//! The shipped kernel (`gemm_mxfp8_bs_v4`): CTA 256 x 256, 4 waves 2 x 2 of 128 x 128, `v_mfma_scale_f32_16x16x128_f8f6f4` through
//! `TiledMma(swap_ab)`; the vendor's register-staged structure (`Cadence::RegStaged`, u2 unroll): operands in AGPRs reloaded in place from tile t+1,
//! accumulators partly pinned in AGPRs; A in the padded LDS image (1056-byte blocks), B preshuffled, both by MUBUF LDS-DMA with layout-derived
//! voffsets; the scales once per CTA into a transposed panel (VGPR loads + in-register byte transposes); the LDS-staged epilogue; the slot-filling
//! scheduler places the DMA issues.

use amdgpu_tile::{ev, LdsPlan, Lane, Role, Shared, SlotIdx, TileIt};
use amdgpu_dsl::{DType, Ext, FnBuilder, Function, Op, Opnd, ValueId, VarId, I32};
use amdgpu_isa::{MmaAtom, DS_READ_B128X2};
use amdgpu_layout::{layout_to_offsets, make_layout, tup, ComposedLayout, ImmRange, IssuePlan, Ix, Swizzle};
use amdgpu_tile::{make_tiled_copy_a, make_tiled_copy_b, make_tiled_mma, Ctx, TiledMma};

pub mod graph;
pub mod reduce;
mod skeleton_v4;
mod abi;
mod epilogue;
use abi::*;
use amdgpu_stage::gemm::body::*;
pub use amdgpu_stage::gemm::body::verify_sfa_transpose;
pub use amdgpu_stage::gemm::{Elem, Scales};
use amdgpu_tile::dma_plan::*;
use epilogue::*;
use amdgpu_stage::{Cadence, PipeBody, Pipeline};

/// Slots per wave record of the optional timing stamps (`GEMM_STAMP`): 0 kernel start, 1 setup done, 2 prologue done, 3 loop done, 4 epilogue stores issued, 16 + tile = the tile matrix phase.
const STAMP_SLOTS: usize = 96;


/// Build configuration of the shipped MXFP8 blockscale GEMM (`GemmConfig::v4`).
#[derive(Clone, Copy, Debug)]
pub struct GemmConfig {
    pub cta_m: i64,
    pub cta_n: i64,
    /// accumulators (16x16 blocks, m-block-major) pinned in AGPRs: the first `acc_agpr`
    pub acc_agpr: usize,
    /// MMAs of the register tile issued before the wait + barrier of the iteration
    pub pre_sync_mmas: usize,
    /// MMAs of the register tile issued between the top-of-tile waits and the barrier (after `pre_sync_mmas`)
    pub mid_sync_mmas: usize,
    /// LDS epilogue in this many row phases (conversion / LDS writes of the next phase overlap the stores of this one)
    pub epi_phases: usize,
    /// extra scheduler tokens (Function::sched_opts, on top of the Gemm profile's wait pass)
    pub sched: &'static str,
    /// epilogue through LDS (bf16 C block transposed in a wave-private staging area, 16-byte row stores; the shipped form). `false` = the direct-C
    /// epilogue (8-byte stores from the accumulator layout), used by the E3 IR proof (tests/functional_gemm_ir.rs)
    pub lds_epilogue: bool,
    /// the quantization scheme of A and B
    pub scales: Scales,
    /// operand element type (the LDS / register tile is 128 bytes of K either way)
    pub elem: Elem,
    /// split-K: grid z = split index over `S` (kernarg 36) splits of KS (kernarg 88) tiles each; the fp32 partial C goes to `ptr_c` as
    /// [S][M][N] (row / column scales are left to the reduce, `gemm_splitk_reduce`)
    pub split_k: bool,
    /// fused-MoE grouped GEMM over aiter-style sorted routes (`MoeStage`)
    pub moe: Option<MoeStage>,
    /// implicit-GEMM convolution (`ConvCfg`)
    pub conv: Option<ConvCfg>,
    /// `Scales::Mx32` with the shuffled scale layout [rows / 16][KT / 2][64 lanes][4 bytes] (one coalesced dword per lane, block and tile pair)
    pub scale_shuf: bool,
    /// batched GEMM (`_bmm`): grid z = batch (kernarg 36); A, B, C and the scales of batch z are dense blocks of the per-batch sizes, each
    /// operand's base moved by a 64-bit `ptr_mad`
    pub batched: bool,
    /// with `Scales::RowCol`: per-TENSOR scales (`_pt`): `ptr_sfa` / `ptr_sfb` each one fp32 (A and B scale), applied in the epilogue
    pub per_tensor: bool,
    /// MoE gpt-oss form (`_oss`): per-expert fp32 bias at kernarg 144 (stage 1 [E][2 n_out] gate then up; stage 2 [E][n_out]); stage 1's
    /// activation is the clamped swiglu (gate = min(g + bg, 7), up = clamp(u + bu, -7, 7), y = (up + 1) gate sigmoid(1.702 gate))
    pub moe_bias: bool,
    /// 3-slot LDS ring (`Pipeline::rs_depth` 3): tile t + 3 is staged while t + 2 is still in flight (counted top wait) - small CTAs whose
    /// per-tile compute is shorter than the memory latency. Unit-scale forms only (no other VMEM in the loop).
    pub depth3: bool,
    /// split-K with the reduce in the kernel (`_lc`): the last of the S split CTAs of an output tile (atomic ticket on `ptr_cnt[by NT + bx]`,
    /// kernarg 104, u32 [tiles][4 waves], zero before the first launch; the last CTA resets it) sums the S fp32 partials of the tile's real rows, applies the
    /// per-token x per-channel scales and stores bf16 C to `ptr_out` (kernarg 96): one launch instead of the main kernel + `gemm_splitk_reduce`
    pub lastc: bool,
}

/// Implicit-GEMM convolution, NDHWC input [NB, DI, H, W, C] bf16 (2d: DI = 1, kernel depth 1), weights [K][C / 64][T][R][S][64] (the 128-byte
/// K tiles = 64 channels of one tap; then aiter shuffle_weight(16, 16)), output [NB, OD, OH, OW, K] bf16 = C [M, K] with M = NB OD OH OW.
/// GEMM M row m -> (nb, od, oh, ow); K tile t -> (channel block t / TRS, tap t % TRS -> (dz, r, s)); the A row of (m, t) is the input pixel
/// (od st - pad + dz dil, oh st - pad + r dil, ow st - pad + s dil) of batch nb, zero when outside (an out-of-range voffset). ABI: the 96-byte
/// GEMM ABI (A = input, B = weights, C = output, m = M, n = K, k = C T R S) + 96 NB, 100 DI, 104 H, 108 W, 112 C, 116 OD, 120 OH, 124 OW.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConvCfg {
    pub t: i32,
    pub r: i32,
    pub s: i32,
    pub stride: i32,
    pub pad: i32,
    pub dil: i32,
}

/// Per-CTA conv values: per A issue the row's base offset and its first input coordinates; the input sizes.
#[derive(Clone)]
struct ConvRt {
    cfg: ConvCfg,
    rows: Vec<(ValueId, ValueId, ValueId, ValueId)>,
    di: ValueId,
    h: ValueId,
    w: ValueId,
    c: ValueId,
    zero: ValueId,
}

/// q = n / d (both >= 0, n < 2^22) by a float reciprocal and one integer correction step.
impl amdgpu_stage::gemm::body::ConvSource for ConvRt {
    fn tile_voffsets(&self, b: &mut FnBuilder, key0: &I32) -> Vec<ValueId> {
        let cc = self.cfg;
        let trs = cc.t * cc.r * cc.s;
        let rs = cc.r * cc.s;
        // t / trs, rs / s by multiply-shift (exact for t < 2^12 = 4096 K tiles, checked)
        let magic = |d: i32| -> i64 { ((1i64 << 24) + d as i64 - 1) / d as i64 };
        for d in [trs, rs, cc.s] {
            let m = magic(d);
            for x in 0..(1i64 << 12) {
                assert_eq!((x * m) >> 24, x / d as i64, "conv: multiply-shift by {d}");
            }
        }
        let t = b.bind_i32(key0.clone().lshr(7));
        let cb = b.bind_i32((ev(t) * magic(trs) as i32).lshr(24));
        let tap = b.bind_i32(ev(t) - ev(cb) * trs);
        let dz = b.bind_i32((ev(tap) * magic(rs) as i32).lshr(24));
        let tr = b.bind_i32(ev(tap) - ev(dz) * rs);
        let rr = b.bind_i32((ev(tr) * magic(cc.s) as i32).lshr(24));
        let ss = b.bind_i32(ev(tr) - ev(rr) * cc.s);
        let toff = b.bind_i32((((ev(dz) * cc.dil) * ev(self.h) + ev(rr) * cc.dil) * ev(self.w) + ev(ss) * cc.dil) * ev(self.c) * 2 + (ev(cb) << 7));
        let big = b.c_i32(0x7fff_fff0);
        self.rows
            .iter()
            .map(|&(base, id0, ih0, iw0)| {
                let (zi, yi, xi) = (b.bind_i32(ev(id0) + ev(dz) * cc.dil), b.bind_i32(ev(ih0) + ev(rr) * cc.dil), b.bind_i32(ev(iw0) + ev(ss) * cc.dil));
                let ok = b.bind_i32(
                    ev(zi).ge(0).select_i32(1, 0) & ev(zi).lt(ev(self.di)).select_i32(1, 0) & ev(yi).ge(0).select_i32(1, 0) & ev(yi).lt(ev(self.h)).select_i32(1, 0)
                        & ev(xi).ge(0).select_i32(1, 0) & ev(xi).lt(ev(self.w)).select_i32(1, 0),
                );
                let ok = b.bind_cond(ev(ok).eq(1));
                let v = b.bind_i32(ev(base) + ev(toff));
                b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(ok), a: Opnd::Val(v), b: Opnd::Val(big) })
            })
            .collect::<Vec<_>>()
    }
    fn zero(&self) -> ValueId {
        self.zero
    }
}

fn div_rt(b: &mut FnBuilder, n: ValueId, d: ValueId) -> ValueId {
    // (v_cvt_f32_i32 = CvtF32I32: int -> float; v_cvt_i32_f32 = CvtI32F32: float -> int)
    let nf = b.un(amdgpu_dsl::UnOp::CvtF32I32, n);
    let df = b.un(amdgpu_dsl::UnOp::CvtF32I32, d);
    let r = b.rcp_f32(df);
    let qf = b.mul_f32(nf, r);
    let qf = b.un(amdgpu_dsl::UnOp::FloorF32, qf);
    let q = b.un(amdgpu_dsl::UnOp::CvtI32F32, qf);
    let rem = b.bind_i32(ev(n) - ev(q) * ev(d));
    b.bind_i32(ev(q) + ev(rem).ge(ev(d)).select_i32(1, 0) - ev(rem).lt(0).select_i32(1, 0))
}

/// The two grouped GEMMs of a fused MoE over the moe_sorting outputs (sorted_token_ids `(slot << 24) | token`, padding `(topk << 24) | M`;
/// sorted_expert_ids per M tile; num_valid_ids [padded rows, M]; sorted_weights). M tile = cta_m sorted rows of ONE expert; tiles past
/// num_valid_ids[0] exit. ABI = the 96-byte GEMM ABI (A, B = all experts, C, m = A rows, n = B rows per expert, k) + 96 ptr_sorted_ids,
/// 104 ptr_sorted_expert_ids, 112 ptr_num_valid_ids, 120 ptr_sorted_weights, 128 topk, 132 E, 136 n_out, 140 M (tokens).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeStage {
    /// gate / up: A = X [M, k] gathered by token, B = W1 [E, 2 n_out, k] with the gate and up rows INTERLEAVED in 16-row blocks (block 2 p =
    /// gate rows 16 p.., block 2 p + 1 = up rows 16 p..; then aiter shuffle_weight(16, 16)) so a lane holds both for the same output columns;
    /// C = act(gate) up (x the routing weight with `doweight`) bf16 [M, topk, n_out] at row token topk + slot
    Up { gelu: bool, doweight: bool },
    /// down: A = the stage-1 output [M topk, k] gathered by (token, slot), B = W2 [E, n_out, k]; C = fp32 [M, topk, n_out] x the routing
    /// weight (summed over topk by `reduce::moe_sum_bf16`)
    Down,
    /// `Down` with a bf16 per-route intermediate [M, topk, n_out] (half the bytes; summed in fp32 by `moe_sum_bf16h_k<K>`)
    DownH,
}

impl GemmConfig {
    /// The shipped configuration.
    pub fn v4() -> GemmConfig {
        GemmConfig { cta_m: 256, cta_n: 256, acc_agpr: 24, pre_sync_mmas: 5, mid_sync_mmas: 3, epi_phases: 8, sched: "rewait,slotfill", lds_epilogue: true, scales: Scales::Mx128, elem: Elem::Fp8, split_k: false, moe: None, conv: None, scale_shuf: false, batched: false, per_tensor: false, moe_bias: false, depth3: false, lastc: false }
    }
    /// fp8 A8W8 with per-token x per-channel fp32 scales (`gemm_a8w8_rowcol_fp8`): the v4 structure, unit MFMA scales, scaled epilogue.
    pub fn a8w8_rowcol() -> GemmConfig {
        GemmConfig { scales: Scales::RowCol, ..GemmConfig::v4() }
    }
    /// bf16 A16W16 (`gemm_a16w16_bf16`): the v4 structure with bf16 MFMAs, no scales.
    pub fn a16w16() -> GemmConfig {
        GemmConfig { scales: Scales::None, elem: Elem::Bf16, ..GemmConfig::v4() }
    }
    /// bf16 A16W16 with a 128 x 128 CTA.
    pub fn a16w16_128() -> GemmConfig {
        GemmConfig { cta_m: 128, cta_n: 128, ..GemmConfig::a16w16() }
    }
    /// fp8 A8W8 with fp32 1 x 128 / 128 x 128 block scales (`gemm_a8w8_blockscale`): 128 x 128 CTA, VGPR accumulators (acc += t s).
    pub fn a8w8_blockscale() -> GemmConfig {
        GemmConfig { cta_m: 128, cta_n: 128, acc_agpr: 0, scales: Scales::Block128, elem: Elem::Fp8Bs, ..GemmConfig::v4() }
    }
    /// The fused-MoE stage `st` (128-row tiles = block_m 128; stage 1: 256 interleaved gate / up B rows = 128 output columns per CTA).
    pub fn moe(elem: Elem, scales: Scales, st: MoeStage) -> GemmConfig {
        let cta_n = if matches!(st, MoeStage::Up { .. }) { 256 } else { 128 };
        GemmConfig { cta_m: 128, cta_n, lds_epilogue: false, scales, elem, moe: Some(st), ..GemmConfig::v4() }
    }
    /// Implicit-GEMM bf16 convolution (`ConvCfg`) on the bf16 GEMM configuration.
    pub fn conv(cc: ConvCfg, cta: i64) -> GemmConfig {
        GemmConfig { cta_m: cta, cta_n: cta, conv: Some(cc), ..GemmConfig::a16w16() }
    }
    /// Split-K form of a 128 x 128 configuration (fp32 partials + `reduce::gemm_splitk_reduce`).
    pub fn splitk(self) -> GemmConfig {
        GemmConfig { split_k: true, ..self }
    }
    /// Split-K with the in-kernel last-arriver reduce (`GemmConfig::lastc`).
    pub fn splitk_lc(self) -> GemmConfig {
        GemmConfig { split_k: true, lastc: true, ..self }
    }
    /// MXFP4 A4W4 (`gemm_a4w4_mxfp4`): fp4 operands, 1 x 32 E8M0 scales in the scaled MFMAs.
    pub fn a4w4() -> GemmConfig {
        GemmConfig { scales: Scales::Mx32, elem: Elem::Fp4, scale_shuf: true, ..GemmConfig::v4() }
    }
    /// MXFP4 A4W4 with a 128 x 128 CTA.
    /// A8W4 (MXFP8 activations x MXFP4 weights, both shuffled 1 x 32 scales), 128 x 128 CTA.
    pub fn a8w4_128() -> GemmConfig {
        GemmConfig { elem: Elem::Fp8W4, ..GemmConfig::a4w4_128() }
    }
    /// A16W4 (bf16 activations x MXFP4 weights, B scales [N][K / 32] unshuffled), 64 x 128 CTA (four bf16 A images per 256-K tile: 2 x 4 x
    /// 8.25 KB + B 2 x 16 KB of LDS), every MMA in the tile body.
    pub fn a16w4_128() -> GemmConfig {
        GemmConfig { elem: Elem::Bf16W4, scale_shuf: false, pre_sync_mmas: 0, mid_sync_mmas: 0, cta_m: 64, ..GemmConfig::a4w4_128() }
    }
    pub fn a4w4_128() -> GemmConfig {
        GemmConfig { cta_m: 128, cta_n: 128, ..GemmConfig::a4w4() }
    }
    /// The same with a 128 x 128 CTA (4 waves of 64 x 64): 4x the CTAs for narrow M / N.
    pub fn a8w8_rowcol_128() -> GemmConfig {
        GemmConfig { cta_m: 128, cta_n: 128, ..GemmConfig::a8w8_rowcol() }
    }
}

type Cfg = GemmConfig;
/// Epilogue staging row pitch (bytes): 256-byte bf16 rows + 16 bytes of padding (the 16 rows a write instruction touches spread over the banks).
const EPI_PITCH: i64 = 272;

/// Scale panel capacity in K groups of 128 (the vendor `sp64`): K <= 8192.
const PANEL_KB: i64 = 64;
impl GemmConfig {
    /// the numbers the main-loop body reads (ody::BodyParams)
    fn body_params(&self) -> BodyParams {
        BodyParams { elem: self.elem, scales: self.scales, wave_m: self.wave_m(), wave_n: self.wave_n(), pre_sync_mmas: self.pre_sync_mmas, mid_sync_mmas: self.mid_sync_mmas }
    }
    fn wave_m(&self) -> i64 {
        self.cta_m / 2
    }
    fn wave_n(&self) -> i64 {
        self.cta_n / 2
    }
}

/// The shipped MXFP8 blockscale GEMM.
pub fn gemm_mxfp8_bs_v4() -> Function {
    gemm_mxfp8_bs("gemm_mxfp8_bs_v4", GemmConfig::v4())
}

pub fn by_name(name: &str) -> Option<Function> {
    match name {
        "gemm_mxfp8_bs_v4" => Some(gemm_mxfp8_bs_v4()),
        "gemm_a8w8_rowcol_fp8" => Some(gemm_mxfp8_bs("gemm_a8w8_rowcol_fp8", GemmConfig::a8w8_rowcol())),
        "gemm_a8w8_i8" => Some(gemm_mxfp8_bs("gemm_a8w8_i8", GemmConfig { elem: Elem::I8, ..GemmConfig::a8w8_rowcol() })),
        "gemm_a8w8_i8_128" => Some(gemm_mxfp8_bs("gemm_a8w8_i8_128", GemmConfig { elem: Elem::I8, ..GemmConfig::a8w8_rowcol_128() })),
        // 64 x 128 forms: the 3-slot ring (depth3: 256x7168x4096 19.8 -> 15.1 us, 1024x4096x8192 50.5 -> 47.0, 128x4096x8192 S4 16.0 -> 13.9)
        // bf16 64 x 128 on the 3-slot ring (256x7168x4096 41.3 (128) -> 23.5 us, 128x4096x8192 74.0 -> 37.3)
        "gemm_a16w16_bf16_64" => Some(gemm_mxfp8_bs("gemm_a16w16_bf16_64", GemmConfig { cta_m: 64, depth3: true, ..GemmConfig::a16w16_128() })),
        "gemm_a16w16_bf16_64_splitk" => Some(gemm_mxfp8_bs("gemm_a16w16_bf16_64_splitk", GemmConfig { cta_m: 64, depth3: true, ..GemmConfig::a16w16_128() }.splitk())),
        "gemm_a16w16_bf16_64_splitk_lc" => Some(gemm_mxfp8_bs("gemm_a16w16_bf16_64_splitk_lc", GemmConfig { cta_m: 64, depth3: true, ..GemmConfig::a16w16_128() }.splitk_lc())),
        "gemm_a8w8_i8_64" => Some(gemm_mxfp8_bs("gemm_a8w8_i8_64", GemmConfig { elem: Elem::I8, cta_m: 64, depth3: true, ..GemmConfig::a8w8_rowcol_128() })),
        "gemm_a8w8_i8_64_splitk" => Some(gemm_mxfp8_bs("gemm_a8w8_i8_64_splitk", GemmConfig { elem: Elem::I8, cta_m: 64, depth3: true, ..GemmConfig::a8w8_rowcol_128() }.splitk())),
        "gemm_a8w8_i8_64_splitk_lc" => Some(gemm_mxfp8_bs("gemm_a8w8_i8_64_splitk_lc", GemmConfig { elem: Elem::I8, cta_m: 64, depth3: true, ..GemmConfig::a8w8_rowcol_128() }.splitk_lc())),
        // (fp8 per-token: 256x7168x4096 22.98 (128) -> 13.60 us, 1024x4096x8192 51.6 -> 46.0)
        "gemm_a8w8_rowcol_fp8_64" => Some(gemm_mxfp8_bs("gemm_a8w8_rowcol_fp8_64", GemmConfig { cta_m: 64, depth3: true, ..GemmConfig::a8w8_rowcol_128() })),
        "gemm_a8w8_rowcol_fp8_64_splitk" => Some(gemm_mxfp8_bs("gemm_a8w8_rowcol_fp8_64_splitk", GemmConfig { cta_m: 64, depth3: true, ..GemmConfig::a8w8_rowcol_128() }.splitk())),
        "gemm_a8w8_rowcol_fp8_64_splitk_lc" => Some(gemm_mxfp8_bs("gemm_a8w8_rowcol_fp8_64_splitk_lc", GemmConfig { cta_m: 64, depth3: true, ..GemmConfig::a8w8_rowcol_128() }.splitk_lc())),
        "gemm_a8w8_rowcol_fp8_128" => Some(gemm_mxfp8_bs("gemm_a8w8_rowcol_fp8_128", GemmConfig::a8w8_rowcol_128())),
        "gemm_a16w16_bf16" => Some(gemm_mxfp8_bs("gemm_a16w16_bf16", GemmConfig::a16w16())),
        "gemm_a16w16_bf16_128" => Some(gemm_mxfp8_bs("gemm_a16w16_bf16_128", GemmConfig::a16w16_128())),
        "gemm_a8w8_blockscale" => Some(gemm_mxfp8_bs("gemm_a8w8_blockscale", GemmConfig::a8w8_blockscale())),
        "gemm_a8w8_blockscale_splitk" => Some(gemm_mxfp8_bs("gemm_a8w8_blockscale_splitk", GemmConfig::a8w8_blockscale().splitk())),
        // the mid / small-M forms (`dispatch::blockscale_config`)
        "gemm_a8w8_blockscale_64" => Some(gemm_mxfp8_bs("gemm_a8w8_blockscale_64", GemmConfig { cta_m: 64, depth3: true, ..GemmConfig::a8w8_blockscale() })),
        "gemm_a8w8_blockscale_64_splitk" => Some(gemm_mxfp8_bs("gemm_a8w8_blockscale_64_splitk", GemmConfig { cta_m: 64, depth3: true, ..GemmConfig::a8w8_blockscale() }.splitk())),
        "gemm_a8w8_blockscale_64_splitk_lc" => Some(gemm_mxfp8_bs("gemm_a8w8_blockscale_64_splitk_lc", GemmConfig { cta_m: 64, depth3: true, ..GemmConfig::a8w8_blockscale() }.splitk_lc())),
        "gemm_a8w8_rowcol_fp8_128_splitk" => Some(gemm_mxfp8_bs("gemm_a8w8_rowcol_fp8_128_splitk", GemmConfig::a8w8_rowcol_128().splitk())),
        "gemm_a16w16_bf16_128_splitk" => Some(gemm_mxfp8_bs("gemm_a16w16_bf16_128_splitk", GemmConfig::a16w16_128().splitk())),
        n if n.starts_with("moe_sum_bf16h_k") => n["moe_sum_bf16h_k".len()..].parse::<i64>().ok().filter(|k| (1..=16).contains(k)).map(|k| reduce::moe_sum_bf16_x(n, k, true)),
        n if n.starts_with("moe_sum_bf16_k") => n["moe_sum_bf16_k".len()..].parse::<i64>().ok().filter(|k| (1..=16).contains(k)).map(|k| reduce::moe_sum_bf16(n, k)),
        n if n.starts_with("conv2d_bf16_") || n.starts_with("conv3d_bf16_") => {
            // conv{2,3}d_bf16_[t<T>]r<R>s<S>_st<st>_p<pad>_d<dil>[_c128]
            let num = |key: &str, dflt: i32| -> Option<i32> {
                match n.find(key) {
                    None => Some(dflt),
                    Some(i) => n[i + key.len()..].chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse().ok(),
                }
            };
            let body = &n[11..];
            let t = if n.starts_with("conv3d") { num("_t", 1).or_else(|| body.strip_prefix('t').and_then(|x| x.split('r').next()?.parse().ok()))? } else { 1 };
            let r = body.split('r').nth(1)?.split('s').next()?.parse().ok()?;
            let s = body.split('s').nth(1)?.split('_').next()?.parse().ok()?;
            let cc = ConvCfg { t, r, s, stride: num("_st", 1)?, pad: num("_p", 0)?, dil: num("_d", 1)? };
            Some(gemm_mxfp8_bs(n, GemmConfig::conv(cc, if n.ends_with("_c128") { 128 } else { 256 })))
        }
        n if n.starts_with("gemm_moe") => {
            // gemm_moe<1|2>_<bf16|fp8>[_silu|_gelu][_dw]
            let (elem, scales) = if n.contains("_fp8bs") {
                (Elem::Fp8Bs, Scales::Block128)
            } else if n.contains("_fp8") {
                (Elem::Fp8, Scales::RowCol)
            } else if n.contains("_bf16w4") {
                // A16W4: bf16 A (exact) x MXFP4 B dequantized in the mainloop (K-permuted B, unshuffled [rows][K / 32] scales); 64-row tiles
                (Elem::Bf16W4, Scales::Mx32)
            } else if n.contains("_fp4") {
                // MXFP4 A x MXFP4 B (1 x 32 E8M0, unshuffled [rows][K / 32])
                (Elem::Fp4, Scales::Mx32)
            } else {
                (Elem::Bf16, Scales::None)
            };
            // `_bm64` suffix: block_m 64 (64-row tiles, sparse experts)
            let (n0, bm64) = match n.strip_suffix("_bm64") { Some(x) => (x, true), None => (n, false) };
            // `_oss`: gpt-oss bias + clamped swiglu (stage 1) / bias (stage 2)
            let (n0, oss) = match n0.strip_suffix("_oss") { Some(x) => (x, true), None => (n0, false) };
            let st = if n0.starts_with("gemm_moe1") {
                MoeStage::Up { gelu: n0.contains("_gelu"), doweight: n0.ends_with("_dw") }
            } else if n0.ends_with("_h") {
                MoeStage::DownH
            } else {
                MoeStage::Down
            };
            let cfg = GemmConfig::moe(elem, scales, st);
            let cfg = GemmConfig { moe_bias: oss, ..cfg };
            // A16W4: every MMA in the tile body, block_m 64 (four bf16 A images per slot)
            let cfg = if elem == Elem::Bf16W4 {
                assert!(bm64, "a16w4 MoE: block_m 64 (`_bm64`)");
                GemmConfig { pre_sync_mmas: 0, mid_sync_mmas: 0, scale_shuf: false, ..cfg }
            } else {
                cfg
            };
            // stage 1, block_m 64, unit scales (fp8 per-token / bf16): the 3-slot ring (fp8 D4096 I1024 E64 topk 8, same run: M 16 99.6 -> 92.7 us,
            // M 64 110.0 -> 107.9, M 256 112.7 -> 112.4; stage 2 measured mixed, block_m 128 worse: unchanged)
            let d3 = bm64 && matches!(st, MoeStage::Up { .. }) && matches!(scales, Scales::RowCol | Scales::None);
            let cfg = GemmConfig { depth3: d3, ..cfg };
            Some(gemm_mxfp8_bs(n, if bm64 { GemmConfig { cta_m: 64, ..cfg } } else { cfg }))
        }
        n if n.starts_with("gemm_splitk_reduce_s") => {
            // gemm_splitk_reduce_s<S>[_rc]
            let rest = &n["gemm_splitk_reduce_s".len()..];
            let (s, rc) = match rest.strip_suffix("_rc") { Some(s) => (s, true), None => (rest, false) };
            s.parse::<i64>().ok().filter(|s| (1..=16).contains(s)).map(|s| reduce::gemm_splitk_reduce(n, s, rc))
        }
        // batched forms (`_bmm`: grid z = batch)
        "gemm_a16w16_bf16_bmm" => Some(gemm_mxfp8_bs("gemm_a16w16_bf16_bmm", GemmConfig { batched: true, ..GemmConfig::a16w16() })),
        "gemm_mxfp8_bs_v4_bmm" => Some(gemm_mxfp8_bs("gemm_mxfp8_bs_v4_bmm", GemmConfig { batched: true, ..GemmConfig::v4() })),
        // per-tensor fp8 (one fp32 scale for A and one for B)
        "gemm_a8w8_pertensor_fp8" => Some(gemm_mxfp8_bs("gemm_a8w8_pertensor_fp8", GemmConfig { per_tensor: true, ..GemmConfig::a8w8_rowcol() })),
        "gemm_a8w8_pertensor_fp8_128" => Some(gemm_mxfp8_bs("gemm_a8w8_pertensor_fp8_128", GemmConfig { per_tensor: true, ..GemmConfig::a8w8_rowcol_128() })),
        "gemm_a8w8_rowcol_fp8_bmm" => Some(gemm_mxfp8_bs("gemm_a8w8_rowcol_fp8_bmm", GemmConfig { batched: true, ..GemmConfig::a8w8_rowcol() })),
        "gemm_a4w4_mxfp4" => Some(gemm_mxfp8_bs("gemm_a4w4_mxfp4", GemmConfig::a4w4())),
        "gemm_a8w4_mxfp_128" => Some(gemm_mxfp8_bs("gemm_a8w4_mxfp_128", GemmConfig::a8w4_128())),
        "gemm_a16w4_mxfp4" => Some(gemm_mxfp8_bs("gemm_a16w4_mxfp4", GemmConfig::a16w4_128())),
        "gemm_a4w4_mxfp4_128" => Some(gemm_mxfp8_bs("gemm_a4w4_mxfp4_128", GemmConfig::a4w4_128())),
        _ => None,
    }
}

/// Builds the GEMM for `cfg` under `name`.
pub fn gemm_mxfp8_bs(name: &str, cfg: GemmConfig) -> Function {
    build(name, cfg)
}
fn build(name: &str, cfg: Cfg) -> Function {
    let mut slots = GemmSlots { cfg };
    skeleton_v4::build_v4(name, &mut slots)
}

/// Everything the slots of the GEMM share (the locals of the former monolithic builder).
/// MoE per-CTA values.
#[derive(Clone)]
struct MoeCx {
    st: MoeStage,
    rsrc_sid: ValueId,
    rsrc_sw: ValueId,
    topk: ValueId,
    mtok: ValueId,
    nout: ValueId,
    e: ValueId,
    live: ValueId,
    row0: ValueId,
}

pub(crate) struct GemmCx {
    cfg: Cfg,
    moe: Option<MoeCx>,
    rsrc_sfa_e: Option<(ValueId, ValueId)>,
    body: Body,
    pipe: Option<Pipeline>,
    ctx: Ctx,
    mma: TiledMma,
    epi: Option<Shared>,
    lane: Lane,
    s_lane: Ix,
    s_n: Ix,
    s_wm: Ix,
    s_wn: Ix,
    ptr_c: ValueId,
    m_v: ValueId,
    n_v: ValueId,
    bx: ValueId,
    by: ValueId,
    wm: ValueId,
    wn: ValueId,
    lg_m: i32,
    lg_n: i32,
    wave_m: i64,
    wave_n: i64,
    stamp_ctx: Option<(ValueId, ValueId, bool)>,
    /// `lastc`: the LDS word that broadcasts the CTA's ticket
    tk: Option<Shared>,
}

/// The slots of the GEMM v4 skeleton (`skeleton_v4.rs`, generated from `graph::graph()`): the orchestration the graph leaves to the kernel.
pub(crate) struct GemmSlots {
    cfg: Cfg,
}

impl skeleton_v4::V4Slots for GemmSlots {
    type Cx = GemmCx;
    fn prelude(&mut self, b: &mut FnBuilder) -> GemmCx {
        let cfg = self.cfg;
    let (cta_m, cta_n, wave_m, wave_n) = (cfg.cta_m, cfg.cta_n, cfg.wave_m(), cfg.wave_n());
    // compile policy: lowering `lgkmn,zeroinit` + the scoreboard wait pass (partial waits across the loop back edge, ds_read hoisting); nothing name-keyed
    b.compile_profile(amdgpu_dsl::CompileProfile::Gemm);
    if !cfg.sched.is_empty() {
        b.sched_opts(cfg.sched);
    }
    declare_abi(b);
    declare_contract(b, cta_m, cta_n, cfg.scales, cfg.elem.per_tile(), cfg.split_k, cfg.moe, cfg.conv, cfg.batched, cfg.per_tensor, cfg.elem.a_images(), cfg.moe_bias);
    // timing stamps (opt-in debug build, GEMM_STAMP=ends|tiles): one extra pointer argument after the ABI (offset 96); see docs/stamps.md
    let stamp_mode = std::env::var("GEMM_STAMP").ok().filter(|v| v == "ends" || v == "tiles");
    if cfg.lastc {
        assert!(cfg.split_k && stamp_mode.is_none() && !cfg.batched && cfg.moe.is_none() && cfg.conv.is_none() && cta_n == 128, "lastc: plain split-K, 128 columns");
        b.arg("ptr_out", DType::Ptr, 96, 8);
        b.arg("ptr_cnt", DType::Ptr, 104, 8);
        let (mt, nt) = (Ext::sym("MT"), Ext::sym("NT"));
        b.arg_tensor("ptr_out", Ext::c(2 * cta_m * cta_n) * mt.clone() * nt.clone());
        b.arg_tensor("ptr_cnt", Ext::c(16) * mt.clone() * nt.clone());
        // the S slices of fp32 partials: written by the S split CTAs (generation = split id), read in full by the tile's last arriver
        b.buf_region("ws", "ptr_c", Ext::c(0), Ext::c(4 * cta_m * cta_n) * Ext::sym("S") * mt * nt);
        b.buf_last_arriver("ws", "ptr_cnt", Ext::sym("S"));
    }
    if stamp_mode.is_some() {
        b.arg("dbg_stamps", DType::Ptr, 96, 8);
        b.arg_tensor("dbg_stamps", Ext::sym("MT") * Ext::sym("NT") * Ext::c((4 * STAMP_SLOTS * 4) as i64));
    }
    let ptr_a = b.let_(Op::KernargPtr { offset: 0 });
    let ptr_b = b.let_(Op::KernargPtr { offset: 8 });
    let ptr_c = b.let_(Op::KernargPtr { offset: 16 });
    let ptr_sfa = b.let_(Op::KernargPtr { offset: 64 });
    let ptr_sfb = b.let_(Op::KernargPtr { offset: 72 });
    // batched: batch z's operands (dense per-batch blocks: A M KB, B N KB bytes (KB = 128 KT), C 2 M N, the scales by kind)
    let (ptr_a, ptr_b, ptr_c, ptr_sfa, ptr_sfb) = if cfg.batched {
        assert!(!cfg.split_k && cfg.moe.is_none() && cfg.conv.is_none(), "batched: plain GEMMs only");
        let z = b.let_(Op::WorkgroupIdZ);
        let m0 = b.let_(Op::KernargI32 { offset: 24 });
        let n0 = b.let_(Op::KernargI32 { offset: 28 });
        let kel = b.let_(Op::KernargI32 { offset: 32 });
        let kt0 = b.let_(Op::KernargI32 { offset: 84 });
        let sz = |b: &mut FnBuilder, e: amdgpu_dsl::I32| { let v = b.bind_i32(e); b.read_first_lane(v) };
        let a_sz = sz(b, ev(m0) * ev(kt0) * 128);
        let b_sz = sz(b, ev(n0) * ev(kt0) * 128);
        let c_sz = sz(b, ev(m0) * ev(n0) * 2);
        let (sa_sz, sb_sz) = match cfg.scales {
            Scales::None => (sz(b, amdgpu_dsl::I32::c(0)), sz(b, amdgpu_dsl::I32::c(0))),
            Scales::RowCol if cfg.per_tensor => (sz(b, amdgpu_dsl::I32::c(4)), sz(b, amdgpu_dsl::I32::c(4))),
            Scales::RowCol => (sz(b, ev(m0) * 4), sz(b, ev(n0) * 4)),
            Scales::Mx128 => (sz(b, ev(kel).lshr(7) * ev(m0)), sz(b, ev(n0).lshr(7) * ev(kel).lshr(7))),
            Scales::Block128 => (sz(b, ev(kel).lshr(7) * ev(m0) * 4), sz(b, ev(n0).lshr(7) * ev(kel).lshr(7) * 4)),
            Scales::Mx32 => (sz(b, ev(m0) * ev(kel).lshr(5)), sz(b, ev(n0) * ev(kel).lshr(5))),
        };
        (b.ptr_mad(ptr_a, z, a_sz), b.ptr_mad(ptr_b, z, b_sz), b.ptr_mad(ptr_c, z, c_sz), b.ptr_mad(ptr_sfa, z, sa_sz), b.ptr_mad(ptr_sfb, z, sb_sz))
    } else {
        (ptr_a, ptr_b, ptr_c, ptr_sfa, ptr_sfb)
    };
    let m_v = b.let_(Op::KernargI32 { offset: 24 });
    let n_v = b.let_(Op::KernargI32 { offset: 28 });
    let k_el = b.let_(Op::KernargI32 { offset: 32 });
    // K in bytes (the tiles are 128 bytes of K)
    let k_v = match cfg.elem.per_tile() {
        128 => k_el,
        64 => b.bind_i32(ev(k_el) << 1),
        _ => b.bind_i32(ev(k_el).lshr(1)),
    };
    let kb_v = b.let_(Op::KernargI32 { offset: 84 });
    let tid = b.let_(Op::WorkitemId);
    let lane = Lane::from_tid(b, tid);
    let bx0 = b.let_(Op::WorkgroupIdX); // N tile
    let by0 = b.let_(Op::WorkgroupIdY); // M tile
    let stamp_ctx: Option<(ValueId, ValueId, bool)> = stamp_mode.as_ref().map(|mode| {
        let buf = b.let_(Op::KernargPtr { offset: 96 });
        let nt = b.bind_i32(ev(n_v).lshr(cta_n.trailing_zeros() as i32));
        let lin = b.bind_i32(ev(bx0) + ev(by0) * ev(nt));
        let rec = b.stamp_rec_off(lin, STAMP_SLOTS as u32, 4);
        b.assume_trusted(rec, Ext::c(0), Ext::sym("MT") * Ext::sym("NT") * Ext::c((4 * STAMP_SLOTS * 4) as i64) - Ext::c(4 * STAMP_SLOTS as i64), "debug stamps: the record offset (workgroup * 4 + wave) * slots * 4 is inside dbg_stamps by construction (grid = NT x MT, 4 waves)");
        let t0 = b.memtime_lo();
        b.stamp_store(buf, rec, 0, t0); // slot 0: kernel start
        (buf, rec, mode == "tiles")
    });
    // XCD-aware raster: workgroups are dispatched round robin over the 8 XCDs (linear id % 8), each with its own L2;
    // give every XCD whole 4 x 4 blocks of tiles (4 A + 4 B panels instead of all A panels). Only for a 16 x 16 tile grid here; identity otherwise.
    let (bx, by) = if cfg.moe.is_some() || cfg.conv.is_some() { (bx0, by0) } else {
        amdgpu_stage::templates::grid::xcd_raster_16x16(b, bx0, by0, n_v, m_v, (16 * cfg.cta_n) as i32, (16 * cfg.cta_m) as i32)
    };
    let (lg_m, lg_n) = (cta_m.trailing_zeros() as i32, cta_n.trailing_zeros() as i32);
    let wm = b.bind_i32(lane.wave() & 1);
    let wn = b.bind_i32(lane.wave().lshr(1));

    // ---- layout symbols ----
    let mut ctx = Ctx::new();
    let s_lane = ctx.sym("lane", 0, 63, lane.lane());
    let s_tid = ctx.sym("tid", 0, 255, ev(tid));
    let s_k = ctx.sym("K", 128, 1 << 20, ev(k_v));
    // A row bytes (mixed precision: A fp8 = K, A bf16 = 2 K bytes per row)
    let ka_v = match cfg.elem {
        Elem::Fp8W4 => k_el,
        Elem::Bf16W4 => b.bind_i32(ev(k_el) << 1),
        _ => k_v,
    };
    let s_ka = if cfg.elem.a_images() > 1 { ctx.sym("KA", 256, 1 << 22, ev(ka_v)) } else { s_k.clone() };
    let s_n = ctx.sym("N", 128, 1 << 20, ev(n_v));
    let s_wm = ctx.sym("wm", 0, 1, ev(wm));
    let s_wn = ctx.sym("wn", 0, 1, ev(wn));

    // ---- MMA + operand copies ----
    let mma = make_tiled_mma(MmaAtom::mfma_scale_16x16x128_f8(), true);
    let copy_a = make_tiled_copy_a(DS_READ_B128X2, &mma);
    let copy_b = make_tiled_copy_b(DS_READ_B128X2, &mma);

    // ---- LDS ring: 2 slots of the padded A image + B 32 KB (+ an unused per-tile scale ring kept for the shipped LDS layout) ----
    let mut lds = LdsPlan::new();
    let slots = if cfg.depth3 { 3 } else { 2 };
    assert!(!cfg.depth3 || (matches!(cfg.scales, Scales::RowCol | Scales::None) || (cfg.scales == Scales::Block128 && cfg.moe.is_none())) && cfg.conv.is_none(), "depth3: unit-scale plain GEMMs");
    let a_bytes = cta_m / 8 * 1056;
    // mixed precision: a_images padded A images per slot (the 128-byte pieces of the 256-element K tile)
    let nimg = cfg.elem.a_images() as i64;
    let a_s = lds.ring("A", nimg * a_bytes, slots, amdgpu_tile::Layout::Linear);
    let b_s = lds.ring("B", cta_n * 128, slots, amdgpu_tile::Layout::Linear);
    let sc_s = lds.ring("SC", 4 * 512, slots, amdgpu_tile::Layout::Linear);
    // scale panel: A [PANEL_KB][256 rows] + B [2 blocks][PANEL_KB] (+ 4 bytes of source alignment slack, padded to 256 for the dword DMA)
    let panel_sh = (cfg.scales == Scales::Mx128).then(|| lds.tile("SCP", PANEL_KB * 256 + 512, amdgpu_tile::Layout::Linear));
    // epilogue staging (wave-private, reusing the dead ring / panel storage): the wave's C block, bf16 rows of EPI_PITCH bytes
    let epi = if cfg.lds_epilogue {
        let mut ov: Vec<&Shared> = vec![&a_s, &b_s, &sc_s];
        ov.extend(panel_sh.as_ref());
        let per_wave = wave_m * EPI_PITCH;
        assert!(4 * per_wave <= 160 * 1024 - 16, "epilogue staging does not fit the LDS");
        Some(lds.private("EPI", per_wave, 4, &ov))
    } else {
        None
    };
    let tk: Option<Shared> = None;
    lds.commit(b);

    // ---- operand reads: the 16-row blocks of the wave (padded A: a 16-row block = two 1056-byte blocks of 8 rows; B 2048 bytes) ----
    let view = ComposedLayout::new(Swizzle::none(), smem_block());
    let a_view = ComposedLayout::new(Swizzle::none(), make_layout(tup![tup![8, 2], 128], tup![tup![128, 1056], 1]));
    let a_blk = 2112;
    let a_place: Vec<i64> = (0..wave_m / 16).map(|i| a_blk * i).collect();
    let b_place: Vec<i64> = (0..wave_n / 16).map(|j| 2048 * j).collect();
    // ---- DMA (A row-major into the padded image, B preshuffled), wave parts of M0, CTA origins ----
    let a_glob = make_layout(tup![cta_m, 128], tup![s_ka.clone(), 1]);
    let b_glob = make_layout(tup![tup![16, cta_n / 16], tup![16, 8]], tup![tup![16, s_k.mul_c(16)], tup![1, 256]]);
    // MoE: the tile's rows, its expert, the live test (tiles past num_valid_ids[0] do nothing)
    let moe_cx = cfg.moe.map(|st| {
        let kp = |b: &mut FnBuilder, o: u32| b.let_(Op::KernargPtr { offset: o });
        let ki = |b: &mut FnBuilder, o: u32| b.let_(Op::KernargI32 { offset: o });
        let (p_sid, p_eid, p_nv, p_sw) = (kp(b, 96), kp(b, 104), kp(b, 112), kp(b, 120));
        let (topk, ne, nout, mtok) = (ki(b, 128), ki(b, 132), ki(b, 136), ki(b, 140));
        let nrows = { let v = b.bind_i32(((ev(by) + 1) << lg_m) << 2); b.read_first_lane(v) };
        let rsrc_sid = b.buffer_desc_dyn(p_sid, nrows);
        let rsrc_sw = b.buffer_desc_dyn(p_sw, nrows);
        let rsrc_nv = b.buffer_desc(p_nv, 8);
        let neid = { let v = b.bind_i32((ev(by) + 1) << 2); b.read_first_lane(v) };
        let rsrc_e = b.buffer_desc_dyn(p_eid, neid);
        let c0 = b.c_i32(0);
        let nv = { let v = b.buffer_load_f32(rsrc_nv, c0, 0i32, 0); let v = b.bitcast_i32(v); b.read_first_lane(v) };
        let eo = { let v = b.bind_i32(ev(by) << 2); b.read_first_lane(v) };
        let e = { let v = b.buffer_load_f32(rsrc_e, c0, eo, 0); let v = b.bitcast_i32(v); let v = b.bind_i32(ev(v).max(0).min(ev(ne) - 1)); b.read_first_lane(v) };
        let row0 = { let v = b.bind_i32(ev(by) << lg_m); b.read_first_lane(v) };
        let live = b.bind_cond(ev(row0).lt(ev(nv)));
        MoeCx { st, rsrc_sid, rsrc_sw, topk, mtok, nout, e, live, row0 }
    });
    let mut conv_rt: Option<ConvRt> = None;
    let a_dma = match &moe_cx {
        None if cfg.conv.is_some() => {
            // implicit-GEMM conv: per issue row m = by cta_m + r -> (nb, od, oh, ow), the first input coordinates and the row's base offset
            let cc = cfg.conv.unwrap();
            let ki = |b: &mut FnBuilder, o: u32| b.let_(Op::KernargI32 { offset: o });
            let (di, h, w, c, od, oh, ow) = (ki(b, 100), ki(b, 104), ki(b, 108), ki(b, 112), ki(b, 116), ki(b, 120), ki(b, 124));
            let p2 = b.bind_i32(ev(oh) * ev(ow));
            let p1 = b.bind_i32(ev(od) * ev(p2));
            let issues = cta_m * 128 / 1024 / 4;
            let mut rows = Vec::new();
            let mut out = Vec::new();
            for q in 0..issues {
                let r = (lane.wave() * issues as i32 + q as i32) * 8 + lane.lane().lshr(3);
                let m = b.bind_i32(((ev(by) << lg_m) + r).min(ev(m_v) - 1));
                let nb = div_rt(b, m, p1);
                let r1 = b.bind_i32(ev(m) - ev(nb) * ev(p1));
                let odv = div_rt(b, r1, p2);
                let r2 = b.bind_i32(ev(r1) - ev(odv) * ev(p2));
                let ohv = div_rt(b, r2, ow);
                let owv = b.bind_i32(ev(r2) - ev(ohv) * ev(ow));
                // 2d (kernel depth 1): no depth stride / padding
                let (sd, pd) = if cc.t == 1 { (1, 0) } else { (cc.stride, cc.pad) };
                let id0 = b.bind_i32(ev(odv) * sd - pd);
                let ih0 = b.bind_i32(ev(ohv) * cc.stride - cc.pad);
                let iw0 = b.bind_i32(ev(owv) * cc.stride - cc.pad);
                let base = b.bind_i32((((ev(nb) * ev(di) + ev(id0)) * ev(h) + ev(ih0)) * ev(w) + ev(iw0)) * ev(c) * 2 + ((lane.lane() & 7) << 4));
                rows.push((base, id0, ih0, iw0));
                out.push((base, 1024 * q as i32));
            }
            let zero = { let z = b.c_i32(0); b.read_first_lane(z) };
            conv_rt = Some(ConvRt { cfg: cc, rows, di, h, w, c, zero });
            Dma { issues: out, pitch: 1056, gather: true }
        }
        None => dma_plan_rows_pad(b, &ctx, &s_tid, cta_m, &a_glob, 8192, 4),
        Some(mc) => {
            let mc = mc.clone();
            // (mixed precision: the A row stride is the A element's K bytes)
            dma_plan_gather(b, &lane, cta_m, 4, ka_v, &|b, r| {
                // source row of sorted row row0 + r: the token (stage 1) or token topk + slot (stage 2), clamped inside A
                let vo = b.bind_i32(r << 2);
                let so = { let v = b.bind_i32(ev(mc.row0) << 2); b.read_first_lane(v) };
                let sid = b.buffer_load_f32(mc.rsrc_sid, vo, so, 0);
                let sid = b.bitcast_i32(sid);
                let tok = b.bind_i32(ev(sid) & 0xff_ffff);
                match mc.st {
                    MoeStage::Up { .. } => b.bind_i32(ev(tok).min(ev(mc.mtok) - 1)),
                    MoeStage::Down | MoeStage::DownH => b.bind_i32((ev(tok) * ev(mc.topk) + ev(sid).lshr(24)).min(ev(m_v) - 1).max(0)),
                }
            })
        }
    };
    let b_dma = dma_plan(b, &ctx, &s_tid, cta_n, &b_glob, 8192, 4);
    let a_wave = b.bind_i32(lane.wave() * (a_bytes / 4) as i32);
    let b_wave = b.bind_i32(lane.wave() * (cta_n * 128 / 4) as i32);
    // buffer descriptors over A (M K bytes) and the preshuffled B (N K bytes)
    let rsrc_a = match cfg.conv {
        // conv: the input tensor NB DI H W C bf16
        Some(_) => {
            let ki = |b: &mut FnBuilder, o: u32| b.let_(Op::KernargI32 { offset: o });
            let (nb, di, h, w, c) = (ki(b, 96), ki(b, 100), ki(b, 104), ki(b, 108), ki(b, 112));
            let n = b.bind_i32(ev(nb) * ev(di) * ev(h) * ev(w) * ev(c) * 2);
            let n = b.read_first_lane(n);
            b.buffer_desc_dyn(ptr_a, n)
        }
        None => { let n = b.bind_i32(ev(m_v) * ev(ka_v)); let n = b.read_first_lane(n); b.buffer_desc_dyn(ptr_a, n) }
    };
    let rsrc_b = match &moe_cx {
        None => { let n = b.bind_i32(ev(n_v) * ev(k_v)); let n = b.read_first_lane(n); b.buffer_desc_dyn(ptr_b, n) }
        // MoE: one descriptor over the tile's EXPERT, its base moved by a 64-bit `ptr_mad` (E n K bytes passes 4 GiB at production sizes,
        // e.g. E 256, n 4096, K 7168 fp8 = 7.5 GB); num_records = n K of one expert
        Some(mc) => {
            let nk = { let v = b.bind_i32(ev(n_v) * ev(k_v)); b.read_first_lane(v) };
            let pe = b.ptr_mad(ptr_b, mc.e, nk);
            b.buffer_desc_dyn(pe, nk)
        }
    };
    let (rsrc_sfa, rsrc_sfb) = match cfg.scales {
        Scales::Mx128 => (
            { let n = b.bind_i32(ev(m_v) * ev(kb_v)); let n = b.read_first_lane(n); b.buffer_desc_dyn(ptr_sfa, n) },
            { let n = b.bind_i32((ev(n_v) >> 7) * ev(kb_v)); let n = b.read_first_lane(n); b.buffer_desc_dyn(ptr_sfb, n) },
        ),
        Scales::RowCol if cfg.per_tensor => {
            let n = b.c_i32(4);
            let n = b.read_first_lane(n);
            (b.buffer_desc_dyn(ptr_sfa, n), b.buffer_desc_dyn(ptr_sfb, n))
        }
        Scales::RowCol => (
            { let n = b.bind_i32(ev(m_v) << 2); let n = b.read_first_lane(n); b.buffer_desc_dyn(ptr_sfa, n) },
            // MoE: the weight scales of every expert [E, n]
            if let Some(mc) = &moe_cx {
                // MoE: the tile's expert's n weight scales (base moved per expert)
                let n = b.bind_i32(ev(n_v) << 2); let n = b.read_first_lane(n);
                let pe = b.ptr_mad(ptr_sfb, mc.e, n);
                b.buffer_desc_dyn(pe, n)
            } else {
                let n = b.bind_i32(ev(n_v) << 2); let n = b.read_first_lane(n); b.buffer_desc_dyn(ptr_sfb, n)
            },
        ),
        // unused (no scales)
        Scales::None => (rsrc_a, rsrc_b),
        Scales::Block128 => (
            { let n = b.bind_i32(ev(m_v) * ev(kb_v) << 2); let n = b.read_first_lane(n); b.buffer_desc_dyn(ptr_sfa, n) },
            if let Some(mc) = &moe_cx {
                // MoE: the tile's expert's [n / 128, KT] block scales (base moved per expert)
                let n = b.bind_i32((ev(n_v) >> 7) * ev(kb_v) << 2); let n = b.read_first_lane(n);
                let pe = b.ptr_mad(ptr_sfb, mc.e, n);
                b.buffer_desc_dyn(pe, n)
            } else {
                let n = b.bind_i32((ev(n_v) >> 7) * ev(kb_v) << 2); let n = b.read_first_lane(n); b.buffer_desc_dyn(ptr_sfb, n)
            },
        ),
        // [rows][K / 32] bytes = rows * K bytes / 64 for fp4 (K bytes = K / 2)
        Scales::Mx32 => (
            { let n = b.bind_i32(ev(m_v) * ev(kb_v) << 3); let n = b.read_first_lane(n); b.buffer_desc_dyn(ptr_sfa, n) },
            if let Some(mc) = &moe_cx {
                // MoE: the tile's expert's [n][8 KT] scale bytes (base moved per expert)
                let n = b.bind_i32(ev(n_v) * ev(kb_v) << 3); let n = b.read_first_lane(n);
                let pe = b.ptr_mad(ptr_sfb, mc.e, n);
                b.buffer_desc_dyn(pe, n)
            } else {
                let n = b.bind_i32(ev(n_v) * ev(kb_v) << 3); let n = b.read_first_lane(n); b.buffer_desc_dyn(ptr_sfb, n)
            },
        ),
    };
    // split-K: this split's first tile zb = z KS (A advances 128 bytes per tile, the preshuffled B 16 x 128)
    let (ks_v, zb) = if cfg.split_k {
        assert!(matches!(cfg.scales, Scales::RowCol | Scales::None | Scales::Block128), "split-K: scales applied per tile or in the reduce");
        let ks = b.let_(Op::KernargI32 { offset: 88 });
        let z = b.let_(Op::WorkgroupIdZ);
        let zb = { let v = b.bind_i32(ev(z) * ev(ks)); b.read_first_lane(v) };
        (ks, Some(zb))
    } else {
        (kb_v, None)
    };
    let (a_org, b_org) = match zb {
        // MoE: rows are gathered (A origin 0); B = the tile's expert (stage 1: the CTA's gate rows bx 128, up rows + n_out)
        None if moe_cx.is_some() => {
            let mc = moe_cx.as_ref().unwrap();
            let _ = mc; // the expert is in the descriptor base (rsrc_b)
            (b.c_i32(0), b.bind_i32((ev(bx) << lg_n) * ev(k_v)))
        }
        None if cfg.conv.is_some() => (b.c_i32(0), b.bind_i32((ev(bx) << lg_n) * ev(k_v))),
        None => (b.bind_i32((ev(by) << lg_m) * ev(ka_v)), b.bind_i32((ev(bx) << lg_n) * ev(k_v))),
        Some(zb) => (b.bind_i32((ev(by) << lg_m) * ev(k_v) + (ev(zb) << 7)), b.bind_i32((ev(bx) << lg_n) * ev(k_v) + (ev(zb) << 11))),
    };

    let panel = panel_sh.map(|panel_sh| {
    // ---- the scale panel (written in the pipeline prologue, `prologue_stores`) ----
    assert!(cta_m == 256 && cta_n == 256, "scale panel: 256 x 256 CTA");
    let sh = panel_sh;
    // this lane's A offset in the transposed panel: 128 wm + 8 (lane % 16)
    let tr = b.bind_i32((ev(wm) << 7) + ((lane.lane() & 15) << 3));
    // the VGPR scale path: A by 4 passes of buffer_load_dwordx4 + in-register 4x4 byte transposes (dpp quad_perm + v_perm) + ds_write, B one dword
    sfa_transpose_proof();
    // A: lane g = 64 w + lane (+ 256 p per pass) holds 16 M rows (chunk c = g % 16) of K group g / 16: voffset (K group) M + m_cta + 16 c
    let g0 = b.bind_i32(lane.tid());
    let voff = b.bind_i32((ev(g0).lshr(4)) * ev(m_v) + (ev(by) << 8) + ((ev(g0) & 15) << 4));
    let j = b.bind_i32(ev(g0) & 3);
    // v_perm selectors (byte k of the result = sel byte k: 0..3 S1, 4..7 S0): step 1 [0,4,2,6] / [5,1,7,3], step 2 [0,1,4,5] / [6,7,2,3]
    let sel1 = b.bind_i32((ev(j) & 1).eq(0).select_i32(0x0602_0400, 0x0307_0105));
    let sel2 = b.bind_i32(ev(j).lt(2).select_i32(0x0504_0100, 0x0302_0706));
    // destination of dword d of pass p: panel + 256 kb + 128 wm + 8 (4 d + j) + 4 a  (wm = c / 8, a = c / 4 % 2) = base + 4096 p + 32 d
    let c = b.bind_i32(ev(g0) & 15);
    let dst = b.bind_i32(((ev(g0).lshr(4)) << 8) + ((ev(c).lshr(3)) << 7) + (ev(j) << 3) + (((ev(c).lshr(2)) & 1) << 2) + sh.base as i32);
    // B: thread tid % 128 -> (blk, kb) = (tid / 64 % 2, tid % 64): the dword holding byte (2 bx + blk) KB + kb, shifted to byte 0
    let blk = b.bind_i32((ev(tid).lshr(6)) & 1);
    let kbt = b.bind_i32(ev(tid) & 63);
    let boff = b.bind_i32(((ev(bx) << 1) + ev(blk)) * ev(kb_v) + ev(kbt));
    let bvo = b.bind_i32(ev(boff).lshr(2) << 2);
    let bsh = b.bind_i32((ev(boff) & 3) << 3);
    let bdst = b.bind_i32(((ev(tid) & 127) << 2) + (sh.base + PANEL_KB * 256) as i32);
    let pso: Vec<ValueId> = (0..4).map(|p| { let v = b.bind_i32(ev(m_v) * (16 * p)); b.read_first_lane(v) }).collect();
    let b_dw = b.bind_i32(ev(wn) * (4 * PANEL_KB as i32));
    Panel { sh, b_base: PANEL_KB * 256, tr, b_dw, sfa: SfaJob { voff, pso, sel1, sel2, dst, bvo, bsh, bdst } }
    });

    let mut pipe = Pipeline::new(Cadence::RegStaged, ks_v, k_v, 128, if cfg.split_k { Ext::ka(88) } else { Ext::ka(84) });
    // two tiles per pass, static slots; tile 1 staged after the scale-panel stores with its DMAs in flight at the prologue barrier
    pipe.late_tile1_vm = (a_dma.issues.len() + b_dma.issues.len()) as u8;
    pipe.scalar_exit = true;
    if cfg.depth3 {
        pipe.rs_depth = 3;
        if cfg.scales == Scales::Block128 {
            // (the two scale DMAs per wave and stage)
            pipe.late_tile1_vm += 2;
        }
    }
    // operand read partitions (the wave offset added to the bases)
    let mut a_rd = copy_a.partition_lds(b, &ctx, &s_lane, &a_s, &a_view, &a_place);
    let mut b_rd = copy_b.partition_lds(b, &ctx, &s_lane, &b_s, &view, &b_place);
    let a_wo = ctx.lower(&s_wm.mul_c(wave_m / 16 * a_blk));
    let b_wo = ctx.lower(&s_wn.mul_c(wave_n / 16 * 2048));
    a_rd.bases = a_rd.bases.iter().map(|x| b.let_i32(x + &a_wo)).collect();
    b_rd.bases = b_rd.bases.iter().map(|x| b.let_i32(x + &b_wo)).collect();
    let zero4 = b.frag_zero(DType::Frag4);
    let unit = matches!(cfg.scales, Scales::RowCol | Scales::None | Scales::Block128).then(|| (b.bind_i32(lane.lane().lshr(6) + 0x7F7F_7F7F), b.bind_i32(lane.lane().lshr(6) + 0x7F)));
    // 1 x 32 scales: per m block i the row (CTA + wave + 16 i + lane % 16) of [rows][8 KT] bytes, the dword of K groups 4 (g / 2).. (g = lane / 16)
    // fp32 block scales: per m block the row's [KT] floats, the wave's 128-column block of B (wave_n <= 128, aligned)
    let bs128 = (cfg.scales == Scales::Block128).then(|| {
        let r16 = b.bind_i32(lane.lane() & 15);
        let rowb = b.bind_i32(ev(kb_v) << 2);
        if let Some(mc) = &moe_cx {
            // MoE: A scale rows of the gathered rows ([A rows][KT] floats); B scales [E, n / 128, KT] of the tile's expert. Stage 1: the
            // wave's 128 interleaved rows = 64 gate rows of gate block gb and the matching 64 up rows of block n_out / 128 + gb (two words,
            // n block j uses word j % 2)
            let so = { let v = b.bind_i32(ev(mc.row0) << 2); b.read_first_lane(v) };
            let voff_a = (0..wave_m / 16)
                .map(|i| {
                    let vo = b.bind_i32((ev(wm) * (wave_m as i32) + (16 * i) as i32 + ev(r16)) << 2);
                    let sid = b.buffer_load_f32(mc.rsrc_sid, vo, so, 0);
                    let sid = b.bitcast_i32(sid);
                    let tok = b.bind_i32(ev(sid) & 0xff_ffff);
                    let row = match mc.st {
                        MoeStage::Up { .. } => b.bind_i32(ev(tok).min(ev(mc.mtok) - 1)),
                        MoeStage::Down | MoeStage::DownH => b.bind_i32((ev(tok) * ev(mc.topk) + ev(sid).lshr(24)).min(ev(m_v) - 1).max(0)),
                    };
                    b.bind_i32(ev(row) * ev(rowb))
                })
                .collect();
            b.wait_vm(0);
            let nbe = b.c_i32(0); // the expert is in the descriptor base (rsrc_sfb)
            let voff_b = if matches!(mc.st, MoeStage::Up { .. }) {
                let gb = b.bind_i32(((ev(bx) << 1) + ev(wn)).lshr(1));
                vec![b.bind_i32((ev(nbe) + ev(gb)) * ev(rowb)), b.bind_i32((ev(nbe) + ev(mc.nout).lshr(7) + ev(gb)) * ev(rowb))]
            } else {
                let nb = b.bind_i32(((ev(bx) << lg_n) + ev(wn) * (wave_n as i32)).lshr(7));
                vec![b.bind_i32((ev(nbe) + ev(nb)) * ev(rowb))]
            };
            return Mx32 { voff_a, voff_b, shift: r16 , shuf: None };
        }
        let a0 = b.bind_i32((ev(by) << lg_m) + ev(wm) * (wave_m as i32) + ev(r16));
        let voff_a = (0..wave_m / 16).map(|i| b.bind_i32((ev(a0) + (16 * i) as i32) * ev(rowb))).collect();
        let nb = b.bind_i32(((ev(bx) << lg_n) + ev(wn) * (wave_n as i32)).lshr(7));
        let voff_b = vec![b.bind_i32(ev(nb) * ev(rowb))];
        Mx32 { voff_a, voff_b, shift: r16 , shuf: None }
    });
    let mx32 = (cfg.scales == Scales::Mx32).then(|| {
        let r16 = b.bind_i32(lane.lane() & 15);
        let rowb = b.bind_i32(ev(kb_v) << 3);
        if let Some(mc) = &moe_cx {
            // MoE: A scale rows = the gathered rows (token, or token topk + slot), B rows of the tile's expert (descriptor base)
            let so = { let v = b.bind_i32(ev(mc.row0) << 2); b.read_first_lane(v) };
            let voff_a = (0..wave_m / 16)
                .map(|i| {
                    let vo = b.bind_i32((ev(wm) * (wave_m as i32) + (16 * i) as i32 + ev(r16)) << 2);
                    let sid = b.buffer_load_f32(mc.rsrc_sid, vo, so, 0);
                    let sid = b.bitcast_i32(sid);
                    let tok = b.bind_i32(ev(sid) & 0xff_ffff);
                    let row = match mc.st {
                        MoeStage::Up { .. } => b.bind_i32(ev(tok).min(ev(mc.mtok) - 1)),
                        MoeStage::Down | MoeStage::DownH => b.bind_i32((ev(tok) * ev(mc.topk) + ev(sid).lshr(24)).min(ev(m_v) - 1).max(0)),
                    };
                    b.bind_i32(ev(row) * ev(rowb))
                })
                .collect();
            b.wait_vm(0);
            let b0 = b.bind_i32((ev(bx) << lg_n) + ev(wn) * (wave_n as i32) + ev(r16));
            let voff_b = (0..wave_n / 16).map(|j| b.bind_i32((ev(b0) + (16 * j) as i32) * ev(rowb))).collect();
            let gg = b.bind_i32(lane.lane().lshr(4));
            let shift = b.bind_i32(ev(gg) + ((ev(gg) + 4) << 8) + 0x0C0C_0000);
            return Mx32 { voff_a, voff_b, shift, shuf: None };
        }
        let a0 = b.bind_i32((ev(by) << lg_m) + ev(wm) * (wave_m as i32) + ev(r16));
        let b0 = b.bind_i32((ev(bx) << lg_n) + ev(wn) * (wave_n as i32) + ev(r16));
        let (voff_a, voff_b) = if cfg.scale_shuf {
            // shuffled: block (row / 16) of [rows / 16][KT / 2][64][4]: (row / 16) KT 128 + 4 lane (KT even)
            let blk = b.bind_i32(ev(kb_v) << 7);
            let l4 = b.bind_i32(lane.lane() << 2);
            let ra = b.bind_i32(((ev(by) << lg_m) + ev(wm) * (wave_m as i32)).lshr(4));
            let rb = b.bind_i32(((ev(bx) << lg_n) + ev(wn) * (wave_n as i32)).lshr(4));
            (
                (0..wave_m / 16).map(|i| b.bind_i32((ev(ra) + i as i32) * ev(blk) + ev(l4))).collect::<Vec<_>>(),
                (0..wave_n / 16).map(|j| b.bind_i32((ev(rb) + j as i32) * ev(blk) + ev(l4))).collect::<Vec<_>>(),
            )
        } else {
            (
                (0..wave_m / 16).map(|i| b.bind_i32((ev(a0) + (16 * i) as i32) * ev(rowb))).collect(),
                (0..wave_n / 16).map(|j| b.bind_i32((ev(b0) + (16 * j) as i32) * ev(rowb))).collect(),
            )
        };
        // v_perm selector: byte 0 = byte g of the low word (S1 bytes 0..3), byte 1 = byte g of the high word (S0 bytes 4..7), bytes 2, 3 = 0
        let gg = b.bind_i32(lane.lane().lshr(4));
        let shift = b.bind_i32(ev(gg) + ((ev(gg) + 4) << 8) + 0x0C0C_0000);
        Mx32 { voff_a, voff_b, shift, shuf: cfg.scale_shuf.then(|| std::cell::RefCell::new(None)) }
    });
    // depth3 + Block128: the scale DMA voffsets (wave 0: A row of the CTA per lane, lane 0 the B block; others out of range) and the lane's
    // read offset (4 (wm wave_m + lane % 16), + 64 per m block)
    let sc_dma = (cfg.depth3 && cfg.scales == Scales::Block128).then(|| {
        assert!(cta_m == 64 && cta_n == 128, "depth3 blockscale: 64 x 128 CTAs");
        let big = b.c_i32(0x7fff_fff0);
        let w0 = b.bind_cond(lane.wave().eq(0));
        let va = { let v = b.bind_i32(((ev(by) << lg_m) + lane.lane()).min(ev(m_v) - 1) * ev(kb_v) << 2); b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(w0), a: Opnd::Val(v), b: Opnd::Val(big) }) };
        let l0 = b.bind_cond((lane.wave() + lane.lane()).eq(0));
        let vb = { let v = b.bind_i32(((ev(bx) << lg_n).lshr(7)) * ev(kb_v) << 2); b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(l0), a: Opnd::Val(v), b: Opnd::Val(big) }) };
        let wofs = { let v = b.bind_i32(lane.wave() << 9); b.read_first_lane(v) };
        let rd = b.bind_i32((ev(wm) * (wave_m as i32) + (lane.lane() & 15)) << 2);
        (sc_s.clone(), va, vb, wofs, rd)
    });
    let body = Body {
        params: cfg.body_params(), kb_v: ks_v, panel, mma: mma.clone(), a_s, b_s, a_dma, b_dma, a_org, b_org, a_wave, b_wave, rsrc_a, rsrc_b, rsrc_sfa, rsrc_sfb,
        a_rd_x: (1..nimg).map(|k| { let mut h = a_rd.clone(); h.bases = h.bases.iter().map(|x| b.let_i32(x.clone() + (k * a_bytes) as i32)).collect(); h }).collect(),
        a_hi: a_bytes as i32,
        w4_shift: (cfg.elem == Elem::Bf16W4).then(|| b.bind_i32((lane.lane().lshr(5)) << 3)),
        a_rd, b_rd, zero4, acc: Vec::new(), staged: None, rv: None, npre_done: 0, stamp: stamp_ctx, unit, mx32, bs128, conv: conv_rt.map(|c| Box::new(c) as Box<dyn amdgpu_stage::gemm::body::ConvSource>), pending: Default::default(), zb, odd_by: std::cell::Cell::new(0), tail_shifted: std::cell::Cell::new(false), sc_dma,
    };
        // MoE: everything after this runs only for tiles below num_valid_ids[0] (closed at the end of the epilogue)
    if let Some(mc) = &moe_cx {
        b.begin_if(Opnd::Val(mc.live));
    }
    GemmCx { cfg, moe: moe_cx, rsrc_sfa_e: Some((rsrc_sfa, rsrc_sfb)), body, pipe: Some(pipe), ctx, mma, epi, lane, s_lane, s_n, s_wm, s_wn, ptr_c, m_v, n_v, bx, by, wm, wn, lg_m, lg_n, wave_m, wave_n, stamp_ctx, tk }
    }
    fn pipeline(&mut self, b: &mut FnBuilder, cx: &mut GemmCx) -> Pipeline {
        if let Some((buf, rec, _)) = cx.stamp_ctx {
            let st = b.memtime_lo();
            b.stamp_store(buf, rec, 1, st); // slot 1: setup done, entering the pipeline (prologue)
        }
        cx.pipe.take().expect("the pipeline is taken once")
    }
    fn stage(&mut self, b: &mut FnBuilder, cx: &mut GemmCx, role: Role, key0: &I32, slot: &SlotIdx) {
        PipeBody::stage(&mut cx.body, b, role, key0, slot)
    }
    fn prologue_stores(&mut self, b: &mut FnBuilder, cx: &mut GemmCx, role: Role) {
        PipeBody::prologue_stores(&mut cx.body, b, role)
    }
    fn prologue_compute(&mut self, b: &mut FnBuilder, cx: &mut GemmCx, role: Role, t0: &TileIt) {
        PipeBody::prologue_compute(&mut cx.body, b, role, t0)
    }
    fn loop_state(&mut self, b: &mut FnBuilder, cx: &mut GemmCx) -> Vec<(DType, Opnd)> {
        PipeBody::loop_state(&mut cx.body, b)
    }
    fn bind_state(&mut self, cx: &mut GemmCx, vars: &[VarId]) {
        PipeBody::bind_state(&mut cx.body, vars)
    }
    fn m_phase(&mut self, b: &mut FnBuilder, cx: &mut GemmCx, role: Role, t: &TileIt) {
        PipeBody::m_phase(&mut cx.body, b, role, t)
    }
    fn pre_sync(&mut self, b: &mut FnBuilder, cx: &mut GemmCx, role: Role, t: &TileIt) {
        PipeBody::pre_sync(&mut cx.body, b, role, t)
    }
    fn post_wait(&mut self, b: &mut FnBuilder, cx: &mut GemmCx, role: Role, t: &TileIt) {
        PipeBody::post_wait(&mut cx.body, b, role, t)
    }
    fn epilogue(&mut self, b: &mut FnBuilder, cx: &mut GemmCx) {
        let cx_tk = cx.tk.clone();
        let GemmCx { cfg, body, ctx, mma, epi, lane, s_lane, s_n, s_wm, s_wn, ptr_c, m_v, n_v, bx, by, wm, wn, lg_m, lg_n, wave_m, wave_n, stamp_ctx, .. } = cx;
        let (cfg, lane, ptr_c, m_v, n_v, bx, by, wm, wn, lg_m, lg_n, wave_m, wave_n, stamp_ctx) = (*cfg, *lane, *ptr_c, *m_v, *n_v, *bx, *by, *wm, *wn, *lg_m, *lg_n, *wave_m, *wave_n, *stamp_ctx);
        let (ctx, mma, s_lane, s_n, s_wm, s_wn) = (&*ctx, &*mma, &*s_lane, &*s_n, &*s_wm, &*s_wn);
    if let Some((buf, rec, _)) = stamp_ctx {
        let st = b.memtime_lo();
        b.stamp_store(buf, rec, 3, st); // slot 3: the tile loop and the peeled last tile are done (epilogue next)
    }
    for v in body.acc.iter().take(cfg.acc_agpr) {
        b.pin_agpr(*v);
    }
    if let Some((bv, av, _, _)) = &body.rv {
        for v in bv.iter().chain(av.iter()) {
            b.pin_agpr(*v);
        }
    }
    if cfg.elem == Elem::I8 {
        // int8: the i32 accumulators to f32 in place (every epilogue below - split-K partials, the scaled LDS epilogue - is the fp32 one)
        for acc in &body.acc {
            for r in 0..4u8 {
                let x = b.frag4_lane(Opnd::Var(*acc), r);
                let x = b.bitcast_i32(x);
                let f = b.un(amdgpu_dsl::UnOp::CvtF32I32, x);
                b.var_set_lane(*acc, r, f);
            }
        }
    }
    if let Some(mc) = cx.moe.clone() {
        let (rsa, rsb) = cx.rsrc_sfa_e.expect("descriptors");
        moe_epilogue(b, &ctx, &mma, &body.acc, s_lane, &mc, cfg, rsa, rsb, ptr_c, m_v, n_v, bx, wm, wn, wave_m, wave_n);
        b.end_if();
        return;
    }
    if cfg.split_k {
        // split-K: the fp32 partial C straight from the accumulator layout (lane = row m, 4 consecutive n: one 16-byte store) into
        // ptr_c[z][m][n]
        let z = b.let_(Op::WorkgroupIdZ);
        let c_blk = make_layout(tup![16, 16], tup![s_n.mul_c(4), 4]);
        let c_plan = IssuePlan::new(mma.c_tv(), 4);
        // slices of MP = ceil(M / cta_m) cta_m rows (kernarg 92, from the host; a partial last row tile writes its padding rows inside its own
        // slice)
        let mp = b.let_(Op::KernargI32 { offset: 92 });
        let c_org = b.bind_i32(((((ev(by) << lg_m) * ev(n_v)) + (ev(bx) << lg_n)) << 2) + ((ev(z) * ev(mp) * ev(n_v)) << 2));
        let cb = b.ptr_add(ptr_c, c_org);
        let mut exprs = Vec::new();
        for i in 0..wave_m / 16 {
            for j in 0..wave_n / 16 {
                let org = s_wm.mul_c(wave_m).add_c(16 * i).mul(s_n).mul_c(4).add(&s_wn.mul_c(wave_n).add_c(16 * j).mul_c(4));
                exprs.extend(c_plan.offsets(&c_blk, s_lane, &org, &ctx.dyns));
            }
        }
        let plan = layout_to_offsets(&exprs, &Swizzle::none(), ImmRange::GLOBAL, &ctx.dyns);
        if cfg.lastc {
            lastc_epilogue(b, &plan, ctx, &body.acc, (body.rsrc_sfa, body.rsrc_sfb), cfg, c_org, z, mp, m_v, n_v, bx, by, wm, wn, lg_m, lg_n, wave_m, wave_n, ptr_c);
            return;
        }
        let bases: Vec<ValueId> = plan.bases.iter().map(|e| {
            let o = b.bind_i32(ctx.lower(e));
            b.ptr_add(cb, o)
        }).collect();
        // only the 16-row blocks that hold real rows (small M: a 64-row tile of an M = 1 GEMV writes one block, not four): a wave-uniform
        // branch per block of the wave (the reduce reads rows < M only)
        let nj = (wave_n / 16) as usize;
        for i in 0..(wave_m / 16) as usize {
            let r0 = b.bind_i32((ev(by) << lg_m) + ev(wm) * wave_m as i32 + 16 * i as i32);
            let live = b.bind_cond(ev(r0).lt(ev(m_v)));
            b.begin_if(Opnd::Val(live));
            for j in 0..nj {
                let n = i * nj + j;
                let (bi, imm) = plan.issues[n];
                b.store_f32x4(bases[bi], imm as u32, Opnd::Var(body.acc[n]));
            }
            b.end_if();
        }
        return;
    }
    if let Some(epi) = &epi {
        let rowcol = (cfg.scales == Scales::RowCol).then_some((body.rsrc_sfa, body.rsrc_sfb));
        lds_epilogue(b, &ctx, &mma, epi, &body.acc, &lane, &s_lane, &s_n, ptr_c, m_v, n_v, bx, by, wm, wn, lg_m, lg_n, wave_m, wave_n, cfg.epi_phases, rowcol, cfg.per_tensor);
        if let Some((buf, rec, _)) = stamp_ctx {
            let st = b.memtime_lo();
            b.stamp_store(buf, rec, 4, st); // slot 4: epilogue stores issued
        }
        return;
    }
    // ---- epilogue: the (swapped) C layout of the atom x C row-major (bf16); 4 consecutive N per lane = one 8-byte store ----
    let c_blk = make_layout(tup![16, 16], tup![s_n.mul_c(2), 2]); // element m + 16 n
    let c_plan = IssuePlan::new(mma.c_tv(), 4);
    let c_org = b.bind_i32((((ev(by) << lg_m) * ev(n_v)) + (ev(bx) << lg_n)) << 1);
    let cb = b.ptr_add(ptr_c, c_org);
    let mut exprs = Vec::new();
    for i in 0..wave_m / 16 {
        for j in 0..wave_n / 16 {
            let org = s_wm.mul_c(wave_m).add_c(16 * i).mul(&s_n).mul_c(2).add(&s_wn.mul_c(wave_n).add_c(16 * j).mul_c(2));
            exprs.extend(c_plan.offsets(&c_blk, &s_lane, &org, &ctx.dyns));
        }
    }
    let plan = layout_to_offsets(&exprs, &Swizzle::none(), ImmRange::GLOBAL, &ctx.dyns);
    let bases: Vec<ValueId> = plan.bases.iter().map(|e| {
        let o = b.bind_i32(ctx.lower(e));
        b.ptr_add(cb, o)
    }).collect();
    for (n, acc) in body.acc.iter().enumerate() {
        let v: Vec<ValueId> = (0..4u8).map(|r| {
            let l = b.frag4_lane(Opnd::Var(*acc), r);
            b.bitcast_f32(l)
        }).collect();
        let w0 = b.cvt_pk_bf16_f32(v[0], v[1]);
        let w1 = b.cvt_pk_bf16_f32(v[2], v[3]);
        let f2 = b.frag2(w0, w1);
        let (bi, imm) = plan.issues[n];
        b.store_f32x2(bases[bi], imm as u32, f2);
    }
    if let Some((buf, rec, _)) = stamp_ctx {
        let st = b.memtime_lo();
        b.stamp_store(buf, rec, 4, st);
    }
    }
}

/// The `lastc` split-K epilogue (the fused-combine pattern of `pa_decode16_*_c`, verified by `buf_last_arriver`): every WAVE writes its fp32
/// partial blocks (real rows only, write-through `sc1` buffer stores) into slice z of `ptr_c`, releases (`fence_release_lite`) and takes one
/// ticket on `ptr_cnt[by NT + bx]` (lane 0; the other lanes' offsets are out of range). The 4 S arrivals of an output tile see tickets
/// 0 .. 4 S - 1; the wave with ticket 4 S - 1 acquires (`fence_acquire_lite`, coherent `sc1` loads), sums the S slices of the tile's real rows,
/// applies the per-token x per-channel scales, stores bf16 to `ptr_out` and resets the counter. No barrier: the waves arrive independently.
#[allow(clippy::too_many_arguments)]
fn lastc_epilogue(b: &mut FnBuilder, plan: &amdgpu_layout::OffsetPlan, ctx: &Ctx, acc: &[VarId], scl: (ValueId, ValueId), cfg: Cfg, c_org: ValueId, z: ValueId, mp: ValueId, m_v: ValueId, n_v: ValueId, bx: ValueId, by: ValueId, wm: ValueId, wn: ValueId, lg_m: i32, lg_n: i32, wave_m: i64, wave_n: i64, ptr_c: ValueId) {
    let sc1 = amdgpu_dsl::CachePolicy { sc0: false, sc1: true, nt: false };
    let s_v = b.let_(Op::KernargI32 { offset: 36 });
    let c0 = b.c_i32(0);
    let big = b.c_i32(0x7fff_fff0);
    let r_ws = { let v = b.bind_i32(ev(s_v) * ev(mp) * ev(n_v) * 4); let v = b.read_first_lane(v); b.buffer_desc_dyn(ptr_c, v) };
    let nj = (wave_n / 16) as usize;
    for i in 0..(wave_m / 16) as usize {
        let r0 = b.bind_i32((ev(by) << lg_m) + ev(wm) * wave_m as i32 + 16 * i as i32);
        let live = b.bind_cond(ev(r0).lt(ev(m_v)));
        b.begin_if(Opnd::Val(live));
        let vos: Vec<ValueId> = (0..nj).map(|j| { let (bi, _) = plan.issues[i * nj + j]; b.bind_i32(ev(c_org) + ctx.lower(&plan.bases[bi])) }).collect();
        b.buf_produce("ws", z);
        for j in 0..nj {
            let n = i * nj + j;
            let (_, imm) = plan.issues[n];
            b.buffer_store_c(r_ws, Some(Opnd::Val(vos[j])), c0, imm as u32, Opnd::Var(acc[n]), amdgpu_dsl::MemTy::B128, false, sc1);
        }
        b.tile_end();
        b.end_if();
    }
    // ---- release, the wave's ticket, the last arriving wave of the output tile combines ----
    b.fence_release_lite();
    let tid = b.let_(Op::WorkitemId);
    let lane = b.bind_i32(ev(tid) & 63);
    let is0 = b.bind_cond(ev(lane).lt(1));
    let p_cnt = b.let_(Op::KernargPtr { offset: 104 });
    let nt = b.bind_i32(ev(n_v).lshr(lg_n));
    let mtiles = b.bind_i32(ev(mp).lshr(lg_m));
    // one counter per (output tile, wave): the S waves that own the same wave_m x wave_n block arrive on it
    let wave = b.bind_i32(ev(wm) + (ev(wn) << 1));
    let gofs = b.bind_i32((((ev(by) * ev(nt) + ev(bx)) << 2) + ev(wave)) << 2);
    let ncnt = { let v = b.bind_i32(ev(mtiles) * ev(nt) * 16); b.read_first_lane(v) };
    let r_cnt = b.buffer_desc_dyn(p_cnt, ncnt);
    let cvo = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(is0), a: Opnd::Val(gofs), b: Opnd::Val(big) });
    let c1 = b.c_i32(1);
    let old = b.buffer_atomic_add_ret(r_cnt, cvo, c0, 0, c1);
    let last = b.bind_i32(ev(s_v) - 1);
    let is_last = b.let_(Op::CmpGeI32(Opnd::Val(old), Opnd::Val(last)));
    let any = b.any_lane(is_last);
    b.wg_winner(any, &[2], "last arriver: the S waves owning one wave block of an output tile (one per split CTA) take tickets 0..S-1 on one counter; only ticket S - 1 runs the combine of that block");
    b.begin_if(Opnd::Val(any));
    {
        b.fence_acquire_lite();
        let zf = b.c_f32(0.0);
        // the wave's wave_m x wave_n block; lane l: columns 4 (l % 16) .. + 3 of it, rows l / 16 + 4 q
        let cg = b.bind_i32(ev(lane) & 15);
        let rr = b.bind_i32(ev(lane).lshr(4));
        let col = b.bind_i32((ev(bx) << lg_n) + ev(wn) * wave_n as i32 + (ev(cg) << 2));
        let mn = { let v = b.bind_i32(ev(m_v) * ev(n_v)); b.read_first_lane(v) };
        let p_out = b.let_(Op::KernargPtr { offset: 96 });
        let r_out = { let v = b.bind_i32(ev(mn) << 1); let v = b.read_first_lane(v); b.buffer_desc_dyn(p_out, v) };
        let slice4 = { let v = b.bind_i32(ev(mp) * ev(n_v) * 4); b.read_first_lane(v) };
        let rowcol = cfg.scales == Scales::RowCol;
        let sb = rowcol.then(|| { let o = b.bind_i32(ev(col) << 2); b.buffer_load_b128(scl.1, o, 0i32, 0) });
        // the row quads of the block that hold real rows: nq = ceil((M - row0) / 4), at most wave_m / 4 (the winner only runs with row0 < M)
        let row0 = b.bind_i32((ev(by) << lg_m) + ev(wm) * wave_m as i32);
        let rows = b.bind_i32((ev(m_v) - ev(row0)).min(wave_m as i32).max(1));
        let nq = { let v = b.bind_i32((ev(rows) + 3).lshr(2)); b.read_first_lane(v) };
        let qv = b.begin_loop(&[(DType::I32, Opnd::I32(0))]);
        b.assume_trusted(qv[0], Ext::c(0), Ext::c(wave_m / 4 - 1), "row-quad counter: starts at 0, the loop exits once q + 1 >= nq <= wave_m / 4");
        {
            let row = b.bind_i32(ev(row0) + (amdgpu_dsl::varx(qv[0]) << 2) + ev(rr));
            let rlive = b.bind_cond(ev(row).lt(ev(m_v)));
            let e0 = b.bind_i32((ev(row) * ev(n_v) + ev(col)) << 2);
            let vo = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(rlive), a: Opnd::Val(e0), b: Opnd::Val(big) });
            let st: Vec<(DType, Opnd)> = std::iter::once((DType::I32, Opnd::I32(0))).chain((0..4).map(|_| (DType::F32, Opnd::Val(zf)))).collect();
            let v = b.begin_loop(&st);
            let si = amdgpu_dsl::varx(v[0]);
            b.assume_trusted(v[0], Ext::c(0), Ext::sym("S") - Ext::c(1), "split counter: starts at 0, steps by 4, the loop exits once s + 4 >= S");
            // 4 splits per iteration, their loads in flight together (a split past S reads out of range: 0)
            let xs: Vec<ValueId> = (0..4i32)
                .map(|u| {
                    let su = b.bind_i32(si.clone() + u);
                    let lv = b.bind_cond(ev(su).lt(ev(s_v)));
                    let vu = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(lv), a: Opnd::Val(vo), b: Opnd::Val(big) });
                    let so = { let x = b.bind_i32(ev(su) * ev(slice4)); b.read_first_lane(x) };
                    b.buf_consume("ws", Opnd::Val(su));
                    let x = b.buffer_load_c(r_ws, Some(Opnd::Val(vu)), so, 0, amdgpu_dsl::MemTy::B128, sc1);
                    b.tile_end();
                    x
                })
                .collect();
            for r in 0..4u8 {
                let mut a = Opnd::Var(v[1 + r as usize]);
                for x in &xs {
                    let f = b.frag4_lane(Opnd::Val(*x), r);
                    let f = b.bitcast_f32(f);
                    a = Opnd::Val(b.add_f32(a, f));
                }
                b.set_var(v[1 + r as usize], a);
            }
            let n1 = b.bind_i32(si + 4);
            b.set_var(v[0], n1);
            let dn = b.bind_cond(ev(n1).ge(ev(s_v)));
            b.break_if(dn);
            b.end_loop();
            let mut y: Vec<ValueId> = (0..4).map(|r| { let z0 = b.c_f32(0.0); b.add_f32(Opnd::Var(v[1 + r]), z0) }).collect();
            if let Some(sb) = sb {
                let ra = b.bind_i32(ev(row) << 2);
                let ra = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(rlive), a: Opnd::Val(ra), b: Opnd::Val(big) });
                let sa = b.buffer_load_f32(scl.0, ra, 0i32, 0);
                for (r, yv) in y.iter_mut().enumerate() {
                    let c = b.frag4_lane(Opnd::Val(sb), r as u8);
                    let c = b.bitcast_f32(c);
                    let s = b.mul_f32(sa, c);
                    *yv = b.mul_f32(*yv, s);
                }
            }
            let w0 = b.cvt_pk_bf16_f32(y[0], y[1]);
            let w1 = b.cvt_pk_bf16_f32(y[2], y[3]);
            let f2 = b.frag2(w0, w1);
            let oo = b.bind_i32((ev(row) * ev(n_v) + ev(col)) << 1);
            let oo = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(rlive), a: Opnd::Val(oo), b: Opnd::Val(big) });
            b.buffer_store_b64(r_out, oo, 0i32, 0, f2);
        }
        let q1 = b.bind_i32(amdgpu_dsl::varx(qv[0]) + 1);
        b.set_var(qv[0], q1);
        let qd = b.bind_cond(ev(q1).ge(ev(nq)));
        b.break_if(qd);
        b.end_loop();
        // the counter back to 0 for the next launch (lane 0)
        let zc = b.c_i32(0);
        b.buffer_store_c(r_cnt, Some(Opnd::Val(cvo)), c0, 0, zc, amdgpu_dsl::MemTy::B32, false, sc1);
    }
    b.end_if();
}