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

use amdgpu_tile::{ev, ox, Cadence, LdsPlan, Lane, PipeBody, Pipeline, Role, Shared, SlotIdx, TileIt};
use amdgpu_dsl::{DType, Ext, FnBuilder, Function, Op, Opnd, ValueId, VarId, I32};
use amdgpu_isa::{MmaAtom, DS_READ_B128X2};
use amdgpu_layout::{composition, layout_to_offsets, make_layout, right_inverse, tup, ComposedLayout, ImmRange, IssuePlan, Ix, Layout, Swizzle};
use amdgpu_tile::{make_tiled_copy_a, make_tiled_copy_b, make_tiled_mma, Ctx, LdsPart, TiledMma};

pub mod graph;
pub mod reduce;
mod skeleton_v4;

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

/// Operand element type of the GEMM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Elem {
    /// fp8 e4m3: one scaled `v_mfma_scale_f32_16x16x128_f8f6f4` per 16 x 16 block and 128-byte K tile
    Fp8,
    /// bf16: two `v_mfma_f32_16x16x32_bf16` per block and K tile (one per 16-byte half of the operand fragments; A and B halves share the K
    /// permutation, so the sum is unchanged)
    Bf16,
    /// fp8 e4m3 with fp32 block scales (`Scales::Block128`): one zero-C scaled MFMA (unit E8M0) per block and K tile, then acc += t sa sb
    Fp8Bs,
    /// fp4 e2m1 (two per byte, low nibble first): two scaled `v_mfma_scale_f32_16x16x128_f8f6f4` (fp4 x fp4) per block and 128-byte K tile
    /// (= 256 elements), one per 16-byte half (32 K-contiguous elements per lane)
    Fp4,
    /// A8W4 (`gemm_a8w4_mxfp_128`): A fp8 e4m3 (MXFP8), B fp4 e2m1 (MXFP4), 1 x 32 E8M0 scales on both. A K tile is 256 elements: the B tile
    /// is 128 bytes, the A tile two 128-byte halves (lo / hi images of the ring slot, two A register fragments per m block); per block two
    /// scaled `16x16x128` fp8 x fp4 MFMAs (A half h with the 16-byte B half h, scale byte h)
    Fp8W4,
    /// A16W4 (`gemm_a16w4_mxfp4`): A bf16 (exact), B fp4 e2m1 with 1 x 32 E8M0 scales, dequantized to bf16 in the mainloop. K tile 256: the
    /// A tile is four 128-byte images (K 64 each); the host lays B with a K permutation inside each 256-K tile so that the lane's standard
    /// fp4 fragment dword 2 i + h holds the 8 K values of A image i half h (`bf16w4_k_perm`); B scales [N][K / 32] (not shuffled)
    Bf16W4,
}

impl Elem {
    /// K elements per 128-byte tile
    fn per_tile(self) -> i64 {
        match self {
            Elem::Fp8 | Elem::Fp8Bs => 128,
            Elem::Bf16 => 64,
            Elem::Fp4 | Elem::Fp8W4 | Elem::Bf16W4 => 256,
        }
    }
    /// 128-byte A images per K tile (the mixed-precision forms stage several A images per B tile)
    fn a_images(self) -> usize {
        match self {
            Elem::Fp8W4 => 2,
            Elem::Bf16W4 => 4,
            _ => 1,
        }
    }
}

/// Scale scheme of the fp8 operands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scales {
    /// E8M0 per 1 x 128 (A: row, K group) and 128 x 128 (B: N block, K group), applied by the scaled MFMA (`ptr_sfa` [K/128][M], `ptr_sfb`
    /// [N/128][K/128] bytes)
    Mx128,
    /// fp32 per row of A (per token, `ptr_sfa` [M]) and per row of B (per output channel, `ptr_sfb` [N]), applied to the accumulators in the
    /// epilogue; the MFMA runs with unit E8M0 scales (127)
    RowCol,
    /// no scales (bf16 / fp16 operands)
    None,
    /// E8M0 per 1 x 32 (MX): A [M][K / 32], B [N][K / 32] bytes, row-major (our layout), applied by the scaled MFMAs
    Mx32,
    /// fp32 per 1 x 128 (A [M][K / 128], our row-major layout) and 128 x 128 (B [N / 128][K / 128]) (DeepSeek-V3 blockscale), applied to the
    /// per-tile products (`Elem::Fp8Bs`)
    Block128,
}

impl GemmConfig {
    /// The shipped configuration.
    pub fn v4() -> GemmConfig {
        GemmConfig { cta_m: 256, cta_n: 256, acc_agpr: 24, pre_sync_mmas: 5, mid_sync_mmas: 3, epi_phases: 8, sched: "rewait,slotfill", lds_epilogue: true, scales: Scales::Mx128, elem: Elem::Fp8, split_k: false, moe: None, conv: None, scale_shuf: false, batched: false, per_tensor: false }
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
    fn wave_m(&self) -> i64 {
        self.cta_m / 2
    }
    fn wave_n(&self) -> i64 {
        self.cta_n / 2
    }
}

/// LDS image of a [rows x 128 bytes] K-tile: 16-row x 16-byte interleave (the aiter B preshuffle pattern with K = 128):
/// (r, k) -> (r / 16) 2048 + (k / 16) 256 + (r % 16) 16 + k % 16. A 16-lane group of every operand read covers 256 consecutive bytes (all banks once).
fn smem_tile(rows: i64) -> Layout {
    make_layout(tup![tup![16, rows / 16], tup![16, 8]], tup![tup![16, 2048], tup![1, 256]])
}
/// The same image restricted to one 16-row MMA operand block (element index r + 16 k, 16 x 128).
fn smem_block() -> Layout {
    make_layout(tup![16, tup![16, 8]], tup![16, tup![1, 256]])
}

/// `gemm` on loop variables (the operands of the register-staged tile are loop-carried): same operand / scale exchange as `amdgpu_tile::gemm`.
#[allow(clippy::too_many_arguments)]
fn mma_var(b: &mut FnBuilder, mma: &TiledMma, elem: Elem, unit: Option<ValueId>, acc: VarId, a: VarId, bf: VarId, sa: VarId, sb: VarId, byte_a: u8) {
    let p = mma_blocks(b, mma, elem, unit, &[(acc, a, bf, sa, sb, byte_a)], 0, None);
    assert!(p.is_empty(), "mma_var: block-scaled MMAs go through mma_blocks");
}

/// The MMAs of several 16 x 16 blocks of the K tile (acc, A frag, B frag, A scale, B scale, A scale byte). fp8: one scaled MFMA per block.
/// bf16: the 32-byte fragments are two 16-byte bf16 operands (8 elements each, the same K permutation for A and B); the first halves of every
/// block, then the second halves, so the two MFMAs into one accumulator are not back to back.
/// `a_hi_off` (A8W4): the hi A fragment of the A variable v is VarId(v + a_hi_off) (the loop state lays the hi fragments right after the lo ones).
fn mma_blocks(b: &mut FnBuilder, mma: &TiledMma, elem: Elem, unit: Option<ValueId>, blocks: &[(VarId, VarId, VarId, VarId, VarId, u8)], a_hi_off: u32, w4: Option<ValueId>) -> Vec<(VarId, ValueId, VarId, VarId)> {
    assert_eq!(mma.atom().name, "MfmaScale16x16x128F8");
    if elem == Elem::Bf16W4 {
        // A16W4: per (A image k, half h) step: every B fragment of the unit dequantized once (dword 2 k + h = 8 fp4 -> 8 bf16 times the
        // group scale 2^(e - 127), e = byte 2 (k % 2) of B scale word k / 2), then the bf16 MFMAs of all blocks (distance = the unit's blocks)
        let _ = w4;
        let kind = amdgpu_dsl::MmaKind::new(amdgpu_dsl::MmaShape::M16N16K32, amdgpu_dsl::MmaTy::Bf16);
        let mut bsets: Vec<(VarId, VarId)> = Vec::new();
        for &(_, _, bf, _, sb, _) in blocks {
            if !bsets.iter().any(|x| x.0 == bf) {
                bsets.push((bf, sb));
            }
        }
        // every (k, h) operand of the unit's B fragments first, then the MFMAs (the dequant results are well ahead of their MFMA reads)
        let mut all: Vec<Vec<(VarId, ValueId)>> = Vec::new();
        for k in 0..4u32 {
            for h in 0..2u8 {
                let ops: Vec<(VarId, ValueId)> = bsets
                    .iter()
                    .map(|&(bf, sb)| {
                        // dword d of the fragment holds K group d of the tile (measured with the host's W4DBG=6 probe): scale byte d % 4 of
                        // word d / 4 (the block's two scale words)
                        let d0 = 2 * k + h as u32;
                        let sw = b.bitcast_i32(Opnd::Var(VarId(sb.0 + d0 / 4)));
                        let e = b.bind_i32((ev(sw).lshr(8 * (d0 % 4) as i32)) & 0xff);
                        let sbits = b.bind_i32(ev(e) << 23);
                        let sf = b.bitcast_f32(sbits);
                        let d = 2 * k as u8 + h;
                        // dword d of the B fragment (a one-dword view of the variable)
                        let dw = b.frag_slice(Opnd::Var(bf), d, 1);
                        let dwi = b.bitcast_i32(dw);
                        let w: Vec<ValueId> = (0..4i32)
                            .map(|byte| {
                                let x = if byte == 0 { dwi } else { b.bind_i32(ev(dwi).lshr(8 * byte)) };
                                // (fp4 pair -> two f32 -> bf16 pair: exact for these values)
                                let f2 = b.cvt(amdgpu_dsl::CvtOp::ScalePkF32Fp4, &[Opnd::Val(x), Opnd::Val(sf)], 0);
                                let (lo, hi) = (b.frag2_lane(f2, 0), b.frag2_lane(f2, 1));
                                let (lo, hi) = (b.bitcast_f32(lo), b.bitcast_f32(hi));
                                b.cvt_pk_bf16_f32(lo, hi)
                            })
                            .collect();
                        (bf, b.pack4(w[0], w[1], w[2], w[3]))
                    })
                    .collect();
                all.push(ops);
            }
        }
        for k in 0..4u32 {
            for h in 0..2u8 {
                let ops = &all[(2 * k + h as u32) as usize];
                for &(acc, a, bf, _, _, _) in blocks {
                    let bop = ops.iter().find(|x| x.0 == bf).unwrap().1;
                    let ak = b.var_slice4(VarId(a.0 + k * a_hi_off), h);
                    if mma.swap_ab {
                        b.var_mfma(acc, kind, bop, ak, false);
                    } else {
                        b.var_mfma(acc, kind, ak, bop, false);
                    }
                }
            }
        }
        return Vec::new();
    }
    if elem == Elem::Fp8W4 {
        // per block, half h: A fragment (lo / hi, 32 bytes = 128 fp8), B 16-byte half h (128 fp4), scale byte h of both scale words
        for h in 0..2u8 {
            for &(acc, a, bf, sa, sb, byte_a) in blocks {
                let ah = if h == 0 { a } else { VarId(a.0 + a_hi_off) };
                let ys = b.var_slice4(bf, h);
                let _ = byte_a;
                if mma.swap_ab {
                    let k = amdgpu_dsl::MmaKind::scaled_f8f6f4(amdgpu_dsl::MmaShape::M16N16K128, amdgpu_dsl::MmaTy::Fp4, amdgpu_dsl::MmaTy::Fp8);
                    b.var_mfma_scaled(acc, k, ys, Opnd::Var(ah), amdgpu_dsl::MmaScale { sa: Opnd::Var(sb), sb: Opnd::Var(sa), opsel: h | (h << 4) }, false);
                } else {
                    let k = amdgpu_dsl::MmaKind::scaled_f8f6f4(amdgpu_dsl::MmaShape::M16N16K128, amdgpu_dsl::MmaTy::Fp8, amdgpu_dsl::MmaTy::Fp4);
                    b.var_mfma_scaled(acc, k, Opnd::Var(ah), ys, amdgpu_dsl::MmaScale { sa: Opnd::Var(sa), sb: Opnd::Var(sb), opsel: h | (h << 4) }, false);
                }
            }
        }
        return Vec::new();
    }
    let ops = |&(acc, a, bf, sa, sb, byte_a): &(VarId, VarId, VarId, VarId, VarId, u8)| {
        if mma.swap_ab { (acc, bf, a, sb, sa, 0, byte_a) } else { (acc, a, bf, sa, sb, byte_a, 0) }
    };
    match elem {
        Elem::Fp8W4 | Elem::Bf16W4 => unreachable!("mixed precision handled above"),
        Elem::Fp8 => {
            for blk in blocks {
                let (acc, x, y, sx, sy, bx, by) = ops(blk);
                b.var_mfma_scale16(acc, Opnd::Var(x), Opnd::Var(y), Opnd::Var(sx), Opnd::Var(sy), bx, by);
            }
        }
        Elem::Fp8Bs => {
            // fp32 block scales (1 x 128 A, 128 x 128 B): per block the K tile's product into a fresh accumulator (unit E8M0 scales, zero C);
            // the acc += t (sa sb) updates are returned and emitted by `bs_fma` at the end of the tile (MFMA latency off the critical path)
            let k = amdgpu_dsl::MmaKind::scaled_f8f6f4(amdgpu_dsl::MmaShape::M16N16K128, amdgpu_dsl::MmaTy::Fp8, amdgpu_dsl::MmaTy::Fp8);
            let u = Opnd::Val(unit.expect("unit scales"));
            return blocks
                .iter()
                .map(|blk| {
                    let (_, x, y, ..) = ops(blk);
                    let t = b.mfma_scaled(k, Opnd::Var(x), Opnd::Var(y), None, amdgpu_dsl::MmaScale { sa: u, sb: u, opsel: 0 });
                    (blk.0, t, blk.3, blk.4)
                })
                .collect();
        }
        Elem::Fp4 => {
            // two fp4 x fp4 scaled MFMAs per block: 16-byte half h of each fragment, the scale byte h of the (shifted) scale words
            let k = amdgpu_dsl::MmaKind::scaled_f8f6f4(amdgpu_dsl::MmaShape::M16N16K128, amdgpu_dsl::MmaTy::Fp4, amdgpu_dsl::MmaTy::Fp4);
            for h in 0..2u8 {
                for blk in blocks {
                    let (acc, x, y, sx, sy, ..) = ops(blk);
                    let xs = b.var_slice4(x, h);
                    let ys = b.var_slice4(y, h);
                    b.var_mfma_scaled(acc, k, xs, ys, amdgpu_dsl::MmaScale { sa: Opnd::Var(sx), sb: Opnd::Var(sy), opsel: h | (h << 4) }, false);
                }
            }
        }
        Elem::Bf16 => {
            let k = amdgpu_dsl::MmaKind::new(amdgpu_dsl::MmaShape::M16N16K32, amdgpu_dsl::MmaTy::Bf16);
            for h in 0..2u8 {
                for blk in blocks {
                    let (acc, x, y, ..) = ops(blk);
                    let xs = b.var_slice4(x, h);
                    let ys = b.var_slice4(y, h);
                    b.var_mfma(acc, k, xs, ys, false);
                }
            }
        }
    }
    Vec::new()
}

/// The deferred block-scale updates: acc += t (sa sb) lane by lane.
fn bs_fma(b: &mut FnBuilder, pending: &[(VarId, ValueId, VarId, VarId)]) {
    for &(acc, t, sa, sb) in pending {
        let (fa, fb) = (b.bitcast_f32(Opnd::Var(sa)), b.bitcast_f32(Opnd::Var(sb)));
        let s = b.mul_f32(fa, fb);
        for r in 0..4u8 {
            let tl = b.frag4_lane(t, r);
            let tl = b.bitcast_f32(tl);
            let al = b.frag4_lane(Opnd::Var(acc), r);
            let al = b.bitcast_f32(al);
            let f = b.let_(Op::FmaF32(Opnd::Val(tl), Opnd::Val(s), Opnd::Val(al)));
            b.var_set_lane(acc, r, f);
        }
    }
}

/// The 96-byte ABI (opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950), declared in offset order.
fn declare_abi(b: &mut FnBuilder) {
    let ptrs = [("ptr_a", 0u32), ("ptr_b", 8), ("ptr_c", 16), ("ptr_sfa", 64), ("ptr_sfb", 72)];
    let ints = [("m", 24u32), ("n", 28), ("k", 32), ("batch", 36), ("stride_a", 40), ("stride_b", 44), ("stride_c", 48), ("stride_a_batch", 52), ("stride_b_batch", 56), ("stride_c_batch", 60), ("stride_sfa", 80), ("stride_sfb", 84), ("stride_sfa_batch", 88), ("stride_sfb_batch", 92)];
    let mut all: Vec<(&str, u32, bool)> = ptrs.iter().map(|&(n, o)| (n, o, true)).chain(ints.iter().map(|&(n, o)| (n, o, false))).collect();
    all.sort_by_key(|x| x.1);
    for (name, off, is_ptr) in all {
        if is_ptr {
            b.arg(name, DType::Ptr, off, 8)
        } else {
            b.arg(name, DType::I32, off, 4)
        }
    }
}

/// The host contract of the ABI for a `cta_m x cta_n` CTA (what `gpu-verify` proves the accesses against): M = MT cta_m, N = NT cta_n, K = KT 128
/// (the kernel has no partial tiles), the grid (NT, MT), dense strides (stride_a = stride_b = K, stride_c = N, stride_sfa = M, stride_sfb = K / 128),
/// batch 1, and the tensor extents A M K, B N K (preshuffled, same size), C 2 M N, sa (K / 128) M, sb (N / 128) (K / 128) bytes.
fn declare_contract(b: &mut FnBuilder, cta_m: i64, cta_n: i64, scales: Scales, per_tile: i64, split: bool, moe: Option<MoeStage>, conv: Option<ConvCfg>, batched: bool, per_tensor: bool, a8w4: usize) {
    if let Some(cc) = conv {
        return declare_contract_conv(b, cta_m, cta_n, cc);
    }
    if let Some(st) = moe {
        return declare_contract_moe(b, cta_m, cta_n, scales, per_tile, st);
    }
    let (mt, nt) = (Ext::sym("MT"), Ext::sym("NT"));
    // int32 offsets: M K, N K and 2 M N stay below 2^31 (16384 x 65536 tiles of bytes at most)
    b.sym_range("MT", 1, (1 << 14) / cta_m);
    b.sym_range("NT", 1, (1 << 14) / cta_n);
    // K <= 128 PANEL_KB: the scale panel holds every K group of the CTA (the vendor sp64 limit); split-K: KT = S KS
    let kt = if split {
        b.sym_range("S", 1, 16);
        b.sym_range("KS", 1, PANEL_KB);
        Ext::sym("S") * Ext::sym("KS")
    } else {
        b.sym_range("KT", 1, PANEL_KB);
        Ext::sym("KT")
    };
    let m = mt.clone() * Ext::c(cta_m);
    let n = nt.clone() * Ext::c(cta_n);
    // batched: BT batches, grid z = BT, every tensor BT times its per-batch extent (the per-batch base checks are ptr_mad's)
    let bt = if batched {
        b.sym_range("BT", 1, 4096);
        b.grid_is(2, Ext::sym("BT"));
        Ext::sym("BT")
    } else {
        Ext::c(1)
    };
    // K in elements (kernarg) and in bytes (extents): KT tiles of 128 bytes
    let k = kt.clone() * Ext::c(128);
    let k_el = kt.clone() * Ext::c(per_tile);
    b.ka_is(24, m.clone());
    b.ka_is(28, n.clone());
    b.ka_is(32, k_el.clone());
    b.ka_is(36, if split { Ext::sym("S") } else if batched { Ext::sym("BT") } else { Ext::c(1) });
    if split {
        b.ka_is(88, Ext::sym("KS"));
        b.ka_is(92, m.clone());
        b.grid_is(2, Ext::sym("S"));
    }
    b.ka_is(40, k_el.clone());
    b.ka_is(44, k_el);
    b.ka_is(48, n.clone());
    b.ka_is(80, m.clone());
    b.ka_is(84, kt.clone());
    b.grid_is(0, nt.clone());
    b.grid_is(1, mt.clone());
    // (mixed precision: A has a_images 128-byte images per 128-byte B tile)
    b.arg_tensor("ptr_a", bt.clone() * m.clone() * k.clone() * Ext::c(a8w4 as i64));
    b.arg_tensor("ptr_b", bt.clone() * n.clone() * k.clone());
    if split {
        b.arg_tensor("ptr_c", bt.clone() * Ext::c(4) * Ext::sym("S") * m.clone() * n.clone());
    } else {
        b.arg_tensor("ptr_c", bt.clone() * Ext::c(2) * m.clone() * n.clone());
    }
    match scales {
        Scales::Mx128 => {
            b.arg_tensor("ptr_sfa", bt.clone() * kt.clone() * m);
            b.arg_tensor("ptr_sfb", bt.clone() * nt * Ext::c(cta_n / 128) * kt);
        }
        Scales::RowCol if per_tensor => {
            b.arg_tensor("ptr_sfa", bt.clone() * Ext::c(4));
            b.arg_tensor("ptr_sfb", bt.clone() * Ext::c(4));
        }
        Scales::RowCol => {
            b.arg_tensor("ptr_sfa", bt.clone() * Ext::c(4) * m);
            b.arg_tensor("ptr_sfb", bt.clone() * Ext::c(4) * n);
        }
        Scales::None => {}
        Scales::Mx32 => {
            b.arg_tensor("ptr_sfa", bt.clone() * m * kt.clone() * Ext::c(8));
            b.arg_tensor("ptr_sfb", bt.clone() * n * kt * Ext::c(8));
        }
        Scales::Block128 => {
            b.arg_tensor("ptr_sfa", bt.clone() * m * kt.clone() * Ext::c(4));
            b.arg_tensor("ptr_sfb", bt.clone() * nt * Ext::c(cta_n / 128) * kt * Ext::c(4));
        }
    }
}

/// The implicit-GEMM conv contract (bf16): M = NB OD OH OW rows (the last M tile partial), N = NT cta_n output channels, C = 64 CB input
/// channels, K = C T R S; tensors: input NB DI H W C, weights N K, output M N (bf16).
fn declare_contract_conv(b: &mut FnBuilder, cta_m: i64, cta_n: i64, cc: ConvCfg) {
    for (n, o) in [("NB", 96u32), ("DI", 100), ("H", 104), ("W", 108), ("C", 112), ("OD", 116), ("OH", 120), ("OW", 124)] {
        b.arg(n, DType::I32, o, 4);
    }
    for o in [96u32, 100, 104, 108, 116, 120, 124] {
        b.ka_range(o, 1, 4096);
    }
    b.sym_range("CB", 1, 64);
    b.sym_range("NT", 1, (1 << 14) / cta_n);
    let cb = Ext::sym("CB");
    b.ka_is(112, cb.clone() * Ext::c(64));
    let trs = (cc.t * cc.r * cc.s) as i64;
    let m = Ext::ka(96) * Ext::ka(116) * Ext::ka(120) * Ext::ka(124);
    let n = Ext::sym("NT") * Ext::c(cta_n);
    let kt = cb.clone() * Ext::c(trs);
    let k_el = kt.clone() * Ext::c(64);
    b.ka_is(24, m.clone());
    b.ka_is(28, n.clone());
    b.ka_is(32, k_el.clone());
    b.ka_is(36, Ext::c(1));
    b.ka_is(40, k_el.clone());
    b.ka_is(44, k_el);
    b.ka_is(48, n.clone());
    b.ka_is(80, m.clone());
    b.ka_is(84, kt.clone());
    b.grid_is(0, Ext::sym("NT"));
    b.grid_is(1, m.clone().ceil_div(cta_m));
    b.arg_tensor("ptr_a", Ext::ka(96) * Ext::ka(100) * Ext::ka(104) * Ext::ka(108) * cb * Ext::c(128));
    b.arg_tensor("ptr_b", n.clone() * kt * Ext::c(128));
    b.arg_tensor("ptr_c", m * n * Ext::c(2));
}

/// The MoE contract: MT = max M tiles (grid y), NT = output N tiles (grid x), A rows m (tokens or tokens x topk), B rows per expert n, E
/// experts, KT 128-byte K tiles.
fn declare_contract_moe(b: &mut FnBuilder, cta_m: i64, cta_n: i64, scales: Scales, per_tile: i64, st: MoeStage) {
    for (n, o) in [("ptr_sorted_ids", 96u32), ("ptr_sorted_expert_ids", 104), ("ptr_num_valid_ids", 112), ("ptr_sorted_weights", 120)] {
        b.arg(n, DType::Ptr, o, 8);
    }
    for (n, o) in [("topk", 128u32), ("E", 132), ("n_out", 136), ("M_tokens", 140)] {
        b.arg(n, DType::I32, o, 4);
    }
    let (mt, nt, kt, ee) = (Ext::sym("MT"), Ext::sym("NT"), Ext::sym("KT"), Ext::sym("E"));
    let nt_c = nt.clone();
    b.sym_range("MT", 1, 4096);
    b.sym_range("NT", 1, 256);
    b.sym_range("KT", 1, PANEL_KB);
    b.sym_range("E", 1, 512);
    b.sym_range("AR", 1, 1 << 20);
    b.ka_range(128, 1, 16);
    b.ka_range(140, 1, 1 << 20);
    let k = kt.clone() * Ext::c(128);
    let k_el = kt.clone() * Ext::c(per_tile);
    // output columns per CTA: stage 1 half the B rows (gate / up interleaved)
    let ncta = if matches!(st, MoeStage::Up { .. }) { cta_n / 2 } else { cta_n };
    let n_out = nt.clone() * Ext::c(ncta);
    let n = if matches!(st, MoeStage::Up { .. }) { n_out.clone() * Ext::c(2) } else { n_out.clone() };
    let m = Ext::sym("AR");
    b.ka_is(24, m.clone());
    b.ka_is(28, n.clone());
    b.ka_is(32, k_el.clone());
    b.ka_is(36, Ext::c(1));
    b.ka_is(40, k_el.clone());
    b.ka_is(44, k_el);
    b.ka_is(48, n_out.clone());
    b.ka_is(84, kt.clone());
    b.ka_is(132, ee.clone());
    b.ka_is(136, n_out.clone());
    b.grid_is(0, nt);
    b.grid_is(1, mt.clone());
    b.arg_tensor("ptr_a", m.clone() * k.clone());
    b.arg_tensor("ptr_b", ee.clone() * n.clone() * k);
    let out_rows = Ext::ka(140) * Ext::ka(128);
    b.arg_tensor("ptr_c", out_rows * n_out * Ext::c(if matches!(st, MoeStage::Up { .. } | MoeStage::DownH) { 2 } else { 4 }));
    b.arg_tensor("ptr_sorted_ids", mt.clone() * Ext::c(4 * cta_m));
    b.arg_tensor("ptr_sorted_weights", mt.clone() * Ext::c(4 * cta_m));
    b.arg_tensor("ptr_sorted_expert_ids", mt * Ext::c(4));
    b.arg_tensor("ptr_num_valid_ids", Ext::c(8));
    if scales == Scales::RowCol {
        b.arg_tensor("ptr_sfa", Ext::c(4) * m);
        b.arg_tensor("ptr_sfb", Ext::c(4) * ee * n);
    } else if scales == Scales::Block128 {
        b.arg_tensor("ptr_sfa", Ext::c(4) * m * kt.clone());
        b.arg_tensor("ptr_sfb", Ext::c(4 * cta_n / 128) * ee * nt_c * kt);
    } else if scales == Scales::Mx32 {
        // MXFP4 MoE: 1 x 32 E8M0 [rows][8 KT] bytes (A rows, every expert's n rows; not shuffled: gathered rows)
        b.arg_tensor("ptr_sfa", Ext::c(8) * m * kt.clone());
        b.arg_tensor("ptr_sfb", Ext::c(8) * ee * n * kt);
    }
}

/// One DMA stage of a [rows x 128 B] tile: per issue the lane's voffset (bound once, loop invariant) and the LDS offset of its 1 KB block.
struct Dma {
    issues: Vec<(ValueId, i32)>,
    /// per-lane gathered rows (MoE): every issue its own M0, no instruction offset (the voffset is the exact source offset)
    gather: bool,
    /// LDS bytes per 1 KB issue block (1024, or 1056 for the padded A image); the issue LDS offsets above are in 1 KB units (`act`)
    pitch: i32,
}

impl Dma {
    /// (M0 offset of the issue's group of 4, instruction offset) for the virtual 1 KB offset `lds`
    fn act(&self, lds: i32) -> (i32, u32) {
        if self.gather {
            return (lds / 1024 * self.pitch, 0);
        }
        ((lds - lds % 4096) / 1024 * self.pitch, ((lds % 4096) / 1024 * self.pitch) as u32)
    }
    /// the M0 group of an issue
    fn group(&self, lds: i32) -> i32 {
        if self.gather { lds } else { lds - lds % 4096 }
    }
}

/// Turns the per-issue source offsets (minus the instruction offset `pitch (q % 4)`) into voffsets, lowered through the GF(2) bit-form normal form
/// (amdgpu_layout::BitForm: shift groups, CSE across plans by Ctx::bind).
fn finish_dma_p(b: &mut FnBuilder, ctx: &Ctx, gofs: Vec<Ix>, pitch: i32) -> Dma {
    let adj: Vec<Ix> = gofs.iter().enumerate().map(|(q, g)| g.add_c(-(pitch as i64) * (q as i64 % 4))).collect();
    ctx.set_bitfield(true);
    let issues = adj.iter().enumerate().map(|(q, g)| { let e = ctx.bind(b, g); (b.bind_i32(e), 1024 * q as i32) }).collect();
    ctx.set_bitfield(false);
    Dma { issues, pitch, gather: false }
}
/// LDS byte `d = 1024 (issues (tid / 64) + q) + 16 (tid % 64)` holds tile element `smem^-1(d)`, so the source offset is `glob o right_inverse(smem)`;
/// proved here for sampled threads against a direct search of the LDS layout.
fn dma_plan(b: &mut FnBuilder, ctx: &Ctx, s_tid: &Ix, rows: i64, glob: &Layout, proof_k: i64, waves: i64) -> Dma {
    let issues = rows * 128 / 1024 / waves;
    let sm = smem_tile(rows);
    let tv = make_layout(tup![tup![64, waves], tup![16, issues]], tup![tup![16, 1024 * issues], tup![1, 1024]]);
    let g = composition(glob, &right_inverse(&sm));
    let gofs = IssuePlan::new(tv, 16).offsets(&g, s_tid, &Ix::C(0), &ctx.dyns);
    let tid_id = match s_tid {
        Ix::Sym(s) => *s,
        _ => unreachable!(),
    };
    for th in [0i64, 1, 15, 16, 63, 64, 100, 127, 255].into_iter().filter(|t| *t < 64 * waves) {
        let env = |s: u32| if s == tid_id { th } else { proof_k };
        for q in 0..issues {
            let lds = 1024 * (issues * (th / 64) + q) + 16 * (th % 64);
            let (r, c) = (0..rows).flat_map(|r| (0..8).map(move |c| (r, 16 * c))).find(|&(r, c)| sm.at(r + rows * c) == lds).expect("LDS byte covered");
            let want = glob.crd2idx(&tup![r, c], &ctx.dyns).eval(&env);
            assert_eq!(gofs[q as usize].eval(&env), want, "DMA rows {rows} thread {th} issue {q}");
        }
    }
    // MUBUF form: voffset (loop invariant) + soffset + the instruction offset 1024 (q % 4) = the global offset; M0 = the group's first 1 KB block
    finish_dma_p(b, ctx, gofs, 1024)
}

/// The A DMA of the padded image with GATHERED rows (MoE): the same lane -> (tile row lane / 8 + 8 block, bytes 16 (lane % 8)) map as
/// `dma_plan_rows_pad`; the source row of tile row r is `row_src(r)` (per lane, loop invariant); exact voffsets, no instruction offset.
fn dma_plan_gather(b: &mut FnBuilder, lane: &Lane, rows: i64, waves: i64, k_v: ValueId, row_src: &dyn Fn(&mut FnBuilder, I32) -> ValueId) -> Dma {
    let issues = rows * 128 / 1024 / waves;
    let mut out = Vec::new();
    for q in 0..issues {
        let r = (lane.wave() * issues as i32 + q as i32) * 8 + lane.lane().lshr(3);
        let src = row_src(b, r);
        let v = b.bind_i32(ev(src) * ev(k_v) + ((lane.lane() & 7) << 4));
        out.push((v, 1024 * q as i32));
    }
    // the row-index loads complete here: the pipeline's counted vmcnt waits count only its DMAs
    b.wait_vm(0);
    Dma { issues: out, pitch: 1056, gather: true }
}

/// A-tile DMA for the PADDED row-major LDS image (the vendor form): 1 KB blocks of 8 whole 128-byte rows at a pitch of 1056 bytes (32 bytes of
/// padding per block spread the banks of the fragment reads), lanes linear in the source (lane -> row lane / 8, bytes 16 (lane % 8)): fully
/// monotone 128-byte segments, no source-side permutation. LDS image: (r, k) -> 1056 (r / 8) + 128 (r % 8) + k.
fn dma_plan_rows_pad(b: &mut FnBuilder, ctx: &Ctx, s_tid: &Ix, rows: i64, glob: &Layout, proof_k: i64, waves: i64) -> Dma {
    let issues = rows * 128 / 1024 / waves;
    let d = &ctx.dyns;
    let tid_id = match s_tid {
        Ix::Sym(s) => *s,
        _ => unreachable!(),
    };
    let mut out = Vec::new();
    for q in 0..issues {
        // block index (issues (tid / 64) + q), row inside the block lane / 8, bytes 16 (lane % 8)
        let blk = s_tid.div_c(64, d).mul_c(issues).add_c(q);
        let lane = s_tid.mod_c(64, d);
        let r = blk.mul_c(8).add(&lane.div_c(8, d));
        let k = lane.mod_c(8, d).mul_c(16);
        let g = glob.crd2idx(&tup![r.clone(), k.clone()], d);
        for th in [0i64, 1, 7, 8, 63, 64, 100, 127, 255].into_iter().filter(|t| *t < 64 * waves) {
            let env = |s: u32| if s == tid_id { th } else { proof_k };
            let (bb, ll) = (issues * (th / 64) + q, th % 64);
            let want = glob.crd2idx(&tup![8 * bb + ll / 8, 16 * (ll % 8)], d).eval(&env);
            assert_eq!(g.eval(&env), want, "padded A DMA thread {th} issue {q}");
        }
        out.push(g);
    }
    finish_dma_p(b, ctx, out, 1056)
}

/// The scale panel: every K group's scales of the CTA in LDS, written once in the prologue (`prologue_stores`).
struct Panel {
    sh: Shared,
    /// LDS offset of the B part inside the panel
    b_base: i64,
    /// A part transposed: [kb][wm][lane % 16][m block i] (8 bytes per lane: 2 dwords, MFMA byte selects); this lane's offset
    tr: ValueId,
    /// B part expanded to dwords [wn][PANEL_KB] (byte 0 = the scale): this wave's dword offset (wn 4 PANEL_KB)
    b_dw: ValueId,
    /// the VGPR scale path emitted in `prologue_stores`
    sfa: SfaJob,
}

/// The VGPR scale path: A scales by 4 passes of buffer_load_dwordx4 (lane = 16 M rows of one K group), a 4x4 byte transpose per dword across each
/// lane quad (dpp quad_perm [1,0,3,2] + v_perm, dpp quad_perm [2,3,0,1] + v_perm) straight into the transposed panel layout, one ds_write per dword;
/// B scales by one dword load per thread. Layout proof: `sfa_transpose_proof`.
struct SfaJob {
    voff: ValueId,
    pso: Vec<ValueId>,
    sel1: ValueId,
    sel2: ValueId,
    dst: ValueId,
    bvo: ValueId,
    bsh: ValueId,
    bdst: ValueId,
}

/// Build-time proof of the VGPR scale path: simulates every lane of every pass byte by byte (the global load, both dpp/perm steps) and checks that
/// the byte written at panel offset 256 kb + 128 wm + 8 r + i is sfa[kb][128 wm + 16 i + r] (the transposed layout `scale_issue` reads).
/// Public wrapper for the VGPR/DPP scale-panel proof (item E1 calls it explicitly; the v4 builder
/// calls it build-time through `sfa_transpose_proof`).
pub fn verify_sfa_transpose() { sfa_transpose_proof() }

fn sfa_transpose_proof() {
    // a byte is identified by (kb, m)
    let perm = |s0: [(i64, i64); 4], s1: [(i64, i64); 4], sel: u32| -> [(i64, i64); 4] {
        let mut o = [(0, 0); 4];
        for (k, ok) in o.iter_mut().enumerate() {
            let sb = (sel >> (8 * k)) & 0xff;
            *ok = if sb < 4 { s1[sb as usize] } else { s0[(sb - 4) as usize] };
        }
        o
    };
    let mut seen = std::collections::HashMap::new();
    for p in 0..4i64 {
        for d in 0..4i64 {
            // x of every thread (256 threads, one pass): bytes of rows 16 c + 4 d + k of K group g / 16
            let x: Vec<[(i64, i64); 4]> = (0..256i64).map(|tid| { let g = tid + 256 * p; let (kb, c) = (g / 16, g % 16); [0, 1, 2, 3].map(|k| (kb, 16 * c + 4 * d + k)) }).collect();
            let qp = |v: &Vec<[(i64, i64); 4]>, pm: [usize; 4]| -> Vec<[(i64, i64); 4]> { (0..256usize).map(|l| v[4 * (l / 4) + pm[l % 4]]).collect() };
            let y = qp(&x, [1, 0, 3, 2]);
            let n1: Vec<[(i64, i64); 4]> = (0..256usize).map(|l| perm(y[l], x[l], if l % 2 == 0 { 0x0602_0400 } else { 0x0307_0105 })).collect();
            let z = qp(&n1, [2, 3, 0, 1]);
            for tid in 0..256i64 {
                let l = tid as usize;
                let out = perm(z[l], n1[l], if l % 4 < 2 { 0x0504_0100 } else { 0x0302_0706 });
                let g = tid + 256 * p;
                let (kb, c, j) = (g / 16, g % 16, tid % 4);
                let addr = 256 * kb + 128 * (c / 8) + 8 * (4 * d + j) + 4 * ((c / 4) % 2);
                for (k, byte) in out.iter().enumerate() {
                    let a = addr + k as i64;
                    let (pk, r, i, wm) = (a / 256, (a % 128) / 8, a % 8, (a % 256) / 128);
                    assert_eq!(*byte, (pk, 128 * wm + 16 * i + r), "sfa transpose: tid {tid} pass {p} dword {d} byte {k}");
                    seen.insert(a, ());
                }
            }
        }
    }
    assert_eq!(seen.len(), 64 * 256, "sfa transpose: every panel byte written once");
}

struct Body {
    cfg: Cfg,
    /// K tiles of this CTA's loop (K / 128, or KS with split-K)
    kb_v: ValueId,
    /// the E8M0 scale panel (Scales::Mx128)
    panel: Option<Panel>,
    mma: TiledMma,
    a_s: Shared,
    b_s: Shared,
    a_dma: Dma,
    b_dma: Dma,
    a_org: ValueId,
    b_org: ValueId,
    a_wave: ValueId,
    b_wave: ValueId,
    rsrc_a: ValueId,
    rsrc_b: ValueId,
    rsrc_sfa: ValueId,
    rsrc_sfb: ValueId,
    a_rd: LdsPart,
    b_rd: LdsPart,
    /// mixed precision: the A images 1.. (the image-0 partition moved by k `a_hi`) and the image stride in the slot
    a_rd_x: Vec<LdsPart>,
    a_hi: i32,
    /// A16W4: the lane's B-scale byte shift (8 (g / 2), g = lane / 16)
    w4_shift: Option<ValueId>,
    zero4: ValueId,
    acc: Vec<VarId>,
    /// MMAs of the register tile already issued by `pre_sync` / `post_wait`
    npre_done: usize,
    /// timing stamps (GEMM_STAMP=ends|tiles): debug buffer, this wave's record offset, per-tile stamps
    stamp: Option<(ValueId, ValueId, bool)>,
    /// `Scales::RowCol`: the unit E8M0 scale words as VGPR values (lane / 64 = 0 keeps them vector: a wave-uniform constant would be held in an
    /// SGPR and copied to a VGPR before every scaled MFMA, with a VALU -> MFMA wait)
    unit: Option<(ValueId, ValueId)>,
    /// `Scales::Mx32`: the per-tile scale loads
    mx32: Option<Mx32>,
    /// `Scales::Block128`: the per-tile fp32 scale loads (voffsets; `shift` unused)
    bs128: Option<Mx32>,
    /// implicit-GEMM convolution: the per-tile A addressing
    conv: Option<ConvRt>,
    /// split-K: this split's first K tile
    zb: Option<ValueId>,
    /// `Elem::Fp8Bs`: the block products of the tile waiting for their scaled accumulation (`bs_fma`)
    pending: std::cell::RefCell<Vec<(VarId, ValueId, VarId, VarId)>>,
    /// register-staged tile: initial values (tile 0) and the loop variables: B fragments, A fragments, A scales, B scale
    staged: Option<(Vec<ValueId>, Vec<ValueId>, Vec<ValueId>, Vec<ValueId>)>,
    rv: Option<(Vec<VarId>, Vec<VarId>, Vec<VarId>, Vec<VarId>)>,
}

/// The 1 x 32 E8M0 scale loads of `Scales::Mx32`: per m block (n block) the lane's loop-invariant voffset of its row's 8 groups of the tile,
/// and the per-lane v_perm selector (`shift`) that picks the bytes of K groups g and 4 + g (g = lane / 16).
struct Mx32 {
    voff_a: Vec<ValueId>,
    voff_b: Vec<ValueId>,
    shift: ValueId,
    /// `scale_shuf`: [rows / 16][KT / 2][lane][4 bytes] with lane 16 g + r holding the bytes of K groups g, 4 + g of the even
    /// tile, then of the odd tile (one coalesced dword per lane, block and tile pair); the bit shift of the last `load_scales` (0 / 16)
    shuf: Option<std::cell::RefCell<Option<ValueId>>>,
}

impl Body {
    /// MMAs `from..to` (in row-major order over the first half of the B fragments) of the register tile.
    fn early_mmas(&mut self, b: &mut FnBuilder, from: usize, to: usize) {
        let (bv, av, sav, sbv) = self.rv.clone().expect("state bound");
        let nn = bv.len();
        let half = nn / 2;
        let mut n = 0;
        let mut blocks = Vec::new();
        let nma = self.n_a();
        'o: for i in 0..nma {
            for j in 0..half {
                if n >= to {
                    break 'o;
                }
                if n >= from {
                    let (w, by) = self.sa_sel(i);
                    blocks.push((self.acc[nn * i + j], av[i], bv[j], sav[w], sbv[self.sb_sel(j)], by));
                }
                n += 1;
            }
        }
        let p = mma_blocks(b, &self.mma, self.cfg.elem, self.unit.map(|u| u.0), &blocks, nma as u32, self.w4_shift);
        self.pending.borrow_mut().extend(p);
        self.npre_done = n;
    }
    /// The scale words of tile `kb` from the transposed panel: the 8 A-scale bytes of this lane (one ds_read_b64; the MMA selects byte i % 4 of
    /// word i / 4: `sa_sel`) and the B scale (the dword-expanded panel: byte 0, no shift).
    fn load_scales(&self, b: &mut FnBuilder, kb: I32) -> (Vec<ValueId>, Vec<ValueId>) {
        if self.cfg.scales == Scales::RowCol || self.cfg.scales == Scales::None {
            // unit E8M0 scales (127 in every byte): the fp32 row / column scales are applied in the epilogue
            let (u, v) = self.unit.expect("unit scales");
            return (vec![u, u], vec![v]);
        }
        // K group of a real tile (t + 1 is read only when it exists): the clamp is a no-op that bounds it for the verifier's loop-exit-blind intervals
        let kb = b.bind_i32(kb.lt(ev(self.kb_v)).select_i32(kb.clone(), ev(self.kb_v) - 1));
        if let Some(bs) = &self.bs128 {
            // fp32 [rows][KT]: tile kb (+ the split's first tile) at byte 4 kb (SGPR soffset), per m block one float; the B block float (raw:
            // installed as words)
            let kb = match self.zb {
                Some(zb) => b.bind_i32(ev(kb) + ev(zb)),
                None => kb,
            };
            let so = { let v = b.bind_i32(ev(kb) << 2); b.read_first_lane(v) };
            let sa = bs.voff_a.iter().map(|v| b.buffer_load_f32(self.rsrc_sfa, *v, so, 0)).collect();
            let sb = bs.voff_b.iter().map(|v| b.buffer_load_f32(self.rsrc_sfb, *v, so, 0)).collect();
            return (sa, sb);
        }
        if let (Some(mx), Some(sh)) = (&self.mx32, self.mx32.as_ref().and_then(|m| m.shuf.as_ref())) {
            // shuffled: the tile pair kb / 2 (256 bytes per block), the tile's half selected by a shift at the install
            let so = { let v = b.bind_i32(ev(kb).lshr(1) << 8); b.read_first_lane(v) };
            let s = b.bind_i32((ev(kb) & 1) << 4);
            *sh.borrow_mut() = Some(s);
            let ld = |b: &mut FnBuilder, rs: ValueId, vo: &[ValueId]| -> Vec<ValueId> { vo.iter().map(|v| b.buffer_load_f32(rs, *v, so, 0)).collect() };
            return (ld(b, self.rsrc_sfa, &mx.voff_a), ld(b, self.rsrc_sfb, &mx.voff_b));
        }
        if let (Some(mx), Elem::Bf16W4) = (&self.mx32, self.cfg.elem) {
            // A16W4: no A scales (zeros, unused); B [N][K / 32]: the 8 group bytes of the tile per n block, as two dwords (words 2 j, 2 j + 1)
            let so = { let v = b.bind_i32(ev(kb) << 3); b.read_first_lane(v) };
            let sa = (0..mx.voff_a.len()).map(|_| b.c_i32(0)).collect();
            // the tile's 8 group bytes of the block's row as two words (groups 0..3, 4..7)
            let mut sb = Vec::new();
            for v in &mx.voff_b {
                let f = b.buffer_load_b64(self.rsrc_sfb, *v, so, 0);
                sb.push(b.frag2_lane(f, 0));
                sb.push(b.frag2_lane(f, 1));
            }
            return (sa, sb);
        }
        if let Some(mx) = &self.mx32 {
            // 1 x 32 E8M0 [rows][K / 32] bytes: the 8 groups of the tile (one 8-byte load per m / n block); the lane group g = lane / 16 of
            // MFMA h (fragment bytes 64 h + 16 g ..) covers K group 4 h + g: v_perm puts byte g of the low word in byte 0 and byte g of the high
            // word in byte 1 (byte h = the scale of MFMA h)
            let so = { let v = b.bind_i32(ev(kb) << 3); b.read_first_lane(v) };
            // the raw 8-byte loads: the byte selection (`scale_word`) is applied where the words are installed, so the loads stay in flight
            // behind the MMAs instead of being waited for at once
            let ld = |b: &mut FnBuilder, rs: ValueId, vo: &[ValueId]| -> Vec<ValueId> { vo.iter().map(|v| b.buffer_load_b64(rs, *v, so, 0)).collect() };
            let sa = ld(b, self.rsrc_sfa, &mx.voff_a);
            let sb = ld(b, self.rsrc_sfb, &mx.voff_b);
            return (sa, sb);
        }
        let kb = ev(kb);
        let p = self.panel.as_ref().expect("scale panel");
        let (base, pr) = (p.sh.base, p.sh.at(&SlotIdx::new(0, 0)));
        let sab = b.bind_i32((kb.clone() << 8) + (base as i32) + ev(p.tr));
        let sbb = b.bind_i32((kb << 2) + ev(p.b_dw) + (base + p.b_base) as i32);
        pr.read(b, |b| {
            let w2 = b.load_lds_b64(sab);
            let wa = vec![b.frag2_lane(w2, 0), b.frag2_lane(w2, 1)];
            (wa, vec![b.load_lds_b32(sbb)])
        })
    }
    /// The scale word of a `load_scales` value: `Scales::Mx32` selects the lane's two group bytes of the 8-byte load (v_perm); identity otherwise.
    fn scale_word(&self, b: &mut FnBuilder, v: ValueId) -> ValueId {
        if self.w4_shift.is_some() {
            // A16W4: the words are selected at the load
            return b.bitcast_i32(v);
        }
        match &self.mx32 {
            Some(mx) if mx.shuf.is_some() => {
                let s = mx.shuf.as_ref().unwrap().borrow().expect("shift of the last scale load");
                let w = b.bitcast_i32(v);
                b.bind_i32(ev(w).lshr(ev(s)))
            }
            Some(mx) => {
                let lo = b.frag2_lane(v, 0);
                let hi = b.frag2_lane(v, 1);
                let (lo, hi) = (b.bitcast_i32(lo), b.bitcast_i32(hi));
                b.perm_b32(hi, lo, mx.shift)
            }
            None if self.bs128.is_some() => b.bitcast_i32(v),
            None => v,
        }
    }
    /// The (word, byte) of the A scale of m block i in the `load_scales` words.
    fn sa_sel(&self, i: usize) -> (usize, u8) {
        if self.mx32.is_some() || self.bs128.is_some() {
            return (i, 0);
        }
        (i / 4, (i % 4) as u8)
    }
    /// The B scale word of n block j.
    fn sb_sel(&self, j: usize) -> usize {
        if self.cfg.elem == Elem::Bf16W4 {
            2 * j
        } else if self.mx32.is_some() {
            j
        } else if self.bs128.as_ref().is_some_and(|bs| bs.voff_b.len() == 2) {
            // MoE stage 1 blockscale: gate / up blocks alternate
            j % 2
        } else {
            0
        }
    }
    /// An operand fragment read into AGPRs.
    /// A fragments (m blocks) of the wave (A8W4 keeps lo + hi fragments: the A variables are twice as many)
    fn n_a(&self) -> usize {
        (self.cfg.wave_m() / 16) as usize
    }
    fn frag(&self, b: &mut FnBuilder, part: &amdgpu_tile::LdsSlotPart, o: usize) -> ValueId {
        let v = part.read(b, o);
        b.agpr(v)
    }
    /// The whole tile `t` into registers (prologue: tile 0).
    fn load_tile(&mut self, b: &mut FnBuilder, t: &TileIt) {
        let (nm, nn) = ((self.cfg.wave_m() / 16) as usize, (self.cfg.wave_n() / 16) as usize);
        let (sa, sb) = self.load_scales(b, ox(t.it));
        let sa: Vec<ValueId> = sa.into_iter().map(|v| self.scale_word(b, v)).collect();
        let sb: Vec<ValueId> = sb.into_iter().map(|v| self.scale_word(b, v)).collect();
        let ar = self.a_rd.at(b, &self.a_s.at(&t.cur));
        let br = self.b_rd.at(b, &self.b_s.at(&t.cur));
        let bf: Vec<ValueId> = (0..nn).map(|j| self.frag(b, &br, j)).collect();
        let mut af: Vec<ValueId> = (0..nm).map(|i| self.frag(b, &ar, i)).collect();
        // mixed precision: the fragments of A image k right after those of image k - 1 (mma_blocks finds image k at lo + k nm)
        for part in &self.a_rd_x {
            let arh = part.at(b, &self.a_s.at(&t.cur));
            let hi: Vec<ValueId> = (0..nm).map(|i| self.frag(b, &arh, i)).collect();
            af.extend(hi);
        }
        self.staged = Some((bf, af, sa, sb));
    }
    /// The MMAs of the register tile; when `t.staged`, each fragment is reloaded from tile t+1 right after its last use: first half of the B
    /// fragments with every A fragment, then the B halves are reloaded; second half, after which every A fragment is reloaded; the scales of t+1
    /// are read before the MMAs.
    fn compute_tile(&mut self, b: &mut FnBuilder, t: &TileIt) {
        let roles = vec!["all".to_string()];
        // pipelined loop: the linear barrier count is not a stable property (StageContract::ANY_PHASE)
        let phase = amdgpu_dsl::StageContract::ANY_PHASE;
        let me: &Body = self;
        // the slot offsets of t + 1 are bound before the region (they are shared with the staging instance through the SlotIdx cache)
        if t.staged {
            let _ = me.a_s.at(&t.next).off_v(b);
            let _ = me.b_s.at(&t.next).off_v(b);
        }
        me.tile_stage(b, t, phase, roles).emit(b);
    }
    /// Stage `mma.blockscale_tile`: the MMAs of the register tile; when `t.staged`, each fragment is reloaded from tile t+1 right after its last
    /// use (first half of the B fragments with every A fragment, then the B halves are reloaded; second half, after which every A fragment is
    /// reloaded); the scales of t+1 are read before the MMAs and installed at the end (partial lgkm waits). Captured units.
    fn tile_stage(&self, b: &mut FnBuilder, t: &TileIt, phase: u32, roles: Vec<String>) -> amdgpu_tile::stage::StageInst<'static> {
        use amdgpu_tile::weave::Units;
        let (bv, av, sav, sbv) = self.rv.clone().expect("state bound");
        let (nm, nn) = (self.n_a(), bv.len());
        let half = nn / 2;
        let mut u = Units::new();
        let (_, (next, early)) = u.capture(b, "tile.setup", |b| {
            let next = t.staged.then(|| (self.a_rd.at(b, &self.a_s.at(&t.next)), self.b_rd.at(b, &self.b_s.at(&t.next))));
            // mixed precision: the other A images of t + 1
            let next_hi: Vec<amdgpu_tile::LdsSlotPart> = if t.staged { self.a_rd_x.iter().map(|p| p.at(b, &self.a_s.at(&t.next))).collect() } else { Vec::new() };
            // the scales of t+1 are read FIRST (before the fragment reloads queue up behind them) and installed at the end (partial lgkm waits)
            let early = t.staged.then(|| self.load_scales(b, ox(t.it) + 1));
            ((next, next_hi), early)
        });
        let (next, next_hi) = next;
        // MMAs of the register tile with the in-place reloads of t+1 (into `u`)
        let body = |b: &mut FnBuilder, u: &mut Units| {
            // bf16 (two dependent K-half MFMAs per block): the blocks of a PAIR of rows form one unit (first halves of all, then second
            // halves), so a block's second MFMA issues 8 MFMAs after its first (4 per row left 120 s_nops of MFMA dependency stalls)
            let rows_per_unit = if self.cfg.elem == Elem::Bf16 && nm % 2 == 0 { 2 } else if self.cfg.elem == Elem::Bf16W4 { nm } else { 1 };
            for (lo, hi) in [(0, half), (half, nn)] {
                for i0 in (0..nm).step_by(rows_per_unit) {
                    let rows = i0..i0 + rows_per_unit;
                    let mut blocks = Vec::new();
                    for i in rows.clone() {
                        let todo: Vec<usize> = (lo..hi).filter(|&j| !(lo == 0 && i * half + j < self.npre_done)).collect(); // the rest were issued before the barrier
                        let (w, by) = self.sa_sel(i);
                        if self.cfg.elem == Elem::Fp8 {  // one unit per block
                            for &j in &todo {
                                u.capture(b, format!("mma.i{i}j{j}"), |b| mma_var(b, &self.mma, self.cfg.elem, self.unit.map(|u| u.0), self.acc[nn * i + j], av[i], bv[j], sav[w], sbv[self.sb_sel(j)], by));
                            }
                        } else {
                            blocks.extend(todo.iter().map(|&j| (self.acc[nn * i + j], av[i], bv[j], sav[w], sbv[self.sb_sel(j)], by)));
                        }
                    }
                    if !blocks.is_empty() {
                        u.capture(b, format!("mma.i{i0}+{}h{lo}", rows_per_unit), |b| {
                            let p = mma_blocks(b, &self.mma, self.cfg.elem, self.unit.map(|u| u.0), &blocks, nm as u32, self.w4_shift);
                            self.pending.borrow_mut().extend(p);
                        });
                    }
                    for i in rows {
                        if let (true, Some((ar, _))) = (lo == half, &next) {
                            // in-place reload: the ds_reads write the variable's own AGPRs (coalesced set_var)
                            u.capture(b, format!("reload.a{i}"), |b| {
                                let v = ar.read(b, i);
                                b.set_var(av[i], v);
                            });
                            for (k, ah) in next_hi.iter().enumerate() {
                                u.capture(b, format!("reload.ah{k}.{i}"), |b| {
                                    let v = ah.read(b, i);
                                    b.set_var(av[(k + 1) * nm + i], v);
                                });
                            }
                        }
                    }
                }
                if let Some((_, br)) = &next {
                    for j in lo..hi {
                        u.capture(b, format!("reload.b{j}"), |b| {
                            let v = br.read(b, j);
                            b.set_var(bv[j], v);
                        });
                    }
                }
            }
        };
        // E3 proof hook for the reg-staged compute path: a real IR branch drops one tile's MMAs, reproducing the historical "tile 1 swallowed"
        // regression in emitted kernel code (tests/functional_gemm_ir.rs); the branch region is one captured unit
        match std::env::var("GEMM_PROOF_DROP_KB").ok().and_then(|v| v.parse::<i32>().ok()) {
            Some(kb) => {
                u.capture(b, "tile.proof_drop", |b| {
                    let keep = b.bind_cond(ox(t.it).ne(kb));
                    b.begin_if(keep);
                    let mut inner = Units::new();
                    body(b, &mut inner);
                    inner.emit_all(b);
                    b.end_if();
                });
            }
            None => body(b, &mut u),
        }
        // the deferred block-scale updates of this tile, before the scales of t + 1 replace the scale variables
        let pend: Vec<_> = self.pending.borrow_mut().drain(..).collect();
        if !pend.is_empty() {
            u.capture(b, "bs.fma", |b| bs_fma(b, &pend));
        }
        if let Some((sa, sb)) = early {
            u.capture(b, "scales.install", |b| {
                for (i, v) in sa.iter().enumerate() {
                    let w = self.scale_word(b, *v);
                    b.set_var(sav[i], w);
                }
                for (j, v) in sb.iter().enumerate() {
                    let w = self.scale_word(b, *v);
                    b.set_var(sbv[j], w);
                }
            });
        }
        let contract = amdgpu_dsl::StageContract { barrier_phase: phase, roles, ..Default::default() };
        amdgpu_tile::stage::StageInst::new("mma.blockscale_tile", amdgpu_tile::stage::Prim::A, "gemm.tile", contract, u)
            .param("atom", "mfma_scale_16x16x128_f8")
            .param("wave_tile", format!("{}x{}", 16 * nm, 16 * nn))
            .param("staged", t.staged)
    }    /// The DMA of one tile (A issues, then B issues): soffset = the wave-uniform tile origin, M0 = the first 1 KB block of each group of 4 issues.
    fn stage_now(&self, b: &mut FnBuilder, key0: &I32, slot: &SlotIdx) {
        let phase = amdgpu_dsl::StageContract::ANY_PHASE;
        // the slot offsets are bound before the region (shared with the readers through the SlotIdx cache)
        let _ = self.a_s.at(slot).off_v(b);
        let _ = self.b_s.at(slot).off_v(b);
        self.ab_stage(b, key0, slot, phase).emit(b);
    }
    /// Stage `move.ab_stage`: the DMA of one tile (A issues, then B issues); soffset = the wave-uniform tile origin, M0 = the first 1 KB block of
    /// each group of 4 issues; one captured unit per DMA instruction (each under the slot's write marker).
    fn ab_stage(&self, b: &mut FnBuilder, key0: &I32, slot: &SlotIdx, phase: u32) -> amdgpu_tile::stage::StageInst<'static> {
        let mut u = amdgpu_tile::weave::Units::new();
        let (_, orgs) = u.capture(b, "ab.setup", |b| {
            // A advances by key0 bytes, the preshuffled B by (key0 / 16) 256 = 16 key0; the tile origins are wave uniform: SGPR soffsets
            // mixed precision: A advances (images) key0 bytes (128 A bytes per image per 128-byte fp4 B tile)
            let nimg = self.cfg.elem.a_images();
            let ka = if nimg > 1 {
                let v = b.bind_i32((key0.clone() << (nimg.trailing_zeros() as i32)) + ev(self.a_org)); b.read_first_lane(v)
            } else {
                let v = b.bind_i32(key0.clone() + ev(self.a_org)); b.read_first_lane(v)
            };
            let kb = { let v = b.bind_i32((key0.clone() << 4) + ev(self.b_org)); b.read_first_lane(v) };
            // (mixed precision only: the other images' origins; other kernels keep their exact code)
            let mut o = vec![ka, kb];
            for k in 1..nimg {
                let v = b.bind_i32(ev(ka) + 128 * k as i32);
                o.push(b.read_first_lane(v));
            }
            o
        });
        // implicit-GEMM conv: this tile's channel block and tap (wave uniform), then per A issue the input pixel's voffset (or out of range)
        let conv_vo: Option<Vec<ValueId>> = self.conv.as_ref().map(|cv| {
            let (_, vo) = u.capture(b, "ab.conv", |b| {
                let cc = cv.cfg;
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
                let toff = b.bind_i32((((ev(dz) * cc.dil) * ev(cv.h) + ev(rr) * cc.dil) * ev(cv.w) + ev(ss) * cc.dil) * ev(cv.c) * 2 + (ev(cb) << 7));
                let big = b.c_i32(0x7fff_fff0);
                cv.rows
                    .iter()
                    .map(|&(base, id0, ih0, iw0)| {
                        let (zi, yi, xi) = (b.bind_i32(ev(id0) + ev(dz) * cc.dil), b.bind_i32(ev(ih0) + ev(rr) * cc.dil), b.bind_i32(ev(iw0) + ev(ss) * cc.dil));
                        let ok = b.bind_i32(
                            ev(zi).ge(0).select_i32(1, 0) & ev(zi).lt(ev(cv.di)).select_i32(1, 0) & ev(yi).ge(0).select_i32(1, 0) & ev(yi).lt(ev(cv.h)).select_i32(1, 0)
                                & ev(xi).ge(0).select_i32(1, 0) & ev(xi).lt(ev(cv.w)).select_i32(1, 0),
                        );
                        let ok = b.bind_cond(ev(ok).eq(1));
                        let v = b.bind_i32(ev(base) + ev(toff));
                        b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(ok), a: Opnd::Val(v), b: Opnd::Val(big) })
                    })
                    .collect::<Vec<_>>()
            });
            vo
        });
        // (A8W4: a third pass, the hi A image: LDS + a_hi, global + 128 bytes)
        let mut passes = vec![(&self.a_s, &self.a_dma, self.rsrc_a, self.a_wave, 0usize, 0i32), (&self.b_s, &self.b_dma, self.rsrc_b, self.b_wave, 1, 0)];
        for k in 1..self.cfg.elem.a_images() {
            passes.push((&self.a_s, &self.a_dma, self.rsrc_a, self.a_wave, k + 1, k as i32 * self.a_hi));
        }
        for (bi, (sh, dma, rsrc, wave, oi, lds_off)) in passes.into_iter().enumerate() {
            let uname = if bi < 2 { sh.name.to_string() } else if bi == 2 { format!("{}h", sh.name) } else { format!("{}h{bi}", sh.name) };
            let (_, mine) = u.capture(b, format!("ab.{uname}.setup"), |b| {
                let o = sh.at(slot).off_v(b);
                let sbase = if lds_off == 0 { b.bind_i32(o + sh.base as i32 + ev(wave)) } else { b.bind_i32(o + (sh.base as i32 + lds_off) + ev(wave)) };
                let mut m0s: Vec<(i32, ValueId)> = Vec::new();
                let mut mine: Vec<(ValueId, i32, ValueId)> = Vec::new();
                for (g, lds) in dma.issues.iter() {
                    let grp = dma.group(*lds);
                    if !m0s.iter().any(|x| x.0 == grp) {
                        let v = b.bind_i32(ev(sbase) + dma.act(*lds).0);
                        let v = b.read_first_lane(v);
                        m0s.push((grp, v));
                    }
                    mine.push((*g, *lds, m0s.iter().find(|x| x.0 == grp).unwrap().1));
                }
                mine
            });
            for (q, &(g, lds, m0)) in mine.iter().enumerate() {
                // buffer_load_dwordx4 ... lds: voffset = the lane's loop-invariant offset, soffset = the tile origin (wave uniform), M0 = the first 1 KB
                // block of each group of 4 issues, instruction offset 1024 (q % 4) moves both the LDS and the global address
                let (g, so) = match (&conv_vo, bi) {
                    (Some(vo), 0) => (vo[q], self.conv.as_ref().map(|c| c.zero).unwrap()),
                    _ => (g, orgs[oi]),
                };
                u.capture(b, format!("ab.{uname}.{q}"), |b| sh.at(slot).write(b, |b| b.load_buffer_lds(rsrc, g, so, m0, dma.act(lds).1, 4)));
            }
        }
        let contract = amdgpu_dsl::StageContract { barrier_phase: phase, roles: vec!["all".to_string()], ..Default::default() };
        amdgpu_tile::stage::StageInst::new("move.ab_stage", amdgpu_tile::stage::Prim::M, "gemm.ab_stage", contract, u).param("slots", 2)
    }}

impl PipeBody for Body {
    fn stage(&mut self, b: &mut FnBuilder, _role: Role, key0: &I32, slot: &SlotIdx) {
        self.stage_now(b, key0, slot);
    }
    fn prologue_stores(&mut self, b: &mut FnBuilder, _role: Role) {
        if self.cfg.scales == Scales::Mx128 {
            let p = self.panel.as_ref().expect("scale panel");
            let sj = &p.sfa;
            let pr = p.sh.at(&SlotIdx::new(0, 0));
            // loads (A: 4 x 16 bytes, B: one dword), then the transposes and the panel writes; published by the pipeline's prologue barrier
            let fr: Vec<ValueId> = sj.pso.iter().map(|so| b.buffer_load_b128(self.rsrc_sfa, sj.voff, *so, 0)).collect();
            let bw = b.buffer_load_f32(self.rsrc_sfb, sj.bvo, 0i32, 0);
            let bwi = b.bitcast_i32(bw);
            let bval = b.bind_i32(ev(bwi).lshr(ev(sj.bsh)) & 255);
            pr.write(b, |b| {
                for (pi, f) in fr.iter().enumerate() {
                    for d in 0..4u8 {
                        let xl = b.frag4_lane(Opnd::Val(*f), d);
                        let x = b.bitcast_i32(xl);
                        let y = b.dpp_quad_perm(x, [1, 0, 3, 2]);
                        let y = b.bitcast_i32(y);
                        let n1 = b.perm_b32(y, x, sj.sel1);
                        let z = b.dpp_quad_perm(n1, [2, 3, 0, 1]);
                        let z = b.bitcast_i32(z);
                        let out = b.perm_b32(z, n1, sj.sel2);
                        let a = b.bind_i32(ev(sj.dst) + (4096 * pi as i32 + 32 * d as i32));
                        b.store_lds_b32(a, out);
                    }
                }
                b.store_lds_b32(sj.bdst, bval);
            });
        }
    }
    fn prologue_compute(&mut self, b: &mut FnBuilder, _role: Role, t0: &TileIt) {
        if let Some((buf, rec, _)) = self.stamp {
            let st = b.memtime_lo();
            b.stamp_store(buf, rec, 9, st); // slot 9: prologue wait + barrier passed (tiles 0/1 + transposed panel visible)
        }
        self.load_tile(b, t0);
        if let Some((buf, rec, _)) = self.stamp {
            let st = b.memtime_lo();
            b.stamp_store(buf, rec, 2, st); // slot 2: prologue done (tile 0 fragments requested), about to enter the tile loop
        }
    }
    fn loop_state(&mut self, _b: &mut FnBuilder) -> Vec<(DType, Opnd)> {
        let mut v: Vec<(DType, Opnd)> = (0..(self.cfg.wave_m() / 16) * (self.cfg.wave_n() / 16)).map(|_| (DType::Frag4, Opnd::Val(self.zero4))).collect();
        if let Some((bf, af, sa, sb)) = &self.staged {
            v.extend(bf.iter().chain(af.iter()).map(|x| (DType::Frag8, Opnd::Val(*x))));
            v.extend(sa.iter().map(|x| (DType::I32, Opnd::Val(*x))));
            v.extend(sb.iter().map(|x| (DType::I32, Opnd::Val(*x))));
        }
        v
    }
    fn bind_state(&mut self, vars: &[VarId]) {
        let na = ((self.cfg.wave_m() / 16) * (self.cfg.wave_n() / 16)) as usize;
        self.acc = vars[..na].to_vec();
        if let Some((bf, af, sa, sb)) = self.staged.take() {
            let (nb, nf, ns, nsb) = (bf.len(), af.len(), sa.len(), sb.len());
            let r = &vars[na..];
            self.rv = Some((r[..nb].to_vec(), r[nb..nb + nf].to_vec(), r[nb + nf..nb + nf + ns].to_vec(), r[nb + nf + ns..nb + nf + ns + nsb].to_vec()));
            let nimg = self.cfg.elem.a_images();
            if nimg > 1 {
                let av = &r[nb..nb + nf];
                let nm = nf / nimg;
                assert!((1..nimg).all(|k| (0..nm).all(|i| av[k * nm + i].0 == av[i].0 + (k * nm) as u32)), "mixed precision: the A image variables are consecutive");
            }
        }
    }
    /// The next MMAs of the register tile between the waits and the barrier (the vendor issues 3 there): the barrier latency hides behind them.
    fn post_wait(&mut self, b: &mut FnBuilder, _role: Role, _t: &TileIt) {
        if self.cfg.mid_sync_mmas > 0 {
            let from = self.npre_done;
            self.early_mmas(b, from, from + self.cfg.mid_sync_mmas);
        }
    }
    /// The first MMAs of the register tile run while the wave waits for the DMA and the barrier (the vendor issues 4 there).
    fn pre_sync(&mut self, b: &mut FnBuilder, _role: Role, _t: &TileIt) {
        let npre = self.cfg.pre_sync_mmas;
        self.early_mmas(b, 0, npre);
    }
    fn m_phase(&mut self, b: &mut FnBuilder, _role: Role, t: &TileIt) {
        if let Some((buf, rec, true)) = self.stamp {
            // slot 16 + tile: the wave reached this tile's matrix phase (after the tile's top wait + barrier and its DMA issue)
            let st = b.memtime_lo();
            let four = b.c_i32(4);
            let ti = b.mul_i32(t.it, four);
            let off = b.add_i32(rec, ti);
            b.stamp_store(buf, off, 16, st);
        }
        self.compute_tile(b, t);
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
        "gemm_a8w8_rowcol_fp8_128" => Some(gemm_mxfp8_bs("gemm_a8w8_rowcol_fp8_128", GemmConfig::a8w8_rowcol_128())),
        "gemm_a16w16_bf16" => Some(gemm_mxfp8_bs("gemm_a16w16_bf16", GemmConfig::a16w16())),
        "gemm_a16w16_bf16_128" => Some(gemm_mxfp8_bs("gemm_a16w16_bf16_128", GemmConfig::a16w16_128())),
        "gemm_a8w8_blockscale" => Some(gemm_mxfp8_bs("gemm_a8w8_blockscale", GemmConfig::a8w8_blockscale())),
        "gemm_a8w8_blockscale_splitk" => Some(gemm_mxfp8_bs("gemm_a8w8_blockscale_splitk", GemmConfig::a8w8_blockscale().splitk())),
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
            } else if n.contains("_fp4") {
                // MXFP4 A x MXFP4 B (1 x 32 E8M0, unshuffled [rows][K / 32])
                (Elem::Fp4, Scales::Mx32)
            } else {
                (Elem::Bf16, Scales::None)
            };
            // `_bm64` suffix: block_m 64 (64-row tiles, sparse experts)
            let (n0, bm64) = match n.strip_suffix("_bm64") { Some(x) => (x, true), None => (n, false) };
            let st = if n0.starts_with("gemm_moe1") {
                MoeStage::Up { gelu: n0.contains("_gelu"), doweight: n0.ends_with("_dw") }
            } else if n0.ends_with("_h") {
                MoeStage::DownH
            } else {
                MoeStage::Down
            };
            let cfg = GemmConfig::moe(elem, scales, st);
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
    declare_contract(b, cta_m, cta_n, cfg.scales, cfg.elem.per_tile(), cfg.split_k, cfg.moe, cfg.conv, cfg.batched, cfg.per_tensor, cfg.elem.a_images());
    // timing stamps (opt-in debug build, GEMM_STAMP=ends|tiles): one extra pointer argument after the ABI (offset 96); see docs/stamps.md
    let stamp_mode = std::env::var("GEMM_STAMP").ok().filter(|v| v == "ends" || v == "tiles");
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
        let okv = b.bind_i32((ev(n_v) ^ (16 * cfg.cta_n) as i32) | (ev(m_v) ^ (16 * cfg.cta_m) as i32));
        let id = b.bind_i32(ev(bx0) + (ev(by0) << 4));
        let loc = b.bind_i32(ev(id).lshr(3));
        // block (of 16) = 2 (id % 8) + loc / 16 -> (row, col) = (block / 4, block % 4); inside the block: loc % 16 -> (n4, m4)
        let blk = b.bind_i32(((ev(id) & 7) << 1) + ev(loc).lshr(4));
        let (bc, br) = (ev(blk) & 3, ev(blk).lshr(2));
        let nx = b.bind_i32((bc << 2) + (ev(loc) & 3));
        let ny = b.bind_i32((br << 2) + (ev(loc).lshr(2) & 3));
        let s = b.bind_i32(ev(okv).eq(0).select_i32(1, 0));
        let px = b.bind_i32(ev(bx0) + ev(s) * (ev(nx) - ev(bx0)));
        let py = b.bind_i32(ev(by0) + ev(s) * (ev(ny) - ev(by0)));
        b.assume_trusted(px, Ext::c(0), Ext::sym("NT") - Ext::c(1), "XCD raster: (px, py) is a permutation of the 16 x 16 tile grid when the select applies (N = 16 cta_n, M = 16 cta_m: 4 x 4 blocks, 2 per XCD), the identity (workgroup ids) otherwise");
        b.assume_trusted(py, Ext::c(0), Ext::sym("MT") - Ext::c(1), "XCD raster: see px");
        (px, py)
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
    let slots = 2;
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
            dma_plan_gather(b, &lane, cta_m, 4, k_v, &|b, r| {
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
    let body = Body {
        cfg, kb_v: ks_v, panel, mma: mma.clone(), a_s, b_s, a_dma, b_dma, a_org, b_org, a_wave, b_wave, rsrc_a, rsrc_b, rsrc_sfa, rsrc_sfb,
        a_rd_x: (1..nimg).map(|k| { let mut h = a_rd.clone(); h.bases = h.bases.iter().map(|x| b.let_i32(x.clone() + (k * a_bytes) as i32)).collect(); h }).collect(),
        a_hi: a_bytes as i32,
        w4_shift: (cfg.elem == Elem::Bf16W4).then(|| b.bind_i32((lane.lane().lshr(5)) << 3)),
        a_rd, b_rd, zero4, acc: Vec::new(), staged: None, rv: None, npre_done: 0, stamp: stamp_ctx, unit, mx32, bs128, conv: conv_rt, pending: Default::default(), zb,
    };
        // MoE: everything after this runs only for tiles below num_valid_ids[0] (closed at the end of the epilogue)
    if let Some(mc) = &moe_cx {
        b.begin_if(Opnd::Val(mc.live));
    }
    GemmCx { cfg, moe: moe_cx, rsrc_sfa_e: Some((rsrc_sfa, rsrc_sfb)), body, pipe: Some(pipe), ctx, mma, epi, lane, s_lane, s_n, s_wm, s_wn, ptr_c, m_v, n_v, bx, by, wm, wn, lg_m, lg_n, wave_m, wave_n, stamp_ctx }
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
        let bases: Vec<ValueId> = plan.bases.iter().map(|e| {
            let o = b.bind_i32(ctx.lower(e));
            b.ptr_add(cb, o)
        }).collect();
        for (n, acc) in body.acc.iter().enumerate() {
            let (bi, imm) = plan.issues[n];
            b.store_f32x4(bases[bi], imm as u32, Opnd::Var(*acc));
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

/// The MoE epilogue straight from the accumulators (lane: row = the c_tv row, 4 consecutive columns): per m block the sorted row's
/// (token, slot) and routing weight; stage 1: act(gate) up of the interleaved block pairs -> bf16 [M topk, n_out] at row token topk + slot;
/// stage 2: fp32 [M topk, n_out] x the weight. Padding rows (token >= M) store past the descriptor (dropped).
#[allow(clippy::too_many_arguments)]
fn moe_epilogue(
    b: &mut FnBuilder, ctx: &Ctx, mma: &TiledMma, acc: &[VarId], s_lane: &Ix, mc: &MoeCx, cfg: Cfg, rsa: ValueId, rsb: ValueId, ptr_c: ValueId, m_v: ValueId,
    n_v: ValueId, bx: ValueId, wm: ValueId, wn: ValueId, wave_m: i64, wave_n: i64,
) {
    let d = &ctx.dyns;
    let (nm, nn) = ((wave_m / 16) as usize, (wave_n / 16) as usize);
    let idx = IssuePlan::new(mma.c_tv(), 4).offsets(&make_layout(tup![16, 16], tup![1024, 1]), s_lane, &Ix::C(0), d);
    let o = b.bind_i32(ctx.lower(&idx[0]));
    let (rl, cl) = (b.bind_i32(ev(o).lshr(10)), b.bind_i32(ev(o) & 1023));
    let up = matches!(mc.st, MoeStage::Up { .. });
    let half_out = mc.st == MoeStage::DownH;
    let esz = if up || half_out { 2 } else { 4 };
    let rsrc_c = { let n = b.bind_i32(ev(mc.mtok) * ev(mc.topk) * ev(mc.nout) * esz); let n = b.read_first_lane(n); b.buffer_desc_dyn(ptr_c, n) };
    let so = { let v = b.bind_i32(ev(mc.row0) << 2); b.read_first_lane(v) };
    let big = b.c_i32(0x7fff_fff0);
    let c0 = b.c_i32(0);
    // the first output column of this lane in the wave
    let col0 = if up {
        b.bind_i32((ev(bx) << (cfg.cta_n.trailing_zeros() as i32 - 1)) + ev(wn) * (wave_n as i32 / 2) + ev(cl))
    } else {
        b.bind_i32((ev(bx) << cfg.cta_n.trailing_zeros() as i32) + ev(wn) * (wave_n as i32) + ev(cl))
    };
    let l2e = b.c_f32(std::f32::consts::LOG2_E);
    let one = b.c_f32(1.0);
    for i in 0..nm {
        let sr = b.bind_i32(ev(wm) * (wave_m as i32) + 16 * i as i32 + ev(rl));
        let vo = b.bind_i32(ev(sr) << 2);
        let sid = b.buffer_load_f32(mc.rsrc_sid, vo, so, 0);
        let sid = b.bitcast_i32(sid);
        // the routing weight only where it is used (a dead load's destination must not be reused while it is in flight: backend issue)
        let use_w = !matches!(mc.st, MoeStage::Up { doweight: false, .. });
        let sw = if use_w { b.buffer_load_f32(mc.rsrc_sw, vo, so, 0) } else { b.c_f32(1.0) };
        let tok = b.bind_i32(ev(sid) & 0xff_ffff);
        let orow = b.bind_i32(ev(tok) * ev(mc.topk) + ev(sid).lshr(24));
        let live = b.bind_cond(ev(tok).lt(ev(mc.mtok)));
        // per-token (A row) scale
        let xs = (cfg.scales == Scales::RowCol).then(|| {
            let arow = if up { tok } else { orow };
            let ar = b.bind_i32(ev(arow).min(ev(m_v) - 1) << 2);
            b.buffer_load_f32(rsa, ar, 0i32, 0)
        });
        let wsc = |b: &mut FnBuilder, brow: ValueId| -> Vec<ValueId> {
            let a = b.bind_i32(ev(brow) << 2); // the expert is in the descriptor base (rsrc_sfb)
            let f = b.buffer_load_b128(rsb, a, 0i32, 0);
            (0..4u8).map(|k| { let x = b.frag4_lane(f, k); b.bitcast_f32(x) }).collect()
        };
        let vals = |b: &mut FnBuilder, n: usize| -> Vec<ValueId> {
            (0..4u8).map(|r| { let l = b.frag4_lane(Opnd::Var(acc[n]), r); b.bitcast_f32(l) }).collect()
        };
        if up {
            let MoeStage::Up { gelu, doweight } = mc.st else { unreachable!() };
            // the row / first column part of the store offset (padding rows: past the descriptor); the 16-column block p in the instruction offset
            let vr0 = b.bind_i32((ev(orow) * ev(mc.nout) + ev(col0)) << 1);
            let vrow = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(live), a: Opnd::Val(vr0), b: Opnd::Val(big) });
            for p in 0..nn / 2 {
                let col = b.bind_i32(ev(col0) + 16 * p as i32);
                let mut g = vals(b, nn * i + 2 * p);
                let mut u = vals(b, nn * i + 2 * p + 1);
                if let Some(xs) = xs {
                    // the weight scales follow the interleaved row order: output column c = 16 q + r -> gate row 32 q + r, up row 32 q + 16 + r
                    let grow = b.bind_i32(((ev(col).lshr(4)) << 5) + (ev(col) & 15));
                    let urow = b.bind_i32(ev(grow) + 16);
                    let (wg, wu) = (wsc(b, grow), wsc(b, urow));
                    for k in 0..4 {
                        let sg = b.mul_f32(xs, wg[k]);
                        g[k] = b.mul_f32(g[k], sg);
                        let su = b.mul_f32(xs, wu[k]);
                        u[k] = b.mul_f32(u[k], su);
                    }
                }
                let y: Vec<ValueId> = (0..4)
                    .map(|k| {
                        let a = if gelu {
                            // gelu (tanh form): g (1 - 1 / (exp(2 t) + 1)), t = 0.7978845608 (g + 0.044715 g^3)
                            let g2 = b.mul_f32(g[k], g[k]);
                            let c1 = b.c_f32(0.044715);
                            let inner = b.let_(Op::FmaF32(Opnd::Val(g2), Opnd::Val(c1), Opnd::Val(one)));
                            let t0 = b.mul_f32(g[k], inner);
                            let c2 = b.c_f32(2.0 * 0.797_884_56 * std::f32::consts::LOG2_E);
                            let x2 = b.mul_f32(t0, c2);
                            let ex = b.exp2_f32(x2);
                            let den = b.add_f32(ex, one);
                            let r = b.rcp_f32(den);
                            let om = b.sub_f32(one, r);
                            b.mul_f32(g[k], om)
                        } else {
                            // silu: g / (1 + exp(-g))
                            let zf = b.c_f32(0.0);
                            let ng = b.sub_f32(zf, g[k]);
                            let x = b.mul_f32(ng, l2e);
                            let ex = b.exp2_f32(x);
                            let den = b.add_f32(ex, one);
                            let r = b.rcp_f32(den);
                            b.mul_f32(g[k], r)
                        };
                        let yy = b.mul_f32(a, u[k]);
                        if doweight { b.mul_f32(yy, sw) } else { yy }
                    })
                    .collect();
                let w0 = b.cvt_pk_bf16_f32(y[0], y[1]);
                let w1 = b.cvt_pk_bf16_f32(y[2], y[3]);
                let f2 = b.frag2(w0, w1);
                b.buffer_store_b64(rsrc_c, vrow, c0, (32 * p) as u32, f2);
            }
        } else {
            let vr0 = if half_out {
                b.bind_i32((ev(orow) * ev(mc.nout) + ev(col0)) << 1)
            } else {
                b.bind_i32((ev(orow) * ev(mc.nout) + ev(col0)) << 2)
            };
            let vrow = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(live), a: Opnd::Val(vr0), b: Opnd::Val(big) });
            for j in 0..nn {
                let col = b.bind_i32(ev(col0) + 16 * j as i32);
                let mut v = vals(b, nn * i + j);
                let ws = xs.map(|_| wsc(b, col));
                for k in 0..4 {
                    if let (Some(xs), Some(ws)) = (xs, &ws) {
                        let s = b.mul_f32(xs, ws[k]);
                        v[k] = b.mul_f32(v[k], s);
                    }
                    let y = b.mul_f32(v[k], sw);
                    v[k] = if half_out { y } else { b.bitcast_i32(y) };
                }
                if half_out {
                    let w0 = b.cvt_pk_bf16_f32(v[0], v[1]);
                    let w1 = b.cvt_pk_bf16_f32(v[2], v[3]);
                    let f2 = b.frag2(w0, w1);
                    b.buffer_store_b64(rsrc_c, vrow, c0, (32 * j) as u32, f2);
                    continue;
                }
                let f4 = b.frag4(v[0], v[1], v[2], v[3]);
                b.buffer_store_b128(rsrc_c, vrow, c0, (64 * j) as u32, f4);
            }
        }
    }
}

/// The LDS-staged epilogue: every accumulator (16 x 16, lane = row m, 4 consecutive n) is converted to bf16 and written to the wave's staging
/// block E[m][n] (row pitch EPI_PITCH); then each lane reads 16 bytes = 8 consecutive n of one row and stores them with one `buffer_store_dwordx4`
/// (16 lanes cover a 256-byte row segment). All addresses from layouts: the C TV layout of the atom over E, a row-contiguous read TV layout over E
/// and the same TV layout over the global C (voffset per lane, the wave-uniform per-issue row step in the SGPR soffset).
#[allow(clippy::too_many_arguments)]
fn lds_epilogue(
    b: &mut FnBuilder, ctx: &Ctx, mma: &TiledMma, epi: &Shared, acc: &[VarId], lane: &Lane, s_lane: &Ix, s_n: &Ix, ptr_c: ValueId, m_v: ValueId,
    n_v: ValueId, bx: ValueId, by: ValueId, wm: ValueId, wn: ValueId, lg_m: i32, lg_n: i32, wave_m: i64, wave_n: i64, phases: usize,
    rowcol: Option<(ValueId, ValueId)>,
    per_tensor: bool,
) {
    let d = &ctx.dyns;
    let (nm, nn) = ((wave_m / 16) as usize, (wave_n / 16) as usize);
    assert!(wave_n % 8 == 0 && 64 % (wave_n / 8) == 0 && wave_n * 2 <= EPI_PITCH, "lds_epilogue: wave rows of 8-column chunks that tile a wave");
    // per-token x per-channel scales: the lane's row of every m block (one f32) and its 4 consecutive columns of every n block (one 16-byte
    // load), from the C TV layout of the atom over a (row 1024 + col) index; loaded before the staging barrier
    let scl: Option<(Vec<ValueId>, Vec<ValueId>)> = rowcol.map(|(rs_a, rs_b)| {
        if per_tensor {
            // one A and one B scale (the whole tensor): the same value for every row / column
            let z = b.c_i32(0);
            let sa = b.buffer_load_f32(rs_a, z, 0i32, 0);
            let sbv = b.buffer_load_f32(rs_b, z, 0i32, 0);
            let sbi = b.bitcast_i32(sbv);
            let sb4 = b.frag4(sbi, sbi, sbi, sbi);
            return ((0..nm).map(|_| sa).collect(), (0..nn).map(|_| sb4).collect());
        }
        let idx = IssuePlan::new(mma.c_tv(), 4).offsets(&make_layout(tup![16, 16], tup![1024, 1]), s_lane, &Ix::C(0), d);
        let o = b.bind_i32(ctx.lower(&idx[0]));
        let row = b.bind_i32((ev(by) << lg_m) + ev(wm) * (wave_m as i32) + ev(o).lshr(10));
        let col = b.bind_i32((ev(bx) << lg_n) + ev(wn) * (wave_n as i32) + (ev(o) & 1023));
        let va = b.bind_i32(ev(row) << 2);
        let vb = b.bind_i32(ev(col) << 2);
        let sa: Vec<ValueId> = (0..nm).map(|i| b.buffer_load_f32(rs_a, va, 0i32, (64 * i) as u32)).collect();
        let sb: Vec<ValueId> = (0..nn).map(|j| b.buffer_load_b128(rs_b, vb, 0i32, (64 * j) as u32)).collect();
        (sa, sb)
    });
    // every wave is done with the ring / panel storage the staging reuses
    b.wait_lds(0);
    b.barrier();
    let wid = b.bind_i32(lane.wave());
    let me = epi.mine(wid);
    let base = b.bind_i32(me.base());
    // staging layout over the C block element index m + 16 n of one 16 x 16 block: (m, n) -> m EPI_PITCH + 2 n
    let e_blk = make_layout(tup![16, 16], tup![EPI_PITCH, 2]);
    let w_plan = IssuePlan::new(mma.c_tv(), 4); // 4 bf16 = one ds_write_b64
    let mut exprs = Vec::new();
    for i in 0..nm as i64 {
        for j in 0..nn as i64 {
            exprs.extend(w_plan.offsets(&e_blk, s_lane, &Ix::C(16 * i * EPI_PITCH + 32 * j), d));
        }
    }
    let wp = layout_to_offsets(&exprs, &Swizzle::none(), ImmRange::DS, d);
    let wbases: Vec<ValueId> = wp.bases.iter().map(|e| b.bind_i32(ev(base) + ctx.lower(e))).collect();
    // read / store TV over the wave block element index n + wave_n m: lane -> (chunk = lane % 16: 8 n, row = lane / 16), value -> (8 n, row step 4)
    let rows_per = 64 / (wave_n / 8); // 4
    let issues = wave_m / rows_per; // 32
    let tv = make_layout(tup![tup![wave_n / 8, rows_per], tup![8, issues]], tup![tup![8, wave_n], tup![1, wave_n * rows_per]]);
    let e_blk2 = make_layout(tup![wave_n, wave_m], tup![2, EPI_PITCH]);
    let rp = IssuePlan::new(tv.clone(), 8);
    let rexprs = rp.offsets(&e_blk2, s_lane, &Ix::C(0), d);
    let rpl = layout_to_offsets(&rexprs, &Swizzle::none(), ImmRange::DS, d);
    let rbases: Vec<ValueId> = rpl.bases.iter().map(|e| b.bind_i32(ev(base) + ctx.lower(e))).collect();
    // global: C row-major bf16, the wave block at rows m_cta + wave_m wm, cols n_cta + wave_n wn
    let g_blk = make_layout(tup![wave_n, wave_m], tup![2, s_n.mul_c(2)]);
    let gofs = rp.offsets(&g_blk, s_lane, &Ix::C(0), d);
    let voff = b.bind_i32(ctx.lower(&gofs[0]) + (ev(wm) * (wave_m as i32) * ev(n_v) + ev(wn) * (wave_n as i32)) * 2);
    let rsrc_c = {
        let nr = b.bind_i32(ev(m_v) * ev(n_v) * 2);
        let nr = b.read_first_lane(nr);
        b.buffer_desc_dyn(ptr_c, nr)
    };
    let c_org = { let v = b.bind_i32((((ev(by) << lg_m) * ev(n_v)) + (ev(bx) << lg_n)) << 1); b.read_first_lane(v) };
    // phases of wave_m / phases rows: the conversion + LDS writes of phase p+1 are issued before the stores of phase p (which only wait for
    // phase p's reads), so conversion, LDS traffic and global stores overlap (the vendor stages quadrant by quadrant)
    // a phase writes whole 16-row blocks (all nn column blocks) and reads back exactly those rows: at most nm phases, dividing nm
    let mut phases = phases.clamp(1, nm);
    while nm % phases != 0 {
        phases -= 1;
    }
    let accs_per = acc.len() / phases;
    let iss_per = issues as usize / phases;
    assert_eq!(accs_per % nn, 0, "lds_epilogue: a phase covers whole row blocks");
    let write_phase = |b: &mut FnBuilder, p: usize| {
        me.write(b, p as i32, |b| {
            for n in p * accs_per..(p + 1) * accs_per {
                let a = acc[n];
                let mut v: Vec<ValueId> = (0..4u8).map(|r| {
                    let l = b.frag4_lane(Opnd::Var(a), r);
                    b.bitcast_f32(l)
                }).collect();
                if let Some((sa, sb)) = &scl {
                    // acc n = block (n / nn, n % nn): times the row scale and the column scales
                    let (i, j) = (n / nn, n % nn);
                    for (r, x) in v.iter_mut().enumerate() {
                        let cl = b.frag4_lane(Opnd::Val(sb[j]), r as u8);
                        let c = b.bitcast_f32(cl);
                        let s = b.mul_f32(sa[i], c);
                        *x = b.mul_f32(*x, s);
                    }
                }
                // 4 bf16 = one ds_write_b64
                let w0 = b.cvt_pk_bf16_f32(v[0], v[1]);
                let w1 = b.cvt_pk_bf16_f32(v[2], v[3]);
                let f2 = b.frag2(w0, w1);
                let (bi, imm) = wp.issues[n];
                let ad = b.bind_i32(ev(wbases[bi]) + imm as i32);
                b.store_lds_b64(ad, f2);
            }
        });
    };
    write_phase(b, 0);
    for p in 0..phases {
        b.wait_lds(0);
        let vals: Vec<ValueId> = me.read_back(b, p as i32, |b| {
            rpl.issues[p * iss_per..(p + 1) * iss_per].iter().map(|&(bi, imm)| {
                let ad = b.bind_i32(ev(rbases[bi]) + imm as i32);
                b.load_lds_b128(ad)
            }).collect()
        });
        if p + 1 < phases {
            write_phase(b, p + 1);
        }
        for (k, v) in vals.iter().enumerate() {
            let q = p * iss_per + k;
            // the per-issue step is wave-uniform (rows_per rows): checked on the layout expressions
            let dq = gofs[q].sub(&gofs[0]);
            let dv = b.bind_i32(ev(c_org) + ctx.lower(&dq));
            let so = b.read_first_lane(dv);
            b.buffer_store_b128(rsrc_c, voff, so, 0, *v);
        }
    }
}