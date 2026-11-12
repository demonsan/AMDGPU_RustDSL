//! Host-side dispatch rules for the shipped kernel families (measured same-run against the vendor kernels; docs/op-coverage-design.md
//! §7.4 / §7.5). Pure functions of the problem shape and the CU count: a launcher picks the kernel name and the grid with them.

/// fp8 blockscale GEMM: (CTA rows, split-K S). 128 x 128 (`gemm_a8w8_blockscale`) when that grid is at least two CTAs per CU; else 64 x 128
/// (`gemm_a8w8_blockscale_64`, `_64_splitk` + `gemm_splitk_reduce_s<S>`) with the smallest S <= 4 (power of two, K whole 128-groups per
/// part) giving S CTAs >= CUs. Same run (us, ours / best aiter BpreShuffle tile): 1024x4096x8192 68.0 / 62.5 (was 71.4 with 128 S2),
/// 512x7168x7168 60.3 / 51.3 (64.1), 256x7168x4096 S2 25.3 / 16.9 (29.4), 128x4096x8192 S4 20.9 / 22.3 (27.1), 64x8192^2 S4 22.5 / 23.0
/// (27.2); 4096^3 and 2048x7168^2 stay 128 x 128 (122.7 / 129.3, 185.2 / 187.2).
/// The 64-row forms run the 3-slot ring with the scales through the DMA stage (`GemmConfig::depth3`): 1024x4096x8192 64.1, 512x7168^2 51.0,
/// 256x7168x4096 18.5 (S 1 now best; the rule's S 2 23.6), 128x4096x8192 S4 15.4, 64x8192^2 S4 19.2 us.
pub fn blockscale_config(m: u64, n: u64, k: u64, cus: u64) -> (u32, u32) {
    if m.div_ceil(128) * n.div_ceil(128) >= 2 * cus {
        return (128, 1);
    }
    let c = m.div_ceil(64) * n.div_ceil(128);
    // (3-slot 64-row forms: a grid of >= 3/4 of the CUs runs unsplit - 256x7168: S1 18.5 vs S2 23.6 us blockscale)
    let mut s = 1u64;
    while s < 4 && c * s < cus && 4 * c < 3 * cus && k % (256 * s) == 0 {
        s *= 2;
    }
    (64, s as u32)
}
/// bf16 prefill (`fmha_fwd_hd128_bf16_v3[_causal]`): the number of key parts S of the `_ks` form (1 = the plain kernel). Small grids (fewer
/// CTAs than CUs; causal grids are halved by the q-tile pairing) split each q tile's key range: measured B1 H8 4096^2 533 -> 601 TF/s (S = 2),
/// causal 259 -> 346 (S = 4), causal B2 H8 1000x1300 78 -> 94; S <= 4.
pub fn bf16_prefill_key_splits(b: u64, h: u64, sq: u64, sk: u64, causal: bool, cus: u64) -> u32 {
    let tiles = sq.div_ceil(256);
    let ctas = b * h * if causal { tiles.div_ceil(2) } else { tiles };
    let mut s = 1u64;
    // at least two 128-key tiles per part
    while s < 4 && ctas * s < cus && sk.div_ceil(128) >= 4 * s {
        s *= 2;
    }
    s as u32
}

/// The decode16 kernel for a launch (bf16, hd128, G <= 16).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decode16Plan {
    /// `pa_decode16_bf16_hd128_p<P>_f`: S = 1, the CTA's 4 wave partials merged in LDS, bf16 out, no combine kernel
    Fused,
    /// `pa_decode16_bf16_hd128_p<P>_c` with S splits: the last-arriving CTA of a (b, hk) merges the S partials (single-lane ticket), no
    /// combine kernel
    LastC(u32),
}

/// decode16 dispatch (B, HK, context length): grids of n = B HK >= 256 work groups run the fused form; smaller grids the last-arriver form
/// `_c` with the smallest power-of-two S giving n S >= 256 CTAs, capped at ctx / 512 keys per split (at least 1; S = 1 runs the fused form).
/// Measured (H 64, HK 8, page 16, us; same run; aiter `pa_bf16_noquant_gqa8_1tg_4w` in brackets) - the previous rule (split + combine
/// kernel with S = smallest n S >= 512, 64-key tiles `_k64` for n < 64; removed) in parentheses:
///   B4   ctx 1024 / 2048 / 8192: `_c` S2 11.9 / S4 15.7 / S8 28.8   (18.6 / 22.4 / 33.0)   [8.0 / 14.3 / 35.7]
///   B8                          : `_c` S2 12.0 / S4 19.6 / S4 40.5   (25.7 / 31.5 / 53.0)   [8.1 / 14.8 / 42.8]
///   B16                         : `_c` S2 15.8 / S2 24.1 / S2 89.8   (24.0 / 31.9 / 98.6)   [11.0 / 18.5 / 88.3]
///   B32                         : fused  20.4 / 35.8 / 166.6        (26.7 / 42.1 / 175.6)  [17.0 / 31.2 / 149.6]
///   B64, B128                   : fused (unchanged) 33.9 / 83.8 / 303.0, 81.9 / 157.8 / 588.3   [30.6 / 77.4 / 271.6, 74.6 / 146.5 / 540.8]
/// (`_c` with S = 1 and the split forms were slower than these everywhere in the grid.)
pub fn decode16_plan(b: u64, hk: u64, ctx: u64) -> Decode16Plan {
    let n = b * hk;
    if n >= 256 {
        return Decode16Plan::Fused;
    }
    let cap = (ctx / 512).max(1);
    let mut s = 1u64;
    while n * s < 256 && s * 2 <= cap {
        s *= 2;
    }
    if s == 1 {
        Decode16Plan::Fused
    } else {
        Decode16Plan::LastC(s as u32)
    }
}

/// The kernel name and split count of a decode16 plan for page size `page` (the last-arriver form exists for pages >= 16; smaller pages run
/// the fused form).
pub fn decode16_kernel(page: u32, plan: Decode16Plan) -> (String, u32) {
    let plan = if page < 16 { Decode16Plan::Fused } else { plan };
    match plan {
        Decode16Plan::Fused => (format!("pa_decode16_bf16_hd128_p{page}_f"), 1),
        Decode16Plan::LastC(s) => (format!("pa_decode16_bf16_hd128_p{page}_c"), s),
    }
}

/// Causal attention backward: the causal kernels (longest-first launch order built in: `fmha_bwd_dq_hd128_bf16_causal` / `fmha_bwd_dkdv_hd128_bf16_causal` grid
/// (H, B, tiles), heaviest tile first: the hardware dispatcher takes workgroups in linear order, so long tiles start first). Same run vs the
/// key-tile-fastest grid / the paired dkdv (archive/bwd-dkdv-pair): B1 H16 4096² 598 / 450 -> 407 us, B4 H8 2048² 323 / 249 -> 228,
/// B1 H8 HK2 2048² 174 / 186 -> 167, B2 H8 HK2 1000x1500 118 / 151 -> 113.
pub fn bwd_causal_kernels() -> (&'static str, &'static str) {
    ("fmha_bwd_dq_hd128_bf16_causal", "fmha_bwd_dkdv_hd128_bf16_causal")
}

/// Fill of the CUs (percent of `cus`, counted in 256 x 256 CTAs) from which the 256-row GEMM form beats the 128-row one. Measured same-run on MI355X (node 53, GPU 1, 7 interleaved repetitions,
/// K = 4096, N = 4096 and 7168, M = 1024 .. 4096 in steps of 256; bf16, int8 and fp8 per-token x per-channel agree): the 128 x 128 form wins up to 128 CTAs of 256 (fill 50 %: bf16 2048x4096 74.3 vs 80.1 us,
/// int8 41.8 vs 47.0, fp8 40.3 vs 46.1; 112 CTAs 1024x7168: 37.7 vs 44.7 int8), the 256 x 256 form wins from 140 CTAs (fill 55 %: 1280x7168 int8 48.6 vs 56.7, bf16 83.6 vs 96.7, fp8 47.4 vs 55.6;
/// 144 CTAs 2304x4096: 48.9 vs 57.5) with the gap growing to 17 - 28 % at fill 88 - 98 % (2048x7168 int8 59.5 vs 74.4, fp8 58.3 vs 73.2, bf16 101.4 vs 126.5). The threshold is the midpoint of the
/// last loss (128 CTAs) and the first win (140), 53 %. (The previous rule, 100 %, picked the 128 form for the whole 50 - 100 % band, 15 - 28 % slower.)
const F256_PERCENT: u64 = 53;

/// The 64 x 128 GEMM form's time model (us), fitted to the measured same-run times of `tools/compose_search.py check-gemm` (node 53, GPU 1, 7 interleaved repetitions, M = 64 .. 512, N = 4096 / 7168, K = 4096 /
/// 8192; docs/compose-gemm-evidence.md): `a + b K / (4096 S) + [S > 1] (r0 + r1 S M N / 8e6)` for a one-round grid (`c S <= CUs`, c = ceil(M/64) ceil(N/128)): a fixed launch + epilogue part, the K loop split S
/// ways, and the split-K reduce (fixed + the S fp32 partials of the M x N output). Constants per operand width: (a, b, r0, r1). Mean error 5.6 % (8-bit, 72 points), 7.6 % (bf16, 27 points); the rule built on
/// it has 0.0 - 0.1 % mean regret against the measured best split (the previous rule: 9.5 % mean, 45 % worst, on the 8-bit kinds).
const SPLIT64_8BIT: (f64, f64, f64, f64) = (2.55, 10.13, 0.86, 22.25);
const SPLIT64_BF16: (f64, f64, f64, f64) = (0.63, 22.18, 0.96, 27.90);

fn split64_us(p: (f64, f64, f64, f64), m: u64, n: u64, k: u64, s: u64) -> f64 {
    let split = if s > 1 { p.2 + p.3 * (s * m * n) as f64 / 8.0e6 } else { 0.0 };
    p.0 + p.1 * k as f64 / (4096.0 * s as f64) + split
}

/// The split-K factor of the 64-row form: the S in {1, 2, 4} the model predicts fastest among those the kernels allow (the reduce covers N in 1024-column blocks, the 3-slot ring needs whole 256-element K
/// pairs per part) and that keep the grid within one round (`c S <= CUs`).
fn split64(p: (f64, f64, f64, f64), m: u64, n: u64, k: u64, cus: u64) -> u32 {
    let c = m.div_ceil(64) * n.div_ceil(128);
    let mut best = (split64_us(p, m, n, k, 1), 1u64);
    for s in [2u64, 4] {
        if c * s <= cus && n % 1024 == 0 && k % (256 * s) == 0 {
            let t = split64_us(p, m, n, k, s);
            if t < best.0 {
                best = (t, s);
            }
        }
    }
    // S = 8 where S = 4 leaves half of the CUs idle (N = 4096: c = 32, 8 x 32 = 256 CTAs): measured on node 53 / GPU 1, 7 interleaved repetitions, `compose_search check-gemm` M = 1 .. 64 x N 4096 x K 4096 / 8192 / 14336
    // (docs/compose-gemm-sweeps.md): M <= 16 S8 beats S4 by 5 - 20 % at K <= 8192 (bf16 M1 K4096 7.4 vs 9.2 us, int8 M1 K8192 7.4 vs 9.1, fp8 M16 K8192 7.9 vs 9.1) and ties at K = 14336 (+-1.5 %); M = 32 only ties (K 4096
    // int8 / fp8 S8 +1.3 % worse) and M = 64 loses (S8 5 - 9 % worse), so S4 stays there; the fitted model alone cannot see the fill effect.
    if best.1 == 4 && m <= 16 && c * 8 <= cus && n % 1024 == 0 && k % 2048 == 0 {
        return 8;
    }
    best.1 as u32
}

/// int8 A8W8 GEMM (`gemm_a8w8_i8[_128|_64[_splitk]]`, per-token x per-channel scales; the same rule picks `gemm_a8w8_rowcol_fp8[_128|_64[_splitk]]`): (CTA rows, split-K S). 256 x 256 from 53 % fill of the CUs
/// (`F256_PERCENT`), else 128 x 128 when that grid fills the CUs, else the 64 x 128 3-slot form with the split `split64` predicts. Same run (node 53) vs the measured best of the whole candidate set: the
/// evidence shapes of docs/compose-gemm-evidence.md and the sweeps behind `F256_PERCENT` / `SPLIT64_*`. (Earlier evidence vs the best aiter i8gemm tile: 4096^3 68.9 vs 112 us, 2048x4096^2 42.3 vs 55.2,
/// 1024x4096x8192 51.9 vs 50.8, 512x7168x4096 23.6 vs 25.5, 256x7168x4096 20.0 vs 14.0, 128x4096x8192 (S 4) 16.2 vs 17.7, 64x8192^2 (S 4) 19.1 vs 18.9.)
pub fn i8_gemm_config(m: u64, n: u64, k: u64, cus: u64) -> (u32, u32) {
    // the 256 x 256 form from a little over half the CUs filled (`F256_PERCENT`): measured crossover (docs/compose-gemm-evidence.md, fill sweep)
    if (m / 256) * (n / 256) * 100 >= F256_PERCENT * cus {
        return (256, 1);
    }
    if m.div_ceil(128) * n.div_ceil(128) >= cus {
        return (128, 1);
    }
    (64, split64(SPLIT64_8BIT, m, n, k, cus))
}

/// bf16 A16W16 GEMM (`gemm_a16w16_bf16[_128|_128_splitk|_64[_splitk]]`): the int8 rule with the bf16 split constants, plus the 128 x 128 split form: for K >= 6144 and a 128-row grid of 0.625 .. 1 x CUs (bf16
/// is compute-bound enough that two half-K CTAs per CU beat one: K = 8192, M 640 .. 1024 x N 4096, M 384 .. 512 x N 7168: `_128_splitk` S2 5 - 12 % faster than `_64` S1 / `_128` S1, 512x7168x7168 8.9 %;
/// at K = 4096 and above the grid it loses).
pub fn bf16_gemm_config(m: u64, n: u64, k: u64, cus: u64) -> (u32, u32) {
    if (m / 256) * (n / 256) * 100 >= F256_PERCENT * cus {
        return (256, 1);
    }
    let g128 = m.div_ceil(128) * n.div_ceil(128);
    if k >= 6144 && n % 1024 == 0 && k % 256 == 0 && 8 * g128 >= 5 * cus && g128 <= cus {
        return (128, 2);
    }
    if g128 >= cus {
        return (128, 1);
    }
    (64, split64(SPLIT64_BF16, m, n, k, cus))
}

/// MLA decode: (head-parallel form `mla_decode_bf16_hp_p<P>`?, S) of a launch with batch `b`, `h` heads and at most `ctx` context tokens (`u64::MAX` = unknown). The head-parallel form (64 heads per CTA share the KV
/// tiles) for H >= 64 (H128 B64 ctx 4096 P16 232 us vs 348 us for the per-wave form) and, measured on node 53 / GPU 1 (docs/compose-mla-moe-evidence.md), also for small batches at short context: B <= 16 and ctx <= 4096
/// (H16: ctx 1024 18.0 vs 35.8 us, ctx 2048 23.9 vs 36.1, ctx 4096 29.4 vs 40.4 at B1; B4 / B16 +22 - 107 %; H32 ctx 2048 B1 +50 %, B16 +25 %; H48 B1 +44 %); the per-wave form (`mla_decode_bf16_p<P>`, 16 heads per CTA,
/// 4 key parts per CTA) otherwise (H16 ctx >= 6144 at B1, every ctx at B64). S = the smallest power of two giving >= 256 CTAs (<= 16): H16 B64 -> 4 (76 us), H128 B64 -> 2.
pub fn mla_config(b: u64, h: u64, ctx: u64) -> (bool, u32) {
    let hp = h >= 64 || (b <= 16 && ctx <= 4096);
    let ctas = b * if hp { h.div_ceil(64) } else { h.div_ceil(16) };
    let mut s = 1u64;
    while s < 16 && ctas * s < 256 {
        s *= 2;
    }
    (hp, s as u32)
}

/// hd128 bf16 prefill: the kernel and key splits. Non-causal: `fmha_fwd_hd128_bf16_v3` (S = 1) or its `_ks` form (small grids,
/// `bf16_prefill_key_splits`). Causal: the paired v3 (`_causal`, `_causal_ks` for small grids), except tiny grids (< 64 paired CTAs), where
/// the 128-row v1 `fmha_fwd_hd128_bf16_causal` measured faster (B2 H8 HK2 1000 x 1300: 123 vs 94 TF/s). (hd64 / hd256, sliding window and
/// varlen stay on the v1 family.)
/// Causal re-measured 2026-11-09 with the assembling `_causal_ks` (backend SGPR fix) and the q16 combine (us; aiter / paired / v1 / ks2 / ks4):
/// B1 H8 4096^2 91 / 119 / 98 / 86 / 74; 2048^2 52 / 69 / 49 / 51 / 42; B2 H8 HK2 1000 x 1300 47 / 63 / 36 / 50 / 43; B1 H16 4096^2
/// 108 / 133 / 150 / 122 / 157; B4 H16 1024^2 42 / 51 / 51 / 59 / 86; B1 H8 8192^2 192 / 239 / 280 / 203 / 226. So: v1 only for tiny grids
/// with short keys (< 16 key tiles), and no causal key split below 16 key tiles.
/// With the 128-row / 64-key `_causal_q128` form (same run, us; aiter / paired / v1 / q128): B4 H16 1024^2 41.6 / 51.5 / 51.5 / 40.6;
/// B1 H32 HK8 2048^2 62.7 / 77.2 / 81.7 / 65.1; B2 H16 1024^2 36.7 / 47.6 / 32.9 / 34.6; B4 H8 1024^2 36.9 / 47.2 / 32.9 / 34.4; B8 H16 1024^2
/// 54.5 / 65.5 / 94.2 / 69.3; B1 H8 2048^2 (P = 32) keeps `_causal_ks` S4 (42.3 vs 51.4). With P = the paired 256-row grid: short keys and
/// P < CUs / 2 -> v1; CUs / 2 <= P < CUs with <= 16 key tiles -> `_causal_q128`.
pub fn bf16_prefill_kernel(b: u64, h: u64, sq: u64, sk: u64, causal: bool, cus: u64) -> (&'static str, u32) {
    let short = sk.div_ceil(128) < 16;
    let paired = b * h * sq.div_ceil(256).div_ceil(2);
    if causal && paired < cus / 2 && short {
        return ("fmha_fwd_hd128_bf16_causal", 1);
    }
    if causal && paired >= cus / 2 && paired < cus && sk.div_ceil(128) <= 16 {
        return ("fmha_fwd_hd128_bf16_v3_causal_q128", 1);
    }
    let s = if causal && short { 1 } else { bf16_prefill_key_splits(b, h, sq, sk, causal, cus) };
    let name = match (causal, s > 1) {
        (false, false) => "fmha_fwd_hd128_bf16_v3",
        (false, true) => "fmha_fwd_hd128_bf16_v3_ks",
        (true, false) => "fmha_fwd_hd128_bf16_v3_causal",
        (true, true) => "fmha_fwd_hd128_bf16_v3_causal_ks",
    };
    (name, s)
}

/// Per-tensor FP8 prefill (`fmha_fwd_hd128_fp8pt_v3*`): the kernel and key splits. U = B H ceil(SQ / 256) (the unpaired q-tile grid). U < CUs:
/// the KV-split form (`_kvsplit` / `_causal_kvsplit` + `fmha_fwd_hd128_mxfp8_v3_kvsplit_combine8h`) with the smallest S giving U S >= CUs
/// (S <= 4, at least two key tiles per part); else the plain kernel (causal: the paired one). Measured (same run, us, plain / split / aiter):
/// B1 H8 2048^2 37.7 / 24.6 (S4) / 33.2, 4096^2 71.0 / 55.3 (S2) / 63.7; causal 2048^2 50.4 / 25.2 (S4) / 41.1, 4096^2 90.8 / 51.3 (S2) / 68.2;
/// at U = 256 the split loses (B4 H16 1024^2 32.3 vs 47.6 S2; causal 40.0 vs 45.6).
pub fn fp8pt_prefill_kernel(b: u64, h: u64, sq: u64, sk: u64, causal: bool, cus: u64) -> (&'static str, u32) {
    let u = b * h * sq.div_ceil(256);
    let mut s = 1u64;
    while s < 4 && u * s < cus && sk.div_ceil(128) >= 4 * s {
        s *= 2;
    }
    let name = match (causal, s > 1) {
        (false, false) => "fmha_fwd_hd128_fp8pt_v3",
        (false, true) => "fmha_fwd_hd128_fp8pt_v3_kvsplit",
        (true, false) => "fmha_fwd_hd128_fp8pt_v3_causal",
        (true, true) => "fmha_fwd_hd128_fp8pt_v3_causal_kvsplit",
    };
    (name, s as u32)
}

/// MLA prefill hd192/hd128 (`fmha_fwd_hd192_hd128_bf16_v3*`): the kernel and key splits. Non-causal: the plain kernel, `_ks` on small grids. Causal, by the
/// unpaired grid U = B H ceil(SQ / 256): U >= 2 CUs -> the paired `_causal` (B4 H16 2048^2 173 -> 130 us, B1 H128 1024^2 101 -> 80, vendor
/// 128 / 82); smaller grids the unpaired `_causal_np` (B1 H16 4096^2 143 vs 156 paired, B2 H16 2048^2 89 vs 93), and when U <= CUs / 2 with
/// >= 32 key tiles the unpaired key split `_causal_ks` S = 2 (B1 H8 4096^2 104 vs 121; B1 H16 2048^2, 16 key tiles, stays unsplit: 72 vs 75).
pub fn hd192_prefill_kernel(b: u64, h: u64, sq: u64, sk: u64, causal: bool, cus: u64) -> (&'static str, u32) {
    let u = b * h * sq.div_ceil(256);
    if !causal {
        // small grids: the key split `_ks` (fp32 partials + attn_combine_bf16_hd128_q16). U <= CUs / 4: S = 4 (B1 H8 2048^2 66.6 -> 41.3 us);
        // U < CUs with >= 32 key tiles: S = 2 (B1 H8 4096^2 134 -> 112); U = CUs / 2 with 16 key tiles loses (B1 H16 2048^2 76.1 vs 78.9 with
        // the one-row combine)
        if 4 * u <= cus && sk.div_ceil(128) >= 16 {
            return ("fmha_fwd_hd192_hd128_bf16_v3_ks", 4);
        }
        // re-measured after sched's region policy (2026-11-09, us; plain / ks2): B1 H16 2048^2 73.0 / 62.5, B1 H8 4096^2 132.6 / 111.7: S = 2 from
        // 16 key tiles
        if u < cus && sk.div_ceil(128) >= 16 {
            return ("fmha_fwd_hd192_hd128_bf16_v3_ks", 2);
        }
        return ("fmha_fwd_hd192_hd128_bf16_v3", 1);
    }
    if u >= 2 * cus {
        ("fmha_fwd_hd192_hd128_bf16_v3_causal", 1)
    } else if 2 * u <= cus && (sk.div_ceil(128) >= 32 || (4 * u <= cus && sk.div_ceil(128) >= 16)) {
        // (re-measured after the region policy, us; np / np_q128 / ks2: B1 H8 2048^2 64.2 / 52.6 / 48.0, B1 H8 4096^2 120.3 / 103.1 / 89.2;
        // B1 H16 2048^2 70.3 / 57.1 / 59.3)
        ("fmha_fwd_hd192_hd128_bf16_v3_causal_ks", 2)
    } else if 2 * u <= cus {
        // the 128-row / 64-key form (same run, us; np / np_q128 / aiter): B1 H16 2048^2 72.1 / 65.7 / 50.3, B2 H16 1024^2 42.3 / 39.8 / 31.4,
        // B1 H8 2048^2 65.9 / 60.9 / 45.4; at U = 256 it loses (B4 H16 1024^2 52.6 / 60.7)
        ("fmha_fwd_hd192_hd128_bf16_v3_causal_np_q128", 1)
    } else {
        ("fmha_fwd_hd192_hd128_bf16_v3_causal_np", 1)
    }
}

/// Implicit-GEMM conv (`conv{2,3}d_bf16_*`): the CTA (256, or `_c128`). M = NB OD OH OW rows, N = output channels. 256 x 256 CTAs when there
/// are at least CUs / 2 of them, else 128 x 128 (measured: 3x3 C64 K256 NB8 56^2 408 vs 255 TF/s, NB16 28^2 C256 K512 567 vs 345, small
/// shapes ~2x; NB32 28^2 C256 K512 639 (256) vs 595). (Split-K over the conv K was tried: slower and needs K tiles % S == 0 - not kept.)
pub fn conv_cta(m: u64, n: u64, cus: u64) -> u32 {
    if m.div_ceil(256) * n.div_ceil(256) >= cus / 2 {
        256
    } else {
        128
    }
}

/// Total token-expert pairs (M topk) up to which the 64-row MoE tile also wins for 128 - 192 rows per expert (the grid of the 128-row form is too small to fill the CUs).
const MOE_BM64_PAIRS: u64 = 5120;
/// Stage-1 output columns (2 I) per CTA tile in the grid model of the band below.
const MOE_N_TILE: u64 = 128;

/// Fused MoE: block_m (the routing sort's block and the `gemm_moe{1,2}_*` tile rows) of a launch with `m` tokens, `topk`, `experts`, intermediate size `inter` (`u64::MAX` = unknown) on `cus` CUs. 64 (`_bm64` kernels)
/// when the average rows per expert r = M topk / E is below 128; also in the measured band r <= 192 with at most `MOE_BM64_PAIRS` pairs in total when the 128-row stage-1 grid (E x blocks per expert x ceil(2 I / 128) CTAs)
/// fits the CUs (r = 128 exactly: half of them); else 128. Measured, stage 1 + stage 2 (docs/compose-mla-moe-evidence.md; node 53 / GPU 1, D 4096 I 256, 7 interleaved repetitions): the 64-row form wins E16 r 128 / 160 / 192
/// (fp8 +24 / +25 / +12 %, bf16 +27 / +25 / +20 %) and E32 r 128 / 160 (fp8 +7.6 / +12.9 %, bf16 +8.6 / +4.4 %); the 128-row form wins E32 r >= 224 and every E64 shape from r = 128 (E64 r 128: fp8 +12.7 %, bf16 +19 %),
/// E32 r 192 is mixed (pairs 6144 > the cap). With a larger intermediate size the grid of the 128-row form already fills the CUs and the 64-row form LOSES (perf_ab, E16 D2048 I768 M320 / M384 bf16: stage 1 -35 %, hence
/// the grid test). Earlier (stage 1, silu): M256 D7168 I2048 E64 topk8 fp8 420 -> 381 us (aiter 392), fp8bs 530 -> 417 us (aiter 418); M128 D4096 I1024 E8 topk2 fp8bs 53.8 -> 38.0 us; stage 2 + sum also faster
/// (251 -> 222, 262 -> 213 us).
pub fn moe_block_m(m: u64, topk: u64, experts: u64, inter: u64, cus: u64) -> u32 {
    let pairs = m * topk;
    if pairs < 128 * experts {
        return 64;
    }
    if pairs <= 192 * experts && pairs <= MOE_BM64_PAIRS && inter != u64::MAX {
        let blocks = pairs.div_ceil(128 * experts).max(1);
        let grid128 = experts * blocks * (2 * inter).div_ceil(MOE_N_TILE);
        let fits = if pairs > 128 * experts { grid128 <= cus } else { 2 * grid128 <= cus };
        if fits {
            return 64;
        }
    }
    128
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moe_rule() {
        assert_eq!(moe_block_m(256, 8, 64, 2048, 256), 64);
        assert_eq!(moe_block_m(128, 2, 8, 1024, 256), 64);
        assert_eq!(moe_block_m(1024, 2, 8, 1024, 256), 128);
        assert_eq!(moe_block_m(4096, 8, 32, 2048, 256), 128);
        // the measured band at I = 256 (stage-1 grid <= CUs): E16 r 128 / 192 and E32 r 128 / 160 take 64 rows; E32 r 192 / 224 and E64 r 128 / 192 keep 128
        assert_eq!(moe_block_m(256, 8, 16, 256, 256), 64);
        assert_eq!(moe_block_m(384, 8, 16, 256, 256), 64);
        assert_eq!(moe_block_m(512, 8, 32, 256, 256), 64);
        assert_eq!(moe_block_m(640, 8, 32, 256, 256), 64);
        assert_eq!(moe_block_m(768, 8, 32, 256, 256), 128);
        assert_eq!(moe_block_m(896, 8, 32, 256, 256), 128);
        assert_eq!(moe_block_m(1024, 8, 64, 256, 256), 128);
        assert_eq!(moe_block_m(1536, 8, 64, 256, 256), 128);
        // a larger intermediate size fills the CUs at 128 rows: the band does not apply (E16 I768 measured -35 % with 64 rows); an unknown size keeps the old rule
        assert_eq!(moe_block_m(320, 8, 16, 768, 256), 128);
        assert_eq!(moe_block_m(384, 8, 16, 768, 256), 128);
        assert_eq!(moe_block_m(384, 8, 16, u64::MAX, 256), 128);
    }

    #[test]
    fn conv_rule() {
        assert_eq!(conv_cta(25088, 512, 256), 256);
        assert_eq!(conv_cta(12544, 512, 256), 128);
        assert_eq!(conv_cta(25088, 256, 256), 128);
        assert_eq!(conv_cta(6272, 256, 256), 128);
    }

    #[test]
    fn prefill_kernel_rule() {
        assert_eq!(bf16_prefill_kernel(2, 8, 1000, 1300, true, 256).0, "fmha_fwd_hd128_bf16_causal");
        assert_eq!(bf16_prefill_kernel(2, 32, 8192, 8192, true, 256), ("fmha_fwd_hd128_bf16_v3_causal", 1));
        assert_eq!(bf16_prefill_kernel(2, 32, 8192, 8192, false, 256), ("fmha_fwd_hd128_bf16_v3", 1));
        assert_eq!(bf16_prefill_kernel(1, 8, 4096, 4096, false, 256), ("fmha_fwd_hd128_bf16_v3_ks", 2));
        assert_eq!(bf16_prefill_kernel(1, 8, 2048, 2048, true, 256), ("fmha_fwd_hd128_bf16_v3_causal_ks", 4));
        assert_eq!(bf16_prefill_kernel(1, 8, 4096, 4096, true, 256), ("fmha_fwd_hd128_bf16_v3_causal_ks", 4));
        assert_eq!(bf16_prefill_kernel(1, 16, 4096, 4096, true, 256), ("fmha_fwd_hd128_bf16_v3_causal_ks", 2));
        assert_eq!(bf16_prefill_kernel(4, 16, 1024, 1024, true, 256), ("fmha_fwd_hd128_bf16_v3_causal_q128", 1));
        assert_eq!(bf16_prefill_kernel(1, 32, 2048, 2048, true, 256), ("fmha_fwd_hd128_bf16_v3_causal_q128", 1));
        assert_eq!(bf16_prefill_kernel(2, 16, 1024, 1024, true, 256), ("fmha_fwd_hd128_bf16_causal", 1));
        assert_eq!(bf16_prefill_kernel(8, 16, 1024, 1024, true, 256), ("fmha_fwd_hd128_bf16_v3_causal", 1));
    }

    #[test]
    fn blockscale_rule() {
        assert_eq!(blockscale_config(1024, 4096, 8192, 256), (64, 1));
        assert_eq!(blockscale_config(256, 7168, 4096, 256), (64, 1));
        assert_eq!(blockscale_config(512, 7168, 7168, 256), (64, 1));
        assert_eq!(blockscale_config(128, 4096, 8192, 256), (64, 4));
        assert_eq!(blockscale_config(4096, 4096, 4096, 256), (128, 1));
    }

    #[test]
    fn fp8pt_rule() {
        assert_eq!(fp8pt_prefill_kernel(1, 8, 2048, 2048, false, 256), ("fmha_fwd_hd128_fp8pt_v3_kvsplit", 4));
        assert_eq!(fp8pt_prefill_kernel(1, 8, 4096, 4096, true, 256), ("fmha_fwd_hd128_fp8pt_v3_causal_kvsplit", 2));
        assert_eq!(fp8pt_prefill_kernel(4, 16, 1024, 1024, true, 256), ("fmha_fwd_hd128_fp8pt_v3_causal", 1));
        assert_eq!(fp8pt_prefill_kernel(2, 32, 8192, 8192, false, 256), ("fmha_fwd_hd128_fp8pt_v3", 1));
    }

    #[test]
    fn hd192_rule() {
        assert_eq!(hd192_prefill_kernel(4, 16, 2048, 2048, true, 256), ("fmha_fwd_hd192_hd128_bf16_v3_causal", 1));
        assert_eq!(hd192_prefill_kernel(1, 128, 1024, 1024, true, 256), ("fmha_fwd_hd192_hd128_bf16_v3_causal", 1));
        assert_eq!(hd192_prefill_kernel(1, 16, 4096, 4096, true, 256), ("fmha_fwd_hd192_hd128_bf16_v3_causal_np", 1));
        assert_eq!(hd192_prefill_kernel(1, 16, 2048, 2048, true, 256), ("fmha_fwd_hd192_hd128_bf16_v3_causal_np_q128", 1));
        assert_eq!(hd192_prefill_kernel(1, 8, 4096, 4096, true, 256), ("fmha_fwd_hd192_hd128_bf16_v3_causal_ks", 2));
        assert_eq!(hd192_prefill_kernel(1, 8, 4096, 4096, false, 256), ("fmha_fwd_hd192_hd128_bf16_v3_ks", 2));
        assert_eq!(hd192_prefill_kernel(1, 8, 2048, 2048, false, 256), ("fmha_fwd_hd192_hd128_bf16_v3_ks", 4));
        assert_eq!(hd192_prefill_kernel(1, 16, 2048, 2048, false, 256), ("fmha_fwd_hd192_hd128_bf16_v3_ks", 2));
        assert_eq!(hd192_prefill_kernel(1, 8, 2048, 2048, true, 256), ("fmha_fwd_hd192_hd128_bf16_v3_causal_ks", 2));
        assert_eq!(hd192_prefill_kernel(2, 16, 1024, 1024, true, 256), ("fmha_fwd_hd192_hd128_bf16_v3_causal_np_q128", 1));
        assert_eq!(hd192_prefill_kernel(1, 16, 4096, 4096, false, 256), ("fmha_fwd_hd192_hd128_bf16_v3", 1));
    }

    #[test]
    fn mla_rule() {
        assert_eq!(mla_config(64, 16, u64::MAX), (false, 4));
        assert_eq!(mla_config(64, 128, u64::MAX), (true, 2));
        // short context at a small batch: the head-parallel form (measured +22 - 107 % at H16 B1 / 4 / 16, ctx 1024 .. 4096); a long context or a large batch keeps the per-wave form
        assert_eq!(mla_config(4, 16, 2048), (true, 16));
        assert_eq!(mla_config(16, 16, 4096), (true, 16));
        assert_eq!(mla_config(1, 16, 8192), (false, 16));
        assert_eq!(mla_config(64, 16, 2048), (false, 4));
        assert_eq!(mla_config(4, 16, u64::MAX), (false, 16));
    }

    #[test]
    fn prefill_rule() {
        assert_eq!(bf16_prefill_key_splits(1, 8, 4096, 4096, false, 256), 2);
        assert_eq!(bf16_prefill_key_splits(1, 8, 4096, 4096, true, 256), 4);
        assert_eq!(bf16_prefill_key_splits(2, 32, 8192, 8192, false, 256), 1);
    }

    #[test]
    fn decode16_plan_rule() {
        use super::{decode16_kernel, decode16_plan, Decode16Plan as P};
        // the measured winners of the grid in the doc comment
        assert_eq!(decode16_plan(4, 8, 1024), P::LastC(2));
        assert_eq!(decode16_plan(4, 8, 2048), P::LastC(4));
        assert_eq!(decode16_plan(4, 8, 8192), P::LastC(8));
        assert_eq!(decode16_plan(8, 8, 1024), P::LastC(2));
        assert_eq!(decode16_plan(8, 8, 2048), P::LastC(4));
        assert_eq!(decode16_plan(8, 8, 8192), P::LastC(4));
        for ctx in [1024, 2048, 8192] {
            assert_eq!(decode16_plan(16, 8, ctx), P::LastC(2));
            for b in [32, 64, 128] {
                assert_eq!(decode16_plan(b, 8, ctx), P::Fused);
            }
        }
        assert_eq!(decode16_plan(1, 8, 256), P::Fused);
        assert_eq!(decode16_kernel(16, P::LastC(4)), ("pa_decode16_bf16_hd128_p16_c".to_string(), 4));
    }

    #[test]
    fn i8_gemm_rule() {
        assert_eq!(super::i8_gemm_config(4096, 4096, 4096, 256), (256, 1));
        assert_eq!(super::i8_gemm_config(2048, 4096, 4096, 256), (128, 1));
        // the 256 x 256 form from 53 % fill: 128 CTAs (50 %) stay 128, 140 CTAs (55 %) and the 224 / 252 CTA grids take 256
        assert_eq!(super::i8_gemm_config(1792, 4096, 4096, 256), (128, 1));
        assert_eq!(super::i8_gemm_config(1280, 7168, 4096, 256), (256, 1));
        assert_eq!(super::i8_gemm_config(2304, 4096, 4096, 256), (256, 1));
        assert_eq!(super::i8_gemm_config(2048, 7168, 4096, 256), (256, 1));
        assert_eq!(super::i8_gemm_config(512, 7168, 4096, 256), (64, 1));
        assert_eq!(super::i8_gemm_config(128, 4096, 8192, 256), (64, 4));
        assert_eq!(super::i8_gemm_config(64, 8192, 8192, 256), (64, 4));
        // the split follows K and the grid: a short K or a larger grid runs unsplit, a mid grid at a long K splits in two (measured: 192x4096x4096 S1 12.0 us vs the old S4 16.2; 256x4096x8192 S2 18.3 vs S1 21.8)
        assert_eq!(super::i8_gemm_config(192, 4096, 4096, 256), (64, 1));
        assert_eq!(super::i8_gemm_config(128, 7168, 4096, 256), (64, 1));
        assert_eq!(super::i8_gemm_config(256, 4096, 8192, 256), (64, 2));
        assert_eq!(super::i8_gemm_config(64, 4096, 4096, 256), (64, 4));
        // S8 for M <= 16 where S4 fills half of the CUs (N = 4096), S4 at M = 32 / 64, and where the grid or the K step does not allow it
        assert_eq!(super::i8_gemm_config(1, 4096, 8192, 256), (64, 8));
        assert_eq!(super::i8_gemm_config(16, 4096, 4096, 256), (64, 8));
        assert_eq!(super::i8_gemm_config(32, 4096, 8192, 256), (64, 4));
        assert_eq!(super::i8_gemm_config(1, 8192, 8192, 256), (64, 4));
        assert_eq!(super::bf16_gemm_config(4, 4096, 4096, 256), (64, 8));
        // bf16: the 128 x 128 split form at a long K and a 0.625 .. 1 x CUs 128-row grid
        assert_eq!(super::bf16_gemm_config(768, 4096, 8192, 256), (128, 2));
        assert_eq!(super::bf16_gemm_config(512, 7168, 8192, 256), (128, 2));
        assert_eq!(super::bf16_gemm_config(768, 4096, 4096, 256), (64, 1));
        assert_eq!(super::bf16_gemm_config(1280, 4096, 8192, 256), (128, 1));
    }
}

/// The dense GEMM launch of `out[M, N] = A B^T` for an operand kind: (main kernel, split-K S, the split-K reduce kernel when S > 1). The tile and
/// split come from the family rules (`bf16_gemm_config`, `i8_gemm_config`, `blockscale_config`):
/// - 256 rows: the base kernel; 128: `_128`; 64: `_64`; 64 with S > 1: `_64_splitk` + `gemm_splitk_reduce_s<S>`;
/// - the per-token x per-channel kinds (`fp8_rowcol`, `i8_rowcol`) reduce with the scales (`_rc`).
///
/// The reduce covers N in 1024-column blocks, so N % 1024 != 0 runs unsplit. `mxfp4` has one kernel (`gemm_a4w4_mxfp4`). None: unknown kind.
pub fn gemm_kernels(kind: &str, m: u64, n: u64, k: u64, cus: u64) -> Option<(String, u32, Option<String>)> {
    let (base, rc) = match kind {
        "bf16" => ("gemm_a16w16_bf16", false),
        "fp8_rowcol" => ("gemm_a8w8_rowcol_fp8", true),
        "i8_rowcol" => ("gemm_a8w8_i8", true),
        "fp8_blockscale" => ("gemm_a8w8_blockscale", false),
        "mxfp4" => return Some(("gemm_a4w4_mxfp4".into(), 1, None)),
        _ => return None,
    };
    let (rows, s) = match kind {
        "fp8_blockscale" => blockscale_config(m, n, k, cus),
        "bf16" => bf16_gemm_config(m, n, k, cus),
        _ => i8_gemm_config(m, n, k, cus),
    };
    let s = if s > 1 && n % 1024 != 0 { 1 } else { s };
    let name = match (rows, s > 1) {
        (256, _) => base.to_string(),
        (128, _) if kind == "fp8_blockscale" => base.to_string(),
        (128, true) => format!("{base}_128_splitk"),
        (128, _) => format!("{base}_128"),
        (_, false) => format!("{base}_64"),
        (_, true) => format!("{base}_64_splitk"),
    };
    let reduce = (s > 1).then(|| format!("gemm_splitk_reduce_s{s}{}", if rc { "_rc" } else { "" }));
    Some((name, s, reduce))
}

/// MXFP8 prefill (`fmha_fwd_hd128_mxfp8_v3_*`, the vendor ABI): (path, splits), the shipped dispatch of `tests/gpu/ref_host.cpp` fitted on
/// realistic data. With N256 = ceil(S / 256) H B and T = ceil(SK / 128) key tiles:
/// - the KV split `_kvsplit` (+ `_kvsplit_combine8h`) with 4 parts when T >= 16 and N256 <= 64, or 2 parts when T >= 32 and N256 <= 128,
///   provided every part keeps a tile ((S - 1) ceil(T / S) < T);
/// - otherwise `_t128` when N256 <= 64, else `_vs128`.
///
/// Formerly a Python-side rule (`amdgpu_jit.select_path`).
pub fn mxfp8_prefill_path(b: u64, h: u64, s: u64, sk: u64) -> (&'static str, u32) {
    let blocks256 = s.div_ceil(256) * h * b;
    let ntiles = sk.div_ceil(128);
    let sp = if ntiles >= 16 && blocks256 <= 64 {
        4
    } else if ntiles >= 32 && blocks256 <= 128 {
        2
    } else {
        0
    };
    if sp > 0 {
        let tps = ntiles.div_ceil(sp);
        if (sp - 1) * tps < ntiles {
            return ("kvsplit", sp as u32);
        }
    }
    (if blocks256 <= 64 { "t128" } else { "vs128" }, 0)
}

/// The kernels of an MXFP8 prefill path (`mxfp8_prefill_path`).
pub fn mxfp8_prefill_kernels(path: &str) -> Vec<&'static str> {
    match path {
        "vs128" => vec!["fmha_fwd_hd128_mxfp8_v3_vs128"],
        "t128" => vec!["fmha_fwd_hd128_mxfp8_v3_t128"],
        _ => vec!["fmha_fwd_hd128_mxfp8_v3_kvsplit", "fmha_fwd_hd128_mxfp8_v3_kvsplit_combine8h"],
    }
}
