//! Host-side dispatch rules for the shipped kernel families (measured same-run against the vendor kernels; docs/op-coverage-design.md
//! §7.4 / §7.5). Pure functions of the problem shape and the CU count: a launcher picks the kernel name and the grid with them.

/// fp8 blockscale GEMM (`gemm_a8w8_blockscale` 128 x 128 CTAs): the split-K factor. Mid-M shapes whose CTA count M N / 128^2 is at most
/// about two per CU run `gemm_a8w8_blockscale_splitk` with S = 2 (+ `gemm_splitk_reduce_s2`): measured 1024x4096x8192 864 -> 975,
/// 256x7168x4096 384 -> 516, 512x7168x7168 761 -> 840 TF/s; S = 4 never won; large shapes stay unsplit (4096^3: S = 2 loses).
pub fn blockscale_split_k(m: u64, n: u64, k: u64, cus: u64) -> u32 {
    let ctas = m.div_ceil(128) * n.div_ceil(128);
    // both halves of K must stay whole 128-wide scale groups
    if ctas <= 2 * cus && k % 256 == 0 {
        2
    } else {
        1
    }
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

/// Causal attention backward: the longest-first forms `fmha_bwd_dq_hd128_bf16_causal_lpt` / `fmha_bwd_dkdv_hd128_bf16_causal_lpt` (grid
/// (H, B, tiles), heaviest tile first: the hardware dispatcher takes workgroups in linear order, so long tiles start first). Same run vs the
/// key-tile-fastest grid / the paired dkdv (archive/bwd-dkdv-pair): B1 H16 4096² 598 / 450 -> 407 us, B4 H8 2048² 323 / 249 -> 228,
/// B1 H8 HK2 2048² 174 / 186 -> 167, B2 H8 HK2 1000x1500 118 / 151 -> 113.
pub fn bwd_causal_kernels() -> (&'static str, &'static str) {
    ("fmha_bwd_dq_hd128_bf16_causal_lpt", "fmha_bwd_dkdv_hd128_bf16_causal_lpt")
}

/// MLA decode: (head-parallel form `mla_decode_bf16_hp_p<P>`?, S). H >= 64 runs the head-parallel form (64 heads per CTA share the KV
/// tiles: H128 B64 ctx 4096 P16 232 us vs 348 us for the per-wave form); smaller H the per-wave form (`mla_decode_bf16_p<P>`, 16 heads per CTA,
/// 4 key parts per CTA). S = the smallest power of two giving >= 256 CTAs (<= 16): H16 B64 -> 4 (76 us), H128 B64 -> 2.
pub fn mla_config(b: u64, h: u64) -> (bool, u32) {
    let hp = h >= 64;
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
pub fn bf16_prefill_kernel(b: u64, h: u64, sq: u64, sk: u64, causal: bool, cus: u64) -> (&'static str, u32) {
    if causal && b * h * sq.div_ceil(256).div_ceil(2) < 64 {
        return ("fmha_fwd_hd128_bf16_causal", 1);
    }
    let s = bf16_prefill_key_splits(b, h, sq, sk, causal, cus);
    let name = match (causal, s > 1) {
        (false, false) => "fmha_fwd_hd128_bf16_v3",
        (false, true) => "fmha_fwd_hd128_bf16_v3_ks",
        (true, false) => "fmha_fwd_hd128_bf16_v3_causal",
        (true, true) => "fmha_fwd_hd128_bf16_v3_causal_ks",
    };
    (name, s)
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

/// Fused MoE: block_m (the routing sort's block and the `gemm_moe{1,2}_*` tile rows): 64 (`_bm64` kernels) when the average rows per
/// expert M topk / E is below 128, else 128. Measured (stage 1, silu): M256 D7168 I2048 E64 topk8 fp8 420 -> 381 us (aiter 392), fp8bs 530
/// -> 417 us (aiter 418); M128 D4096 I1024 E8 topk2 fp8bs 53.8 -> 38.0 us; stage 2 + sum also faster (251 -> 222, 262 -> 213 us).
pub fn moe_block_m(m: u64, topk: u64, experts: u64) -> u32 {
    if m * topk < 128 * experts {
        64
    } else {
        128
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moe_rule() {
        assert_eq!(moe_block_m(256, 8, 64), 64);
        assert_eq!(moe_block_m(128, 2, 8), 64);
        assert_eq!(moe_block_m(1024, 2, 8), 128);
        assert_eq!(moe_block_m(4096, 8, 32), 128);
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
    }

    #[test]
    fn blockscale_rule() {
        assert_eq!(blockscale_split_k(1024, 4096, 8192, 256), 2);
        assert_eq!(blockscale_split_k(256, 7168, 4096, 256), 2);
        assert_eq!(blockscale_split_k(512, 7168, 7168, 256), 2);
        assert_eq!(blockscale_split_k(4096, 4096, 4096, 256), 1);
    }

    #[test]
    fn mla_rule() {
        assert_eq!(mla_config(64, 16), (false, 4));
        assert_eq!(mla_config(64, 128), (true, 2));
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
        assert!(crate::kernel_by_name("pa_decode16_bf16_hd128_p16_c").is_some());
    }

    #[test]
    fn bwd_causal_lpt_registered() {
        let (q, k) = super::bwd_causal_kernels();
        assert!(crate::registry::kernel_by_name(q).is_some() && crate::registry::kernel_by_name(k).is_some());
    }
}