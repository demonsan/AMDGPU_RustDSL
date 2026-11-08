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

/// Paged decode with 16x16 MFMAs (`pa_decode16_bf16_hd128_p<P>[_k64]`, GQA G <= 16): (keys per tile, splits per wave group S). n = B HK
/// work groups before the split: large n runs 32-key tiles (two CTAs per CU) with the smallest S giving n S >= 512; small grids (n < 64) run
/// 64-key tiles with S up to 8. Measured (B, HK, ctx): (16, 8, 4096) 32-key S4 4035 GB/s vs 64-key best 3739; (64, 8, 2048) 32-key S1 4182 vs
/// 3483; (4, 4, 8192) 64-key S8 1398 vs 32-key 1233.
pub fn decode16_config(b: u64, hk: u64) -> (u32, u32) {
    let n = b * hk;
    if n < 64 {
        let mut s = 1u64;
        while s < 8 && n * s * 4 < 1024 {
            s *= 2;
        }
        return (64, s as u32);
    }
    let mut s = 1u64;
    while s < 16 && n * s < 512 {
        s *= 2;
    }
    (32, s as u32)
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn decode16_rule() {
        assert_eq!(decode16_config(16, 8), (32, 4));
        assert_eq!(decode16_config(64, 8), (32, 1));
        assert_eq!(decode16_config(4, 4), (64, 8));
    }

    #[test]
    fn prefill_rule() {
        assert_eq!(bf16_prefill_key_splits(1, 8, 4096, 4096, false, 256), 2);
        assert_eq!(bf16_prefill_key_splits(1, 8, 4096, 4096, true, 256), 4);
        assert_eq!(bf16_prefill_key_splits(2, 32, 8192, 8192, false, 256), 1);
    }
}