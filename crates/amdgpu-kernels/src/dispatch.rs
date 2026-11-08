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
    fn prefill_rule() {
        assert_eq!(bf16_prefill_key_splits(1, 8, 4096, 4096, false, 256), 2);
        assert_eq!(bf16_prefill_key_splits(1, 8, 4096, 4096, true, 256), 4);
        assert_eq!(bf16_prefill_key_splits(2, 32, 8192, 8192, false, 256), 1);
    }
}