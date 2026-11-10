//! Operand element formats and scale schemes of the GEMM family (docs/l2-family-bodies.md §2.3): operand formats are L2 data, not shape choices. The kernel crate's `GemmConfig` picks them.

/// Operand element type of the GEMM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Elem {
    /// fp8 e4m3: one scaled `v_mfma_scale_f32_16x16x128_f8f6f4` per 16 x 16 block and 128-byte K tile
    Fp8,
    /// bf16: two `v_mfma_f32_16x16x32_bf16` per block and K tile (one per 16-byte half of the operand fragments; A and B halves share the K
    /// permutation, so the sum is unchanged)
    Bf16,
    /// int8 (`gemm_a8w8_i8`, per-token x per-channel fp32 scales): two `v_mfma_i32_16x16x64_i8` per block and K tile (one per 16-byte half of
    /// the fragments, the shared K permutation as `Bf16`); the i32 accumulators are converted to f32 before the scaled epilogue
    I8,
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
    pub fn per_tile(self) -> i64 {
        match self {
            Elem::Fp8 | Elem::Fp8Bs | Elem::I8 => 128,
            Elem::Bf16 => 64,
            Elem::Fp4 | Elem::Fp8W4 | Elem::Bf16W4 => 256,
        }
    }
    /// 128-byte A images per K tile (the mixed-precision forms stage several A images per B tile)
    pub fn a_images(self) -> usize {
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

