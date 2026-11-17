//! The gfx1250 (MI450 / CDNA5, wave32) matrix ATOMS of the tile layer. They are not re-exported at the crate root: the shipped kernels target
//! gfx950.
//!
//! Layouts from the public "CDNA5" Instruction Set Architecture, section 7.12 (Wave Matrix Multiply Accumulate): 7.12.2 "Matrix Element Storage in
//! VGPRs" for the dense FP8 ops, 7.12.6.1 "Matrix Data Layout" and 7.12.6.2 "Matrix Scale Values" for the block-scaled F4 op. Conventions as
//! [`crate::gfx950`]: an operand layout maps (lane, value) to the element index of the operand tile, A `m + M k`, B `n + N k`, C `m + M n`, with
//! `index = lane + 32 value` (wave32). Values are elements of the operand width in VGPR order (dword-major, low bits first).
//!
//! Provenance: measured on MI450 by the probes `probe_g1250_wmma_{fp8,fp8_k128,f4}` (`tests/gpu/g1250_wmma_probe_host.cpp` fills every operand through these layouts
//! and compares D with a CPU product: fp8 within 1e-4 of the dot-product magnitude, the scaled f4 form bit-exact).

use crate::gfx950::{MmaAtom, Provenance};
use amdgpu_layout::{make_layout, tup};

/// Lanes per wave.
pub const WAVE: i64 = 32;

impl MmaAtom {
    /// `v_wmma_f32_16x16x64_fp8_fp8` (dense, 8 VGPRs per A / B, 8 VGPRs C; ISA 7.12.2 "8-bit A-Matrix 16x64"):
    /// - A lane l = 16 h + m holds row m; VGPR v byte b holds k = b + 4 (v % 2) + 8 h + 16 ((v / 2) % 2) + 32 (v / 4);
    /// - C lane l = 16 h + n, VGPR v holds row m = v + 8 h, column n.
    pub fn wmma_f32_16x16x64_fp8() -> MmaAtom {
        let ab = make_layout(tup![tup![16, 2], tup![4, 2, 2, 2]], tup![tup![1, 8 * 16], tup![16, 4 * 16, 16 * 16, 32 * 16]]);
        MmaAtom {
            name: "Wmma16x16x64Fp8",
            mnemonic: "v_wmma_f32_16x16x64_fp8_fp8",
            m: 16,
            n: 16,
            k: 64,
            ab_bits: 8,
            a_tv: ab.clone(),
            b_tv: ab,
            c_tv: c_16x16(),
            a_dwords: 8,
            b_dwords: 8,
            c_dwords: 8,
            scaled: false,
            passes: 4,
            provenance: Provenance::Probe("probe_g1250_wmma_* on MI450 (tests/gpu/g1250_wmma_probe_host.cpp)"),
        }
    }
    /// `v_wmma_f32_16x16x128_fp8_fp8`: "A 16x128 8-bit matrix is just two 16x64 matrices in consecutive VGPRs" (16 VGPRs per A / B).
    pub fn wmma_f32_16x16x128_fp8() -> MmaAtom {
        let ab = make_layout(tup![tup![16, 2], tup![4, 2, 2, 2, 2]], tup![tup![1, 8 * 16], tup![16, 4 * 16, 16 * 16, 32 * 16, 64 * 16]]);
        MmaAtom {
            name: "Wmma16x16x128Fp8",
            mnemonic: "v_wmma_f32_16x16x128_fp8_fp8",
            m: 16,
            n: 16,
            k: 128,
            ab_bits: 8,
            a_tv: ab.clone(),
            b_tv: ab,
            c_tv: c_16x16(),
            a_dwords: 16,
            b_dwords: 16,
            c_dwords: 8,
            scaled: false,
            passes: 8,
            provenance: Provenance::Probe("probe_g1250_wmma_* on MI450 (tests/gpu/g1250_wmma_probe_host.cpp)"),
        }
    }
    /// `v_wmma_scale_f32_32x16x128_f4` (VOP3PX2, block size 32, E8M0 scales; ISA 7.12.6.1.3 and 7.12.6.2):
    /// - A (32 x 128, 16 VGPRs) lane l = 16 h + m: VGPR v nibble j holds row m + 16 (v / 8), k = j + 8 (v % 4) + 32 h + 64 ((v / 4) % 2);
    /// - B (16 x 128, 8 VGPRs): the same per 8 VGPRs with n for m;
    /// - C (32 x 16 f32, 16 VGPRs): lane l = 16 h + n, VGPR v holds row (v % 8) + 8 h + 16 (v / 8), column n.
    ///
    /// Scales ([`scale_32x16x128_f4_a`]): 1 VGPR for A, lane l byte j = scale of row l, k block j (rows 0..31 over all 32 lanes).
    pub fn wmma_scale_f32_32x16x128_f4() -> MmaAtom {
        MmaAtom {
            name: "WmmaScale32x16x128F4",
            mnemonic: "v_wmma_scale_f32_32x16x128_f4",
            m: 32,
            n: 16,
            k: 128,
            ab_bits: 4,
            a_tv: make_layout(tup![tup![16, 2], tup![8, 4, 2, 2]], tup![tup![1, 32 * 32], tup![32, 8 * 32, 64 * 32, 16]]),
            b_tv: make_layout(tup![tup![16, 2], tup![8, 4, 2]], tup![tup![1, 32 * 16], tup![16, 8 * 16, 64 * 16]]),
            c_tv: make_layout(tup![tup![16, 2], tup![8, 2]], tup![tup![32, 8], tup![1, 16]]),
            a_dwords: 16,
            b_dwords: 8,
            c_dwords: 16,
            scaled: true,
            passes: 8,
            provenance: Provenance::Probe("probe_g1250_wmma_* on MI450 (tests/gpu/g1250_wmma_probe_host.cpp)"),
        }
    }
}

/// The 32-bit C / D matrix of the 16 x 16 WMMAs (ISA 7.12.2): lane l = 16 h + n, VGPR v holds (m = v + 8 h, n).
fn c_16x16() -> amdgpu_layout::Layout {
    make_layout(tup![tup![16, 2], 8], tup![tup![16, 8], 1])
}

/// The A scale operand of `v_wmma_scale_f32_32x16x128_f4` (block 32, one VGPR, OPSEL unused): (lane, byte) -> `m + 32 kb` for row m = lane,
/// k block kb = byte (ISA 7.12.6.2 "F4 A-matrix (32x128) with 32 values per scale").
pub fn scale_32x16x128_f4_a() -> amdgpu_layout::Layout {
    make_layout(tup![32, 4], tup![1, 32])
}

/// The B scale operand of `v_wmma_scale_f32_32x16x128_f4` (16 x 128 B, block 32): bytes of lanes 0..15 (SCL_OPSEL_HI[0] = 0) or 16..31
/// (= 1): (lane % 16, byte) -> `n + 16 kb`.
pub fn scale_32x16x128_f4_b() -> amdgpu_layout::Layout {
    make_layout(tup![16, 4], tup![1, 16])
}

/// The gfx1250 matrix atoms (for table-wide checks).
pub fn mma_atoms() -> Vec<MmaAtom> {
    vec![MmaAtom::wmma_f32_16x16x64_fp8(), MmaAtom::wmma_f32_16x16x128_fp8(), MmaAtom::wmma_scale_f32_32x16x128_f4()]
}
