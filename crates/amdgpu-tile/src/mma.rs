//! TiledMma: an MMA atom with an optional A/B exchange, and the in-place accumulate.

use amdgpu_dsl::{FnBuilder, Opnd, ValueId, VarId};
use amdgpu_isa::MmaAtom;
use amdgpu_layout::Layout;

/// An MMA atom used by one wave. `swap_ab`: the hardware srcA is the B fragment (and srcB the A fragment, scales and their byte selects exchanged), so the
/// accumulator holds `C^T` (the aiter 4-wave GEMM's form: lane % 16 = M row).
#[derive(Clone, Debug)]
pub struct TiledMma {
    atom: MmaAtom,
    pub swap_ab: bool,
}

pub fn make_tiled_mma(atom: MmaAtom, swap_ab: bool) -> TiledMma {
    TiledMma { atom, swap_ab }
}

impl TiledMma {
    pub fn atom(&self) -> &MmaAtom {
        &self.atom
    }
    /// The register layout the A fragment must have (lane, value -> element `m + M k` of the A tile): with `swap_ab` the hardware reads the A
    /// fragment through its srcB port, so it is the atom's B layout.
    pub fn a_tv(&self) -> Layout {
        if self.swap_ab {
            self.atom.b_tv.clone()
        } else {
            self.atom.a_tv.clone()
        }
    }
    pub fn b_tv(&self) -> Layout {
        if self.swap_ab {
            self.atom.a_tv.clone()
        } else {
            self.atom.b_tv.clone()
        }
    }
    /// The accumulator layout (lane, value -> element `m + M n` of the C tile): transposed with `swap_ab`.
    pub fn c_tv(&self) -> Layout {
        if !self.swap_ab {
            return self.atom.c_tv.clone();
        }
        // hardware C' = C^T (square C tiles: the transposed element of row + R col is col + R row)
        assert_eq!(self.atom.m, self.atom.n, "c_tv with swap_ab: square C tiles only");
        transpose_tv(&self.atom.c_tv, self.atom.m)
    }
}

/// The TV layout whose element is the transpose (in an `r x r` column-major tile) of `tv`'s element, for the gfx950 C layouts (power-of-two strides).
fn transpose_tv(tv: &Layout, r: i64) -> Layout {
    // element e = row + r col -> col + r row: on the flat modes, a stride s < r (row part) becomes s * r, a stride s >= r (column part) becomes s / r
    let mut it = tv.stride.flatten().into_iter().map(|s| {
        let s = s.st();
        if s < r {
            amdgpu_layout::Ix::C(s * r)
        } else {
            assert_eq!(s % r, 0, "transpose_tv: stride {s} is not a multiple of {r}");
            amdgpu_layout::Ix::C(s / r)
        }
    });
    Layout { shape: tv.shape.clone(), stride: tv.stride.with_leaves(&mut it) }
}

/// `acc += A * B` with the atom of `mma` (operands as the TiledMma presents them; the exchange of `swap_ab` is applied here). `byte_a` / `byte_b` select
/// the E8M0 byte of the scale dwords (16x16x128 only; 0 otherwise).
#[allow(clippy::too_many_arguments)]
pub fn gemm(b: &mut FnBuilder, mma: &TiledMma, acc: VarId, a: ValueId, bf: ValueId, scale_a: impl Into<Opnd>, scale_b: impl Into<Opnd>, byte_a: u8, byte_b: u8) {
    let (sa, sb): (Opnd, Opnd) = (scale_a.into(), scale_b.into());
    let (x, y, sx, sy, bx, by) = if mma.swap_ab { (bf, a, sb, sa, byte_b, byte_a) } else { (a, bf, sa, sb, byte_a, byte_b) };
    match mma.atom.name {
        "MfmaScale32x32x64F8" => {
            assert!(bx == 0 && by == 0, "the 32x32x64 scaled MFMA reads scale byte 0");
            b.var_mfma_scale(acc, x, y, sx, sy);
        }
        "MfmaScale16x16x128F8" => b.var_mfma_scale16(acc, x, y, sx, sy, bx, by),
        other => panic!("gemm: no IR accumulate form for the atom {other}"),
    }
}
