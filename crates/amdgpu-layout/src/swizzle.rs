//! GF(2)-linear swizzles and swizzled layouts (`Swizzle o Layout`).
//!
//! A swizzle XORs a GF(2)-linear function of some SOURCE bits of an offset into disjoint DESTINATION bits: `y = x ^ (M * x[src] << dst_lo)`. CuTe's
//! `Swizzle<B, M, S>` (`x ^ ((x >> S) & mask_B) << M`, i.e. bits [M+S, M+S+B) into [M, M+B)) is the special case with the identity matrix. The FMHA K/V
//! tiles use a non-identity matrix (`FMHA128`, docs/status.md: found by brute force, zero LDS bank conflicts for the K reads, the transposing V reads and
//! the DMA writes): 16-byte chunk `c` (offset bits 4..6) of row `r` (offset bits 7..10 = r mod 16) is stored at chunk `c ^ s(r)` with
//! `s0 = r2 ^ r0, s1 = r1, s2 = r3 ^ r0`.

use crate::ix::{Dyns, Ix};
use crate::layout::Layout;
use crate::tuple::IntTuple;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Swizzle {
    /// first source bit and number of source bits
    pub src_lo: u32,
    pub src_n: u32,
    /// first destination bit and number of destination bits
    pub dst_lo: u32,
    pub dst_n: u32,
    /// `rows[i]` = mask of the source bits whose parity is XORed into destination bit i
    pub rows: Vec<u32>,
}

impl Swizzle {
    /// CuTe `Swizzle<B, M, S>`: bits `[M+S, M+S+B)` XORed into `[M, M+B)`.
    pub fn cute(b: u32, m: u32, s: u32) -> Swizzle {
        Swizzle { src_lo: m + s, src_n: b, dst_lo: m, dst_n: b, rows: (0..b).map(|i| 1u32 << i).collect() }
    }
    /// The identity (no swizzle).
    pub fn none() -> Swizzle {
        Swizzle { src_lo: 0, src_n: 0, dst_lo: 0, dst_n: 0, rows: vec![] }
    }
    /// The FMHA 128-byte-row swizzle (see the module doc).
    pub fn fmha128() -> Swizzle {
        Swizzle { src_lo: 7, src_n: 4, dst_lo: 4, dst_n: 3, rows: vec![0b0101, 0b0010, 0b1001] }
    }
    pub fn is_identity(&self) -> bool {
        self.rows.iter().all(|r| *r == 0)
    }
    pub fn src_mask(&self) -> u64 {
        (((1u64 << self.src_n) - 1) << self.src_lo) as u64
    }
    pub fn dst_mask(&self) -> u64 {
        ((1u64 << self.dst_n) - 1) << self.dst_lo
    }
    /// Disjoint source and destination bits: the map is an involution, hence a bijection on every aligned block.
    pub fn is_bijective(&self) -> bool {
        self.rows.len() == self.dst_n as usize && (self.src_mask() & self.dst_mask() == 0 || self.is_identity())
    }
    /// The XOR term `M * x[src] << dst_lo` of a concrete offset.
    pub fn term(&self, x: i64) -> i64 {
        let s = ((x as u64) >> self.src_lo) & ((1u64 << self.src_n) - 1);
        let mut t = 0i64;
        for (i, row) in self.rows.iter().enumerate() {
            if ((s as u32) & row).count_ones() % 2 == 1 {
                t |= 1 << (self.dst_lo as usize + i);
            }
        }
        t
    }
    pub fn apply(&self, x: i64) -> i64 {
        x ^ self.term(x)
    }
    /// The XOR term of a symbolic offset (bit extractions + parity XORs + shifts, like the manual `swz_s`).
    pub fn term_ix(&self, x: &Ix, d: &Dyns) -> Ix {
        let _ = d;
        let mut acc = Ix::C(0);
        for (i, row) in self.rows.iter().enumerate() {
            let mut par = Ix::C(0);
            for j in 0..self.src_n {
                if row & (1 << j) != 0 {
                    par = par.xor(&x.shr(self.src_lo + j).and_c(1));
                }
            }
            // the destination bits are disjoint: OR == ADD
            acc = acc.add(&par.shl(self.dst_lo + i as u32));
        }
        acc
    }
    pub fn apply_ix(&self, x: &Ix, d: &Dyns) -> Ix {
        if self.is_identity() {
            return x.clone();
        }
        x.xor(&self.term_ix(x, d))
    }
}

/// `swizzle o layout`: coordinates -> layout offset -> swizzled offset (CuTe `ComposedLayout`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ComposedLayout {
    pub swz: Swizzle,
    pub inner: Layout,
}

impl ComposedLayout {
    pub fn new(swz: Swizzle, inner: Layout) -> ComposedLayout {
        assert!(swz.is_bijective(), "ComposedLayout: the swizzle is not a bijection (source and destination bits overlap)");
        ComposedLayout { swz, inner }
    }
    pub fn plain(inner: Layout) -> ComposedLayout {
        ComposedLayout { swz: Swizzle::none(), inner }
    }
    pub fn crd2idx(&self, crd: &IntTuple, d: &Dyns) -> Ix {
        self.swz.apply_ix(&self.inner.crd2idx(crd, d), d)
    }
    pub fn at(&self, i: i64) -> i64 {
        self.swz.apply(self.inner.at(i))
    }
    /// The transposed view of a rank-2 layout (same storage).
    pub fn transpose(&self) -> ComposedLayout {
        ComposedLayout { swz: self.swz.clone(), inner: self.inner.transpose() }
    }
}
