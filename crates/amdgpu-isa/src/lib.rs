//! `amdgpu-isa`: the gfx950 instruction ATOMS of the tile layer (docs/dsl-layout-refactor-design.md §3).
//!
//! A copy atom is one memory instruction seen per lane: how many bytes each lane moves, where (global / LDS / registers), the immediate field, the completion
//! counter (`vmcnt` / `lgkmcnt`) and, for the LDS-direct DMA forms, the fixed destination pattern (`M0 + imm + 16 lane`). An MMA atom is one matrix
//! instruction with the thread-value layouts of its A, B, C operands (element index of the operand tile: A `m + M k`, B `n + N k`, C `m + M n`).
//! Every atom records where its layout comes from (`Provenance`): the ISA, a hardware probe in this repository, a shipped kernel that depends on it, or
//! not yet confirmed - unconfirmed layouts are usable but flagged so a kernel built on them is checked against hardware before it ships.

#![forbid(unsafe_code)]

use amdgpu_layout::{make_layout, tup, ImmRange, Layout};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Space {
    Global,
    Lds,
    Reg,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Counter {
    /// `s_waitcnt vmcnt`
    Vm,
    /// `s_waitcnt lgkmcnt`
    Lgkm,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Provenance {
    /// stated by the gfx950 ISA / LLVM
    Isa,
    /// measured by a hardware probe of this repository (`tests/gpu/...`)
    Probe(&'static str),
    /// relied upon by a shipped, gate-passing kernel of this repository
    Shipped(&'static str),
    /// derived from a vendor kernel's data layout, not yet confirmed on hardware here
    Derived(&'static str),
}

/// One memory instruction (per lane).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CopyAtom {
    pub name: &'static str,
    pub mnemonic: &'static str,
    pub src: Space,
    pub dst: Space,
    /// bytes moved per lane per instruction
    pub bytes: i64,
    /// instructions per atom operation (`ds_read_b128x2` = 2, `ds_read_b64_tr_b8 x4` = 4)
    pub insts: u32,
    /// required address alignment (bytes)
    pub align: i64,
    pub imm: ImmRange,
    pub counter: Counter,
    /// LDS-direct DMA: lane i writes `M0 + imm + lds_lane_stride * i` (the destination layout is fixed, the swizzle must sit on the source)
    pub lds_lane_stride: Option<i64>,
    /// transposing read (the register layout is the 8x8-byte transpose of the source quad-words)
    pub transpose: bool,
    pub provenance: Provenance,
}

const fn copy(name: &'static str, mnemonic: &'static str, src: Space, dst: Space, bytes: i64, insts: u32, imm: ImmRange, counter: Counter) -> CopyAtom {
    CopyAtom { name, mnemonic, src, dst, bytes, insts, align: bytes, imm, counter, lds_lane_stride: None, transpose: false, provenance: Provenance::Isa }
}

pub const GLOBAL_LOAD_X4: CopyAtom = copy("GlobalLoad128", "global_load_dwordx4", Space::Global, Space::Reg, 16, 1, ImmRange::GLOBAL, Counter::Vm);
pub const GLOBAL_LOAD_X2: CopyAtom = copy("GlobalLoad64", "global_load_dwordx2", Space::Global, Space::Reg, 8, 1, ImmRange::GLOBAL, Counter::Vm);
pub const GLOBAL_LOAD_X1: CopyAtom = copy("GlobalLoad32", "global_load_dword", Space::Global, Space::Reg, 4, 1, ImmRange::GLOBAL, Counter::Vm);
pub const GLOBAL_STORE_X4: CopyAtom = copy("GlobalStore128", "global_store_dwordx4", Space::Reg, Space::Global, 16, 1, ImmRange::GLOBAL, Counter::Vm);
pub const GLOBAL_STORE_X2: CopyAtom = copy("GlobalStore64", "global_store_dwordx2", Space::Reg, Space::Global, 8, 1, ImmRange::GLOBAL, Counter::Vm);
pub const BUFFER_LOAD_X4: CopyAtom = copy("BufferLoad128", "buffer_load_dwordx4", Space::Global, Space::Reg, 16, 1, ImmRange::BUFFER, Counter::Vm);
pub const BUFFER_STORE_X4: CopyAtom = copy("BufferStore128", "buffer_store_dwordx4", Space::Reg, Space::Global, 16, 1, ImmRange::BUFFER, Counter::Vm);
pub const GLOBAL_LDS_X4: CopyAtom = CopyAtom {
    lds_lane_stride: Some(16),
    provenance: Provenance::Shipped("fmha _vs128 K/V DMA (tile::copy::RowsCopy)"),
    ..copy("GlobalLds128", "global_load_lds_dwordx4", Space::Global, Space::Lds, 16, 1, ImmRange::GLOBAL, Counter::Vm)
};
pub const GLOBAL_LDS_X1: CopyAtom = CopyAtom {
    lds_lane_stride: Some(4),
    provenance: Provenance::Shipped("fmha _vs128 K-scale DMA (tile::copy::DwordRowsCopy)"),
    ..copy("GlobalLds32", "global_load_lds_dword", Space::Global, Space::Lds, 4, 1, ImmRange::GLOBAL, Counter::Vm)
};
pub const BUFFER_LDS_X4: CopyAtom = CopyAtom {
    lds_lane_stride: Some(16),
    provenance: Provenance::Probe("buffer_host (LDS-direct zero fill)"),
    ..copy("BufferLds128", "buffer_load_dwordx4 ... lds", Space::Global, Space::Lds, 16, 1, ImmRange::BUFFER, Counter::Vm)
};
pub const DS_READ_B128: CopyAtom = copy("DsRead128", "ds_read_b128", Space::Lds, Space::Reg, 16, 1, ImmRange::DS, Counter::Lgkm);
pub const DS_READ_B128X2: CopyAtom = copy("DsRead128x2", "ds_read_b128 x2", Space::Lds, Space::Reg, 16, 2, ImmRange::DS, Counter::Lgkm);
pub const DS_READ_B64: CopyAtom = copy("DsRead64", "ds_read_b64", Space::Lds, Space::Reg, 8, 1, ImmRange::DS, Counter::Lgkm);
pub const DS_READ_B32: CopyAtom = copy("DsRead32", "ds_read_b32", Space::Lds, Space::Reg, 4, 1, ImmRange::DS, Counter::Lgkm);
pub const DS_READ_TR_B8X4: CopyAtom = CopyAtom {
    transpose: true,
    provenance: Provenance::Probe("probe_lds_tr_b8 + fmha _vs128 V^T reads"),
    ..copy("DsReadTrB8x4", "ds_read_b64_tr_b8 x4", Space::Lds, Space::Reg, 8, 4, ImmRange::DS, Counter::Lgkm)
};
pub const DS_WRITE_B128: CopyAtom = copy("DsWrite128", "ds_write_b128", Space::Reg, Space::Lds, 16, 1, ImmRange::DS, Counter::Lgkm);
pub const DS_WRITE_B64: CopyAtom = copy("DsWrite64", "ds_write_b64", Space::Reg, Space::Lds, 8, 1, ImmRange::DS, Counter::Lgkm);
pub const DS_WRITE_B32: CopyAtom = copy("DsWrite32", "ds_write_b32", Space::Reg, Space::Lds, 4, 1, ImmRange::DS, Counter::Lgkm);

/// Every copy atom (for table-wide checks).
pub const COPY_ATOMS: &[CopyAtom] = &[
    GLOBAL_LOAD_X4,
    GLOBAL_LOAD_X2,
    GLOBAL_LOAD_X1,
    GLOBAL_STORE_X4,
    GLOBAL_STORE_X2,
    BUFFER_LOAD_X4,
    BUFFER_STORE_X4,
    GLOBAL_LDS_X4,
    GLOBAL_LDS_X1,
    BUFFER_LDS_X4,
    DS_READ_B128,
    DS_READ_B128X2,
    DS_READ_B64,
    DS_READ_B32,
    DS_READ_TR_B8X4,
    DS_WRITE_B128,
    DS_WRITE_B64,
    DS_WRITE_B32,
];

// compile-time sanity of the table: widths are dword multiples, alignment = width, the immediate fields are the ISA's (rustc rejects a bad entry: E0080)
const _: () = {
    let mut i = 0;
    while i < COPY_ATOMS.len() {
        let a = COPY_ATOMS[i];
        assert!(a.bytes == 4 || a.bytes == 8 || a.bytes == 16, "copy atom width");
        assert!(a.align == a.bytes, "copy atom alignment");
        assert!(a.insts >= 1 && a.insts <= 4, "copy atom instruction count");
        if let Some(s) = a.lds_lane_stride {
            assert!(s == a.bytes, "LDS-direct DMA lands bytes-per-lane apart");
        }
        i += 1;
    }
};

/// One matrix instruction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MmaAtom {
    pub name: &'static str,
    pub mnemonic: &'static str,
    pub m: i64,
    pub n: i64,
    pub k: i64,
    /// bits per A/B element
    pub ab_bits: u32,
    /// (lane, value) -> element of the operand tile: A `m + M k`, B `n + N k`, C `m + M n`
    pub a_tv: Layout,
    pub b_tv: Layout,
    pub c_tv: Layout,
    pub a_dwords: u32,
    pub b_dwords: u32,
    pub c_dwords: u32,
    /// MX block scales (E8M0) per operand
    pub scaled: bool,
    /// XDL passes (issue cycles = 4 x passes)
    pub passes: u32,
    pub provenance: Provenance,
}

impl MmaAtom {
    /// `v_mfma_scale_f32_32x32x64_f8f6f4` (E4M3): lane l holds row l % 32 and k = 16 (l / 32) + 0..15 and 32 + 16 (l / 32) + 0..15; C lane l value r is
    /// (row 8 (r / 4) + 4 (l / 32) + r % 4, column l % 32).
    pub fn mfma_scale_32x32x64_f8() -> MmaAtom {
        let ab = make_layout(tup![tup![32, 2], tup![16, 2]], tup![tup![1, 512], tup![32, 1024]]);
        MmaAtom {
            name: "MfmaScale32x32x64F8",
            mnemonic: "v_mfma_scale_f32_32x32x64_f8f6f4",
            m: 32,
            n: 32,
            k: 64,
            ab_bits: 8,
            a_tv: ab.clone(),
            b_tv: ab,
            c_tv: make_layout(tup![tup![32, 2], tup![4, 4]], tup![tup![32, 4], tup![1, 8]]),
            a_dwords: 8,
            b_dwords: 8,
            c_dwords: 16,
            scaled: true,
            passes: 16,
            provenance: Provenance::Shipped("fmha _vs128 QK/PV (K rows, Q columns, KeyMask C rows)"),
        }
    }
    /// `v_mfma_scale_f32_16x16x128_f8f6f4` (E4M3): lane l holds row l % 16 and k = 16 (l / 16) + 0..15 and 64 + 16 (l / 16) + 0..15; C lane l value r is
    /// (row 4 (l / 16) + r, column l % 16).
    pub fn mfma_scale_16x16x128_f8() -> MmaAtom {
        let ab = make_layout(tup![tup![16, 4], tup![16, 2]], tup![tup![1, 256], tup![16, 1024]]);
        MmaAtom {
            name: "MfmaScale16x16x128F8",
            mnemonic: "v_mfma_scale_f32_16x16x128_f8f6f4",
            m: 16,
            n: 16,
            k: 128,
            ab_bits: 8,
            a_tv: ab.clone(),
            b_tv: ab,
            c_tv: make_layout(tup![tup![16, 4], 4], tup![tup![16, 4], 1]),
            a_dwords: 8,
            b_dwords: 8,
            c_dwords: 4,
            scaled: true,
            passes: 8,
            provenance: Provenance::Derived("aiter 4-wave blockscale GEMM (B preshuffle (n%16)*16 + k%16 per 256-byte block), docs/mxfp8-gemm-4wave-analysis.md"),
        }
    }
    /// `v_mfma_f32_32x32x8_f16` (the FMHA bias trick): lane l holds row l % 32 and k = 4 (l / 32) + 0..3; C as the 32x32 atoms.
    pub fn mfma_32x32x8_f16() -> MmaAtom {
        let ab = make_layout(tup![tup![32, 2], 4], tup![tup![1, 128], 32]);
        MmaAtom {
            name: "Mfma32x32x8F16",
            mnemonic: "v_mfma_f32_32x32x8_f16",
            m: 32,
            n: 32,
            k: 8,
            ab_bits: 16,
            a_tv: ab.clone(),
            b_tv: ab,
            c_tv: make_layout(tup![tup![32, 2], tup![4, 4]], tup![tup![32, 4], tup![1, 8]]),
            a_dwords: 2,
            b_dwords: 2,
            c_dwords: 16,
            scaled: false,
            passes: 8,
            provenance: Provenance::Shipped("fmha _vs128 ones x bias reference MFMA"),
        }
    }
    /// `v_smfmac_f32_16x16x128_fp8_fp8` (sparse A, 2:4 compressed; recovered with one-hot hardware probes, `data/smfmac_16x16x128_fp8.map`, docs/isa-probes-smfmac-tr.md).
    /// Operand tiles in the usual element convention (A `m + 16 c`, B `n + 16 k`, C `m + 16 n`):
    /// * A is the COMPRESSED tile, 16 rows x 64 stored values: lane l (row m = l % 16, lane group g = l / 16) holds stored values s = 0..15 (register s / 4, byte s % 4) =
    ///   compressed column c = 16 g + s. Which logical K each stored value multiplies is chosen by the lane's SRC2 index word (see [`smfmac`]).
    /// * B is the dense 128 x 16 tile: lane l (column n = l % 16, lane group h = l / 16) holds k = 32 h + 4 r + y (register r = 0..7, byte y).
    /// * C: lane l dword r = (row 4 (l / 16) + r, column l % 16).
    pub fn smfmac_16x16x128_fp8() -> MmaAtom {
        MmaAtom {
            name: "Smfmac16x16x128Fp8",
            mnemonic: "v_smfmac_f32_16x16x128_fp8_fp8",
            m: 16,
            n: 16,
            k: 128,
            ab_bits: 8,
            a_tv: make_layout(tup![tup![16, 4], 16], tup![tup![1, 256], 16]),
            b_tv: make_layout(tup![tup![16, 4], 32], tup![tup![1, 512], 16]),
            c_tv: make_layout(tup![tup![16, 4], 4], tup![tup![16, 4], 1]),
            a_dwords: 4,
            b_dwords: 8,
            c_dwords: 4,
            scaled: false,
            passes: 8,
            provenance: Provenance::Probe("tests/gpu/isa_probe_host.cpp (probe_smfmac_gen), crates/amdgpu-isa/data/smfmac_16x16x128_fp8.map"),
        }
    }
    /// The atom with its A and B operands exchanged (hardware srcA = the B fragment): computes `C^T`, so the C layout is transposed (row and column swap).
    pub fn swap_ab(&self) -> MmaAtom {
        let mut s = self.clone();
        std::mem::swap(&mut s.a_tv, &mut s.b_tv);
        std::mem::swap(&mut s.m, &mut s.n);
        std::mem::swap(&mut s.a_dwords, &mut s.b_dwords);
        s
    }
}

/// Index semantics of the sparse matrix instruction `v_smfmac_f32_16x16x128_fp8_fp8`, measured (docs/isa-probes-smfmac-tr.md).
pub mod smfmac {
    /// Number of stored A values per lane (register s / 4, byte s % 4).
    pub const STORED: i64 = 16;
    /// The 2-bit index field of stored value `s` of a lane: bits `2 s ..= 2 s + 1` of THAT lane's SRC2 dword. The two values of a pair (2 q, 2 q + 1) are the
    /// two kept entries of one group of 4 (their fields are the low and the high half of nibble q); the hardware accepts any two fields.
    pub const fn field(idx: u32, s: i64) -> i64 {
        ((idx >> (2 * s)) & 3) as i64
    }
    /// The B operand dword that pair `q` (stored values 2 q, 2 q + 1) of an A lane in lane group `g` contracts with: (B lane group, B register); the group's 4
    /// candidate k's are the four bytes of that dword and the index field picks the byte.
    pub const fn b_dword(g: i64, q: i64) -> (i64, i64) {
        (2 * (g & 1) + q / 4, 4 * (g >> 1) + q % 4)
    }
    /// The logical K (in the B tile's `32 h + 4 r + y` numbering) that stored value `s` of A lane `l` multiplies under index word `idx`.
    pub const fn k_of(l: i64, s: i64, idx: u32) -> i64 {
        let (h, r) = b_dword(l / 16, s / 2);
        32 * h + 4 * r + field(idx, s)
    }
    /// Index nibble of a pair kept at positions (p0 < p1) of its group of 4: p0 in the low 2 bits, p1 in the high 2 bits.
    pub const fn nibble(p0: u32, p1: u32) -> u32 {
        p0 | (p1 << 2)
    }
    /// An index word with the same pair of kept positions in all 8 nibbles (`0x44444444` = positions (0, 1), `0xeeeeeeee` = (2, 3): together they cover all 4 bytes of
    /// every B dword, i.e. two accumulated smfmacs with these words contract all 128 k).
    pub const fn uniform(p0: u32, p1: u32) -> u32 {
        nibble(p0, p1) * 0x1111_1111
    }
}

/// `ds_read_b64_tr_b8` (one instruction; the shipped atom [`DS_READ_TR_B8X4`] issues four), measured (data/lds_tr_b8.map, docs/isa-probes-smfmac-tr.md).
pub mod tr_b8 {
    use super::*;
    /// Destination lane `d`, byte `j` (register j / 4, byte j % 4) receives byte `d % 8` of the 8-byte word read by lane `16 (d / 16) + 2 j + (d % 16) / 8`
    /// (independent of the addresses: inside each 16-lane group the eight words of the even lanes are transposed 8 x 8 bytes into dest lanes 0..7 and the eight
    /// words of the odd lanes into dest lanes 8..15). Returns (source lane, byte offset inside its word).
    pub const fn source(d: i64, j: i64) -> (i64, i64) {
        (16 * (d / 16) + 2 * j + (d % 16) / 8, d % 8)
    }
    /// The same map as a TV layout: (dest lane, byte j) -> element `8 s' + h` of the source tile "64 lanes x 8 bytes" (s' = source lane, h = byte offset).
    pub fn dest_tv() -> Layout {
        make_layout(tup![tup![8, 2, 4], 8], tup![tup![1, 8, 128], 16])
    }
}
