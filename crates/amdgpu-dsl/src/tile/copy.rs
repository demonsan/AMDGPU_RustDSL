//! L3a: global -> LDS copies (direct DMA: `global_load_lds_dwordx4` / `global_load_lds_dword`). The per-lane addressing (source-side swizzle, M0 at the top
//! of the wave's slice with negative instruction offsets and a positive address compensation, row clamps) is computed here once per kernel; `issue` adds
//! only the per-tile terms and wraps the DMA in the protocol markers of the destination slot.

use super::shared::{Layout, Shared, SlotRef, Swz};
use super::tensor::{Clamp, TileView};
use super::{add0, ev, fmha_swz, Lane};
use crate::{FnBuilder, ValueId, I32};

/// A row-addressed global source: `ptr + base + row * stride (+ column bytes)`, rows clamped by the view's clamp.
#[derive(Clone, Debug)]
pub struct RowSource {
    pub ptr: ValueId,
    /// byte stride of one row
    pub stride: I32,
    /// tile-invariant byte offset (all the other coordinates + the origin row), bound once
    pub base: I32,
    pub clamp: Option<Clamp>,
}

impl TileView {
    /// The view as a row source along `row_dim`: the offset of the origin (every coordinate, unclamped) is bound ONCE here; rows are relative to the origin.
    pub fn row_source(&self, b: &mut FnBuilder, row_dim: usize) -> RowSource {
        let off = self.t.offset(&self.origin);
        let base = if super::is_zero(&off) { off } else { b.let_i32(off) };
        RowSource { ptr: self.t.ptr, stride: self.t.strides[row_dim].clone(), base, clamp: self.clamps[row_dim].clone() }
    }
}

impl RowSource {
    /// The (clamped) relative row index.
    pub fn row(&self, r: I32) -> I32 {
        match &self.clamp {
            None => r,
            Some(c) if c.use_min => r.min(&c.last),
            Some(c) => r.lt(&c.limit).select_i32(r.clone(), c.last.clone()),
        }
    }
}

/// Copy of a [rows x 128 B] tile into a `Layout::FMHA_SWZ` buffer: wave w of the producer group copies rows `rows_per_wave * w ..`; one DMA instruction moves
/// 8 rows (64 lanes x 16 B); lane i of DMA q fetches chunk `(i & 7) ^ s(row)` of row `8 q + (i >> 3)` so that the LINEAR LDS landing is the swizzled layout.
pub struct RowsCopy {
    src: RowSource,
    dst: Shared,
    rowb: I32,
    cbytes: [I32; 2],
    lds_wave: I32,
    dmas: i32,
}

impl RowsCopy {
    /// `wave` = the wave index inside the producer group (0 .. waves-1).
    pub fn new(b: &mut FnBuilder, src: RowSource, dst: &Shared, wave: ValueId, lane: &Lane, waves: i64) -> RowsCopy {
        let (row_bytes, swz) = match dst.layout {
            Layout::Rows { row_bytes, swz } => (row_bytes, swz),
            Layout::Linear => panic!("RowsCopy: destination {} needs a row layout", dst.name),
        };
        assert_eq!(row_bytes, 128, "RowsCopy: 128-byte rows (8 chunks of 16 B; 8 rows per DMA instruction)");
        let rows = dst.bytes / row_bytes;
        assert_eq!(rows % (8 * waves), 0, "RowsCopy: {} rows over {} waves", rows, waves);
        let rows_per_wave = rows / waves;
        let dmas = (rows_per_wave / 8) as i32;
        let lane_hi = b.let_i32(lane.lane().lshr(3));
        let lane_lo = b.let_i32(lane.lane() & 7);
        // the swizzle depends on the row bits 0..3: rows 8q + lane_hi of an even / odd q
        let cbytes = if swz == Swz::Fmha {
            let lane_hi8 = b.let_i32(&lane_hi + 8);
            let s_even = fmha_swz(b, &lane_hi);
            let s_odd = fmha_swz(b, &lane_hi8);
            let ce = b.let_i32((&lane_lo ^ s_even) << 4);
            let co = b.let_i32((&lane_lo ^ s_odd) << 4);
            [ce, co]
        } else {
            let c = b.let_i32(&lane_lo << 4);
            [c.clone(), c]
        };
        let shift = (rows_per_wave as u32).trailing_zeros() as i32;
        assert_eq!(1i64 << shift, rows_per_wave, "RowsCopy: rows per wave must be a power of two");
        let rowb = b.let_i32((ev(wave) << shift) + &lane_hi);
        let slice = (rows_per_wave * row_bytes) as i32;
        // M0 at the top of the wave's slice: instruction offsets -(1024 (dmas-1-q)), compensated by a positive address term (a 32-bit pointer offset is
        // zero-extended: a negative one would wrap to +4 GB)
        let top = slice - 1024;
        let lds_wave = b.let_i32((ev(wave) * slice + top) + dst.base as i32);
        RowsCopy { src, dst: dst.clone(), rowb, cbytes, lds_wave, dmas }
    }
    /// DMA of the tile whose first row is `key0` (relative to the source origin) into `slot` (marked as a write of its generation).
    pub fn issue(&self, b: &mut FnBuilder, key0: &I32, slot: &SlotRef) {
        assert_eq!(slot.sh.name, self.dst.name, "RowsCopy: slot of another buffer");
        let rb = b.let_i32(key0 + &self.rowb);
        let off = slot.off_v(b);
        let lbase = b.bind_i32(&self.lds_wave + off);
        slot.write(b, |b| {
            for q in 0..self.dmas {
                let comp = 1024 * (self.dmas - 1 - q);
                let row = self.src.row(&rb + 8 * q);
                let off = b.bind_i32((add0(row * &self.src.stride, &self.src.base) + &self.cbytes[(q & 1) as usize]) + comp);
                let sp = b.ptr_add(self.src.ptr, off);
                b.load_global_lds(sp, lbase, -comp, 4);
            }
        });
    }
}

/// Copy of one dword per row (e.g. the K scales: 128 rows x 4 B): wave w covers rows `64 (w & 1) + lane` (waves 2, 3 duplicate waves 0, 1), one
/// `global_load_lds_dword` per wave.
pub struct DwordRowsCopy {
    src: RowSource,
    dst: Shared,
    row0: I32,
    lds_wave: I32,
}

impl DwordRowsCopy {
    pub fn new(b: &mut FnBuilder, src: RowSource, dst: &Shared, wave: ValueId, lane: &Lane) -> DwordRowsCopy {
        assert_eq!(dst.bytes, 512, "DwordRowsCopy: 128 rows x 4 B per slot");
        let wlo = b.let_i32(ev(wave) & 1);
        let row0 = b.let_i32((&wlo << 6) + lane.lane());
        let lds_wave = b.let_i32(&wlo * 256 + dst.base as i32);
        DwordRowsCopy { src, dst: dst.clone(), row0, lds_wave }
    }
    pub fn issue(&self, b: &mut FnBuilder, key0: &I32, slot: &SlotRef) {
        assert_eq!(slot.sh.name, self.dst.name, "DwordRowsCopy: slot of another buffer");
        let off = slot.off_v(b);
        slot.write(b, |b| {
            let row = self.src.row(key0 + &self.row0);
            let off = b.bind_i32(add0(row * &self.src.stride, &self.src.base));
            let p = b.ptr_add(self.src.ptr, off);
            let la = b.bind_i32(&self.lds_wave + off);
            b.load_global_lds(p, la, 0, 1);
        });
    }
}
