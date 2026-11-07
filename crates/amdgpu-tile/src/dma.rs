//! Global -> LDS DMA partitions (`global_load_lds_dwordx4`, and the MUBUF `buffer_load_dwordx4 ... lds` form with the same lane pattern).
//!
//! The DMA's destination is FIXED: lane i of instruction q writes LDS bytes `M0 + imm + 16 i`. So the copy is derived from the destination side: the slot
//! byte `d = wave * slice + step * q + 16 lane` holds tile element `view^-1(d)`; for a row-major view with an XOR swizzle on the 16-byte chunks,
//! `row = d / row_bytes` (the swizzle does not touch the row bits) and `chunk' = chunk ^ s(row)` - the swizzle moves to the SOURCE address. Per lane the
//! row (without the tile's first row) and the distinct column offsets are bound once; per tile only `key0 + row`, the clamp and the stride multiply are
//! emitted, with M0 at the top of the wave's slice and negative instruction offsets compensated on the global side (a 32-bit pointer offset is
//! zero-extended: a negative one would wrap) - the instruction stream of the hand `RowsCopy`.

use crate::Ctx;
use crate::rows::RowSource;
use crate::shared::{Shared, SlotRef};
use amdgpu_dsl::{FnBuilder, ValueId, I32};
use amdgpu_isa::CopyAtom;
use amdgpu_layout::{ComposedLayout, Ix};

pub struct DmaPart {
    atom: CopyAtom,
    src: RowSource,
    sh: Shared,
    /// lane part of the tile row of every instruction (bound) + the instruction's static row offset
    row_dyn: I32,
    row_c: Vec<i64>,
    /// distinct column byte offsets (bound) and the index used by every instruction
    cols: Vec<I32>,
    col_of: Vec<usize>,
    /// `M0 = sh.base + wave * slice + step * (n - 1)` without the slot offset (bound)
    lds_wave: I32,
    step: i64,
    /// bytes of one row of the view (the record-range slack of the buffer form)
    row_bytes: i64,
}

/// Partitions a DMA of `issues` instructions per wave: wave `wave` (0..waves) fills bytes `[wave * slice, +slice)` of the slot, `slice = issues * 64 *
/// atom.bytes`. `view` = the LDS layout of the tile (row-major rows, XOR swizzle on 16-byte chunks), `src` = the tile's rows in global memory.
#[allow(clippy::too_many_arguments)]
pub fn partition_dma(b: &mut FnBuilder, ctx: &Ctx, atom: CopyAtom, lane: &Ix, wave: &Ix, wave_val: &I32, issues: i64, src: RowSource, sh: &Shared, view: &ComposedLayout) -> DmaPart {
    let lane_stride = atom.lds_lane_stride.unwrap_or_else(|| panic!("partition_dma: {} is not an LDS-direct DMA", atom.name));
    let step = 64 * lane_stride;
    let slice = issues * step;
    let fm = view.inner.flat_modes();
    assert!(fm.len() == 2 && fm[1].1 == Ix::C(1), "partition_dma: the view must be row-major [rows, row_bytes]: {}", view.inner);
    let row_bytes = fm[0].1.st();
    assert_eq!(fm[1].0.st(), row_bytes, "partition_dma: dense rows only");
    assert!(view.swz.is_identity() || view.swz.dst_lo >= 4 && view.swz.dst_lo + view.swz.dst_n <= row_bytes.trailing_zeros(), "partition_dma: the swizzle must move whole 16-byte chunks inside a row");
    let d = &ctx.dyns;
    // the slot byte of lane / instruction q, split into its dynamic part and the static step
    let a = lane.mul_c(lane_stride).add(&wave.mul_c(slice));
    let row_a = a.div_c(row_bytes, d);
    let row_dyn = ctx.bind(b, &row_a);
    let mut cols: Vec<Ix> = Vec::new();
    let mut col_of = Vec::new();
    let mut row_c = Vec::new();
    for q in 0..issues {
        let c = step * q;
        assert!(a.bits(d) & c as u64 == 0, "partition_dma: lane bytes and the instruction step overlap");
        row_c.push(c / row_bytes);
        // column of the destination byte, un-swizzled: (a mod row_bytes) ^ s(row of a + c)
        // in 16-byte chunks (the swizzle moves whole chunks): (chunk ^ s) << 4 keeps the verifier's interval of the column inside the row
        let chunk0 = a.add_c(c).mod_c(row_bytes, d).div_c(16, d);
        let chunk = if view.swz.is_identity() { chunk0 } else { chunk0.xor(&view.swz.term_ix_down(&a.add_c(c), d, 4)) };
        let col = chunk.shl(4);
        let i = match cols.iter().position(|x| *x == col) {
            Some(i) => i,
            None => {
                cols.push(col);
                cols.len() - 1
            }
        };
        col_of.push(i);
    }
    let cols: Vec<I32> = cols.iter().map(|c| ctx.bind(b, c)).collect();
    let top = slice - step;
    let lds_wave = b.let_i32((wave_val * slice as i32 + top as i32) + sh.base as i32);
    DmaPart { atom, src, sh: sh.clone(), row_dyn, row_c, cols, col_of, lds_wave, step, row_bytes }
}

impl DmaPart {
    /// The DMA of the tile whose first row is `key0` (relative to the source origin) into `slot` (a write of its generation).
    pub fn issue(&self, b: &mut FnBuilder, key0: &I32, slot: &SlotRef) {
        assert_eq!(slot.sh.name, self.sh.name, "DmaPart: slot of another buffer");
        let rb = b.let_i32(key0 + &self.row_dyn);
        let off = slot.off_v(b);
        let lbase: ValueId = b.bind_i32(&self.lds_wave + off);
        let n = self.row_c.len() as i64;
        let dwords = (self.atom.bytes / 4) as u8;
        slot.write(b, |b| {
            for q in 0..n as usize {
                let comp = (self.step * (n - 1 - q as i64)) as i32;
                let row = self.src.row(&rb + self.row_c[q] as i32);
                let o = b.bind_i32(add_c(add0(add0(row * &self.src.stride, &self.src.base), &self.cols[self.col_of[q]]), comp));
                let sp = b.ptr_add(self.src.ptr, o);
                b.load_global_lds(sp, lbase, -comp, dwords);
            }
        });
    }
}

/// The MUBUF form of a `DmaPart` (the vendor's staging): `buffer_load_dwordx4 ... lds` with ONE loop-invariant voffset per instruction (lane row x
/// stride + swizzled column - instruction step), the per-tile advance `key0 x stride` in the SGPR soffset and M0 = the wave's slice base, the
/// instruction immediate = the step. Rows past the source's clamp limit are not clamped but fall outside the descriptor's record range (they
/// read zeros: the keys past the sequence are masked, the V rows multiply zero P).
pub struct BufForm {
    rsrc: ValueId,
    /// the tile origin byte offset (SGPR)
    base: ValueId,
    voffs: Vec<ValueId>,
}

impl DmaPart {
    pub fn buffer_form(&self, b: &mut FnBuilder) -> BufForm {
        assert!(self.atom.bytes == 16 || self.atom.bytes == 4, "DmaPart::buffer_form: 16- or 4-byte DMA only");
        let n = self.row_c.len();
        // the descriptor base must be the kernarg pointer: the tile origin (wave-uniform) goes into the soffset and the record range
        let bo = b_bind(b, &self.src.base);
        let bo = b.read_first_lane(bo);
        let nrec = match &self.src.clamp {
            Some(c) => b.bind_i32(ev_(bo) + (c.limit.clone() - 1) * &self.src.stride + self.row_bytes as i32),
            None => b.c_i32(0x7fff_ffff),
        };
        let nrec = b.read_first_lane(nrec);
        let rsrc = b.buffer_desc_dyn(self.src.ptr, nrec);
        let voffs = (0..n)
            .map(|q| {
                let r = &self.row_dyn + self.row_c[q] as i32;
                b.bind_i32(add0(r * &self.src.stride, &self.cols[self.col_of[q]]) - (self.step * q as i64) as i32)
            })
            .collect();
        BufForm { rsrc, base: bo, voffs }
    }
    /// The DMA of the tile at `key0` into `slot` through the buffer form.
    pub fn issue_buffer(&self, b: &mut FnBuilder, bf: &BufForm, key0: &I32, slot: &SlotRef) {
        self.issue_buffer_units(bf, key0, slot).emit_all(b);
    }
    /// `issue_buffer` as emission units: the per-tile setup (soffset, M0) and one unit per DMA instruction (each under the slot's write marker).
    pub fn issue_buffer_units<'s>(&'s self, bf: &'s BufForm, key0: &I32, slot: &SlotRef<'s>) -> crate::weave::Units<'s> {
        use crate::weave::{Hint, Out, UnitClass, Units};
        assert_eq!(slot.sh.name, self.sh.name, "DmaPart: slot of another buffer");
        let n = self.row_c.len() as i64;
        let mut u = Units::new();
        let regs: Out<(ValueId, ValueId)> = Out::new();
        let (k0, sl, r2) = (key0.clone(), slot.clone(), regs.clone());
        let setup = u.add(format!("dma.{}.setup", self.sh.name), UnitClass::Setup, Hint::Anchor, &[], move |b| {
            let so = {
                let v = b.bind_i32(k0 * &self.src.stride + ev_(bf.base));
                b.read_first_lane(v)
            };
            let off = sl.off_v(b);
            let m0 = b.bind_i32(&self.lds_wave + off - (self.step * (n - 1)) as i32);
            // wave-uniform: ONE readfirstlane per tile (the DMA's M0 / soffset are SGPRs; a VGPR value is re-read before every issue)
            let m0 = b.read_first_lane(m0);
            r2.set((so, m0));
        });
        for q in 0..n as usize {
            let (sl, r3) = (slot.clone(), regs.clone());
            u.add(format!("dma.{}.{q}", self.sh.name), UnitClass::Dma, Hint::LatencySensitive, &[setup], move |b| {
                let (so, m0) = r3.get();
                sl.write(b, |b| {
                    if self.atom.bytes == 4 {
                        b.buffer_load_lds_b32(bf.rsrc, bf.voffs[q], so, m0, (self.step * q as i64) as u32);
                    } else {
                        b.buffer_load_lds_b128(bf.rsrc, bf.voffs[q], so, m0, (self.step * q as i64) as u32);
                    }
                });
            });
        }
        u
    }
}

fn ev_(v: ValueId) -> I32 {
    amdgpu_dsl::ev(v)
}

fn b_bind(b: &mut FnBuilder, x: &I32) -> ValueId {
    b.bind_i32(x.clone())
}

fn add_c(a: I32, k: i32) -> I32 {
    if k == 0 {
        a
    } else {
        a + k
    }
}

fn add0(a: I32, b: &I32) -> I32 {
    if b.as_const() == Some(0) {
        a
    } else {
        a + b
    }
}

/// Several DMA partitions over the SAME rows (one producer group staging K and V of a tile, `_t128`): the clamped rows are computed once and shared;
/// each partition adds its own stride, base and column offsets and writes its own slot (own marker).
pub fn issue_rows_shared(b: &mut FnBuilder, key0: &I32, parts: &[(&DmaPart, &SlotRef)]) {
    let (first, _) = parts[0];
    assert!(parts.iter().all(|(p, _)| p.row_c == first.row_c), "issue_rows_shared: partitions of different row shapes");
    let rb = b.let_i32(key0 + &first.row_dyn);
    let rows: Vec<I32> = first.row_c.iter().map(|&c| b.let_i32(first.src.row(&rb + c as i32))).collect();
    for (p, slot) in parts {
        assert_eq!(slot.sh.name, p.sh.name, "issue_rows_shared: slot of another buffer");
        let off = slot.off_v(b);
        let lbase: ValueId = b.bind_i32(&p.lds_wave + off);
        let n = p.row_c.len() as i64;
        let dwords = (p.atom.bytes / 4) as u8;
        slot.write(b, |b| {
            for q in 0..n as usize {
                let comp = (p.step * (n - 1 - q as i64)) as i32;
                let o = b.bind_i32(add_c(add0(add0(&rows[q] * &p.src.stride, &p.src.base), &p.cols[p.col_of[q]]), comp));
                let sp = b.ptr_add(p.src.ptr, o);
                b.load_global_lds(sp, lbase, -comp, dwords);
            }
        });
    }
}
