//! TiledCopy: an atom + a thread-value layout, partitioned over an LDS view.

use crate::mma::TiledMma;
use crate::Ctx;
use amdgpu_dsl::tile::shared::{PrivRef, Shared, SlotRef};
use amdgpu_dsl::{FnBuilder, Opnd, ValueId, I32};
use amdgpu_isa::{CopyAtom, Space};
use amdgpu_layout::{layout_to_offsets, proofs, ComposedLayout, IssuePlan, Ix, Layout};

/// An atom applied by every thread of a TV layout. `elem_bits` = bits of one element of the tile (8 for fp8).
#[derive(Clone, Debug)]
pub struct TiledCopy {
    pub atom: CopyAtom,
    pub tv: Layout,
    pub elem_bits: u32,
}

impl TiledCopy {
    /// Elements per instruction.
    pub fn vec(&self) -> i64 {
        if self.atom.transpose {
            1 // a transposing read is addressed by one quad-word per lane: the TV layout lists the address (first byte) of each read
        } else {
            self.atom.bytes * 8 / self.elem_bits as i64
        }
    }
}

/// `make_tiled_copy(atom, tv, tile)`: proves (build time) that the TV layout covers the `tile`-element tile exactly once (skipped for transposing atoms,
/// whose TV layout lists one source address per instruction) and that every thread's values split into whole instructions.
pub fn make_tiled_copy(atom: CopyAtom, tv: Layout, tile_elems: i64, elem_bits: u32) -> TiledCopy {
    let tc = TiledCopy { atom, tv, elem_bits };
    let v = tc.tv.get(1).size().st();
    assert!(v % tc.vec() == 0, "make_tiled_copy({}): {v} values per thread are not whole instructions of {} elements", atom.name, tc.vec());
    if !atom.transpose {
        if let Err(e) = proofs::check_tv(&tc.tv, tile_elems, false) {
            panic!("make_tiled_copy({}): {e}", atom.name);
        }
    }
    tc
}

/// `make_tiled_copy` for a BROADCAST TV layout (several threads read the same elements, e.g. both lane halves of the K-scale words): every element is
/// covered the same number of times.
pub fn make_tiled_copy_bcast(atom: CopyAtom, tv: Layout, tile_elems: i64, elem_bits: u32) -> TiledCopy {
    let tc = TiledCopy { atom, tv, elem_bits };
    if let Err(e) = proofs::check_tv(&tc.tv, tile_elems, true) {
        panic!("make_tiled_copy_bcast({}): {e}", atom.name);
    }
    tc
}

/// The copy that loads the A operand of `mma` (registers laid out as the atom expects, K permutation included).
pub fn make_tiled_copy_a(atom: CopyAtom, mma: &TiledMma) -> TiledCopy {
    let a = mma.atom();
    make_tiled_copy(atom, mma.a_tv(), a.m * a.k, a.ab_bits)
}
/// The copy that loads the B operand of `mma`.
pub fn make_tiled_copy_b(atom: CopyAtom, mma: &TiledMma) -> TiledCopy {
    let a = mma.atom();
    make_tiled_copy(atom, mma.b_tv(), a.n * a.k, a.ab_bits)
}

/// A TiledCopy partitioned for this lane over an LDS buffer: base registers (bound once) and, per operation, its instructions' (base, immediate).
#[derive(Clone, Debug)]
pub struct LdsPart {
    pub sh: Shared,
    pub atom: CopyAtom,
    pub bases: Vec<I32>,
    /// per operation (= `atom.insts` consecutive instructions): (base index, immediate)
    pub ops: Vec<Vec<(usize, i64)>>,
}

impl TiledCopy {
    /// Partitions the copy over the LDS buffer `sh` viewed through `view` (tile coordinates -> slot-relative byte offset, swizzle included). Every entry of
    /// `extras` is one placement of the tile inside the slot (a static byte offset, e.g. `4096 s + 64 c` for the K operand tiles of `_vs128`); the
    /// operations are ordered placement-major, then by the atom's instruction groups. The lane parts are bound here (once per kernel).
    pub fn partition_lds(&self, b: &mut FnBuilder, ctx: &Ctx, lane: &Ix, sh: &Shared, view: &ComposedLayout, extras: &[i64]) -> LdsPart {
        assert!((self.atom.src == Space::Lds && self.atom.dst == Space::Reg) || (self.atom.src == Space::Reg && self.atom.dst == Space::Lds), "partition_lds: {} is not an LDS read or write", self.atom.name);
        let vec = self.vec();
        let plan = IssuePlan::new(self.tv.clone(), vec);
        if !self.atom.transpose {
            if let Err(e) = proofs::check_contiguous(&self.tv, view, vec, (self.elem_bits / 8) as i64, self.atom.align) {
                panic!("partition_lds({} over {}): {e}", self.atom.name, sh.name);
            }
        }
        let mut exprs: Vec<Ix> = Vec::new();
        for &x in extras {
            exprs.extend(plan.offsets(&view.inner, lane, &Ix::C(x), &ctx.dyns));
        }
        let op = layout_to_offsets(&exprs, &view.swz, self.atom.imm, &ctx.dyns);
        // build-time proofs on the concrete lanes (the lane symbol must be the first dynamic symbol of a single-symbol plan)
        let insts = self.atom.insts as usize;
        assert!(op.issues.len() % insts == 0, "partition_lds: {} instructions are not whole {}-instruction operations", op.issues.len(), insts);
        let bases: Vec<I32> = op.bases.iter().map(|x| { let v = ctx.bind(b, x); if sh.base == 0 { v } else { b.let_i32(v + sh.base as i32) } }).collect();
        let ops: Vec<Vec<(usize, i64)>> = op.issues.chunks(insts).map(|c| c.to_vec()).collect();
        LdsPart { sh: sh.clone(), atom: self.atom, bases, ops }
    }
}

impl LdsPart {
    /// The partition at one ring slot (one add per base, once per tile; the slot offset is shared with the other users of the slot).
    pub fn at<'a>(&self, b: &mut FnBuilder, slot: &SlotRef<'a>) -> LdsSlotPart<'a> {
        assert_eq!(slot.sh.name, self.sh.name, "LdsPart::at: slot of another buffer");
        let off = slot.off_v(b);
        let bases = if off.as_const() == Some(0) { self.bases.clone() } else { self.bases.iter().map(|x| b.let_i32(x + &off)).collect() };
        LdsSlotPart { slot: slot.clone(), atom: self.atom, bases, ops: self.ops.clone() }
    }
    /// The partition in this wave's slice of a private region (`base + wave * size`, the address derived from the region's wave key).
    pub fn at_private<'a>(&self, b: &mut FnBuilder, p: &'a PrivRef<'a>) -> LdsPrivPart<'a> {
        assert_eq!(p.sh.name, self.sh.name, "LdsPart::at_private: another region");
        assert_eq!(self.sh.base, 0, "at_private: partition the private region with base 0 (the slice base adds it)");
        let lbase = b.let_i32(p.base());
        let bases = self.bases.iter().map(|x| if x.as_const() == Some(0) { lbase.clone() } else { b.let_i32(&lbase + x) }).collect();
        LdsPrivPart { p: p.clone(), atom: self.atom, bases, ops: self.ops.clone() }
    }
    pub fn len(&self) -> usize {
        self.ops.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

fn addrs(b: &mut FnBuilder, bases: &[I32], op: &[(usize, i64)]) -> Vec<ValueId> {
    op.iter().map(|&(bi, imm)| if imm == 0 { b.bind_i32(&bases[bi]) } else { b.bind_i32(&bases[bi] + imm as i32) }).collect()
}

fn emit_write(b: &mut FnBuilder, a: CopyAtom, addrs: &[ValueId], vals: &[ValueId]) {
    assert_eq!(addrs.len(), vals.len(), "one value per instruction");
    for (ad, v) in addrs.iter().zip(vals) {
        match a.bytes {
            4 => b.store_lds_b32(Opnd::Val(*ad), Opnd::Val(*v)),
            16 => b.store_lds_b128(Opnd::Val(*ad), Opnd::Val(*v)),
            n => panic!("no {n}-byte LDS write form"),
        }
    }
}

/// A partition bound to a wave's private slice (generation-marked accesses of the wave's own data).
#[derive(Clone, Debug)]
pub struct LdsPrivPart<'a> {
    pub p: PrivRef<'a>,
    pub atom: CopyAtom,
    bases: Vec<I32>,
    ops: Vec<Vec<(usize, i64)>>,
}

impl LdsPrivPart<'_> {
    /// Writes operation `o` (one value per instruction) as generation `gen`.
    pub fn write(&self, b: &mut FnBuilder, o: usize, gen: i32, vals: &[ValueId]) {
        let ad = addrs(b, &self.bases, &self.ops[o]);
        let a = self.atom;
        self.p.write(b, gen, |b| emit_write(b, a, &ad, vals));
    }
    /// Reads operation `o` back (generation `gen`; the writes must have completed: `wait_lds(0)`).
    pub fn read(&self, b: &mut FnBuilder, o: usize, gen: i32) -> ValueId {
        let ad = addrs(b, &self.bases, &self.ops[o]);
        assert!(self.atom.bytes == 16 && self.atom.insts == 1, "LdsPrivPart::read: ds_read_b128 only");
        self.p.read_back(b, gen, |b| b.load_lds_b128(Opnd::Val(ad[0])))
    }
    pub fn len(&self) -> usize {
        self.ops.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

/// A partition bound to one slot.
#[derive(Clone, Debug)]
pub struct LdsSlotPart<'a> {
    pub slot: SlotRef<'a>,
    pub atom: CopyAtom,
    bases: Vec<I32>,
    ops: Vec<Vec<(usize, i64)>>,
}

impl LdsSlotPart<'_> {
    /// Writes operation `o` (one value per instruction, under the slot's write marker).
    pub fn write(&self, b: &mut FnBuilder, o: usize, vals: &[ValueId]) {
        let ad = addrs(b, &self.bases, &self.ops[o]);
        let a = self.atom;
        self.slot.write(b, |b| emit_write(b, a, &ad, vals));
    }
    /// Emits operation `o` (its instructions, under the slot's read marker) and returns the fragment.
    pub fn read(&self, b: &mut FnBuilder, o: usize) -> ValueId {
        let addrs: Vec<ValueId> = self.ops[o].iter().map(|&(bi, imm)| if imm == 0 { b.bind_i32(&self.bases[bi]) } else { b.bind_i32(&self.bases[bi] + imm as i32) }).collect();
        let a = self.atom;
        self.slot.read(b, |b| match (a.bytes, a.insts, a.transpose) {
            (16, 1, false) => b.load_lds_b128(Opnd::Val(addrs[0])),
            (16, 2, false) => b.load_lds_b128x2(Opnd::Val(addrs[0]), Opnd::Val(addrs[1])),
            (4, 1, false) => b.load_lds_b32(Opnd::Val(addrs[0])),
            (8, 1, false) => b.load_lds_b64(Opnd::Val(addrs[0])),
            (8, 4, true) => b.load_lds_tr_b8x4(addrs[0], addrs[1], addrs[2], addrs[3]),
            (8, 1, true) => b.load_lds_tr_b8(Opnd::Val(addrs[0])),
            _ => panic!("LdsSlotPart::read: no IR form for the atom {}", a.name),
        })
    }
    pub fn len(&self) -> usize {
        self.ops.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}
