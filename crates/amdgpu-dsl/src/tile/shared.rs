//! L2: shared (LDS) buffers. An `LdsPlan` allocates every buffer once; `commit` emits the segment size and the `lds_ring/lds_tile/lds_private` declarations,
//! plus `lds_alias` ONLY where it is justified: (a) the declared extents of interleaved rings overlap although their real slots do not, (b) the author
//! declared time-multiplexing (`overlays`). A real overlap that was not declared is a build-time panic (it would otherwise be silently aliased).

use super::{ev, ox};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::{FnBuilder, Opnd, ValueId, I32};

/// Byte layout of a buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// offset = byte
    Linear,
    /// rows of `row_bytes`; 16-byte chunk c of row r stored at chunk `c ^ swz(r)`
    Rows { row_bytes: i64, swz: Swz },
}

/// XOR swizzle of the 16-byte chunks of a row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Swz {
    None,
    /// the FMHA bank-conflict-free swizzle (docs/archive/status.md, `swz`): `s2 = u2 ^ p, s1 = u0, s0 = u1 ^ p` with `u = (r >> 1) & 7`, `p = r & 1`
    Fmha,
}

impl Layout {
    /// The FMHA K/V tile layout: 128-byte rows, `Swz::Fmha`.
    pub const FMHA_SWZ: Layout = Layout::Rows { row_bytes: 128, swz: Swz::Fmha };
    /// Swizzled chunk index of chunk `c` of row `r`.
    pub fn chunk(&self, b: &mut FnBuilder, r: &I32, c: &I32) -> I32 {
        match self {
            Layout::Rows { swz: Swz::Fmha, .. } => c ^ fmha_swz(b, r),
            _ => c.clone(),
        }
    }
}

/// The FMHA swizzle `s(r)` of a row index (the four bit extractions are bound first, in the order of the manual `swz_s`, then combined).
pub fn fmha_swz(b: &mut FnBuilder, r: &I32) -> I32 {
    let p = b.let_i32(r & 1);
    let u0 = b.let_i32(r.lshr(1) & 1);
    let u1 = b.let_i32(r.lshr(2) & 1);
    let u2 = b.let_i32(r.lshr(3) & 1);
    let s2 = b.let_i32(&u2 ^ &p);
    let s0 = b.let_i32(&u1 ^ &p);
    b.let_i32(((s2 << 2) + (u0 << 1)) + s0)
}

/// One declared LDS buffer (a tile = a ring with one slot; a private region = one slot per wave).
#[derive(Clone, Debug)]
pub struct Shared {
    pub name: String,
    pub base: i64,
    /// distance between slots (= `bytes` unless the ring is interleaved with others in a ring group)
    pub slot_stride: i64,
    /// bytes of one slot
    pub bytes: i64,
    pub slots: u32,
    pub layout: Layout,
    /// per-wave private region (`lds_private`)
    pub private: bool,
}

impl Shared {
    /// The buffer at (slot, generation): read/write it through the returned handle.
    pub fn at<'a>(&'a self, idx: &SlotIdx) -> SlotRef<'a> {
        SlotRef { sh: self, idx: idx.clone() }
    }
    /// Byte offset of slot `slot` relative to the first slot (`slot * slot_stride`).
    pub fn slot_off(&self, slot: &I32) -> I32 {
        if self.slots == 1 {
            I32::c(0)
        } else {
            slot * (self.slot_stride as i32)
        }
    }
    /// The private slice of wave `wave` (the SAME value must be the address key, see `lds_private`).
    pub fn mine(&self, wave: ValueId) -> PrivRef<'_> {
        assert!(self.private, "{}: `mine` on a non-private buffer", self.name);
        PrivRef { sh: self, wave }
    }
    /// Real byte intervals occupied by the buffer.
    fn footprint(&self) -> Vec<(i64, i64)> {
        (0..self.slots as i64).map(|k| (self.base + k * self.slot_stride, self.base + k * self.slot_stride + self.bytes)).collect()
    }
    /// The extent the verifier sees (`lds_ring(name, base, slot_stride, slots)` declares `[base, base + slots * stride)`).
    fn declared(&self) -> (i64, i64) {
        let span = if self.slots == 1 { self.bytes } else { self.slot_stride * self.slots as i64 };
        (self.base, self.base + span)
    }
}

/// Ring slot index and generation (version) of one tile instance. Built by the pipeline iterator; never written by the kernel author.
#[derive(Clone, Debug)]
pub struct SlotIdx {
    pub slot: Opnd,
    pub gen: Opnd,
    /// bound slot * stride values, shared by every consumer of this slot instance (clones share the cache): the manual kernels compute
    /// slot_cur / slot_nxt / kds_nxt once per iteration. A cached value is only valid in the scope it was bound in: the pipeline creates a fresh
    /// SlotIdx per iteration at the iteration's top-level scope, and the components bind at that scope.
    offs: Rc<RefCell<HashMap<i64, ValueId>>>,
}

impl SlotIdx {
    pub fn new(slot: impl Into<Opnd>, gen: impl Into<Opnd>) -> SlotIdx {
        SlotIdx { slot: slot.into(), gen: gen.into(), offs: Default::default() }
    }
    /// slot * stride bound once per slot instance and stride (0 for a constant slot 0).
    pub fn off_v(&self, b: &mut FnBuilder, stride: i64) -> I32 {
        if let Opnd::I32(c) = self.slot {
            return I32::c(c * stride as i32);
        }
        if let Some(v) = self.offs.borrow().get(&stride) {
            return ev(*v);
        }
        let v = b.bind_i32(self.slot() * (stride as i32));
        self.offs.borrow_mut().insert(stride, v);
        ev(v)
    }
    /// The slot index as an expression.
    pub fn slot(&self) -> I32 {
        ox(self.slot)
    }
}

/// A buffer at one (slot, generation): every access goes through `read`/`write`, which bracket it with the protocol markers.
#[derive(Clone, Debug)]
pub struct SlotRef<'a> {
    pub sh: &'a Shared,
    pub idx: SlotIdx,
}

impl SlotRef<'_> {
    /// Slot offset relative to the buffer base (`slot * stride`) as an unbound expression (re-emitted at every bind: prefer `off_v`).
    pub fn off(&self) -> I32 {
        self.sh.slot_off(&self.idx.slot())
    }
    /// Slot offset bound once per slot instance and stride (shared with every other buffer of the same slot stride, e.g. the K and V halves of a ring group).
    pub fn off_v(&self, b: &mut FnBuilder) -> I32 {
        if self.sh.slots == 1 {
            return I32::c(0);
        }
        self.idx.off_v(b, self.sh.slot_stride)
    }
    /// Absolute LDS byte address of the slot.
    pub fn base(&self) -> I32 {
        self.off() + self.sh.base as i32
    }
    /// The accesses issued by `f` WRITE this tile instance (`tile_produce .. tile_end`).
    pub fn write<R>(&self, b: &mut FnBuilder, f: impl FnOnce(&mut FnBuilder) -> R) -> R {
        b.tile_produce(&self.sh.name, self.idx.slot, self.idx.gen);
        let r = f(b);
        b.tile_end();
        r
    }
    /// The accesses issued by `f` READ this tile instance (`tile_consume .. tile_end`).
    pub fn read<R>(&self, b: &mut FnBuilder, f: impl FnOnce(&mut FnBuilder) -> R) -> R {
        b.tile_consume(&self.sh.name, self.idx.slot, self.idx.gen);
        let r = f(b);
        b.tile_end();
        r
    }
}

/// One wave's slice of a private region.
#[derive(Clone, Debug)]
pub struct PrivRef<'a> {
    pub sh: &'a Shared,
    pub wave: ValueId,
}

impl PrivRef<'_> {
    /// Absolute LDS byte address of the slice (`base + wave * size`): every address of an access must be derived from it.
    pub fn base(&self) -> I32 {
        ev(self.wave) * (self.sh.slot_stride as i32) + self.sh.base as i32
    }
    /// Writes generation `gen` of the slice (the wave's own data).
    pub fn write<R>(&self, b: &mut FnBuilder, gen: impl Into<Opnd>, f: impl FnOnce(&mut FnBuilder) -> R) -> R {
        b.tile_produce(&self.sh.name, Opnd::Val(self.wave), gen);
        let r = f(b);
        b.tile_end();
        r
    }
    /// Reads generation `gen` back (the wave's own data: no barrier; the writes must have completed - `wait_lds(0)` once after the batch of writes,
    /// otherwise the verifier reports GV-TILE-STALE).
    pub fn read_back<R>(&self, b: &mut FnBuilder, gen: impl Into<Opnd>, f: impl FnOnce(&mut FnBuilder) -> R) -> R {
        b.tile_consume(&self.sh.name, Opnd::Val(self.wave), gen);
        let r = f(b);
        b.tile_end();
        r
    }
}

/// Allocation of the LDS segment.
#[derive(Default, Debug)]
pub struct LdsPlan {
    cursor: i64,
    bufs: Vec<Shared>,
    overlays: Vec<(String, String)>,
}

fn align16(x: i64) -> i64 {
    (x + 15) / 16 * 16
}

impl LdsPlan {
    pub fn new() -> LdsPlan {
        LdsPlan::default()
    }
    /// Places the next buffer at `off` (for byte-exact reproductions of a manual layout).
    pub fn at(&mut self, off: i64) -> &mut LdsPlan {
        self.cursor = off;
        self
    }
    /// Current allocation cursor.
    pub fn cursor(&self) -> i64 {
        self.cursor
    }
    fn push(&mut self, s: Shared) -> Shared {
        assert!(!self.bufs.iter().any(|x| x.name == s.name), "LdsPlan: buffer `{}` declared twice", s.name);
        self.bufs.push(s.clone());
        s
    }
    /// Interleaved ring group: slot k holds every member's tile k back to back (`_vs128`: slot = [K 16 KB | V 16 KB]).
    pub fn ring_group<const N: usize>(&mut self, slots: u32, parts: [(&str, i64, Layout); N]) -> [Shared; N] {
        let base = align16(self.cursor);
        let stride: i64 = parts.iter().map(|p| align16(p.1)).sum();
        let mut off = base;
        let mut out: Vec<Shared> = Vec::with_capacity(N);
        for (name, bytes, layout) in parts {
            out.push(self.push(Shared { name: name.to_string(), base: off, slot_stride: stride, bytes, slots, layout, private: false }));
            off += align16(bytes);
        }
        self.cursor = base + stride * slots as i64;
        out.try_into().unwrap_or_else(|_| unreachable!())
    }
    /// A ring of `slots` contiguous slots.
    pub fn ring(&mut self, name: &str, bytes: i64, slots: u32, layout: Layout) -> Shared {
        let base = align16(self.cursor);
        self.cursor = base + bytes * slots as i64;
        self.push(Shared { name: name.to_string(), base, slot_stride: bytes, bytes, slots, layout, private: false })
    }
    /// A single tile.
    pub fn tile(&mut self, name: &str, bytes: i64, layout: Layout) -> Shared {
        self.ring(name, bytes, 1, layout)
    }
    /// A per-wave private region (`waves` slices of `per_wave` bytes). With `overlays` non-empty it is placed at the base of the first buffer and reuses their storage
    /// (they must be dead when the region is used - e.g. the Q staging during the epilogue); otherwise it is allocated at the cursor.
    pub fn private(&mut self, name: &str, per_wave: i64, waves: u32, overlays: &[&Shared]) -> Shared {
        let base = match overlays.first() {
            Some(o) => o.base,
            None => {
                let b = align16(self.cursor);
                self.cursor = b + per_wave * waves as i64;
                b
            }
        };
        for o in overlays {
            self.overlays.push((name.to_string(), o.name.clone()));
        }
        self.push(Shared { name: name.to_string(), base, slot_stride: per_wave, bytes: per_wave, slots: waves, layout: Layout::Linear, private: true })
    }
    /// Declare that `a` reuses the storage of `b` (time multiplexing) without moving it.
    pub fn overlay(&mut self, a: &Shared, b: &Shared) {
        self.overlays.push((a.name.clone(), b.name.clone()));
    }
    /// Total segment size: the end of the furthest declared extent or allocation.
    pub fn total(&self) -> i64 {
        self.bufs.iter().map(|s| s.declared().1).max().unwrap_or(0).max(self.cursor)
    }
    /// Emits `lds(total)`, the declarations and the justified aliases (see the module doc). Returns the alias pairs.
    pub fn commit(self, b: &mut FnBuilder) -> Vec<(String, String)> {
        b.lds(self.total() as u32);
        for s in &self.bufs {
            if s.private {
                b.lds_private(&s.name, s.base, s.bytes, s.slots);
            } else if s.slots == 1 {
                b.lds_tile(&s.name, s.base, s.bytes);
            } else {
                b.lds_ring(&s.name, s.base, s.slot_stride, s.slots);
            }
        }
        let declared_overlay = |a: &str, c: &str| self.overlays.iter().any(|(x, y)| (x == a && y == c) || (x == c && y == a));
        let mut aliases: Vec<(String, String)> = Vec::new();
        for i in 0..self.bufs.len() {
            for k in i + 1..self.bufs.len() {
                let (x, y) = (&self.bufs[i], &self.bufs[k]);
                let (dx, dy) = (x.declared(), y.declared());
                let decl_overlap = dx.0 < dy.1 && dy.0 < dx.1;
                let real_overlap = x.footprint().iter().any(|p| y.footprint().iter().any(|q| p.0 < q.1 && q.0 < p.1));
                let ov = declared_overlay(&x.name, &y.name);
                if real_overlap && !ov {
                    panic!("LdsPlan: `{}` {:?} and `{}` {:?} really overlap but no time multiplexing was declared (overlays)", x.name, x.footprint(), y.name, y.footprint());
                }
                if decl_overlap || ov {
                    b.lds_alias(&x.name, &y.name);
                    aliases.push((x.name.clone(), y.name.clone()));
                }
            }
        }
        aliases
    }
}
