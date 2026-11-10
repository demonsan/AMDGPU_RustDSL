//! L2 template T9 (docs/l2-templates.md §3.9), the block inclusive scan: Hillis–Steele in LDS over `n` threads, ping-pong between two declared tiles.
//!
//! Used by `sampling` (f32 sums over the 1024 thread sums), `moe_sorting` and the multi-block `moe_scan` (u32 padded counts over the experts). The caller stores its value into the first tile (version 0),
//! completes the store and barriers; `steps` then emits the log2(n) steps and returns where the inclusive result is. Every access is under the tile markers: step `j` reads tile `cur` at version `j` and
//! writes the other tile at version `j + 1` (a different version of a different tile: no race by construction), then completes the store and barriers. The neighbour address is a `max(tid - s, 0)` (provably
//! inside the tile; the lanes it does not apply to are masked by the compare). The emitted sequence is the one of the three hand-written copies it replaces (the golden md5 manifest is the gate).

use amdgpu_tile::{ev, Shared, SlotIdx};
use amdgpu_dsl::{CmpPred, CmpTy, DType, FnBuilder, MemTy, Op, Opnd, ValueId};

/// The element type of the scan (the combine is an add).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScanElem {
    U32,
    /// f32 sums; the scan needs an f32 zero constant (`zero_f32`)
    F32,
}

/// The two ping-pong tiles of the scan (`base0` / `base1` are their byte offsets in the workgroup's LDS; each holds `n` dwords).
pub struct HsScan<'a> {
    pub s0: &'a Shared,
    pub s1: &'a Shared,
    pub base0: i32,
    pub base1: i32,
    /// number of threads (a power of two): the scan runs `log2(n)` steps
    pub n: i32,
    pub elem: ScanElem,
}

/// Where the inclusive result of a scan sits: tile, byte offset and version.
pub struct ScanOut<'a> {
    pub tile: &'a Shared,
    pub base: i32,
    pub ver: i32,
}

impl<'a> HsScan<'a> {
    /// Emit the scan steps. `tid` is the workitem id, `tid4 = tid << 2`; `zero_f32` is required for `ScanElem::F32`.
    pub fn steps(&self, b: &mut FnBuilder, tid: ValueId, tid4: ValueId, zero_f32: Option<ValueId>) -> ScanOut<'a> {
        let (b0, b1) = (self.base0, self.base1);
        let mut cur = b0;
        let mut s = 1i32;
        let mut ver = 0i32; // the version of the tile `cur` (ping-pong: every step writes the other tile one version up)
        while s < self.n {
            let nxt = if cur == b0 { b1 } else { b0 };
            let (cur_t, nxt_t) = if cur == b0 { (self.s0, self.s1) } else { (self.s1, self.s0) };
            let a_own = b.bind_i32(ev(tid4) + cur);
            let own = cur_t.at(&SlotIdx::new(0, ver)).read(b, |b| b.load_lds(a_own, MemTy::B32));
            let prev_i = b.bind_i32(ev(tid) - s);
            let ok = b.cmp(CmpPred::Ge, CmpTy::I32, prev_i, 0);
            // the neighbour below (clamped to 0 for the first s threads: masked by `ok` below; a max so that the address is provably inside the scan tile)
            let pi = b.bind_i32(ev(prev_i).max(0));
            let a_pv = b.bind_i32((ev(pi) << 2) + cur);
            let pv = cur_t.at(&SlotIdx::new(0, ver)).read(b, |b| b.load_lds(a_pv, MemTy::B32));
            b.wait_lds(0);
            let sum = match self.elem {
                ScanElem::U32 => {
                    let add = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(ok), a: Opnd::Val(pv), b: Opnd::I32(0) });
                    b.add_i32(own, add)
                }
                ScanElem::F32 => {
                    let ownf = b.bitcast_f32(own);
                    let pvf = b.bitcast_f32(pv);
                    let zero = zero_f32.expect("ScanElem::F32 needs the f32 zero constant");
                    let addf = b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(ok), a: Opnd::Val(pvf), b: Opnd::Val(zero) });
                    let sum = b.add_f32(ownf, addf);
                    b.bitcast_i32(sum)
                }
            };
            let a_nx = b.bind_i32(ev(tid4) + nxt);
            nxt_t.at(&SlotIdx::new(0, ver + 1)).write(b, |b| b.store_lds(a_nx, sum, MemTy::B32, false));
            b.wait_lds(0);
            b.barrier();
            cur = nxt;
            ver += 1;
            s *= 2;
        }
        let tile = if cur == b0 { self.s0 } else { self.s1 };
        ScanOut { tile, base: cur, ver }
    }
}

impl ScanOut<'_> {
    /// This thread's inclusive value and the total (the last thread's value, word `last`): two reads; the caller completes them (`wait_lds(0)`).
    pub fn incl_total(&self, b: &mut FnBuilder, tid4: ValueId, last: i32) -> (ValueId, ValueId) {
        let a_incl = b.bind_i32(ev(tid4) + self.base);
        let incl = self.tile.at(&SlotIdx::new(0, self.ver)).read(b, |b| b.load_lds(a_incl, MemTy::B32));
        let a_tot = b.c_i32(self.base + 4 * last);
        let total = self.tile.at(&SlotIdx::new(0, self.ver)).read(b, |b| b.load_lds(a_tot, MemTy::B32));
        (incl, total)
    }
}
