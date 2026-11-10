//! L2 template T9 (docs/l2-templates.md §3.9), the shared part of the radix select: the histogram clear and the descending cumulative scan that finds the bin where a target rank is reached.
//!
//! Used by `sampling` (argmax / top-k / top-p select over bf16 logits) and `topk::perrow` (four-pass radix select of the K-th largest key). What stays in the kernels (per-shape orchestration): the key
//! transform, the element source and the histogram pass itself (loads, bin choice, the sink bin of the masked lanes), the pass driver (two passes with an f32-weight option vs four 8-bit digits), and the
//! emit pass. The LDS tiles are declared by the kernel (`hist`, `wt`, `rb`); every access here is under the tile markers with the compile-time version counters of [`SelectVer`]:
//!   * `hv`: the histogram version, bumped by every `clear` (the atomics of the following histogram pass use [`SelectVer::hist`]);
//!   * `rv`: the version of the scan result words / wave totals, bumped by every `scan_find`.
//!
//! The emitted instruction sequence is exactly the sequence of the two hand-written copies it replaces (the golden md5 manifest is the gate); the two differences between the copies are parameters:
//! `clamp_bin` (the bin index of the idle threads is clamped to 0 so the read stays inside the declared tile: `sampling`) and `norm_hit` (an extra, redundant 0/1 normalisation of the hit flag that
//! `sampling` carries: kept so that the move is md5-neutral; flip it with a re-baselined manifest to drop the redundant select).

use crate::{ev, Shared, SlotIdx};
use amdgpu_dsl::{CmpPred, CmpTy, Dpp, DppAluOp, DppCtrl, DType, FnBuilder, MemTy, Op, Opnd, ValueId};
use std::cell::Cell;

/// The compile-time version counters of one select (shared by `clear` and `scan_find`).
#[derive(Debug, Default)]
pub struct SelectVer {
    hv: Cell<i32>,
    rv: Cell<i32>,
}

impl SelectVer {
    pub fn new() -> SelectVer {
        SelectVer::default()
    }
    /// The current version of the histogram tile (use it for the atomics of a histogram pass).
    pub fn hist(&self) -> i32 {
        self.hv.get()
    }
}

/// The per-thread values the scan needs (all computed once by the kernel).
#[derive(Clone, Copy, Debug)]
pub struct ScanThreads {
    /// workitem id
    pub tid: ValueId,
    /// the thread id clamped to the 256 bins' owners: bin `255 - tcl` is owned by thread `tcl`
    pub tcl: ValueId,
    /// `tcl << 2`: this thread's histogram word offset
    pub tcl4: ValueId,
    /// wave id
    pub wave: ValueId,
    /// the constant 0 (i32)
    pub zero: ValueId,
}

/// A descending 256-bin cumulative scan over the declared tiles.
pub struct DescScan<'a> {
    pub hist: &'a Shared,
    pub wt: &'a Shared,
    pub rb: &'a Shared,
    /// byte offsets of the histogram, of the 16 wave-total dwords and of the 3 result words (bin, before, sink) in the workgroup's LDS
    pub hist_base: i32,
    pub wt_base: i32,
    pub res_base: i32,
    /// threads >= 256 own no bin: clamp their bin index to 0 (the value is masked) so the read provably stays inside the histogram tile
    pub clamp_bin: bool,
    /// keep the redundant 0/1 normalisation of the hit flag (see the module comment)
    pub norm_hit: bool,
}

fn sel_f32(b: &mut FnBuilder, cond: ValueId, x: ValueId, y: ValueId) -> ValueId {
    b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(cond), a: Opnd::Val(x), b: Opnd::Val(y) })
}

impl DescScan<'_> {
    /// Zero the bins (thread `t` zeroes word `t`), then barrier; bumps the histogram version.
    pub fn clear(&self, b: &mut FnBuilder, th: &ScanThreads, ver: &SelectVer) {
        ver.hv.set(ver.hv.get() + 1);
        let a = b.bind_i32(ev(th.tcl4) + self.hist_base);
        self.hist.at(&SlotIdx::new(0, ver.hv.get())).write(b, |b| b.store_lds(a, th.zero, MemTy::B32, false));
        b.wait_lds(0);
        b.barrier();
    }

    /// Descending cumulative scan over the 256 bins; returns `(bin, cumulative before the bin)` of the bin where the cumulative reaches `tgt` (bin 0 / 0 when it never does); `float`: the bins hold f32 sums.
    /// Thread `t < 256` owns bin `255 - t`: a DPP inclusive scan inside each wave (row_shr 1 / 2 / 4 / 8, then row_bcast 15 / 31), the wave totals through LDS, one barrier.
    pub fn scan_find(&self, b: &mut FnBuilder, th: &ScanThreads, ver: &SelectVer, tgt: ValueId, float: bool) -> (ValueId, ValueId) {
        let (tid, wave, zero_i) = (th.tid, th.wave, th.zero);
        let (hist, wt, resl) = (self.hist_base, self.wt_base, self.res_base);
        ver.rv.set(ver.rv.get() + 1);
        let rs = SlotIdx::new(0, ver.rv.get());
        let rb = if self.clamp_bin { b.bind_i32((255 - ev(th.tcl)).max(0)) } else { b.bind_i32(255 - ev(th.tcl)) };
        let a_h = b.bind_i32((ev(rb) << 2) + hist);
        let v0 = self.hist.at(&SlotIdx::new(0, ver.hv.get())).read(b, |b| b.load_lds(a_h, MemTy::B32));
        b.wait_lds(0);
        let in256 = b.cmp(CmpPred::Lt, CmpTy::I32, tid, 256);
        let v0z = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(in256), a: Opnd::Val(v0), b: Opnd::I32(0) });
        let vty = if float { b.bitcast_f32(v0z) } else { v0z };
        let aop = if float { DppAluOp::AddF32 } else { DppAluOp::AddU32 };
        let mut v = vty;
        for c in [DppCtrl::RowShr(1), DppCtrl::RowShr(2), DppCtrl::RowShr(4), DppCtrl::RowShr(8)] {
            v = b.dpp_alu(aop, Dpp { ctrl: c, row_mask: 0xf, bank_mask: 0xf, bound_zero: true }, v, v, v);
        }
        v = b.dpp_alu(aop, Dpp { ctrl: DppCtrl::RowBcast15, row_mask: 0xa, bank_mask: 0xf, bound_zero: false }, v, v, v);
        v = b.dpp_alu(aop, Dpp { ctrl: DppCtrl::RowBcast31, row_mask: 0xc, bank_mask: 0xf, bound_zero: false }, v, v, v);
        // the wave total (lane 63) -> LDS; defaults of the result slots (thread 0)
        let tot = b.read_lane(v, 63);
        let a_wt = b.bind_i32((ev(wave) << 2) + wt);
        self.wt.at(&rs).write(b, |b| b.store_lds(a_wt, tot, MemTy::B32, false));
        let is0 = b.cmp(CmpPred::Eq, CmpTy::I32, tid, 0);
        let dflt_a = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(is0), a: Opnd::I32(resl), b: Opnd::I32(resl + 8) });
        self.rb.at(&rs).write(b, |b| b.store_lds(dflt_a, zero_i, MemTy::B32, false));
        let dflt_b = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(is0), a: Opnd::I32(resl + 4), b: Opnd::I32(resl + 8) });
        self.rb.at(&rs).write(b, |b| b.store_lds(dflt_b, zero_i, MemTy::B32, false));
        b.wait_lds(0);
        b.barrier();
        let mut tots = Vec::new();
        for w in 0..3i32 {
            let a = b.c_i32(wt + 4 * w);
            tots.push(self.wt.at(&rs).read(b, |b| b.load_lds(a, MemTy::B32)));
        }
        b.wait_lds(0);
        // this wave's offset = the sum of the totals of the lower waves (waves >= 4 are idle; their value is unused)
        let incl = if float {
            let mut pre = b.c_f32(0.0);
            for (w, t) in tots.iter().enumerate() {
                let tf = b.bitcast_f32(*t);
                let sum = b.add_f32(pre, tf);
                let is_hi = b.cmp(CmpPred::Gt, CmpTy::I32, wave, w as i32);
                pre = sel_f32(b, is_hi, sum, pre);
            }
            b.add_f32(v, pre)
        } else {
            let mut pre = b.c_i32(0);
            for (w, t) in tots.iter().enumerate() {
                let sum = b.add_i32(pre, *t);
                let is_hi = b.cmp(CmpPred::Gt, CmpTy::I32, wave, w as i32);
                pre = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(is_hi), a: Opnd::Val(sum), b: Opnd::Val(pre) });
            }
            b.add_i32(v, pre)
        };
        // exclusive = inclusive - own value; the owner is the thread with exclusive < tgt <= inclusive
        let (g12, hit, before) = if float {
            let vf = b.bitcast_f32(v0z);
            let exf = b.sub_f32(incl, vf);
            let g1 = b.cmp(CmpPred::Lt, CmpTy::F32, exf, tgt);
            let g2 = b.cmp(CmpPred::Ge, CmpTy::F32, incl, tgt);
            let g12 = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(g1), a: Opnd::I32(1), b: Opnd::I32(0) });
            let g12 = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(g2), a: Opnd::Val(g12), b: Opnd::I32(0) });
            let hit = self.norm_hit.then(|| b.cmp(CmpPred::Ne, CmpTy::I32, g12, 0));
            (g12, hit, b.bitcast_i32(exf))
        } else {
            let ex = b.bind_i32(ev(incl) - ev(v0z));
            let g1 = b.cmp(CmpPred::Lt, CmpTy::I32, ex, tgt);
            let g2 = b.cmp(CmpPred::Ge, CmpTy::I32, incl, tgt);
            let g12 = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(g1), a: Opnd::I32(1), b: Opnd::I32(0) });
            let g12 = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(g2), a: Opnd::Val(g12), b: Opnd::I32(0) });
            let hit = self.norm_hit.then(|| b.cmp(CmpPred::Ne, CmpTy::I32, g12, 0));
            (g12, hit, ex)
        };
        // (`norm_hit`: the redundant re-normalisation of the 0/1 flag: cmp + select)
        let go = match hit {
            Some(hit) => b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(hit), a: Opnd::I32(1), b: Opnd::I32(0) }),
            None => g12,
        };
        let go = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(in256), a: Opnd::Val(go), b: Opnd::I32(0) });
        let wr = b.cmp(CmpPred::Ne, CmpTy::I32, go, 0);
        let addr_bin = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(wr), a: Opnd::I32(resl), b: Opnd::I32(resl + 8) });
        self.rb.at(&rs).write(b, |b| b.store_lds(addr_bin, rb, MemTy::B32, false));
        let addr_bef = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(wr), a: Opnd::I32(resl + 4), b: Opnd::I32(resl + 8) });
        self.rb.at(&rs).write(b, |b| b.store_lds(addr_bef, before, MemTy::B32, false));
        b.wait_lds(0);
        b.barrier();
        let r_bin_a = b.c_i32(resl);
        let r_bef_a = b.c_i32(resl + 4);
        let bin = self.rb.at(&rs).read(b, |b| b.load_lds(r_bin_a, MemTy::B32));
        let bef = self.rb.at(&rs).read(b, |b| b.load_lds(r_bef_a, MemTy::B32));
        b.wait_lds(0);
        b.barrier();
        (bin, bef)
    }
}
