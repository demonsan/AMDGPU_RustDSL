//! The shipped FMHA forward kernels (`_vs128`, `_kvsplit`, `_kvsplit_combine8h`, `_t128`) composed from the tile library (`crate::tile`,
//! docs/dsl-tile-abstractions.md). They replace the hand-written `kernels.rs` builders (`fmha_t_build` + `emit_fmha_vs128_role` /
//! `emit_fmha_t128_role`, ~3,300 lines), default recipe only. The FMHA ABI, block setup and K/V ring are library components (`tile::fmha`);
//! this file holds the attention algorithm (`AttnBody`: QK -> online softmax -> PV per 128-key tile, as a `PipeBody`) and one short builder per kernel.
use crate::tile::epilogue::{combine_splits, kvsplit_partial_f16, store_bf16};
use crate::tile::frag::*;
use crate::tile::reduce::KeyMask;
use crate::tile::softmax::{OnlineSoftmax, Recipe};
use crate::tile::fmha::{FmhaBlock, FmhaIo, KdCopy, KvRing};
use crate::tile::*;
use crate::{CompileProfile, DType, Ext, FnBuilder, Function, Op, Opnd, ValueId, VarId, I32};

pub struct AttnBody<'a> {
    ring: &'a KvRing,
    blk: &'a FmhaBlock,
    wid: ValueId,
    pub sm: OnlineSoftmax,
    zero16: ValueId,
    c127: ValueId,
    c123: ValueId,
    s_init: Vec<ValueId>,
    pub s: Option<AccTile<C32x32F32, 4>>,
    pub o: Option<AccTile<C32x32F32, 4>>,
    /// P words of the tile + the new bias word, from the V phase to the M phase
    p: Vec<ValueId>,
    /// the K-scale word in flight (VGPR staging)
    kd_pending: Option<ValueId>,
}

impl<'a> AttnBody<'a> {
    pub fn new(b: &mut FnBuilder, ring: &'a KvRing, blk: &'a FmhaBlock, wid: ValueId, lut: bool) -> AttnBody<'a> {
        let sm = OnlineSoftmax::new(b, &blk.lane, Recipe::vsx(lut));
        let zero16 = b.frag_zero(DType::Frag16);
        let (c127, c123) = (b.c_i32(127), b.c_i32(123));
        AttnBody { ring, blk, wid, sm, zero16, c127, c123, s_init: Vec::new(), s: None, o: None, p: Vec::new(), kd_pending: None }
    }
    /// QK of the tile in `slot`: either in place into S (each S block first set to ones x bias) or (prologue) as fresh values.
    fn qk(&self, b: &mut FnBuilder, slot: &SlotIdx, out: QkOut, bias: ValueId) -> Vec<ValueId> {
        let (r, blk) = (self.ring, self.blk);
        let k = r.kr.at(b, &r.ks.at(slot));
        let kd = r.kdr.at(b, &r.kd.at(slot));
        qk_tile(b, out, Some((self.sm.ones_f16, bias)), &k, &kd, &blk.q.frags, blk.sq, &blk.sel)
    }
    /// PV of the tile in `slot` with the P words `w` (16), the row sums first (4 smfmacs).
    fn pv(&self, b: &mut FnBuilder, slot: &SlotIdx, w: &[ValueId]) {
        let r = self.ring;
        let v = r.vr.at(b, &r.vs.at(slot));
        let p = [Frag::<B32x32x64Fp8>::from_words(b, &w[0..8]), Frag::<B32x32x64Fp8>::from_words(b, &w[8..16])];
        self.sm.row_sum(b, p[0].v, p[1].v);
        pv_tile(b, &self.o.unwrap(), &v, &p, self.c127, self.c123);
    }
}

impl PipeBody for AttnBody<'_> {
    fn stage(&mut self, b: &mut FnBuilder, role: Role, key0: &I32, slot: &SlotIdx) {
        let r = self.ring;
        match (role, &r.kdc) {
            (Role::A, KdCopy::Dma(kd)) => {
                kd.issue(b, key0, &r.kd.at(slot));
                r.kcp.issue(b, key0, &r.ks.at(slot));
            }
            (Role::B, _) => r.vcp.issue(b, key0, &r.vs.at(slot)),
            (_, KdCopy::Vgpr(kd)) => {
                self.kd_pending = Some(kd.issue(b, key0));
                issue_rows_shared(b, key0, &[(&r.kcp, &r.ks.at(slot)), (&r.vcp, &r.vs.at(slot))]);
            }
            _ => unreachable!("role / K-scale staging combination"),
        }
    }
    fn stage_complete(&mut self, b: &mut FnBuilder, _role: Role, slot: &SlotIdx) {
        if let (KdCopy::Vgpr(kd), Some(v)) = (&self.ring.kdc, self.kd_pending.take()) {
            kd.complete(b, v, &self.ring.kd.at(slot));
        }
    }
    fn prologue_stores(&mut self, b: &mut FnBuilder, _role: Role) {
        // the shipped kernel's constant LDS data: a block of E4M3 ones and this wave's Q fragments (32 B per lane each)
        let (r, lane) = (self.ring, &self.blk.lane);
        let s0 = SlotIdx::new(0, 0);
        let ones_w = b.c_i32(0x38383838);
        let oa = b.bind_i32((ev(self.wid) * 64 + lane.lane()) * 4 + r.ones.base as i32);
        r.ones.at(&s0).write(b, |b| b.store_lds_b32(oa, ones_w));
        for (sh, w) in [(&r.q0, &self.blk.q.words[0]), (&r.q1, &self.blk.q.words[1])] {
            let a0 = b.bind_i32((ev(self.wid) * 2048 + sh.base as i32) + (lane.lane() << 4));
            let a1 = b.bind_i32(ev(a0) + 1024);
            let (g0, g1) = (b.frag4(w[0], w[1], w[2], w[3]), b.frag4(w[4], w[5], w[6], w[7]));
            sh.at(&s0).write(b, |b| {
                b.store_lds_b128(a0, g0);
                b.store_lds_b128(a1, g1);
            });
        }
    }
    fn prologue_compute(&mut self, b: &mut FnBuilder, _role: Role, t0: &TileIt) {
        self.s_init = self.qk(b, &t0.cur, QkOut::Values(self.zero16), self.sm.bias_init);
    }
    fn loop_state(&mut self, _b: &mut FnBuilder) -> Vec<(DType, Opnd)> {
        let st = self.sm.state();
        let mut v = vec![st[0]];
        v.extend((0..4).map(|_| (DType::Frag16, Opnd::Val(self.zero16))));
        v.extend(self.s_init.iter().map(|&s| (DType::Frag16, Opnd::Val(s))));
        v.push(st[1]);
        v
    }
    fn bind_state(&mut self, vars: &[VarId]) {
        self.sm.bind(&[vars[0], vars[9]]);
        self.o = Some(AccTile::from_vars(&vars[1..5]));
        self.s = Some(AccTile::from_vars(&vars[5..9]));
    }
    /// softmax of tile t: one max / reference decision over the tile, P = fp8(2^S')
    fn v_phase(&mut self, b: &mut FnBuilder, _role: Role, t: &TileIt) {
        let (s, o) = (self.s.unwrap(), self.o.unwrap());
        let mask = t.masked.then(|| KeyMask::new(b, t, self.blk.g4, self.blk.kv_len));
        let nb = self.sm.decide(b, &s, &o, mask.as_ref());
        self.p = self.sm.convert(b, &s, mask.as_ref(), 0, 4);
        self.p.push(nb);
    }
    /// PV(t) with the row sums, then QK(t+1) (its K tile was staged during the V phase)
    fn m_phase(&mut self, b: &mut FnBuilder, _role: Role, t: &TileIt) {
        let mut w = std::mem::take(&mut self.p);
        let bias = w.pop().expect("bias word");
        self.pv(b, &t.cur, &w);
        if t.staged {
            self.qk(b, &t.next, QkOut::Vars(&self.s.unwrap()), bias);
        }
    }
    /// `_t128`: ONE window per tile - the decision of tile t, then per 32-key block its P conversion and QK(t+1) into the freed S block, the bias variable
    /// written last (an early write costs the scheduler its fit), then PV(t) with the row sums
    fn window(&mut self, b: &mut FnBuilder, _role: Role, t: &TileIt) {
        let (r, blk) = (self.ring, self.blk);
        let (s, o) = (self.s.unwrap(), self.o.unwrap());
        // slot offsets of this window bound at its top (as the hand kernel's `ring()`), not at their first use after the decision
        let _ = r.ks.at(&t.cur).off_v(b);
        if t.staged {
            let _ = (r.ks.at(&t.next).off_v(b), r.kd.at(&t.next).off_v(b));
        }
        let mask = t.masked.then(|| KeyMask::new(b, t, blk.g4, blk.kv_len));
        let nb = self.sm.decide_deferred(b, &s, &o, mask.as_ref());
        let next = t.staged.then(|| (r.kr.at(b, &r.ks.at(&t.next)), r.kdr.at(b, &r.kd.at(&t.next))));
        let mut w = Vec::with_capacity(16);
        for i in 0..4usize {
            w.extend(self.sm.convert(b, &s, mask.as_ref(), i, i + 1));
            if let Some((k, kd)) = &next {
                qk_block(b, &QkOut::Vars(&s), Some((self.sm.ones_f16, nb)), k, kd, &blk.q.frags, blk.sq, &blk.sel, i);
            }
        }
        self.sm.commit_bias(b, nb);
        self.pv(b, &t.cur, &w);
    }
}

// ------------------------------------------------------------------------------------------------------------------------------------------------------
// Kernels
// ------------------------------------------------------------------------------------------------------------------------------------------------------

/// `_vs128` on the tile library: 256 query rows per block (8 waves in two groups of 4, ping-pong), 128-key tiles in a 2-slot LDS ring,
/// two code copies (LUT-corrected P conversion for sequences shorter than 9 tiles).
pub fn fmha_fwd_hd128_mxfp8_dsl_vs128() -> Function {
    let mut fb = FnBuilder::new("fmha_fwd_hd128_mxfp8_dsl_vs128", 512);
    fb.compile_profile(CompileProfile::Vs128);
    let b = &mut fb;
    let mut ka = KArgs::new();
    let io = FmhaIo::declare(b, &mut ka, 256, false);
    ka.arg(b, "abi_tail", DType::I32, 0x280, 16);
    let blk = io.block(b, 8);
    let roles = Roles::split(b, &blk.lane, 4);
    let ring = KvRing::new(b, &io, &blk, &roles, 2);
    let pipe = Pipeline::new(Cadence::TwoPhaseOffset, blk.nkv, blk.kv_len, 128, Ext::ka(0x110).ceil_div(128));
    let short = b.bind_cond(ev(blk.nkv).lt(9));
    versioned(b, short, |b, lut| {
        roles.each(b, |b, role| {
            let wid = roles.wave_id(b, role);
            let mut body = AttnBody::new(b, &ring, &blk, wid, lut);
            pipe.run(b, role, &mut body);
            // O * (V descale / l) -> bf16 (rows past S recompute and store the last row)
            let l = body.sm.l_total(b);
            let inv_l = b.rcp_f32(l);
            let fin = b.mul_f32(inv_l, blk.vd);
            let (row_ptr, live) = io.out_row(b, &blk);
            store_bf16(b, &body.o.unwrap(), fin, row_ptr, live);
        });
    });
    fb.finish()
}

/// `_kvsplit` (phase 1 of the sequence-parallel attention, docs/kvsplit-design.md): the `_vs128` block over one key range per block; writes the fp16
/// normalised partial `Ohat` and `lse` instead of the output. Grid `(ceil(S/256), H, B * splits)`; kernarg = the base ABI + 0x270..0x287.
pub fn fmha_fwd_hd128_mxfp8_dsl_kvsplit() -> Function {
    let mut fb = FnBuilder::new("fmha_fwd_hd128_mxfp8_dsl_kvsplit", 512);
    fb.compile_profile(CompileProfile::KvSplit);
    let b = &mut fb;
    let mut ka = KArgs::new();
    let io = FmhaIo::declare(b, &mut ka, 256, true);
    let heads = ka.i32(b, "s_q_head_num", 0x140);
    let kv = io.kvs.as_ref().expect("kv-split args");
    let blk = io.block(b, 8);
    let roles = Roles::split(b, &blk.lane, 4);
    let ring = KvRing::new(b, &io, &blk, &roles, 2);
    let mut pipe = Pipeline::new(Cadence::TwoPhaseOffset, blk.nkv, blk.kv_len, 128, Ext::ka(0x110).ceil_div(128));
    pipe.ring_var_last = true;
    // linear partial row of a query row: split * rows_total + (b * S + row) * H + h
    let part_row0 = b.bind_i32(ev(blk.split.expect("split")) * ev(kv.rows_total));
    let prow = |row: I32| ev(part_row0) + (((ev(blk.bidx) * io.s_len.clone()) + row) * ev(heads) + ev(blk.hidx));
    // the recipe copy is chosen by the tile count of the WHOLE sequence: identical numerics to `_vs128` per split
    let short = b.bind_cond(ev(blk.ntiles_total).lt(9));
    versioned(b, short, |b, lut| {
        roles.each(b, |b, role| {
            let wid = roles.wave_id(b, role);
            let mut body = AttnBody::new(b, &ring, &blk, wid, lut);
            pipe.run(b, role, &mut body);
            // every wave: the epilogue scratch reuses the Q staging of the neighbouring waves (the drain's last barrier is role-dependent)
            b.barrier();
            let l = body.sm.l_total(b);
            let inv_l = b.rcp_f32(l);
            let fin = b.mul_f32(inv_l, blk.vd);
            let lse = body.sm.lse(b, l);
            let lse_ptr = {
                let off = b.bind_i32(prow(blk.row_c.clone()) << 2);
                b.ptr_add(kv.lse, off)
            };
            let live = b.bind_cond(blk.row0.lt(io.s_len.clone()));
            let lane = blk.lane;
            kvsplit_partial_f16(b, &body.o.unwrap(), fin, &lane, &ring.epi.mine(wid), lse_ptr, lse, live, &mut |b, i| {
                let r = b.let_i32((&blk.row0 + 8 * i as i32) + lane.lane().lshr(3));
                let rg = r.lt(io.s_len.clone()).select_i32(r.clone(), blk.s_last.clone());
                let off = b.bind_i32((prow(rg) << 8) + ((lane.lane() & 7) << 4));
                b.ptr_add(kv.ohat, off)
            });
        });
    });
    fb.finish()
}

/// KV-split combine (phase 2), 8 output elements per thread: `out = sum_s 2^(lse_s - m) Ohat_s / sum_s 2^(lse_s - m)` (bf16).
/// Kernarg: 0x00 Ohat, 0x08 lse, 0x10 out, 0x18 rows_total, 0x1C D (128), 0x20 splits; grid `ceil(rows_total * 16 / 256)` x 256 threads.
pub fn fmha_fwd_hd128_mxfp8_dsl_kvsplit_combine8h() -> Function {
    let mut fb = FnBuilder::new("fmha_fwd_hd128_mxfp8_dsl_kvsplit_combine8h", 256);
    fb.compile_profile(CompileProfile::Combine);
    let b = &mut fb;
    let mut ka = KArgs::new();
    let ohat = ka.ptr(b, "ptr_ohat", 0x00);
    let lse = ka.ptr(b, "ptr_lse", 0x08);
    let out = ka.ptr(b, "ptr_out", 0x10);
    let rows = ka.dim(b, "rows_total", 0x18, 1, i64::MAX).val().unwrap();
    ka.dim(b, "d", 0x1C, 128, 128);
    let splits = ka.dim(b, "splits", 0x20, 1, i64::MAX).val().unwrap();
    b.arg_tensor("ptr_ohat", Ext::ka(0x20) * Ext::ka(0x18) * Ext::c(128 * 4));
    b.arg_tensor("ptr_lse", Ext::ka(0x20) * Ext::ka(0x18) * Ext::c(4));
    b.arg_tensor("ptr_out", Ext::ka(0x18) * Ext::c(128 * 2));
    // thread -> 8-element group of the output (clamped: the tail threads redo the last group), its row
    let tid = b.let_(Op::WorkitemId);
    let bx = b.let_(Op::WorkgroupIdX);
    let groups = b.let_i32(rows.clone() << 4);
    let t_raw = b.let_i32((ev(bx) << 8) + ev(tid));
    let t = b.bind_i32(t_raw.lt(groups.clone()).select_i32(t_raw.clone(), groups - 1));
    let row = b.bind_i32(ev(t).lshr(4));
    b.assume_trusted(row, Ext::c(0), Ext::ka(0x18) - Ext::c(1), "row = clamped thread index >> k <= rows - 1: the engine keeps the rational upper bound of a shift");
    let (rows_v, splits_v) = (b.bind_i32(rows), b.bind_i32(splits));
    combine_splits(b, ohat, lse, out, rows_v, splits_v, t, row, 8, Ext::ka(0x20) - Ext::c(1));
    fb.finish()
}
/// `_t128`: 128 query rows per block (ONE group of 4 waves), a 3-slot LDS ring staged by all waves (K and V share the row clamp, the K scales go through
/// a VGPR), one window and one barrier per 128-key tile, two code copies as `_vs128`.
pub fn fmha_fwd_hd128_mxfp8_dsl_t128() -> Function {
    let mut fb = FnBuilder::new("fmha_fwd_hd128_mxfp8_dsl_t128", 256);
    fb.compile_profile(CompileProfile::T128);
    let b = &mut fb;
    let mut ka = KArgs::new();
    let io = FmhaIo::declare(b, &mut ka, 128, false);
    ka.arg(b, "abi_tail", DType::I32, 0x280, 16);
    let blk = io.block(b, 7);
    let roles = Roles::single(&blk.lane, 4);
    let ring = KvRing::new(b, &io, &blk, &roles, 3);
    let pipe = Pipeline::new(Cadence::OneWindow, blk.nkv, blk.kv_len, 128, Ext::ka(0x110).ceil_div(128));
    let short = b.bind_cond(ev(blk.nkv).lt(9));
    versioned(b, short, |b, lut| {
        let mut body = AttnBody::new(b, &ring, &blk, blk.lane.wave, lut);
        pipe.run(b, Role::All, &mut body);
        let l = body.sm.l_total(b);
        let inv_l = b.rcp_f32(l);
        let fin = b.mul_f32(inv_l, blk.vd);
        let (row_ptr, live) = io.out_row(b, &blk);
        store_bf16(b, &body.o.unwrap(), fin, row_ptr, live);
    });
    fb.finish()
}