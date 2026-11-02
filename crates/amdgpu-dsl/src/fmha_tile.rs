//! The shipped FMHA forward kernels (`_vs128`, `_kvsplit`, `_kvsplit_combine8h`, `_t128`) composed from the tile library (`crate::tile`,
//! docs/dsl-tile-abstractions.md). They replace the hand-written `kernels.rs` builders (`fmha_t_build` + `emit_fmha_vs128_role` /
//! `emit_fmha_t128_role`, ~3,300 lines), default recipe only. The FMHA ABI, block setup and K/V ring are library components (`tile::fmha`);
//! this file holds the attention algorithm (`AttnBody`: QK -> online softmax -> PV per 128-key tile, as a `PipeBody`) and one short builder per kernel.
use crate::tile::frag::*;
use crate::tile::reduce::KeyMask;
use crate::tile::softmax::{OnlineSoftmax, Recipe};
use crate::tile::fmha::{combine_kernel, FmhaBlock, FmhaIo, KvRing};
use crate::tile::*;
use crate::{CompileProfile, DType, Ext, FnBuilder, Function, Opnd, ValueId, VarId, I32};

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
    /// The output scale after the pipeline: `V descale / l`.
    pub fn out_scale(&self, b: &mut FnBuilder) -> ValueId {
        let l = self.sm.l_total(b);
        self.blk.out_scale(b, l)
    }
}

impl PipeBody for AttnBody<'_> {
    fn stage(&mut self, b: &mut FnBuilder, role: Role, key0: &I32, slot: &SlotIdx) {
        self.kd_pending = self.ring.stage(b, role, key0, slot);
    }
    fn stage_complete(&mut self, b: &mut FnBuilder, _role: Role, slot: &SlotIdx) {
        self.ring.stage_complete(b, self.kd_pending.take(), slot);
    }
    fn prologue_stores(&mut self, b: &mut FnBuilder, _role: Role) {
        self.ring.store_constants(b, self.wid, &self.blk.lane, &self.blk.q);
    }
    fn prologue_compute(&mut self, b: &mut FnBuilder, _role: Role, t0: &TileIt) {
        self.s_init = self.qk(b, &t0.cur, QkOut::Values(self.zero16), self.sm.bias_init);
    }
    /// loop state, in the hand kernels' order: [bias, O x4, S x4, row sums]
    fn loop_state(&mut self, _b: &mut FnBuilder) -> Vec<(DType, Opnd)> {
        let [bias, la]: [(DType, Opnd); 2] = self.sm.state().try_into().expect("softmax state");
        let o = (0..4).map(|_| (DType::Frag16, Opnd::Val(self.zero16)));
        let s = self.s_init.iter().map(|&s| (DType::Frag16, Opnd::Val(s)));
        std::iter::once(bias).chain(o).chain(s).chain(std::iter::once(la)).collect()
    }
    fn bind_state(&mut self, vars: &[VarId]) {
        let (bias, rest) = vars.split_first().expect("state");
        let (o, rest) = rest.split_at(4);
        let (s, la) = rest.split_at(4);
        self.sm.bind(&[*bias, la[0]]);
        self.o = Some(AccTile::from_vars(o));
        self.s = Some(AccTile::from_vars(s));
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

/// `_vs128`: 256 query rows per block (8 waves in two groups of 4, ping-pong), 128-key tiles in a 2-slot LDS ring, two code copies (LUT-corrected P
/// conversion for sequences shorter than 9 tiles).
pub fn fmha_fwd_hd128_mxfp8_dsl_vs128() -> Function {
    let mut fb = FnBuilder::new("fmha_fwd_hd128_mxfp8_dsl_vs128", 512);
    fb.compile_profile(CompileProfile::Vs128);
    let b = &mut fb;
    let io = FmhaIo::declare(b, 256, false);
    let blk = io.block(b, 8);
    let roles = Roles::split(b, &blk.lane, 4);
    let ring = KvRing::new(b, &io, &blk, &roles, 2);
    let pipe = Pipeline::new(Cadence::TwoPhaseOffset, blk.nkv, blk.kv_len, 128, Ext::ka(0x110).ceil_div(128));
    let lut = blk.lut_copy(b);
    versioned(b, lut, |b, lut| {
        roles.each(b, |b, role| {
            let wid = roles.wave_id(b, role);
            let mut body = AttnBody::new(b, &ring, &blk, wid, lut);
            pipe.run(b, role, &mut body);
            let fin = body.out_scale(b);
            io.store_out(b, &blk, &body.o.unwrap(), fin);
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
    let io = FmhaIo::declare(b, 256, true);
    let blk = io.block(b, 8);
    let roles = Roles::split(b, &blk.lane, 4);
    let ring = KvRing::new(b, &io, &blk, &roles, 2);
    let pipe = Pipeline::new(Cadence::TwoPhaseOffset, blk.nkv, blk.kv_len, 128, Ext::ka(0x110).ceil_div(128));
    let lut = blk.lut_copy(b);
    versioned(b, lut, |b, lut| {
        roles.each(b, |b, role| {
            let wid = roles.wave_id(b, role);
            let mut body = AttnBody::new(b, &ring, &blk, wid, lut);
            pipe.run(b, role, &mut body);
            let fin = body.out_scale(b);
            let l = body.sm.l_total(b);
            let lse = body.sm.lse(b, l);
            io.store_partial(b, &blk, &ring.epi.mine(wid), &body.o.unwrap(), fin, lse);
        });
    });
    fb.finish()
}

/// KV-split combine (phase 2), 8 output elements per thread: `out = sum_s 2^(lse_s - m) Ohat_s / sum_s 2^(lse_s - m)` (bf16).
pub fn fmha_fwd_hd128_mxfp8_dsl_kvsplit_combine8h() -> Function {
    let mut fb = FnBuilder::new("fmha_fwd_hd128_mxfp8_dsl_kvsplit_combine8h", 256);
    fb.compile_profile(CompileProfile::Combine);
    combine_kernel(&mut fb, 8);
    fb.finish()
}

/// `_t128`: 128 query rows per block (ONE group of 4 waves), a 3-slot LDS ring staged by all waves (K and V share the row clamp, the K scales go through
/// a VGPR), one window and one barrier per 128-key tile, two code copies as `_vs128`.
pub fn fmha_fwd_hd128_mxfp8_dsl_t128() -> Function {
    let mut fb = FnBuilder::new("fmha_fwd_hd128_mxfp8_dsl_t128", 256);
    fb.compile_profile(CompileProfile::T128);
    let b = &mut fb;
    let io = FmhaIo::declare(b, 128, false);
    let blk = io.block(b, 7);
    let roles = Roles::single(&blk.lane, 4);
    let ring = KvRing::new(b, &io, &blk, &roles, 3);
    let mut pipe = Pipeline::new(Cadence::OneWindow, blk.nkv, blk.kv_len, 128, Ext::ka(0x110).ceil_div(128));
    pipe.prebind = vec![32768, 512]; // ring slot offsets (K|V, K scales) bound at the top of every iteration, as the hand kernel's `ring()`
    let lut = blk.lut_copy(b);
    versioned(b, lut, |b, lut| {
        let mut body = AttnBody::new(b, &ring, &blk, blk.lane.wave, lut);
        pipe.run(b, Role::All, &mut body);
        let fin = body.out_scale(b);
        io.store_out(b, &blk, &body.o.unwrap(), fin);
    });
    fb.finish()
}