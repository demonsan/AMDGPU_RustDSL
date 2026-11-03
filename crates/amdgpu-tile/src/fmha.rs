//! The four shipped FMHA forward kernels on the layout layer (P3 of docs/dsl-layout-refactor-design.md): `fmha_fwd_hd128_mxfp8_v3_{vs128, kvsplit,
//! kvsplit_combine8h, t128}`. Same algorithm and protocol as `amdgpu_dsl::fmha_tile` (the dsl set), but every LDS address of the K/V ring is DERIVED:
//! * the K fragment reads = `make_tiled_copy_a(ds_read_b128 x2, mma)` over the swizzled [key, d] view (placements `4096 s + 64 c`);
//! * the transposed V reads = the `ds_read_b64_tr_b8 x4` source layout over the same view;
//! * the K / V staging = `partition_dma` (destination-side derivation: the swizzle on the source, M0 at the top of the wave slice);
//! * QK / PV = `gemm` of the scaled 32x32x64 atom.
//! The FMHA ABI, block setup, K-scale staging/reader, softmax, epilogues and the pipeline cadence are the (layout-free) components of `amdgpu_dsl::tile`.

use crate::copy::{make_tiled_copy, make_tiled_copy_a, LdsPart, LdsSlotPart};
use crate::dma::{issue_rows_shared, partition_dma, DmaPart};
use crate::mma::{gemm, make_tiled_mma, TiledMma};
use crate::Ctx;
use amdgpu_dsl::tile::fmha::{combine_kernel, FmhaBlock, FmhaIo, KdCopy};
use amdgpu_dsl::tile::frag::{AccTile, Frag, QkOut, RowScaleReader, B32x32x64Fp8, C32x32F32};
use amdgpu_dsl::tile::reduce::KeyMask;
use amdgpu_dsl::tile::softmax::{OnlineSoftmax, Recipe};
use amdgpu_dsl::tile::*;
use amdgpu_dsl::{CompileProfile, DType, Ext, FnBuilder, Function, Opnd, ValueId, VarId, I32};
use amdgpu_isa::{MmaAtom, DS_READ_B128X2, DS_READ_TR_B8X4, GLOBAL_LDS_X4};
use amdgpu_layout::{make_layout, tup, ComposedLayout, Layout as LLayout, Swizzle};

/// The K/V tile in LDS: [key, d] bytes, 128-byte rows, the FMHA swizzle.
pub fn kv_view() -> ComposedLayout {
    ComposedLayout::new(Swizzle::fmha128(), make_layout(tup![128, 128], tup![128, 1]))
}

/// Source addresses of the transposed V reads (`ds_read_b64_tr_b8`, the A operand `V^T` of the PV MFMA) in [key, d] coordinates (`key + 128 d`):
/// lane bit 0 -> byte +8, bits 1, 2 -> key +1, +2, bit 3 -> key +8, bit 4 -> byte +16, bit 5 -> key +4; values (read r: key +16 r, half hh: key +64 hh,
/// column group e: byte +32 e).
pub fn vt_tr_source() -> LLayout {
    make_layout(tup![tup![2, 2, 2, 2, 2, 2], tup![4, 2, 4]], tup![tup![8 * 128, 1, 2, 8, 16 * 128, 4], tup![16, 64, 32 * 128]])
}

/// The K/V/K-scale ring with layout-derived copies and readers.
pub struct KvRing3 {
    pub ks: Shared,
    pub vs: Shared,
    pub kd: Shared,
    pub ones: Shared,
    pub q1: Shared,
    pub q0: Shared,
    pub epi: Shared,
    pub kdma: DmaPart,
    pub vdma: DmaPart,
    pub kdc: KdCopy,
    pub kpart: LdsPart,
    pub vpart: LdsPart,
    pub kdr: RowScaleReader,
    pub mma: TiledMma,
}

impl KvRing3 {
    pub fn new(b: &mut FnBuilder, io: &FmhaIo, blk: &FmhaBlock, roles: &Roles, slots: u32) -> KvRing3 {
        let mut lds = LdsPlan::new();
        let [ks, vs] = lds.ring_group(slots, [("K", 16384, Layout::FMHA_SWZ), ("V", 16384, Layout::FMHA_SWZ)]);
        let kd = lds.ring("KD", 512, slots, Layout::Linear);
        lds.at(32768 * slots as i64 + 2048);
        let ones = lds.tile("ones", 2048, Layout::Linear);
        let q1 = lds.tile("Q1", 16384, Layout::Linear);
        let q0 = lds.tile("Q0", 16384, Layout::Linear);
        let epi = lds.private("epi", 4096, 8, &[&q1, &q0]);
        lds.commit(b);
        let lane = &blk.lane;
        let mut ctx = Ctx::new();
        let l = ctx.sym("lane", 0, 63, lane.lane());
        let w = ctx.sym("wave", 0, 3, ev(roles.wave_in_group));
        let mma = make_tiled_mma(MmaAtom::mfma_scale_32x32x64_f8(), false);
        // K: the A operand (32 keys x 64 bytes) at rows 32 s, k-group c
        let kview = ComposedLayout::new(Swizzle::fmha128(), make_layout(tup![32, 64], tup![128, 1]));
        let placements: Vec<i64> = (0..4).flat_map(|s| (0..2).map(move |c| 4096 * s + 64 * c)).collect();
        let kpart = make_tiled_copy_a(DS_READ_B128X2, &mma).partition_lds(b, &ctx, &l, &ks, &kview, &placements);
        let vpart = make_tiled_copy(DS_READ_TR_B8X4, vt_tr_source(), 128 * 128, 8).partition_lds(b, &ctx, &l, &vs, &kv_view(), &[0]);
        let kdr = RowScaleReader::new(b, &kd, lane);
        let (kt, vt, kdt) = (blk.kv_tile(b, &io.k), blk.kv_tile(b, &io.v), blk.kv_tile(b, &io.ksc));
        let (ksrc, vsrc, kdsrc) = (kt.row_source(b, 1), vt.row_source(b, 1), kdt.row_source(b, 1));
        let wv = ev(roles.wave_in_group);
        let kdma = partition_dma(b, &ctx, GLOBAL_LDS_X4, &l, &w, &wv, 4, ksrc, &ks, &kv_view());
        let vdma = partition_dma(b, &ctx, GLOBAL_LDS_X4, &l, &w, &wv, 4, vsrc, &vs, &kv_view());
        let kdc = if slots == 2 {
            KdCopy::Dma(copy::DwordRowsCopy::new(b, kdsrc, &kd, roles.wave_in_group, lane))
        } else {
            let rt_row = b.bind_i32(lane.tid().lshr(1));
            KdCopy::Vgpr(copy::DwordRowsVgpr::new(b, kdsrc, &kd, rt_row))
        };
        KvRing3 { ks, vs, kd, ones, q1, q0, epi, kdma, vdma, kdc, kpart, vpart, kdr, mma }
    }

    /// The copies of the tile at `key0` into `slot` that `role` produces (as `KvRing::stage`).
    pub fn stage(&self, b: &mut FnBuilder, role: Role, key0: &I32, slot: &SlotIdx) -> Option<ValueId> {
        match (role, &self.kdc) {
            (Role::A, KdCopy::Dma(kd)) => {
                kd.issue(b, key0, &self.kd.at(slot));
                self.kdma.issue(b, key0, &self.ks.at(slot));
                None
            }
            (Role::B, _) => {
                self.vdma.issue(b, key0, &self.vs.at(slot));
                None
            }
            (_, KdCopy::Vgpr(kd)) => {
                let v = kd.issue(b, key0);
                issue_rows_shared(b, key0, &[(&self.kdma, &self.ks.at(slot)), (&self.vdma, &self.vs.at(slot))]);
                Some(v)
            }
            _ => unreachable!("KvRing3::stage: role / K-scale staging combination"),
        }
    }
    pub fn stage_complete(&self, b: &mut FnBuilder, pending: Option<ValueId>, slot: &SlotIdx) {
        if let (KdCopy::Vgpr(kd), Some(v)) = (&self.kdc, pending) {
            kd.complete(b, v, &self.kd.at(slot));
        }
    }
    /// The ones block and the wave's Q fragments (as `KvRing::store_constants`).
    pub fn store_constants(&self, b: &mut FnBuilder, wid: ValueId, lane: &Lane, q: &frag::RowFrags) {
        let s0 = SlotIdx::new(0, 0);
        let ones_w = b.c_i32(0x38383838);
        let oa = b.bind_i32((ev(wid) * 64 + lane.lane()) * 4 + self.ones.base as i32);
        self.ones.at(&s0).write(b, |b| b.store_lds_b32(oa, ones_w));
        for (sh, w) in [(&self.q0, &q.words[0]), (&self.q1, &q.words[1])] {
            let a0 = b.bind_i32((ev(wid) * 2048 + sh.base as i32) + (lane.lane() << 4));
            let a1 = b.bind_i32(ev(a0) + 1024);
            let (g0, g1) = (b.frag4(w[0], w[1], w[2], w[3]), b.frag4(w[4], w[5], w[6], w[7]));
            sh.at(&s0).write(b, |b| {
                b.store_lds_b128(a0, g0);
                b.store_lds_b128(a1, g1);
            });
        }
    }
}

pub struct AttnBody3<'a> {
    ring: &'a KvRing3,
    blk: &'a FmhaBlock,
    wid: ValueId,
    pub sm: OnlineSoftmax,
    zero16: ValueId,
    c127: ValueId,
    c123: ValueId,
    s_init: Vec<ValueId>,
    pub s: Option<AccTile<C32x32F32, 4>>,
    pub o: Option<AccTile<C32x32F32, 4>>,
    p: Vec<ValueId>,
    kd_pending: Option<ValueId>,
}

impl<'a> AttnBody3<'a> {
    pub fn new(b: &mut FnBuilder, ring: &'a KvRing3, blk: &'a FmhaBlock, wid: ValueId, lut: bool) -> AttnBody3<'a> {
        let sm = OnlineSoftmax::new(b, &blk.lane, Recipe::vsx(lut));
        let zero16 = b.frag_zero(DType::Frag16);
        let (c127, c123) = (b.c_i32(127), b.c_i32(123));
        AttnBody3 { ring, blk, wid, sm, zero16, c127, c123, s_init: Vec::new(), s: None, o: None, p: Vec::new(), kd_pending: None }
    }
    /// One 32-key block of QK: the f16 bias MFMA, the two K fragments, the K-scale word and its bytes, two scaled MFMAs (the hand order).
    #[allow(clippy::too_many_arguments)]
    fn qk_block(&self, b: &mut FnBuilder, out: &QkOut, bias: ValueId, k: &LdsSlotPart, kd: &(SlotRef, I32), s: usize) -> Option<ValueId> {
        let blk = self.blk;
        let start = match out {
            QkOut::Vars(t) => {
                b.var_mfma_f16_z(t.accs[s].var, self.sm.ones_f16, bias);
                None
            }
            QkOut::Values(z) => Some(b.mfma_f16_wide(self.sm.ones_f16, bias, *z)),
        };
        let k0 = k.read(b, 2 * s);
        let k1 = k.read(b, 2 * s + 1);
        let kw = RowScaleReader::read(b, kd, s);
        let (sk0, sk1) = blk.sel.pair(b, kw);
        let q = &blk.q.frags;
        match out {
            QkOut::Vars(t) => {
                gemm(b, &self.ring.mma, t.accs[s].var, k0, q[0].v, sk0, blk.sq[0], 0, 0);
                gemm(b, &self.ring.mma, t.accs[s].var, k1, q[1].v, sk1, blk.sq[1], 0, 0);
                None
            }
            QkOut::Values(_) => {
                let s0 = b.mfma_scale_fp8(k0, q[0].v, start.expect("start"), sk0, blk.sq[0]);
                Some(b.mfma_scale_fp8(k1, q[1].v, s0, sk1, blk.sq[1]))
            }
        }
    }
    fn qk(&self, b: &mut FnBuilder, slot: &SlotIdx, out: QkOut, bias: ValueId) -> Vec<ValueId> {
        let r = self.ring;
        let k = r.kpart.at(b, &r.ks.at(slot));
        let kd = r.kdr.at(b, &r.kd.at(slot));
        (0..4).filter_map(|s| self.qk_block(b, &out, bias, &k, &kd, s)).collect()
    }
    /// PV with the row sums first: per half hh and dim group e, the transposed V fragment then the scaled MFMA (fixed scales).
    fn pv(&self, b: &mut FnBuilder, slot: &SlotIdx, w: &[ValueId]) {
        let r = self.ring;
        let v = r.vpart.at(b, &r.vs.at(slot));
        let p = [Frag::<B32x32x64Fp8>::from_words(b, &w[0..8]), Frag::<B32x32x64Fp8>::from_words(b, &w[8..16])];
        self.sm.row_sum(b, p[0].v, p[1].v);
        let o = self.o.expect("O");
        for (hh, pf) in p.iter().enumerate() {
            for e in 0..4usize {
                let vf = v.read(b, hh + 2 * e);
                gemm(b, &r.mma, o.accs[e].var, vf, pf.v, self.c127, self.c123, 0, 0);
            }
        }
    }
    pub fn out_scale(&self, b: &mut FnBuilder) -> ValueId {
        let l = self.sm.l_total(b);
        self.blk.out_scale(b, l)
    }
}

impl PipeBody for AttnBody3<'_> {
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
    fn v_phase(&mut self, b: &mut FnBuilder, _role: Role, t: &TileIt) {
        let (s, o) = (self.s.expect("S"), self.o.expect("O"));
        let mask = t.masked.then(|| KeyMask::new(b, t, self.blk.g4, self.blk.kv_len));
        let nb = self.sm.decide(b, &s, &o, mask.as_ref());
        self.p = self.sm.convert(b, &s, mask.as_ref(), 0, 4);
        self.p.push(nb);
    }
    fn m_phase(&mut self, b: &mut FnBuilder, _role: Role, t: &TileIt) {
        let mut w = std::mem::take(&mut self.p);
        let bias = w.pop().expect("bias word");
        self.pv(b, &t.cur, &w);
        if t.staged {
            let s = self.s.expect("S");
            self.qk(b, &t.next, QkOut::Vars(&s), bias);
        }
    }
    fn window(&mut self, b: &mut FnBuilder, _role: Role, t: &TileIt) {
        let r = self.ring;
        let (s, o) = (self.s.expect("S"), self.o.expect("O"));
        let mask = t.masked.then(|| KeyMask::new(b, t, self.blk.g4, self.blk.kv_len));
        let nb = self.sm.decide_deferred(b, &s, &o, mask.as_ref());
        let next = t.staged.then(|| (r.kpart.at(b, &r.ks.at(&t.next)), r.kdr.at(b, &r.kd.at(&t.next))));
        let mut w = Vec::with_capacity(16);
        for i in 0..4usize {
            w.extend(self.sm.convert(b, &s, mask.as_ref(), i, i + 1));
            if let Some((k, kd)) = &next {
                self.qk_block(b, &QkOut::Vars(&s), nb, k, kd, i);
            }
        }
        self.sm.commit_bias(b, nb);
        self.pv(b, &t.cur, &w);
    }
}

pub fn fmha_fwd_hd128_mxfp8_v3_vs128() -> Function {
    let mut fb = FnBuilder::new("fmha_fwd_hd128_mxfp8_v3_vs128", 512);
    fb.compile_profile(CompileProfile::Vs128);
    let b = &mut fb;
    let io = FmhaIo::declare(b, 256, false);
    let blk = io.block(b, 8);
    let roles = Roles::split(b, &blk.lane, 4);
    let ring = KvRing3::new(b, &io, &blk, &roles, 2);
    let pipe = Pipeline::new(Cadence::TwoPhaseOffset, blk.nkv, blk.kv_len, 128, Ext::ka(0x110).ceil_div(128));
    let lut = blk.lut_copy(b);
    versioned(b, lut, |b, lut| {
        roles.each(b, |b, role| {
            let wid = roles.wave_id(b, role);
            let mut body = AttnBody3::new(b, &ring, &blk, wid, lut);
            pipe.run(b, role, &mut body);
            let fin = body.out_scale(b);
            io.store_out(b, &blk, &body.o.expect("O"), fin);
        });
    });
    fb.finish()
}

pub fn fmha_fwd_hd128_mxfp8_v3_kvsplit() -> Function {
    let mut fb = FnBuilder::new("fmha_fwd_hd128_mxfp8_v3_kvsplit", 512);
    fb.compile_profile(CompileProfile::KvSplit);
    let b = &mut fb;
    let io = FmhaIo::declare(b, 256, true);
    let blk = io.block(b, 8);
    let roles = Roles::split(b, &blk.lane, 4);
    let ring = KvRing3::new(b, &io, &blk, &roles, 2);
    let pipe = Pipeline::new(Cadence::TwoPhaseOffset, blk.nkv, blk.kv_len, 128, Ext::ka(0x110).ceil_div(128));
    let lut = blk.lut_copy(b);
    versioned(b, lut, |b, lut| {
        roles.each(b, |b, role| {
            let wid = roles.wave_id(b, role);
            let mut body = AttnBody3::new(b, &ring, &blk, wid, lut);
            pipe.run(b, role, &mut body);
            let fin = body.out_scale(b);
            let l = body.sm.l_total(b);
            let lse = body.sm.lse(b, l);
            io.store_partial(b, &blk, &ring.epi.mine(wid), &body.o.expect("O"), fin, lse);
        });
    });
    fb.finish()
}

pub fn fmha_fwd_hd128_mxfp8_v3_kvsplit_combine8h() -> Function {
    let mut fb = FnBuilder::new("fmha_fwd_hd128_mxfp8_v3_kvsplit_combine8h", 256);
    fb.compile_profile(CompileProfile::Combine);
    combine_kernel(&mut fb, 8);
    fb.finish()
}

pub fn fmha_fwd_hd128_mxfp8_v3_t128() -> Function {
    let mut fb = FnBuilder::new("fmha_fwd_hd128_mxfp8_v3_t128", 256);
    fb.compile_profile(CompileProfile::T128);
    let b = &mut fb;
    let io = FmhaIo::declare(b, 128, false);
    let blk = io.block(b, 7);
    let roles = Roles::single(&blk.lane, 4);
    let ring = KvRing3::new(b, &io, &blk, &roles, 3);
    let mut pipe = Pipeline::new(Cadence::OneWindow, blk.nkv, blk.kv_len, 128, Ext::ka(0x110).ceil_div(128));
    pipe.prebind = vec![32768, 512];
    let lut = blk.lut_copy(b);
    versioned(b, lut, |b, lut| {
        let mut body = AttnBody3::new(b, &ring, &blk, blk.lane.wave, lut);
        pipe.run(b, Role::All, &mut body);
        let fin = body.out_scale(b);
        io.store_out(b, &blk, &body.o.expect("O"), fin);
    });
    fb.finish()
}
