//! The MXFP8 / per-tensor FP8 attention body shared by the `_vs128`, `_kvsplit`, `_t128` and `fp8pt` kernels (L2 template candidate,
//! docs/l2-templates.md T2 / T6): the derived orientation and views, the K / V / K-scale ring (`KvRing3`) and the phase body (`AttnBody3`).

use amdgpu_dsl::{DType, FnBuilder, I32, Op, Opnd, StageContract, ValueId, VarId};
use amdgpu_isa::{DS_READ_B128X2, DS_READ_B32, DS_READ_TR_B8X4, GLOBAL_LDS_X1, GLOBAL_LDS_X4, MmaAtom};
use amdgpu_layout::{ComposedLayout, make_layout, Swizzle, tup};
use amdgpu_tile::{Ctx, ev, Lane, LdsPlan, Role, Roles, Shared, SlotIdx, TileIt};
use amdgpu_tile::copy::{LdsPart, LdsSlotPart, make_tiled_copy, make_tiled_copy_a, make_tiled_copy_bcast};
use amdgpu_tile::dma::{DmaPart, issue_rows_shared, partition_dma};
use amdgpu_tile::frag::{AccTile, B32x32x64Fp8, C32x32F32, Frag, QkOut};
use amdgpu_tile::mma::{gemm, make_tiled_mma, TiledMma};
use amdgpu_tile::reduce::KeyMask;
use amdgpu_tile::softmax::{OnlineSoftmax, Recipe};
use amdgpu_codegen::weave::Units;
use amdgpu_layout::Layout as LLayout;
use amdgpu_tile::frag;
use amdgpu_tile::Layout;
use super::block3::Operands;
use super::io::FmhaBlock;
use amdgpu_stage::{PipeBody};

/// The derived S -> P orientation (amdgpu_tile::orient::search_attention over the scaled fp8 atoms): the producer / consumer atom, checked to be the one this
/// file implements (S^T = K Q^T, P = the B operand of O^T = V^T P^T, zero conversion cost).
pub fn attention_orientation() -> (MmaAtom, amdgpu_tile::orient::Orientation) {
    let atoms = [MmaAtom::mfma_scale_32x32x64_f8(), MmaAtom::mfma_scale_16x16x128_f8()];
    let best = amdgpu_tile::orient::search_attention(&atoms).into_iter().next().expect("candidates");
    assert!(best.s_transposed && best.p_operand == amdgpu_tile::orient::POperand::B && best.cost() == 0 && best.producer == best.consumer, "FMHA: the derived S -> P orientation {best:?} is not the implemented one");
    let atom = atoms.into_iter().find(|a| a.name == best.producer).expect("atom");
    (atom, best)
}

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

/// K-scale staging: a layout-derived DMA (one dword per row, waves 2 and 3 repeat waves 0 and 1) or the `_t128` VGPR path.
pub enum KdCopy3 {
    Dma(DmaPart),
    Vgpr(amdgpu_tile::rows::DwordRowsVgpr),
    /// no K scales (per-tensor FP8: the unit word)
    None,
}

/// The K-scale reader TV: thread (j = lane % 32, h = lane / 32 broadcast) x value (row group s) -> row `j + 32 s` of the 128-word tile.
pub fn kd_read_tv() -> LLayout {
    make_layout(tup![tup![32, 2], 4], tup![tup![1, 0], 32])
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
    pub kdc: KdCopy3,
    pub kpart: LdsPart,
    pub vpart: LdsPart,
    /// the K-scale words: lane (j, h) reads word j + 32 s (both lane halves read the same word: a broadcast TV layout); None without K scales
    pub kdpart: Option<LdsPart>,
    pub mma: TiledMma,
    /// MUBUF form of the K / V DMA (`_vs128`): SGPR per-tile advance, loop-invariant voffsets
    pub buf: Option<(amdgpu_tile::dma::BufForm, amdgpu_tile::dma::BufForm)>,
    /// MUBUF form of the K-scale DMA
    pub kd_buf: Option<amdgpu_tile::dma::BufForm>,
}

impl KvRing3 {
    pub fn new(b: &mut FnBuilder, io: &dyn Operands, blk: &FmhaBlock, roles: &Roles, slots: u32) -> KvRing3 {
        Self::new_x(b, io, blk, roles, slots, true)
    }
    /// `kscale = false`: no K-scale staging or reads (per-tensor FP8; the KD ring keeps its place so the LDS plan is the shipped one).
    pub fn new_x(b: &mut FnBuilder, io: &dyn Operands, blk: &FmhaBlock, roles: &Roles, slots: u32, kscale: bool) -> KvRing3 {
        let mut lds = LdsPlan::new();
        let [ks, vs] = lds.ring_group(slots, [("K", 16384, Layout::FMHA_SWZ), ("V", 16384, Layout::FMHA_SWZ)]);
        let kd = lds.ring("KD", 512, slots, Layout::Linear);
        lds.at(32768 * slots as i64 + 2048);
        let ones = lds.tile("ones", 2048, Layout::Linear);
        let q1 = lds.tile("Q1", 16384, Layout::Linear);
        let q0 = lds.tile("Q0", 16384, Layout::Linear);
        let epi = lds.private("epi", 4096, 8, &[&q1, &q0]);
        lds.commit(b);
        Self::parts(b, io, blk, roles, slots, kscale, [ks, vs, kd, ones, q1, q0, epi])
    }
    /// The same ring (LDS regions) with every lane partition and copy rebuilt from `blk` / `roles` (fresh values: a second pass over the ring
    /// does not keep the first pass's address registers live; the paired causal form).
    pub fn rebuilt(&self, b: &mut FnBuilder, io: &dyn Operands, blk: &FmhaBlock, roles: &Roles, slots: u32) -> KvRing3 {
        let sh = [self.ks.clone(), self.vs.clone(), self.kd.clone(), self.ones.clone(), self.q1.clone(), self.q0.clone(), self.epi.clone()];
        let mut r = Self::parts(b, io, blk, roles, slots, self.kdpart.is_some(), sh);
        if self.buf.is_some() {
            r.enable_buffer_dma(b);
        }
        if self.kd_buf.is_some() {
            r.enable_buffer_kd(b);
        }
        r
    }
    fn parts(b: &mut FnBuilder, io: &dyn Operands, blk: &FmhaBlock, roles: &Roles, slots: u32, kscale: bool, sh: [Shared; 7]) -> KvRing3 {
        let [ks, vs, kd, ones, q1, q0, epi] = sh;
        let lane = &blk.lane;
        let mut ctx = Ctx::new();
        let l = ctx.sym("lane", 0, 63, lane.lane());
        let w = ctx.sym("wave", 0, 3, ev(roles.wave_in_group));
        // the S -> P hand-off orientation is DERIVED (amdgpu_tile::orient): the cheapest candidate computes S^T (A = K, B = Q) with the 32x32x64 atom and feeds P
        // as the B operand of the 32x32x64 PV MMA (O^T = V^T P^T) with a zero-cost conversion (the fp8 packing in accumulator order + the key
        // permutation of the transposed V reads); this builder implements exactly that orientation
        let o = attention_orientation();
        let mma = make_tiled_mma(o.0, false);
        // K: the A operand (32 keys x 64 bytes) at rows 32 s, k-group c
        let kview = ComposedLayout::new(Swizzle::fmha128(), make_layout(tup![32, 64], tup![128, 1]));
        let placements: Vec<i64> = (0..4).flat_map(|s| (0..2).map(move |c| 4096 * s + 64 * c)).collect();
        let kpart = make_tiled_copy_a(DS_READ_B128X2, &mma).partition_lds(b, &ctx, &l, &ks, &kview, &placements);
        let vpart = make_tiled_copy(DS_READ_TR_B8X4, vt_tr_source(), 128 * 128, 8).partition_lds(b, &ctx, &l, &vs, &kv_view(), &[0]);
        let kdpart = kscale.then(|| make_tiled_copy_bcast(DS_READ_B32, kd_read_tv(), 128, 32).partition_lds(b, &ctx, &l, &kd, &ComposedLayout::plain(make_layout(128, 4)), &[0]));
        let (kt, vt) = (blk.kv_tile(b, io.k()), blk.kv_tile(b, io.v()));
        let kdt = kscale.then(|| blk.kv_tile(b, io.ksc()));
        let (ksrc, vsrc) = (kt.row_source(b, 1), vt.row_source(b, 1));
        let kdsrc = kdt.map(|t| t.row_source(b, 1));
        let wv = ev(roles.wave_in_group);
        let kdma = partition_dma(b, &ctx, GLOBAL_LDS_X4, &l, &w, &wv, 4, ksrc, &ks, &kv_view());
        let vdma = partition_dma(b, &ctx, GLOBAL_LDS_X4, &l, &w, &wv, 4, vsrc, &vs, &kv_view());
        let kdc = if let Some(kdsrc) = kdsrc { if slots == 2 {
            // 128 rows x 4 bytes, one dword per lane: waves 0 / 1 stage rows 0..63 / 64..127, waves 2 / 3 repeat them (the wave index taken mod 2)
            let w2 = w.mod_c(2, &ctx.dyns);
            let wv2 = b.let_i32(ev(roles.wave_in_group) & 1);
            KdCopy3::Dma(partition_dma(b, &ctx, GLOBAL_LDS_X1, &l, &w2, &wv2, 1, kdsrc, &kd, &ComposedLayout::plain(make_layout(tup![128, 4], tup![4, 1]))))
        } else {
            let rt_row = b.bind_i32(lane.tid().lshr(1));
            KdCopy3::Vgpr(amdgpu_tile::rows::DwordRowsVgpr::new(b, kdsrc, &kd, rt_row))
        } } else {
            KdCopy3::None
        };
        KvRing3 { ks, vs, kd, ones, q1, q0, epi, kdma, vdma, kdc, kpart, vpart, kdpart, mma, buf: None, kd_buf: None }
    }
    /// Switches the K / V staging to the buffer form (the vendor's `buffer_load_dwordx4 ... lds` with an SGPR soffset).
    pub fn enable_buffer_dma(&mut self, b: &mut FnBuilder) {
        let k = self.kdma.buffer_form(b);
        let v = self.vdma.buffer_form(b);
        self.buf = Some((k, v));
    }
    /// The K-scale DMA in the buffer form too.
    pub fn enable_buffer_kd(&mut self, b: &mut FnBuilder) {
        if let KdCopy3::Dma(kd) = &self.kdc {
            self.kd_buf = Some(kd.buffer_form(b));
        }
    }

    /// The copies of the tile at `key0` into `slot` that `role` produces, as emission units (buffer form: one unit per DMA instruction). `None` for
    /// the VGPR K-scale staging (`_t128`), which keeps the direct form.
    pub fn stage_units(&self, b: &mut FnBuilder, role: Role, key0: &I32, slot: &SlotIdx) -> Option<Units> {
        match (role, &self.kdc, &self.buf, &self.kd_buf) {
            (Role::A, KdCopy3::Dma(kd), Some((kb, _)), Some(kdb)) => {
                let mut u = kd.issue_buffer_units(b, kdb, key0, &self.kd.at(slot));
                u.append(self.kdma.issue_buffer_units(b, kb, key0, &self.ks.at(slot)), &[]);
                Some(u)
            }
            (Role::A, KdCopy3::None, Some((kb, _)), _) => Some(self.kdma.issue_buffer_units(b, kb, key0, &self.ks.at(slot))),
            (Role::B, _, Some((_, vb)), _) => Some(self.vdma.issue_buffer_units(b, vb, key0, &self.vs.at(slot))),
            _ => None,
        }
    }
    /// The copies of the tile at `key0` into `slot` that `role` produces (as `KvRing::stage`).
    pub fn stage(&self, b: &mut FnBuilder, role: Role, key0: &I32, slot: &SlotIdx) -> Option<ValueId> {
        // the slot offsets bound before this stage are its scalar inputs (taken before the units are captured: capturing binds new ones)
        let bound = slot.offset_scalars();
        if let Some(u) = self.stage_units(b, role, key0, slot) {
            // the slot offsets are shared with every reader of the slot instance (SlotIdx cache): declared scalar hand-overs
            let c = StageContract { barrier_phase: StageContract::ANY_PHASE, roles: vec!["group".into()], scalars_in: bound, ..Default::default() };
            let sl = slot.clone();
            amdgpu_stage::stage::StageInst::new("move.kv_stage", amdgpu_stage::stage::Prim::M, "kv.stage", c, u)
                .param("role", format!("{role:?}"))
                .handover(move |_b| (Vec::new(), sl.offset_scalars()))
                .emit(b);
            return None;
        }
        match (role, &self.kdc) {
            (Role::A, KdCopy3::Dma(kd)) => {
                match &self.kd_buf {
                    Some(kb) => kd.issue_buffer(b, kb, key0, &self.kd.at(slot)),
                    None => kd.issue(b, key0, &self.kd.at(slot)),
                }
                match &self.buf {
                    Some((kb, _)) => self.kdma.issue_buffer(b, kb, key0, &self.ks.at(slot)),
                    None => self.kdma.issue(b, key0, &self.ks.at(slot)),
                }
                None
            }
            (Role::A, KdCopy3::None) => {
                match &self.buf {
                    Some((kb, _)) => self.kdma.issue_buffer(b, kb, key0, &self.ks.at(slot)),
                    None => self.kdma.issue(b, key0, &self.ks.at(slot)),
                }
                None
            }
            (Role::B, _) => {
                match &self.buf {
                    Some((_, vb)) => self.vdma.issue_buffer(b, vb, key0, &self.vs.at(slot)),
                    None => self.vdma.issue(b, key0, &self.vs.at(slot)),
                }
                None
            }
            (_, KdCopy3::Vgpr(kd)) => {
                let v = kd.issue(b, key0);
                issue_rows_shared(b, key0, &[(&self.kdma, &self.ks.at(slot)), (&self.vdma, &self.vs.at(slot))]);
                Some(v)
            }
            _ => unreachable!("KvRing3::stage: role / K-scale staging combination"),
        }
    }
    pub fn stage_complete(&self, b: &mut FnBuilder, pending: Option<ValueId>, slot: &SlotIdx) {
        if let (KdCopy3::Vgpr(kd), Some(v)) = (&self.kdc, pending) {
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
    /// the vendor's priority ping-pong (`_vs128`): `s_setprio 1` at the start of the MFMA phase, `s_setprio 0` at the softmax phase
    pub setprio: bool,
    /// K scales: one shift per scale word + op_sel byte select on the scaled QK MFMA (`_vs128`)
    pub kscale_sel: bool,
    /// the reference update entirely inside the rollback branch (`OnlineSoftmax::decide_lazy`)
    pub lazy_bias: bool,
    bias_next: Option<Opnd>,
    /// one f16 bias MFMA per tile (the vendor's single helper): its result is the C input of the four S tiles' first QK MFMA (`_vs128`)
    pub biasonce: bool,
    bias_tile: std::cell::Cell<Option<Opnd>>,
    /// the P words in `p` were handed over by a softmax STAGE (declared live-in of the matrix stage)
    p_staged: bool,
    /// causal key mask (per-tensor FP8 `_causal`)
    pub causal: Option<CausalMask>,
}

/// The causal mask of a block: key k of this lane's query row is visible when `k < min(row + qoff1, kv_len)` (computed in the masking branch); the
/// tiles with a key at or past `lim_min` (the wave's smallest limit, wave-uniform) take the in-place masking branch.
#[derive(Clone, Copy, Debug)]
pub struct CausalMask {
    /// SK - SQ + 1 (wave-uniform): the lane limit is min(row + qoff1, kv_len)
    pub qoff1: ValueId,
    pub lim_min: ValueId,
    pub neg_big: ValueId,
}

/// The P hand-over of the softmax stage (16 packed fp8 words; element table: word index, the layout fingerprint of the in-place packing).
fn p_val(words: Vec<ValueId>) -> amdgpu_dsl::StageVal {
    let n = words.len() as i64;
    amdgpu_dsl::StageVal { name: "P".into(), values: words, table: (0..n).collect(), elem_bits: 32, storage: "vgpr".into(), layout: String::new() }
}

/// The reference words of the lazy recipe when they are a value (a loop variable needs no hand-over).
fn bias_scalar(o: Opnd) -> Vec<amdgpu_dsl::StageScalar> {
    match o {
        Opnd::Val(v) => vec![amdgpu_dsl::StageScalar { name: "ref".into(), values: vec![v], owner: "row".into() }],
        _ => Vec::new(),
    }
}

impl<'a> AttnBody3<'a> {
    pub fn new(b: &mut FnBuilder, ring: &'a KvRing3, blk: &'a FmhaBlock, wid: ValueId, lut: bool) -> AttnBody3<'a> {
        let sm = OnlineSoftmax::new(b, &blk.lane, Recipe::vsx(lut));
        let zero16 = b.frag_zero(DType::Frag16);
        let (c127, c123) = (b.c_i32(127), b.c_i32(123));
        AttnBody3 { ring, blk, wid, sm, zero16, c127, c123, s_init: Vec::new(), s: None, o: None, p: Vec::new(), kd_pending: None, setprio: false, kscale_sel: false, lazy_bias: false, bias_next: None, biasonce: false, bias_tile: std::cell::Cell::new(None), p_staged: false, causal: None }
    }
    /// The K-scale reader of `slot` (None without K scales).
    fn kd_at(&self, b: &mut FnBuilder, slot: &SlotIdx) -> Option<LdsSlotPart> {
        let r = self.ring;
        r.kdpart.as_ref().map(|p| p.at(b, &r.kd.at(slot)))
    }
    /// The K-scale word of 32-key block `s`: read from the ring, or the unit word (per-tensor FP8).
    fn kword(&self, b: &mut FnBuilder, kd: Option<&LdsSlotPart>, s: usize) -> ValueId {
        match kd {
            Some(p) => p.read(b, s),
            None => self.blk.pt.expect("no K scales without the per-tensor scales").kunit,
        }
    }
    /// The bias MFMA operand: the reference words themselves, or (per-tensor FP8) the two-word encoding of `b / m` (`b = 4 (hi + lo)`; the
    /// encoding of `b / m` is `hi' + lo'` = (hi + lo) / m split into an f16 and its f16 residual).
    fn bias_in(&self, b: &mut FnBuilder, bias: Opnd) -> Opnd {
        let Some(pt) = self.blk.pt else { return bias };
        let wu = b.frag2_lane(bias, 0);
        let w = b.bitcast_i32(wu);
        let c16 = b.c_i32(16);
        let hi = b.let_(Op::CvtF16ToF32(Opnd::Val(w)));
        let lo_s = b.let_(Op::LshrI32(Opnd::Val(w), Opnd::Val(c16)));
        let lo = b.let_(Op::CvtF16ToF32(Opnd::Val(lo_s)));
        let s = b.add_f32(hi, lo);
        let q = b.mul_f32(s, pt.inv_m);
        let h_raw = b.let_(Op::CvtF32ToF16(Opnd::Val(q)));
        let hq = b.let_(Op::CvtF16ToF32(Opnd::Val(h_raw)));
        let r = b.sub_f32(q, hq);
        let l_raw = b.let_(Op::CvtF32ToF16(Opnd::Val(r)));
        let l_sh = b.shl_i32(l_raw, c16);
        let wq = b.let_(Op::AddI32(Opnd::Val(h_raw), Opnd::Val(l_sh)));
        Opnd::Val(b.frag2(wq, wq))
    }
    /// The causal mask (in place, only on tiles that reach the wave's smallest limit). (The per-tensor m scale of S is applied where S is
    /// produced: `prologue_compute` and the end of `m_phase`.)
    fn pt_prepare(&self, b: &mut FnBuilder, s: &AccTile<C32x32F32, 4>, t: &TileIt) {
        if let Some(cm) = self.causal {
            let key_t = b.bind_i32(t.key0());
            let partial = b.bind_cond((ev(key_t) + 127).ge(ev(cm.lim_min)));
            b.begin_if(Opnd::Val(partial));
            // in place, two keys per v_pk_fma: S += hidden * (-1e30, -1e30) with hidden in {0, 1}; key (i, r) of the lane = key0 + g4 + 32 i +
            // row_const(r) is visible iff its offset 32 i + row_const(r) < rel = lim - key0 - g4 (one compare against a constant per key)
            let rel = b.bind_i32((&self.blk.row0 + self.blk.lane.j() + ev(cm.qoff1)).min(ev(self.blk.kv_len)) - t.key0() - ev(self.blk.g4));
            let (z, one) = (b.c_f32(0.0), b.c_f32(1.0));
            let nb2 = b.frag2_dup(cm.neg_big);
            for (i, a) in s.accs.iter().enumerate() {
                for r in (0..16u8).step_by(2) {
                    let h: Vec<ValueId> = (0..2u8)
                        .map(|d| {
                            let ok = b.bind_cond(ev(rel).gt(C32x32F32::row_const(r + d) + 32 * i as i32));
                            let x = b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(ok), a: Opnd::Val(z), b: Opnd::Val(one) });
                            b.bitcast_i32(x)
                        })
                        .collect();
                    let hp = b.frag2(h[0], h[1]);
                    b.var_pk_fma(a.var, r, hp, nb2);
                }
            }
            b.end_if();
        }
    }
    /// One 32-key block of QK: the f16 bias MFMA, the two K fragments, the K-scale word and its bytes, two scaled MFMAs (the hand order).
    #[allow(clippy::too_many_arguments)]
    fn qk_block(&self, b: &mut FnBuilder, out: &QkOut, bias: Opnd, k: &LdsSlotPart, kd: Option<&LdsSlotPart>, s: usize) -> Option<ValueId> {
        let blk = self.blk;
        let start = match out {
            QkOut::Vars(t) => {
                if self.bias_tile.get().is_none() {
                    b.var_mfma_f16_z(t.accs[s].var, self.sm.ones_f16, bias);
                }
                None
            }
            QkOut::Values(z) => Some(b.mfma_f16_wide(self.sm.ones_f16, bias, *z)),
        };
        let k0 = k.read(b, 2 * s);
        let k1 = k.read(b, 2 * s + 1);
        let kw = self.kword(b, kd, s);
        let q = &blk.q.frags;
        if let (true, QkOut::Vars(t)) = (self.kscale_sel, out) {
            // the vendor form: ONE shift by 8 x lane half, then the scaled MFMA picks byte 0 (k chunk 0) / byte 2 (k chunk 1) with op_sel
            let w = if blk.pt.is_some() { kw } else { b.bind_i32(ev(kw).lshr(ev(blk.sel.g8))) };
            if let (Some(bt), true) = (self.bias_tile.get(), s < 3) {
                b.var_mfma_scale_c(t.accs[s].var, k0, q[0].v, bt, w, blk.sq[0]);
            } else {
                gemm(b, &self.ring.mma, t.accs[s].var, k0, q[0].v, w, blk.sq[0], 0, 0);
            }
            gemm(b, &self.ring.mma, t.accs[s].var, k1, q[1].v, w, blk.sq[1], 2, 0);
            return None;
        }
        let (sk0, sk1) = if blk.pt.is_some() { (kw, kw) } else { blk.sel.pair(b, kw) };
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
    /// QK into the S variables as captured units: the setup (slot-bound K / K-scale partitions; the one bias MFMA of the `biasonce` form) and per
    /// 32-key block a read unit (K fragments, K-scale word) and an MMA unit (the shift and the two scaled MFMAs).
    fn qk_units(&self, b: &mut FnBuilder, slot: &SlotIdx, s: AccTile<C32x32F32, 4>, bias: Opnd) -> Units {
        let r = self.ring;
        let mut u = Units::new();
        let once = self.biasonce && self.kscale_sel;
        let (_, (k, kd)) = u.capture(b, "qk.setup", |b| {
            let parts = (r.kpart.at(b, &r.ks.at(slot)), self.kd_at(b, slot));
            if once {
                // the one bias MFMA writes S tile 3 (zero accumulator: no zero-fragment copies); tiles 0-2 take it as the C input of their first
                // scaled MFMA, tile 3 accumulates in place last
                b.var_mfma_f16_z(s.accs[3].var, self.sm.ones_f16, bias);
                self.bias_tile.set(Some(Opnd::Var(s.accs[3].var)));
            }
            parts
        });
        for blk in 0..4usize {
            let (_, (k0, k1, kw)) = u.capture(b, format!("qk.read{blk}"), |b| {
                if self.bias_tile.get().is_none() {
                    b.var_mfma_f16_z(s.accs[blk].var, self.sm.ones_f16, bias);
                }
                (k.read(b, 2 * blk), k.read(b, 2 * blk + 1), self.kword(b, kd.as_ref(), blk))
            });
            u.capture(b, format!("qk.mma{blk}"), |b| self.qk_block_mma(b, &s, k0, k1, kw, blk));
        }
        self.bias_tile.set(None);
        u
    }
    /// PV as captured units: the setup (slot-bound V partition, the two P fragments), the row sums, per (half hh, dim group e) a read unit and an
    /// MMA unit.
    fn pv_units(&self, b: &mut FnBuilder, slot: &SlotIdx, w: &[ValueId]) -> Units {
        let r = self.ring;
        let mut u = Units::new();
        let (_, (v, p)) = u.capture(b, "pv.setup", |b| {
            (r.vpart.at(b, &r.vs.at(slot)), [Frag::<B32x32x64Fp8>::from_words(b, &w[0..8]).v, Frag::<B32x32x64Fp8>::from_words(b, &w[8..16]).v])
        });
        u.capture(b, "pv.rowsum", |b| self.sm.row_sum(b, p[0], p[1]));
        let o = self.o.expect("O");
        for hh in 0..2usize {
            for e in 0..4usize {
                let (_, vf) = u.capture(b, format!("pv.read.h{hh}e{e}"), |b| v.read(b, hh + 2 * e));
                u.capture(b, format!("pv.mma.h{hh}e{e}"), |b| gemm(b, &r.mma, o.accs[e].var, vf, p[hh], self.c127, self.c123, 0, 0));
            }
        }
        u
    }
    /// The MMA half of a QK block into the S variables (the K fragments / scale word already read).
    fn qk_block_mma(&self, b: &mut FnBuilder, t: &AccTile<C32x32F32, 4>, k0: ValueId, k1: ValueId, kw: ValueId, s: usize) {
        let blk = self.blk;
        let q = &blk.q.frags;
        if self.kscale_sel {
            let w = if blk.pt.is_some() { kw } else { b.bind_i32(ev(kw).lshr(ev(blk.sel.g8))) };
            if let (Some(bt), true) = (self.bias_tile.get(), s < 3) {
                b.var_mfma_scale_c(t.accs[s].var, k0, q[0].v, bt, w, blk.sq[0]);
            } else {
                gemm(b, &self.ring.mma, t.accs[s].var, k0, q[0].v, w, blk.sq[0], 0, 0);
            }
            gemm(b, &self.ring.mma, t.accs[s].var, k1, q[1].v, w, blk.sq[1], 2, 0);
            return;
        }
        let (sk0, sk1) = if blk.pt.is_some() { (kw, kw) } else { blk.sel.pair(b, kw) };
        gemm(b, &self.ring.mma, t.accs[s].var, k0, q[0].v, sk0, blk.sq[0], 0, 0);
        gemm(b, &self.ring.mma, t.accs[s].var, k1, q[1].v, sk1, blk.sq[1], 0, 0);
    }
    fn qk(&self, b: &mut FnBuilder, slot: &SlotIdx, out: QkOut, bias: Opnd) -> Vec<ValueId> {
        if let QkOut::Vars(t) = &out {
            self.qk_units(b, slot, **t, bias).emit_all(b);
            return Vec::new();
        }
        let r = self.ring;
        let k = r.kpart.at(b, &r.ks.at(slot));
        let kd = self.kd_at(b, slot);
        if let (true, true, QkOut::Vars(t)) = (self.biasonce, self.kscale_sel, &out) {
            // the one bias MFMA writes S tile 3 (zero accumulator: no zero-fragment copies); tiles 0-2 take it as the C input of their first
            // scaled MFMA, tile 3 accumulates in place last
            b.var_mfma_f16_z(t.accs[3].var, self.sm.ones_f16, bias);
            self.bias_tile.set(Some(Opnd::Var(t.accs[3].var)));
        }
        let res = (0..4).filter_map(|s| self.qk_block(b, &out, bias, &k, kd.as_ref(), s)).collect();
        self.bias_tile.set(None);
        res
    }
    /// PV with the row sums first: per half hh and dim group e, the transposed V fragment then the scaled MFMA (fixed scales).
    fn pv(&self, b: &mut FnBuilder, slot: &SlotIdx, w: &[ValueId]) {
        self.pv_units(b, slot, w).emit_all(b);
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
        let bias = self.bias_in(b, Opnd::Val(self.sm.bias_init));
        self.s_init = self.qk(b, &t0.cur, QkOut::Values(self.zero16), bias);
        if let Some(pt) = self.blk.pt {
            // tile 0: the m scale on the S values (two dwords per v_pk_mul_f32)
            self.s_init = self
                .s_init
                .clone()
                .into_iter()
                .map(|v| {
                    let parts: Vec<Opnd> = (0..8u8)
                        .map(|k| {
                            let p = b.frag_slice(v, 2 * k, 2);
                            Opnd::Val(b.pk_mul_f32(p, pt.m2))
                        })
                        .collect();
                    b.frag_cat(&parts)
                })
                .collect();
        }
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
        if self.setprio {
            b.set_prio(0);
        }
        let (s, o) = (self.s.expect("S"), self.o.expect("O"));
        let mask = t.masked.then(|| KeyMask::new(b, t, self.blk.g4, self.blk.kv_len));
        if self.lazy_bias {
            // no seed needed: the initial reference (bias words 0x4800: b = 64) puts the first tile's max far above the threshold, so tile 0
            // always rebases
            let mut u = Units::new();
            if self.blk.pt.is_some() || self.causal.is_some() {
                u.capture(b, "softmax.pt", |b| self.pt_prepare(b, &s, t));
            }
            let (_, nb) = u.capture(b, "softmax.decide", |b| self.sm.decide_lazy(b, &s, &o, mask.as_ref()));
            let (_, pw) = u.capture(b, "softmax.convert", |b| self.sm.convert(b, &s, mask.as_ref(), 0, 4));
            let nb = if self.blk.pt.is_some() { u.capture(b, "softmax.pt_bias", |b| self.bias_in(b, nb)).1 } else { nb };
            let c = StageContract { barrier_phase: StageContract::ANY_PHASE, roles: vec!["group".into()], ..Default::default() };
            let pw2 = pw.clone();
            amdgpu_stage::stage::StageInst::new("online.softmax", amdgpu_stage::stage::Prim::O, "v_phase", c, u)
                .param("recipe", "vendor-lazy")
                .handover(move |_b| (vec![p_val(pw2)], bias_scalar(nb)))
                .emit(b);
            self.bias_next = Some(nb);
            self.p = pw;
            self.p_staged = true;
            return;
        }
        self.pt_prepare(b, &s, t);
        let nb = self.sm.decide(b, &s, &o, mask.as_ref());
        self.p = self.sm.convert(b, &s, mask.as_ref(), 0, 4);
        let nb = match self.bias_in(b, Opnd::Val(nb)) {
            Opnd::Val(v) => v,
            _ => unreachable!("bias_in of a value"),
        };
        self.p.push(nb);
        self.p_staged = false;
    }
    fn m_phase(&mut self, b: &mut FnBuilder, _role: Role, t: &TileIt) {
        let mut w = std::mem::take(&mut self.p);
        let bias = match self.bias_next.take() {
            Some(op) => op,
            None => Opnd::Val(w.pop().expect("bias word")),
        };
        if self.setprio {
            b.set_prio(1);
        }
        let s = self.s.expect("S");
        // the matrix phase = stage `mma.pv` (tile t) then stage `mma.qk` (tile t + 1), merged into one region and woven in declaration order;
        // it consumes the P words (and the reference words of the lazy recipe) handed over by the softmax stage
        let mut c = StageContract { barrier_phase: StageContract::ANY_PHASE, roles: vec!["group".into()], ..Default::default() };
        if self.p_staged {
            c.live_in.push(p_val(w.clone()));
            c.scalars_in = bias_scalar(bias);
        }
        c.scalars_in.extend(t.cur.offset_scalars());
        if t.staged {
            c.scalars_in.extend(t.next.offset_scalars());
        }
        let mut insts = vec![amdgpu_stage::stage::StageInst::new("mma.pv", amdgpu_stage::stage::Prim::A, "pv", Default::default(), self.pv_units(b, &t.cur, &w))];
        if t.staged {
            insts.push(amdgpu_stage::stage::StageInst::new("mma.qk", amdgpu_stage::stage::Prim::A, "qk", Default::default(), self.qk_units(b, &t.next, s, bias)));
        }
        let (cur, next, staged) = (t.cur.clone(), t.next.clone(), t.staged);
        amdgpu_stage::stage::merge("m_phase", c, insts)
            .handover(move |_b| {
                let mut s = cur.offset_scalars();
                if staged {
                    s.extend(next.offset_scalars());
                }
                (Vec::new(), s)
            })
            .emit(b);
        if let (Some(pt), true) = (self.blk.pt, t.staged) {
            // per-tensor FP8: the next tile's S scaled by m at the end of the matrix phase, where it is produced. At the start of the softmax
            // phase (the first version) the 32 v_pk_mul lengthened the VALU phase that both ping-pong groups wait on (sched's ATT: VALU stall
            // 1.12 M vs 0.25 M for the vendor); moved here, same run: B2 H32 8192^2 1083 -> 933 us (1.20x aiter), causal 640 -> 556, B4 H16
            // 2048^2 90 -> 71, KV split B1 H8 4096^2 51 -> 44 / causal 48.5 -> 39.7
            for a in &s.accs {
                b.var_scale_pk(a.var, pt.m2);
            }
        }
    }
    fn window(&mut self, b: &mut FnBuilder, _role: Role, t: &TileIt) {
        let r = self.ring;
        let (s, o) = (self.s.expect("S"), self.o.expect("O"));
        let mask = t.masked.then(|| KeyMask::new(b, t, self.blk.g4, self.blk.kv_len));
        let (nb, lazy) = if self.lazy_bias {
            (self.sm.decide_lazy(b, &s, &o, mask.as_ref()), true)
        } else {
            (Opnd::Val(self.sm.decide_deferred(b, &s, &o, mask.as_ref())), false)
        };
        // the window: conversion chunk i, then QK block i of the next tile (captured units, woven in this order)
        let bound = if t.staged { t.next.offset_scalars() } else { Vec::new() };
        let mut u = Units::new();
        let parts = t.staged.then(|| u.capture(b, "window.qk.setup", |b| (r.kpart.at(b, &r.ks.at(&t.next)), self.kd_at(b, &t.next))).1);
        let mut w: Vec<ValueId> = Vec::with_capacity(16);
        for i in 0..4usize {
            w.extend(u.capture(b, format!("window.convert{i}"), |b| self.sm.convert(b, &s, mask.as_ref(), i, i + 1)).1);
            if let Some((k, kd)) = &parts {
                u.capture(b, format!("window.qk.block{i}"), |b| self.qk_block(b, &QkOut::Vars(&s), nb, k, kd.as_ref(), i));
            }
        }
        // the window = stage `online.softmax` (conversion chunks of tile t) woven with stage `mma.qk` (blocks of tile t + 1), alternating
        let c = StageContract { barrier_phase: StageContract::ANY_PHASE, roles: vec!["group".into()], scalars_in: bound, ..Default::default() };
        let (nx, staged) = (t.next.clone(), t.staged);
        amdgpu_stage::stage::StageInst::new("online.softmax", amdgpu_stage::stage::Prim::O, "window", c, u)
            .param("with", "mma.qk blocks")
            .handover(move |_b| (Vec::new(), if staged { nx.offset_scalars() } else { Vec::new() }))
            .emit(b);
        if !lazy {
            if let Opnd::Val(v) = nb {
                self.sm.commit_bias(b, v);
            }
        }
        self.pv(b, &t.cur, &w);
    }
}
