//! The four shipped FMHA forward kernels `fmha_fwd_hd128_mxfp8_v3_{vs128, kvsplit, kvsplit_combine8h, t128}` (docs/fmha-dsl-tour.md). Every LDS
//! / global address is DERIVED from layouts (amdgpu-layout) and instruction atoms (amdgpu-isa) through the amdgpu-tile abstractions:
//! * the K fragment reads = `make_tiled_copy_a(ds_read_b128 x2, mma)` over the swizzled [key, d] view (placements `4096 s + 64 c`);
//! * the transposed V reads = the `ds_read_b64_tr_b8 x4` source layout over the same view;
//! * the K / V staging = `partition_dma` (destination-side derivation: the swizzle on the source, M0 at the top of the wave slice), MUBUF form in `_vs128`;
//! * QK / PV = `gemm` of the scaled 32x32x64 atom; the online softmax, row sums, pipeline cadence and epilogues come from amdgpu-tile.
//! The ABI types, per-block state and the KV-split combine body live in `io.rs`.

use amdgpu_tile::copy::{make_tiled_copy, make_tiled_copy_a, make_tiled_copy_bcast, LdsPart, LdsSlotPart};
use amdgpu_tile::dma::{issue_rows_shared, partition_dma, DmaPart};
use amdgpu_tile::mma::{gemm, make_tiled_mma, TiledMma};
use amdgpu_tile::{contract, Ctx};
pub mod io;
pub mod graph;
mod skeleton_kvsplit;
mod skeleton_t128;
mod skeleton_vs128;

use io::{combine_kernel, FmhaBlock, FmhaIo};
use amdgpu_tile::frag::{AccTile, Frag, QkOut, B32x32x64Fp8, C32x32F32};
use amdgpu_tile::reduce::KeyMask;
use amdgpu_tile::softmax::{OnlineSoftmax, Recipe};
use amdgpu_tile::*;
use amdgpu_tile::weave::{Hint, Out, UnitClass, Units};
use amdgpu_dsl::{CompileProfile, DType, Ext, FnBuilder, Function, Opnd, StageContract, ValueId, VarId, I32};
use amdgpu_isa::{MmaAtom, DS_READ_B128, DS_READ_B128X2, DS_READ_B32, DS_READ_TR_B8X4, DS_WRITE_B32, GLOBAL_LDS_X1, GLOBAL_LDS_X4, GLOBAL_LOAD_X4, GLOBAL_STORE_X2};
use amdgpu_layout::{make_layout, tup, ComposedLayout, Ix, Layout as LLayout, Swizzle};

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
    /// the K-scale words: lane (j, h) reads word j + 32 s (both lane halves read the same word: a broadcast TV layout)
    pub kdpart: LdsPart,
    pub mma: TiledMma,
    /// MUBUF form of the K / V DMA (`_vs128`): SGPR per-tile advance, loop-invariant voffsets
    pub buf: Option<(amdgpu_tile::dma::BufForm, amdgpu_tile::dma::BufForm)>,
    /// MUBUF form of the K-scale DMA
    pub kd_buf: Option<amdgpu_tile::dma::BufForm>,
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
        let kdpart = make_tiled_copy_bcast(DS_READ_B32, kd_read_tv(), 128, 32).partition_lds(b, &ctx, &l, &kd, &ComposedLayout::plain(make_layout(128, 4)), &[0]);
        let (kt, vt, kdt) = (blk.kv_tile(b, &io.k), blk.kv_tile(b, &io.v), blk.kv_tile(b, &io.ksc));
        let (ksrc, vsrc, kdsrc) = (kt.row_source(b, 1), vt.row_source(b, 1), kdt.row_source(b, 1));
        let wv = ev(roles.wave_in_group);
        let kdma = partition_dma(b, &ctx, GLOBAL_LDS_X4, &l, &w, &wv, 4, ksrc, &ks, &kv_view());
        let vdma = partition_dma(b, &ctx, GLOBAL_LDS_X4, &l, &w, &wv, 4, vsrc, &vs, &kv_view());
        let kdc = if slots == 2 {
            // 128 rows x 4 bytes, one dword per lane: waves 0 / 1 stage rows 0..63 / 64..127, waves 2 / 3 repeat them (the wave index taken mod 2)
            let w2 = w.mod_c(2, &ctx.dyns);
            let wv2 = b.let_i32(ev(roles.wave_in_group) & 1);
            KdCopy3::Dma(partition_dma(b, &ctx, GLOBAL_LDS_X1, &l, &w2, &wv2, 1, kdsrc, &kd, &ComposedLayout::plain(make_layout(tup![128, 4], tup![4, 1]))))
        } else {
            let rt_row = b.bind_i32(lane.tid().lshr(1));
            KdCopy3::Vgpr(amdgpu_tile::rows::DwordRowsVgpr::new(b, kdsrc, &kd, rt_row))
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
    pub fn stage_units<'s>(&'s self, role: Role, key0: &I32, slot: &SlotIdx) -> Option<Units<'s>> {
        match (role, &self.kdc, &self.buf) {
            (Role::A, KdCopy3::Dma(kd), Some((kb, _))) => {
                let kdb = self.kd_buf.as_ref()?;
                let mut u = kd.issue_buffer_units(kdb, key0, &self.kd.at(slot));
                u.append(self.kdma.issue_buffer_units(kb, key0, &self.ks.at(slot)), &[]);
                Some(u)
            }
            (Role::B, _, Some((_, vb))) => Some(self.vdma.issue_buffer_units(vb, key0, &self.vs.at(slot))),
            _ => None,
        }
    }
    /// The copies of the tile at `key0` into `slot` that `role` produces (as `KvRing::stage`).
    pub fn stage(&self, b: &mut FnBuilder, role: Role, key0: &I32, slot: &SlotIdx) -> Option<ValueId> {
        if let Some(u) = self.stage_units(role, key0, slot) {
            // the slot offsets are shared with every reader of the slot instance (SlotIdx cache): declared scalar hand-overs
            let c = StageContract { barrier_phase: StageContract::ANY_PHASE, roles: vec!["group".into()], scalars_in: slot.offset_scalars(), ..Default::default() };
            let sl = slot.clone();
            amdgpu_tile::stage::StageInst::new("move.kv_stage", amdgpu_tile::stage::Prim::M, "kv.stage", c, u)
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
        AttnBody3 { ring, blk, wid, sm, zero16, c127, c123, s_init: Vec::new(), s: None, o: None, p: Vec::new(), kd_pending: None, setprio: false, kscale_sel: false, lazy_bias: false, bias_next: None, biasonce: false, bias_tile: std::cell::Cell::new(None), p_staged: false }
    }
    /// One 32-key block of QK: the f16 bias MFMA, the two K fragments, the K-scale word and its bytes, two scaled MFMAs (the hand order).
    #[allow(clippy::too_many_arguments)]
    fn qk_block(&self, b: &mut FnBuilder, out: &QkOut, bias: Opnd, k: &LdsSlotPart, kd: &LdsSlotPart, s: usize) -> Option<ValueId> {
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
        let kw = kd.read(b, s);
        let q = &blk.q.frags;
        if let (true, QkOut::Vars(t)) = (self.kscale_sel, out) {
            // the vendor form: ONE shift by 8 x lane half, then the scaled MFMA picks byte 0 (k chunk 0) / byte 2 (k chunk 1) with op_sel
            let w = b.bind_i32(ev(kw).lshr(ev(blk.sel.g8)));
            if let (Some(bt), true) = (self.bias_tile.get(), s < 3) {
                b.var_mfma_scale_c(t.accs[s].var, k0, q[0].v, bt, w, blk.sq[0]);
            } else {
                gemm(b, &self.ring.mma, t.accs[s].var, k0, q[0].v, w, blk.sq[0], 0, 0);
            }
            gemm(b, &self.ring.mma, t.accs[s].var, k1, q[1].v, w, blk.sq[1], 2, 0);
            return None;
        }
        let (sk0, sk1) = blk.sel.pair(b, kw);
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
    /// QK into the S variables as emission units: the setup (slot-bound K / K-scale partitions; the one bias MFMA of the `biasonce` form) and one
    /// unit per 32-key block (K fragments, K-scale word, the two scaled MFMAs), then an end marker (no IR).
    fn qk_units<'s>(&'s self, slot: &SlotIdx, s: AccTile<C32x32F32, 4>, bias: Opnd) -> Units<'s> {
        let r = self.ring;
        let mut u = Units::new();
        let parts: Out<(LdsSlotPart<'s>, LdsSlotPart<'s>)> = Out::new();
        let (p2, sl) = (parts.clone(), slot.clone());
        let once = self.biasonce && self.kscale_sel;
        let setup = u.add("qk.setup", UnitClass::Setup, Hint::Anchor, &[], move |b| {
            let k = r.kpart.at(b, &r.ks.at(&sl));
            let kd = r.kdpart.at(b, &r.kd.at(&sl));
            p2.set((k, kd));
            if once {
                // the one bias MFMA writes S tile 3 (zero accumulator: no zero-fragment copies); tiles 0-2 take it as the C input of their first
                // scaled MFMA, tile 3 accumulates in place last
                b.var_mfma_f16_z(s.accs[3].var, self.sm.ones_f16, bias);
                self.bias_tile.set(Some(Opnd::Var(s.accs[3].var)));
            }
        });
        let mut ends = Vec::new();
        for blk in 0..4usize {
            // read unit: (the per-block bias MFMA when there is no shared bias tile,) the two K fragments and the K-scale word; MMA unit: the shift and
            // the two scaled MFMAs
            let (p3, rd) = (parts.clone(), Out::<(ValueId, ValueId, ValueId)>::new());
            let rd2 = rd.clone();
            let r_id = u.add(format!("qk.read{blk}"), UnitClass::LdsRead, Hint::LatencySensitive, &[setup], move |b| {
                p3.with(|(k, kd)| {
                    if self.bias_tile.get().is_none() {
                        b.var_mfma_f16_z(s.accs[blk].var, self.sm.ones_f16, bias);
                    }
                    let k0 = k.read(b, 2 * blk);
                    let k1 = k.read(b, 2 * blk + 1);
                    let kw = kd.read(b, blk);
                    rd2.set((k0, k1, kw));
                });
            });
            ends.push(u.add(format!("qk.mma{blk}"), UnitClass::Mfma, Hint::Skeleton, &[r_id], move |b| {
                let (k0, k1, kw) = rd.get();
                self.qk_block_mma(b, &s, k0, k1, kw, blk);
            }));
        }
        u.add("qk.end", UnitClass::Setup, Hint::Anchor, &ends, move |_b| self.bias_tile.set(None));
        u
    }
    /// PV as emission units: the setup (slot-bound V partition, the two P fragments), the row sums, one unit per (half hh, dim group e).
    fn pv_units<'s>(&'s self, slot: &SlotIdx, w: Vec<ValueId>) -> Units<'s> {
        let r = self.ring;
        let mut u = Units::new();
        let st: Out<(LdsSlotPart<'s>, [ValueId; 2])> = Out::new();
        let (s2, sl) = (st.clone(), slot.clone());
        let setup = u.add("pv.setup", UnitClass::Setup, Hint::Anchor, &[], move |b| {
            let v = r.vpart.at(b, &r.vs.at(&sl));
            let p = [Frag::<B32x32x64Fp8>::from_words(b, &w[0..8]).v, Frag::<B32x32x64Fp8>::from_words(b, &w[8..16]).v];
            s2.set((v, p));
        });
        let s3 = st.clone();
        u.add("pv.rowsum", UnitClass::Mfma, Hint::Filler, &[setup], move |b| {
            let p = s3.with(|(_, p)| *p);
            self.sm.row_sum(b, p[0], p[1]);
        });
        let o = self.o.expect("O");
        for hh in 0..2usize {
            for e in 0..4usize {
                let (s4, s5, vf) = (st.clone(), st.clone(), Out::<ValueId>::new());
                let vf2 = vf.clone();
                let r_id = u.add(format!("pv.read.h{hh}e{e}"), UnitClass::LdsRead, Hint::LatencySensitive, &[setup], move |b| {
                    s4.with(|(v, _)| vf2.set(v.read(b, hh + 2 * e)));
                });
                u.add(format!("pv.mma.h{hh}e{e}"), UnitClass::Mfma, Hint::Skeleton, &[r_id], move |b| {
                    let p = s5.with(|(_, p)| *p);
                    gemm(b, &r.mma, o.accs[e].var, vf.get(), p[hh], self.c127, self.c123, 0, 0);
                });
            }
        }
        u
    }
    /// The MMA half of a QK block into the S variables (the K fragments / scale word already read).
    fn qk_block_mma(&self, b: &mut FnBuilder, t: &AccTile<C32x32F32, 4>, k0: ValueId, k1: ValueId, kw: ValueId, s: usize) {
        let blk = self.blk;
        let q = &blk.q.frags;
        if self.kscale_sel {
            let w = b.bind_i32(ev(kw).lshr(ev(blk.sel.g8)));
            if let (Some(bt), true) = (self.bias_tile.get(), s < 3) {
                b.var_mfma_scale_c(t.accs[s].var, k0, q[0].v, bt, w, blk.sq[0]);
            } else {
                gemm(b, &self.ring.mma, t.accs[s].var, k0, q[0].v, w, blk.sq[0], 0, 0);
            }
            gemm(b, &self.ring.mma, t.accs[s].var, k1, q[1].v, w, blk.sq[1], 2, 0);
            return;
        }
        let (sk0, sk1) = blk.sel.pair(b, kw);
        gemm(b, &self.ring.mma, t.accs[s].var, k0, q[0].v, sk0, blk.sq[0], 0, 0);
        gemm(b, &self.ring.mma, t.accs[s].var, k1, q[1].v, sk1, blk.sq[1], 0, 0);
    }
    fn qk(&self, b: &mut FnBuilder, slot: &SlotIdx, out: QkOut, bias: Opnd) -> Vec<ValueId> {
        if let QkOut::Vars(t) = &out {
            self.qk_units(slot, **t, bias).emit_all(b);
            return Vec::new();
        }
        let r = self.ring;
        let k = r.kpart.at(b, &r.ks.at(slot));
        let kd = r.kdpart.at(b, &r.kd.at(slot));
        if let (true, true, QkOut::Vars(t)) = (self.biasonce, self.kscale_sel, &out) {
            // the one bias MFMA writes S tile 3 (zero accumulator: no zero-fragment copies); tiles 0-2 take it as the C input of their first
            // scaled MFMA, tile 3 accumulates in place last
            b.var_mfma_f16_z(t.accs[3].var, self.sm.ones_f16, bias);
            self.bias_tile.set(Some(Opnd::Var(t.accs[3].var)));
        }
        let res = (0..4).filter_map(|s| self.qk_block(b, &out, bias, &k, &kd, s)).collect();
        self.bias_tile.set(None);
        res
    }
    /// PV with the row sums first: per half hh and dim group e, the transposed V fragment then the scaled MFMA (fixed scales).
    fn pv(&self, b: &mut FnBuilder, slot: &SlotIdx, w: &[ValueId]) {
        self.pv_units(slot, w.to_vec()).emit_all(b);
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
        self.s_init = self.qk(b, &t0.cur, QkOut::Values(self.zero16), Opnd::Val(self.sm.bias_init));
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
            let (nb, pw): (Out<Opnd>, Out<Vec<ValueId>>) = (Out::new(), Out::new());
            {
                let me: &Self = self;
                let (nb2, pw2, m1, m2) = (nb.clone(), pw.clone(), mask.as_ref(), mask.as_ref());
                let mut u = Units::new();
                let d = u.add("softmax.decide", UnitClass::Branch, Hint::Anchor, &[], move |b| nb2.set(me.sm.decide_lazy(b, &s, &o, m1)));
                u.add("softmax.convert", UnitClass::Valu, Hint::Filler, &[d], move |b| pw2.set(me.sm.convert(b, &s, m2, 0, 4)));
                let c = StageContract { barrier_phase: StageContract::ANY_PHASE, roles: vec!["group".into()], ..Default::default() };
                let (nb3, pw3) = (nb.clone(), pw.clone());
                amdgpu_tile::stage::StageInst::new("online.softmax", amdgpu_tile::stage::Prim::O, "v_phase", c, u)
                    .param("recipe", "vendor-lazy")
                    .handover(move |_b| (vec![p_val(pw3.get())], bias_scalar(nb3.get())))
                    .emit(b);
            }
            self.bias_next = Some(nb.get());
            self.p = pw.get();
            self.p_staged = true;
            return;
        }
        let nb = self.sm.decide(b, &s, &o, mask.as_ref());
        self.p = self.sm.convert(b, &s, mask.as_ref(), 0, 4);
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
        let mut insts = vec![amdgpu_tile::stage::StageInst::new("mma.pv", amdgpu_tile::stage::Prim::A, "pv", Default::default(), self.pv_units(&t.cur, w))];
        if t.staged {
            insts.push(amdgpu_tile::stage::StageInst::new("mma.qk", amdgpu_tile::stage::Prim::A, "qk", Default::default(), self.qk_units(&t.next, s, bias)));
        }
        let (cur, next, staged) = (t.cur.clone(), t.next.clone(), t.staged);
        amdgpu_tile::stage::merge("m_phase", c, insts)
            .handover(move |_b| {
                let mut s = cur.offset_scalars();
                if staged {
                    s.extend(next.offset_scalars());
                }
                (Vec::new(), s)
            })
            .emit(b);
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
        // the window: conversion chunk i, then QK block i of the next tile (units, woven in this explicit order)
        let mut u = Units::new();
        let parts: Out<(LdsSlotPart, LdsSlotPart)> = Out::new();
        let setup = t.staged.then(|| {
            let (p2, nx) = (parts.clone(), t.next.clone());
            u.add("window.qk.setup", UnitClass::Setup, Hint::Anchor, &[], move |b| p2.set((r.kpart.at(b, &r.ks.at(&nx)), r.kdpart.at(b, &r.kd.at(&nx)))))
        });
        let chunks: Vec<Out<Vec<ValueId>>> = (0..4).map(|_| Out::new()).collect();
        {
            let me: &Self = self;
            for i in 0..4usize {
                let (c, m) = (chunks[i].clone(), mask.as_ref());
                u.add(format!("window.convert{i}"), UnitClass::Valu, Hint::Filler, &[], move |b| c.set(me.sm.convert(b, &s, m, i, i + 1)));
                if let Some(st) = setup {
                    let p3 = parts.clone();
                    u.add(format!("window.qk.block{i}"), UnitClass::Mfma, Hint::Skeleton, &[st], move |b| {
                        p3.with(|(k, kd)| {
                            me.qk_block(b, &QkOut::Vars(&s), nb, k, kd, i);
                        });
                    });
                }
            }
            // the window = stage `online.softmax` (conversion chunks of tile t) woven with stage `mma.qk` (blocks of tile t + 1), alternating
            let mut c = StageContract { barrier_phase: StageContract::ANY_PHASE, roles: vec!["group".into()], ..Default::default() };
            if t.staged {
                c.scalars_in = t.next.offset_scalars();
            }
            let (nx, staged) = (t.next.clone(), t.staged);
            amdgpu_tile::stage::StageInst::new("online.softmax", amdgpu_tile::stage::Prim::O, "window", c, u)
                .param("with", "mma.qk blocks")
                .handover(move |_b| (Vec::new(), if staged { nx.offset_scalars() } else { Vec::new() }))
                .emit(b);
        }
        let w: Vec<ValueId> = chunks.iter().flat_map(|c| c.get()).collect();
        if !lazy {
            if let Opnd::Val(v) = nb {
                self.sm.commit_bias(b, v);
            }
        }
        self.pv(b, &t.cur, &w);
    }
}

/// `_vs128`: 256-row blocks, two 4-wave groups offset by one phase (`Cadence::TwoPhaseOffset`), MUBUF K / V / K-scale staging, the vendor's
/// priority ping-pong, two-smfmac row sums (`amdgpu_tile::rowsum`), K-scale byte select, the vendor's reference recipe with the lazy reference
/// update from 4 tiles on (`OnlineSoftmax::set_vendor_ref`, `decide_lazy`; shorter sequences keep the exact LUT-corrected recipe: with < 4 tiles the
/// lazy recipe's error does not average out) and the LDS-staged epilogue.
pub fn fmha_fwd_hd128_mxfp8_v3_vs128() -> Function {
    vs128_named("fmha_fwd_hd128_mxfp8_v3_vs128", 0)
}

/// EXPERIMENT: `_vs128` with pipeline ablations (timing bounds only)
pub fn vs128_named(name: &str, ablate: u8) -> Function {
    let mut f = skeleton_vs128::build_vs128(name, &mut Vs128Slots(ablate));
    // scheduler: sink every ds_read to at most one MFMA before its first use (one fragment group in flight, the vendor's lgkmcnt(4);
    // sched/ldahead.rs, docs/lgkm-lookahead.md): +0.6 % B1H8S8192, +1.3 % per-CU on realistic data, VGPR 256 -> 248
    // + wsink (explicit lgkm waits sunk past the compute) and ldahead32=1 (the K-scale ds_read_b32 one MFMA ahead): the two M-phase lgkmcnt(0)
    // become counted waits (sched, docs/lgkm-lookahead.md)
    f.sched_opts = Some("ldsink=1,wsink,ldahead32=1".to_string());
    f
}

/// Values shared between the slots of the FMHA kernels (what the generated skeletons' `Cx` carries).
pub(crate) struct FmhaCx {
    io: FmhaIo,
    blk: FmhaBlock,
    roles: Roles,
    ring: KvRing3,
    pipe: Option<Pipeline>,
    lut: ValueId,
}

/// The slots of `_vs128` (`skeleton_vs128.rs`, generated from `graph::graph_vs128()`).
pub(crate) struct Vs128Slots(u8);

impl skeleton_vs128::Vs128Slots for Vs128Slots {
    type Cx = FmhaCx;
    fn prelude(&mut self, b: &mut FnBuilder) -> FmhaCx {
        b.compile_profile(CompileProfile::Vs128);
        let io = declare_io(b, 256, false);
        let blk = block3_x(&io, b, 8, self.0 & 64 != 0);
        let roles = Roles::split(b, &blk.lane, 4);
        let mut ring = KvRing3::new(b, &io, &blk, &roles, 2);
        ring.enable_buffer_dma(b);
        ring.enable_buffer_kd(b);
        let mut pipe = Pipeline::new(Cadence::TwoPhaseOffset, blk.nkv, blk.kv_len, 128, Ext::ka(0x110).ceil_div(128));
        // every stage reads through a range-checked buffer descriptor: no clamp of the next tile's key
        pipe.key_unclamped = true;
        pipe.ablate = self.0 & 63;
        let lut = blk.lut_copy(b);
        FmhaCx { io, blk, roles, ring, pipe: Some(pipe), lut }
    }
    fn pipeline(&mut self, _b: &mut FnBuilder, cx: &mut FmhaCx) -> Pipeline {
        cx.pipe.take().expect("the pipeline is taken once")
    }
    fn dispatch(&mut self, b: &mut FnBuilder, cx: &mut FmhaCx, pipe: &Pipeline) {
        let (io, blk, roles, ring, lut) = (&cx.io, &cx.blk, &cx.roles, &cx.ring, cx.lut);
        versioned(b, lut, |b, lut| {
            roles.each(b, |b, role| {
                let wid = roles.wave_id(b, role);
                let mut body = AttnBody3::new(b, ring, blk, wid, lut);
                body.kscale_sel = true;
                let p = amdgpu_tile::rowsum::plan();
                body.sm.use_rows2(b, &blk.lane, p.dwords, p.idx);
                if !lut {
                    body.sm.set_vendor_ref(b);
                    body.lazy_bias = true;
                }
                body.setprio = true;
                body.biasonce = true;
                pipe.run(b, role, &mut body);
                let fin = body.out_scale(b);
                store_out3_lds(io, b, blk, &ring.epi, wid, &body.o.expect("O"), fin);
            });
        });
    }
}

/// `_kvsplit`: the `_vs128` structure over a key range per split (two-smfmac row sums, the vendor recipe with the lazy update from 4 tiles of the
/// whole sequence), the partial-row epilogue + log-sum-exp.
pub fn fmha_fwd_hd128_mxfp8_v3_kvsplit() -> Function {
    kvsplit_named("fmha_fwd_hd128_mxfp8_v3_kvsplit", false)
}

/// EXPERIMENT: `_kvsplit` with the XCD-aware block order
pub fn kvsplit_named(name: &str, xcd: bool) -> Function {
    let mut f = skeleton_kvsplit::build_kvsplit(name, &mut KvsplitSlots(xcd));
    // scheduler: sink every ds_read to at most one MFMA before its first use (sched/ldahead.rs, docs/lgkm-lookahead.md): +5.5..+6.8 % on realistic data (B1H8S2048 / B1H4S4096 split 4, B1H4S8192 split
    // + wsink, ldahead32=1 (counted M-phase LDS waits; kvsplit +1.1..+1.8 %, t128 -0.2..+1.5 % on realistic data)
    f.sched_opts = Some("ldsink=1,wsink,ldahead32=1".to_string());
    f
}

/// The slots of `_kvsplit` (`skeleton_kvsplit.rs`, generated from `graph::graph_kvsplit()`).
pub(crate) struct KvsplitSlots(bool);

impl skeleton_kvsplit::KvsplitSlots for KvsplitSlots {
    type Cx = FmhaCx;
    fn prelude(&mut self, b: &mut FnBuilder) -> FmhaCx {
        b.compile_profile(CompileProfile::KvSplit);
        let io = declare_io(b, 256, true);
        let blk = block3_x(&io, b, 8, self.0);
        let roles = Roles::split(b, &blk.lane, 4);
        let ring = KvRing3::new(b, &io, &blk, &roles, 2);
        let pipe = Pipeline::new(Cadence::TwoPhaseOffset, blk.nkv, blk.kv_len, 128, Ext::ka(0x110).ceil_div(128));
        let lut = blk.lut_copy(b);
        FmhaCx { io, blk, roles, ring, pipe: Some(pipe), lut }
    }
    fn pipeline(&mut self, _b: &mut FnBuilder, cx: &mut FmhaCx) -> Pipeline {
        cx.pipe.take().expect("the pipeline is taken once")
    }
    fn dispatch(&mut self, b: &mut FnBuilder, cx: &mut FmhaCx, pipe: &Pipeline) {
        let (io, blk, roles, ring, lut) = (&cx.io, &cx.blk, &cx.roles, &cx.ring, cx.lut);
        versioned(b, lut, |b, lut| {
            roles.each(b, |b, role| {
                let wid = roles.wave_id(b, role);
                let mut body = AttnBody3::new(b, ring, blk, wid, lut);
                let p = amdgpu_tile::rowsum::plan();
                body.sm.use_rows2(b, &blk.lane, p.dwords, p.idx);
                if !lut {
                    body.sm.set_vendor_ref(b);
                    body.lazy_bias = true;
                }
                pipe.run(b, role, &mut body);
                let fin = body.out_scale(b);
                let l = body.sm.l_total(b);
                let lse = body.sm.lse(b, l);
                store_partial3(io, b, blk, &ring.epi, wid, &body.o.expect("O"), fin, lse);
            });
        });
    }
}

pub fn fmha_fwd_hd128_mxfp8_v3_kvsplit_combine8h() -> Function {
    let mut fb = FnBuilder::new("fmha_fwd_hd128_mxfp8_v3_kvsplit_combine8h", 256);
    fb.compile_profile(CompileProfile::Combine);
    combine_kernel(&mut fb, 8);
    fb.finish()
}

/// `_t128`: 128-row blocks, one 4-wave group, a 3-slot ring with one window per tile (`Cadence::OneWindow`); the vendor recipe with the lazy
/// update from 4 tiles.
pub fn fmha_fwd_hd128_mxfp8_v3_t128() -> Function {
    t128_named("fmha_fwd_hd128_mxfp8_v3_t128", false)
}

/// EXPERIMENT: `_t128` with the XCD-aware block order
pub fn t128_named(name: &str, xcd: bool) -> Function {
    let mut f = skeleton_t128::build_t128(name, &mut T128Slots(xcd));
    // scheduler: sink every ds_read to at most one MFMA before its first use (sched/ldahead.rs, docs/lgkm-lookahead.md): +0.4..+0.8 % on realistic data
    // + wsink, ldahead32=1 (counted M-phase LDS waits; kvsplit +1.1..+1.8 %, t128 -0.2..+1.5 % on realistic data)
    f.sched_opts = Some("ldsink=1,wsink,ldahead32=1".to_string());
    f
}

/// The slots of `_t128` (`skeleton_t128.rs`, generated from `graph::graph_t128()`).
pub(crate) struct T128Slots(bool);

impl skeleton_t128::T128Slots for T128Slots {
    type Cx = FmhaCx;
    fn prelude(&mut self, b: &mut FnBuilder) -> FmhaCx {
        b.compile_profile(CompileProfile::T128);
        let io = declare_io(b, 128, false);
        let blk = block3_x(&io, b, 7, self.0);
        let roles = Roles::single(&blk.lane, 4);
        let ring = KvRing3::new(b, &io, &blk, &roles, 3);
        let mut pipe = Pipeline::new(Cadence::OneWindow, blk.nkv, blk.kv_len, 128, Ext::ka(0x110).ceil_div(128));
        pipe.prebind = vec![32768, 512];
        let lut = blk.lut_copy(b);
        FmhaCx { io, blk, roles, ring, pipe: Some(pipe), lut }
    }
    fn pipeline(&mut self, _b: &mut FnBuilder, cx: &mut FmhaCx) -> Pipeline {
        cx.pipe.take().expect("the pipeline is taken once")
    }
    fn dispatch(&mut self, b: &mut FnBuilder, cx: &mut FmhaCx, pipe: &Pipeline) {
        let (io, blk, ring, lut) = (&cx.io, &cx.blk, &cx.ring, cx.lut);
        versioned(b, lut, |b, lut| {
            let mut body = AttnBody3::new(b, ring, blk, blk.lane.wave, lut);
            if !lut {
                body.sm.set_vendor_ref(b);
                body.lazy_bias = true;
            }
            pipe.run(b, Role::All, &mut body);
            let fin = body.out_scale(b);
            store_out3(io, b, blk, &body.o.expect("O"), fin);
        });
    }
}

/// The 656-byte FMHA kernarg ABI (+ the KV-split extension) with the host contract DERIVED from the tensors' layouts (`contract::compact_tensor` /
/// `same_strides`: allocation = cosize, compactness relations); the kernarg loads and the runtime tensors are those of `FmhaIo::declare`.
pub fn declare_io(b: &mut FnBuilder, rows_per_block: i64, kvsplit: bool) -> FmhaIo {
    let ka = &mut KArgs::new();
    let gqa = ka.i32(b, "s_gqa", 0x0B0);
    b.ka_range(0x0B0, 1, 16);
    b.ka_is(0x140, Ext::sym("HK") * Ext::ka(0x0B0));
    let s = ka.dim(b, "s_seq_len", 0x060, 1, i64::MAX);
    let skv = ka.dim(b, "s_kv_seq_len", 0x110, 1, i64::MAX);
    let heads = ka.i32(b, "s_q_head_num", 0x140);
    let h = Dim::Ka { off: 0x140, val: heads };
    let (hk, d) = (Dim::sym("HK"), Dim::c(128));
    let bat = if kvsplit { Dim::sym("B") } else { Dim::grid(2) };
    let bshd = |ka: &mut KArgs, b: &mut FnBuilder, name: &str, off: u32, eb: i64, n: &Dim, hh: &Dim, st: [(&'static str, u32); 3]| {
        ka.tensor(name, off, eb, vec![bat.clone(), n.clone(), hh.clone(), d.clone()], vec![Stride::Ka(st[0].0, st[0].1), Stride::Ka(st[1].0, st[1].1), Stride::Ka(st[2].0, st[2].1), Stride::Unit])
            .external_contract()
            .declare(ka, b)
    };
    let q = bshd(ka, b, "ptr_q", 0x010, 1, &s, &h, [("s_Bs", 0x0A0), ("s_Seqs", 0x070), ("s_Hs", 0x090)]);
    let k = bshd(ka, b, "ptr_k", 0x020, 1, &skv, &hk, [("s_k_Bs", 0x0E0), ("s_k_Seqs", 0x0C0), ("s_k_Hs", 0x0D0)]);
    let v = bshd(ka, b, "ptr_v", 0x030, 1, &skv, &hk, [("s_v_Bs", 0x170), ("s_v_Seqs", 0x150), ("s_v_Hs", 0x160)]);
    let o = bshd(ka, b, "ptr_o", 0x000, 2, &s, &h, [("s_o_Bs", 0x1A0), ("s_o_Seqs", 0x180), ("s_o_Hs", 0x190)]);
    let qd_stride = ev(b.bind_i32(q.strides[1].clone().lshr(5)));
    let kd_stride = ev(b.bind_i32(k.strides[1].clone().lshr(5)));
    let scale = |ka: &mut KArgs, b: &mut FnBuilder, name: &str, off: u32, n: &Dim, hh: &Dim, bs: (&'static str, u32), der: I32, hs: (&'static str, u32)| {
        ka.tensor(name, off, 1, vec![bat.clone(), n.clone(), hh.clone(), Dim::c(4)], vec![Stride::Ka(bs.0, bs.1), Stride::Derived(der), Stride::Ka(hs.0, hs.1), Stride::Unit])
            .external_contract()
            .declare(ka, b)
    };
    let qsc = scale(ka, b, "ptr_q_descale", 0x200, &s, &h, ("s_descale_q_Bs", 0x230), qd_stride, ("s_descale_q_Hs", 0x240));
    let ksc = scale(ka, b, "ptr_k_descale", 0x210, &skv, &hk, ("s_descale_k_Bs", 0x250), kd_stride, ("s_descale_k_Hs", 0x260));
    // the contract, from the layouts (bytes)
    let mut ctx = Ctx::new();
    let big = i64::MAX / 4;
    let cb = if kvsplit { ctx.sym_ext("B", 1, big, None, Ext::sym("B")) } else { ctx.sym_ext("grid_z", 1, big, None, Ext::grid_z()) };
    let kas = |ctx: &mut Ctx, name: &str, off: u32| ctx.sym_ext(name, 0, big, None, Ext::ka(off));
    let (cs, cskv, ch) = (kas(&mut ctx, "S", 0x060), kas(&mut ctx, "SK", 0x110), kas(&mut ctx, "H", 0x140));
    let chk = ctx.sym_ext("HK", 1, big, None, Ext::sym("HK"));
    let lay = |sh: [&Ix; 4], st: [Ix; 4]| amdgpu_layout::make_layout(tup![sh[0].clone(), sh[1].clone(), sh[2].clone(), sh[3].clone()], tup![st[0].clone(), st[1].clone(), st[2].clone(), st[3].clone()]);
    let c = |k: i64| Ix::C(k);
    let ql = lay([&cb, &cs, &ch, &c(128)], [kas(&mut ctx, "s_Bs", 0x0A0), kas(&mut ctx, "s_Seqs", 0x070), c(128), c(1)]);
    contract::compact_tensor(b, &ctx, "ptr_q", &ql, &[Some(0x0A0), Some(0x070), Some(0x090), None], None);
    let kl = lay([&cb, &cskv, &chk, &c(128)], [kas(&mut ctx, "s_k_Bs", 0x0E0), kas(&mut ctx, "s_k_Seqs", 0x0C0), c(128), c(1)]);
    let k_ka = [Some(0x0E0), Some(0x0C0), Some(0x0D0), None];
    contract::compact_tensor(b, &ctx, "ptr_k", &kl, &k_ka, None);
    let vl = lay([&cb, &cskv, &chk, &c(128)], [kas(&mut ctx, "s_v_Bs", 0x170), kas(&mut ctx, "s_v_Seqs", 0x150), c(128), c(1)]);
    contract::same_strides(b, &ctx, "ptr_v", &vl, &[Some(0x170), Some(0x150), Some(0x160), None], &kl, &k_ka);
    let ol = lay([&cb, &cs, &ch, &c(128)], [kas(&mut ctx, "s_o_Bs", 0x1A0), kas(&mut ctx, "s_o_Seqs", 0x180), c(256), c(2)]);
    contract::compact_tensor(b, &ctx, "ptr_o", &ol, &[Some(0x1A0), Some(0x180), Some(0x190), None], None);
    let qsl = lay([&cb, &cs, &ch, &c(4)], [kas(&mut ctx, "s_descale_q_Bs", 0x230), ch.mul_c(4), c(4), c(1)]);
    contract::compact_tensor(b, &ctx, "ptr_q_descale", &qsl, &[Some(0x230), None, Some(0x240), None], Some(Ext::c(512 * 4) * Ext::ka(0x140)));
    let ksl = lay([&cb, &cskv, &chk, &c(4)], [kas(&mut ctx, "s_descale_k_Bs", 0x250), chk.mul_c(4), c(4), c(1)]);
    contract::compact_tensor(b, &ctx, "ptr_k_descale", &ksl, &[Some(0x250), None, Some(0x260), None], Some(Ext::c(640 * 4) * Ext::ka(0x140)));
    let ptr_vd = ka.ptr(b, "ptr_v_descale", 0x220);
    b.arg_tensor("ptr_v_descale", Ext::c(4096));
    b.grid_is(1, Ext::ka(0x140));
    b.grid_is(0, Ext::ka(0x060).ceil_div(rows_per_block));
    let kvs = if kvsplit {
        let splits = ka.dim(b, "kvs_splits", 0x280, 1, i64::MAX).val().expect("splits");
        let rows_total = ka.dim(b, "kvs_rows_total", 0x284, 1, i64::MAX).val().expect("rows");
        b.grid_is(2, Ext::sym("B") * Ext::ka(0x280));
        b.ka_is(0x284, Ext::sym("B") * Ext::ka(0x060) * Ext::ka(0x140));
        let ohat = ka.ptr(b, "ptr_ohat", 0x270);
        let lse = ka.ptr(b, "ptr_lse", 0x278);
        b.arg_tensor("ptr_ohat", Ext::ka(0x280) * Ext::ka(0x284) * Ext::c(128 * 4));
        b.arg_tensor("ptr_lse", Ext::ka(0x280) * Ext::ka(0x284) * Ext::c(4));
        Some(io::KvsArgs { ohat, lse, splits: b.bind_i32(splits), rows_total: b.bind_i32(rows_total) })
    } else {
        ka.arg(b, "abi_tail", DType::I32, 0x280, 16);
        None
    };
    FmhaIo { q, k, v, o, qsc, ksc, ptr_vd, gqa, heads, s_len: s.val().expect("S"), kv_len: skv.val().expect("SK"), kvs }
}

/// `FmhaIo::block` with the Q fragments loaded through a TiledCopy: the B operand of the scaled 32x32x64 atom over the lane's (clamped) query row, two
/// k-groups of 64 bytes (placements 0 / 64): one pointer (row + 16 (lane >> 5)) and four `global_load_dwordx4` with immediates 0 / 32 / 64 / 96.
pub fn block3(io: &FmhaIo, b: &mut FnBuilder, rows_lg: i32) -> FmhaBlock {
    block3_x(io, b, rows_lg, false)
}

/// `xcd`: the vendor's XCD-aware block order. Workgroups are dispatched round robin over the 8 XCDs (linear id % 8), each with its own L2;
/// with the plain order the 8 query tiles of one head land on 8 different XCDs and every XCD streams the same K / V from memory. The remap
/// gives XCD x the contiguous logical range [x N / 8, (x + 1) N / 8) of the (query tile, head) plane of the batch (N = tiles x heads), so the
/// blocks that share K / V share an L2. Applied when N % 8 == 0 and H <= 32 (the vendor's conditions; the head is recovered by at most 32
/// compare-subtract steps), the plain order otherwise.
pub fn block3_x(io: &FmhaIo, b: &mut FnBuilder, rows_lg: i32, xcd: bool) -> FmhaBlock {
    let (qtile, hidx) = if xcd {
        let x0u = b.let_(amdgpu_dsl::Op::WorkgroupIdX);
        let y0u = b.let_(amdgpu_dsl::Op::WorkgroupIdY);
        let x0 = b.let_(amdgpu_dsl::Op::AddI32(amdgpu_dsl::Opnd::Val(x0u), amdgpu_dsl::Opnd::I32(0)));
        let y0 = b.let_(amdgpu_dsl::Op::AddI32(amdgpu_dsl::Opnd::Val(y0u), amdgpu_dsl::Opnd::I32(0)));
        let hh = b.let_(amdgpu_dsl::Op::KernargI32 { offset: 0x140 });
        let xq = b.bind_i32((io.s_len.clone() + ((1 << rows_lg) - 1)).lshr(rows_lg));
        let n = b.bind_i32(ev(xq) * ev(hh));
        // ok = (N % 8 == 0) && (H <= 32), as 0 / 1
        // + X <= 16: with long sequences (X = 32, B8H8S8192) the order measured -6.9 % (all 32 CUs of an XCD on one head in lockstep)
        let ok = b.bind_i32((ev(n) & 7).eq(0).select_i32(ev(hh).le(32).select_i32(ev(xq).le(16).select_i32(1, 0), 0), 0));
        let l = b.bind_i32(ev(y0) * ev(xq) + ev(x0));
        let mut r = b.bind_i32((ev(l) & 7) * ev(n).lshr(3) + ev(l).lshr(3));
        // head = M / X by binary long division (H <= 32: 5 steps)
        let mut y = b.c_i32(0);
        for bit in [16i32, 8, 4, 2, 1] {
            let xb = b.bind_i32(ev(xq) * bit);
            let take = b.bind_i32(ev(r).ge(ev(xb)).select_i32(1, 0));
            y = b.bind_i32(ev(y) + ev(take) * bit);
            r = b.bind_i32(ev(r) - ev(take) * ev(xb));
        }
        // clamps (no-ops for N % 8 == 0, H <= 32) bound the logical ids for the verifier
        let rc = b.bind_i32(ev(r).max(0).min(ev(xq) - 1));
        let yc = b.bind_i32(ev(y).max(0).min(ev(hh) - 1));
        // wave-uniform: back to SGPRs like the workgroup ids (otherwise the address math downstream turns vector)
        let qt = b.bind_i32(ev(ok).eq(1).select_i32(ev(rc), ev(x0)));
        let hi = b.bind_i32(ev(ok).eq(1).select_i32(ev(yc), ev(y0)));
        let qt = b.read_first_lane(qt);
        let hi = b.read_first_lane(hi);
        let qt = b.let_(amdgpu_dsl::Op::AddI32(amdgpu_dsl::Opnd::Val(qt), amdgpu_dsl::Opnd::I32(0)));
        let hi = b.let_(amdgpu_dsl::Op::AddI32(amdgpu_dsl::Opnd::Val(hi), amdgpu_dsl::Opnd::I32(0)));
        // the remap is a bijection of [0, X) x [0, H) (N % 8 == 0); the interval engine cannot follow the division: trusted contract facts
        b.assume_trusted(qt, Ext::c(0), Ext::ka(0x060).ceil_div(1 << rows_lg) - Ext::c(1), "XCD block order: the logical query tile M % X of the bijection M = (L % 8) N / 8 + L / 8 over [0, X H) lies in [0, X)");
        b.assume_trusted(hi, Ext::c(0), Ext::ka(0x140) - Ext::c(1), "XCD block order: the logical head M / X of the bijection over [0, X H) lies in [0, H)");
        (qt, hi)
    } else {
        (b.let_(amdgpu_dsl::Op::WorkgroupIdX), b.let_(amdgpu_dsl::Op::WorkgroupIdY))
    };
    let zz = b.let_(amdgpu_dsl::Op::WorkgroupIdZ);
    let lane = Lane::new(b);
    let kv_all = b.bind_i32(io.kv_len.clone());
    let ntiles_total = b.bind_i32((ev(kv_all) + 127).lshr(7));
    let (bidx, kv_len, key_lo, split) = match &io.kvs {
        None => (zz, kv_all, I32::c(0), None),
        Some(kv) => {
            let (bi, si, lg) = components::kv_split_decode(b, zz, kv.splits, 0x280, "B");
            let (key_lo, kv_eff) = components::kv_split_range(b, kv_all, si, lg, 0x110);
            (bi, kv_eff, ev(key_lo), Some(si))
        }
    };
    let row0 = b.let_i32((ev(qtile) << rows_lg) + (lane.wave() << 5));
    let row = b.let_i32(&row0 + lane.j());
    let s_last = b.let_i32(&io.s_len - 1);
    let row_c = b.let_i32(row.lt(io.s_len.clone()).select_i32(row.clone(), s_last.clone()));
    let origin = vec![ev(bidx), I32::c(0), ev(hidx), I32::c(0)];
    let qt = io.q.tile(origin.clone()).clamp(1, io.s_len.clone(), s_last.clone());
    let q_row = b.let_i32(qt.offset(&[I32::c(0), row.clone(), I32::c(0), I32::c(0)]));
    let mut ctx = Ctx::new();
    let l = ctx.sym("lane", 0, 63, lane.lane());
    let mma = make_tiled_mma(attention_orientation().0, false);
    let qcp = amdgpu_tile::copy::make_tiled_copy_b(GLOBAL_LOAD_X4, &mma);
    // the row is the lane's (clamped) query row, supplied above: the view gives the byte inside it (k), row stride 0
    let qpart = qcp.partition_global(b, &ctx, &l, &make_layout(tup![32, 64], tup![0, 1]), &[0, 64]);
    let ptrs = qpart.ptrs(b, io.q.ptr, &q_row);
    let mut frags = Vec::with_capacity(2);
    let mut words = [[0 as ValueId; 8]; 2];
    for c in 0..2usize {
        let mut w: Vec<ValueId> = Vec::with_capacity(8);
        for f4 in qpart.load(b, &ptrs, c) {
            for k in 0..4u8 {
                let lv = b.frag4_lane(f4, k);
                w.push(b.bitcast_i32(lv));
            }
        }
        words[c].copy_from_slice(&w);
        frags.push(Frag::<B32x32x64Fp8>::from_words(b, &w));
    }
    let q = frag::RowFrags { frags: [frags[0], frags[1]], words };
    let qdp = io.qsc.tile(origin).clamp(1, io.s_len.clone(), s_last.clone()).ptr(b, &[I32::c(0), row.clone(), I32::c(0), I32::c(0)]);
    let qd_raw = b.load_f32(qdp, 0);
    let qd = b.bitcast_i32(qd_raw);
    let sel = frag::ScaleSel::new(b, &lane);
    let (sq0, sq1) = sel.pair(b, qd);
    let sq = [b.bind_i32(ev(sq0) - 9), b.bind_i32(ev(sq1) - 9)];
    let vd = b.load_f32(io.ptr_vd, 0);
    let kvh = components::gqa_kv_head(b, hidx, io.gqa, "HK");
    let nkv = b.bind_i32((ev(kv_len) + 127).lshr(7));
    b.assume_trusted(nkv, Ext::c(1), Ext::ka(0x110).ceil_div(128), "nkv = (kv_len + 127) >> 7 with 1 <= kv_len <= SK; the engine floors the lower bound of a shift to 0");
    let g4 = b.bind_i32(C32x32F32::half_base(&lane));
    FmhaBlock { lane, bidx, hidx, kvh, row0, row_c, s_last, kv_len, key_lo, nkv, ntiles_total, split, q, sq, sel, vd, g4 }
}

/// The bf16 epilogue through a TiledCopy: the C layout of the scaled 32x32x64 atom (O^T: rows = dims, columns = queries) stored with
/// `global_store_dwordx2` (4 bf16 per instruction) into the lane's (clamped) output row: one pointer (row + 8 (lane >> 5)) and the immediates
/// `64 e + 16 q` - the shipped instruction stream, under `if row0 < S`.
pub fn store_out3(io: &FmhaIo, b: &mut FnBuilder, blk: &FmhaBlock, o: &AccTile<C32x32F32, 4>, fin: ValueId) {
    let o_tile = io.o.tile(vec![ev(blk.bidx), I32::c(0), ev(blk.hidx), I32::c(0)]).clamp(1, io.s_len.clone(), blk.s_last.clone());
    let row = b.let_i32(o_tile.offset(&[I32::c(0), &blk.row0 + blk.lane.j(), I32::c(0), I32::c(0)]));
    let live = b.bind_cond(blk.row0.lt(io.s_len.clone()));
    let mut ctx = Ctx::new();
    let l = ctx.sym("lane", 0, 63, blk.lane.lane());
    let mma = make_tiled_mma(attention_orientation().0, false);
    let st = make_tiled_copy(GLOBAL_STORE_X2, mma.c_tv(), 32 * 32, 16);
    let part = st.partition_global(b, &ctx, &l, &make_layout(tup![32, 32], tup![2, 0]), &[0, 64, 128, 192]);
    b.begin_if(Opnd::Val(live));
    let ptrs = part.ptrs(b, io.o.ptr, &row);
    for e in 0..4usize {
        let mut vals = Vec::with_capacity(4);
        for q in 0..4usize {
            let mut v4: Vec<ValueId> = Vec::with_capacity(4);
            for k in 0..4usize {
                let lv = b.frag16_lane(Opnd::Var(o.accs[e].var), (4 * q + k) as u8);
                let f = b.bitcast_f32(lv);
                v4.push(b.mul_f32(f, fin));
            }
            let w0 = b.cvt_pk_bf16_f32(v4[0], v4[1]);
            let w1 = b.cvt_pk_bf16_f32(v4[2], v4[3]);
            vals.push(b.frag2(w0, w1));
        }
        part.store(b, &ptrs, e, &vals);
    }
    b.end_if();
}

/// The bf16 epilogue through the wave-private scratch (the `store_partial3` transpose with bf16 words): per 64-dim half, the C layout is written as
/// 2 x `ds_write_b32` per 4 dims into a [32 queries x 64 dims] slice (16-byte chunks XOR-swizzled by the query), read back as 16-byte row chunks
/// (`ds_read_b128`, 8 lanes per row) and stored as 4 x 16-byte segments of the output rows: 8 `global_store_dwordx4` per wave (the vendor count)
/// instead of 16 `global_store_dwordx2`.
pub fn store_out3_lds(io: &FmhaIo, b: &mut FnBuilder, blk: &FmhaBlock, epi: &Shared, wid: ValueId, o: &AccTile<C32x32F32, 4>, fin: ValueId) {
    b.barrier();
    let o_tile = io.o.tile(vec![ev(blk.bidx), I32::c(0), ev(blk.hidx), I32::c(0)]).clamp(1, io.s_len.clone(), blk.s_last.clone());
    let live = b.bind_cond(blk.row0.lt(io.s_len.clone()));
    let lane = blk.lane;
    let mut ctx = Ctx::new();
    let l = ctx.sym("lane", 0, 63, lane.lane());
    let view = ComposedLayout::new(Swizzle::cute(3, 4, 3), make_layout(tup![64, 32], tup![2, 128]));
    let wtv = make_layout(tup![tup![32, 2], tup![4, 4, 2]], tup![tup![64, 4], tup![1, 8, 32]]);
    let epi0 = Shared { base: 0, ..epi.clone() };
    let wpart = make_tiled_copy(DS_WRITE_B32, wtv, 64 * 32, 16).partition_lds(b, &ctx, &l, &epi0, &view, &[0]);
    let rtv = make_layout(tup![tup![8, 8], tup![8, 4]], tup![tup![8, 64], tup![1, 512]]);
    let rpart = make_tiled_copy(DS_READ_B128, rtv, 64 * 32, 16).partition_lds(b, &ctx, &l, &epi0, &view, &[0]);
    b.begin_if(Opnd::Val(live));
    let me = epi.mine(wid);
    let wp = wpart.at_private(b, &me);
    let rp = rpart.at_private(b, &me);
    let gptr: Vec<ValueId> = (0..4)
        .map(|i| {
            // round i: row 8 i + (lane >> 3) of the wave's 32 queries (clamped by the tile), chunk lane & 7 (8 dims = 16 bytes)
            let r = b.let_i32((&blk.row0 + 8 * i as i32) + lane.lane().lshr(3));
            let off = b.bind_i32(o_tile.offset(&[I32::c(0), r, I32::c(0), I32::c(0)]) + ((lane.lane() & 7) << 4));
            b.ptr_add(io.o.ptr, off)
        })
        .collect();
    for pr in 0..2usize {
        for ec in 0..2usize {
            let e = 2 * pr + ec;
            for q in 0..4usize {
                let mut v4: Vec<ValueId> = Vec::with_capacity(4);
                for k in 0..4usize {
                    let lv = b.frag16_lane(Opnd::Var(o.accs[e].var), (4 * q + k) as u8);
                    let f = b.bitcast_f32(lv);
                    v4.push(b.mul_f32(f, fin));
                }
                let w0 = b.cvt_pk_bf16_f32(v4[0], v4[1]);
                let w1 = b.cvt_pk_bf16_f32(v4[2], v4[3]);
                let n = 2 * (4 * ec + q);
                wp.write(b, n, 0, &[w0]);
                wp.write(b, n + 1, 0, &[w1]);
            }
        }
        b.wait_lds(0);
        for (i, g) in gptr.iter().enumerate() {
            let v = rp.read(b, i, 0);
            b.store_f32x4(*g, (128 * pr) as u32, v);
        }
    }
    b.end_if();
}

/// The KV-split partial epilogue with the private transpose scratch on TiledCopies: the C layout written as 2 x `ds_write_b32` per 4 dims into a
/// [32 queries x 64 dims] fp16 slice with the 16-byte chunks XOR-swizzled by the query (`Swizzle<3,4,3>`), read back as rows of 16 bytes
/// (`ds_read_b128`, 8 lanes per row) and stored as 16-byte segments of the partial rows (as `FmhaIo::store_partial`).
pub fn store_partial3(io: &FmhaIo, b: &mut FnBuilder, blk: &FmhaBlock, epi: &Shared, wid: ValueId, o: &AccTile<C32x32F32, 4>, fin: ValueId, lse: ValueId) {
    let kv = io.kvs.as_ref().expect("store_partial3: not a KV-split launch");
    b.barrier();
    let part_row0 = b.bind_i32(ev(blk.split.expect("split")) * ev(kv.rows_total));
    let prow = |row: I32| ev(part_row0) + (((ev(blk.bidx) * io.s_len.clone()) + row) * ev(io.heads) + ev(blk.hidx));
    let lse_ptr = {
        let off = b.bind_i32(prow(blk.row_c.clone()) << 2);
        b.ptr_add(kv.lse, off)
    };
    let live = b.bind_cond(blk.row0.lt(io.s_len.clone()));
    let lane = blk.lane;
    let mut ctx = Ctx::new();
    let l = ctx.sym("lane", 0, 63, lane.lane());
    // the per-wave slice [64 dims x 32 queries] fp16 (element dim + 64 query), chunks swizzled by the query
    let view = ComposedLayout::new(Swizzle::cute(3, 4, 3), make_layout(tup![64, 32], tup![2, 128]));
    // R2S: thread (query j = lane % 32: element +64, half: dim +4) x value (dim +1 x4, +8 x4, column pair ec: +32)
    let wtv = make_layout(tup![tup![32, 2], tup![4, 4, 2]], tup![tup![64, 4], tup![1, 8, 32]]);
    let epi0 = Shared { base: 0, ..epi.clone() };
    let wpart = make_tiled_copy(DS_WRITE_B32, wtv, 64 * 32, 16).partition_lds(b, &ctx, &l, &epi0, &view, &[0]);
    // S2R: thread (chunk = lane % 8: dim +8, row = lane / 8: query +1) x value (8 dims, round i: query +8)
    let rtv = make_layout(tup![tup![8, 8], tup![8, 4]], tup![tup![8, 64], tup![1, 512]]);
    let rpart = make_tiled_copy(DS_READ_B128, rtv, 64 * 32, 16).partition_lds(b, &ctx, &l, &epi0, &view, &[0]);
    b.begin_if(Opnd::Val(live));
    b.store_f32(lse_ptr, 0, lse);
    let me = epi.mine(wid);
    let wp = wpart.at_private(b, &me);
    let rp = rpart.at_private(b, &me);
    let gptr: Vec<ValueId> = (0..4)
        .map(|i| {
            // round i: the lane reads back 16 bytes of row 8 i + (lane >> 3) (clamped), chunk lane & 7, of the 256-byte fp16 partial row
            let r = b.let_i32((&blk.row0 + 8 * i as i32) + lane.lane().lshr(3));
            let rg = r.lt(io.s_len.clone()).select_i32(r.clone(), blk.s_last.clone());
            let off = b.bind_i32((prow(rg) << 8) + ((lane.lane() & 7) << 4));
            b.ptr_add(kv.ohat, off)
        })
        .collect();
    let c16 = b.c_i32(16);
    for pr in 0..2usize {
        for ec in 0..2usize {
            let e = 2 * pr + ec;
            for q in 0..4usize {
                let mut hv: Vec<ValueId> = Vec::with_capacity(4);
                for k in 0..4usize {
                    let lv = b.frag16_lane(Opnd::Var(o.accs[e].var), (4 * q + k) as u8);
                    let f = b.bitcast_f32(lv);
                    let v0 = b.mul_f32(f, fin);
                    hv.push(b.let_(amdgpu_dsl::Op::CvtF32ToF16(Opnd::Val(v0))));
                }
                let s0 = b.shl_i32(hv[1], c16);
                let s1 = b.shl_i32(hv[3], c16);
                let w0 = b.let_(amdgpu_dsl::Op::AddI32(Opnd::Val(hv[0]), Opnd::Val(s0)));
                let w1 = b.let_(amdgpu_dsl::Op::AddI32(Opnd::Val(hv[2]), Opnd::Val(s1)));
                // instructions 2 (4 ec + q) and + 1 of the write partition
                let n = 2 * (4 * ec + q);
                wp.write(b, n, 0, &[w0]);
                wp.write(b, n + 1, 0, &[w1]);
            }
        }
        b.wait_lds(0);
        for (i, g) in gptr.iter().enumerate() {
            let v = rp.read(b, i, 0);
            b.store_f32x4(*g, (128 * pr) as u32, v);
        }
        b.wait_lds(0);
    }
    b.end_if();
}
