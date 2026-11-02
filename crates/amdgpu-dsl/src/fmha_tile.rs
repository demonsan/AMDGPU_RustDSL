//! FMHA forward kernels written on the tile library (`crate::tile`, docs/dsl-tile-abstractions.md): ports of the shipped hand-written kernels
//! (`kernels.rs`: `fmha_t_build` + `emit_fmha_vs128_role`, ~1,600 lines), default recipe only (no `FMHA_OPT` experiments).
//!
//! Layout of this file: `FmhaIo` / `FmhaBlock` = the FMHA kernarg ABI and the per-block setup shared by every FMHA kernel (tensors + contract, block ids,
//! GQA head, key range, resident Q fragments); `AttnBody` = the per-tile work of the transposed attention (`S^T = K Q^T`, online softmax, `O^T += V^T P`)
//! as a `PipeBody`; then one short builder per kernel.

use crate::tile::epilogue::{combine_splits, kvsplit_partial_f16, store_bf16};
use crate::tile::frag::*;
use crate::tile::reduce::KeyMask;
use crate::tile::softmax::{OnlineSoftmax, Recipe};
use crate::tile::*;
use crate::{CompileProfile, DType, Ext, FnBuilder, Function, Op, Opnd, ValueId, VarId, I32};

// ------------------------------------------------------------------------------------------------------------------------------------------------------
// ABI + per-block setup (shared by the FMHA kernels)
// ------------------------------------------------------------------------------------------------------------------------------------------------------

/// The 656-byte vendor FMHA kernarg ABI as tensors: Q/K/V (fp8, BSHD), O (bf16), the E8M0 Q/K descales and the V descale; declaring them emits the host
/// contract (`arg_tensor`, `ka_is`, `ka_range`) the GPU-semantics verifier proves every access against.
pub struct FmhaIo {
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
    pub o: Tensor,
    pub qsc: Tensor,
    pub ksc: Tensor,
    pub ptr_vd: ValueId,
    pub gqa: ValueId,
    pub s_len: I32,
    pub kv_len: I32,
    /// KV-split launch (phase 1): the partial outputs and the split count
    pub kvs: Option<KvsArgs>,
}

/// KV-split kernarg extension (0x270 Ohat, 0x278 lse, 0x280 splits, 0x284 rows_total = B*S*H; grid z = B * splits).
pub struct KvsArgs {
    pub ohat: ValueId,
    pub lse: ValueId,
    pub splits: ValueId,
    pub rows_total: ValueId,
}

impl FmhaIo {
    pub fn declare(b: &mut FnBuilder, ka: &mut KArgs, rows_per_block: i64, kvsplit: bool) -> FmhaIo {
        // GQA (H = HK * gqa, gqa a power of two <= 16) and the grid; the tensor contracts below are stated in terms of H
        let gqa = ka.i32(b, "s_gqa", 0x0B0);
        b.ka_range(0x0B0, 1, 16);
        b.ka_is(0x140, Ext::sym("HK") * Ext::ka(0x0B0));
        let s = ka.dim(b, "s_seq_len", 0x060, 1, i64::MAX);
        let skv = ka.dim(b, "s_kv_seq_len", 0x110, 1, i64::MAX);
        let h = Dim::Ka { off: 0x140, val: ka.i32(b, "s_q_head_num", 0x140) };
        let (hk, d) = (Dim::sym("HK"), Dim::c(128));
        // the batch: grid z for the single-kernel launch, the symbol B with grid z = B * splits for the KV-split launch
        let bat = if kvsplit { Dim::sym("B") } else { Dim::grid(2) };
        let bshd = |ka: &mut KArgs, b: &mut FnBuilder, name: &str, off: u32, eb: i64, n: &Dim, hh: &Dim, st: [(&'static str, u32); 3]| {
            ka.tensor(name, off, eb, vec![bat.clone(), n.clone(), hh.clone(), d.clone()], vec![Stride::Ka(st[0].0, st[0].1), Stride::Ka(st[1].0, st[1].1), Stride::Ka(st[2].0, st[2].1), Stride::Unit])
        };
        let q = bshd(ka, b, "ptr_q", 0x010, 1, &s, &h, [("s_Bs", 0x0A0), ("s_Seqs", 0x070), ("s_Hs", 0x090)]).packed().declare(ka, b);
        let k = bshd(ka, b, "ptr_k", 0x020, 1, &skv, &hk, [("s_k_Bs", 0x0E0), ("s_k_Seqs", 0x0C0), ("s_k_Hs", 0x0D0)]).packed().declare(ka, b);
        let v = bshd(ka, b, "ptr_v", 0x030, 1, &skv, &hk, [("s_v_Bs", 0x170), ("s_v_Seqs", 0x150), ("s_v_Hs", 0x160)]).same_strides_as(&k).declare(ka, b);
        let o = bshd(ka, b, "ptr_o", 0x000, 2, &s, &h, [("s_o_Bs", 0x1A0), ("s_o_Seqs", 0x180), ("s_o_Hs", 0x190)]).packed().declare(ka, b);
        // E8M0 descales [B, S, H, 4] / [B, SK, HK, 4] (row stride = seq stride / 32) with zeroed slack rows behind them (read by unguarded gathers)
        let qd_stride = ev(b.bind_i32(q.strides[1].clone().lshr(5)));
        let kd_stride = ev(b.bind_i32(k.strides[1].clone().lshr(5)));
        let qsc = ka
            .tensor("ptr_q_descale", 0x200, 1, vec![bat.clone(), s.clone(), h.clone(), Dim::c(4)], vec![Stride::Ka("s_descale_q_Bs", 0x230), Stride::Derived(qd_stride), Stride::Ka("s_descale_q_Hs", 0x240), Stride::Unit])
            .packed()
            .slack(Ext::c(512 * 4) * Ext::ka(0x140))
            .declare(ka, b);
        let ksc = ka
            .tensor("ptr_k_descale", 0x210, 1, vec![bat, skv.clone(), hk, Dim::c(4)], vec![Stride::Ka("s_descale_k_Bs", 0x250), Stride::Derived(kd_stride), Stride::Ka("s_descale_k_Hs", 0x260), Stride::Unit])
            .packed()
            .slack(Ext::c(640 * 4) * Ext::ka(0x140))
            .declare(ka, b);
        let ptr_vd = ka.ptr(b, "ptr_v_descale", 0x220);
        b.arg_tensor("ptr_v_descale", Ext::c(4096));
        b.grid_is(1, Ext::ka(0x140));
        b.grid_is(0, Ext::ka(0x060).ceil_div(rows_per_block));
        let kvs = if kvsplit {
            let splits = ka.dim(b, "kvs_splits", 0x280, 1, i64::MAX).val().unwrap();
            let rows_total = ka.dim(b, "kvs_rows_total", 0x284, 1, i64::MAX).val().unwrap();
            b.grid_is(2, Ext::sym("B") * Ext::ka(0x280));
            b.ka_is(0x284, Ext::sym("B") * Ext::ka(0x060) * Ext::ka(0x140));
            let ohat = ka.ptr(b, "ptr_ohat", 0x270);
            let lse = ka.ptr(b, "ptr_lse", 0x278);
            // fp16 partials in an fp32-sized allocation (the host serves both widths)
            b.arg_tensor("ptr_ohat", Ext::ka(0x280) * Ext::ka(0x284) * Ext::c(128 * 4));
            b.arg_tensor("ptr_lse", Ext::ka(0x280) * Ext::ka(0x284) * Ext::c(4));
            Some(KvsArgs { ohat, lse, splits: b.bind_i32(splits), rows_total: b.bind_i32(rows_total) })
        } else {
            None
        };
        FmhaIo { q, k, v, o, qsc, ksc, ptr_vd, gqa, s_len: s.val().unwrap(), kv_len: skv.val().unwrap(), kvs }
    }

    /// Block ids, lane ids, this lane's query row (clamped), the resident Q fragments + scales, the KV head and the key range (all keys).
    pub fn block(&self, b: &mut FnBuilder, rows_lg: i32) -> FmhaBlock {
        let qtile = b.let_(Op::WorkgroupIdX);
        let hidx = b.let_(Op::WorkgroupIdY);
        let zz = b.let_(Op::WorkgroupIdZ);
        let lane = Lane::new(b);
        // KV split: grid z = batch * splits + split (splits in {2, 4, 8}: log2 = (s >> 1) - (s >> 3)); this block owns the 128-key tiles
        // [split * tps, min(ntiles, (split + 1) * tps)) - every key index below is relative to its first key
        let kv_all = b.bind_i32(self.kv_len.clone());
        let ntiles_total = b.bind_i32((ev(kv_all) + 127).lshr(7));
        let (bidx, kv_len, key_lo, split) = match &self.kvs {
            None => (zz, kv_all, I32::c(0), None),
            Some(kv) => {
                let (bi, si, lg) = components::kv_split_decode(b, zz, kv.splits, 0x280, "B");
                let (key_lo, kv_eff) = components::kv_split_range(b, kv_all, si, lg, 0x110);
                (bi, kv_eff, ev(key_lo), Some(si))
            }
        };
        let row0 = b.let_i32((ev(qtile) << rows_lg) + (lane.wave() << 5));
        let row = b.let_i32(&row0 + lane.j());
        let s_last = b.let_i32(&self.s_len - 1);
        let row_c = b.let_i32(row.lt(self.s_len.clone()).select_i32(row.clone(), s_last.clone()));
        let origin = vec![ev(bidx), I32::c(0), ev(hidx), I32::c(0)];
        // Q: two 16-byte loads per 64-byte k-group, kept in registers for the whole kernel; its E8M0 scales (- 9: S in units of 1/512)
        let qt = self.q.tile(origin.clone()).clamp(1, self.s_len.clone(), s_last.clone());
        let q_off = b.bind_i32(qt.offset(&[I32::c(0), row.clone(), I32::c(0), lane.half() << 4]));
        let q = RowFrags::load(b, self.q.ptr, q_off);
        let qdp = self.qsc.tile(origin.clone()).clamp(1, self.s_len.clone(), s_last.clone()).ptr(b, &[I32::c(0), row.clone(), I32::c(0), I32::c(0)]);
        let qd_raw = b.load_f32(qdp, 0);
        let qd = b.bitcast_i32(qd_raw);
        let sel = ScaleSel::new(b, &lane);
        let (sq0, sq1) = sel.pair(b, qd);
        let sq = [b.bind_i32(ev(sq0) - 9), b.bind_i32(ev(sq1) - 9)];
        let vd = b.load_f32(self.ptr_vd, 0);
        let kvh = components::gqa_kv_head(b, hidx, self.gqa, "HK");
        let nkv = b.bind_i32((ev(kv_len) + 127).lshr(7));
        b.assume_trusted(nkv, Ext::c(1), Ext::ka(0x110).ceil_div(128), "nkv = (kv_len + 127) >> 7 with 1 <= kv_len <= SK; the engine floors the lower bound of a shift to 0");
        let g4 = b.bind_i32(C32x32F32::half_base(&lane));
        FmhaBlock { lane, bidx, hidx, kvh, row0, row_c, s_last, kv_len, key_lo, nkv, ntiles_total, split, q, sq, sel, vd, g4 }
    }
}

pub struct FmhaBlock {
    pub lane: Lane,
    pub bidx: ValueId,
    pub hidx: ValueId,
    pub kvh: ValueId,
    /// first query row of this wave
    pub row0: I32,
    /// this lane's query row clamped to S - 1, and S - 1
    pub row_c: I32,
    pub s_last: I32,
    /// keys of this block (relative to its first key `key_lo`) / number of 128-key tiles; tiles of the whole sequence
    pub kv_len: ValueId,
    pub key_lo: I32,
    pub nkv: ValueId,
    pub ntiles_total: ValueId,
    /// KV split index of this block
    pub split: Option<ValueId>,
    pub q: RowFrags,
    pub sq: [ValueId; 2],
    pub sel: ScaleSel,
    pub vd: ValueId,
    /// `4 (lane >> 5)`: key offset of this lane half in a C32x32 accumulator
    pub g4: ValueId,
}

impl FmhaBlock {
    /// A K/V-shaped tensor (rows = keys) as a tile of this block's KV head, rows clamped to the last key.
    fn kv_tile(&self, b: &mut FnBuilder, t: &Tensor) -> TileView {
        let last = b.let_i32(ev(self.kv_len) - 1);
        t.tile(vec![ev(self.bidx), self.key_lo.clone(), ev(self.kvh), I32::c(0)]).clamp(1, ev(self.kv_len), last)
    }
}

// ------------------------------------------------------------------------------------------------------------------------------------------------------
// The attention body: per 128-key tile, QK (S = ones x bias + K Q^T), online softmax, PV + row sums
// ------------------------------------------------------------------------------------------------------------------------------------------------------

/// LDS buffers + copies + readers of the K/V/K-scale ring (the `_vs128` / `_kvsplit` plan = `fmha_vs_tiles`).
pub struct KvRing {
    pub ks: Shared,
    pub vs: Shared,
    pub kd: Shared,
    pub ones: Shared,
    pub q1: Shared,
    pub q0: Shared,
    pub epi: Shared,
    kcp: RowsCopy,
    vcp: RowsCopy,
    kdcp: DwordRowsCopy,
    kr: RowsReader,
    vr: TrReader,
    kdr: RowScaleReader,
}

impl KvRing {
    pub fn new(b: &mut FnBuilder, io: &FmhaIo, blk: &FmhaBlock, roles: &Roles) -> KvRing {
        let mut lds = LdsPlan::new();
        let [ks, vs] = lds.ring_group(2, [("K", 16384, Layout::FMHA_SWZ), ("V", 16384, Layout::FMHA_SWZ)]);
        let kd = lds.ring("KD", 512, 2, Layout::Linear);
        lds.at(67584);
        let ones = lds.tile("ones", 2048, Layout::Linear);
        let q1 = lds.tile("Q1", 16384, Layout::Linear);
        let q0 = lds.tile("Q0", 16384, Layout::Linear);
        let epi = lds.private("epi", 4096, 8, &[&q1, &q0]);
        lds.commit(b);
        let lane = &blk.lane;
        let kr = RowsReader::new(b, &ks, lane);
        let vr = TrReader::new(b, &vs, lane);
        let kdr = RowScaleReader::new(b, &kd, lane);
        let (kt, vt, kdt) = (blk.kv_tile(b, &io.k), blk.kv_tile(b, &io.v), blk.kv_tile(b, &io.ksc));
        let (ksrc, vsrc, kdsrc) = (kt.row_source(b, 1), vt.row_source(b, 1), kdt.row_source(b, 1));
        let kcp = RowsCopy::new(b, ksrc, &ks, roles.wave_in_group, lane, 4);
        let vcp = RowsCopy::new(b, vsrc, &vs, roles.wave_in_group, lane, 4);
        let kdcp = DwordRowsCopy::new(b, kdsrc, &kd, roles.wave_in_group, lane);
        KvRing { ks, vs, kd, ones, q1, q0, epi, kcp, vcp, kdcp, kr, vr, kdr }
    }
}

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
}

impl<'a> AttnBody<'a> {
    pub fn new(b: &mut FnBuilder, ring: &'a KvRing, blk: &'a FmhaBlock, wid: ValueId, lut: bool) -> AttnBody<'a> {
        let sm = OnlineSoftmax::new(b, &blk.lane, Recipe::vsx(lut));
        let zero16 = b.frag_zero(DType::Frag16);
        let (c127, c123) = (b.c_i32(127), b.c_i32(123));
        AttnBody { ring, blk, wid, sm, zero16, c127, c123, s_init: Vec::new(), s: None, o: None, p: Vec::new() }
    }
    /// QK of the tile in `slot`: either in place into S (each S block first set to ones x bias) or (prologue) as fresh values.
    fn qk(&self, b: &mut FnBuilder, slot: &SlotIdx, out: QkOut, bias: ValueId) -> Vec<ValueId> {
        let (r, blk) = (self.ring, self.blk);
        let k = r.kr.at(b, &r.ks.at(slot));
        let kd = r.kdr.at(b, &r.kd.at(slot));
        qk_tile(b, out, Some((self.sm.ones_f16, bias)), &k, &kd, &blk.q.frags, blk.sq, &blk.sel)
    }
}

impl PipeBody for AttnBody<'_> {
    fn stage(&mut self, b: &mut FnBuilder, role: Role, key0: &I32, slot: &SlotIdx) {
        let r = self.ring;
        match role {
            Role::A => {
                r.kdcp.issue(b, key0, &r.kd.at(slot));
                r.kcp.issue(b, key0, &r.ks.at(slot));
            }
            _ => r.vcp.issue(b, key0, &r.vs.at(slot)),
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
        let r = self.ring;
        let mut w = std::mem::take(&mut self.p);
        let bias = w.pop().expect("bias word");
        let v = r.vr.at(b, &r.vs.at(&t.cur));
        let p = [Frag::<B32x32x64Fp8>::from_words(b, &w[0..8]), Frag::<B32x32x64Fp8>::from_words(b, &w[8..16])];
        self.sm.row_sum(b, p[0].v, p[1].v);
        pv_tile(b, &self.o.unwrap(), &v, &p, self.c127, self.c123);
        if t.staged {
            self.qk(b, &t.next, QkOut::Vars(&self.s.unwrap()), bias);
        }
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
    let ring = KvRing::new(b, &io, &blk, &roles);
    let pipe = Pipeline::new(Cadence::TwoPhaseOffset, blk.nkv, blk.kv_len, 128, Ext::ka(0x110).ceil_div(128));
    let o_tile = io.o.tile(vec![ev(blk.bidx), I32::c(0), ev(blk.hidx), I32::c(0)]).clamp(1, io.s_len.clone(), &io.s_len - 1);
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
            let row_ptr = o_tile.ptr(b, &[I32::c(0), &blk.row0 + blk.lane.j(), I32::c(0), blk.lane.half() << 2]);
            let live = b.bind_cond(blk.row0.lt(io.s_len.clone()));
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
    let ring = KvRing::new(b, &io, &blk, &roles);
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