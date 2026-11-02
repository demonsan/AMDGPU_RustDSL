//! L6c: the FMHA kernel-family components - the 656-byte vendor FMHA kernarg ABI as tensors (+ the KV-split extension) with the host contract they imply
//! (`FmhaIo`), the per-block setup every FMHA kernel starts with (`FmhaBlock`: block/lane ids, clamped query row, resident Q fragments and scales, GQA KV
//! head, KV-split key range) and the K/V/K-scale LDS ring with its copies and fragment readers (`KvRing`: the `_vs128`/`_kvsplit` 2-slot plan or the
//! `_t128` 3-slot plan). Kernels (`crate::fmha_tile`) compose these with a `PipeBody`.

use super::components;
use super::frag::*;
use super::*;
use crate::{DType, Ext, FnBuilder, Op, ValueId, I32};
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
    pub fn kv_tile(&self, b: &mut FnBuilder, t: &Tensor) -> TileView {
        let last = b.let_i32(ev(self.kv_len) - 1);
        t.tile(vec![ev(self.bidx), self.key_lo.clone(), ev(self.kvh), I32::c(0)]).clamp(1, ev(self.kv_len), last)
    }
}

// ------------------------------------------------------------------------------------------------------------------------------------------------------
// The attention body: per 128-key tile, QK (S = ones x bias + K Q^T), online softmax, PV + row sums
// ------------------------------------------------------------------------------------------------------------------------------------------------------

/// K-scale staging: straight into LDS by DMA (`_vs128` / `_kvsplit`: one `global_load_lds_dword` per wave) or through a VGPR and a `ds_write` after the
/// DMA wait (`_t128`).
pub enum KdCopy {
    Dma(DwordRowsCopy),
    Vgpr(DwordRowsVgpr),
}

/// LDS buffers + copies + readers of the K/V/K-scale ring: `_vs128` / `_kvsplit` (= `fmha_vs_tiles`: 2 slots, K by group A, V by group B) or `_t128`
/// (3 slots, one group stages K and V with a shared row clamp).
pub struct KvRing {
    pub ks: Shared,
    pub vs: Shared,
    pub kd: Shared,
    pub ones: Shared,
    pub q1: Shared,
    pub q0: Shared,
    pub epi: Shared,
    pub kcp: RowsCopy,
    pub vcp: RowsCopy,
    pub kdc: KdCopy,
    pub kr: RowsReader,
    pub vr: TrReader,
    pub kdr: RowScaleReader,
}

impl KvRing {
    pub fn new(b: &mut FnBuilder, io: &FmhaIo, blk: &FmhaBlock, roles: &Roles, slots: u32) -> KvRing {
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
        let kr = RowsReader::new(b, &ks, lane);
        let vr = TrReader::new(b, &vs, lane);
        let kdr = RowScaleReader::new(b, &kd, lane);
        let (kt, vt, kdt) = (blk.kv_tile(b, &io.k), blk.kv_tile(b, &io.v), blk.kv_tile(b, &io.ksc));
        let (ksrc, vsrc, kdsrc) = (kt.row_source(b, 1), vt.row_source(b, 1), kdt.row_source(b, 1));
        let kcp = RowsCopy::new(b, ksrc, &ks, roles.wave_in_group, lane, 4);
        let vcp = RowsCopy::new(b, vsrc, &vs, roles.wave_in_group, lane, 4);
        let kdc = if slots == 2 {
            KdCopy::Dma(DwordRowsCopy::new(b, kdsrc, &kd, roles.wave_in_group, lane))
        } else {
            // one dword per row: thread t covers row t >> 1 (two threads per row)
            let rt_row = b.bind_i32(lane.tid().lshr(1));
            KdCopy::Vgpr(DwordRowsVgpr::new(b, kdsrc, &kd, rt_row))
        };
        KvRing { ks, vs, kd, ones, q1, q0, epi, kcp, vcp, kdc, kr, vr, kdr }
    }
}

impl FmhaIo {
    /// The lane's bf16 output row pointer (the clamped row, INCLUDING the lane-half term `8 (lane >> 5)` bytes) and the wave-live condition.
    pub fn out_row(&self, b: &mut FnBuilder, blk: &FmhaBlock) -> (ValueId, ValueId) {
        let o_tile = self.o.tile(vec![ev(blk.bidx), I32::c(0), ev(blk.hidx), I32::c(0)]).clamp(1, self.s_len.clone(), blk.s_last.clone());
        let row_ptr = o_tile.ptr(b, &[I32::c(0), &blk.row0 + blk.lane.j(), I32::c(0), blk.lane.half() << 2]);
        let live = b.bind_cond(blk.row0.lt(self.s_len.clone()));
        (row_ptr, live)
    }
}