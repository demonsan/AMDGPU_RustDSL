//! FMHA kernel-family components: the 656-byte vendor FMHA kernarg ABI as tensors (+ the KV-split extension) with the host contract they imply
//! (`FmhaIo`), the per-block state every FMHA kernel starts with (`FmhaBlock`: block/lane ids, clamped query row, resident Q fragments and scales, GQA KV
//! head, KV-split key range, the recipe-copy condition and the output scale) and the KV-split combine kernel body (`combine_kernel`). The kernels in
//! `mod.rs` build these through `declare_io` / `block3` and compose them with a `PipeBody` (`AttnBody3`).

use amdgpu_tile::components;
use amdgpu_tile::frag::*;
use amdgpu_tile::*;
use amdgpu_dsl::{DType, Ext, FnBuilder, Op, ValueId, I32};
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
    /// number of query heads H
    pub heads: ValueId,
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
    /// Declares the whole kernarg segment (656 bytes; 720 with the KV-split extension) and its host contract for a launch of `rows_per_block` query rows
    /// per block.
    pub fn declare(b: &mut FnBuilder, rows_per_block: i64, kvsplit: bool) -> FmhaIo {
        let ka = &mut KArgs::new();
        // GQA (H = HK * gqa, gqa a power of two <= 16) and the grid; the tensor contracts below are stated in terms of H
        let gqa = ka.i32(b, "s_gqa", 0x0B0);
        b.ka_range(0x0B0, 1, 16);
        b.ka_is(0x140, Ext::sym("HK") * Ext::ka(0x0B0));
        let s = ka.dim(b, "s_seq_len", 0x060, 1, i64::MAX);
        let skv = ka.dim(b, "s_kv_seq_len", 0x110, 1, i64::MAX);
        let heads = ka.i32(b, "s_q_head_num", 0x140);
        let h = Dim::Ka { off: 0x140, val: heads };
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
            ka.arg(b, "abi_tail", DType::I32, 0x280, 16);
            None
        };
        FmhaIo { q, k, v, o, qsc, ksc, ptr_vd, gqa, heads, s_len: s.val().unwrap(), kv_len: skv.val().unwrap(), kvs }
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

impl FmhaIo {
    /// The lane's bf16 output row pointer (the clamped row, INCLUDING the lane-half term `8 (lane >> 5)` bytes) and the wave-live condition.
    pub fn out_row(&self, b: &mut FnBuilder, blk: &FmhaBlock) -> (ValueId, ValueId) {
        let o_tile = self.o.tile(vec![ev(blk.bidx), I32::c(0), ev(blk.hidx), I32::c(0)]).clamp(1, self.s_len.clone(), blk.s_last.clone());
        let row_ptr = o_tile.ptr(b, &[I32::c(0), &blk.row0 + blk.lane.j(), I32::c(0), blk.lane.half() << 2]);
        let live = b.bind_cond(blk.row0.lt(self.s_len.clone()));
        (row_ptr, live)
    }
}
impl FmhaIo {
    /// KV-split partial epilogue: `lse` and the fp16 `Ohat = O * fin` of the lane's row at partial row `split * rows_total + (b * S + row) * H + h`, through
    /// the wave's private transpose scratch `epi`. The scratch reuses the Q staging of the neighbouring waves, whose last reads may still be in flight
    /// (the drain's last barrier is role-dependent): every wave meets one barrier first.
    pub fn store_partial(&self, b: &mut FnBuilder, blk: &FmhaBlock, epi: &PrivRef, o: &AccTile<C32x32F32, 4>, fin: ValueId, lse: ValueId) {
        let kv = self.kvs.as_ref().expect("store_partial: not a KV-split launch");
        b.barrier();
        let part_row0 = b.bind_i32(ev(blk.split.expect("split")) * ev(kv.rows_total));
        let prow = |row: I32| ev(part_row0) + (((ev(blk.bidx) * self.s_len.clone()) + row) * ev(self.heads) + ev(blk.hidx));
        let lse_ptr = {
            let off = b.bind_i32(prow(blk.row_c.clone()) << 2);
            b.ptr_add(kv.lse, off)
        };
        let live = b.bind_cond(blk.row0.lt(self.s_len.clone()));
        let lane = blk.lane;
        amdgpu_tile::epilogue::kvsplit_partial_f16(b, o, fin, &lane, epi, lse_ptr, lse, live, &mut |b, i| {
            // round i: the lane reads back 16 bytes of row 8 i + (lane >> 3) (clamped), chunk lane & 7, of the 256-byte fp16 partial row
            let r = b.let_i32((&blk.row0 + 8 * i as i32) + lane.lane().lshr(3));
            let rg = r.lt(self.s_len.clone()).select_i32(r.clone(), blk.s_last.clone());
            let off = b.bind_i32((prow(rg) << 8) + ((lane.lane() & 7) << 4));
            b.ptr_add(kv.ohat, off)
        });
    }
}

impl FmhaBlock {
    /// The recipe copy: sequences of fewer than 4 tiles of 128 keys (the WHOLE sequence, also for a KV split) keep the exact LUT-corrected recipe
    /// (from 2 tiles on, the vendor recipe missed parity at S=300 SK=129: 1.31x the vendor's mean error).
    pub fn lut_copy(&self, b: &mut FnBuilder) -> ValueId {
        b.bind_cond(ev(self.ntiles_total).lt(4))
    }
    /// The output scale `V descale / l`.
    pub fn out_scale(&self, b: &mut FnBuilder, l: ValueId) -> ValueId {
        let inv_l = b.rcp_f32(l);
        b.mul_f32(inv_l, self.vd)
    }
}

/// The whole KV-split combine kernel body (`..._kvsplit_combine{8,16}h`): kernarg 0x00 Ohat, 0x08 lse, 0x10 out, 0x18 rows_total, 0x1C D (= 128),
/// 0x20 splits with its host contract; thread -> `per_thread`-element group of the output (the tail threads redo the last group) -> `combine_splits`.
/// Launch: grid `ceil(rows_total * 128 / per_thread / 256)` x 256 threads.
pub fn combine_kernel(b: &mut FnBuilder, per_thread: usize) {
    let ka = &mut KArgs::new();
    let ohat = ka.ptr(b, "ptr_ohat", 0x00);
    let lse = ka.ptr(b, "ptr_lse", 0x08);
    let out = ka.ptr(b, "ptr_out", 0x10);
    let rows = ka.dim(b, "rows_total", 0x18, 1, i64::MAX).val().unwrap();
    ka.dim(b, "d", 0x1C, 128, 128);
    let splits = ka.dim(b, "splits", 0x20, 1, i64::MAX).val().unwrap();
    b.arg_tensor("ptr_ohat", Ext::ka(0x20) * Ext::ka(0x18) * Ext::c(128 * 4));
    b.arg_tensor("ptr_lse", Ext::ka(0x20) * Ext::ka(0x18) * Ext::c(4));
    b.arg_tensor("ptr_out", Ext::ka(0x18) * Ext::c(128 * 2));
    let lg = (128 / per_thread as i32).trailing_zeros() as i32; // groups per row, log2
    let tid = b.let_(Op::WorkitemId);
    let bx = b.let_(Op::WorkgroupIdX);
    let groups = b.let_i32(rows.clone() << lg);
    let t_raw = b.let_i32((ev(bx) << 8) + ev(tid));
    let t = b.bind_i32(t_raw.lt(groups.clone()).select_i32(t_raw.clone(), groups - 1));
    let row = b.bind_i32(ev(t).lshr(lg));
    b.assume_trusted(row, Ext::c(0), Ext::ka(0x18) - Ext::c(1), "row = clamped thread index >> k <= rows - 1: the engine keeps the rational upper bound of a shift");
    let (rows_v, splits_v) = (b.bind_i32(rows), b.bind_i32(splits));
    amdgpu_tile::epilogue::combine_splits(b, ohat, lse, out, rows_v, splits_v, t, row, per_thread, Ext::ka(0x20) - Ext::c(1));
}
