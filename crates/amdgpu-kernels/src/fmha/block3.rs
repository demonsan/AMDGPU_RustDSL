//! The FMHA block launch (L2 template candidate): the kernarg ABI with its derived host contract (`declare_io`), the per-block state (grid
//! map, key range, split decode: `block3*`) and the output / partial epilogues (`store_out3*`, `store_partial3`).

use super::*;

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
        Some(io::KvsArgs { ohat, lse, splits: b.bind_i32(splits), rows_total: b.bind_i32(rows_total), splits_off: 0x280 })
    } else {
        ka.arg(b, "abi_tail", DType::I32, 0x280, 16);
        None
    };
    FmhaIo { q, k, v, o, qsc, ksc, ptr_vd, vsc: None, gqa, heads, s_len: s.val().expect("S"), kv_len: skv.val().expect("SK"), kvs }
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
    block3_x_mode(io, b, rows_lg, xcd, None)
}

/// `block3_x` with the per-tensor mode.
pub fn block3_x_mode(io: &FmhaIo, b: &mut FnBuilder, rows_lg: i32, xcd: bool, pt: Option<PtMode>) -> FmhaBlock {
    let (qtile, hidx) = if xcd {
        amdgpu_tile::templates::grid::xcd_tile_head(b, io.s_len.clone(), rows_lg, 0x140, 0x060)
    } else {
        (b.let_(amdgpu_dsl::Op::WorkgroupIdX), b.let_(amdgpu_dsl::Op::WorkgroupIdY))
    };
    block3_at(io, b, rows_lg, qtile, hidx, pt)
}

/// The per-tensor FP8 form of a block (`fp8pt`): the [B, HK] f32 descales and the softmax scale instead of the E8M0 Q scales; `causal`: the key
/// range ends at the last visible key of the block (bottom-right alignment, keys k <= q + SK - SQ).
#[derive(Clone, Copy, Debug)]
pub struct PtMode {
    pub causal: bool,
}

/// `block3_x` at a given (query tile, head).
pub fn block3_at(io: &FmhaIo, b: &mut FnBuilder, rows_lg: i32, qtile: ValueId, hidx: ValueId, pt: Option<PtMode>) -> FmhaBlock {
    let zz = b.let_(amdgpu_dsl::Op::WorkgroupIdZ);
    let lane = Lane::new(b);
    let kv_all = b.bind_i32(io.kv_len.clone());
    let ntiles_total = b.bind_i32((ev(kv_all) + 127).lshr(7));
    let causal = matches!(pt, Some(PtMode { causal: true }));
    let kv_end = if causal {
        // the block's last visible key + 1 = min(256 (qtile + 1) + SK - SQ, SK), at least 1 (rows that see no key: undefined output)
        let v = b.bind_i32((((ev(qtile) + 1) << rows_lg) + (ev(kv_all) - io.s_len.clone())).min(ev(kv_all)).max(1));
        let r = b.read_first_lane(v);
        b.let_(Op::AddI32(Opnd::Val(r), Opnd::I32(0)))
    } else {
        kv_all
    };
    let mut split_empty = None;
    let (bidx, kv_len, key_lo, split) = match &io.kvs {
        None => (zz, kv_all, I32::c(0), None),
        Some(kv) => {
            let (bi, si, lg) = amdgpu_tile::templates::grid::kv_split_decode(b, zz, kv.splits, kv.splits_off, "B");
            if causal {
                // the split covers this block's causal key range (kv_end, computed above); a split past it is empty
                let (key_lo, kv_eff, empty) = components::kv_split_range_var(b, kv_end, si, lg, 0x110);
                split_empty = Some(empty);
                (bi, kv_eff, ev(key_lo), Some(si))
            } else {
                let (key_lo, kv_eff) = components::kv_split_range(b, kv_all, si, lg, 0x110);
                (bi, kv_eff, ev(key_lo), Some(si))
            }
        }
    };
    let kv_len = if causal && io.kvs.is_none() { kv_end } else { kv_len };
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
    let (sel, sq, vd, kvh, ptsc) = if pt.is_none() {
        let qdp = io.qsc.tile(origin).clamp(1, io.s_len.clone(), s_last.clone()).ptr(b, &[I32::c(0), row.clone(), I32::c(0), I32::c(0)]);
        let qd_raw = b.load_f32(qdp, 0);
        let qd = b.bitcast_i32(qd_raw);
        let sel = frag::ScaleSel::new(b, &lane);
        let (sq0, sq1) = sel.pair(b, qd);
        let sq = [b.bind_i32(ev(sq0) - 9), b.bind_i32(ev(sq1) - 9)];
        let vd = b.load_f32(io.ptr_vd, 0);
        let kvh = components::gqa_kv_head(b, hidx, io.gqa, "HK");
        (sel, sq, vd, kvh, None)
    } else {
        let sel = frag::ScaleSel::new(b, &lane);
        let kvh = components::gqa_kv_head(b, hidx, io.gqa, "HK");
        // the [B, HK] descales of this (batch, KV head)
        let at = |b: &mut FnBuilder, t: &Tensor| {
            let off = b.bind_i32(t.offset(&[ev(bidx), ev(kvh)]));
            let p = b.ptr_add(t.ptr, off);
            b.load_f32(p, 0)
        };
        let qd = at(b, &io.qsc);
        let kd = at(b, &io.ksc);
        let vd = at(b, io.vsc.as_ref().expect("fp8pt: the V descale tensor"));
        let scale = b.let_(Op::KernargF32 { offset: 0x050 });
        let l2e = b.c_f32(std::f32::consts::LOG2_E);
        let c0 = b.mul_f32(scale, l2e);
        let c1 = b.mul_f32(c0, qd);
        let c = b.mul_f32(c1, kd);
        // c = 2^(ex - 127) m: the E8M0 Q scale ex - 9 (S in units of 1/512 as the MX kernels), m = c with the exponent of 1.0
        let cb = b.bitcast_i32(c);
        let ex = b.bind_i32(((ev(cb).lshr(23) & 255) - 9).max(0));
        let mb = b.bind_i32((ev(cb) & 0x007F_FFFF) | 0x3F80_0000);
        let m = b.bitcast_f32(mb);
        let inv_m = b.rcp_f32(m);
        let m2 = b.frag2_dup(m);
        let kunit = b.c_i32(0x7F7F_7F7F);
        (sel, [ex, ex], vd, kvh, Some(io::PtScale { m2, inv_m, kunit }))
    };
    let nkv = b.bind_i32((ev(kv_len) + 127).lshr(7));
    b.assume_trusted(nkv, Ext::c(1), Ext::ka(0x110).ceil_div(128), "nkv = (kv_len + 127) >> 7 with 1 <= kv_len <= SK; the engine floors the lower bound of a shift to 0");
    let g4 = b.bind_i32(C32x32F32::half_base(&lane));
    // causal: the LUT-copy decision counts the block's tiles; the paired form's ring reads the whole sequence (kv_clamp = SK), the split form
    // stays inside its part
    let (kv_clamp, ntiles_total) = match (causal, io.kvs.is_some()) {
        (true, false) => (kv_all, nkv),
        (true, true) => (kv_len, b.bind_i32((ev(kv_end) + 127).lshr(7))),
        _ => (kv_len, ntiles_total),
    };
    FmhaBlock { lane, bidx, hidx, kvh, row0, row_c, s_last, kv_len, kv_clamp, key_lo, nkv, ntiles_total, split, split_empty, q, sq, sel, vd, g4, pt: ptsc }
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
            // tileop: O[e] elements 4q .. 4q+4 scaled by fin, packed to bf16
            let acc = amdgpu_tile::tileop::VarTile::f32(o.accs[e].var, 16);
            let v4 = amdgpu_tile::tileop::ewise_elems(b, &acc, (4 * q as u8)..(4 * q as u8 + 4), |b, x| b.mul_f32(x, fin));
            let w = amdgpu_tile::tileop::pack_bf16(b, &v4);
            vals.push(b.frag2(w[0], w[1]));
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
    store_out3_lds_x(io, b, blk, epi, wid, o, fin, false)
}

/// `store_out3_lds`; `stores_only`: the `row0 < S` guard covers only the global stores (the wave-private LDS transpose runs unconditionally),
/// so the epilogue adds no protocol-relevant branch (the paired causal form keeps the verifier's path enumeration within its 64 paths).
#[allow(clippy::too_many_arguments)]
pub fn store_out3_lds_x(io: &FmhaIo, b: &mut FnBuilder, blk: &FmhaBlock, epi: &Shared, wid: ValueId, o: &AccTile<C32x32F32, 4>, fin: ValueId, stores_only: bool) {
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
    if !stores_only {
        b.begin_if(Opnd::Val(live));
    }
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
                // tileop: O[e] elements 4q .. 4q+4 scaled by fin, packed to bf16
                let acc = amdgpu_tile::tileop::VarTile::f32(o.accs[e].var, 16);
                let v4 = amdgpu_tile::tileop::ewise_elems(b, &acc, (4 * q as u8)..(4 * q as u8 + 4), |b, x| b.mul_f32(x, fin));
                let w = amdgpu_tile::tileop::pack_bf16(b, &v4);
                let (w0, w1) = (w[0], w[1]);
                let n = 2 * (4 * ec + q);
                wp.write(b, n, 0, &[w0]);
                wp.write(b, n + 1, 0, &[w1]);
            }
        }
        b.wait_lds(0);
        if stores_only {
            let vs: Vec<ValueId> = (0..gptr.len()).map(|i| rp.read(b, i, 0)).collect();
            b.begin_if(Opnd::Val(live));
            for (g, v) in gptr.iter().zip(vs) {
                b.store_f32x4(*g, (128 * pr) as u32, v);
            }
            b.end_if();
            continue;
        }
        for (i, g) in gptr.iter().enumerate() {
            let v = rp.read(b, i, 0);
            b.store_f32x4(*g, (128 * pr) as u32, v);
        }
    }
    if !stores_only {
        b.end_if();
    }
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
                // tileop: O[e] elements 4q .. 4q+4 scaled by fin, to f16
                let acc = amdgpu_tile::tileop::VarTile::f32(o.accs[e].var, 16);
                let hv = amdgpu_tile::tileop::ewise_elems(b, &acc, (4 * q as u8)..(4 * q as u8 + 4), |b, x| {
                    let v0 = b.mul_f32(x, fin);
                    b.let_(amdgpu_dsl::Op::CvtF32ToF16(Opnd::Val(v0)))
                });
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
