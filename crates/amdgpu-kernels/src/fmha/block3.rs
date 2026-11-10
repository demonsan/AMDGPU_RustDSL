//! The FMHA block launch (L2 template candidate): the kernarg ABI with its derived host contract (`declare_io`), the per-block state (grid
//! map, key range, split decode: `block3*`) and the output / partial epilogues (`store_out3*`, `store_partial3`).

use amdgpu_dsl::{Ext, FnBuilder, I32, Op, Opnd, ValueId};
use amdgpu_isa::{DS_READ_B128, DS_WRITE_B32, GLOBAL_LOAD_X4, GLOBAL_STORE_X2};
use amdgpu_layout::{ComposedLayout, make_layout, Swizzle, tup};
use amdgpu_tile::{Ctx, ev, Lane, Shared, Tensor};
use amdgpu_tile::copy::make_tiled_copy;
use amdgpu_tile::frag::{AccTile, B32x32x64Fp8, C32x32F32, Frag};
use amdgpu_tile::mma::make_tiled_mma;
use amdgpu_tile::components;
use amdgpu_tile::frag;
use super::attn3::attention_orientation;
use super::io::FmhaBlock;
use super::io;

/// The kernarg offsets the generic block launch reads (the ABI stays with the kernel crate's `declare_io`; docs/l2-family-bodies.md §2.2).
#[derive(Clone, Copy, Debug)]
pub struct LaunchAbi {
    /// H (query heads)
    pub h_ka: u32,
    /// SQ (query length)
    pub sq_ka: u32,
    /// SK (key length)
    pub sk_ka: u32,
    /// the softmax scale (f32; the per-tensor FP8 form folds it into the Q scale)
    pub scale_ka: u32,
}

/// The global operands of an attention launch, as the attention bodies see them (docs/l2-family-bodies.md §2.1). `FmhaIo` implements it.
pub trait Operands {
    fn q(&self) -> &Tensor;
    fn k(&self) -> &Tensor;
    fn v(&self) -> &Tensor;
    fn o(&self) -> &Tensor;
    fn qsc(&self) -> &Tensor;
    fn ksc(&self) -> &Tensor;
    fn vsc(&self) -> Option<&Tensor>;
    fn ptr_vd(&self) -> ValueId;
    fn gqa(&self) -> ValueId;
    fn heads(&self) -> ValueId;
    fn s_len(&self) -> I32;
    fn kv_len(&self) -> I32;
    fn kvs(&self) -> Option<&io::KvsArgs>;
    fn abi(&self) -> LaunchAbi;
}
/// `FmhaIo::block` with the Q fragments loaded through a TiledCopy: the B operand of the scaled 32x32x64 atom over the lane's (clamped) query row, two
/// k-groups of 64 bytes (placements 0 / 64): one pointer (row + 16 (lane >> 5)) and four `global_load_dwordx4` with immediates 0 / 32 / 64 / 96.
pub fn block3(io: &dyn Operands, b: &mut FnBuilder, rows_lg: i32) -> FmhaBlock {
    block3_x(io, b, rows_lg, false)
}

/// `xcd`: the vendor's XCD-aware block order. Workgroups are dispatched round robin over the 8 XCDs (linear id % 8), each with its own L2;
/// with the plain order the 8 query tiles of one head land on 8 different XCDs and every XCD streams the same K / V from memory. The remap
/// gives XCD x the contiguous logical range [x N / 8, (x + 1) N / 8) of the (query tile, head) plane of the batch (N = tiles x heads), so the
/// blocks that share K / V share an L2. Applied when N % 8 == 0 and H <= 32 (the vendor's conditions; the head is recovered by at most 32
/// compare-subtract steps), the plain order otherwise.
pub fn block3_x(io: &dyn Operands, b: &mut FnBuilder, rows_lg: i32, xcd: bool) -> FmhaBlock {
    block3_x_mode(io, b, rows_lg, xcd, None)
}

/// `block3_x` with the per-tensor mode.
pub fn block3_x_mode(io: &dyn Operands, b: &mut FnBuilder, rows_lg: i32, xcd: bool, pt: Option<PtMode>) -> FmhaBlock {
    let (qtile, hidx) = if xcd {
        amdgpu_tile::templates::grid::xcd_tile_head(b, io.s_len(), rows_lg, io.abi().h_ka, io.abi().sq_ka)
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
pub fn block3_at(io: &dyn Operands, b: &mut FnBuilder, rows_lg: i32, qtile: ValueId, hidx: ValueId, pt: Option<PtMode>) -> FmhaBlock {
    let zz = b.let_(amdgpu_dsl::Op::WorkgroupIdZ);
    let lane = Lane::new(b);
    let kv_all = b.bind_i32(io.kv_len());
    let ntiles_total = b.bind_i32((ev(kv_all) + 127).lshr(7));
    let causal = matches!(pt, Some(PtMode { causal: true }));
    let kv_end = if causal {
        // the block's last visible key + 1 = min(256 (qtile + 1) + SK - SQ, SK), at least 1 (rows that see no key: undefined output)
        let v = b.bind_i32((((ev(qtile) + 1) << rows_lg) + (ev(kv_all) - io.s_len())).min(ev(kv_all)).max(1));
        let r = b.read_first_lane(v);
        b.let_(Op::AddI32(Opnd::Val(r), Opnd::I32(0)))
    } else {
        kv_all
    };
    let mut split_empty = None;
    let (bidx, kv_len, key_lo, split) = match io.kvs() {
        None => (zz, kv_all, I32::c(0), None),
        Some(kv) => {
            let (bi, si, lg) = amdgpu_tile::templates::grid::kv_split_decode(b, zz, kv.splits, kv.splits_off, "B");
            if causal {
                // the split covers this block's causal key range (kv_end, computed above); a split past it is empty
                let (key_lo, kv_eff, empty) = components::kv_split_range_var(b, kv_end, si, lg, io.abi().sk_ka);
                split_empty = Some(empty);
                (bi, kv_eff, ev(key_lo), Some(si))
            } else {
                let (key_lo, kv_eff) = components::kv_split_range(b, kv_all, si, lg, io.abi().sk_ka);
                (bi, kv_eff, ev(key_lo), Some(si))
            }
        }
    };
    let kv_len = if causal && io.kvs().is_none() { kv_end } else { kv_len };
    let row0 = b.let_i32((ev(qtile) << rows_lg) + (lane.wave() << 5));
    let row = b.let_i32(&row0 + lane.j());
    let s_last = b.let_i32(io.s_len() - 1);
    let row_c = b.let_i32(row.lt(io.s_len()).select_i32(row.clone(), s_last.clone()));
    let origin = vec![ev(bidx), I32::c(0), ev(hidx), I32::c(0)];
    let qt = io.q().tile(origin.clone()).clamp(1, io.s_len(), s_last.clone());
    let q_row = b.let_i32(qt.offset(&[I32::c(0), row.clone(), I32::c(0), I32::c(0)]));
    let mut ctx = Ctx::new();
    let l = ctx.sym("lane", 0, 63, lane.lane());
    let mma = make_tiled_mma(attention_orientation().0, false);
    let qcp = amdgpu_tile::copy::make_tiled_copy_b(GLOBAL_LOAD_X4, &mma);
    // the row is the lane's (clamped) query row, supplied above: the view gives the byte inside it (k), row stride 0
    let qpart = qcp.partition_global(b, &ctx, &l, &make_layout(tup![32, 64], tup![0, 1]), &[0, 64]);
    let ptrs = qpart.ptrs(b, io.q().ptr, &q_row);
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
        let qdp = io.qsc().tile(origin).clamp(1, io.s_len(), s_last.clone()).ptr(b, &[I32::c(0), row.clone(), I32::c(0), I32::c(0)]);
        let qd_raw = b.load_f32(qdp, 0);
        let qd = b.bitcast_i32(qd_raw);
        let sel = frag::ScaleSel::new(b, &lane);
        let (sq0, sq1) = sel.pair(b, qd);
        let sq = [b.bind_i32(ev(sq0) - 9), b.bind_i32(ev(sq1) - 9)];
        let vd = b.load_f32(io.ptr_vd(), 0);
        let kvh = components::gqa_kv_head(b, hidx, io.gqa(), "HK");
        (sel, sq, vd, kvh, None)
    } else {
        let sel = frag::ScaleSel::new(b, &lane);
        let kvh = components::gqa_kv_head(b, hidx, io.gqa(), "HK");
        // the [B, HK] descales of this (batch, KV head)
        let at = |b: &mut FnBuilder, t: &Tensor| {
            let off = b.bind_i32(t.offset(&[ev(bidx), ev(kvh)]));
            let p = b.ptr_add(t.ptr, off);
            b.load_f32(p, 0)
        };
        let qd = at(b, io.qsc());
        let kd = at(b, io.ksc());
        let vd = at(b, io.vsc().expect("fp8pt: the V descale tensor"));
        let scale = b.let_(Op::KernargF32 { offset: io.abi().scale_ka });
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
    b.assume_trusted(nkv, Ext::c(1), Ext::ka(io.abi().sk_ka).ceil_div(128), "nkv = (kv_len + 127) >> 7 with 1 <= kv_len <= SK; the engine floors the lower bound of a shift to 0");
    let g4 = b.bind_i32(C32x32F32::half_base(&lane));
    // causal: the LUT-copy decision counts the block's tiles; the paired form's ring reads the whole sequence (kv_clamp = SK), the split form
    // stays inside its part
    let (kv_clamp, ntiles_total) = match (causal, io.kvs().is_some()) {
        (true, false) => (kv_all, nkv),
        (true, true) => (kv_len, b.bind_i32((ev(kv_end) + 127).lshr(7))),
        _ => (kv_len, ntiles_total),
    };
    FmhaBlock { lane, bidx, hidx, kvh, row0, row_c, s_last, kv_len, kv_clamp, key_lo, nkv, ntiles_total, split, split_empty, q, sq, sel, vd, g4, pt: ptsc }
}

/// The bf16 epilogue through a TiledCopy: the C layout of the scaled 32x32x64 atom (O^T: rows = dims, columns = queries) stored with
/// `global_store_dwordx2` (4 bf16 per instruction) into the lane's (clamped) output row: one pointer (row + 8 (lane >> 5)) and the immediates
/// `64 e + 16 q` - the shipped instruction stream, under `if row0 < S`.
pub fn store_out3(io: &dyn Operands, b: &mut FnBuilder, blk: &FmhaBlock, o: &AccTile<C32x32F32, 4>, fin: ValueId) {
    let o_tile = io.o().tile(vec![ev(blk.bidx), I32::c(0), ev(blk.hidx), I32::c(0)]).clamp(1, io.s_len(), blk.s_last.clone());
    let row = b.let_i32(o_tile.offset(&[I32::c(0), &blk.row0 + blk.lane.j(), I32::c(0), I32::c(0)]));
    let live = b.bind_cond(blk.row0.lt(io.s_len()));
    let mut ctx = Ctx::new();
    let l = ctx.sym("lane", 0, 63, blk.lane.lane());
    let mma = make_tiled_mma(attention_orientation().0, false);
    let st = make_tiled_copy(GLOBAL_STORE_X2, mma.c_tv(), 32 * 32, 16);
    let part = st.partition_global(b, &ctx, &l, &make_layout(tup![32, 32], tup![2, 0]), &[0, 64, 128, 192]);
    b.begin_if(Opnd::Val(live));
    let ptrs = part.ptrs(b, io.o().ptr, &row);
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
pub fn store_out3_lds(io: &dyn Operands, b: &mut FnBuilder, blk: &FmhaBlock, epi: &Shared, wid: ValueId, o: &AccTile<C32x32F32, 4>, fin: ValueId) {
    store_out3_lds_x(io, b, blk, epi, wid, o, fin, false)
}

/// `store_out3_lds`; `stores_only`: the `row0 < S` guard covers only the global stores (the wave-private LDS transpose runs unconditionally),
/// so the epilogue adds no protocol-relevant branch (the paired causal form keeps the verifier's path enumeration within its 64 paths).
#[allow(clippy::too_many_arguments)]
pub fn store_out3_lds_x(io: &dyn Operands, b: &mut FnBuilder, blk: &FmhaBlock, epi: &Shared, wid: ValueId, o: &AccTile<C32x32F32, 4>, fin: ValueId, stores_only: bool) {
    b.barrier();
    let o_tile = io.o().tile(vec![ev(blk.bidx), I32::c(0), ev(blk.hidx), I32::c(0)]).clamp(1, io.s_len(), blk.s_last.clone());
    let live = b.bind_cond(blk.row0.lt(io.s_len()));
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
            b.ptr_add(io.o().ptr, off)
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
pub fn store_partial3(io: &dyn Operands, b: &mut FnBuilder, blk: &FmhaBlock, epi: &Shared, wid: ValueId, o: &AccTile<C32x32F32, 4>, fin: ValueId, lse: ValueId) {
    let kv = io.kvs().expect("store_partial3: not a KV-split launch");
    b.barrier();
    let part_row0 = b.bind_i32(ev(blk.split.expect("split")) * ev(kv.rows_total));
    let prow = |row: I32| ev(part_row0) + (((ev(blk.bidx) * io.s_len()) + row) * ev(io.heads()) + ev(blk.hidx));
    let lse_ptr = {
        let off = b.bind_i32(prow(blk.row_c.clone()) << 2);
        b.ptr_add(kv.lse, off)
    };
    let live = b.bind_cond(blk.row0.lt(io.s_len()));
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
            let rg = r.lt(io.s_len()).select_i32(r.clone(), blk.s_last.clone());
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
