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
mod attn3;
mod block3;
pub use attn3::*;
pub use block3::*;
pub mod graph;
pub mod bf16;
pub mod bf16v3;
pub mod bwd;
pub mod decode;
pub mod decode16;
pub mod paged;
pub mod mla;
pub mod fp8pt;
mod skeleton_kvsplit;
mod skeleton_t128;
mod skeleton_vs128;

use io::{combine_kernel, FmhaBlock, FmhaIo};
use amdgpu_tile::frag::{AccTile, Frag, QkOut, B32x32x64Fp8, C32x32F32};
use amdgpu_tile::reduce::KeyMask;
use amdgpu_tile::softmax::{OnlineSoftmax, Recipe};
use amdgpu_tile::*;
use amdgpu_tile::weave::Units;
use amdgpu_dsl::{CompileProfile, DType, Ext, FnBuilder, Function, Op, Opnd, StageContract, ValueId, VarId, I32};
use amdgpu_isa::{MmaAtom, DS_READ_B128, DS_READ_B128X2, DS_READ_B32, DS_READ_TR_B8X4, DS_WRITE_B32, GLOBAL_LDS_X1, GLOBAL_LDS_X4, GLOBAL_LOAD_X4, GLOBAL_STORE_X2};
use amdgpu_layout::{make_layout, tup, ComposedLayout, Ix, Layout as LLayout, Swizzle};

/// `_vs128`: 256-row blocks, two 4-wave groups offset by one phase (`Cadence::TwoPhaseOffset`), MUBUF K / V / K-scale staging, the vendor's
/// priority ping-pong, two-smfmac row sums (`amdgpu_tile::rowsum`), K-scale byte select, the vendor's reference recipe with the lazy reference
/// update from 4 tiles on (`OnlineSoftmax::set_vendor_ref`, `decide_lazy`; shorter sequences keep the exact LUT-corrected recipe: with < 4 tiles the
/// lazy recipe's error does not average out) and the LDS-staged epilogue.
pub fn fmha_fwd_hd128_mxfp8_v3_vs128() -> Function {
    let mut f = skeleton_vs128::build_vs128("fmha_fwd_hd128_mxfp8_v3_vs128", &mut Vs128Slots);
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
pub(crate) struct Vs128Slots;

impl skeleton_vs128::Vs128Slots for Vs128Slots {
    type Cx = FmhaCx;
    fn prelude(&mut self, b: &mut FnBuilder) -> FmhaCx {
        b.compile_profile(CompileProfile::Vs128);
        let io = declare_io(b, 256, false);
        let blk = block3_x(&io, b, 8, true);
        let roles = Roles::split(b, &blk.lane, 4);
        let mut ring = KvRing3::new(b, &io, &blk, &roles, 2);
        ring.enable_buffer_dma(b);
        ring.enable_buffer_kd(b);
        let mut pipe = Pipeline::new(Cadence::TwoPhaseOffset, blk.nkv, blk.kv_len, 128, Ext::ka(0x110).ceil_div(128));
        // every stage reads through a range-checked buffer descriptor: no clamp of the next tile's key
        pipe.key_unclamped = true;
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
    let mut f = skeleton_kvsplit::build_kvsplit("fmha_fwd_hd128_mxfp8_v3_kvsplit", &mut KvsplitSlots);
    // scheduler: sink every ds_read to at most one MFMA before its first use (sched/ldahead.rs, docs/lgkm-lookahead.md): +5.5..+6.8 % on realistic data (B1H8S2048 / B1H4S4096 split 4, B1H4S8192 split
    // + wsink, ldahead32=1 (counted M-phase LDS waits; kvsplit +1.1..+1.8 %, t128 -0.2..+1.5 % on realistic data)
    f.sched_opts = Some("ldsink=1,wsink,ldahead32=1".to_string());
    f
}

/// The slots of `_kvsplit` (`skeleton_kvsplit.rs`, generated from `graph::graph_kvsplit()`).
pub(crate) struct KvsplitSlots;

impl skeleton_kvsplit::KvsplitSlots for KvsplitSlots {
    type Cx = FmhaCx;
    fn prelude(&mut self, b: &mut FnBuilder) -> FmhaCx {
        b.compile_profile(CompileProfile::KvSplit);
        let io = declare_io(b, 256, true);
        let blk = block3(&io, b, 8);
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
    let mut f = skeleton_t128::build_t128("fmha_fwd_hd128_mxfp8_v3_t128", &mut T128Slots);
    // scheduler: sink every ds_read to at most one MFMA before its first use (sched/ldahead.rs, docs/lgkm-lookahead.md): +0.4..+0.8 % on realistic data
    // + wsink, ldahead32=1 (counted M-phase LDS waits; kvsplit +1.1..+1.8 %, t128 -0.2..+1.5 % on realistic data)
    f.sched_opts = Some("ldsink=1,wsink,ldahead32=1".to_string());
    f
}

/// The slots of `_t128` (`skeleton_t128.rs`, generated from `graph::graph_t128()`).
pub(crate) struct T128Slots;

impl skeleton_t128::T128Slots for T128Slots {
    type Cx = FmhaCx;
    fn prelude(&mut self, b: &mut FnBuilder) -> FmhaCx {
        b.compile_profile(CompileProfile::T128);
        // `edgepad` (MFMA hazard windows carried across blocks, no block-end padding) and `salubr` (wave-uniform branch conditions on SCC): same-run A/B,
        // B2H8S2048 layout median +3.3% together; `saddrlds` (the SADDR DMA form: +1.8..+2.1%); all three are -0.2..-1.5% on _vs128 / _kvsplit
        // (docs/dsl-layout-refactor-design.md section 9.4)
        b.lower_opts(&["saddrlds", "edgepad", "salubr"]);
        let io = declare_io(b, 128, false);
        let blk = block3_x(&io, b, 7, true);
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
        Some(io::KvsArgs { ohat, lse, splits: b.bind_i32(splits), rows_total: b.bind_i32(rows_total), splits_off: 0x280 })
    } else {
        ka.arg(b, "abi_tail", DType::I32, 0x280, 16);
        None
    };
    FmhaIo { q, k, v, o, qsc, ksc, ptr_vd, vsc: None, gqa, heads, s_len: s.val().expect("S"), kv_len: skv.val().expect("SK"), kvs }
}
