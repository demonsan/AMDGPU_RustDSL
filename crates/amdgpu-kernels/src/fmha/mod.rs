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
