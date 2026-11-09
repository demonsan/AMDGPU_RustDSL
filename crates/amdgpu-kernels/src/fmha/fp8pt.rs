//! Per-tensor FP8 attention prefill `fmha_fwd_hd128_fp8pt_v3[_causal]` (docs/op-coverage-design.md §7.5): Q / K / V e4m3 (BSHD), O bf16, one
//! f32 descale per (batch, KV head) for each of Q / K / V - the aiter `fmha_v3_fwd` fp8 ABI (`fwd_hd128_fp8[_causal].co`, `fmha_fwd_v3_args`
//! slots 32-40: `ptr_{q,k,v}_descale`, `s_descale_{q,k,v}_{Bs,Hs}` in bytes; slot 5 the softmax scale).
//!
//! The body is the shipped MXFP8 `_vs128` structure (`skeleton_vs128`, `AttnBody3`): 256-row blocks, two 4-wave groups one phase apart,
//! MUBUF K / V staging, the vendor's two-word reference recipe with the lazy update, two-smfmac row sums, the LDS-staged epilogue. What changes:
//! * no MX scales: the K-scale ring / DMA / reads disappear (the unit E8M0 word), the Q scale is one uniform E8M0 exponent;
//! * the softmax factor `c = scale log2(e) q_ds k_ds = 2^e m`: 2^e rides the Q scale, m in [1, 2) multiplies S in fp32 before the softmax
//!   (`AttnBody3::pt_prepare`), the bias MFMA takes the reference of `b / m` (`AttnBody3::bias_in`); v_ds goes into the output scale;
//! * `_causal`: bottom-right causal (keys k <= q + SK - SQ), the vendor's q-tile pairing (CTA x runs q tiles x and nt - 1 - x), an in-place
//!   mask on the tiles that reach the diagonal.

use super::*;
use amdgpu_tile::tensor::{Dim, KArgs, Stride};

/// The 656-byte ABI of the vendor's fp8 fmha_v3 with the per-tensor descales and its host contract. `kvsplit`: our extension behind it
/// (0x290 fp16 partials Ohat, 0x298 partial log-sum-exp, 0x2A0 splits in {2, 4, 8}, 0x2A4 rows_total = B S H; grid z = B splits) - the
/// partial layout of the MXFP8 `_kvsplit`, merged by `fmha_fwd_hd128_mxfp8_v3_kvsplit_combine8h`.
fn declare_io_pt(b: &mut FnBuilder, pair: bool, kvsplit: bool) -> FmhaIo {
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
    // the descales: [B, HK] f32, dense (the [1] scalar form of the aiter API = all strides 0 also runs, outside this contract)
    let desc = |ka: &mut KArgs, b: &mut FnBuilder, name: &str, off: u32, bs: (&'static str, u32), hs: (&'static str, u32)| {
        ka.tensor(name, off, 4, vec![bat.clone(), hk.clone()], vec![Stride::Ka(bs.0, bs.1), Stride::Ka(hs.0, hs.1)]).packed().declare(ka, b)
    };
    let qsc = desc(ka, b, "ptr_q_descale", 0x200, ("s_descale_q_Bs", 0x230), ("s_descale_q_Hs", 0x240));
    let ksc = desc(ka, b, "ptr_k_descale", 0x210, ("s_descale_k_Bs", 0x250), ("s_descale_k_Hs", 0x260));
    let vsc = desc(ka, b, "ptr_v_descale", 0x220, ("s_descale_v_Bs", 0x270), ("s_descale_v_Hs", 0x280));
    ka.arg(b, "scalar", DType::F32, 0x050, 4);
    // the contract, from the layouts (bytes), as `declare_io`
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
    b.grid_is(1, Ext::ka(0x140));
    // causal: two q tiles per block (the vendor's pairing), the grid is halved
    b.grid_is(0, if pair { Ext::ka(0x060).ceil_div(256).ceil_div(2) } else { Ext::ka(0x060).ceil_div(256) });
    let kvs = kvsplit.then(|| {
        let splits = ka.dim(b, "kvs_splits", 0x2A0, 1, i64::MAX).val().expect("splits");
        let rows_total = ka.dim(b, "kvs_rows_total", 0x2A4, 1, i64::MAX).val().expect("rows");
        b.grid_is(2, Ext::sym("B") * Ext::ka(0x2A0));
        b.ka_is(0x2A4, Ext::sym("B") * Ext::ka(0x060) * Ext::ka(0x140));
        let ohat = ka.ptr(b, "ptr_ohat", 0x290);
        let lse = ka.ptr(b, "ptr_lse_part", 0x298);
        // fp16 partials in an fp32-sized allocation (as the MXFP8 split)
        b.arg_tensor("ptr_ohat", Ext::ka(0x2A0) * Ext::ka(0x2A4) * Ext::c(128 * 4));
        b.arg_tensor("ptr_lse_part", Ext::ka(0x2A0) * Ext::ka(0x2A4) * Ext::c(4));
        io::KvsArgs { ohat, lse, splits: b.bind_i32(splits), rows_total: b.bind_i32(rows_total), splits_off: 0x2A0 }
    });
    let ptr_vd = vsc.ptr;
    FmhaIo { q, k, v, o, qsc, ksc, ptr_vd, vsc: Some(vsc), gqa, heads, s_len: s.val().expect("S"), kv_len: skv.val().expect("SK"), kvs }
}

/// The context of the per-tensor kernels.
pub(crate) struct PtCx {
    io: FmhaIo,
    blk: FmhaBlock,
    roles: Roles,
    ring: KvRing3,
    pipe: Option<Pipeline>,
    lut: ValueId,
}

/// The `_vs128` slots for the per-tensor FP8 kernels.
pub(crate) struct PtSlots {
    causal: bool,
}

fn pt_pipeline(blk: &FmhaBlock) -> Pipeline {
    let mut pipe = Pipeline::new(Cadence::TwoPhaseOffset, blk.nkv, blk.kv_len, 128, Ext::ka(0x110).ceil_div(128));
    pipe.key_unclamped = true;
    pipe
}

/// The causal mask of the block: lane limit min(row + SK - SQ + 1, kv_len), the wave's smallest limit (its first row) wave-uniform.
fn causal_mask(b: &mut FnBuilder, io: &FmhaIo, blk: &FmhaBlock) -> CausalMask {
    // a KV split: keys relative to the split's first key
    let qoff = if blk.split.is_some() { b.bind_i32(io.kv_len.clone() - io.s_len.clone() - blk.key_lo.clone()) } else { b.bind_i32(io.kv_len.clone() - io.s_len.clone()) };
    let lm = b.bind_i32((&blk.row0 + ev(qoff) + 1).min(ev(blk.kv_len)));
    let lim_min = rfl(b, lm);
    let neg_big = b.c_f32(-1.0e30);
    let qoff1 = { let v = b.bind_i32(ev(qoff) + 1); rfl(b, v) };
    CausalMask { qoff1, lim_min, neg_big }
}

/// One pass over the block's q tile: the vs128 dispatch (versions x roles) with the per-tensor body.
fn run_pass(b: &mut FnBuilder, io: &FmhaIo, blk: &FmhaBlock, roles: &Roles, ring: &KvRing3, pipe: &Pipeline, lut: ValueId, causal: bool) {
    let cm = causal.then(|| causal_mask(b, io, blk));
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
            body.causal = cm;
            pipe.run(b, role, &mut body);
            let mut fin = body.out_scale(b);
            let mut any = None;
            if causal {
                // SQ > SK: rows above the diagonal see no key (l = 0): O = 0 (as the reference / the vendor), not 0 * inf; a causal split: rows
                // with no key in this part, and every row of an empty part
                let l = body.sm.l_total(b);
                let z = b.c_f32(0.0);
                let a = b.cmp_lt_f32(z, l);
                fin = b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(a), a: Opnd::Val(fin), b: Opnd::Val(z) });
                if let Some(e) = blk.split_empty {
                    let keep = b.bind_cond(ev(e).eq(0));
                    fin = b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(keep), a: Opnd::Val(fin), b: Opnd::Val(z) });
                    any = Some((a, Some(keep)));
                } else {
                    any = Some((a, None));
                }
            }
            if io.kvs.is_some() {
                let l = body.sm.l_total(b);
                let mut lse = body.sm.lse(b, l);
                if let Some((a, keep)) = any {
                    let ninf = b.c_f32(f32::NEG_INFINITY);
                    lse = b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(a), a: Opnd::Val(lse), b: Opnd::Val(ninf) });
                    if let Some(k) = keep {
                        lse = b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(k), a: Opnd::Val(lse), b: Opnd::Val(ninf) });
                    }
                }
                store_partial3(io, b, blk, &ring.epi, wid, &body.o.expect("O"), fin, lse);
                return;
            }
            // causal (two passes): the guard covers only the global stores (verifier path count)
            store_out3_lds_x(io, b, blk, &ring.epi, wid, &body.o.expect("O"), fin, causal);
        });
    });
}

impl skeleton_vs128::Vs128Slots for PtSlots {
    type Cx = PtCx;
    fn prelude(&mut self, b: &mut FnBuilder) -> PtCx {
        b.compile_profile(CompileProfile::Vs128);
        let io = declare_io_pt(b, self.causal, false);
        let mode = Some(PtMode { causal: self.causal });
        // non-causal: the XCD block order of `_vs128`; causal (paired): the plain order
        let blk = if self.causal {
            let (x, y) = (b.let_(Op::WorkgroupIdX), b.let_(Op::WorkgroupIdY));
            block3_at(&io, b, 8, x, y, mode)
        } else {
            block3_x_pt(&io, b, mode)
        };
        let roles = Roles::split(b, &blk.lane, 4);
        let mut ring = KvRing3::new_x(b, &io, &blk, &roles, 2, false);
        ring.enable_buffer_dma(b);
        let pipe = pt_pipeline(&blk);
        let lut = blk.lut_copy(b);
        PtCx { io, blk, roles, ring, pipe: Some(pipe), lut }
    }
    fn pipeline(&mut self, _b: &mut FnBuilder, cx: &mut PtCx) -> Pipeline {
        cx.pipe.take().expect("the pipeline is taken once")
    }
    fn dispatch(&mut self, b: &mut FnBuilder, cx: &mut PtCx, pipe: &Pipeline) {
        let (io, blk, roles, ring, lut) = (&cx.io, &cx.blk, &cx.roles, &cx.ring, cx.lut);
        run_pass(b, io, blk, roles, ring, pipe, lut, self.causal);
        if !self.causal {
            return;
        }
        // the paired q tile nt - 1 - x (when it differs from x): the ring is reused - every wave is done with the first pass
        let (x, y) = (b.let_(Op::WorkgroupIdX), b.let_(Op::WorkgroupIdY));
        let nt = b.bind_i32((io.s_len.clone() + 255).lshr(8));
        let qb = { let v = b.bind_i32((ev(nt) - 1 - ev(x)).max(0)); rfl(b, v) };
        let second = b.bind_cond(ev(qb).gt(ev(x)));
        b.wait_lds(0);
        b.barrier();
        b.begin_if(Opnd::Val(second));
        let blk2 = block3_at(io, b, 8, qb, y, Some(PtMode { causal: true }));
        // fresh roles and ring partitions: reusing the first pass's address registers keeps them live through its last (unstaged) tile
        let roles2 = Roles::split(b, &blk2.lane, 4);
        let ring2 = ring.rebuilt(b, io, &blk2, &roles2, 2);
        let pipe2 = pt_pipeline(&blk2);
        let lut2 = blk2.lut_copy(b);
        run_pass(b, io, &blk2, &roles2, &ring2, &pipe2, lut2, true);
        b.end_if();
    }
}

/// `block3_x(.., xcd = true)` for the per-tensor form.
fn block3_x_pt(io: &FmhaIo, b: &mut FnBuilder, mode: Option<PtMode>) -> FmhaBlock {
    block3_x_mode(io, b, 8, true, mode)
}

/// Per-tensor FP8 prefill, hd128, non-causal.
pub fn fmha_fwd_hd128_fp8pt_v3() -> Function {
    let mut f = skeleton_vs128::build_vs128("fmha_fwd_hd128_fp8pt_v3", &mut PtSlots { causal: false });
    f.sched_opts = Some("ldsink=1,wsink,ldahead32=1".to_string());
    f
}

/// Per-tensor FP8 prefill, hd128, bottom-right causal (q-tile pairing).
pub fn fmha_fwd_hd128_fp8pt_v3_causal() -> Function {
    let mut f = skeleton_vs128::build_vs128("fmha_fwd_hd128_fp8pt_v3_causal", &mut PtSlots { causal: true });
    f.sched_opts = Some("ldsink=1,wsink,ldahead32=1".to_string());
    f
}

/// `read_first_lane` as an i32 value (the result is typed u32).
fn rfl(b: &mut FnBuilder, v: ValueId) -> ValueId {
    let r = b.read_first_lane(v);
    b.let_(Op::AddI32(Opnd::Val(r), Opnd::I32(0)))
}

/// The `_kvsplit` slots for the per-tensor FP8 kernels (small grids): the q tile's key range (causal: up to its diagonal) cut into
/// `splits` parts (grid z = B splits), fp16 partials + log-sum-exp, merged by `fmha_fwd_hd128_mxfp8_v3_kvsplit_combine8h`. One q tile per
/// block (no pairing), the `_vs128` body (MUBUF staging, ping-pong priorities, one bias MFMA).
pub(crate) struct PtKvSlots {
    causal: bool,
}

impl skeleton_kvsplit::KvsplitSlots for PtKvSlots {
    type Cx = PtCx;
    fn prelude(&mut self, b: &mut FnBuilder) -> PtCx {
        b.compile_profile(CompileProfile::Vs128);
        let io = declare_io_pt(b, false, true);
        let (x, y) = (b.let_(Op::WorkgroupIdX), b.let_(Op::WorkgroupIdY));
        let blk = block3_at(&io, b, 8, x, y, Some(PtMode { causal: self.causal }));
        let roles = Roles::split(b, &blk.lane, 4);
        let mut ring = KvRing3::new_x(b, &io, &blk, &roles, 2, false);
        ring.enable_buffer_dma(b);
        let pipe = pt_pipeline(&blk);
        let lut = blk.lut_copy(b);
        PtCx { io, blk, roles, ring, pipe: Some(pipe), lut }
    }
    fn pipeline(&mut self, _b: &mut FnBuilder, cx: &mut PtCx) -> Pipeline {
        cx.pipe.take().expect("the pipeline is taken once")
    }
    fn dispatch(&mut self, b: &mut FnBuilder, cx: &mut PtCx, pipe: &Pipeline) {
        run_pass(b, &cx.io, &cx.blk, &cx.roles, &cx.ring, pipe, cx.lut, self.causal);
    }
}

/// Per-tensor FP8 prefill, KV split (phase 1; the combine is `fmha_fwd_hd128_mxfp8_v3_kvsplit_combine8h`).
pub fn fmha_fwd_hd128_fp8pt_v3_kvsplit() -> Function {
    let mut f = skeleton_kvsplit::build_kvsplit("fmha_fwd_hd128_fp8pt_v3_kvsplit", &mut PtKvSlots { causal: false });
    f.sched_opts = Some("ldsink=1,wsink,ldahead32=1".to_string());
    f
}

/// Per-tensor FP8 prefill, causal, KV split over each q tile's visible keys.
pub fn fmha_fwd_hd128_fp8pt_v3_causal_kvsplit() -> Function {
    let mut f = skeleton_kvsplit::build_kvsplit("fmha_fwd_hd128_fp8pt_v3_causal_kvsplit", &mut PtKvSlots { causal: true });
    f.sched_opts = Some("ldsink=1,wsink,ldahead32=1".to_string());
    f
}