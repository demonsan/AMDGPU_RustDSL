//! L2 template T4 (docs/l2-templates.md §3.4), merge variant (a): the separate log-sum-exp combine kernels of the attention key splits.
//! The split side (index maths, partial stores) is in `components` / `epilogue`; the kernel-level choice of S and of the merge variant
//! stays in the kernels / dispatch. Variants (b) in-CTA LDS merge and (c) last-arriver have one user each (`pa_decode16_fused`,
//! `pa_decode16_lastc`) and stay in `fmha/decode16.rs` until a second user exists (the generic test of §0).

use amdgpu_dsl::{ev, CompileProfile, DType, Ext, FnBuilder, Function, Op, Opnd, ValueId};

/// The per-(batch, head) partial layout of a split decode / prefill: `dd` dims per row, partials `[B][H][NS]` or (split-major)
/// `[B][NS][H]`, optional attention sink.
pub struct LseCombine {
    pub dd: i32,
    pub split_major: bool,
    pub sink: bool,
}

/// Merges NS splits: O[b, h, d] = sum_s exp(lse_s - M) po_s[d] / sum_s exp(lse_s - M) (bf16), M = max_s lse_s; 0 when every split is
/// empty. Grid (H, B), `dd` threads (one per d). ABI: 0 ptr_o (bf16 [B, H, D]), 1 ptr_po, 2 ptr_plse, 3 H, 4 NS (16-byte slots).
/// `sink` (gpt-oss attention sinks): a per-head logit sink[h] (f32 [H], ptr @80) joins the softmax denominator only:
/// O = sum_s e^(lse_s - M) po_s / (sum_s e^(lse_s - M) + e^(sink_h - M)), M = max(max_s lse_s, sink_h).
pub fn lse_combine(name: &str, p: &LseCombine) -> Function {
    let (dd, split_major, sink) = (p.dd, p.split_major, p.sink);
    let (dv, rowb) = (dd, 2 * dd);
    let mut b = FnBuilder::new(name, dd as u32);
    b.compile_profile(CompileProfile::Gemm);
    for (n, i) in [("ptr_o", 0u32), ("ptr_po", 1), ("ptr_plse", 2)] {
        b.arg(n, DType::Ptr, 16 * i, 8);
    }
    b.arg("H", DType::I32, 48, 4);
    b.arg("NS", DType::I32, 64, 4);
    if sink {
        b.arg("ptr_sink", DType::Ptr, 80, 8);
    }
    let (hh, ns) = (Ext::ka(48), Ext::ka(64));
    b.sym_range("B", 1, 1 << 16);
    b.ka_range(48, 1, 1 << 12);
    b.ka_range(64, 1, 256);
    b.grid_is(0, hh.clone());
    b.grid_is(1, Ext::sym("B"));
    b.arg_tensor("ptr_o", Ext::sym("B") * hh.clone() * Ext::c(rowb as i64));
    b.arg_tensor("ptr_po", Ext::sym("B") * hh.clone() * ns.clone() * Ext::c(4 * dv as i64));
    if sink {
        b.arg_tensor("ptr_sink", hh.clone() * Ext::c(4));
    }
    b.arg_tensor("ptr_plse", Ext::sym("B") * hh * ns * Ext::c(4));
    let (p_o, p_po, p_l) = (b.let_(Op::KernargPtr { offset: 0 }), b.let_(Op::KernargPtr { offset: 16 }), b.let_(Op::KernargPtr { offset: 32 }));
    let hq = b.let_(Op::KernargI32 { offset: 48 });
    let nsv = b.let_(Op::KernargI32 { offset: 64 });
    let h = b.let_(Op::WorkgroupIdX);
    let bat = b.let_(Op::WorkgroupIdY);
    let tid = b.let_(Op::WorkitemId);
    let row = { let v = b.bind_i32(ev(bat) * ev(hq) + ev(h)); b.read_first_lane(v) };
    // partial (split s) of this (b, h): index base + s step
    let (pbase, pstep) = if split_major {
        (b.bind_i32(ev(bat) * ev(nsv) * ev(hq) + ev(h)), b.bind_i32(ev(hq)))
    } else {
        (b.bind_i32(ev(row) * ev(nsv)), b.c_i32(1))
    };
    let (pbase, pstep) = (b.read_first_lane(pbase), b.read_first_lane(pstep));
    let r_l = { let n = b.bind_i32((ev(bat) + 1) * ev(hq) * ev(nsv) << 2); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_l, n) };
    let r_po = { let n = b.bind_i32((ev(bat) + 1) * ev(hq) * ev(nsv) * (4 * dv)); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_po, n) };
    let r_o = { let n = b.bind_i32((ev(row) + 1) * rowb); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_o, n) };
    let l_so = { let v = b.bind_i32(ev(pbase) << 2); b.read_first_lane(v) };
    let c0 = b.c_i32(0);
    let neg_inf = b.c_f32(f32::NEG_INFINITY);
    // pass 1: M
    let v1 = b.begin_loop(&[(DType::I32, Opnd::I32(0)), (DType::F32, Opnd::Val(neg_inf))]);
    let s1 = amdgpu_dsl::varx(v1[0]);
    let off = b.bind_i32((s1.clone() * ev(pstep)) << 2);
    let x = b.buffer_load_f32(r_l, off, l_so, 0);
    let mx = b.max_f32(Opnd::Var(v1[1]), x);
    b.set_var(v1[1], mx);
    let n1 = b.bind_i32(s1 + 1);
    b.set_var(v1[0], n1);
    let d1 = b.bind_cond(ev(n1).ge(ev(nsv)));
    b.break_if(d1);
    b.end_loop();
    // sinks: the head's sink logit joins the max (and the denominator below)
    let snk = sink.then(|| {
        let p = b.let_(Op::KernargPtr { offset: 80 });
        let n = { let v = b.bind_i32(ev(hq) << 2); b.read_first_lane(v) };
        let r = b.buffer_desc_dyn(p, n);
        let so = { let v = b.bind_i32(ev(h) << 2); b.read_first_lane(v) };
        b.buffer_load_f32(r, c0, so, 0)
    });
    let m = match snk { Some(s) => Opnd::Val(b.max_f32(Opnd::Var(v1[1]), s)), None => Opnd::Var(v1[1]) };
    // all empty -> M = -inf: use 0 (the weights exp(-inf - 0) = 0, O = 0 / max(sum, tiny))
    let mbits = b.bitcast_i32(m);
    let fin = b.bind_cond(ev(mbits).ne(0xff80_0000u32 as i32));
    let zf = b.c_f32(0.0);
    let mm = b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(fin), a: m, b: Opnd::Val(zf) });
    // pass 2: sums
    let v2 = b.begin_loop(&[(DType::I32, Opnd::I32(0)), (DType::F32, Opnd::Val(zf)), (DType::F32, Opnd::Val(zf))]);
    let s2 = amdgpu_dsl::varx(v2[0]);
    let off = b.bind_i32((s2.clone() * ev(pstep)) << 2);
    let x = b.buffer_load_f32(r_l, off, l_so, 0);
    let dx = b.sub_f32(x, mm);
    let l2e = b.c_f32(std::f32::consts::LOG2_E);
    let t = b.mul_f32(dx, l2e);
    let w = b.exp2_f32(t);
    let po_vo = b.bind_i32(((ev(pbase) + s2.clone() * ev(pstep)) * dv + ev(tid)) << 2);
    let o = b.buffer_load_f32(r_po, po_vo, c0, 0);
    let acc = b.let_(Op::FmaF32(Opnd::Val(w), Opnd::Val(o), Opnd::Var(v2[1])));
    b.set_var(v2[1], acc);
    let sw = b.add_f32(Opnd::Var(v2[2]), w);
    b.set_var(v2[2], sw);
    let n2 = b.bind_i32(s2 + 1);
    b.set_var(v2[0], n2);
    let d2 = b.bind_cond(ev(n2).ge(ev(nsv)));
    b.break_if(d2);
    b.end_loop();
    let tiny = b.c_f32(1e-30);
    let den = match snk {
        Some(s) => {
            let ds = b.sub_f32(s, mm);
            let l2e = b.c_f32(std::f32::consts::LOG2_E);
            let ts = b.mul_f32(ds, l2e);
            let ws = b.exp2_f32(ts);
            Opnd::Val(b.add_f32(Opnd::Var(v2[2]), ws))
        }
        None => Opnd::Var(v2[2]),
    };
    let den = b.max_f32(den, tiny);
    let inv = b.rcp_f32(den);
    let y = b.mul_f32(Opnd::Var(v2[1]), inv);
    let yb = b.cvt_pk_bf16_f32(y, y);
    let o_vo = b.bind_i32((ev(row) * dv + ev(tid)) << 1);
    b.buffer_store_n(r_o, Some(Opnd::Val(o_vo)), c0, 0, yb, amdgpu_dsl::MemTy::U16, false);
    b.finish()
}

/// Merges the S = 2^lg_s key parts of the bf16 v3 `_ks` prefill forms (hd128 value rows): O[b, q, h, :] (bf16, BSHD) = sum_s w_s po_s /
/// sum_s w_s with w_s = exp(lse_s - M); optional LSE [B, H, SQ] = M + ln(sum w). ABI: 0 ptr_o, 8 ptr_po, 16 ptr_plse, 24 ptr_lse, 32 H,
/// 36 SQ, 40 S, 44 lse_on. 256 threads = 16 query rows of one (batch, head) x 16 threads, 8 dims per thread, one online pass over the parts
/// (running max / rescale); grid (ceil(SQ / 16), H, B).
pub fn lse_combine_rows16_hd128(name: &str) -> Function {
    const D: i32 = 128;
    const ROW: i32 = 2 * D;
    let mut b = FnBuilder::new(name, 256);
    b.compile_profile(CompileProfile::Gemm);
    for (n, o) in [("ptr_o", 0u32), ("ptr_po", 8), ("ptr_plse", 16), ("ptr_lse", 24)] {
        b.arg(n, DType::Ptr, o, 8);
    }
    for (n, o) in [("H", 32u32), ("SQ", 36), ("S", 40), ("lse_on", 44)] {
        b.arg(n, DType::I32, o, 4);
    }
    let (hh, sq) = (Ext::ka(32), Ext::ka(36));
    b.ka_range(32, 1, 256);
    b.ka_range(36, 1, 1 << 20);
    b.sym_range("SPL", 1, 16);
    b.ka_is(40, Ext::sym("SPL"));
    b.sym_range("B", 1, 1 << 12);
    b.grid_is(0, sq.clone().ceil_div(16));
    b.grid_is(1, hh.clone());
    b.grid_is(2, Ext::sym("B"));
    let rows = Ext::sym("B") * hh * sq;
    b.arg_tensor("ptr_o", rows.clone() * Ext::c(ROW as i64));
    b.arg_tensor("ptr_po", rows.clone() * Ext::sym("SPL") * Ext::c(4 * D as i64));
    b.arg_tensor("ptr_plse", rows.clone() * Ext::sym("SPL") * Ext::c(4));
    b.arg_tensor("ptr_lse", rows * Ext::c(4));
    let kp = |b: &mut FnBuilder, o: u32| b.let_(Op::KernargPtr { offset: o });
    let ki = |b: &mut FnBuilder, o: u32| b.let_(Op::KernargI32 { offset: o });
    let (p_o, p_po, p_pl, p_l) = (kp(&mut b, 0), kp(&mut b, 8), kp(&mut b, 16), kp(&mut b, 24));
    let (hq, s_q, ns, lse_on) = (ki(&mut b, 32), ki(&mut b, 36), ki(&mut b, 40), ki(&mut b, 44));
    let (qb, h, bat) = (b.let_(Op::WorkgroupIdX), b.let_(Op::WorkgroupIdY), b.let_(Op::WorkgroupIdZ));
    let tid = b.let_(Op::WorkitemId);
    // the thread's query row (clamped for the loads; the stores are guarded) and dims 8 (tid % 16) ..
    let q = b.bind_i32((ev(qb) << 4) + ev(tid).lshr(4));
    let live = b.bind_cond(ev(q).lt(ev(s_q)));
    let qc = b.bind_i32(ev(q).min(ev(s_q) - 1));
    let d0 = b.bind_i32((ev(tid) & 15) << 3);
    let row = b.bind_i32((ev(bat) * ev(hq) + ev(h)) * ev(s_q) + ev(qc));
    let nrow = { let v = b.bind_i32((((ev(bat) * ev(hq) + ev(h)) + 1) * ev(s_q)) * ev(ns)); b.read_first_lane(v) };
    let r_pl = { let n = b.bind_i32(ev(nrow) << 2); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_pl, n) };
    let r_po = { let n = b.bind_i32(ev(nrow) * (4 * D)); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_po, n) };
    let base = b.bind_i32(ev(row) * ev(ns));
    let c0 = b.c_i32(0);
    let neg_big = b.c_f32(-1.0e30);
    let zf = b.c_f32(0.0);
    let l2e = b.c_f32(std::f32::consts::LOG2_E);
    let mut inits = vec![(DType::I32, Opnd::Val(c0)), (DType::F32, Opnd::Val(neg_big)), (DType::F32, Opnd::Val(zf))];
    inits.extend((0..8).map(|_| (DType::F32, Opnd::Val(zf))));
    let v = b.begin_loop(&inits);
    let s = amdgpu_dsl::varx(v[0]);
    let lo = b.bind_i32((ev(base) + s.clone()) << 2);
    let l = b.buffer_load_f32(r_pl, lo, c0, 0);
    let po = b.bind_i32((((ev(base) + s.clone()) * D) + ev(d0)) << 2);
    let o0 = b.buffer_load_b128(r_po, po, c0, 0);
    let o1 = b.buffer_load_b128(r_po, po, c0, 16);
    let m_new = b.max_f32(Opnd::Var(v[1]), l);
    let d_old = b.sub_f32(Opnd::Var(v[1]), m_new);
    let d_new = b.sub_f32(l, m_new);
    let t_old = b.mul_f32(d_old, l2e);
    let t_new = b.mul_f32(d_new, l2e);
    let a = b.exp2_f32(t_old);
    let w = b.exp2_f32(t_new);
    let den_a = b.mul_f32(Opnd::Var(v[2]), a);
    let den = b.add_f32(den_a, w);
    for k in 0..8usize {
        let f = if k < 4 { o0 } else { o1 };
        let x = b.frag4_lane(f, (k % 4) as u8);
        let x = b.bitcast_f32(x);
        let acc_a = b.mul_f32(Opnd::Var(v[3 + k]), a);
        let acc = b.let_(Op::FmaF32(Opnd::Val(w), Opnd::Val(x), Opnd::Val(acc_a)));
        b.set_var(v[3 + k], acc);
    }
    b.set_var(v[1], m_new);
    b.set_var(v[2], den);
    let n1 = b.bind_i32(s + 1);
    b.set_var(v[0], n1);
    let done = b.bind_cond(ev(n1).ge(ev(ns)));
    b.break_if(done);
    b.end_loop();
    let tiny = b.c_f32(1e-30);
    let den = b.max_f32(Opnd::Var(v[2]), tiny);
    let inv = b.rcp_f32(den);
    let ys: Vec<ValueId> = (0..8).map(|k| b.mul_f32(Opnd::Var(v[3 + k]), inv)).collect();
    let w: Vec<ValueId> = (0..4).map(|k| b.cvt_pk_bf16_f32(ys[2 * k], ys[2 * k + 1])).collect();
    let f4 = b.frag4(w[0], w[1], w[2], w[3]);
    let nob = { let v = b.bind_i32(((ev(bat) + 1) * ev(s_q) * ev(hq)) * ROW); b.read_first_lane(v) };
    let r_o = b.buffer_desc_dyn(p_o, nob);
    let big = b.c_i32(0x7fff_fff0);
    let o_vo = b.bind_i32(((ev(bat) * ev(s_q) + ev(qc)) * ev(hq) + ev(h)) * ROW + (ev(d0) << 1));
    let o_vo = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(live), a: Opnd::Val(o_vo), b: Opnd::Val(big) });
    b.buffer_store_b128(r_o, o_vo, c0, 0, f4);
    let on = b.bind_cond(ev(lse_on).ne(0));
    b.begin_if(Opnd::Val(on));
    {
        // natural-log LSE of the row from its first thread
        let lg2 = b.log2_f32(den);
        let ln2 = b.c_f32(std::f32::consts::LN_2);
        let lnl = b.mul_f32(lg2, ln2);
        let lse = b.add_f32(Opnd::Var(v[1]), lnl);
        let nl = { let v = b.bind_i32(((ev(bat) * ev(hq) + ev(h)) + 1) * ev(s_q) << 2); b.read_first_lane(v) };
        let r_l = b.buffer_desc_dyn(p_l, nl);
        let first = b.bind_cond(ev(d0).eq(0));
        let lvo = b.bind_i32(ev(row) << 2);
        let lvo = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(first), a: Opnd::Val(lvo), b: Opnd::Val(big) });
        let lvo = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(live), a: Opnd::Val(lvo), b: Opnd::Val(big) });
        b.buffer_store_f32(r_l, lvo, c0, 0, lse);
    }
    b.end_if();
    b.finish()
}
