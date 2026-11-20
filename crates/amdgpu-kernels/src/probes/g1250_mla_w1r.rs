//! `mla_g1250_qh16_w1r`: the two-role window kernel, built on `mla_w1_parts::W1Parts` (kernel-opt) with the role split and the schedule of this file.
//!
//! 8 waves: waves 0 - 3 = MMA waves (the TDM stage, K / V fragments, the S WMMAs, the lazy O rescale and the PV WMMAs; the O accumulators live here), waves 4 - 7 = softmax waves (the masked raw
//! scores, the maxima, the lean P, the running max / sum). A SIMD hosts one wave of each role. The roles hand over through LDS rings: SBUF (the S tile, f32), RED (the per-wave maxima), PBUF (P, fp8),
//! ALPHA (the rescale factor), and two barriers per window (phase A: S and the maxima, phase B: P and PV). Iteration t:
//!   A: MMA waves: stage window t + 3, S(t + 2) -> SBUF gen t + 2.  softmax waves: S(t + 1) <- SBUF, masked raw, partial maxima -> RED gen t + 1.            barrier
//!   B: softmax waves: window maximum of t + 1 <- RED, P(t + 1) -> PBUF gen t + 1, alpha(t + 1) -> ALPHA, running max / sum.  MMA waves: PV(t) (P and alpha gen t).  barrier
//! so the ring slot of window t - 1 is free for the stage of window t + 3 (PV(t - 1) ended at the last barrier), as in `_w1p`.

use amdgpu_dsl::{ev, CmpPred, CmpTy, DType, Ext, FnBuilder, Function, Op, Opnd, ValueId, WmmaKind};
use amdgpu_stage::attention::mla::PageLayout;
use amdgpu_tile::shared::{Layout, LdsPlan, SlotIdx};

use super::mla_w1_parts::W1Parts;

pub fn mla_g1250_qh16_w1r() -> Function {
    let (wp, depth) = (2i32, 4i32);
    let in_flight = ((depth - 4) * wp) as u8;
    let layout = PageLayout::plan();
    let mut b = FnBuilder::new("mla_g1250_qh16_w1r", 256);
    b.compile_profile(amdgpu_dsl::CompileProfile::Gemm);
    W1Parts::declare_args(&mut b);
    let mut lds = LdsPlan::new();
    let kv = lds.ring("KV", (wp * layout.slot_bytes) as i64, depth as u32, Layout::Linear);
    let red = lds.ring("RED", 512, 2, Layout::Linear);
    let pbuf = lds.ring("PBUF", 1024 * wp as i64, 2, Layout::Linear);
    let alpha_r = lds.ring("ALPHA", 64, 2, Layout::Linear);
    let sbuf = lds.ring("SBUF", 8192, 2, Layout::Linear);
    let lsum = lds.tile("LSUM", 576, Layout::Linear);
    let out = lds.tile_over("OUT", 16 * 512 * 2, &[&kv]);
    lds.commit(&mut b);
    let tid = b.let_(Op::WorkitemId);
    let lane = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::I32(31)));
    let wv = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::I32(5)));
    let wfull = b.read_first_lane(wv);
    let w = b.let_(Op::AndI32(Opnd::Val(wfull), Opnd::I32(3)));
    let role = b.let_(Op::LshrI32(Opnd::Val(wfull), Opnd::I32(2)));
    let p = W1Parts::new(&mut b, layout, wp, depth, w, lane);
    let nwin = p.nwin(&mut b);
    let neg = p.neg;
    let zf = b.c_f32(0.0);
    let is_mma = {
        let c = b.cmp(CmpPred::Eq, CmpTy::I32, role, Opnd::I32(0));
        b.let_(Op::AnyLane(Opnd::Val(c)))
    };
    let is_sm = {
        let c = b.cmp(CmpPred::Eq, CmpTy::I32, role, Opnd::I32(1));
        b.let_(Op::AnyLane(Opnd::Val(c)))
    };
    // the S tile (two Frag8: [page][lane][8 f32]) through SBUF[slot], wave w's 2 KiB
    let sbuf_store = |b: &mut FnBuilder, gen: ValueId, slot: Opnd, s: &[ValueId]| {
        let r = sbuf.at(&SlotIdx::new(slot, Opnd::Val(gen)));
        let base = b.bind_i32(r.base() + ev(w) * 2048 + ev(lane) * 32);
        r.write(b, |b| {
            for pg in 0..wp as usize {
                for h in 0..2usize {
                    let part = b.frag_slice(Opnd::Val(s[pg]), (4 * h) as u8, 4);
                    let a = b.bind_i32(ev(base) + (1024 * pg as i32 + 16 * h as i32));
                    b.store_lds_b128(a, part);
                }
            }
        });
    };
    let sbuf_load = |b: &mut FnBuilder, gen: ValueId, slot: Opnd| -> Vec<ValueId> {
        let r = sbuf.at(&SlotIdx::new(slot, Opnd::Val(gen)));
        let base = b.bind_i32(r.base() + ev(w) * 2048 + ev(lane) * 32);
        r.read(b, |b| {
            let mut o = Vec::new();
            for pg in 0..wp as usize {
                for h in 0..2usize {
                    let a = b.bind_i32(ev(base) + (1024 * pg as i32 + 16 * h as i32));
                    let v4 = b.load_lds_b128(a);
                    for k in 0..4u8 {
                        let e = b.frag_slice(v4, k, 1);
                        o.push(b.let_(Op::BitcastF32(Opnd::Val(e))));
                    }
                }
            }
            o
        })
    };
    let c0 = b.c_i32(0);
    let c1 = b.c_i32(1);
    let c2 = b.c_i32(2);
    // ---- loop state: O (MMA waves), the running max / sum (softmax waves), the page ids of window t + depth - 1 (MMA waves) ----
    let mut inits: Vec<(DType, Opnd)> = vec![(DType::F32, Opnd::Val(neg)), (DType::F32, Opnd::Val(zf))];
    for _ in 0..8 {
        let z = b.frag_cat(&[Opnd::I32(0); 8]);
        inits.push((WmmaKind::Fp8_16x16x64.acc_ty(), Opnd::Val(z)));
    }
    let nx = 8 * wp as usize;
    for _ in 0..nx {
        inits.push((DType::F32, Opnd::Val(zf)));
    }
    let npid = wp as usize;
    for _ in 0..npid {
        inits.push((DType::I32, Opnd::Val(c0)));
    }
    let st = b.declare_vars(&inits);
    let (vm, vl) = (st[0], st[1]);
    let ov: Vec<_> = st[2..10].to_vec();
    let xv: Vec<_> = st[10..10 + nx].to_vec();
    let pidv: Vec<_> = st[10 + nx..10 + nx + npid].to_vec();

    // ---- prologue ----
    // windows 0 .. depth - 2 staged by the MMA waves
    b.begin_if(is_mma);
    for k in 0..(depth - 1) {
        let ck = b.c_i32(k);
        let pids: Vec<amdgpu_dsl::I32> = p.load_pids(&mut b, ck).into_iter().map(|v| ev(v)).collect();
        p.stage(&mut b, &kv, ck, &pids);
    }
    b.wait_tensor(in_flight);
    b.end_if();
    b.wait_lds(0);
    b.barrier();
    // S(0), S(1) -> SBUF gens 0, 1 (MMA waves)
    b.begin_if(is_mma);
    let k0 = p.k_frags(&mut b, &kv, c0);
    let s0 = p.s_wmma(&mut b, &k0);
    sbuf_store(&mut b, c0, Opnd::I32(0), &s0);
    let k1 = p.k_frags(&mut b, &kv, c1);
    let s1 = p.s_wmma(&mut b, &k1);
    sbuf_store(&mut b, c1, Opnd::I32(1), &s1);
    let tf = b.c_i32(depth - 1);
    let pf = p.load_pids(&mut b, tf);
    for k in 0..npid {
        b.set_var(pidv[k], Opnd::Val(pf[k]));
    }
    b.end_if();
    b.wait_lds(0);
    b.barrier();
    // softmax waves: x(0) masked, its maxima -> RED gen 0
    b.begin_if(is_sm);
    let f0 = sbuf_load(&mut b, c0, Opnd::I32(0));
    let x0 = p.raw_masked(&mut b, c0, &f0);
    p.red_write(&mut b, &red, c0, &x0);
    for k in 0..nx {
        b.set_var(xv[k], x0[k]);
    }
    b.end_if();
    b.wait_lds(0);
    b.barrier();
    // softmax waves: P(0), alpha(0) (the max of window 0 is visible now)
    b.begin_if(is_sm);
    let x0v: Vec<Opnd> = xv.iter().map(|&v| Opnd::Var(v)).collect();
    let pm0 = p.red_read_max(&mut b, &red, c0);
    let (pv0, alpha0, m0, l0) = p.p_lean(&mut b, &x0v, Opnd::Val(neg), Opnd::Val(zf), pm0);
    p.p_store(&mut b, &pbuf, c0, &pv0);
    p.alpha_store(&mut b, &alpha_r, c0, alpha0);
    b.set_var(vm, Opnd::Val(m0));
    b.set_var(vl, Opnd::Val(l0));
    b.end_if();
    b.wait_lds(0);
    b.barrier();
    let _ = c2;

    // ---- the loops ----
    let z = b.c_i32(0);
    b.begin_if(is_mma);
    {
        let lvars = b.begin_loop(&[(DType::I32, Opnd::Val(z))]);
        let it = lvars[0];
        b.assume(it, Ext::c(0), Ext::c(512));
        let t = b.bind_i32(amdgpu_dsl::varx(it) + 0);
        let t2 = b.bind_i32(ev(t) + 2);
        let t3 = b.bind_i32(ev(t) + (depth - 1));
        // phase A: stage window t + 3 (page ids loaded last iteration), then S(t + 2) -> SBUF gen t + 2
        let pe: Vec<amdgpu_dsl::I32> = pidv.iter().map(|&v| amdgpu_dsl::varx(v) + 0).collect();
        p.stage(&mut b, &kv, t3, &pe);
        let t4 = b.bind_i32(ev(t3) + 1);
        let np = p.load_pids(&mut b, t4);
        for k in 0..np.len() {
            b.set_var(pidv[k], Opnd::Val(np[k]));
        }
        let kf = p.k_frags(&mut b, &kv, t2);
        let s2 = p.s_wmma(&mut b, &kf);
        let sl2 = b.bind_i32(ev(t2) & 1);
        sbuf_store(&mut b, t2, Opnd::Val(sl2), &s2);
        b.wait_lds(0);
        b.barrier();
        // phase B: PV(t) with P and alpha of gen t
        let pa = p.p_load(&mut b, &pbuf, t);
        let vf = p.v_frags(&mut b, &kv, t);
        let (rows, grew) = p.alpha_rows_load(&mut b, &alpha_r, t);
        p.rescale_lazy(&mut b, &ov, &rows, grew);
        p.pv_wmma(&mut b, &ov, &pa, &vf);
        b.wait_tensor(in_flight);
        b.wait_lds(0);
        b.barrier();
        let tn = b.bind_i32(amdgpu_dsl::varx(it) + 1);
        b.set_var(it, Opnd::Val(tn));
        let fin = b.cmp(CmpPred::Ge, CmpTy::I32, tn, nwin);
        b.break_if(fin);
        b.end_loop();
    }
    b.else_();
    {
        let lvars = b.begin_loop(&[(DType::I32, Opnd::Val(z))]);
        let it = lvars[0];
        b.assume(it, Ext::c(0), Ext::c(512));
        let t = b.bind_i32(amdgpu_dsl::varx(it) + 0);
        let t1 = b.bind_i32(ev(t) + 1);
        // phase A: S(t + 1) <- SBUF, masked raw scores, the partial maxima -> RED gen t + 1
        let sl1 = b.bind_i32(ev(t1) & 1);
        let f1 = sbuf_load(&mut b, t1, Opnd::Val(sl1));
        let x1 = p.raw_masked(&mut b, t1, &f1);
        p.red_write(&mut b, &red, t1, &x1);
        b.wait_lds(0);
        b.barrier();
        // phase B: the window maximum, P(t + 1), alpha(t + 1)
        let pm1 = p.red_read_max(&mut b, &red, t1);
        let (pv1, alpha1, m1, l1) = p.p_lean(&mut b, &x1, Opnd::Var(vm), Opnd::Var(vl), pm1);
        p.p_store(&mut b, &pbuf, t1, &pv1);
        p.alpha_store(&mut b, &alpha_r, t1, alpha1);
        b.set_var(vm, Opnd::Val(m1));
        b.set_var(vl, Opnd::Val(l1));
        b.wait_lds(0);
        b.barrier();
        let tn = b.bind_i32(amdgpu_dsl::varx(it) + 1);
        b.set_var(it, Opnd::Val(tn));
        let fin = b.cmp(CmpPred::Ge, CmpTy::I32, tn, nwin);
        b.break_if(fin);
        b.end_loop();
    }
    b.end_if();
    b.wait_tensor(0);
    // ---- epilogue: the softmax waves publish the row sums and maxima; the MMA waves normalise and store (the epilogue has one internal barrier: the softmax waves meet it) ----
    b.begin_if(is_sm);
    p.lsum_write(&mut b, &lsum, Opnd::Var(vm), Opnd::Var(vl));
    b.end_if();
    b.wait_lds(0);
    b.barrier();
    b.begin_if(is_mma);
    let tid_m = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::I32(127)));
    p.epilogue(&mut b, &ov, &lsum, &out, tid_m);
    b.else_();
    b.barrier();
    b.end_if();
    b.finish()
}
