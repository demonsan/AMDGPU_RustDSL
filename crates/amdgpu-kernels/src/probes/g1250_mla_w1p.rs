//! `mla_g1250_qh16_w1p`: the one-barrier window kernel (`_w1v`'s lean form) rebuilt on `mla_w1_parts::W1Parts` (the 1-role reference
//! of the parts the role kernels use). Schedule per iteration t (after barrier B(t-1)):
//!   M(t+1) from RED, P(t+1) from the carried masked raw S(t+1) -> PBUF; stage window t+3 (page ids loaded one iteration ahead);
//!   K(t+2), P(t), V(t) fragments; lazy O rescale by alpha(t); S(t+2) and PV(t) WMMAs; mask / max of S(t+2) -> RED; wait; B(t).

use amdgpu_dsl::{ev, CmpPred, CmpTy, DType, Ext, FnBuilder, Function, Opnd, WmmaKind, Op};
use amdgpu_stage::attention::mla::PageLayout;
use amdgpu_tile::shared::{Layout, LdsPlan};

use super::mla_w1_parts::W1Parts;

pub fn mla_g1250_qh16_w1p() -> Function {
    let (wp, depth) = (2i32, 4i32);
    let in_flight = ((depth - 4) * wp) as u8;
    let layout = PageLayout::plan();
    let mut b = FnBuilder::new("mla_g1250_qh16_w1p", 128);
    b.compile_profile(amdgpu_dsl::CompileProfile::Gemm);
    W1Parts::declare_args(&mut b);
    let mut lds = LdsPlan::new();
    let kv = lds.ring("KV", (wp * layout.slot_bytes) as i64, depth as u32, Layout::Linear);
    let red = lds.ring("RED", 512, 2, Layout::Linear);
    let pbuf = lds.ring("PBUF", 1024 * wp as i64, 2, Layout::Linear);
    let lsum = lds.tile("LSUM", 576, Layout::Linear);
    let out = lds.tile_over("OUT", 16 * 512 * 2, &[&kv]);
    lds.commit(&mut b);
    let tid = b.let_(Op::WorkitemId);
    let lane = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::I32(31)));
    let wv = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::I32(5)));
    let w = b.read_first_lane(wv);
    let p = W1Parts::new(&mut b, layout, wp, depth, w, lane);
    let nwin = p.nwin(&mut b);
    let neg = p.neg;
    let zf = b.c_f32(0.0);

    // ---- prologue: windows 0 .. depth - 2 staged; S(0), S(1) masked and their maxima; P(0) ----
    for k in 0..(depth - 1) {
        let ck = b.c_i32(k);
        let pids: Vec<amdgpu_dsl::I32> = p.load_pids(&mut b, ck).into_iter().map(|v| ev(v)).collect();
        p.stage(&mut b, &kv, ck, &pids);
    }
    b.wait_tensor(in_flight);
    b.barrier();
    let c0 = b.c_i32(0);
    let c1 = b.c_i32(1);
    let k0 = p.k_frags(&mut b, &kv, c0);
    let s0 = p.s_wmma(&mut b, &k0);
    let f0 = p.s_floats(&mut b, &s0);
    let x0 = p.raw_masked(&mut b, c0, &f0);
    p.red_write(&mut b, &red, c0, &x0);
    let k1 = p.k_frags(&mut b, &kv, c1);
    let s1 = p.s_wmma(&mut b, &k1);
    let f1 = p.s_floats(&mut b, &s1);
    let x1 = p.raw_masked(&mut b, c1, &f1);
    p.red_write(&mut b, &red, c1, &x1);
    b.wait_lds(0);
    b.barrier();
    let pm0 = p.red_read_max(&mut b, &red, c0);
    let (pv0, alpha0, m0, l0) = p.p_lean(&mut b, &x0, Opnd::Val(neg), Opnd::Val(zf), pm0);
    p.p_store(&mut b, &pbuf, c0, &pv0);
    b.wait_lds(0);
    b.barrier();

    // ---- loop state: m, l, alpha(t), O (8 tiles), masked raw S(t + 1), page ids of window t + depth - 1 ----
    let mut inits = vec![(DType::F32, Opnd::Val(m0)), (DType::F32, Opnd::Val(l0)), (DType::F32, Opnd::Val(alpha0))];
    for _ in 0..8 {
        let z = b.frag_cat(&[Opnd::I32(0); 8]);
        inits.push((WmmaKind::Fp8_16x16x64.acc_ty(), Opnd::Val(z)));
    }
    for x in &x1 {
        inits.push((DType::F32, *x));
    }
    let tf = b.c_i32(depth - 1);
    for pv in p.load_pids(&mut b, tf) {
        inits.push((DType::I32, Opnd::Val(pv)));
    }
    let st = b.declare_vars(&inits);
    let (vm, vl, va) = (st[0], st[1], st[2]);
    let ov: Vec<_> = st[3..11].to_vec();
    let nx = 8 * wp as usize;
    let xv: Vec<_> = st[11..11 + nx].to_vec();
    let pidv: Vec<_> = st[11 + nx..].to_vec();
    let z = b.c_i32(0);
    let lvars = b.begin_loop(&[(DType::I32, Opnd::Val(z))]);
    let it = lvars[0];
    b.assume(it, Ext::c(0), Ext::c(512));
    let t = b.bind_i32(amdgpu_dsl::varx(it) + 0);
    let t1 = b.bind_i32(ev(t) + 1);
    let t2 = b.bind_i32(ev(t) + 2);
    let t3 = b.bind_i32(ev(t) + (depth - 1));
    // M(t + 1), P(t + 1)
    let pm1 = p.red_read_max(&mut b, &red, t1);
    let xo: Vec<Opnd> = xv.iter().map(|&v| Opnd::Var(v)).collect();
    let (pv1, alpha1, m1, l1) = p.p_lean(&mut b, &xo, Opnd::Var(vm), Opnd::Var(vl), pm1);
    p.p_store(&mut b, &pbuf, t1, &pv1);
    // stage window t + depth - 1 (its page ids loaded last iteration); load those of the next window
    let pe: Vec<amdgpu_dsl::I32> = pidv.iter().map(|&v| amdgpu_dsl::varx(v) + 0).collect();
    p.stage(&mut b, &kv, t3, &pe);
    let t4 = b.bind_i32(ev(t3) + 1);
    let np = p.load_pids(&mut b, t4);
    for k in 0..np.len() {
        b.set_var(pidv[k], Opnd::Val(np[k]));
    }
    // fragments: K(t + 2), P(t), V(t)
    let kf = p.k_frags(&mut b, &kv, t2);
    let pa = p.p_load(&mut b, &pbuf, t);
    let vf = p.v_frags(&mut b, &kv, t);
    // O *= alpha(t), lazily
    let (rows, grew) = p.alpha_rows_from_lanes(&mut b, Opnd::Var(va));
    p.rescale_lazy(&mut b, &ov, &rows, grew);
    // S(t + 2), PV(t)
    let s2 = p.s_wmma(&mut b, &kf);
    p.pv_wmma(&mut b, &ov, &pa, &vf);
    let f2 = p.s_floats(&mut b, &s2);
    let x2 = p.raw_masked(&mut b, t2, &f2);
    p.red_write(&mut b, &red, t2, &x2);
    b.set_var(vm, Opnd::Val(m1));
    b.set_var(vl, Opnd::Val(l1));
    b.set_var(va, Opnd::Val(alpha1));
    for k in 0..nx {
        b.set_var(xv[k], x2[k]);
    }
    b.wait_tensor(in_flight);
    b.wait_lds(0);
    b.barrier();
    let tn = b.bind_i32(amdgpu_dsl::varx(it) + 1);
    b.set_var(it, Opnd::Val(tn));
    let fin = b.cmp(CmpPred::Ge, CmpTy::I32, tn, nwin);
    b.break_if(fin);
    b.end_loop();
    b.wait_tensor(0);
    // ---- epilogue ----
    p.lsum_write(&mut b, &lsum, Opnd::Var(vm), Opnd::Var(vl));
    b.wait_lds(0);
    b.barrier();
    p.epilogue(&mut b, &ov, &lsum, &out, tid);
    b.finish()
}
