//! MLA decode (gfx1250), the window structure with ONE barrier per window (vendor-like: the 4 waves split each window's tokens, P goes
//! through LDS, each wave owns 128 V dims, no final cross-wave merge).
//!
//! The first window kernel (`mla_g1250_qh16`) pays two barriers per window (the max partials, then P / alpha) plus the ring's own. Here the
//! three per-window steps are software pipelined over windows so that one barrier publishes everything the next iteration needs:
//!
//! iteration t (after barrier B(t-1)):
//!   1. stage window t + 3 (into the ring slot of window t - 1, free since B(t-1));
//!   2. S of window t + 2 (visible since B(t-1)) -> this wave's head maxima -> RED[(t + 2) & 1];
//!   3. M of window t + 1 from RED[(t + 1) & 1] (published at B(t-1)); P of window t + 1 from the carried S -> PBUF[(t + 1) & 1]; the
//!      running max / sum and the rescale factor of window t + 1 (every wave computes the same M: no alpha exchange);
//!   4. O = alpha(t) O + P(t) V(t) (P(t) from PBUF[t & 1], published at B(t-1));
//!   5. wait for window t + 3's tensor loads; B(t).
//!
//! Windows past the sequence are clamped to its last page and fully masked (they add nothing).

use amdgpu_dsl::{ev, CmpPred, CmpTy, DType, Ext, FnBuilder, Function, MemTy, Op, Opnd, TdmCpol, TensorTile, ValueId, WmmaKind};
use amdgpu_stage::attention::mla::{PageLayout, DV, PAGE_BYTES, PAGE_TOKENS, QUARTER, ROPE_OFF};
use amdgpu_stage::gemm::tdm::tdm_group0;
use amdgpu_tile::shared::{Layout, LdsPlan, SlotIdx};

/// pages per window
const WP: i32 = 2;
/// pages per sequence the LDS page table holds (ctx <= 65536)
const MAX_PAGES: i32 = 1024;

fn f32_of(b: &mut FnBuilder, v: ValueId) -> ValueId {
    b.let_(Op::BitcastF32(Opnd::Val(v)))
}

/// The one-barrier window kernel.
pub fn mla_g1250_qh16_w1() -> Function {
    let name = "mla_g1250_qh16_w1";
    let layout = PageLayout::plan();
    let mut b = FnBuilder::new(name, 128);
    b.compile_profile(amdgpu_dsl::CompileProfile::Gemm);
    let names = ["R", "LSE", "Q", "KV", "LTP", "LTD", "LTL", "QTP", "QS", "KS", "VSC"];
    for (i, n) in names.iter().enumerate() {
        b.arg(n, DType::Ptr, 8 * i as u32, 8);
    }
    for (i, n) in ["out16", "scalar", "qlens", "passes", "stride_page", "log2_page", "stride_q", "flags"].iter().enumerate() {
        b.arg(n, DType::I32, 88 + 4 * i as u32, 4);
    }
    b.arg_tensor("R", Ext::grid_y() * Ext::c(16 * 512 * 2));
    b.arg_tensor("LSE", Ext::grid_y() * Ext::c(16 * 4));
    b.arg_tensor("Q", Ext::grid_y() * Ext::c(16 * 768));
    b.arg_tensor("LTP", (Ext::grid_y() + Ext::c(1)) * Ext::c(4));
    b.arg_tensor("LTL", Ext::grid_y() * Ext::c(4));
    b.arg_tensor("QS", Ext::c(4));
    b.arg_tensor("KS", Ext::c(4));
    b.sym_range("P", 1, 1 << 24);
    b.sym_range("NPG", 1, 1 << 26);
    b.arg_tensor("KV", Ext::sym("P") * Ext::c(PAGE_BYTES as i64));
    b.arg_tensor("LTD", Ext::sym("NPG") * Ext::c(4));
    let mut lds = LdsPlan::new();
    let kv = lds.ring("KV", (WP * layout.slot_bytes) as i64, 4, Layout::Linear);
    let red = lds.ring("RED", 512, 2, Layout::Linear);
    let pbuf = lds.ring("PBUF", 1024 * WP as i64, 2, Layout::Linear);
    let ptab = lds.tile("PTAB", 4 * MAX_PAGES as i64, Layout::Linear);
    let lsb = lds.tile("LSUM", 512, Layout::Linear);
    let out = lds.tile_over("OUT", 16 * 512 * 2, &[&kv]);
    lds.commit(&mut b);
    let kp = |b: &mut FnBuilder, i: u32| b.let_(Op::KernargPtr { offset: 8 * i });
    let (pr, plse, pq, pkv, pltp, pltd, pltl) = (kp(&mut b, 0), kp(&mut b, 1), kp(&mut b, 2), kp(&mut b, 3), kp(&mut b, 4), kp(&mut b, 5), kp(&mut b, 6));
    let (pqs, pks) = (kp(&mut b, 8), kp(&mut b, 9));
    let tid = b.let_(Op::WorkitemId);
    let bidx = b.let_(Op::WorkgroupIdY);
    let lane = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::I32(31)));
    let wv = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::I32(5)));
    let w = b.read_first_lane(wv);
    let m = b.let_(Op::AndI32(Opnd::Val(lane), Opnd::I32(15)));
    let g = b.let_(Op::LshrI32(Opnd::Val(lane), Opnd::I32(4)));
    let ld = |b: &mut FnBuilder, p: ValueId, off: ValueId| {
        let q = b.let_(Op::PtrAdd { base: Opnd::Val(p), byte_off: Opnd::Val(off) });
        b.let_(Op::LoadScalar { base: Opnd::Val(q), byte_off: 0 })
    };
    let zero = b.c_i32(0);
    let qs = ld(&mut b, pqs, zero);
    let ks = ld(&mut b, pks, zero);
    let bo = b.let_(Op::ShlI32(Opnd::Val(bidx), Opnd::I32(2)));
    let kv0 = ld(&mut b, pltp, bo);
    let bo4 = b.add_i32(bo, Opnd::I32(4));
    let kv1 = ld(&mut b, pltp, bo4);
    let last = ld(&mut b, pltl, bo);
    let npages = b.bind_i32(ev(kv1) - ev(kv0));
    b.assume_trusted(npages, Ext::c(1), Ext::c(MAX_PAGES as i64), "host contract: every sequence has 1 .. 1024 pages (ctx <= 65536)");
    b.assume_trusted(kv0, Ext::c(0), Ext::sym("NPG") - Ext::c(1), "host contract: LTP is monotone and LTP[B] <= NPG, the page table's entry count");
    let kv_end = b.bind_i32(ev(kv0) + ev(npages));
    b.assume_trusted(kv_end, Ext::c(1), Ext::sym("NPG"), "host contract: LTP[b + 1] <= NPG (a sequence's entries lie inside the page table)");
    b.assume_trusted(last, Ext::c(1), Ext::c(PAGE_TOKENS as i64), "host contract: the last page holds 1 .. 64 tokens");
    let ctx = b.bind_i32(((ev(npages) - 1) << 6) + ev(last));
    let qsf = b.let_(Op::BitcastF32(Opnd::Val(qs)));
    let ksf = b.let_(Op::BitcastF32(Opnd::Val(ks)));
    let sc = b.let_(Op::KernargI32 { offset: 92 });
    let scf = b.let_(Op::BitcastF32(Opnd::Val(sc)));
    let c1 = b.mul_f32(qsf, ksf);
    let c2 = b.mul_f32(c1, scf);
    let log2e = b.c_f32(std::f32::consts::LOG2_E);
    let cscale = b.mul_f32(c2, log2e);
    // Q fragments: head m, dims 64 c + 32 g .. + 32
    let qrow = b.bind_i32((ev(bidx) * 16 + ev(m)) * 768 + ev(g) * 32);
    let pqr = b.let_(Op::PtrAdd { base: Opnd::Val(pq), byte_off: Opnd::Val(qrow) });
    let q: Vec<ValueId> = (0..9u32)
        .map(|c| {
            let parts: Vec<Opnd> = (0..2u32).map(|j| Opnd::Val(b.load_global(pqr, 64 * c + 16 * j, MemTy::B128))).collect();
            b.frag_cat(&parts)
        })
        .collect();
    // a quarter page per wave per tensor op (144 rows of 64 bytes)
    let tile = TensorTile { data_size: 8, tensor_dim: [Opnd::I32(1152), Opnd::I32(1)], stride0: Opnd::I32(1152), tile: [1152, 1], pad: layout.pad, wg_mask: 0 };
    let (_, g1) = b.tensor_desc_2d(&tile, Opnd::Val(pkv), Opnd::I32(0), Opnd::I32(0));
    // lane offsets inside a page slot: K rows of this wave's 16 tokens, rope rows, the V tr8 fetch of this wave's 128 dims
    let kofs = layout.at_expr(&mut b, (ev(w) * 16 + ev(m)) * DV + ev(g) * 32);
    let rofs = layout.at_expr(&mut b, (ev(w) * 16 + ev(m)) * 64 + ev(g) * 32 + ROPE_OFF);
    let vofs = layout.at_expr(&mut b, (((ev(lane) >> 4) << 3) + (ev(lane) & 3) + (((ev(lane) >> 3) & 1) << 2)) * DV + (((ev(lane) >> 2) & 1) << 3) + ev(w) * 128);
    let kbases: Vec<i32> = (0..128).map(|t| (t / 32 * 16 + (t & 15)) * DV + ((t & 31) >> 4) * 32).collect();
    let kdelta: Vec<i32> = (0..8).map(|c| layout.delta(&kbases, 64 * c)).collect();
    let vbases: Vec<i32> = (0..128).map(|t| { let f = t & 31; (8 * (f >> 4) + (f & 3) + 4 * ((f >> 3) & 1)) * DV + 8 * ((f >> 2) & 1) + (t / 32) * 128 }).collect();
    let vdelta: Vec<i32> = (0..32).map(|i| layout.delta(&vbases, 16 * (i / 8) * DV + 16 * (i % 8))).collect();
    let _ = QUARTER;
    // the sequence's page ids into LDS (indices clamped to the last page)
    let nch = b.bind_i32((ev(npages) + 127) >> 7);
    let k0 = b.c_i32(0);
    let lv = b.begin_loop(&[(DType::I32, Opnd::Val(k0))]);
    let kk = lv[0];
    b.assume(kk, Ext::c(0), Ext::c((MAX_PAGES / 128 - 1) as i64));
    let idx = b.bind_i32(amdgpu_dsl::varx(kk) * 128 + ev(tid));
    let e_raw = b.bind_i32(ev(kv0) + ev(idx));
    let e_last = b.bind_i32(ev(kv_end) - 1);
    let ci = b.let_(Op::MinI32(Opnd::Val(e_raw), Opnd::Val(e_last)));
    let eo = b.bind_i32(ev(ci) * 4);
    let pe = b.let_(Op::PtrAdd { base: Opnd::Val(pltd), byte_off: Opnd::Val(eo) });
    let pv = b.load_global(pe, 0, MemTy::B32);
    let sa = b.bind_i32(ev(idx) * 4 + ptab.base as i32);
    let t00 = SlotIdx::new(Opnd::I32(0), Opnd::I32(0));
    ptab.at(&t00).write(&mut b, |b| b.store_lds(sa, pv, MemTy::B32, false));
    let k1 = b.bind_i32(amdgpu_dsl::varx(kk) + 1);
    b.set_var(kk, Opnd::Val(k1));
    let done = b.cmp(CmpPred::Ge, CmpTy::I32, k1, nch);
    b.break_if(done);
    b.end_loop();
    b.wait_lds(0);
    b.barrier();

    let nwin = b.bind_i32((ev(npages) + (WP - 1)) >> 1);
    let neg = b.c_f32(-1.0e30);
    let zf = b.c_f32(0.0);
    let lastp = b.bind_i32(ev(npages) - 1);
    // ---- the per-window pieces (window index `t` an I32 expression; ring slot t & 3, exchange slots t & 1, generation t) ----
    let stage = |b: &mut FnBuilder, t: ValueId| {
        let si = b.bind_i32(ev(t) & 3);
        let sl = SlotIdx::new(Opnd::Val(si), Opnd::Val(t));
        let r = kv.at(&sl);
        for pg in 0..WP {
            let tp = b.bind_i32(ev(t) * WP + pg);
            let tc = b.let_(Op::MinI32(Opnd::Val(tp), Opnd::Val(lastp)));
            let ta = b.bind_i32((ev(tc) << 2) + ptab.base as i32);
            let pidv = ptab.at(&SlotIdx::new(Opnd::I32(0), Opnd::I32(0))).read(b, |b| b.load_lds(ta, MemTy::B32));
            let pid = b.read_first_lane(pidv);
            b.assume_trusted(pid, Ext::c(0), Ext::sym("P") - Ext::c(1), "host contract: every page-table entry < P, the KV pool's page count (KV holds P x 36864 bytes)");
            let off = b.bind_i32(ev(pid) * PAGE_BYTES + ev(w) * QUARTER);
            let lds_a = b.bind_i32(r.base() + pg * layout.slot_bytes + ev(w) * layout.quarter_lds);
            let g0 = tdm_group0(b, pkv, off, Opnd::Val(lds_a));
            r.write(b, |b| b.tensor_load_to_lds(Opnd::Val(g0), Opnd::Val(g1), None, TdmCpol::default()));
        }
    };
    // S^T of window t: one Frag8 per page (lane (m, g) = head m, tokens 16 w + 8 g + r of the page)
    let s_win = |b: &mut FnBuilder, t: ValueId| -> Vec<ValueId> {
        let si = b.bind_i32(ev(t) & 3);
        let sl = SlotIdx::new(Opnd::Val(si), Opnd::Val(t));
        let r = kv.at(&sl);
        let sb = b.bind_i32(r.base());
        let mut outv = Vec::new();
        for pg in 0..WP {
            let pb = b.bind_i32(ev(sb) + pg * layout.slot_bytes);
            let nope_row = b.bind_i32(ev(pb) + ev(kofs));
            let rope_row = b.bind_i32(ev(pb) + ev(rofs));
            let s = r.read(b, |b| {
                let fa: Vec<ValueId> = (0..9)
                    .map(|c| {
                        let lo = if c < 8 { b.bind_i32(ev(nope_row) + kdelta[c as usize]) } else { b.bind_i32(ev(rope_row)) };
                        let hi = b.bind_i32(ev(lo) + 16);
                        b.load_lds_b128x2(lo, hi)
                    })
                    .collect();
                let mut acc: Option<Opnd> = None;
                for c in 0..9 {
                    let d = b.wmma(WmmaKind::Fp8_16x16x64, fa[c], q[c], acc);
                    acc = Some(Opnd::Val(d));
                }
                match acc {
                    Some(Opnd::Val(v)) => v,
                    _ => unreachable!(),
                }
            });
            outv.push(s);
        }
        outv
    };
    // the masked base-2 logits of window t (16 per lane)
    let logits = |b: &mut FnBuilder, t: ValueId, s: &[Opnd]| -> Vec<ValueId> {
        let mut x = Vec::with_capacity(16);
        for pg in 0..WP {
            let tok0 = b.bind_i32(ev(t) * (64 * WP) + (64 * pg) + ev(w) * 16 + ev(g) * 8);
            for r in 0..8u8 {
                let si = b.frag_slice(s[pg as usize], r, 1);
                let sf = f32_of(b, si);
                let v = b.mul_f32(sf, cscale);
                let tok = b.add_i32(tok0, Opnd::I32(r as i32));
                let ok = b.cmp(CmpPred::Lt, CmpTy::I32, tok, ctx);
                x.push(b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(ok), a: Opnd::Val(v), b: Opnd::Val(neg) }));
            }
        }
        x
    };
    // this wave's head-m maximum over the window -> RED[t & 1]
    let put_max = |b: &mut FnBuilder, t: ValueId, x: &[ValueId]| {
        let mut mx = x[0];
        for r in 1..x.len() {
            mx = b.max_f32(mx, x[r]);
        }
        let si = b.bind_i32(ev(t) & 1);
        let sl = SlotIdx::new(Opnd::Val(si), Opnd::Val(t));
        let r = red.at(&sl);
        let a = b.bind_i32(r.base() + ev(w) * 128 + ev(g) * 64 + ev(m) * 4);
        r.write(b, |b| b.store_lds(a, mx, MemTy::B32, false));
    };
    // the window maximum of head m (8 partials)
    let get_max = |b: &mut FnBuilder, t: ValueId| -> ValueId {
        let si = b.bind_i32(ev(t) & 1);
        let sl = SlotIdx::new(Opnd::Val(si), Opnd::Val(t));
        let r = red.at(&sl);
        let rb = b.bind_i32(r.base() + ev(m) * 4);
        r.read(b, |b| {
            let mut pm: Option<ValueId> = None;
            for k in 0..8 {
                let a = b.bind_i32(ev(rb) + k * 64);
                let v = b.load_lds(a, MemTy::B32);
                let vf = f32_of(b, v);
                pm = Some(match pm {
                    None => vf,
                    Some(p) => b.max_f32(p, vf),
                });
            }
            pm.unwrap()
        })
    };
    // P of window t (exponents, exps, sums batched) -> PBUF[t & 1]; returns (alpha, m_new, l_new)
    let put_p = |b: &mut FnBuilder, t: ValueId, x: &[ValueId], m_old: Opnd, l_old: Opnd, pm: ValueId| -> (ValueId, ValueId, ValueId) {
        let m_new = b.max_f32(m_old, pm);
        let dm = b.sub_f32(m_old, m_new);
        let alpha = b.exp2_f32(dm);
        let mut lsum = b.mul_f32(l_old, alpha);
        let d: Vec<ValueId> = (0..x.len()).map(|r| b.sub_f32(x[r], m_new)).collect();
        let p: Vec<ValueId> = (0..x.len()).map(|r| b.exp2_f32(d[r])).collect();
        for r in 0..x.len() {
            lsum = b.add_f32(lsum, p[r]);
        }
        let si = b.bind_i32(ev(t) & 1);
        let sl = SlotIdx::new(Opnd::Val(si), Opnd::Val(t));
        let r = pbuf.at(&sl);
        let pw = b.bind_i32(r.base() + ev(lane) * 32 + ev(w) * 8);
        r.write(b, |b| {
            for pg in 0..WP as usize {
                let p0 = b.cvt_pk_fp8x4_f32(p[8 * pg], p[8 * pg + 1], p[8 * pg + 2], p[8 * pg + 3]);
                let p1 = b.cvt_pk_fp8x4_f32(p[8 * pg + 4], p[8 * pg + 5], p[8 * pg + 6], p[8 * pg + 7]);
                let a0 = b.bind_i32(ev(pw) + 1024 * pg as i32);
                b.store_lds(a0, p0, MemTy::B32, false);
                let a1 = b.bind_i32(ev(pw) + (1024 * pg as i32 + 4));
                b.store_lds(a1, p1, MemTy::B32, false);
            }
        });
        (alpha, m_new, lsum)
    };

    // ---- prologue: windows 0, 1, 2 staged; S(0), S(1) and their maxima; P(0) ----
    let c0 = b.c_i32(0);
    let c1i = b.c_i32(1);
    let c2i = b.c_i32(2);
    stage(&mut b, c0);
    stage(&mut b, c1i);
    stage(&mut b, c2i);
    b.wait_tensor(0);
    b.barrier();
    let s0 = s_win(&mut b, c0);
    let x0 = logits(&mut b, c0, &s0.iter().map(|&v| Opnd::Val(v)).collect::<Vec<_>>());
    put_max(&mut b, c0, &x0);
    let s1 = s_win(&mut b, c1i);
    let x1 = logits(&mut b, c1i, &s1.iter().map(|&v| Opnd::Val(v)).collect::<Vec<_>>());
    put_max(&mut b, c1i, &x1);
    b.wait_lds(0);
    b.barrier();
    let pm0 = get_max(&mut b, c0);
    let (alpha0, m0, l0) = put_p(&mut b, c0, &x0, Opnd::Val(neg), Opnd::Val(zf), pm0);
    b.wait_lds(0);
    b.barrier();

    // ---- loop state: running max (head m), partial sum (this lane's tokens), alpha of the next PV, O (8 x Frag8), S of window t + 1 ----
    let mut inits = vec![(DType::F32, Opnd::Val(m0)), (DType::F32, Opnd::Val(l0)), (DType::F32, Opnd::Val(alpha0))];
    for _ in 0..8 {
        let z = b.frag_cat(&[Opnd::I32(0); 8]);
        inits.push((WmmaKind::Fp8_16x16x64.acc_ty(), Opnd::Val(z)));
    }
    for s in &s1 {
        inits.push((WmmaKind::Fp8_16x16x64.acc_ty(), Opnd::Val(*s)));
    }
    let st = b.declare_vars(&inits);
    let (vm, vl, va) = (st[0], st[1], st[2]);
    let ov: Vec<_> = st[3..11].to_vec();
    let sv: Vec<_> = st[11..13].to_vec();
    let z = b.c_i32(0);
    let lvars = b.begin_loop(&[(DType::I32, Opnd::Val(z))]);
    let it = lvars[0];
    b.assume(it, Ext::c(0), Ext::c((MAX_PAGES / 2) as i64));
    let t = b.bind_i32(amdgpu_dsl::varx(it) + 0);
    let t1 = b.bind_i32(ev(t) + 1);
    let t2 = b.bind_i32(ev(t) + 2);
    let t3 = b.bind_i32(ev(t) + 3);
    // 1. stage window t + 3
    stage(&mut b, t3);
    // 2. S(t + 2) and its maxima
    let s2 = s_win(&mut b, t2);
    let x2 = logits(&mut b, t2, &s2.iter().map(|&v| Opnd::Val(v)).collect::<Vec<_>>());
    put_max(&mut b, t2, &x2);
    // 3. P(t + 1) from the carried S(t + 1)
    let x1c = logits(&mut b, t1, &sv.iter().map(|&v| Opnd::Var(v)).collect::<Vec<_>>());
    let pm1 = get_max(&mut b, t1);
    let (alpha1, m1, l1) = put_p(&mut b, t1, &x1c, Opnd::Var(vm), Opnd::Var(vl), pm1);
    // 4. O = alpha(t) O + P(t) V(t): alpha of row 8 g + r by v_readlane of head 8 g + r (every wave holds the same alpha)
    let sa16: Vec<ValueId> = (0..16u8).map(|h| b.read_lane(Opnd::Var(va), h)).collect();
    let gbits = b.bind_i32(ev(g) * 0x3F80_0000);
    let gf = b.let_(Op::BitcastF32(Opnd::Val(gbits)));
    let one = b.c_f32(1.0);
    let hf = b.sub_f32(one, gf);
    let al: Vec<ValueId> = (0..8usize)
        .map(|r| {
            let a0 = b.let_(Op::BitcastF32(Opnd::Val(sa16[r])));
            let a1 = b.let_(Op::BitcastF32(Opnd::Val(sa16[8 + r])));
            let u0 = b.mul_f32(a0, hf);
            let u1 = b.mul_f32(a1, gf);
            b.add_f32(u0, u1)
        })
        .collect();
    for nt in 0..8 {
        for r in 0..8u8 {
            b.var_scale_lane(ov[nt], r, al[r as usize]);
        }
    }
    let psi = b.bind_i32(ev(t) & 1);
    let psl = SlotIdx::new(Opnd::Val(psi), Opnd::Val(t));
    let pr_ = pbuf.at(&psl);
    let pbase = b.bind_i32(pr_.base() + ev(lane) * 32);
    let pa: Vec<ValueId> = pr_.read(&mut b, |b| {
        (0..WP)
            .map(|pg| {
                let lo = b.bind_i32(ev(pbase) + 1024 * pg);
                let hi = b.bind_i32(ev(lo) + 16);
                b.load_lds_b128x2(lo, hi)
            })
            .collect()
    });
    let ksi = b.bind_i32(ev(t) & 3);
    let ksl = SlotIdx::new(Opnd::Val(ksi), Opnd::Val(t));
    let kr = kv.at(&ksl);
    let kb = b.bind_i32(kr.base());
    kr.read(&mut b, |b| {
        for pg in 0..WP {
            let vrow = b.bind_i32(ev(kb) + pg * layout.slot_bytes + ev(vofs));
            let fvs: Vec<ValueId> = (0..8)
                .map(|nt| {
                    let a: Vec<ValueId> = (0..4).map(|jj| b.bind_i32(ev(vrow) + vdelta[8 * jj as usize + nt as usize])).collect();
                    b.load_lds_tr_b8x4(a[0], a[1], a[2], a[3])
                })
                .collect();
            for nt in 0..8 {
                let o = ov[nt];
                let d = b.wmma(WmmaKind::Fp8_16x16x64, pa[pg as usize], fvs[nt], Some(Opnd::Var(o)));
                b.set_var(o, Opnd::Val(d));
            }
        }
    });
    b.set_var(vm, Opnd::Val(m1));
    b.set_var(vl, Opnd::Val(l1));
    b.set_var(va, Opnd::Val(alpha1));
    for pg in 0..WP as usize {
        b.set_var(sv[pg], Opnd::Val(s2[pg]));
    }
    // 5. window t + 3 complete; B(t)
    b.wait_tensor(0);
    b.wait_lds(0);
    b.barrier();
    let tn = b.bind_i32(amdgpu_dsl::varx(it) + 1);
    b.set_var(it, Opnd::Val(tn));
    let fin = b.cmp(CmpPred::Ge, CmpTy::I32, tn, nwin);
    b.break_if(fin);
    b.end_loop();
    b.wait_tensor(0);

    // ---- epilogue: the row sums (8 partials per head) through LDS, O / L * ks to bf16 through LDS, LSE ----
    let rw = b.bind_i32(ev(w) * 128 + ev(g) * 64 + ev(m) * 4 + lsb.base as i32);
    lsb.at(&t00).write(&mut b, |b| b.store_lds(rw, Opnd::Var(vl), MemTy::B32, false));
    b.wait_lds(0);
    b.barrier();
    let sum_head = |b: &mut FnBuilder, head: amdgpu_dsl::I32| -> ValueId {
        let hb = b.bind_i32(head * 4 + lsb.base as i32);
        lsb.at(&SlotIdx::new(Opnd::I32(0), Opnd::I32(0))).read(b, |b| {
            let mut acc: Option<ValueId> = None;
            for k in 0..8 {
                let a = b.bind_i32(ev(hb) + k * 64);
                let v = b.load_lds(a, MemTy::B32);
                let vf = f32_of(b, v);
                acc = Some(match acc {
                    None => vf,
                    Some(s) => b.add_f32(s, vf),
                });
            }
            acc.unwrap()
        })
    };
    let inv: Vec<ValueId> = (0..8)
        .map(|r| {
            let l = sum_head(&mut b, ev(g) * 8 + r);
            let rl = b.rcp_f32(l);
            b.mul_f32(rl, ksf)
        })
        .collect();
    let ob = out.base as i32;
    out.at(&t00).write(&mut b, |b| {
        for nt in 0..8u32 {
            let o = ov[nt as usize];
            for r in 0..8u8 {
                let e = b.frag_slice(Opnd::Var(o), r, 1);
                let ef = f32_of(b, e);
                let y = b.mul_f32(ef, inv[r as usize]);
                let h = b.cvt_pk_bf16_f32(y, y);
                let a = b.bind_i32(((ev(g) * 8 + r as i32) * 512 + ev(w) * 128 + (16 * nt as i32) + ev(m)) * 2 + ob);
                b.store_lds(a, h, MemTy::U16, false);
            }
        }
    });
    b.wait_lds(0);
    b.barrier();
    let rb = b.bind_i32(ev(bidx) * (16 * 512 * 2) + ev(tid) * 128);
    let prb = b.let_(Op::PtrAdd { base: Opnd::Val(pr), byte_off: Opnd::Val(rb) });
    out.at(&t00).read(&mut b, |b| {
        let rows: Vec<ValueId> = (0..8).map(|k| {
            let a = b.bind_i32(ev(tid) * 128 + (16 * k + ob));
            b.load_lds_b128(a)
        }).collect();
        for k in 0..8u32 {
            b.store_global(prb, 16 * k, rows[k as usize], MemTy::B128, false);
        }
    });
    let l = sum_head(&mut b, ev(m));
    let lg = b.log2_f32(l);
    let s2l = b.add_f32(Opnd::Var(vm), lg);
    let ln2 = b.c_f32(std::f32::consts::LN_2);
    let lse = b.mul_f32(s2l, ln2);
    let off = b.bind_i32((ev(bidx) * 16 + ev(m)) * 4);
    let pl = b.let_(Op::PtrAdd { base: Opnd::Val(plse), byte_off: Opnd::Val(off) });
    b.store_global(pl, 0, lse, MemTy::B32, false);
    b.finish()
}
