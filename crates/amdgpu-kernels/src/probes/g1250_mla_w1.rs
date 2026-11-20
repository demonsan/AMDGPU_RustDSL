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

/// pages per sequence the LDS page table holds (ctx <= 65536)
const MAX_PAGES: i32 = 1024;

fn f32_of(b: &mut FnBuilder, v: ValueId) -> ValueId {
    b.let_(Op::BitcastF32(Opnd::Val(v)))
}

/// The one-barrier window kernel: 2-page windows, a 4-slot ring (one window of load lead).
pub fn mla_g1250_qh16_w1() -> Function {
    mla_w1_with("mla_g1250_qh16_w1", 2, 4)
}

/// 1-page windows and an 8-slot ring: four windows of load lead (the tensor-load latency exceeds one 2-page window).
pub fn mla_g1250_qh16_w1d() -> Function {
    mla_w1_with("mla_g1250_qh16_w1d", 1, 8)
}

/// `_w1` with the iteration body ordered for instruction-level parallelism: all LDS fragments first (K of window t + 2, P and V of
/// window t), then S(t + 2)'s and PV(t)'s WMMAs interleaved chunk by chunk with P(t + 1)'s VALU / TRANS work between them.
pub fn mla_g1250_qh16_w1i() -> Function {
    ILP.with(|c| c.set(true));
    let f = mla_w1_with("mla_g1250_qh16_w1i", 2, 4);
    ILP.with(|c| c.set(false));
    f
}

thread_local! {
    static ILP: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static STAMPS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static ABL: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
    static LEAN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// `_w1i` with the lean softmax VALU: masked raw S carried (masked once), the per-lane token limit compared against constants, the max on
/// raw S (scale > 0), and P's exponent as one `v_pk_fma_f32` per two values (S * scale - max)
pub fn mla_g1250_qh16_w1v() -> Function {
    ILP.with(|c| c.set(true));
    LEAN.with(|c| c.set(true));
    let f = mla_w1_with("mla_g1250_qh16_w1v", 2, 4);
    LEAN.with(|c| c.set(false));
    ILP.with(|c| c.set(false));
    f
}

/// deletion ablations of `_w1i` (timing only; the output is wrong): 1 = no K reads (Q as the A operand), 2 = no V reads (P as the B
/// operand), 3 = no S / PV WMMAs (the fragments are read and folded into the carried values by a cheap op)
pub fn mla_g1250_qh16_w1x(mode: u8) -> Function {
    ILP.with(|c| c.set(true));
    LEAN.with(|c| c.set(mode >= 6));
    ABL.with(|c| c.set(mode));
    let f = mla_w1_with(&format!("mla_g1250_qh16_w1x{mode}"), 2, 4);
    ABL.with(|c| c.set(0));
    LEAN.with(|c| c.set(false));
    ILP.with(|c| c.set(false));
    f
}

/// timing aid (not registered for tests): `_w1i` storing shader-clock stamps of every wave over R[b] (word 64 w + checkpoint)
pub fn mla_g1250_qh16_w1s() -> Function {
    ILP.with(|c| c.set(true));
    STAMPS.with(|c| c.set(true));
    let f = mla_w1_with("mla_g1250_qh16_w1s", 2, 4);
    STAMPS.with(|c| c.set(false));
    ILP.with(|c| c.set(false));
    f
}

/// `wp` pages per window, a `depth`-slot ring (a power of two >= 4): window t + depth - 1 is staged in iteration t, window t + 3 must be
/// complete at its end (depth - 4 windows stay in flight).
fn mla_w1_with(name: &str, wp: i32, depth: i32) -> Function {
    #[allow(non_snake_case)]
    let WP = wp;
    assert!(depth >= 4 && (depth & (depth - 1)) == 0);
    let in_flight = ((depth - 4) * wp) as u8;
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
    let kv = lds.ring("KV", (WP * layout.slot_bytes) as i64, depth as u32, Layout::Linear);
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

    let nwin = b.bind_i32((ev(npages) + (WP - 1)) >> (WP.trailing_zeros() as i32));
    let neg = b.c_f32(-1.0e30);
    let zf = b.c_f32(0.0);
    let lastp = b.bind_i32(ev(npages) - 1);
    // ---- the per-window pieces (window index `t` an I32 expression; ring slot t & 3, exchange slots t & 1, generation t) ----
    let stage = |b: &mut FnBuilder, t: ValueId| {
        let si = b.bind_i32(ev(t) & (depth - 1));
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
        let si = b.bind_i32(ev(t) & (depth - 1));
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
        let mut x = Vec::with_capacity(8 * WP as usize);
        for pg in 0..WP {
            let tok0 = b.bind_i32(ev(t) * (64 * WP) + (64 * pg) + ev(w) * 16 + ev(g) * 8);
            for r in 0..8u8 {
                let si = b.frag_slice(s[pg as usize], r, 1);
                let sf = f32_of(b, si);
                let v = b.mul_f32(sf, cscale);
                if ABL.with(|c| c.get()) == 4 {
                    x.push(v);
                    continue;
                }
                let tok = b.add_i32(tok0, Opnd::I32(r as i32));
                let ok = b.cmp(CmpPred::Lt, CmpTy::I32, tok, ctx);
                x.push(b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(ok), a: Opnd::Val(v), b: Opnd::Val(neg) }));
            }
        }
        x
    };
    // this wave's head-m maximum over the window -> RED[t & 1]
    let put_max = |b: &mut FnBuilder, t: ValueId, x: &[Opnd]| {
        let mut mx = b.max_f32(x[0], x[1]);
        for r in 2..x.len() {
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

    let lean = LEAN.with(|c| c.get());
    // lean pieces: the masked raw S of window t (lane limit per page vs constant token offsets)
    let raw_masked = |b: &mut FnBuilder, t: ValueId, s: &[Opnd]| -> Vec<ValueId> {
        let mut x = Vec::with_capacity(8 * WP as usize);
        for pg in 0..WP {
            let lim = b.bind_i32(ev(ctx) - (ev(t) * (64 * WP) + (64 * pg) + ev(w) * 16 + ev(g) * 8));
            for r in 0..8u8 {
                let si = b.frag_slice(s[pg as usize], r, 1);
                let sf = f32_of(b, si);
                let ok = b.cmp(CmpPred::Gt, CmpTy::I32, lim, Opnd::I32(r as i32));
                x.push(b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(ok), a: Opnd::Val(sf), b: Opnd::Val(neg) }));
            }
        }
        x
    };
    // P of window t from masked raw values: m_new = max(m_old, pm_raw * scale); exponent pairs by v_pk_fma_f32 (x * scale - m_new)
    let put_p_lean = |b: &mut FnBuilder, t: ValueId, x: &[Opnd], m_old: Opnd, l_old: Opnd, pm_raw: ValueId| -> (ValueId, ValueId, ValueId) {
        let pms = b.mul_f32(pm_raw, cscale);
        let m_new = b.max_f32(m_old, pms);
        let dm = b.sub_f32(m_old, m_new);
        let alpha = b.exp2_f32(dm);
        let sc2 = b.frag2_dup(cscale);
        let nm = b.neg_f32(m_new);
        let nm2 = b.frag2_dup(nm);
        let mut p = Vec::with_capacity(x.len());
        let dp: Vec<ValueId> = (0..x.len() / 2)
            .map(|k| {
                let pair = b.frag_cat(&[x[2 * k], x[2 * k + 1]]);
                b.pk_fma_f32(pair, sc2, nm2)
            })
            .collect();
        for k in 0..dp.len() {
            for h in 0..2u8 {
                let dv = b.frag_slice(Opnd::Val(dp[k]), h, 1);
                let df = f32_of(b, dv);
                p.push(b.exp2_f32(df));
            }
        }
        // the sum: pairwise packed adds, then the two lanes
        let mut pairs: Vec<ValueId> = (0..p.len() / 2).map(|k| b.frag_cat(&[Opnd::Val(p[2 * k]), Opnd::Val(p[2 * k + 1])])).collect();
        while pairs.len() > 1 {
            let mut nx = Vec::new();
            for k in 0..pairs.len() / 2 {
                nx.push(b.pk_add_f32(pairs[2 * k], pairs[2 * k + 1]));
            }
            pairs = nx;
        }
        let s0 = b.frag_slice(Opnd::Val(pairs[0]), 0, 1);
        let s1 = b.frag_slice(Opnd::Val(pairs[0]), 1, 1);
        let s0f = f32_of(b, s0);
        let s1f = f32_of(b, s1);
        let ps = b.add_f32(s0f, s1f);
        let la = b.mul_f32(l_old, alpha);
        let lsum = b.add_f32(la, ps);
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
    let _ = c2i;
    for k in 0..(depth - 1) {
        let ck = b.c_i32(k);
        stage(&mut b, ck);
    }
    b.wait_tensor(in_flight);
    b.barrier();
    let s0 = s_win(&mut b, c0);
    let x0 = if lean { raw_masked(&mut b, c0, &s0.iter().map(|&v| Opnd::Val(v)).collect::<Vec<_>>()) } else { logits(&mut b, c0, &s0.iter().map(|&v| Opnd::Val(v)).collect::<Vec<_>>()) };
    put_max(&mut b, c0, &x0.iter().map(|&v| Opnd::Val(v)).collect::<Vec<_>>());
    let s1 = s_win(&mut b, c1i);
    let x1 = if lean { raw_masked(&mut b, c1i, &s1.iter().map(|&v| Opnd::Val(v)).collect::<Vec<_>>()) } else { logits(&mut b, c1i, &s1.iter().map(|&v| Opnd::Val(v)).collect::<Vec<_>>()) };
    put_max(&mut b, c1i, &x1.iter().map(|&v| Opnd::Val(v)).collect::<Vec<_>>());
    b.wait_lds(0);
    b.barrier();
    let pm0 = get_max(&mut b, c0);
    let (alpha0, m0, l0) = if lean {
        put_p_lean(&mut b, c0, &x0.iter().map(|&v| Opnd::Val(v)).collect::<Vec<_>>(), Opnd::Val(neg), Opnd::Val(zf), pm0)
    } else {
        put_p(&mut b, c0, &x0, Opnd::Val(neg), Opnd::Val(zf), pm0)
    };
    b.wait_lds(0);
    b.barrier();

    // ---- loop state: running max (head m), partial sum (this lane's tokens), alpha of the next PV, O (8 x Frag8), S of window t + 1 ----
    let mut inits = vec![(DType::F32, Opnd::Val(m0)), (DType::F32, Opnd::Val(l0)), (DType::F32, Opnd::Val(alpha0))];
    for _ in 0..8 {
        let z = b.frag_cat(&[Opnd::I32(0); 8]);
        inits.push((WmmaKind::Fp8_16x16x64.acc_ty(), Opnd::Val(z)));
    }
    if lean {
        for x in &x1 {
            inits.push((DType::F32, Opnd::Val(*x)));
        }
    } else {
        for s in &s1 {
            inits.push((WmmaKind::Fp8_16x16x64.acc_ty(), Opnd::Val(*s)));
        }
    }
    let st = b.declare_vars(&inits);
    let (vm, vl, va) = (st[0], st[1], st[2]);
    let ov: Vec<_> = st[3..11].to_vec();
    let sv: Vec<_> = if lean { Vec::new() } else { st[11..11 + WP as usize].to_vec() };
    let xv: Vec<_> = if lean { st[11..11 + 8 * WP as usize].to_vec() } else { Vec::new() };
    let z = b.c_i32(0);
    let lvars = b.begin_loop(&[(DType::I32, Opnd::Val(z))]);
    let it = lvars[0];
    b.assume(it, Ext::c(0), Ext::c((MAX_PAGES / 2) as i64));
    let t = b.bind_i32(amdgpu_dsl::varx(it) + 0);
    let t1 = b.bind_i32(ev(t) + 1);
    let t2 = b.bind_i32(ev(t) + 2);
    let t3 = b.bind_i32(ev(t) + (depth - 1));
    let ilp = ILP.with(|c| c.get());
    let stamps = STAMPS.with(|c| c.get());
    let stamp_n = std::cell::Cell::new(0u32);
    let stamp = |b: &mut FnBuilder| {
        if stamps {
            let c = b.shader_cycles();
            let lo = b.let_(Op::PtrLo(Opnd::Val(c)));
            let i = stamp_n.get();
            stamp_n.set(i + 1);
            let o = b.bind_i32(ev(bidx) * (16 * 512 * 2) + (ev(w) * 64 + i as i32) * 4);
            let pa = b.let_(Op::PtrAdd { base: Opnd::Val(pr), byte_off: Opnd::Val(o) });
            b.store_global(pa, 0, lo, MemTy::B32, false);
        }
    };
    if ilp {
        stamp(&mut b);
        // (lean) M(t + 1) and P(t + 1) first: the max partials' LDS reads must not queue behind the fragment batch (in-order DScnt: a wait
        // for them would wait for every fragment)
        let p1_early = if lean {
            let pm1 = get_max(&mut b, t1);
            Some(put_p_lean(&mut b, t1, &xv.iter().map(|&v| Opnd::Var(v)).collect::<Vec<_>>(), Opnd::Var(vm), Opnd::Var(vl), pm1))
        } else {
            None
        };
        // 1. stage window t + depth - 1
        stage(&mut b, t3);
        stamp(&mut b);
        // 2. the LDS fragments: K of window t + 2 (both pages, 9 chunks each), P(t), V(t) (8 tiles per page)
        let si2 = b.bind_i32(ev(t2) & (depth - 1));
        let sl2 = SlotIdx::new(Opnd::Val(si2), Opnd::Val(t2));
        let r2 = kv.at(&sl2);
        let sb2 = b.bind_i32(r2.base());
        let abl = ABL.with(|c| c.get());
        let kfr: Vec<Vec<ValueId>> = if abl == 1 || abl == 6 { (0..WP).map(|_| (0..9).map(|c| q[(c + 1) % 9]).collect()).collect() } else { (0..WP)
            .map(|pg| {
                let pb = b.bind_i32(ev(sb2) + pg * layout.slot_bytes);
                let nope_row = b.bind_i32(ev(pb) + ev(kofs));
                let rope_row = b.bind_i32(ev(pb) + ev(rofs));
                r2.read(&mut b, |b| {
                    (0..9)
                        .map(|c| {
                            let lo = if c < 8 { b.bind_i32(ev(nope_row) + kdelta[c as usize]) } else { b.bind_i32(ev(rope_row)) };
                            let hi = b.bind_i32(ev(lo) + 16);
                            b.load_lds_b128x2(lo, hi)
                        })
                        .collect()
                })
            })
            .collect() };
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
        let ksi = b.bind_i32(ev(t) & (depth - 1));
        let ksl = SlotIdx::new(Opnd::Val(ksi), Opnd::Val(t));
        let kr = kv.at(&ksl);
        let kb = b.bind_i32(kr.base());
        let vfr: Vec<Vec<ValueId>> = if abl == 2 || abl == 6 { (0..WP).map(|_| (0..8).map(|nt| q[nt]).collect()).collect() } else { (0..WP)
            .map(|pg| {
                let vrow = b.bind_i32(ev(kb) + pg * layout.slot_bytes + ev(vofs));
                kr.read(&mut b, |b| {
                    (0..8)
                        .map(|nt| {
                            let a: Vec<ValueId> = (0..4).map(|jj| b.bind_i32(ev(vrow) + vdelta[8 * jj as usize + nt as usize])).collect();
                            b.load_lds_tr_b8x4(a[0], a[1], a[2], a[3])
                        })
                        .collect()
                })
            })
            .collect() };
        stamp(&mut b);
        // 3. O *= alpha(t) (rows 8 g + r: v_readlane of head 8 g + r, select-free choice of the lane half); lean: only when some head's
        // running max grew (a uniform branch)
        if lean {
            let grew = b.cmp(CmpPred::Lt, CmpTy::F32, Opnd::Var(va), Opnd::F32(1.0));
            let anyg = b.let_(Op::AnyLane(Opnd::Val(grew)));
            b.begin_if(anyg);
        }
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
        if lean {
            // O rows (r, r + 1) by one in-place v_pk_mul_f32 with the alpha pair
            let ap: Vec<ValueId> = (0..4).map(|k| b.frag_cat(&[Opnd::Val(al[2 * k]), Opnd::Val(al[2 * k + 1])])).collect();
            for nt in 0..8 {
                for k in 0..4u8 {
                    b.var_pk_mul(ov[nt], 2 * k, ap[k as usize]);
                }
            }
            b.end_if();
        } else if abl != 4 {
            for nt in 0..8 {
                for r in 0..8u8 {
                    b.var_scale_lane(ov[nt], r, al[r as usize]);
                }
            }
        }
        // 4. S(t + 2) and PV(t) WMMAs interleaved (independent chains); P(t + 1)'s logits / max read / exps between the first groups
        let mut sacc: Vec<Option<Opnd>> = vec![None; WP as usize];
        let mut p1 = p1_early;
        let mut x1c_keep = Vec::new();
        for c in 0..9usize {
            for pg in 0..WP as usize {
                if abl == 3 {
                    // keep the read alive: fold the fragment into the carried S (no WMMA)
                    let prev = match sacc[pg] { Some(x) => x, None => Opnd::Var(sv[pg]) };
                    let e = b.frag_slice(Opnd::Val(kfr[pg][c]), 0, 1);
                    let pv0 = b.frag_slice(prev, 0, 1);
                    let s0 = b.add_i32(pv0, Opnd::Val(e));
                    let mut parts = vec![Opnd::Val(s0)];
                    for k in 1..8u8 {
                        parts.push(Opnd::Val(b.frag_slice(prev, k, 1)));
                    }
                    sacc[pg] = Some(Opnd::Val(b.frag_cat(&parts)));
                } else {
                    let d = b.wmma(WmmaKind::Fp8_16x16x64, kfr[pg][c], q[c], sacc[pg]);
                    sacc[pg] = Some(Opnd::Val(d));
                }
            }
            if c < 8 {
                for pg in 0..WP as usize {
                    let o = ov[c];
                    if abl == 3 {
                        let e = b.frag_slice(Opnd::Val(vfr[pg][c]), 0, 1);
                        let e2 = b.frag_slice(Opnd::Val(pa[pg]), 0, 1);
                        let o0 = b.frag_slice(Opnd::Var(o), 0, 1);
                        let s0 = b.add_i32(o0, Opnd::Val(e));
                        let s1 = b.add_i32(s0, Opnd::Val(e2));
                        let mut parts = vec![Opnd::Val(s1)];
                        for k in 1..8u8 {
                            parts.push(Opnd::Val(b.frag_slice(Opnd::Var(o), k, 1)));
                        }
                        let d = b.frag_cat(&parts);
                        b.set_var(o, Opnd::Val(d));
                    } else {
                        let d = b.wmma(WmmaKind::Fp8_16x16x64, pa[pg], vfr[pg][c], Some(Opnd::Var(o)));
                        b.set_var(o, Opnd::Val(d));
                    }
                }
            }
            if c == 1 && !lean {
                x1c_keep = logits(&mut b, t1, &sv.iter().map(|&v| Opnd::Var(v)).collect::<Vec<_>>());
            }
            if c == 3 && !lean {
                let pm1 = get_max(&mut b, t1);
                p1 = Some(if lean {
                    put_p_lean(&mut b, t1, &xv.iter().map(|&v| Opnd::Var(v)).collect::<Vec<_>>(), Opnd::Var(vm), Opnd::Var(vl), pm1)
                } else {
                    put_p(&mut b, t1, &x1c_keep, Opnd::Var(vm), Opnd::Var(vl), pm1)
                });
            }
        }
        stamp(&mut b);
        let s2: Vec<ValueId> = sacc.into_iter().map(|a| match a { Some(Opnd::Val(v)) => v, _ => unreachable!() }).collect();
        let x2: Vec<Opnd> = if lean {
            // raw S; the mask only in a window that reaches past ctx (a uniform branch over VALU)
            let raw: Vec<(DType, Opnd)> = (0..WP as usize)
                .flat_map(|pg| (0..8u8).map(move |r| (pg, r)))
                .map(|(pg, r)| {
                    let si = b.frag_slice(Opnd::Val(s2[pg]), r, 1);
                    (DType::F32, Opnd::Val(f32_of(&mut b, si)))
                })
                .collect();
            let xm = b.declare_vars(&raw);
            let wend = b.bind_i32((ev(t2) + 1) * (64 * WP));
            let part = b.cmp(CmpPred::Gt, CmpTy::I32, wend, ctx);
            b.begin_if(part);
            let mk = raw_masked(&mut b, t2, &s2.iter().map(|&v| Opnd::Val(v)).collect::<Vec<_>>());
            for k in 0..xm.len() {
                b.set_var(xm[k], Opnd::Val(mk[k]));
            }
            b.end_if();
            xm.iter().map(|&v| Opnd::Var(v)).collect::<Vec<Opnd>>()
        } else { logits(&mut b, t2, &s2.iter().map(|&v| Opnd::Val(v)).collect::<Vec<_>>()).into_iter().map(Opnd::Val).collect() };
        put_max(&mut b, t2, &x2);
        stamp(&mut b);
        let (alpha1, m1, l1) = p1.unwrap();
        b.set_var(vm, Opnd::Val(m1));
        b.set_var(vl, Opnd::Val(l1));
        b.set_var(va, Opnd::Val(alpha1));
        if lean {
            for k in 0..xv.len() {
                b.set_var(xv[k], x2[k]);
            }
        } else {
            for pg in 0..WP as usize {
                b.set_var(sv[pg], Opnd::Val(s2[pg]));
            }
        }
    } else {
    // 1. stage window t + depth - 1 (into the slot of window t - 1)
        stage(&mut b, t3);
        // 2. S(t + 2) and its maxima
        let s2 = s_win(&mut b, t2);
        let x2 = logits(&mut b, t2, &s2.iter().map(|&v| Opnd::Val(v)).collect::<Vec<_>>());
        put_max(&mut b, t2, &x2.iter().map(|&v| Opnd::Val(v)).collect::<Vec<_>>());
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
        let ksi = b.bind_i32(ev(t) & (depth - 1));
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
    }
    // 5. window t + 3 complete (depth - 4 windows stay in flight); B(t)
    b.wait_tensor(in_flight);
    stamp(&mut b);
    b.wait_lds(0);
    b.barrier();
    stamp(&mut b);
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
            if !STAMPS.with(|c| c.get()) {
                b.store_global(prb, 16 * k, rows[k as usize], MemTy::B128, false);
            }
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


/// WMMA issue probe (gfx1250): every wave runs 64 iterations of 34 `v_wmma_f32_16x16x64_fp8_fp8` and stores its shader-clock cycle count
/// (out[wave]) and a sink of its accumulators (out[64 + wave]). `k` accumulators used round-robin (dependency distance k); k = 0 is the
/// one-barrier MLA window's pattern (two 9-deep S chains interleaved with 8 PV accumulators of 2 each).
pub fn probe_g1250_wmma_lat(k: u32) -> Function {
    let mut b = FnBuilder::new(&format!("probe_g1250_wmma_lat{k}"), 256);
    b.arg("out", DType::Ptr, 0, 8);
    b.arg_tensor("out", Ext::c(128 * 4));
    let po = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);
    let wv = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::I32(5)));
    let w = b.read_first_lane(wv);
    let lane = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::I32(31)));
    // operands: distinct per-lane fp8 bytes (small values), held in VGPRs
    let mk = |b: &mut FnBuilder, s: i32| -> ValueId {
        let parts: Vec<Opnd> = (0..8).map(|j| Opnd::Val(b.bind_i32((ev(lane) & 7) * 0x0101_0101 + (0x3030_3030 + s * 8 + j)))).collect();
        b.frag_cat(&parts)
    };
    let a: Vec<ValueId> = (0..4).map(|s| mk(&mut b, s)).collect();
    let bb: Vec<ValueId> = (0..4).map(|s| mk(&mut b, 4 + s)).collect();
    let nacc: usize = if k == 0 { 10 } else { k as usize };
    let mut inits = Vec::new();
    for _ in 0..nacc {
        let z = b.frag_cat(&[Opnd::I32(0); 8]);
        inits.push((WmmaKind::Fp8_16x16x64.acc_ty(), Opnd::Val(z)));
    }
    let accs = b.declare_vars(&inits);
    let c0 = b.shader_cycles();
    let c0l = b.let_(Op::PtrLo(Opnd::Val(c0)));
    let z = b.c_i32(0);
    let lv = b.begin_loop(&[(DType::I32, Opnd::Val(z))]);
    let it = lv[0];
    b.assume(it, Ext::c(0), Ext::c(64));
    if k == 0 {
        // S chains: accs[0] (page 0), accs[1] (page 1), each 9 deep, started from zero each iteration; PV: accs[2..10], 2 per accumulator
        let mut s0: Option<Opnd> = None;
        let mut s1: Option<Opnd> = None;
        for c in 0..9usize {
            let d0 = b.wmma(WmmaKind::Fp8_16x16x64, a[c % 4], bb[c % 4], s0);
            s0 = Some(Opnd::Val(d0));
            let d1 = b.wmma(WmmaKind::Fp8_16x16x64, a[(c + 1) % 4], bb[c % 4], s1);
            s1 = Some(Opnd::Val(d1));
            if c < 8 {
                for pg in 0..2usize {
                    let o = accs[2 + c];
                    let d = b.wmma(WmmaKind::Fp8_16x16x64, a[(pg + 2) % 4], bb[(c + pg) % 4], Some(Opnd::Var(o)));
                    b.set_var(o, Opnd::Val(d));
                }
            }
        }
        if let (Some(Opnd::Val(x)), Some(Opnd::Val(y))) = (s0, s1) {
            b.set_var(accs[0], Opnd::Val(x));
            b.set_var(accs[1], Opnd::Val(y));
        }
    } else {
        for i in 0..34usize {
            let o = accs[i % nacc];
            let d = b.wmma(WmmaKind::Fp8_16x16x64, a[i % 4], bb[(i / 4) % 4], Some(Opnd::Var(o)));
            b.set_var(o, Opnd::Val(d));
        }
    }
    let n = b.bind_i32(amdgpu_dsl::varx(it) + 1);
    b.set_var(it, Opnd::Val(n));
    let fin = b.cmp(CmpPred::Ge, CmpTy::I32, n, Opnd::I32(64));
    b.break_if(fin);
    b.end_loop();
    let c1 = b.shader_cycles();
    let c1l = b.let_(Op::PtrLo(Opnd::Val(c1)));
    let dt = b.bind_i32(ev(c1l) - ev(c0l));
    let mut sink: Option<ValueId> = None;
    for &v in &accs {
        let e = b.frag_slice(Opnd::Var(v), 0, 1);
        sink = Some(match sink {
            None => e,
            Some(s) => b.add_i32(s, Opnd::Val(e)),
        });
    }
    let o1 = b.bind_i32(ev(w) * 4);
    let p1 = b.let_(Op::PtrAdd { base: Opnd::Val(po), byte_off: Opnd::Val(o1) });
    b.store_global(p1, 0, dt, MemTy::B32, false);
    b.store_global(p1, 256, sink.unwrap(), MemTy::B32, false);
    b.finish()
}
