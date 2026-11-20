//! MLA decode (gfx1250): per-phase microbenchmarks of the one-barrier window kernel (`g1250_mla_w1.rs`, kernel-opt's `_w1i`), measured in isolation without counters.
//!
//! The same ABI, extents, host contract, LDS layout (padded page ring, RED / PBUF exchange rings) and fragment address expressions as `_w1`; window 0 is staged once into ring slot 0 and a P stub
//! is written into PBUF slot 0, then the loop runs ONLY the blocks selected by the mask, `nwin` times (the number of 2-page windows of the sequence), on that data:
//!   bit 0 S block: the K fragments (2 pages x 9 chunks of ds_load_b128 x 2) and the 18 S WMMAs, carried in the S variables;
//!   bit 1 PV block: the P fragments, the V fragments (ds_load_tr_b8 x 4 per tile, 16 tiles) and the 16 PV WMMAs into the O accumulators;
//!   bit 2 alpha: the O rescale (v_readlane of the alpha, 64 per-lane scalings), only with the PV block;
//!   bit 3 softmax block: logits (mask, scale), the per-wave head maxima to RED, the exchange barrier, the window maximum, the exponents / sums, P to PBUF2;
//!   bit 4 a work-group barrier.
//! The result (garbage numbers, a digest of every accumulator so that nothing is dead code) goes to R: per wave, word 0 = the loop's shader cycles, word 1 = the window count. Names:
//! `mla_g1250_qh16_ph_<mask in hex>`; host `g1250_mla_host` with `DUMP=<file>` (the first 256 bytes of each wave's region of R[0]).

use amdgpu_dsl::{ev, CmpPred, CmpTy, DType, Ext, FnBuilder, Function, MemTy, Op, Opnd, TdmCpol, TensorTile, ValueId, WmmaKind};
use amdgpu_stage::attention::mla::{PageLayout, DV, PAGE_BYTES, PAGE_TOKENS, QUARTER, ROPE_OFF};
use amdgpu_stage::gemm::tdm::tdm_group0;
use amdgpu_tile::shared::{Layout, LdsPlan, SlotIdx};

/// pages per sequence the LDS page table holds (ctx <= 65536)
const MAX_PAGES: i32 = 1024;

fn f32_of(b: &mut FnBuilder, v: ValueId) -> ValueId {
    b.let_(Op::BitcastF32(Opnd::Val(v)))
}


const PH_S: u32 = 1;
const PH_PV: u32 = 2;
const PH_ALPHA: u32 = 4;
const PH_SM: u32 = 8;
const PH_BAR: u32 = 16;

/// `mla_g1250_qh16_ph_<hex mask>`.
pub fn by_mask(hex: &str) -> Option<Function> {
    let mask = u32::from_str_radix(hex, 16).ok()?;
    if mask > 0x1f {
        return None;
    }
    Some(mla_ph_with(&format!("mla_g1250_qh16_ph_{hex}"), 2, 4, mask))
}

fn mla_ph_with(name: &str, wp: i32, depth: i32, mask: u32) -> Function {
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
    let pbuf2 = lds.ring("PBUF2", 1024 * WP as i64, 2, Layout::Linear);
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
        let r = pbuf2.at(&sl);
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

    // ---- the microbenchmark: window 0 staged ONCE into slot 0, a P stub in PBUF slot 0; the loop then runs only the selected blocks on that data, `nwin` times ----
    let c0 = b.c_i32(0);
    stage(&mut b, c0);
    b.wait_tensor(0);
    {
        let sl = SlotIdx::new(Opnd::I32(0), Opnd::I32(0));
        let r = pbuf.at(&sl);
        let pw = b.bind_i32(r.base() + ev(lane) * 32 + ev(w) * 8);
        let stub = b.c_i32(0x3838_3838);
        r.write(&mut b, |b| {
            for pg in 0..WP {
                let a0 = b.bind_i32(ev(pw) + 1024 * pg);
                b.store_lds(a0, stub, MemTy::B32, false);
                let a1 = b.bind_i32(ev(pw) + (1024 * pg + 4));
                b.store_lds(a1, stub, MemTy::B32, false);
            }
        });
    }
    b.wait_lds(0);
    b.barrier();
    let mut inits = vec![(DType::F32, Opnd::Val(neg)), (DType::F32, Opnd::Val(zf)), (DType::F32, Opnd::Val(zf))];
    for _ in 0..8 {
        let z = b.frag_cat(&[Opnd::I32(0); 8]);
        inits.push((WmmaKind::Fp8_16x16x64.acc_ty(), Opnd::Val(z)));
    }
    for _ in 0..WP {
        let z = b.frag_cat(&[Opnd::I32(0); 8]);
        inits.push((WmmaKind::Fp8_16x16x64.acc_ty(), Opnd::Val(z)));
    }
    let st = b.declare_vars(&inits);
    let (vm, vl, va) = (st[0], st[1], st[2]);
    let ov: Vec<_> = st[3..11].to_vec();
    let sv: Vec<_> = st[11..11 + WP as usize].to_vec();
    let cyc = |b: &mut FnBuilder| {
        let c = b.shader_cycles();
        b.let_(Op::PtrLo(Opnd::Val(c)))
    };
    let t_start = cyc(&mut b);
    let z = b.c_i32(0);
    let lvars = b.begin_loop(&[(DType::I32, Opnd::Val(z))]);
    let it = lvars[0];
    b.assume(it, Ext::c(0), Ext::c((MAX_PAGES / 2) as i64));
    let t = b.bind_i32(amdgpu_dsl::varx(it) + 0);
    if mask & PH_S != 0 {
        // K fragments of the staged window (both pages, 9 chunks each) and the S WMMAs, carried in `sv`
        let kr = kv.at(&SlotIdx::new(Opnd::I32(0), Opnd::I32(0)));
        let sb2 = b.bind_i32(kr.base());
        let kfr: Vec<Vec<ValueId>> = (0..WP)
            .map(|pg| {
                let pb = b.bind_i32(ev(sb2) + pg * layout.slot_bytes);
                let nope_row = b.bind_i32(ev(pb) + ev(kofs));
                let rope_row = b.bind_i32(ev(pb) + ev(rofs));
                kr.read(&mut b, |b| {
                    (0..9)
                        .map(|c| {
                            let lo = if c < 8 { b.bind_i32(ev(nope_row) + kdelta[c as usize]) } else { b.bind_i32(ev(rope_row)) };
                            let hi = b.bind_i32(ev(lo) + 16);
                            b.load_lds_b128x2(lo, hi)
                        })
                        .collect()
                })
            })
            .collect();
        let mut sacc: Vec<Option<Opnd>> = vec![None; WP as usize];
        for c in 0..9usize {
            for pg in 0..WP as usize {
                let d = b.wmma(WmmaKind::Fp8_16x16x64, kfr[pg][c], q[c], sacc[pg]);
                sacc[pg] = Some(Opnd::Val(d));
            }
        }
        for pg in 0..WP as usize {
            if let Some(Opnd::Val(v)) = sacc[pg] {
                b.set_var(sv[pg], Opnd::Val(v));
            }
        }
    }
    if mask & PH_PV != 0 {
        let pr_ = pbuf.at(&SlotIdx::new(Opnd::I32(0), Opnd::I32(0)));
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
        let kr = kv.at(&SlotIdx::new(Opnd::I32(0), Opnd::I32(0)));
        let kb = b.bind_i32(kr.base());
        let vfr: Vec<Vec<ValueId>> = (0..WP)
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
            .collect();
        if mask & PH_ALPHA != 0 {
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
        }
        for c in 0..8usize {
            for pg in 0..WP as usize {
                let o = ov[c];
                let d = b.wmma(WmmaKind::Fp8_16x16x64, pa[pg], vfr[pg][c], Some(Opnd::Var(o)));
                b.set_var(o, Opnd::Val(d));
            }
        }
    }
    if mask & PH_SM != 0 {
        // the softmax block on the carried S: logits, this wave's maxima to RED, the exchange barrier, the window maximum, the exponents / sums, P to PBUF2
        let x = logits(&mut b, t, &sv.iter().map(|&v| Opnd::Var(v)).collect::<Vec<_>>());
        put_max(&mut b, t, &x);
        b.wait_lds(0);
        b.barrier();
        let pm = get_max(&mut b, t);
        let (alpha1, m1, l1) = put_p(&mut b, t, &x, Opnd::Var(vm), Opnd::Var(vl), pm);
        b.set_var(vm, Opnd::Val(m1));
        b.set_var(vl, Opnd::Val(l1));
        b.set_var(va, Opnd::Val(alpha1));
    }
    if mask & PH_BAR != 0 {
        b.wait_lds(0);
        b.barrier();
    }
    let tn = b.bind_i32(amdgpu_dsl::varx(it) + 1);
    b.set_var(it, Opnd::Val(tn));
    let fin = b.cmp(CmpPred::Ge, CmpTy::I32, tn, nwin);
    b.break_if(fin);
    b.end_loop();
    let t_end = cyc(&mut b);
    // the result: per wave the loop's cycles (word 0), the window count (word 1) and a digest of every live accumulator (words 2 ..), so that no block is dead code
    let dt = b.bind_i32(ev(t_end) - ev(t_start));
    let o0 = b.bind_i32(ev(bidx) * (16 * 512 * 2) + ev(w) * 256);
    let pa0 = b.let_(Op::PtrAdd { base: Opnd::Val(pr), byte_off: Opnd::Val(o0) });
    b.store_global(pa0, 0, dt, MemTy::B32, false);
    b.store_global(pa0, 4, nwin, MemTy::B32, false);
    for k in 0..8usize {
        let e = b.frag_slice(Opnd::Var(ov[k]), 0, 1);
        b.store_global(pa0, (8 + 4 * k) as u32, e, MemTy::B32, false);
    }
    for pg in 0..WP as usize {
        let e = b.frag_slice(Opnd::Var(sv[pg]), 0, 1);
        b.store_global(pa0, (40 + 4 * pg) as u32, e, MemTy::B32, false);
    }
    b.store_global(pa0, 56, Opnd::Var(vm), MemTy::B32, false);
    b.store_global(pa0, 60, Opnd::Var(vl), MemTy::B32, false);
    b.store_global(pa0, 64, Opnd::Var(va), MemTy::B32, false);
    let _ = &plse;
    b.finish()
}
