//! MLA decode (gfx1250): a two-role prototype of the one-barrier window kernel (stub data, no KV streaming): 8 waves, waves 0 - 3 = MMA waves (S and PV WMMAs, the S tile to an LDS ring `SBUF`, P from
//! `PBUF`), waves 4 - 7 = softmax waves (the scores from `SBUF`, partial maxima through `RED`, lean arithmetic, P to `PBUF`), one work-group barrier per window; a SIMD hosts one wave of each role.
//! Window 0 is staged once; the loop runs `nwin` times. The result (per wave the loop's shader cycles in word 0 of its 256 bytes of R[0], the window count in word 1, digests) is garbage by design.
//! `mla_g1250_qh16_roles`; launch with 256 work-items (`tools/mi450/mla_roles.sh` patches the MLA host's block size).

use amdgpu_dsl::{ev, CmpPred, CmpTy, DType, Ext, FnBuilder, Function, MemTy, Op, Opnd, TdmCpol, TensorTile, ValueId, WmmaKind};
use amdgpu_stage::attention::mla::{PageLayout, DV, PAGE_BYTES, PAGE_TOKENS, QUARTER, ROPE_OFF};
use amdgpu_stage::gemm::tdm::tdm_group0;
use amdgpu_tile::shared::{Layout, LdsPlan, SlotIdx};

/// pages per sequence the LDS page table holds (ctx <= 65536)
const MAX_PAGES: i32 = 1024;

fn f32_of(b: &mut FnBuilder, v: ValueId) -> ValueId {
    b.let_(Op::BitcastF32(Opnd::Val(v)))
}



pub fn mla_g1250_qh16_roles() -> Function {
    mla_roles_with("mla_g1250_qh16_roles", 2, 2, 0)
}

/// `mla_g1250_qh16_roles_<hex mask>`: bit 0 = the MMA waves skip their window work, bit 1 = the softmax waves skip theirs (the barrier stays).
pub fn by_mask(hex: &str) -> Option<Function> {
    let mask = u32::from_str_radix(hex, 16).ok()?;
    if mask > 31 {
        return None;
    }
    Some(mla_roles_with(&format!("mla_g1250_qh16_roles_{hex}"), 2, 2, mask))
}

fn mla_roles_with(name: &str, wp: i32, depth: i32, mask: u32) -> Function {
    #[allow(non_snake_case)]
    let WP = wp;
    assert!(depth >= 2 && (depth & (depth - 1)) == 0);
    let in_flight = ((depth - 4).max(0) * wp) as u8;
    let layout = PageLayout::plan();
    // windows per barrier (mask bit 3 = 2): every producer -> consumer hop crosses one barrier, so the pipeline is 3 W windows deep and the exchange rings have 2 W slots
    let wpi: i32 = if mask & 8 != 0 { 2 } else { 1 };
    let three = mask & 16 != 0;
    let mut b = FnBuilder::new(name, if three { 384 } else { 256 });
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
    let red = lds.ring("RED", 512, 2 * wpi as u32, Layout::Linear);
    let pbuf = lds.ring("PBUF", 1024 * WP as i64, 2 * wpi as u32, Layout::Linear);
    let pbuf2 = lds.ring("PBUF2", 1024 * WP as i64, 2, Layout::Linear);
    let sbuf = lds.ring("SBUF", 8192, 2 * wpi as u32, Layout::Linear);
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
    let wfull = b.read_first_lane(wv);
    let w = b.let_(Op::AndI32(Opnd::Val(wfull), Opnd::I32(3)));
    let role = b.let_(Op::LshrI32(Opnd::Val(wfull), Opnd::I32(2)));
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
    let idx = b.bind_i32(amdgpu_dsl::varx(kk) * 128 + (ev(tid) & 127));
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

    // ---- the two-role prototype: waves 0 - 3 (role 0) = MMA waves, waves 4 - 7 (role 1) = softmax waves; window 0 staged once into slot 0 (stub data, no KV streaming) ----
    // W = windows per barrier. Window w, produced at one hop and consumed at the next (each hop crosses a barrier): the MMA waves store S(w + 3 W) to SBUF and run PV(w) from PBUF gen w; the softmax waves read S(w + 2 W), put
    // its maxima to RED gen w + 2 W, read the window maximum of w + W (RED gen w + W) and write P(w + W) (PBUF gen w + W) from the scores carried from the previous iteration; one barrier after W windows (all 8 waves).
    let w_n = wpi;
    let nslots = 2 * w_n;
    let smask = nslots - 1;
    let any_role = |b: &mut FnBuilder, r: i32| {
        let c = b.cmp(CmpPred::Eq, CmpTy::I32, role, Opnd::I32(r));
        b.let_(Op::AnyLane(Opnd::Val(c)))
    };
    let is_mma = any_role(&mut b, 0);
    let k_frags = |b: &mut FnBuilder| -> Vec<Vec<ValueId>> {
        let kr = kv.at(&SlotIdx::new(Opnd::I32(0), Opnd::I32(0)));
        let sb2 = b.bind_i32(kr.base());
        (0..WP)
            .map(|pg| {
                let pb = b.bind_i32(ev(sb2) + pg * layout.slot_bytes);
                let nope_row = b.bind_i32(ev(pb) + ev(kofs));
                let rope_row = b.bind_i32(ev(pb) + ev(rofs));
                kr.read(b, |b| {
                    (0..9)
                        .map(|c| {
                            let lo = if c < 8 { b.bind_i32(ev(nope_row) + kdelta[c as usize]) } else { b.bind_i32(ev(rope_row)) };
                            let hi = b.bind_i32(ev(lo) + 16);
                            b.load_lds_b128x2(lo, hi)
                        })
                        .collect()
                })
            })
            .collect()
    };
    let s_chain = |b: &mut FnBuilder, kfr: &[Vec<ValueId>]| -> Vec<Opnd> {
        let mut sacc: Vec<Option<Opnd>> = vec![None; WP as usize];
        for c in 0..9usize {
            for pg in 0..WP as usize {
                let d = b.wmma(WmmaKind::Fp8_16x16x64, kfr[pg][c], q[c], sacc[pg]);
                sacc[pg] = Some(Opnd::Val(d));
            }
        }
        sacc.iter().map(|o| o.unwrap()).collect()
    };
    let sbuf_store = |b: &mut FnBuilder, gen: ValueId, slot: Opnd, sacc: &[Opnd]| {
        let r = sbuf.at(&SlotIdx::new(slot, Opnd::Val(gen)));
        let base = b.bind_i32(r.base() + ev(w) * 2048 + ev(lane) * 32);
        r.write(b, |b| {
            for pg in 0..WP as usize {
                for h in 0..2usize {
                    let part = b.frag_slice(sacc[pg], (4 * h) as u8, 4);
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
            let mut out = Vec::new();
            for pg in 0..WP as usize {
                for h in 0..2usize {
                    let a = b.bind_i32(ev(base) + (1024 * pg as i32 + 16 * h as i32));
                    let v4 = b.load_lds_b128(a);
                    for k in 0..4u8 {
                        let e = b.frag_slice(v4, k, 1);
                        out.push(f32_of(b, e));
                    }
                }
            }
            out
        })
    };
    let put_max_g = |b: &mut FnBuilder, gen: ValueId, slot: Opnd, x: &[ValueId]| {
        let mut mx = x[0];
        for r in 1..x.len() {
            mx = b.max_f32(mx, x[r]);
        }
        let r = red.at(&SlotIdx::new(slot, Opnd::Val(gen)));
        let a = b.bind_i32(r.base() + ev(w) * 128 + ev(g) * 64 + ev(m) * 4);
        r.write(b, |b| b.store_lds(a, mx, MemTy::B32, false));
    };
    let get_max_g = |b: &mut FnBuilder, gen: ValueId, slot: Opnd| -> ValueId {
        let r = red.at(&SlotIdx::new(slot, Opnd::Val(gen)));
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
    let pbuf_write = |b: &mut FnBuilder, gen: ValueId, slot: Opnd, vals: &dyn Fn(&mut FnBuilder, usize) -> (ValueId, ValueId)| {
        let r = pbuf.at(&SlotIdx::new(slot, Opnd::Val(gen)));
        let pw = b.bind_i32(r.base() + ev(lane) * 32 + ev(w) * 8);
        r.write(b, |b| {
            for pg in 0..WP as usize {
                let (p0, p1) = vals(b, pg);
                let a0 = b.bind_i32(ev(pw) + 1024 * pg as i32);
                b.store_lds(a0, p0, MemTy::B32, false);
                let a1 = b.bind_i32(ev(pw) + (1024 * pg as i32 + 4));
                b.store_lds(a1, p1, MemTy::B32, false);
            }
        });
    };
    let put_p_lean = |b: &mut FnBuilder, gen: ValueId, slot: Opnd, x: &[Opnd], vm_old: Opnd, vl_old: Opnd, pm: ValueId| -> (ValueId, ValueId) {
        let pms = b.mul_f32(pm, cscale);
        let m_new = b.max_f32(vm_old, pms);
        let dm = b.sub_f32(vm_old, m_new);
        let alpha = b.exp2_f32(dm);
        let negm = b.sub_f32(zf, m_new);
        let mut lsum = b.mul_f32(vl_old, alpha);
        let p: Vec<ValueId> = x
            .iter()
            .map(|&xi| {
                let d = b.let_(Op::FmaF32(xi, Opnd::Val(cscale), Opnd::Val(negm)));
                b.exp2_f32(d)
            })
            .collect();
        for pi in &p {
            lsum = b.add_f32(lsum, *pi);
        }
        pbuf_write(b, gen, slot, &|b, pg| {
            (
                b.cvt_pk_fp8x4_f32(p[8 * pg], p[8 * pg + 1], p[8 * pg + 2], p[8 * pg + 3]),
                b.cvt_pk_fp8x4_f32(p[8 * pg + 4], p[8 * pg + 5], p[8 * pg + 6], p[8 * pg + 7]),
            )
        });
        (m_new, lsum)
    };
    // loop state: O (8 Frag8, MMA waves), the carried scores (16 f32 per window of the iteration), running max and sum (softmax waves)
    let nx = 16 * w_n as usize;
    let mut inits: Vec<(DType, Opnd)> = Vec::new();
    for _ in 0..8 {
        let z = b.frag_cat(&[Opnd::I32(0); 8]);
        inits.push((WmmaKind::Fp8_16x16x64.acc_ty(), Opnd::Val(z)));
    }
    for _ in 0..nx {
        inits.push((DType::F32, Opnd::Val(zf)));
    }
    inits.push((DType::F32, Opnd::Val(neg)));
    inits.push((DType::F32, Opnd::Val(zf)));
    let st = b.declare_vars(&inits);
    let ov: Vec<_> = st[0..8].to_vec();
    let xs: Vec<_> = st[8..8 + nx].to_vec();
    let (vm, vl) = (st[8 + nx], st[9 + nx]);
    let gc: Vec<ValueId> = (0..3 * w_n).map(|k| b.c_i32(k)).collect();
    let slot_c = |k: i32| Opnd::I32(k & smask);
    // ---- prologue: window 0 into slot 0; S(W .. 3 W - 1) on the MMA waves; the softmax waves take S(W .. 2 W - 1) (carried scores, maxima to RED) and write stub P gens 0 .. W - 1 ----
    b.begin_if(is_mma);
    stage(&mut b, gc[0]);
    b.wait_tensor(0);
    b.end_if();
    b.wait_lds(0);
    b.barrier();
    b.begin_if(is_mma);
    let kf0 = k_frags(&mut b);
    let s0 = s_chain(&mut b, &kf0);
    for k in w_n..3 * w_n {
        sbuf_store(&mut b, gc[k as usize], slot_c(k), &s0);
    }
    b.end_if();
    b.wait_lds(0);
    b.barrier();
    let is_sm = is_sm_marker(&mut b, role, if three { 2 } else { 1 });
    b.begin_if(is_sm);
    for k in w_n..2 * w_n {
        let xk = sbuf_load(&mut b, gc[k as usize], slot_c(k));
        put_max_g(&mut b, gc[k as usize], slot_c(k), &xk);
        for i in 0..16usize {
            b.set_var(xs[16 * (k - w_n) as usize + i], Opnd::Val(xk[i]));
        }
    }
    for k in 0..w_n {
        let stub = b.c_i32(0x3838_3838);
        pbuf_write(&mut b, gc[k as usize], slot_c(k), &|_b, _pg| (stub, stub));
    }
    b.end_if();
    b.wait_lds(0);
    b.barrier();
    let cyc = |b: &mut FnBuilder| {
        let c = b.shader_cycles();
        b.let_(Op::PtrLo(Opnd::Val(c)))
    };
    let t_start = cyc(&mut b);
    let o0 = b.bind_i32(ev(bidx) * (16 * 512 * 2) + ev(wfull) * 256);
    let pa0 = b.let_(Op::PtrAdd { base: Opnd::Val(pr), byte_off: Opnd::Val(o0) });
    let z = b.c_i32(0);
    let nitr = b.bind_i32((ev(nwin) + (w_n - 1)) >> ((w_n as u32).trailing_zeros() as i32));
    // ---- the MMA waves' loop: S and PV in each wave, or (with `sub`) a uniform branch inside the loop that makes the first four waves S waves and the next four PV waves ----
    let mma_loop = |b: &mut FnBuilder, do_s: bool, do_pv: bool, sub: Option<ValueId>| {
        let lvars = b.begin_loop(&[(DType::I32, Opnd::Val(z))]);
        let it = lvars[0];
        b.assume(it, Ext::c(0), Ext::c((MAX_PAGES / 2) as i64));
        for j in 0..w_n {
            let wn = b.bind_i32(amdgpu_dsl::varx(it) * w_n + j);
            let wa = b.bind_i32(ev(wn) + 3 * w_n);
            let sp = b.bind_i32(ev(wn) & smask);
            let sa = b.bind_i32(ev(wa) & smask);
            let kr = kv.at(&SlotIdx::new(Opnd::I32(0), Opnd::I32(0)));
            if let Some(c) = sub {
                b.begin_if(c);
            }
            if do_s {
                let kfr = k_frags(b);
                let sacc = s_chain(b, &kfr);
                sbuf_store(b, wa, Opnd::Val(sa), &sacc);
            }
            if sub.is_some() {
                b.else_();
            }
            if do_pv {
                let pr_ = pbuf.at(&SlotIdx::new(Opnd::Val(sp), Opnd::Val(wn)));
                let pbase = b.bind_i32(pr_.base() + ev(lane) * 32);
                let pa: Vec<ValueId> = pr_.read(b, |b| {
                    (0..WP)
                        .map(|pg| {
                            let lo = b.bind_i32(ev(pbase) + 1024 * pg);
                            let hi = b.bind_i32(ev(lo) + 16);
                            b.load_lds_b128x2(lo, hi)
                        })
                        .collect()
                });
                let kb = b.bind_i32(kr.base());
                let vfr: Vec<Vec<ValueId>> = (0..WP)
                    .map(|pg| {
                        let vrow = b.bind_i32(ev(kb) + pg * layout.slot_bytes + ev(vofs));
                        kr.read(b, |b| {
                            (0..8)
                                .map(|nt| {
                                    let a: Vec<ValueId> = (0..4).map(|jj| b.bind_i32(ev(vrow) + vdelta[8 * jj as usize + nt as usize])).collect();
                                    b.load_lds_tr_b8x4(a[0], a[1], a[2], a[3])
                                })
                                .collect()
                        })
                    })
                    .collect();
                for nt in 0..8usize {
                    for pg in 0..WP as usize {
                        let d = b.wmma(WmmaKind::Fp8_16x16x64, pa[pg], vfr[pg][nt], Some(Opnd::Var(ov[nt])));
                        b.set_var(ov[nt], Opnd::Val(d));
                    }
                }
            }
            if sub.is_some() {
                b.end_if();
            }
        }
        b.wait_lds(0);
        b.barrier();
        let tn = b.bind_i32(amdgpu_dsl::varx(it) + 1);
        b.set_var(it, Opnd::Val(tn));
        let fin = b.cmp(CmpPred::Ge, CmpTy::I32, tn, nitr);
        b.break_if(fin);
        b.end_loop();
    };
    // ---- the softmax waves' loop ----
    let sm_loop = |b: &mut FnBuilder| {
        let lvars = b.begin_loop(&[(DType::I32, Opnd::Val(z))]);
        let it = lvars[0];
        b.assume(it, Ext::c(0), Ext::c((MAX_PAGES / 2) as i64));
        for j in 0..w_n {
            let wn = b.bind_i32(amdgpu_dsl::varx(it) * w_n + j);
            let w1 = b.bind_i32(ev(wn) + w_n);
            let w2 = b.bind_i32(ev(wn) + 2 * w_n);
            let s1 = b.bind_i32(ev(w1) & smask);
            let s2 = b.bind_i32(ev(w2) & smask);
            let xn = sbuf_load(b, w2, Opnd::Val(s2));
            let pm = get_max_g(b, w1, Opnd::Val(s1));
            let xo: Vec<Opnd> = (0..16usize).map(|i| Opnd::Var(xs[16 * j as usize + i])).collect();
            let (m_new, l_new) = put_p_lean(b, w1, Opnd::Val(s1), &xo, Opnd::Var(vm), Opnd::Var(vl), pm);
            put_max_g(b, w2, Opnd::Val(s2), &xn);
            for i in 0..16usize {
                b.set_var(xs[16 * j as usize + i], Opnd::Val(xn[i]));
            }
            b.set_var(vm, Opnd::Val(m_new));
            b.set_var(vl, Opnd::Val(l_new));
        }
        b.wait_lds(0);
        b.barrier();
        let tn = b.bind_i32(amdgpu_dsl::varx(it) + 1);
        b.set_var(it, Opnd::Val(tn));
        let fin = b.cmp(CmpPred::Ge, CmpTy::I32, tn, nitr);
        b.break_if(fin);
        b.end_loop();
    };
    if three {
        // waves 0 - 7: the MMA role, its first four waves the S waves and the next four the PV waves (a uniform branch on the wave inside the role's loop); waves 8 - 11: the softmax waves
        let subc = {
            let c = b.cmp(CmpPred::Eq, CmpTy::I32, role, Opnd::I32(0));
            b.let_(Op::AnyLane(Opnd::Val(c)))
        };
        let is_mma3 = {
            let c = b.cmp(CmpPred::Le, CmpTy::I32, role, Opnd::I32(1));
            b.let_(Op::AnyLane(Opnd::Val(c)))
        };
        b.begin_if(is_mma3);
        mma_loop(&mut b, true, true, Some(subc));
        b.else_();
        sm_loop(&mut b);
        b.end_if();
    } else {
        b.begin_if(is_mma);
        mma_loop(&mut b, true, true, None);
        b.else_();
        sm_loop(&mut b);
        b.end_if();
    }
    let t_end = cyc(&mut b);
    // the result: per wave the loop's cycles (word 0), the window count (word 1), digests
    let dt = b.bind_i32(ev(t_end) - ev(t_start));
    b.store_global(pa0, 0, dt, MemTy::B32, false);
    b.store_global(pa0, 4, nwin, MemTy::B32, false);
    for k in 0..8usize {
        let e = b.frag_slice(Opnd::Var(ov[k]), 0, 1);
        b.store_global(pa0, (8 + 4 * k) as u32, e, MemTy::B32, false);
    }
    for k in 0..16usize {
        b.store_global(pa0, (40 + 4 * k) as u32, Opnd::Var(xs[k]), MemTy::B32, false);
    }
    b.store_global(pa0, 104, Opnd::Var(vm), MemTy::B32, false);
    b.store_global(pa0, 108, Opnd::Var(vl), MemTy::B32, false);
    let _ = (&plse, pbuf2.base, lsb.base, out.base, is_mma);
    b.finish()
}

fn is_sm_marker(b: &mut FnBuilder, role: ValueId, r: i32) -> ValueId {
    let c = b.cmp(CmpPred::Eq, CmpTy::I32, role, Opnd::I32(r));
    b.let_(Op::AnyLane(Opnd::Val(c)))
}
