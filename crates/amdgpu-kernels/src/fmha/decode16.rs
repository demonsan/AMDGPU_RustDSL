//! Paged-KV decode with 16x16x32 bf16 MFMAs (perf pass, GQA G <= 16): `pa_decode16_bf16_hd128_p<P>`, the ABI / layouts / splits of
//! `decode.rs` (and its `attn_combine_bf16_hd128`), but the S^T / O^T tiles are 16 x 16: the 16 columns are the G heads of the KV head (the
//! 32-row form wastes 3/4 of the MFMA work for G = 8). A lane holds head l % 16 and, per 16-key (16-dim) block, the 4 keys (dims) 4 g + 0..3
//! (g = l / 16); the softmax statistics are combined over the 4 lane groups (permlane16 + permlane32 swaps). P^T feeds the 32-key PV step in the
//! key order {4 g + 0..3 of block 2 t, 16 + 4 g + 0..3 of block 2 t + 1}; V^T by ds_read_b64_tr_b16 with rows 16 r + 4 g + s / 4.
use amdgpu_dsl::{CompileProfile, DType, Ext, FnBuilder, Function, MmaKind, MmaShape, MmaTy, Op, Opnd, ValueId, VarId};
use amdgpu_tile::{ev, LdsPlan, Lane};

const D: i32 = 128;
const ROW: i32 = 2 * D;

/// max / sum over the 4 lane groups (lanes l, l ^ 16, l ^ 32, l ^ 48)
fn group4(b: &mut FnBuilder, x: ValueId, max: bool) -> ValueId {
    let f = |b: &mut FnBuilder, a: ValueId, c: ValueId| if max { b.max_f32(a, c) } else { b.add_f32(a, c) };
    let p = b.permlane16_swap(x, x);
    let (p0, p1) = (b.frag2_lane(p, 0), b.frag2_lane(p, 1));
    let (p0, p1) = (b.bitcast_f32(p0), b.bitcast_f32(p1));
    let y = f(b, p0, p1);
    let (a, c) = b.permlane32_swap(y, y);
    let (a, c) = (b.bitcast_f32(a), b.bitcast_f32(c));
    f(b, a, c)
}

/// `kt` keys per tile and wave: 32 (a 16 KB private K / V region: two CTAs of 4 waves per CU; large batches) or 64 (32 KB, one CTA per CU;
/// small grids: more keys per wave). `crate::dispatch::decode16_config` picks it with the split count.
pub fn pa_decode16_bf16_hd128(name: &str, page: i32, kt: i32) -> Function {
    pa_decode16_x(name, page, kt, false)
}

/// `fp8` (`pa_decode16_fp8_hd128_p<P>`): an fp8 e4m3 KV cache with per-tensor scales (`reshape_cache_flash_fp8_h<Hk>_d128`: [pages, P, HK,
/// 128] bytes; k_scale @208, v_scale @224 f32). K: an fp8 tile (rows of 128 bytes, chunk ^ key % 8), each QK operand 8 bytes -> 8 bf16
/// (exact); V: the fp8 rows staged, then converted into the bf16 V tile the transposed reads use. k_scale folds into the softmax scale,
/// v_scale into the output.
pub fn pa_decode16_x(name: &str, page: i32, kt: i32, fp8: bool) -> Function {
    assert!(!fp8 || kt == 32, "decode16 fp8: 32-key tiles");
    assert!(kt == 32 || kt == 64, "decode16: 32- or 64-key tiles");
    #[allow(non_snake_case)]
    let KT = kt;
    #[allow(non_snake_case)]
    let LKT = kt.trailing_zeros() as i32;
    #[allow(non_snake_case)]
    let NKB = (kt / 16) as usize;    let mut b = FnBuilder::new(name, 256);
    b.compile_profile(CompileProfile::Gemm);
    for (n, i) in [("ptr_po", 0u32), ("ptr_q", 1), ("ptr_k", 2), ("ptr_v", 3), ("ptr_bt", 4), ("ptr_cl", 5), ("ptr_plse", 12)] {
        b.arg(n, DType::Ptr, 16 * i, 8);
    }
    b.arg("scale", DType::F32, 96, 4);
    for (n, i) in [("H", 7u32), ("HK", 8), ("max_pages", 9), ("num_pages", 10), ("S", 11)] {
        b.arg(n, DType::I32, 16 * i, 4);
    }
    if fp8 {
        b.arg("k_scale", DType::F32, 208, 4);
        b.arg("v_scale", DType::F32, 224, 4);
    }
    // KV cache bytes per (token, head) row
    let krow: i32 = if fp8 { D } else { ROW };
    // ---- contract ----
    let (hh, hk, mp, np, ss) = (Ext::ka(112), Ext::ka(128), Ext::ka(144), Ext::ka(160), Ext::ka(176));
    b.sym_range("G", 1, 16);
    b.sym_range("B", 1, 1 << 16);
    b.ka_range(128, 1, 64);
    b.ka_is(112, hk.clone() * Ext::sym("G"));
    b.ka_range(144, 1, 1 << 20);
    b.ka_range(160, 1, 1 << 20);
    b.ka_range(176, 1, 64);
    b.grid_is(0, hk.clone());
    b.grid_is(1, Ext::sym("B"));
    b.grid_is(2, ss.clone());
    let bb = Ext::sym("B");
    b.arg_tensor("ptr_q", bb.clone() * hh.clone() * Ext::c(ROW as i64));
    let cache = np * Ext::c(page as i64) * hk * Ext::c(krow as i64);
    b.arg_tensor("ptr_k", cache.clone());
    b.arg_tensor("ptr_v", cache);
    b.arg_tensor("ptr_bt", bb.clone() * mp * Ext::c(4));
    b.arg_tensor("ptr_cl", bb.clone() * Ext::c(4));
    b.arg_tensor("ptr_po", bb.clone() * hh.clone() * ss.clone() * Ext::c(4 * 4 * D as i64));
    b.arg_tensor("ptr_plse", bb * hh * ss * Ext::c(16));

    let kp = |b: &mut FnBuilder, i: u32| b.let_(Op::KernargPtr { offset: 16 * i });
    let ki = |b: &mut FnBuilder, i: u32| b.let_(Op::KernargI32 { offset: 16 * i });
    let (p_po, p_q, p_k, p_v, p_bt, p_cl, p_lse) = (kp(&mut b, 0), kp(&mut b, 1), kp(&mut b, 2), kp(&mut b, 3), kp(&mut b, 4), kp(&mut b, 5), kp(&mut b, 12));
    let scale = b.let_(Op::KernargF32 { offset: 96 });
    let (hq, hkv_n, maxp, npages, s_n) = (ki(&mut b, 7), ki(&mut b, 8), ki(&mut b, 9), ki(&mut b, 10), ki(&mut b, 11));
    let hk = b.let_(Op::WorkgroupIdX);
    let bat = b.let_(Op::WorkgroupIdY);
    let z = b.let_(Op::WorkgroupIdZ);
    let lane = Lane::new(&mut b);
    // 16x16 MFMA: the lane's head column j = l % 16, lane group g = l / 16 (keys / dims 4 g + 0..3 of the C fragments)
    let grp = b.bind_i32(lane.lane().lshr(4));
    let wv = { let v = b.bind_i32(lane.wave()); b.read_first_lane(v) };
    // GQA group G = H / HK (H = HK G): the lane's query head (clamped), live rows j < G
    let gq = b.let_(Op::KernargI32 { offset: 112 });
    let gsz = {
        // G = H / HK by binary long division (G <= 32)
        let mut r = b.bind_i32(ev(gq));
        let mut q = b.c_i32(0);
        for bit in [32i32, 16, 8, 4, 2, 1] {
            let xb = b.bind_i32(ev(hkv_n) * bit);
            let take = b.bind_i32(ev(r).ge(ev(xb)).select_i32(1, 0));
            q = b.bind_i32(ev(q) + ev(take) * bit);
            r = b.bind_i32(ev(r) - ev(take) * ev(xb));
        }
        let q = b.read_first_lane(q);
        b.assume_trusted(q, Ext::c(1), Ext::c(16), "GQA group G = H / HK in [1, 16] (contract H = HK G)");
        q
    };
    let j = b.bind_i32(lane.lane() & 15);
    let row_live = b.bind_cond(ev(j).lt(ev(gsz)));
    let head = b.bind_i32(ev(hk) * ev(gsz) + ev(j).min(ev(gsz) - 1));
    // ---- the wave's key split ----
    let nsplit = b.bind_i32(ev(s_n) << 2);
    let ns = { let v = b.bind_i32((ev(z) << 2) + ev(wv)); b.read_first_lane(v) };
    let ncl = { let v = b.bind_i32((ev(bat) + 1) << 2); b.read_first_lane(v) };
    let r_cl = b.buffer_desc_dyn(p_cl, ncl);
    let cl_so = { let v = b.bind_i32(ev(bat) << 2); b.read_first_lane(v) };
    let zero = b.c_i32(0);
    let ctx = { let v = b.buffer_load_f32(r_cl, zero, cl_so, 0); let v = b.bitcast_i32(v); b.read_first_lane(v) };
    b.assume_trusted(ctx, Ext::c(0), Ext::ka(144) * Ext::c(page as i64), "decode: a context length lies in [0, max_pages P] (host contract)");
    // chunk = ceil(tiles / NS) tiles (NS = 4 S, S a power of two <= 16); [k_lo, k_hi) = [ns chunk, min(ctx, (ns + 1) chunk))
    let lg = {
        let sv = ev(s_n);
        let s1 = sv.clone().gt(1).select_i32(1, 0);
        let s2 = sv.clone().gt(2).select_i32(2, s1);
        let s3 = sv.clone().gt(4).select_i32(3, s2);
        let v = b.bind_i32(sv.gt(8).select_i32(4, s3) + 2);
        b.read_first_lane(v)
    };
    let ntt = b.bind_i32((ev(ctx) + (KT - 1)).lshr(LKT));
    let per = b.bind_i32(((ev(ntt) + ev(nsplit) - 1).lshr(ev(lg))) << LKT);
    let k_lo = { let v = b.bind_i32(ev(ns) * ev(per)); b.read_first_lane(v) };
    let k_hi = { let v = b.bind_i32((ev(k_lo) + ev(per)).min(ev(ctx))); b.read_first_lane(v) };
    let ntile = { let v = b.bind_i32((ev(k_hi) - ev(k_lo) + (KT - 1)).max(0).lshr(LKT)); b.read_first_lane(v) };

    // ---- descriptors ----
    let r_q = { let n = b.bind_i32((ev(bat) + 1) * ev(hq) * ROW); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_q, n) };
    let page_bytes = page * krow;
    let r_k = { let n = b.bind_i32(ev(npages) * ev(hkv_n) * page_bytes); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_k, n) };
    let r_v = { let n = b.bind_i32(ev(npages) * ev(hkv_n) * page_bytes); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_v, n) };
    let r_bt = { let n = b.bind_i32((ev(bat) + 1) * ev(maxp) << 2); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_bt, n) };
    let bt_so = { let v = b.bind_i32(ev(bat) * ev(maxp) << 2); b.read_first_lane(v) };
    let maxk = b.bind_i32((ev(maxp) * page) - 1);


    // ---- Q^T fragments (B operand of 16x16x32: lane = head column, k = d 32 c + 8 g + 0..7) ----
    let q_vo = b.bind_i32((ev(bat) * ev(hq) + ev(head)) * ROW + (ev(grp) << 4));
    let c0 = b.c_i32(0);
    let qf: Vec<ValueId> = (0..4).map(|c| b.buffer_load_b128(r_q, q_vo, c0, (64 * c) as u32)).collect();

    // ---- private LDS: per wave K (16 KB) + V (16 KB), as decode.rs ----
    let mut lds = LdsPlan::new();
    // K and V in separate private regions (their DMAs complete separately: QK waits for K only)
    let kreg = lds.private("K", (KT * krow) as i64, 4, &[]);
    let vreg = lds.private("V", (KT * ROW) as i64, 4, &[]);
    // fp8: the staged fp8 V rows (converted into the bf16 V tile)
    let vsreg = fp8.then(|| lds.private("VS", (KT * D) as i64, 4, &[]));
    lds.commit(&mut b);
    let me = kreg.mine(wv);
    let mev = vreg.mine(wv);
    let mvs = vsreg.as_ref().map(|r| r.mine(wv));
    let mbase = me.base();
    let vbase0 = mev.base();
    let l16 = b.bind_i32(lane.lane() & 15);
    let kk_of = |i: i32| (lane.lane().lshr(4)) + 4 * i;
    // K reads (A: lane row = key l % 16 of a 16-key block, k = d 32 c + 8 g): chunk (4 c + g) ^ (key % 16)
    let kr: Vec<ValueId> = (0..4).map(|c| b.bind_i32((ev(l16)) * ROW + (((ev(grp) + 4 * c) ^ ev(l16)) << 4))).collect();
    // V^T reads (A: lane row = d 16 db + l % 16 ... by tr_b16: lane s = l % 16 at key row 16 r + 4 g + s / 4, dims 16 db + 4 (s % 4))
    let vr = b.bind_i32((ev(grp) * 4 + ev(l16).lshr(2)) * ROW + ((ev(l16) & 3) << 3));
    let c_l2e = b.c_f32(std::f32::consts::LOG2_E);
    let csc = b.mul_f32(scale, c_l2e);
    // fp8: the K scale folds into the softmax scale, the V scale into the output
    let (csc, vsc) = if fp8 {
        let ks = b.let_(Op::KernargF32 { offset: 208 });
        let vs = b.let_(Op::KernargF32 { offset: 224 });
        (b.mul_f32(csc, ks), Some(vs))
    } else {
        (csc, None)
    };
    let neg_inf = b.c_f32(f32::NEG_INFINITY);
    let kind = MmaKind::new(MmaShape::M16N16K32, MmaTy::Bf16);
    let zero4 = b.frag_zero(DType::Frag4);
    let m_init = b.c_f32(-1.0e30);
    let l_init = b.c_f32(0.0);
    // loop state: it, m, l, O^T (8 blocks of 16 d), S^T (4 blocks of 16 keys)
    let mut inits: Vec<(DType, Opnd)> = vec![(DType::I32, Opnd::I32(0)), (DType::F32, Opnd::Val(m_init)), (DType::F32, Opnd::Val(l_init))];
    inits.extend((0..8 + NKB).map(|_| (DType::Frag4, Opnd::Val(zero4))));
    let any = b.bind_cond(ev(ntile).gt(0));
    b.begin_if(Opnd::Val(any));
    let vars = b.begin_loop(&inits);
    let (it, m_v, l_v) = (vars[0], vars[1], vars[2]);
    let ov: Vec<VarId> = vars[3..11].to_vec();
    let sv_: Vec<VarId> = vars[11..11 + NKB].to_vec();
    let itx = amdgpu_dsl::varx(it);
    let key_t = { let v = b.bind_i32(ev(k_lo) + (itx.clone() << LKT)); b.read_first_lane(v) };
    b.wait_lds(0);
    let lookups: Vec<(ValueId, ValueId)> = (0..if fp8 { KT / 8 } else { KT / 4 })
        .map(|i| {
            // (fp8 rows of 128 bytes: 8 lanes per key, keys 8 i + l / 8)
            let key = if fp8 { b.bind_i32(ev(key_t) + (lane.lane().lshr(3)) + 8 * i) } else { b.bind_i32(ev(key_t) + kk_of(i)) };
            let kc = b.bind_i32(ev(key).min(ev(k_hi) - 1).max(0).min(ev(maxk)));
            let bvo = b.bind_i32(ev(kc).lshr(page.trailing_zeros() as i32) << 2);
            (kc, b.buffer_load_f32(r_bt, bvo, bt_so, 0))
        })
        .collect();
    // all K DMAs, then all V DMAs: QK and the softmax wait only for K (counted vmcnt), the V rows land meanwhile
    let nv = lookups.len() as u8;
    let rows: Vec<(ValueId, ValueId)> = lookups
        .iter()
        .map(|&(kc, pv)| {
            let pg = b.bitcast_i32(pv);
            let pg = b.bind_i32(ev(pg).max(0).min(ev(npages) - 1));
            (kc, b.bind_i32(((ev(pg) * page + (ev(kc) & (page - 1))) * ev(hkv_n) + ev(hk)) * krow))
        })
        .collect();
    let l8 = b.bind_i32(lane.lane() & 7);
    me.write(&mut b, Opnd::Var(it), |b| {
        for (i, &(kc, rowoff)) in rows.iter().enumerate() {
            // (fp8: chunk ^ key % 8 within the 8-chunk row)
            let kvo = if fp8 { b.bind_i32(ev(rowoff) + ((ev(l8) ^ (ev(kc) & 7)) << 4)) } else { b.bind_i32(ev(rowoff) + ((ev(l16) ^ (ev(kc) & 15)) << 4)) };
            let m0k = b.bind_i32(mbase.clone() + ((i as i32) << 10));
            b.buffer_load_lds_b128(r_k, kvo, c0, m0k, 0);
        }
    });
    match &mvs {
        Some(ms) => {
            let sb = ms.base();
            ms.write(&mut b, Opnd::Var(it), |b| {
                for (i, &(_, rowoff)) in rows.iter().enumerate() {
                    let vvo = b.bind_i32(ev(rowoff) + (ev(l8) << 4));
                    let m0v = b.bind_i32(sb.clone() + ((i as i32) << 10));
                    b.buffer_load_lds_b128(r_v, vvo, c0, m0v, 0);
                }
            });
        }
        None => mev.write(&mut b, Opnd::Var(it), |b| {
            for (i, &(_, rowoff)) in rows.iter().enumerate() {
                let vvo = b.bind_i32(ev(rowoff) + (ev(l16) << 4));
                let m0v = b.bind_i32(vbase0.clone() + ((i as i32) << 10));
                b.buffer_load_lds_b128(r_v, vvo, c0, m0v, 0);
            }
        }),
    }
    b.wait_vm(nv);
    // ---- S^T: 4 blocks of 16 keys x 4 d chunks (block inner: independent chains) ----
    let kb = b.bind_i32(mbase.clone());
    me.read_back(&mut b, Opnd::Var(it), |b| {
        for c in 0..4usize {
            for blk in 0..NKB as i32 {
                let kf = if fp8 {
                    // 8 fp8 (d 32 c + 8 g ..) -> 8 bf16: chunk (2 c + g / 2) ^ (key % 8), half g % 2
                    let ch = b.bind_i32(ev(grp).lshr(1) + 2 * c as i32);
                    let a = b.bind_i32(ev(kb) + (16 * blk + ev(l16)) * krow + (((ev(ch) ^ (ev(l16) & 7))) << 4) + ((ev(grp) & 1) << 3));
                    let w2 = b.load_lds_b64(a);
                    let one = b.c_f32(1.0);
                    let ws: Vec<ValueId> = (0..2u8)
                        .flat_map(|dw| [(dw, 0u8), (dw, 1u8)])
                        .map(|(dw, sel)| {
                            let x = b.frag2_lane(w2, dw);
                            b.cvt(amdgpu_dsl::CvtOp::ScalePkBf16Fp8, &[Opnd::Val(x), Opnd::Val(one)], sel)
                        })
                        .collect();
                    b.pack4(ws[0], ws[1], ws[2], ws[3])
                } else {
                    let a = b.bind_i32(ev(kr[c]) + ev(kb) + blk * 16 * ROW);
                    b.load_lds_b128(a)
                };
                b.var_mfma(sv_[blk as usize], kind, kf, qf[c], c == 0);
            }
        }
    });
    // ---- softmax: values key_t + 16 blk + 4 g + r ----
    let key0 = b.bind_i32(ev(key_t) + (ev(grp) << 2));
    let mut s: Vec<ValueId> = Vec::with_capacity(16);
    for blk in 0..NKB {
        for r in 0..4u8 {
            let x = b.frag4_lane(Opnd::Var(sv_[blk]), r);
            let x = b.bitcast_f32(x);
            let keep = b.bind_cond((ev(key0) + (16 * blk as i32 + r as i32)).lt(ev(k_hi)));
            s.push(b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(keep), a: Opnd::Val(x), b: Opnd::Val(neg_inf) }));
        }
    }
    let mut mx = s[0];
    for pair in s[1..].chunks(2) {
        mx = if pair.len() == 2 { b.max3_f32(mx, pair[0], pair[1]) } else { b.max_f32(mx, pair[0]) };
    }
    let mt = group4(&mut b, mx, true);
    let mt = b.mul_f32(mt, csc);
    let thr = b.c_f32(8.0);
    let mlim = b.add_f32(Opnd::Var(m_v), thr);
    let grow = b.cmp_lt_f32(mlim, mt);
    let grow = b.any_lane(grow);
    b.begin_if(Opnd::Val(grow));
    {
        let mn = b.max_f32(Opnd::Var(m_v), mt);
        let dm = b.sub_f32(Opnd::Var(m_v), mn);
        let alpha = b.exp2_f32(dm);
        for v in &ov {
            for r in 0..4u8 {
                let x = b.frag4_lane(Opnd::Var(*v), r);
                let x = b.bitcast_f32(x);
                let y = b.mul_f32(x, alpha);
                b.var_set_lane(*v, r, y);
            }
        }
        let l2 = b.mul_f32(Opnd::Var(l_v), alpha);
        b.set_var(l_v, l2);
        b.set_var(m_v, mn);
    }
    b.end_if();
    let zf = b.c_f32(0.0);
    let nm = b.sub_f32(zf, Opnd::Var(m_v));
    let p: Vec<ValueId> = s
        .iter()
        .map(|x| {
            let t = b.let_(Op::FmaF32(Opnd::Val(*x), Opnd::Val(csc), Opnd::Val(nm)));
            b.exp2_f32(t)
        })
        .collect();
    let mut ps = p[0];
    for x in &p[1..] {
        ps = b.add_f32(ps, *x);
    }
    let l_new = b.add_f32(Opnd::Var(l_v), ps);
    b.set_var(l_v, l_new);
    b.wait_vm(0);
    if let Some(ms) = &mvs {
        // fp8: the staged V rows -> the bf16 V tile (lane: key l / 2, d half l % 2: 64 fp8 -> 64 bf16)
        let sb = ms.base();
        let key = b.bind_i32(lane.lane().lshr(1));
        let hf = b.bind_i32(lane.lane() & 1);
        let src = b.bind_i32(sb.clone() + ev(key) * D + (ev(hf) << 6));
        let raw: Vec<ValueId> = ms.read_back(&mut b, Opnd::Var(it), |b| (0..4i32).map(|q| { let a = b.bind_i32(ev(src) + 16 * q); b.load_lds_b128(a) }).collect());
        let one = b.c_f32(1.0);
        let dst = b.bind_i32(vbase0.clone() + ev(key) * ROW + (ev(hf) << 7));
        mev.write(&mut b, Opnd::Var(it), |b| {
            for (q, f) in raw.iter().enumerate() {
                let ws: Vec<ValueId> = (0..4u8)
                    .flat_map(|dw| [(dw, 0u8), (dw, 1u8)])
                    .map(|(dw, sel)| {
                        let x = b.frag4_lane(*f, dw);
                        b.cvt(amdgpu_dsl::CvtOp::ScalePkBf16Fp8, &[Opnd::Val(x), Opnd::Val(one)], sel)
                    })
                    .collect();
                for w in 0..2usize {
                    let v4 = b.pack4(ws[4 * w], ws[4 * w + 1], ws[4 * w + 2], ws[4 * w + 3]);
                    let a = b.bind_i32(ev(dst) + (32 * q + 16 * w) as i32);
                    b.store_lds_b128(a, v4);
                }
            }
        });
        b.wait_lds(0);
    }
    // ---- O^T += V^T P^T: 2 steps of 32 keys x 8 blocks of 16 d ----
    let vb = b.bind_i32(vbase0.clone() + ev(vr));
    mev.read_back(&mut b, Opnd::Var(it), |b| {
        for t in 0..NKB / 2 {
            let w: Vec<ValueId> = [(2 * t, 0usize), (2 * t, 2), (2 * t + 1, 0), (2 * t + 1, 2)].iter().map(|&(blk, r)| b.cvt_pk_bf16_f32(p[4 * blk + r], p[4 * blk + r + 1])).collect();
            let pf = b.pack4(w[0], w[1], w[2], w[3]);
            for db in 0..8i32 {
                let a0 = b.bind_i32(ev(vb) + (32 * t as i32) * ROW + 32 * db);
                let a1 = b.bind_i32(ev(vb) + (32 * t as i32 + 16) * ROW + 32 * db);
                let vf = b.load_lds_tr_b16x2(a0, a1);
                b.var_mfma(ov[db as usize], kind, vf, pf, false);
            }
        }
    });
    let it1 = b.bind_i32(itx.clone() + 1);
    b.set_var(it, it1);
    let done = b.bind_cond(ev(it1).ge(ev(ntile)));
    b.break_if(done);
    b.end_loop();
    // ---- the split's partial: O / l (fp32, 4 consecutive d per lane and block) and LSE ----
    let lv = l_v_val(&mut b, l_v);
    let l_tot = group4(&mut b, lv, false);
    let lbits = b.bitcast_i32(l_tot);
    let pos = b.bind_cond(ev(lbits).gt(0));
    let inv0 = b.rcp_f32(l_tot);
    let inv = b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(pos), a: Opnd::Val(inv0), b: Opnd::Val(zf) });
    let inv = match vsc { Some(vs) => b.mul_f32(inv, vs), None => inv };
    let row_po = b.bind_i32(((ev(bat) * ev(hq) + ev(head)) * ev(nsplit) + ev(ns)) * (4 * D));
    let big = b.c_i32(0x7fff_fff0);
    let po_vo_r = b.bind_i32(ev(row_po) + (ev(grp) << 4));
    let po_vo = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(row_live), a: Opnd::Val(po_vo_r), b: Opnd::Val(big) });
    let r_po = { let n = b.bind_i32((ev(bat) + 1) * ev(hq) * ev(nsplit) * (4 * D)); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_po, n) };
    for db in 0..8usize {
        let v: Vec<ValueId> = (0..4u8)
            .map(|k| {
                let x = b.frag4_lane(Opnd::Var(ov[db]), k);
                let x = b.bitcast_f32(x);
                let y = b.mul_f32(x, inv);
                b.bitcast_i32(y)
            })
            .collect();
        let f4 = b.frag4(v[0], v[1], v[2], v[3]);
        b.buffer_store_b128(r_po, po_vo, c0, (16 * db as u32) * 4, f4);
    }
    let lg2 = b.log2_f32(l_tot);
    let x = b.add_f32(Opnd::Var(m_v), lg2);
    let ln2 = b.c_f32(std::f32::consts::LN_2);
    let lse0 = b.mul_f32(x, ln2);
    let lse = b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(pos), a: Opnd::Val(lse0), b: Opnd::Val(neg_inf) });
    let r_l = { let n = b.bind_i32((ev(bat) + 1) * ev(hq) * ev(nsplit) * 4); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_lse, n) };
    let l_vo_r = b.bind_i32(((ev(bat) * ev(hq) + ev(head)) * ev(nsplit) + ev(ns)) << 2);
    let l_vo = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(row_live), a: Opnd::Val(l_vo_r), b: Opnd::Val(big) });
    b.buffer_store_f32(r_l, l_vo, c0, 0, lse);
    b.else_();
    {
        let r_l = { let n = b.bind_i32((ev(bat) + 1) * ev(hq) * ev(nsplit) * 4); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_lse, n) };
        let l_vo_r = b.bind_i32(((ev(bat) * ev(hq) + ev(head)) * ev(nsplit) + ev(ns)) << 2);
        let big = b.c_i32(0x7fff_fff0);
        let l_vo = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(row_live), a: Opnd::Val(l_vo_r), b: Opnd::Val(big) });
        b.buffer_store_f32(r_l, l_vo, c0, 0, neg_inf);
    }
    b.end_if();
    b.finish()
}

/// The f32 value of a loop variable (after the loop).
fn l_v_val(b: &mut FnBuilder, v: VarId) -> ValueId {
    let z = b.c_f32(0.0);
    b.add_f32(Opnd::Var(v), z)
}