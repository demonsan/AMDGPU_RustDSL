//! Paged-KV decode with 16x16x32 bf16 MFMAs (perf pass, GQA G <= 16): `pa_decode16_bf16_hd128_p<P>`, the ABI / layouts / splits of
//! `decode.rs` (and its `attn_combine_bf16_hd128`), but the S^T / O^T tiles are 16 x 16: the 16 columns are the G heads of the KV head (the
//! 32-row form wastes 3/4 of the MFMA work for G = 8). A lane holds head l % 16 and, per 16-key (16-dim) block, the 4 keys (dims) 4 g + 0..3
//! (g = l / 16); the softmax statistics are combined over the 4 lane groups (permlane16 + permlane32 swaps). P^T feeds the 32-key PV step in the
//! key order {4 g + 0..3 of block 2 t, 16 + 4 g + 0..3 of block 2 t + 1}; V^T by ds_read_b64_tr_b16 with rows 16 r + 4 g + s / 4.
use amdgpu_dsl::{CompileProfile, DType, Ext, FnBuilder, Function, MmaKind, MmaShape, MmaTy, Op, Opnd, ValueId, VarId};
use amdgpu_tile::{ev, LdsPlan, Lane};

const D: i32 = 128;
const ROW: i32 = 2 * D;
const KT: i32 = 64;

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

pub fn pa_decode16_bf16_hd128(name: &str, page: i32) -> Function {    let mut b = FnBuilder::new(name, 256);
    b.compile_profile(CompileProfile::Gemm);
    for (n, i) in [("ptr_po", 0u32), ("ptr_q", 1), ("ptr_k", 2), ("ptr_v", 3), ("ptr_bt", 4), ("ptr_cl", 5), ("ptr_plse", 12)] {
        b.arg(n, DType::Ptr, 16 * i, 8);
    }
    b.arg("scale", DType::F32, 96, 4);
    for (n, i) in [("H", 7u32), ("HK", 8), ("max_pages", 9), ("num_pages", 10), ("S", 11)] {
        b.arg(n, DType::I32, 16 * i, 4);
    }
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
    let cache = np * Ext::c(page as i64) * hk * Ext::c(ROW as i64);
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
    let ntt = b.bind_i32((ev(ctx) + (KT - 1)).lshr(6));
    let per = b.bind_i32(((ev(ntt) + ev(nsplit) - 1).lshr(ev(lg))) << 6);
    let k_lo = { let v = b.bind_i32(ev(ns) * ev(per)); b.read_first_lane(v) };
    let k_hi = { let v = b.bind_i32((ev(k_lo) + ev(per)).min(ev(ctx))); b.read_first_lane(v) };
    let ntile = { let v = b.bind_i32((ev(k_hi) - ev(k_lo) + (KT - 1)).max(0).lshr(6)); b.read_first_lane(v) };

    // ---- descriptors ----
    let r_q = { let n = b.bind_i32((ev(bat) + 1) * ev(hq) * ROW); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_q, n) };
    let page_bytes = page * ROW;
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
    let kv = lds.private("KV", 2 * (KT * ROW) as i64, 4, &[]);
    lds.commit(&mut b);
    let me = kv.mine(wv);
    let mbase = me.base();
    let l16 = b.bind_i32(lane.lane() & 15);
    let kk_of = |i: i32| (lane.lane().lshr(4)) + 4 * i;
    // K reads (A: lane row = key l % 16 of a 16-key block, k = d 32 c + 8 g): chunk (4 c + g) ^ (key % 16)
    let kr: Vec<ValueId> = (0..4).map(|c| b.bind_i32((ev(l16)) * ROW + (((ev(grp) + 4 * c) ^ ev(l16)) << 4))).collect();
    // V^T reads (A: lane row = d 16 db + l % 16 ... by tr_b16: lane s = l % 16 at key row 16 r + 4 g + s / 4, dims 16 db + 4 (s % 4))
    let vr = b.bind_i32((ev(grp) * 4 + ev(l16).lshr(2)) * ROW + ((ev(l16) & 3) << 3));
    let c_l2e = b.c_f32(std::f32::consts::LOG2_E);
    let csc = b.mul_f32(scale, c_l2e);
    let neg_inf = b.c_f32(f32::NEG_INFINITY);
    let kind = MmaKind::new(MmaShape::M16N16K32, MmaTy::Bf16);
    let zero4 = b.frag_zero(DType::Frag4);
    let m_init = b.c_f32(-1.0e30);
    let l_init = b.c_f32(0.0);
    // loop state: it, m, l, O^T (8 blocks of 16 d), S^T (4 blocks of 16 keys)
    let mut inits: Vec<(DType, Opnd)> = vec![(DType::I32, Opnd::I32(0)), (DType::F32, Opnd::Val(m_init)), (DType::F32, Opnd::Val(l_init))];
    inits.extend((0..12).map(|_| (DType::Frag4, Opnd::Val(zero4))));
    let any = b.bind_cond(ev(ntile).gt(0));
    b.begin_if(Opnd::Val(any));
    let vars = b.begin_loop(&inits);
    let (it, m_v, l_v) = (vars[0], vars[1], vars[2]);
    let ov: Vec<VarId> = vars[3..11].to_vec();
    let sv_: Vec<VarId> = vars[11..15].to_vec();
    let itx = amdgpu_dsl::varx(it);
    let key_t = { let v = b.bind_i32(ev(k_lo) + (itx.clone() << 6)); b.read_first_lane(v) };
    b.wait_lds(0);
    let lookups: Vec<(ValueId, ValueId)> = (0..16)
        .map(|i| {
            let key = b.bind_i32(ev(key_t) + kk_of(i));
            let kc = b.bind_i32(ev(key).min(ev(k_hi) - 1).max(0).min(ev(maxk)));
            let bvo = b.bind_i32(ev(kc).lshr(page.trailing_zeros() as i32) << 2);
            (kc, b.buffer_load_f32(r_bt, bvo, bt_so, 0))
        })
        .collect();
    me.write(&mut b, Opnd::Var(it), |b| {
        for (i, &(kc, pv)) in lookups.iter().enumerate() {
            let i = i as i32;
            let pg = b.bitcast_i32(pv);
            let pg = b.bind_i32(ev(pg).max(0).min(ev(npages) - 1));
            let rowoff = b.bind_i32(((ev(pg) * page + (ev(kc) & (page - 1))) * ev(hkv_n) + ev(hk)) * ROW);
            let kvo = b.bind_i32(ev(rowoff) + ((ev(l16) ^ (ev(kc) & 15)) << 4));
            let vvo = b.bind_i32(ev(rowoff) + (ev(l16) << 4));
            let m0k = b.bind_i32(mbase.clone() + (i << 10));
            let m0v = b.bind_i32(mbase.clone() + (KT * ROW + (i << 10)));
            b.buffer_load_lds_b128(r_k, kvo, c0, m0k, 0);
            b.buffer_load_lds_b128(r_v, vvo, c0, m0v, 0);
        }
    });
    b.wait_vm(0);
    // ---- S^T: 4 blocks of 16 keys x 4 d chunks (block inner: independent chains) ----
    let kb = b.bind_i32(mbase.clone());
    me.read_back(&mut b, Opnd::Var(it), |b| {
        for c in 0..4usize {
            for blk in 0..4i32 {
                let a = b.bind_i32(ev(kr[c]) + ev(kb) + blk * 16 * ROW);
                let kf = b.load_lds_b128(a);
                b.var_mfma(sv_[blk as usize], kind, kf, qf[c], c == 0);
            }
        }
    });
    // ---- softmax: values key_t + 16 blk + 4 g + r ----
    let key0 = b.bind_i32(ev(key_t) + (ev(grp) << 2));
    let mut s: Vec<ValueId> = Vec::with_capacity(16);
    for blk in 0..4usize {
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
    // ---- O^T += V^T P^T: 2 steps of 32 keys x 8 blocks of 16 d ----
    let vb = b.bind_i32(mbase.clone() + KT * ROW + ev(vr));
    me.read_back(&mut b, Opnd::Var(it), |b| {
        for t in 0..2usize {
            let w: Vec<ValueId> = [(2 * t, 0usize), (2 * t, 2), (2 * t + 1, 0), (2 * t + 1, 2)].iter().map(|&(blk, r)| b.cvt_pk_bf16_f32(p[4 * blk + r], p[4 * blk + r + 1])).collect();
            let pf = b.frag4(w[0], w[1], w[2], w[3]);
            for db in 0..8i32 {
                let mut words: Vec<ValueId> = Vec::with_capacity(4);
                for r in 0..2i32 {
                    let a = b.bind_i32(ev(vb) + (32 * t as i32 + 16 * r) * ROW + 32 * db);
                    let x = b.load_lds_tr_b16(a);
                    let w0 = b.frag2_lane(x, 0);
                    let w1 = b.frag2_lane(x, 1);
                    words.push(b.bitcast_i32(w0));
                    words.push(b.bitcast_i32(w1));
                }
                let vf = b.frag4(words[0], words[1], words[2], words[3]);
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