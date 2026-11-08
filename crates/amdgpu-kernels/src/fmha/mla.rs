//! MLA decode (DeepSeek latent attention), bf16, with 16x16x32 MFMAs: `mla_decode_bf16_p<P>` (the 32-row first version is archived as tag
//! archive/mla-decode-v1). One latent KV head: cache rows of 576 bf16 (512 latent + 64 rope), K = the row, V = its first 512 dims; ABI
//! (0 ptr_po, 1 ptr_q, 2 ptr_kv [num_pages, P, 576], 3 ptr_bt, 4 ptr_cl, 5 scale, 6 H, 7 max_pages, 8 num_pages, 9 S, 10 ptr_plse), but:
//! * a CTA owns 16 heads (the 16 columns of S^T / O^T: no wasted rows at H = 16) and every wave its own key range: grid (ceil(H / 16), B, S),
//!   NS = 4 S partials per (b, h) ([B, H, NS, 512] fp32 + LSE [B, H, NS], split-minor: `attn_combine_bf16_hd512` with NS);
//! * each wave runs the whole 576-dim QK and 512-dim PV of its 32-key tiles out of a private 36 KB LDS region (the tile's rows, chunk j of
//!   key k at chunk (j & !7) + ((j ^ k) & 7)): no cross-wave S reduction, no barriers;
//! * the DMA writes the tile linearly (chunk c = 72 key + position); the row base of key k comes from lane k (one block-table load per lane
//!   and tile, `ds_bpermute` per issue).
use amdgpu_dsl::{CompileProfile, DType, Ext, FnBuilder, Function, MmaKind, MmaShape, MmaTy, Op, Opnd, ValueId, VarId};
use amdgpu_tile::{ev, LdsPlan, Lane, Layout as TLayout, SlotIdx};

const DK: i32 = 576;
const DV: i32 = 512;
const ROW: i32 = 2 * DK;
const KT: i32 = 32;
const CHUNKS: i32 = ROW / 16; // 72
/// QK d chunks of 32 dims
const QC: usize = (DK / 32) as usize; // 18
/// PV d blocks of 16 dims
const DB: usize = (DV / 16) as usize; // 32

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

pub fn mla_decode_bf16(name: &str, page: i32) -> Function {
    assert!(page.count_ones() == 1, "page size: a power of two");
    let mut b = FnBuilder::new(name, 256);
    b.compile_profile(CompileProfile::Gemm);
    for (n, i) in [("ptr_po", 0u32), ("ptr_q", 1), ("ptr_kv", 2), ("ptr_bt", 3), ("ptr_cl", 4), ("ptr_plse", 10)] {
        b.arg(n, DType::Ptr, 16 * i, 8);
    }
    b.arg("scale", DType::F32, 80, 4);
    for (n, i) in [("H", 6u32), ("max_pages", 7), ("num_pages", 8), ("S", 9)] {
        b.arg(n, DType::I32, 16 * i, 4);
    }
    let (hh, mp, np, ss) = (Ext::ka(96), Ext::ka(112), Ext::ka(128), Ext::ka(144));
    b.sym_range("B", 1, 1 << 16);
    b.ka_range(96, 1, 256);
    b.ka_range(112, 1, 1 << 20);
    b.ka_range(128, 1, 1 << 20);
    b.ka_range(144, 1, 64);
    b.grid_is(0, hh.clone().ceil_div(16));
    b.grid_is(1, Ext::sym("B"));
    b.grid_is(2, ss.clone());
    let bb = Ext::sym("B");
    b.arg_tensor("ptr_q", bb.clone() * hh.clone() * Ext::c(ROW as i64));
    b.arg_tensor("ptr_kv", np * Ext::c((page * ROW) as i64));
    b.arg_tensor("ptr_bt", bb.clone() * mp * Ext::c(4));
    b.arg_tensor("ptr_cl", bb.clone() * Ext::c(4));
    b.arg_tensor("ptr_po", bb.clone() * hh.clone() * ss.clone() * Ext::c(4 * 4 * DV as i64));
    b.arg_tensor("ptr_plse", bb * hh * ss * Ext::c(16));

    let kp = |b: &mut FnBuilder, i: u32| b.let_(Op::KernargPtr { offset: 16 * i });
    let ki = |b: &mut FnBuilder, i: u32| b.let_(Op::KernargI32 { offset: 16 * i });
    let (p_po, p_q, p_kv, p_bt, p_cl, p_lse) = (kp(&mut b, 0), kp(&mut b, 1), kp(&mut b, 2), kp(&mut b, 3), kp(&mut b, 4), kp(&mut b, 10));
    let scale = b.let_(Op::KernargF32 { offset: 80 });
    let (hq, maxp, npages, s_n) = (ki(&mut b, 6), ki(&mut b, 7), ki(&mut b, 8), ki(&mut b, 9));
    let hg = b.let_(Op::WorkgroupIdX);
    let bat = b.let_(Op::WorkgroupIdY);
    let z = b.let_(Op::WorkgroupIdZ);
    let lane = Lane::new(&mut b);
    let wv = { let v = b.bind_i32(lane.wave()); b.read_first_lane(v) };
    let c0 = b.c_i32(0);
    let l16 = b.bind_i32(lane.lane() & 15);
    let grp = b.bind_i32(lane.lane().lshr(4));
    let head_r = b.bind_i32((ev(hg) << 4) + ev(l16));
    let row_live = b.bind_cond(ev(head_r).lt(ev(hq)));
    let head = b.bind_i32(ev(head_r).min(ev(hq) - 1));
    // context length; this wave's key part [k_lo, k_hi) of NS = 4 S parts (S a power of two <= 64)
    let ncl = { let v = b.bind_i32((ev(bat) + 1) << 2); b.read_first_lane(v) };
    let r_cl = b.buffer_desc_dyn(p_cl, ncl);
    let cl_so = { let v = b.bind_i32(ev(bat) << 2); b.read_first_lane(v) };
    let ctx = { let v = b.buffer_load_f32(r_cl, c0, cl_so, 0); let v = b.bitcast_i32(v); b.read_first_lane(v) };
    b.assume_trusted(ctx, Ext::c(0), Ext::ka(112) * Ext::c(page as i64), "MLA decode: a context length lies in [0, max_pages P] (host contract)");
    let lg = {
        let sv = ev(s_n);
        let mut acc = amdgpu_dsl::I32::c(0);
        for (thr, val) in [(1, 1), (2, 2), (4, 3), (8, 4), (16, 5), (32, 6)] {
            acc = sv.clone().gt(thr).select_i32(val, acc);
        }
        let v = b.bind_i32(acc + 2);
        b.read_first_lane(v)
    };
    let nsplit = b.bind_i32(ev(s_n) << 2);
    let ns = { let v = b.bind_i32((ev(z) << 2) + ev(wv)); b.read_first_lane(v) };
    let ntt = b.bind_i32((ev(ctx) + (KT - 1)).lshr(5));
    let per = b.bind_i32(((ev(ntt) + ev(nsplit) - 1).lshr(ev(lg))) << 5);
    let k_lo = { let v = b.bind_i32(ev(ns) * ev(per)); b.read_first_lane(v) };
    let k_hi = { let v = b.bind_i32((ev(k_lo) + ev(per)).min(ev(ctx))); b.read_first_lane(v) };

    // ---- descriptors ----
    let r_q = { let n = b.bind_i32((ev(bat) + 1) * ev(hq) * ROW); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_q, n) };
    let r_kv = { let n = b.bind_i32(ev(npages) * (page * ROW)); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_kv, n) };
    let r_bt = { let n = b.bind_i32((ev(bat) + 1) * ev(maxp) << 2); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_bt, n) };
    let bt_so = { let v = b.bind_i32(ev(bat) * ev(maxp) << 2); b.read_first_lane(v) };
    let maxk = b.bind_i32(ev(maxp) * page - 1);

    // ---- Q^T (B operand of 16x16x32: lane = head column, k = dims 32 c + 8 g): 18 x 16 bytes ----
    let q_vo = b.bind_i32((ev(bat) * ev(hq) + ev(head)) * ROW + (ev(grp) << 4));
    let qf: Vec<ValueId> = (0..QC).map(|c| b.buffer_load_b128(r_q, q_vo, c0, (64 * c) as u32)).collect();

    // ---- private LDS: per wave one 32-key tile (36 KB) ----
    let mut lds = LdsPlan::new();
    // the tile as two 16-key halves in separate private regions: QK of half 0 runs while half 1 lands
    let kv0 = lds.private("KV0", (16 * ROW) as i64, 4, &[]);
    let kv1 = lds.private("KV1", (16 * ROW) as i64, 4, &[]);
    lds.commit(&mut b);
    let mes = [kv0.mine(wv), kv1.mine(wv)];
    let mbases = [mes[0].base(), mes[1].base()];
    for c in 0..2304i64 {
        assert_eq!((c * 1821) >> 17, c / 72, "c / 72 multiply-shift");
    }
    // K reads (A: lane row = key l % 16 of a 16-key block, k = dims 32 c + 8 g -> chunk 4 c + g, swizzled by key % 8)
    let kr: Vec<ValueId> = (0..QC as i32)
        .map(|c| {
            let ch = b.bind_i32(ev(grp) + 4 * c);
            b.bind_i32(ev(l16) * ROW + (((ev(ch).lshr(3) << 3) + ((ev(ch) ^ ev(l16)) & 7)) << 4))
        })
        .collect();
    // V^T reads (tr_b16: lane s = l % 16 at key row 16 r + 4 g + s / 4, dims 16 db + 4 (s % 4) -> chunk 2 db + (s % 4) / 2, byte 8 (s % 2));
    // the swizzle depends on 2 db % 8 (4 offsets) and the row % 8
    let vrow = b.bind_i32(ev(grp) * 4 + ev(l16).lshr(2));
    let c0v = b.bind_i32((ev(l16) & 3).lshr(1));
    let vr: Vec<ValueId> = (0..4)
        .map(|q| {
            let lo = b.bind_i32(((2 * q + ev(c0v)) ^ ev(vrow)) & 7);
            b.bind_i32(ev(vrow) * ROW + (ev(lo) << 4) + ((ev(l16) & 1) << 3))
        })
        .collect();
    let c_l2e = b.c_f32(std::f32::consts::LOG2_E);
    let csc = b.mul_f32(scale, c_l2e);
    let neg_inf = b.c_f32(f32::NEG_INFINITY);
    let kind = MmaKind::new(MmaShape::M16N16K32, MmaTy::Bf16);
    let zero4 = b.frag_zero(DType::Frag4);
    let m_init = b.c_f32(-1.0e30);
    let l_init = b.c_f32(0.0);
    let mut inits: Vec<(DType, Opnd)> = vec![(DType::I32, Opnd::I32(0)), (DType::F32, Opnd::Val(m_init)), (DType::F32, Opnd::Val(l_init))];
    inits.extend((0..DB + 2).map(|_| (DType::Frag4, Opnd::Val(zero4))));
    // no branch around the loop: an empty part runs one fully masked tile (P = 0, l = 0) and the epilogue writes O = 0, LSE = -inf (the
    // previous if / else form hit a backend bug: a constant register materialised only on the then path was read on the else path)
    let nloop = { let v = b.bind_i32((ev(k_hi) - ev(k_lo) + (KT - 1)).max(0).lshr(5).max(1)); b.read_first_lane(v) };
    // the block-table entry of key l % 32 of tile t (the lane's page), loaded one tile ahead (loop-carried; the load of tile t + 1 is issued
    // at the start of tile t and lands during it)
    let bt_load = |b: &mut FnBuilder, tile: ValueId| -> (ValueId, ValueId) {
        let key_t = b.bind_i32(ev(k_lo) + (ev(tile) << 5));
        let kl = b.bind_i32((ev(key_t) + (lane.lane() & 31)).min(ev(k_hi) - 1).max(0).min(ev(maxk)));
        let bvo = b.bind_i32(ev(kl).lshr(page.trailing_zeros() as i32) << 2);
        (kl, b.buffer_load_f32(r_bt, bvo, bt_so, 0))
    };
    let t0 = b.c_i32(0);
    let (_, pg0) = bt_load(&mut b, t0);
    inits.push((DType::F32, Opnd::Val(pg0)));
    let vars = b.begin_loop(&inits);
    let pg_v = vars[5 + DB];
    let (it, m_v, l_v) = (vars[0], vars[1], vars[2]);
    let ov: Vec<VarId> = vars[3..3 + DB].to_vec();
    let sv_: Vec<VarId> = vars[3 + DB..5 + DB].to_vec();
    let itx = amdgpu_dsl::varx(it);
    let key_t = { let v = b.bind_i32(ev(k_lo) + (itx.clone() << 5)); b.read_first_lane(v) };
    b.wait_lds(0);
    // the row base of key l % 32 of the tile (its page was loaded during the previous tile), shared by ds_bpermute
    let kl = b.bind_i32((ev(key_t) + (lane.lane() & 31)).min(ev(k_hi) - 1).max(0).min(ev(maxk)));
    let pg = b.bitcast_i32(Opnd::Var(pg_v));
    {
        let itn = b.bind_i32((itx.clone() + 1).min(ev(nloop) - 1));
        let (_, pn) = bt_load(&mut b, itn);
        let pn = b.inplace(pn);
        b.set_var(pg_v, pn);
    }
    let pg = b.bind_i32(ev(pg).max(0).min(ev(npages) - 1));
    let rowb = b.bind_i32((ev(pg) * page + (ev(kl) & (page - 1))) * ROW);
    // per half: 18 issues of 1 KB (chunk c = 72 key + position, key < 16), row bases by ds_bpermute from lane 16 h + key
    for (h, me) in mes.iter().enumerate() {
        let mb = mbases[h].clone();
        me.write(&mut b, Opnd::Var(it), |b| {
            for g0 in (0..18i32).step_by(9) {
                let parts: Vec<(ValueId, ValueId)> = (g0..g0 + 9)
                    .map(|i| {
                        let c = b.bind_i32((i << 6) + lane.lane());
                        let key = b.bind_i32((ev(c) * 1821).lshr(17));
                        let pos = b.bind_i32(ev(c) - ev(key) * CHUNKS);
                        let src = b.bind_i32((ev(pos).lshr(3) << 3) + ((ev(pos) ^ ev(key)) & 7));
                        let addr = b.bind_i32((ev(key) + 16 * h as i32) << 2);
                        (b.bpermute(addr, rowb), src)
                    })
                    .collect();
                for (k, (rb, src)) in parts.into_iter().enumerate() {
                    let i = g0 + k as i32;
                    let rb = b.bitcast_i32(rb);
                    let vo = b.bind_i32(ev(rb) + (ev(src) << 4));
                    let m0 = b.bind_i32(mb.clone() + (i << 10));
                    b.buffer_load_lds_b128(r_kv, vo, c0, m0, 0);
                }
            }
        });
    }
    // ---- S^T: block (half) 0 after its 18 DMAs complete (half 1 in flight), then block 1 ----
    for blk in 0..2usize {
        b.wait_vm(if blk == 0 { 18 } else { 0 });
        let kb = b.bind_i32(mbases[blk].clone());
        mes[blk].read_back(&mut b, Opnd::Var(it), |b| {
            for c in 0..QC {
                let a = b.bind_i32(ev(kr[c]) + ev(kb));
                let kf = b.load_lds_b128(a);
                b.var_mfma(sv_[blk], kind, kf, qf[c], c == 0);
            }
        });
    }
    // ---- softmax: values key_t + 16 blk + 4 g + r ----
    let key0 = b.bind_i32(ev(key_t) + (ev(grp) << 2));
    let mut s: Vec<ValueId> = Vec::with_capacity(8);
    for blk in 0..2usize {
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
    // ---- O^T += V^T P^T: one 32-key step x 32 blocks of 16 d ----
    let w: Vec<ValueId> = [(0usize, 0usize), (0, 2), (1, 0), (1, 2)].iter().map(|&(blk, r)| b.cvt_pk_bf16_f32(p[4 * blk + r], p[4 * blk + r + 1])).collect();
    let pf = b.pack4(w[0], w[1], w[2], w[3]);
    let vb0: Vec<ValueId> = vr.iter().map(|v| b.bind_i32(mbases[0].clone() + ev(*v))).collect();
    let vb1: Vec<ValueId> = vr.iter().map(|v| b.bind_i32(mbases[1].clone() + ev(*v))).collect();
    // the two halves are read inside one region each: the 32 MFMAs need both, so the reads are split per half (A halves cat'ed)
    for db in 0..DB as i32 {
        let q = (db % 4) as usize;
        let hi = ((2 * db) & !7) << 4;
        let x0 = mes[0].read_back(&mut b, Opnd::Var(it), |b| { let a = b.bind_i32(ev(vb0[q]) + hi); b.load_lds_tr_b16(a) });
        let x1 = mes[1].read_back(&mut b, Opnd::Var(it), |b| { let a = b.bind_i32(ev(vb1[q]) + hi); b.load_lds_tr_b16(a) });
        let vf = b.cat_frag4(x0, x1);
        b.var_mfma(ov[db as usize], kind, vf, pf, false);
    }
    let it1 = b.bind_i32(itx.clone() + 1);
    b.set_var(it, it1);
    let done = b.bind_cond(ev(it1).ge(ev(nloop)));
    b.break_if(done);
    b.end_loop();
    // ---- the part's partial: O / l (fp32, 4 consecutive d per lane and block) and LSE ----
    let lv = { let z0 = b.c_f32(0.0); b.add_f32(Opnd::Var(l_v), z0) };
    let l_tot = group4(&mut b, lv, false);
    let lbits = b.bitcast_i32(l_tot);
    let pos = b.bind_cond(ev(lbits).gt(0));
    let inv0 = b.rcp_f32(l_tot);
    let inv = b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(pos), a: Opnd::Val(inv0), b: Opnd::Val(zf) });
    let row_po = b.bind_i32(((ev(bat) * ev(hq) + ev(head)) * ev(nsplit) + ev(ns)) * (4 * DV));
    let big = b.c_i32(0x7fff_fff0);
    let po_vo_r = b.bind_i32(ev(row_po) + (ev(grp) << 4));
    let po_vo = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(row_live), a: Opnd::Val(po_vo_r), b: Opnd::Val(big) });
    let r_po = { let n = b.bind_i32((ev(bat) + 1) * ev(hq) * ev(nsplit) * (4 * DV)); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_po, n) };
    for db in 0..DB {
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
    b.finish()
}

/// Head-parallel MLA decode for many heads (`mla_decode_bf16_hp_p<P>`, H >= 64): the CTA's 4 waves are 4 groups of 16 heads that SHARE the
/// key tiles of the CTA's key part (a 2-slot 36 KB ring, each wave issuing 9 of the 36 1-KB DMAs; the tile t + 1 DMA overlaps tile t), so
/// the KV is read once per 64 heads instead of once per 16. Grid (ceil(H / 64), B, S), partials [B, H, S, 512] + LSE [B, H, S] (NS = S).
/// Same ABI as `mla_decode_bf16`; per-wave math as there (16x16x32, one head per lane column).
pub fn mla_decode_bf16_hp(name: &str, page: i32) -> Function {
    assert!(page.count_ones() == 1, "page size: a power of two");
    let mut b = FnBuilder::new(name, 256);
    b.compile_profile(CompileProfile::Gemm);
    for (n, i) in [("ptr_po", 0u32), ("ptr_q", 1), ("ptr_kv", 2), ("ptr_bt", 3), ("ptr_cl", 4), ("ptr_plse", 10)] {
        b.arg(n, DType::Ptr, 16 * i, 8);
    }
    b.arg("scale", DType::F32, 80, 4);
    for (n, i) in [("H", 6u32), ("max_pages", 7), ("num_pages", 8), ("S", 9)] {
        b.arg(n, DType::I32, 16 * i, 4);
    }
    let (hh, mp, np, ss) = (Ext::ka(96), Ext::ka(112), Ext::ka(128), Ext::ka(144));
    b.sym_range("B", 1, 1 << 16);
    b.ka_range(96, 1, 256);
    b.ka_range(112, 1, 1 << 20);
    b.ka_range(128, 1, 1 << 20);
    b.ka_range(144, 1, 64);
    b.grid_is(0, hh.clone().ceil_div(64));
    b.grid_is(1, Ext::sym("B"));
    b.grid_is(2, ss.clone());
    let bb = Ext::sym("B");
    b.arg_tensor("ptr_q", bb.clone() * hh.clone() * Ext::c(ROW as i64));
    b.arg_tensor("ptr_kv", np * Ext::c((page * ROW) as i64));
    b.arg_tensor("ptr_bt", bb.clone() * mp * Ext::c(4));
    b.arg_tensor("ptr_cl", bb.clone() * Ext::c(4));
    b.arg_tensor("ptr_po", bb.clone() * hh.clone() * ss.clone() * Ext::c(4 * DV as i64));
    b.arg_tensor("ptr_plse", bb * hh * ss * Ext::c(4));

    let kp = |b: &mut FnBuilder, i: u32| b.let_(Op::KernargPtr { offset: 16 * i });
    let ki = |b: &mut FnBuilder, i: u32| b.let_(Op::KernargI32 { offset: 16 * i });
    let (p_po, p_q, p_kv, p_bt, p_cl, p_lse) = (kp(&mut b, 0), kp(&mut b, 1), kp(&mut b, 2), kp(&mut b, 3), kp(&mut b, 4), kp(&mut b, 10));
    let scale = b.let_(Op::KernargF32 { offset: 80 });
    let (hq, maxp, npages, s_n) = (ki(&mut b, 6), ki(&mut b, 7), ki(&mut b, 8), ki(&mut b, 9));
    let hg = b.let_(Op::WorkgroupIdX);
    let bat = b.let_(Op::WorkgroupIdY);
    let z = b.let_(Op::WorkgroupIdZ);
    let lane = Lane::new(&mut b);
    let wv = { let v = b.bind_i32(lane.wave()); b.read_first_lane(v) };
    let c0 = b.c_i32(0);
    let l16 = b.bind_i32(lane.lane() & 15);
    let grp = b.bind_i32(lane.lane().lshr(4));
    let head_r = b.bind_i32((ev(hg) << 6) + (ev(wv) << 4) + ev(l16));
    let row_live = b.bind_cond(ev(head_r).lt(ev(hq)));
    let head = b.bind_i32(ev(head_r).min(ev(hq) - 1));
    let ncl = { let v = b.bind_i32((ev(bat) + 1) << 2); b.read_first_lane(v) };
    let r_cl = b.buffer_desc_dyn(p_cl, ncl);
    let cl_so = { let v = b.bind_i32(ev(bat) << 2); b.read_first_lane(v) };
    let ctx = { let v = b.buffer_load_f32(r_cl, c0, cl_so, 0); let v = b.bitcast_i32(v); b.read_first_lane(v) };
    b.assume_trusted(ctx, Ext::c(0), Ext::ka(112) * Ext::c(page as i64), "MLA decode: a context length lies in [0, max_pages P] (host contract)");
    let lg = {
        let sv = ev(s_n);
        let mut acc = amdgpu_dsl::I32::c(0);
        for (thr, val) in [(1, 1), (2, 2), (4, 3), (8, 4), (16, 5), (32, 6)] {
            acc = sv.clone().gt(thr).select_i32(val, acc);
        }
        let v = b.bind_i32(acc);
        b.read_first_lane(v)
    };
    let ntt = b.bind_i32((ev(ctx) + (KT - 1)).lshr(5));
    let per = b.bind_i32(((ev(ntt) + ev(s_n) - 1).lshr(ev(lg))) << 5);
    let k_lo = { let v = b.bind_i32(ev(z) * ev(per)); b.read_first_lane(v) };
    let k_hi = { let v = b.bind_i32((ev(k_lo) + ev(per)).min(ev(ctx))); b.read_first_lane(v) };
    // tiles of this part (at least one: an empty part runs a fully masked tile), two per loop pass
    let npair = { let v = b.bind_i32(((ev(k_hi) - ev(k_lo) + (KT - 1)).max(0).lshr(5).max(1) + 1).lshr(1)); b.read_first_lane(v) };

    let r_q = { let n = b.bind_i32((ev(bat) + 1) * ev(hq) * ROW); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_q, n) };
    let r_kv = { let n = b.bind_i32(ev(npages) * (page * ROW)); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_kv, n) };
    let r_bt = { let n = b.bind_i32((ev(bat) + 1) * ev(maxp) << 2); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_bt, n) };
    let bt_so = { let v = b.bind_i32(ev(bat) * ev(maxp) << 2); b.read_first_lane(v) };
    let maxk = b.bind_i32(ev(maxp) * page - 1);
    let q_vo = b.bind_i32((ev(bat) * ev(hq) + ev(head)) * ROW + (ev(grp) << 4));
    let qf: Vec<ValueId> = (0..QC).map(|c| b.buffer_load_b128(r_q, q_vo, c0, (64 * c) as u32)).collect();

    // ---- the shared ring: 2 slots of one 32-key tile ----
    let mut lds = LdsPlan::new();
    let kvs = lds.ring("KV", (KT * ROW) as i64, 2, TLayout::Linear);
    lds.commit(&mut b);
    for c in 0..2304i64 {
        assert_eq!((c * 1821) >> 17, c / 72, "c / 72 multiply-shift");
    }
    // wave w stages blocks 9 w + i (i < 9) of the tile: chunk c = 64 block + lane = 72 key + position (source chunk swizzled by key % 8)
    let dma_lane: Vec<(ValueId, ValueId)> = (0..9)
        .map(|i| {
            let c = b.bind_i32(((ev(wv) * 9 + i) << 6) + lane.lane());
            let key = b.bind_i32((ev(c) * 1821).lshr(17));
            let pos = b.bind_i32(ev(c) - ev(key) * CHUNKS);
            let src = b.bind_i32((ev(pos).lshr(3) << 3) + ((ev(pos) ^ ev(key)) & 7));
            (b.bind_i32(ev(key) << 2), src)
        })
        .collect();
    let stage = |b: &mut FnBuilder, tile: ValueId, slot: i32, gen: Opnd| {
        let key_t = { let v = b.bind_i32(ev(k_lo) + (ev(tile) << 5)); b.read_first_lane(v) };
        // the row base of key l % 32 (one block-table load per lane), shared by ds_bpermute
        let kl = b.bind_i32((ev(key_t) + (lane.lane() & 31)).min(ev(k_hi) - 1).max(0).min(ev(maxk)));
        let bvo = b.bind_i32(ev(kl).lshr(page.trailing_zeros() as i32) << 2);
        let pv = b.buffer_load_f32(r_bt, bvo, bt_so, 0);
        let pg = b.bitcast_i32(pv);
        let pg = b.bind_i32(ev(pg).max(0).min(ev(npages) - 1));
        let rowb = b.bind_i32((ev(pg) * page + (ev(kl) & (page - 1))) * ROW);
        let base = kvs.base as i32 + slot * KT * ROW;
        let parts: Vec<(ValueId, ValueId)> = dma_lane.iter().map(|(addr, src)| (b.bpermute(*addr, rowb), *src)).collect();
        kvs.at(&SlotIdx::new(slot, gen)).write(b, |b| {
            for (i, (rb, src)) in parts.into_iter().enumerate() {
                let rb = b.bitcast_i32(rb);
                let vo = b.bind_i32(ev(rb) + (ev(src) << 4));
                let m0 = { let x = b.bind_i32(((ev(wv) * 9 + i as i32) << 10) + base); b.read_first_lane(x) };
                b.buffer_load_lds_b128(r_kv, vo, c0, m0, 0);
            }
        });
    };
    let kr: Vec<ValueId> = (0..QC as i32)
        .map(|c| {
            let ch = b.bind_i32(ev(grp) + 4 * c);
            b.bind_i32(ev(l16) * ROW + (((ev(ch).lshr(3) << 3) + ((ev(ch) ^ ev(l16)) & 7)) << 4))
        })
        .collect();
    let vrow = b.bind_i32(ev(grp) * 4 + ev(l16).lshr(2));
    let c0v = b.bind_i32((ev(l16) & 3).lshr(1));
    let vr: Vec<ValueId> = (0..4)
        .map(|q| {
            let lo = b.bind_i32(((2 * q + ev(c0v)) ^ ev(vrow)) & 7);
            b.bind_i32(ev(vrow) * ROW + (ev(lo) << 4) + ((ev(l16) & 1) << 3))
        })
        .collect();
    let c_l2e = b.c_f32(std::f32::consts::LOG2_E);
    let csc = b.mul_f32(scale, c_l2e);
    let neg_inf = b.c_f32(f32::NEG_INFINITY);
    let kind = MmaKind::new(MmaShape::M16N16K32, MmaTy::Bf16);
    let zero4 = b.frag_zero(DType::Frag4);
    let m_init = b.c_f32(-1.0e30);
    let l_init = b.c_f32(0.0);
    let t0 = b.c_i32(0);
    stage(&mut b, t0, 0, Opnd::I32(0));
    let mut inits: Vec<(DType, Opnd)> = vec![(DType::I32, Opnd::I32(0)), (DType::F32, Opnd::Val(m_init)), (DType::F32, Opnd::Val(l_init))];
    inits.extend((0..DB + 2).map(|_| (DType::Frag4, Opnd::Val(zero4))));
    let vars = b.begin_loop(&inits);
    let (it, m_v, l_v) = (vars[0], vars[1], vars[2]);
    let ov: Vec<VarId> = vars[3..3 + DB].to_vec();
    let sv_: Vec<VarId> = vars[3 + DB..5 + DB].to_vec();
    let itx = amdgpu_dsl::varx(it);
    for slot in 0..2i32 {
        let tile = b.bind_i32((itx.clone() << 1) + slot);
        b.wait_vm(0);
        b.wait_lds(0);
        b.barrier();
        let last = b.bind_i32((ev(npair) << 1) - 1);
        let nxt = b.bind_i32((ev(tile) + 1).min(ev(last)));
        let gen_n = if slot == 0 { Opnd::Var(it) } else { Opnd::Val(b.bind_i32(itx.clone() + 1)) };
        stage(&mut b, nxt, 1 - slot, gen_n);
        let kbase = kvs.base as i32 + slot * KT * ROW;
        let sl = kvs.at(&SlotIdx::new(slot, Opnd::Var(it)));
        sl.read(&mut b, |b| {
            for c in 0..QC {
                for blk in 0..2i32 {
                    let a = b.bind_i32(ev(kr[c]) + kbase + blk * 16 * ROW);
                    let kf = b.load_lds_b128(a);
                    b.var_mfma(sv_[blk as usize], kind, kf, qf[c], c == 0);
                }
            }
        });
        let key_t = b.bind_i32(ev(k_lo) + (ev(tile) << 5));
        let key0 = b.bind_i32(ev(key_t) + (ev(grp) << 2));
        let mut s: Vec<ValueId> = Vec::with_capacity(8);
        for blk in 0..2usize {
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
        let w: Vec<ValueId> = [(0usize, 0usize), (0, 2), (1, 0), (1, 2)].iter().map(|&(blk, r)| b.cvt_pk_bf16_f32(p[4 * blk + r], p[4 * blk + r + 1])).collect();
        let pf = b.pack4(w[0], w[1], w[2], w[3]);
        let vbase: Vec<ValueId> = vr.iter().map(|v| b.bind_i32(ev(*v) + kbase)).collect();
        sl.read(&mut b, |b| {
            for db in 0..DB as i32 {
                let q = (db % 4) as usize;
                let hi = ((2 * db) & !7) << 4;
                let a0 = b.bind_i32(ev(vbase[q]) + hi);
                let a1 = b.bind_i32(ev(vbase[q]) + hi + 16 * ROW);
                let vf = b.load_lds_tr_b16x2(a0, a1);
                b.var_mfma(ov[db as usize], kind, vf, pf, false);
            }
        });
    }
    let it1 = b.bind_i32(itx.clone() + 1);
    b.set_var(it, it1);
    let done = b.bind_cond(ev(it1).ge(ev(npair)));
    b.break_if(done);
    b.end_loop();
    // ---- the part's partial (as mla_decode_bf16, NS = S) ----
    let zf = b.c_f32(0.0);
    let lv = { let z0 = b.c_f32(0.0); b.add_f32(Opnd::Var(l_v), z0) };
    let l_tot = group4(&mut b, lv, false);
    let lbits = b.bitcast_i32(l_tot);
    let pos = b.bind_cond(ev(lbits).gt(0));
    let inv0 = b.rcp_f32(l_tot);
    let inv = b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(pos), a: Opnd::Val(inv0), b: Opnd::Val(zf) });
    let row_po = b.bind_i32(((ev(bat) * ev(hq) + ev(head)) * ev(s_n) + ev(z)) * (4 * DV));
    let big = b.c_i32(0x7fff_fff0);
    let po_vo_r = b.bind_i32(ev(row_po) + (ev(grp) << 4));
    let po_vo = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(row_live), a: Opnd::Val(po_vo_r), b: Opnd::Val(big) });
    let r_po = { let n = b.bind_i32((ev(bat) + 1) * ev(hq) * ev(s_n) * (4 * DV)); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_po, n) };
    for db in 0..DB {
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
    let r_l = { let n = b.bind_i32((ev(bat) + 1) * ev(hq) * ev(s_n) * 4); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_lse, n) };
    let l_vo_r = b.bind_i32(((ev(bat) * ev(hq) + ev(head)) * ev(s_n) + ev(z)) << 2);
    let l_vo = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(row_live), a: Opnd::Val(l_vo_r), b: Opnd::Val(big) });
    b.buffer_store_f32(r_l, l_vo, c0, 0, lse);
    b.finish()
}