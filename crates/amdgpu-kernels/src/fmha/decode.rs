//! Paged-KV decode attention, bf16, D = 128 (batch 3 of docs/op-coverage-design.md): `pa_decode_bf16_hd128_p<P>` + the split combine
//! `attn_combine_bf16_hd128`.
//!
//! * KV cache in the flash / NHD paged layout K, V [num_pages, P, HK, D] bf16 (page size P = 16 / 64 / 256 / ...: a power of two), block
//!   table int32 [B, max_pages], context lengths int32 [B]; Q [B, H, D] (one query per sequence), GQA G = H / HK <= 32.
//! * Grid (HK, B, S), 4 waves: the queries of one KV head (the G heads) are the 32 "rows" of a wave (S^T / O^T orientation of
//!   `bf16::tile_math`: one query per lane); every wave is its own key split (split index 4 z + wave of NS = 4 S): the keys of a sequence are
//!   cut into NS chunks of a multiple of 64; each wave streams its chunk tile by tile through a private LDS region (K + V, 32 KB), the DMA
//!   translating every key row through the block table.
//! * Output per split: O / l (fp32 [B, H, NS, D]) and LSE = ln(sum exp) (fp32 [B, H, NS]; -inf for an empty split); the combine kernel merges
//!   the NS splits into bf16 O [B, H, D].
//!
//! ABI (16-byte slots): 0 ptr_po, 1 ptr_q, 2 ptr_k, 3 ptr_v, 4 ptr_bt, 5 ptr_cl, 6 scale (f32), 7 H, 8 HK, 9 max_pages, 10 num_pages, 11 S,
//! 12 ptr_plse.
use super::bf16::{tile_math, Tm};
use amdgpu_dsl::{CompileProfile, DType, Ext, FnBuilder, Function, MmaKind, MmaShape, MmaTy, Op, Opnd, ValueId, VarId, I32};
use amdgpu_stage::templates::grid;
use amdgpu_tile::{ev, LdsPlan, Lane};
use amdgpu_stage::templates::split::{lse_combine, LseCombine};

const D: i32 = 128;
const ROW: i32 = 2 * D;
const KT: i32 = 64;

pub fn pa_decode_bf16_hd128(name: &str, page: i32) -> Function {
    assert!(page.count_ones() == 1, "page size: a power of two");
    let mut b = FnBuilder::new(name, 256);
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
    b.sym_range("G", 1, 32);
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
    let half = b.bind_i32(lane.half());
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
        b.assume_trusted(q, Ext::c(1), Ext::c(32), "GQA group G = H / HK in [1, 32] (contract H = HK G)");
        q
    };
    let j = b.bind_i32(lane.j());
    let row_live = b.bind_cond(ev(j).lt(ev(gsz)));
    let head = b.bind_i32(ev(hk) * ev(gsz) + ev(j).min(ev(gsz) - 1));
    // ---- the wave's key split ----
    let (nsplit, ns) = grid::wave_split_index(&mut b, z, wv, s_n);
    let ncl = { let v = b.bind_i32((ev(bat) + 1) << 2); b.read_first_lane(v) };
    let r_cl = b.buffer_desc_dyn(p_cl, ncl);
    let cl_so = { let v = b.bind_i32(ev(bat) << 2); b.read_first_lane(v) };
    let zero = b.c_i32(0);
    let ctx = { let v = b.buffer_load_f32(r_cl, zero, cl_so, 0); let v = b.bitcast_i32(v); b.read_first_lane(v) };
    b.assume_trusted(ctx, Ext::c(0), Ext::ka(144) * Ext::c(page as i64), "decode: a context length lies in [0, max_pages P] (host contract)");
    // chunk = ceil(tiles / NS) tiles (NS = 4 S, S a power of two <= 16); [k_lo, k_hi) = [ns chunk, min(ctx, (ns + 1) chunk))
    let lg = grid::split_lg(&mut b, s_n, 4);
    let (k_lo, k_hi) = grid::key_chunk(&mut b, ctx, ns, nsplit, lg, 6);
    let ntile = { let v = b.bind_i32((ev(k_hi) - ev(k_lo) + (KT - 1)).max(0).lshr(6)); b.read_first_lane(v) };

    // ---- descriptors ----
    let r_q = { let n = b.bind_i32((ev(bat) + 1) * ev(hq) * ROW); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_q, n) };
    let page_bytes = page * ROW;
    let r_k = { let n = b.bind_i32(ev(npages) * ev(hkv_n) * page_bytes); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_k, n) };
    let r_v = { let n = b.bind_i32(ev(npages) * ev(hkv_n) * page_bytes); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_v, n) };
    let r_bt = { let n = b.bind_i32((ev(bat) + 1) * ev(maxp) << 2); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_bt, n) };
    let bt_so = { let v = b.bind_i32(ev(bat) * ev(maxp) << 2); b.read_first_lane(v) };
    let maxk = b.bind_i32((ev(maxp) * page) - 1);

    // ---- Q^T fragments (lane: head row, 16 bytes at d 16 c + 8 h) ----
    let q_vo = b.bind_i32((ev(bat) * ev(hq) + ev(head)) * ROW + (ev(half) << 4));
    let c0 = b.c_i32(0);
    let qf: Vec<ValueId> = (0..D / 16).map(|c| b.buffer_load_b128(r_q, q_vo, c0, (32 * c) as u32)).collect();

    // ---- private LDS: per wave K (16 KB) + V (16 KB) ----
    let mut lds = LdsPlan::new();
    let kv = lds.private("KV", 2 * (KT * ROW) as i64, 4, &[]);
    lds.commit(&mut b);
    let me = kv.mine(wv);
    let mbase = me.base();
    // DMA of the wave's tile: 16 KB per tensor = 16 issues of 1 KB, issue i: keys 4 i + lane / 16, chunk lane % 16 (K source chunk XOR key % 16)
    let l16 = b.bind_i32(lane.lane() & 15);
    let kk_of = |i: i32| (lane.lane().lshr(4)) + 4 * i;
    // per-lane LDS read offsets (as the prefill kernel, relative to the region)
    let kr: Vec<ValueId> = (0..D / 16).map(|c| b.bind_i32(lane.j() * ROW + (((ev(half) + 2 * c) ^ (lane.lane() & 15)) << 4))).collect();
    let vr: Vec<ValueId> = vec![b.bind_i32((ev(half) * 4 + (lane.lane() & 15).lshr(2)) * ROW + ((((lane.lane() & 31).lshr(4) << 4) + ((lane.lane() & 3) << 2)) << 1))];
    let c_l2e = b.c_f32(std::f32::consts::LOG2_E);
    let csc = b.mul_f32(scale, c_l2e);
    let neg_inf = b.c_f32(f32::NEG_INFINITY);
    let kind = MmaKind::new(MmaShape::M32N32K16, MmaTy::Bf16);
    let tm = Tm { d: D, kind, csc, neg_inf, kr, vr, qf, nsb: 2 };
    let zero16 = b.frag_zero(DType::Frag16);
    let m_init = b.c_f32(-1.0e30);
    let l_init = b.c_f32(0.0);
    let mut inits: Vec<(DType, Opnd)> = vec![(DType::I32, Opnd::I32(0)), (DType::F32, Opnd::Val(m_init)), (DType::F32, Opnd::Val(l_init))];
    inits.extend((0..6).map(|_| (DType::Frag16, Opnd::Val(zero16))));
    let any = b.bind_cond(ev(ntile).gt(0));
    b.begin_if(Opnd::Val(any));
    let vars = b.begin_loop(&inits);
    let (it, m_v, l_v) = (vars[0], vars[1], vars[2]);
    let ov: Vec<VarId> = vars[3..7].to_vec();
    let s_vars: Vec<VarId> = vars[7..9].to_vec();
    let itx = amdgpu_dsl::varx(it);
    let key_t = { let v = b.bind_i32(ev(k_lo) + (itx.clone() << 6)); b.read_first_lane(v) };
    // ---- DMA the tile's K and V rows through the block table ----
    b.wait_lds(0);
    // the 16 block-table lookups first (one round trip), then the 32 DMAs
    let lookups: Vec<(ValueId, ValueId)> = (0..16)
        .map(|i| {
            let key = b.bind_i32(ev(key_t) + kk_of(i));
            // the page of the (clamped) key: rows past the context read a valid page, masked
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
            // M0: the wave's private slice (wave-uniform: the slice key is the readfirstlane'd wave index)
            let m0k = b.bind_i32(mbase.clone() + (i << 10));
            let m0v = b.bind_i32(mbase.clone() + (KT * ROW + (i << 10)));
            b.buffer_load_lds_b128(r_k, kvo, c0, m0k, 0);
            b.buffer_load_lds_b128(r_v, vvo, c0, m0v, 0);
        }
    });
    b.wait_vm(0);
    let key0 = b.bind_i32(ev(key_t) + (ev(half) << 2));
    let keep = |b: &mut FnBuilder, off: i32| b.bind_cond((ev(key0) + off).lt(ev(k_hi)));
    let kbase = mbase.clone();
    let vbase = mbase.clone() + KT * ROW;
    tile_math(&mut b, &tm, kbase, vbase, Some((None, &keep)), m_v, l_v, &s_vars, &ov, &|b, f| me.read_back(b, Opnd::Var(it), |b| f(b)), &|b, f| me.read_back(b, Opnd::Var(it), |b| f(b)));
    let it1 = b.bind_i32(itx.clone() + 1);
    b.set_var(it, it1);
    let done = b.bind_cond(ev(it1).ge(ev(ntile)));
    b.break_if(done);
    b.end_loop();
    // ---- the split's partial: O / l (fp32) and LSE (ln); l = 0 (no visible key) -> O = 0, LSE = -inf ----
    let (la, lb) = b.permlane32_swap(Opnd::Var(l_v), Opnd::Var(l_v));
    let (la, lb) = (b.bitcast_f32(la), b.bitcast_f32(lb));
    let l_tot = b.add_f32(la, lb);
    let zf = b.c_f32(0.0);
    let lbits = b.bitcast_i32(l_tot);
    let pos = b.bind_cond(ev(lbits).gt(0));
    let inv0 = b.rcp_f32(l_tot);
    let inv = b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(pos), a: Opnd::Val(inv0), b: Opnd::Val(zf) });
    let row_po = b.bind_i32(((ev(bat) * ev(hq) + ev(head)) * ev(nsplit) + ev(ns)) * (4 * D));
    let big = b.c_i32(0x7fff_fff0);
    let po_vo_r = b.bind_i32(ev(row_po) + (ev(half) << 4));
    let po_vo = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(row_live), a: Opnd::Val(po_vo_r), b: Opnd::Val(big) });
    let r_po = { let n = b.bind_i32((ev(bat) + 1) * ev(hq) * ev(nsplit) * (4 * D)); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_po, n) };
    for e in 0..4usize {
        for u in 0..4u8 {
            let v: Vec<ValueId> = (0..4u8)
                .map(|k| {
                    let x = b.frag16_lane(Opnd::Var(ov[e]), 4 * u + k);
                    let x = b.bitcast_f32(x);
                    let y = b.mul_f32(x, inv);
                    b.bitcast_i32(y)
                })
                .collect();
            let f4 = b.frag4(v[0], v[1], v[2], v[3]);
            b.buffer_store_b128(r_po, po_vo, c0, ((32 * e as u32) + 8 * u as u32) * 4, f4);
        }
    }
    let lg = b.log2_f32(l_tot);
    let x = b.add_f32(Opnd::Var(m_v), lg);
    let ln2 = b.c_f32(std::f32::consts::LN_2);
    let lse0 = b.mul_f32(x, ln2);
    let lse = b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(pos), a: Opnd::Val(lse0), b: Opnd::Val(neg_inf) });
    let r_l = { let n = b.bind_i32((ev(bat) + 1) * ev(hq) * ev(nsplit) * 4); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_lse, n) };
    let l_vo_r = b.bind_i32(((ev(bat) * ev(hq) + ev(head)) * ev(nsplit) + ev(ns)) << 2);
    let l_vo = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(row_live), a: Opnd::Val(l_vo_r), b: Opnd::Val(big) });
    b.buffer_store_f32(r_l, l_vo, c0, 0, lse);
    b.else_();
    // an empty split: LSE = -inf, O = 0
    {
        let r_l = { let n = b.bind_i32((ev(bat) + 1) * ev(hq) * ev(nsplit) * 4); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_lse, n) };
        let l_vo_r = b.bind_i32(((ev(bat) * ev(hq) + ev(head)) * ev(nsplit) + ev(ns)) << 2);
        let big = b.c_i32(0x7fff_fff0);
        let l_vo = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(row_live), a: Opnd::Val(l_vo_r), b: Opnd::Val(big) });
        b.buffer_store_f32(r_l, l_vo, c0, 0, neg_inf);
    }
    b.end_if();
    let _ = I32::c(0);
    b.finish()
}

/// Merges NS = 4 S splits: O[b, h, d] = sum_s exp(lse_s - M) po_s[d] / sum_s exp(lse_s - M) (bf16), M = max_s lse_s; 0 when every split is
/// empty. Grid (H, B), 128 threads (one per d). ABI: 0 ptr_o (bf16 [B, H, D]), 1 ptr_po, 2 ptr_plse, 3 H, 4 NS.
pub fn attn_combine_bf16_hd128(name: &str) -> Function {
    attn_combine_bf16(name, D)
}

/// `attn_combine_bf16_hd128` for head dim `dd` (dd threads, <= 1024).
#[allow(non_snake_case)]
pub fn attn_combine_bf16(name: &str, dd: i32) -> Function {
    attn_combine_bf16_x(name, dd, false)
}

/// `split_major`: partials indexed [B][NS][H] (aiter's MLA stage-1 splitData / splitLse) instead of [B][H][NS].
pub fn attn_combine_bf16_x(name: &str, dd: i32, split_major: bool) -> Function {
    attn_combine_bf16_xs(name, dd, split_major, false)
}

/// `sink` (`attn_combine_bf16_sink_hd128`, gpt-oss attention sinks): a per-head logit sink[h] (f32 [H], ptr @80) joins the softmax
/// denominator only: O = sum_s e^(lse_s - M) po_s / (sum_s e^(lse_s - M) + e^(sink_h - M)), M = max(max_s lse_s, sink_h).
pub fn attn_combine_bf16_xs(name: &str, dd: i32, split_major: bool, sink: bool) -> Function {
    lse_combine(name, &LseCombine { dd, split_major, sink })
}
