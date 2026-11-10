//! Paged-KV decode with 16x16x32 bf16 MFMAs (perf pass, GQA G <= 16): `pa_decode16_bf16_hd128_p<P>`, the ABI / layouts / splits of
//! `decode.rs` (and its `attn_combine_bf16_hd128`), but the S^T / O^T tiles are 16 x 16: the 16 columns are the G heads of the KV head (the
//! 32-row form wastes 3/4 of the MFMA work for G = 8). A lane holds head l % 16 and, per 16-key (16-dim) block, the 4 keys (dims) 4 g + 0..3
//! (g = l / 16); the softmax statistics are combined over the 4 lane groups (permlane16 + permlane32 swaps). P^T feeds the 32-key PV step in the
//! key order {4 g + 0..3 of block 2 t, 16 + 4 g + 0..3 of block 2 t + 1}; V^T by ds_read_b64_tr_b16 with rows 16 r + 4 g + s / 4.
use amdgpu_dsl::{CompileProfile, DType, Ext, FnBuilder, Function, MmaKind, MmaShape, MmaTy, Op, Opnd, ValueId, VarId};
use amdgpu_tile::templates::grid;
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
/// small grids: more keys per wave). The launch dispatch (`crate::dispatch::decode16_plan`) runs the fused (`_f`) or last-arriver (`_c`) forms;
/// these split forms (with the `attn_combine_bf16_hd128` kernel) remain for experiments and the MTP / fp8 variants.
pub fn pa_decode16_bf16_hd128(name: &str, page: i32, kt: i32) -> Function {
    pa_decode16_x(name, page, kt, false)
}

/// fp8 KV in the BLOCKED layout with per-token scales (`pa_decode16_fp8tok_hd128_b<BS>`, page = BS): sched's
/// `reshape_cache_blocked_fp8_h<Hk>_d128_b<BS>` writes K [NB][HK][D / 16][BS][16], V [NB][HK][D][BS] (e4m3 bytes) and f32 scales
/// [HK][S] (S = NB BS, index hk S + slot) - k_scales @208, v_scales @224 (pointers). K rows are gathered into the same fp8 K tile as the
/// flash form; V arrives d-major (V^T) and is converted into a bf16 V^T tile ([128 d][32 keys]) read by plain 8-byte reads; the per-key
/// scales multiply S (k) and P (v, after the row sum).
pub fn pa_decode16_fp8tok(name: &str, page: i32) -> Function {
    pa_decode16_kv(name, page, 32, 2)
}

/// MTP / speculative decode (`pa_decode16_bf16_hd128_p<P>_mtp`): qlen (@240, 1..16, G qlen <= 16) new query tokens per sequence, Q
/// [B][qlen][H][D]; the 16 MFMA columns are (query position, head) pairs j = qpos G + head, query position qpos sees keys
/// < ctx - qlen + 1 + qpos (causal among the new tokens); partials / outputs per row (b qlen + qpos, h) (combine grid (H, B qlen)).
pub fn pa_decode16_mtp(name: &str, page: i32, kt: i32) -> Function {
    pa_decode16_kvm(name, page, kt, 0, true)
}

/// `_f` (fused combine, S = 1): the CTA's 4 wave partials are merged in LDS (over the dead K / V regions) and the bf16 output O [B, H, D] is
/// written at ptr_po (slot 0) - no partials, no combine kernel.
pub fn pa_decode16_fused(name: &str, page: i32, kt: i32) -> Function {
    pa_decode16_all(name, page, kt, 0, false, true)
}

/// `_c` (last-arriver combine, any S): the split form, then each CTA releases its partials (per-wave drain + L2 write-back), thread 0 takes a
/// ticket on counter[b HK + hk] (kernarg 208, u32 [B][HK], zero before the first launch; the last CTA resets it), and the last of the S CTAs
/// of a (b, hk) acquires and merges the 4 S partials of its G heads into the bf16 output (kernarg 224, [B][H][D]) - no combine launch.
pub fn pa_decode16_lastc(name: &str, page: i32) -> Function {
    pa_decode16_gen(name, page, 32, 0, false, false, true)
}

pub fn pa_decode16_x(name: &str, page: i32, kt: i32, fp8: bool) -> Function {
    pa_decode16_kv(name, page, kt, if fp8 { 1 } else { 0 })
}

/// `fp8` (`pa_decode16_fp8_hd128_p<P>`): an fp8 e4m3 KV cache with per-tensor scales (`reshape_cache_flash_fp8_h<Hk>_d128`: [pages, P, HK,
/// 128] bytes; k_scale @208, v_scale @224 f32). K: an fp8 tile (rows of 128 bytes, chunk ^ key % 8), each QK operand 8 bytes -> 8 bf16
/// (exact); V: the fp8 rows staged, then converted into the bf16 V tile the transposed reads use. k_scale folds into the softmax scale,
/// v_scale into the output.
/// `kvm`: 0 bf16 KV, 1 fp8 flash layout (per-tensor scales), 2 fp8 blocked layout (per-token scales).
fn pa_decode16_kv(name: &str, page: i32, kt: i32, kvm: u8) -> Function {
    pa_decode16_kvm(name, page, kt, kvm, false)
}

fn pa_decode16_kvm(name: &str, page: i32, kt: i32, kvm: u8, mtp: bool) -> Function {
    pa_decode16_all(name, page, kt, kvm, mtp, false)
}

fn pa_decode16_all(name: &str, page: i32, kt: i32, kvm: u8, mtp: bool, fused: bool) -> Function {
    pa_decode16_gen(name, page, kt, kvm, mtp, fused, false)
}

fn pa_decode16_gen(name: &str, page: i32, kt: i32, kvm: u8, mtp: bool, fused: bool, lastc: bool) -> Function {
    assert!(!lastc || (kvm == 0 && !mtp && !fused), "decode16 last-arriver combine: bf16, qlen 1, split form");
    assert!(!fused || (kvm == 0 && !mtp), "decode16 fused: bf16, qlen 1");
    let fp8 = kvm >= 1;
    let blocked = kvm == 2;
    assert!(!fp8 || kt == 32, "decode16 fp8: 32-key tiles");
    assert!(!blocked || page >= 16, "decode16 blocked fp8: BS >= 16");
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
    if blocked {
        b.arg("k_scales", DType::Ptr, 208, 8);
        b.arg("v_scales", DType::Ptr, 224, 8);
    } else if fp8 {
        b.arg("k_scale", DType::F32, 208, 4);
        b.arg("v_scale", DType::F32, 224, 4);
    }
    if mtp {
        b.arg("qlen", DType::I32, 240, 4);
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
    // (MTP: qlen rows per sequence: the "batch" of the Q / partial tensors is B qlen)
    let bb = if mtp {
        b.ka_range(240, 1, 16);
        bb * Ext::ka(240)
    } else {
        bb
    };
    b.arg_tensor("ptr_q", bb.clone() * hh.clone() * Ext::c(ROW as i64));
    let cache = np * Ext::c(page as i64) * hk * Ext::c(krow as i64);
    b.arg_tensor("ptr_k", cache.clone());
    b.arg_tensor("ptr_v", cache);
    if blocked {
        let sc = Ext::ka(160) * Ext::c(page as i64) * Ext::ka(128) * Ext::c(4);
        b.arg_tensor("k_scales", sc.clone());
        b.arg_tensor("v_scales", sc);
    }
    b.arg_tensor("ptr_bt", bb.clone() * mp * Ext::c(4));
    b.arg_tensor("ptr_cl", bb.clone() * Ext::c(4));
    if fused {
        b.ka_is(176, Ext::c(1));
        b.arg_tensor("ptr_po", bb.clone() * hh.clone() * Ext::c(2 * D as i64));
    } else {
        b.arg_tensor("ptr_po", bb.clone() * hh.clone() * ss.clone() * Ext::c(4 * 4 * D as i64));
    }
    b.arg_tensor("ptr_plse", bb.clone() * hh.clone() * ss.clone() * Ext::c(16));
    if lastc {
        b.arg("ptr_counter", DType::Ptr, 208, 8);
        b.arg("ptr_out", DType::Ptr, 224, 8);
        b.arg_tensor("ptr_counter", bb.clone() * Ext::ka(128) * Ext::c(4));
        b.arg_tensor("ptr_out", bb * hh * Ext::c(2 * D as i64));
    }

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
        // the quotient IS the contract symbol G (H = HK G, ka_is above): a point fact, so G-scaled offsets stay linear for the store proof
        b.assume_trusted(q, Ext::sym("G"), Ext::sym("G"), "GQA group G = H / HK (contract H = HK G; G in [1, 16])");
        q
    };
    let j = b.bind_i32(lane.lane() & 15);
    // MTP: column j = qpos G + head index; qlen query rows per sequence (row index bat qlen + qpos)
    let (row_live, head, qrow, qpos, qlen) = if mtp {
        let ql = { let v = b.let_(Op::KernargI32 { offset: 240 }); b.read_first_lane(v) };
        b.assume_trusted(ql, Ext::c(1), Ext::c(16), "MTP: qlen in [1, 16], G qlen <= 16 (host contract)");
        let mut qp = b.c_i32(0);
        for t in 1..16i32 {
            qp = b.bind_i32(ev(qp) + ev(j).ge(ev(gsz) * t).select_i32(1, 0));
        }
        let qp = b.bind_i32(ev(qp).min(ev(ql) - 1));
        let hi = b.bind_i32(ev(j) - ev(qp) * ev(gsz));
        let live = b.bind_cond(ev(j).lt(ev(gsz) * ev(ql)));
        let hd = b.bind_i32(ev(hk) * ev(gsz) + ev(hi).min(ev(gsz) - 1));
        let qr = b.bind_i32(ev(bat) * ev(ql) + ev(qp));
        (live, hd, qr, Some(qp), Some(ql))
    } else {
        let live = b.bind_cond(ev(j).lt(ev(gsz)));
        let hd = b.bind_i32(ev(hk) * ev(gsz) + ev(j).min(ev(gsz) - 1));
        (live, hd, bat, None, None)
    };
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
    let (k_lo, k_hi) = grid::key_chunk(&mut b, ctx, ns, nsplit, lg, LKT);
    let ntile = { let v = b.bind_i32((ev(k_hi) - ev(k_lo) + (KT - 1)).max(0).lshr(LKT)); b.read_first_lane(v) };
    // the lane's key limit (MTP: query position qpos sees keys < ctx - qlen + 1 + qpos)
    let kmax = match (qpos, qlen) {
        (Some(qp), Some(ql)) => b.bind_i32(ev(k_hi).min(ev(ctx) - ev(ql) + 1 + ev(qp))),
        _ => k_hi,
    };

    // ---- descriptors ----
    // the rows end of this sequence (MTP: qlen rows per sequence)
    let rend = qlen.map(|ql| { let v = b.bind_i32((ev(bat) + 1) * ev(ql)); b.read_first_lane(v) });
    let re = move || -> amdgpu_dsl::I32 { match rend { Some(r) => ev(r), None => ev(bat) + 1 } };
    let r_q = { let n = b.bind_i32(re() * ev(hq) * ROW); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_q, n) };
    let page_bytes = page * krow;
    let r_k = { let n = b.bind_i32(ev(npages) * ev(hkv_n) * page_bytes); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_k, n) };
    let r_v = { let n = b.bind_i32(ev(npages) * ev(hkv_n) * page_bytes); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_v, n) };
    let r_bt = { let n = b.bind_i32((ev(bat) + 1) * ev(maxp) << 2); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_bt, n) };
    let bt_so = { let v = b.bind_i32(ev(bat) * ev(maxp) << 2); b.read_first_lane(v) };
    let maxk = b.bind_i32((ev(maxp) * page) - 1);


    // ---- Q^T fragments (B operand of 16x16x32: lane = head column, k = d 32 c + 8 g + 0..7) ----
    let q_vo = b.bind_i32((ev(qrow) * ev(hq) + ev(head)) * ROW + (ev(grp) << 4));
    let c0 = b.c_i32(0);
    let qf: Vec<ValueId> = (0..4).map(|c| b.buffer_load_b128(r_q, q_vo, c0, (64 * c) as u32)).collect();

    // ---- private LDS: per wave K (16 KB) + V (16 KB), as decode.rs ----
    let mut lds = LdsPlan::new();
    // K and V in separate private regions (their DMAs complete separately: QK waits for K only)
    let kreg = lds.private("K", (KT * krow) as i64, 4, &[]);
    let vreg = lds.private("V", (KT * ROW) as i64, 4, &[]);
    // fp8: the staged fp8 V rows (converted into the bf16 V tile)
    let vsreg = fp8.then(|| lds.private("VS", (KT * D) as i64, 4, &[]));
    // fused: wave w's normalized partial O [16 cols][128] f32 over its own K region (8 KB w) and its LSE [16] at the start of its own V
    // region (32 KB + 8 KB w): a wave writes only its own dead storage, before the barrier
    const EPI_L: i32 = 32768;
    assert!(!fused || KT == 32, "decode16 fused: 32-key tiles (8 KB K / V per wave)");
    let epi = fused.then(|| lds.tile_over("EPI", (EPI_L + 3 * 8192 + 64) as i64, &[&kreg, &vreg]));
    // last-arriver: the broadcast ticket (thread 0's old counter value)
    let tk = lastc.then(|| lds.tile("TK", 16, amdgpu_tile::Layout::Linear));
    lds.commit(&mut b);
    // fused: this wave's slice of EPI (O at 8 KB w + 512 col + 4 d, LSE at 32 KB + 64 w + 4 col)
    let epi_o = fused.then(|| b.bind_i32((ev(wv) << 13) + (ev(j) << 9) + (ev(grp) << 4)));
    let epi_l = fused.then(|| b.bind_i32((ev(wv) << 13) + (ev(j) << 2) + EPI_L));
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
    let (csc, vsc) = if blocked {
        (csc, None)
    } else if fp8 {
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
    // page groups (bf16 flash layout): k_lo is a multiple of KT, so a tile's keys fall into KT / min(page, KT) groups of consecutive keys that share ONE page (page 16, KT 32: 2 pages); the 8 per-lane
    // block-table loads per tile collapse to one wave-uniform load per group (the 64 lanes of every DMA hold 4 keys of the same group). The page of the clamped key (min(key, k_hi - 1)) is the page of the group's
    // first key clamped the same way.
    // (page >= 16: at most 2 groups per tile, and every DMA issue's 4 keys per lane group stay in one page; small pages keep the per-lane path)
    let grp_path = !fp8 && !blocked && page >= 16;
    // the V granule swizzle: every bf16 flash form (written by both DMA paths, read below)
    let vswz = !fp8 && !blocked;
    let pgsz = page.min(KT);
    let ngrp = if grp_path { KT / pgsz } else { 0 };
    let grp_of = move |i: i32| (4 * i) / pgsz;
    let group_page = |b: &mut FnBuilder, key_t: amdgpu_dsl::I32, g: i32| -> ValueId {
        let key = b.bind_i32((key_t + g * pgsz).min(ev(k_hi) - 1).max(0).min(ev(maxk)));
        let bvo = b.bind_i32(ev(key).lshr(page.trailing_zeros() as i32) << 2);
        b.buffer_load_f32(r_bt, bvo, bt_so, 0)
    };
    // grouped path, per DMA i: the lane's in-page row offset (kk_i % pgsz) HK ROW + hk ROW and the K / V source-chunk swizzles (by the
    // destination row: loop invariant); per tile only the group page base and the cap of rows past the range (min) remain
    let hkrow = { let v = b.bind_i32(ev(hkv_n) * ROW); b.read_first_lane(v) };
    let pgstride = { let v = b.bind_i32(ev(hkrow) * page); b.read_first_lane(v) };
    let ginv: Vec<(ValueId, ValueId, ValueId)> = if grp_path {
        (0..KT / 4)
            .map(|i| {
                let kk = b.bind_i32(kk_of(i));
                let a = b.bind_i32((ev(kk) & (pgsz - 1)) * ev(hkrow) + ev(hk) * ROW);
                (a, b.bind_i32((ev(l16) ^ (ev(kk) & 15)) << 4), b.bind_i32((ev(l16) ^ ((ev(kk) & 7) << 1)) << 4))
            })
            .collect()
    } else {
        Vec::new()
    };
    let any = b.bind_cond(ev(ntile).gt(0));
    b.begin_if(Opnd::Val(any));
    let vars = b.begin_loop(&inits);
    let (it, m_v, l_v) = (vars[0], vars[1], vars[2]);
    let ov: Vec<VarId> = vars[3..11].to_vec();
    let sv_: Vec<VarId> = vars[11..11 + NKB].to_vec();
    let itx = amdgpu_dsl::varx(it);
    let key_t = { let v = b.bind_i32(ev(k_lo) + (itx.clone() << LKT)); b.read_first_lane(v) };
    b.wait_lds(0);
    // grouped pages (bf16 flash): one load per page group, the raw value shared by the group's DMAs
    let gpages: Vec<ValueId> = if grp_path { (0..ngrp).map(|g| group_page(&mut b, ev(key_t), g)).collect() } else { Vec::new() };
    let lookups: Vec<(ValueId, ValueId)> = if grp_path {
        (0..KT / 4)
            .map(|i| {
                let key = b.bind_i32(ev(key_t) + kk_of(i));
                // (key >= 0 and k_hi > k_lo >= 0 inside the loop: min only - `.max(0)` lowers to a compare + select pair with an s_nop)
                let kc = b.bind_i32(ev(key).min(ev(k_hi) - 1).min(ev(maxk)));
                (kc, gpages[grp_of(i) as usize])
            })
            .collect()
    } else { (0..if fp8 { KT / 8 } else { KT / 4 })
        .map(|i| {
            // (fp8 rows of 128 bytes: 8 lanes per key, keys 8 i + l / 8)
            let key = if fp8 { b.bind_i32(ev(key_t) + (lane.lane().lshr(3)) + 8 * i) } else { b.bind_i32(ev(key_t) + kk_of(i)) };
            let kc = b.bind_i32(ev(key).min(ev(k_hi) - 1).max(0).min(ev(maxk)));
            let bvo = b.bind_i32(ev(kc).lshr(page.trailing_zeros() as i32) << 2);
            (kc, b.buffer_load_f32(r_bt, bvo, bt_so, 0))
        })
        .collect() };
    // all K DMAs, then all V DMAs: QK and the softmax wait only for K (counted vmcnt), the V rows land meanwhile
    let nv = lookups.len() as u8;
    let gclamped: Vec<ValueId> = gpages.iter().map(|&pv| { let pg = b.bitcast_i32(pv); b.bind_i32(ev(pg).max(0).min(ev(npages) - 1)) }).collect();
    // grouped path: offset_i = min(tile row base + A_i, cap of the group) + group page base (the cap keeps a lane past k_hi on a row of
    // the page of k_hi - 1 at or below it: real data, masked later)
    let goff: Vec<(ValueId, ValueId)> = if grp_path {
        let tb = { let v = b.bind_i32((ev(key_t) & (page - 1)) * ev(hkrow)); b.read_first_lane(v) };
        let gbase: Vec<(ValueId, ValueId)> = (0..ngrp)
            .map(|g| {
                // the group's last key reaches k_hi - 1: cap at k_hi - 1's in-page row, else no cap (page - 1)
                let lastk = ev(key_t) + (g * pgsz + pgsz - 1);
                let cr = { let v = b.bind_i32(lastk.ge(ev(k_hi) - 1).select_i32((ev(k_hi) - 1).max(0) & (page - 1), amdgpu_dsl::I32::c(page - 1))); b.read_first_lane(v) };
                let cap = { let v = b.bind_i32(ev(cr) * ev(hkrow) + ev(hk) * ROW); b.read_first_lane(v) };
                let pb = b.bind_i32(ev(gclamped[g as usize]) * ev(pgstride));
                (cap, pb)
            })
            .collect();
        (0..KT / 4)
            .map(|i| {
                let (cap, pb) = gbase[grp_of(i) as usize];
                let (a, sk, sv) = ginv[i as usize];
                let o = b.bind_i32((ev(a) + ev(tb)).min(ev(cap)) + ev(pb));
                (b.bind_i32(ev(o) + ev(sk)), b.bind_i32(ev(o) + ev(sv)))
            })
            .collect()
    } else {
        Vec::new()
    };
    let rows: Vec<(ValueId, ValueId)> = lookups
        .iter()
        .enumerate()
        .map(|(i, &(kc, pv))| {
            let pg = if grp_path {
                gclamped[grp_of(i as i32) as usize]
            } else {
                let pg = b.bitcast_i32(pv);
                b.bind_i32(ev(pg).max(0).min(ev(npages) - 1))
            };
            if blocked {
                // blocked K [NB][HK][D / 16][BS][16]: the key's first 16-byte chunk (chunk j at + j BS 16)
                (kc, b.bind_i32((ev(pg) * ev(hkv_n) + ev(hk)) * (D * page) + ((ev(kc) & (page - 1)) << 4)))
            } else {
                (kc, b.bind_i32(((ev(pg) * page + (ev(kc) & (page - 1))) * ev(hkv_n) + ev(hk)) * krow))
            }
        })
        .collect();
    let l8 = b.bind_i32(lane.lane() & 7);
    me.write(&mut b, Opnd::Var(it), |b| {
        for (i, &(kc, rowoff)) in rows.iter().enumerate() {
            // (fp8: chunk ^ key % 8 within the 8-chunk row)
            let kvo = if blocked {
                b.bind_i32(ev(rowoff) + (ev(l8) ^ (ev(kc) & 7)) * (16 * page))
            } else if fp8 { b.bind_i32(ev(rowoff) + ((ev(l8) ^ (ev(kc) & 7)) << 4)) } else if grp_path { goff[i].0 } else { b.bind_i32(ev(rowoff) + ((ev(l16) ^ (ev(kc) & 15)) << 4)) };
            let m0k = b.bind_i32(mbase.clone() + ((i as i32) << 10));
            b.buffer_load_lds_b128(r_k, kvo, c0, m0k, 0);
        }
    });
    // blocked: the per-key scales of the lane's 8 S values (keys 16 blk + 4 g + r of the tile): slot of key l % 32 by one block-table
    // load, shared by ds_bpermute; then the k / v scale loads (in flight behind the QK)
    let kvsc: Option<(Vec<ValueId>, Vec<ValueId>)> = blocked.then(|| {
        let kl = b.bind_i32((ev(key_t) + (lane.lane() & 31)).min(ev(k_hi) - 1).max(0).min(ev(maxk)));
        let bvo = b.bind_i32(ev(kl).lshr(page.trailing_zeros() as i32) << 2);
        let pv = b.buffer_load_f32(r_bt, bvo, bt_so, 0);
        let pg = b.bitcast_i32(pv);
        let pg = b.bind_i32(ev(pg).max(0).min(ev(npages) - 1));
        let sl = b.bind_i32(ev(pg) * page + (ev(kl) & (page - 1)));
        let ns_ = { let v = b.bind_i32(ev(npages) * (page * 4)); b.read_first_lane(v) };
        let so = { let v = b.bind_i32(ev(hk) * ev(npages) * (page * 4)); b.read_first_lane(v) };
        let (pks, pvs) = (b.let_(Op::KernargPtr { offset: 208 }), b.let_(Op::KernargPtr { offset: 224 }));
        let (rks, rvs) = { let n = b.bind_i32(ev(ns_) * ev(hkv_n)); let n = b.read_first_lane(n); (b.buffer_desc_dyn(pks, n), b.buffer_desc_dyn(pvs, n)) };
        let mut ks = Vec::new();
        let mut vs = Vec::new();
        for blk in 0..NKB as i32 {
            for r in 0..4i32 {
                let a = b.bind_i32(((ev(grp) << 2) + (16 * blk + r)) << 2);
                let s = b.bpermute(a, sl);
                let s = b.bitcast_i32(s);
                let vo = b.bind_i32(ev(s) << 2);
                ks.push(b.buffer_load_f32(rks, vo, so, 0));
                vs.push(b.buffer_load_f32(rvs, vo, so, 0));
            }
        }
        (ks, vs)
    });
    match &mvs {
        Some(ms) if blocked => {
            // V [NB][HK][D][BS]: the tile's V^T [128 d][32 keys] bytes, 16-byte chunks c = 2 d + h (keys 16 h ..), 4 issues of 64 chunks
            let sb = ms.base();
            let pgs: Vec<ValueId> = (0..2i32)
                .map(|h| {
                    let kc = b.bind_i32((ev(key_t) + 16 * h).min(ev(k_hi) - 1).max(0).min(ev(maxk)));
                    let bvo = b.bind_i32(ev(kc).lshr(page.trailing_zeros() as i32) << 2);
                    let pv = b.buffer_load_f32(r_bt, bvo, bt_so, 0);
                    let pg = b.bitcast_i32(pv);
                    let pg = b.bind_i32(ev(pg).max(0).min(ev(npages) - 1));
                    b.bind_i32((ev(pg) * ev(hkv_n) + ev(hk)) * (D * page) + (ev(kc) & (page - 1) & !15))
                })
                .collect();
            let hsel = b.bind_cond((lane.lane() & 1).eq(1));
            let pgl = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(hsel), a: Opnd::Val(pgs[1]), b: Opnd::Val(pgs[0]) });
            ms.write(&mut b, Opnd::Var(it), |b| {
                for i in 0..4i32 {
                    // chunk c = 64 i + l: d = c / 2 = 32 i + l / 2
                    let vvo = b.bind_i32(ev(pgl) + ((lane.lane().lshr(1)) + 32 * i) * page);
                    let m0v = b.bind_i32(sb.clone() + (i << 10));
                    b.buffer_load_lds_b128(r_v, vvo, c0, m0v, 0);
                }
            });
        }
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
                // V tile swizzle: the 32-byte granule G of tile row r is stored at granule G ^ (r % 8) (the DMA lands lane-linear, so the swizzle is applied to the SOURCE chunk: l16 ^ 2 (r % 8))
                let vvo = if grp_path { goff[i].1 } else { let vsw = b.bind_i32((kk_of(i as i32) & 7) << 1); b.bind_i32(ev(rowoff) + ((ev(l16) ^ ev(vsw)) << 4)) };
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
    // grouped path: the key mask only on the tile that needs it (uniform branch, VALU only, in place)
    let mask_in_place = grp_path && !mtp;
    if mask_in_place {
        let part = b.bind_cond((ev(key_t) + KT).gt(ev(kmax)));
        b.begin_if(Opnd::Val(part));
        for blk in 0..NKB {
            for r in 0..4u8 {
                let x = b.frag4_lane(Opnd::Var(sv_[blk]), r);
                let x = b.bitcast_f32(x);
                let keep = b.bind_cond((ev(key0) + (16 * blk as i32 + r as i32)).lt(ev(kmax)));
                let y = b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(keep), a: Opnd::Val(x), b: Opnd::Val(neg_inf) });
                b.var_set_lane(sv_[blk], r, y);
            }
        }
        b.end_if();
    }
    for blk in 0..NKB {
        for r in 0..4u8 {
            let x = b.frag4_lane(Opnd::Var(sv_[blk]), r);
            let x = b.bitcast_f32(x);
            let x = match &kvsc { Some((ks, _)) => b.mul_f32(x, ks[4 * blk + r as usize]), None => x };
            if mask_in_place {
                s.push(x);
                continue;
            }
            let keep = b.bind_cond((ev(key0) + (16 * blk as i32 + r as i32)).lt(ev(kmax)));
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
        // fp8: the staged V rows -> the bf16 V tile (lane: key l / 2, d half l % 2: 64 fp8 -> 64 bf16); blocked: the staged V^T chunks
        // 4 l .. 4 l + 3 (d 2 l + q / 2, keys 16 (q % 2) ..) -> the bf16 V^T tile [128 d][32 keys] (rows of 64 bytes)
        let sb = ms.base();
        let key = b.bind_i32(lane.lane().lshr(1));
        let hf = b.bind_i32(lane.lane() & 1);
        let src = if blocked { b.bind_i32(sb.clone() + (lane.lane() << 6)) } else { b.bind_i32(sb.clone() + ev(key) * D + (ev(hf) << 6)) };
        let raw: Vec<ValueId> = ms.read_back(&mut b, Opnd::Var(it), |b| (0..4i32).map(|q| { let a = b.bind_i32(ev(src) + 16 * q); b.load_lds_b128(a) }).collect());
        let one = b.c_f32(1.0);
        // (blocked: chunk q of the lane -> d 2 l + q / 2 row (64 bytes), 32 bytes at 32 (q % 2); the per-q offset 64 (q / 2) + 32 (q % 2) = 32 q)
        let dst = if blocked { b.bind_i32(vbase0.clone() + (lane.lane() << 7)) } else { b.bind_i32(vbase0.clone() + ev(key) * ROW + (ev(hf) << 7)) };
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
    // blocked: P times the per-key V scales (the row sum l is unscaled)
    let p: Vec<ValueId> = match &kvsc { Some((_, vs)) => p.iter().zip(vs).map(|(x, s)| b.mul_f32(*x, *s)).collect(), None => p };
    let vb = b.bind_i32(vbase0.clone() + ev(vr));
    // the swizzled granule offsets of the 8 d blocks (hoisted): (db ^ (r0 % 8)) * 32, r0 = 4 grp + l16 / 4
    let r0m = b.bind_i32(((ev(grp) << 2) + ev(l16).lshr(2)) & 7);
    // (the per-granule base registers are hoisted: the loop's reads keep a constant row offset, which the ds_read immediate absorbs)
    let vgran: Vec<ValueId> = if vswz { (0..8i32).map(|db| { let g = b.bind_i32((ev(r0m) ^ db) << 5); b.bind_i32(ev(vb) + ev(g)) }).collect() } else { Vec::new() };
    // blocked: the V^T row (d 16 db + l % 16) bytes of keys 4 g .. 4 g + 3 and 16 + 4 g ..
    let vtb = blocked.then(|| b.bind_i32(vbase0.clone() + ev(l16) * 64 + (ev(grp) << 3)));
    mev.read_back(&mut b, Opnd::Var(it), |b| {
        for t in 0..NKB / 2 {
            let w: Vec<ValueId> = [(2 * t, 0usize), (2 * t, 2), (2 * t + 1, 0), (2 * t + 1, 2)].iter().map(|&(blk, r)| b.cvt_pk_bf16_f32(p[4 * blk + r], p[4 * blk + r + 1])).collect();
            let pf = b.pack4(w[0], w[1], w[2], w[3]);
            for db in 0..8i32 {
                let vf = if blocked {
                    let vtb = vtb.unwrap();
                    let a0 = b.bind_i32(ev(vtb) + 16 * 64 * db);
                    let a1 = b.bind_i32(ev(vtb) + 16 * 64 * db + 32);
                    let lo = b.load_lds_b64(a0);
                    let hi = b.load_lds_b64(a1);
                    b.cat_frag4(lo, hi)
                } else {
                    // (the swizzle exists on the bf16 flash path only: the fp8 forms convert their V rows into an unswizzled bf16 tile)
                    let (a0, a1) = if vswz {
                        (b.bind_i32(ev(vgran[db as usize]) + (32 * t as i32) * ROW), b.bind_i32(ev(vgran[db as usize]) + (32 * t as i32 + 16) * ROW))
                    } else {
                        (b.bind_i32(ev(vb) + (32 * t as i32) * ROW + 32 * db), b.bind_i32(ev(vb) + (32 * t as i32 + 16) * ROW + 32 * db))
                    };
                    b.load_lds_tr_b16x2(a0, a1)
                };
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
    let row_po = b.bind_i32(((ev(qrow) * ev(hq) + ev(head)) * ev(nsplit) + ev(ns)) * (4 * D));
    let big = b.c_i32(0x7fff_fff0);
    let po_vo_r = b.bind_i32(ev(row_po) + (ev(grp) << 4));
    let po_vo = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(row_live), a: Opnd::Val(po_vo_r), b: Opnd::Val(big) });
    let ov_n: Vec<ValueId> = if fused {
        (0..8usize)
            .map(|db| {
                let v: Vec<ValueId> = (0..4u8)
                    .map(|k| {
                        let x = b.frag4_lane(Opnd::Var(ov[db]), k);
                        let x = b.bitcast_f32(x);
                        let y = b.mul_f32(x, inv);
                        b.bitcast_i32(y)
                    })
                    .collect();
                b.frag4(v[0], v[1], v[2], v[3])
            })
            .collect()
    } else {
        Vec::new()
    };
    let r_po = if fused { c0 } else { let n = b.bind_i32(re() * ev(hq) * ev(nsplit) * (4 * D)); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_po, n) };
    for db in 0..if fused { 0 } else { 8usize } {
        let v: Vec<ValueId> = (0..4u8)
            .map(|k| {
                let x = b.frag4_lane(Opnd::Var(ov[db]), k);
                let x = b.bitcast_f32(x);
                let y = b.mul_f32(x, inv);
                b.bitcast_i32(y)
            })
            .collect();
        let f4 = b.frag4(v[0], v[1], v[2], v[3]);
        // (last-arriver: write-through sc1 stores - the lite release needs no L2 write-back)
        if lastc { b.buffer_store_c(r_po, Some(Opnd::Val(po_vo)), c0, (16 * db as u32) * 4, f4, amdgpu_dsl::MemTy::B128, false, amdgpu_dsl::CachePolicy { sc0: false, sc1: true, nt: false }) } else { b.buffer_store_b128(r_po, po_vo, c0, (16 * db as u32) * 4, f4) };
    }
    let lg2 = b.log2_f32(l_tot);
    let x = b.add_f32(Opnd::Var(m_v), lg2);
    let ln2 = b.c_f32(std::f32::consts::LN_2);
    let lse0 = b.mul_f32(x, ln2);
    let lse = b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(pos), a: Opnd::Val(lse0), b: Opnd::Val(neg_inf) });
    if let (Some(e), Some(eo), Some(el)) = (&epi, epi_o, epi_l) {
        e.at(&amdgpu_tile::SlotIdx::new(0, 0)).write(&mut b, |b| {
            for (db, f4) in ov_n.iter().enumerate() {
                let a = b.bind_i32(ev(eo) + 64 * db as i32);
                b.store_lds_b128(a, *f4);
            }
            let li = b.bitcast_i32(lse);
            b.store_lds_b32(el, li);
        });
    } else {
        let r_l = { let n = b.bind_i32(re() * ev(hq) * ev(nsplit) * 4); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_lse, n) };
        let l_vo_r = b.bind_i32(((ev(qrow) * ev(hq) + ev(head)) * ev(nsplit) + ev(ns)) << 2);
        let l_vo = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(row_live), a: Opnd::Val(l_vo_r), b: Opnd::Val(big) });
        if lastc { let li = b.bitcast_i32(lse); b.buffer_store_c(r_l, Some(Opnd::Val(l_vo)), c0, 0, li, amdgpu_dsl::MemTy::B32, false, amdgpu_dsl::CachePolicy { sc0: false, sc1: true, nt: false }) } else { b.buffer_store_f32(r_l, l_vo, c0, 0, lse) };
    }
    b.else_();
    if let (Some(e), Some(eo), Some(el)) = (&epi, epi_o, epi_l) {
        // an empty wave: O = 0, LSE = -inf
        let z = b.c_i32(0);
        let z4 = b.frag4(z, z, z, z);
        e.at(&amdgpu_tile::SlotIdx::new(0, 0)).write(&mut b, |b| {
            for db in 0..8i32 {
                let a = b.bind_i32(ev(eo) + 64 * db);
                b.store_lds_b128(a, z4);
            }
            let ni = b.bitcast_i32(neg_inf);
            b.store_lds_b32(el, ni);
        });
    } else {
        let r_l = { let n = b.bind_i32(re() * ev(hq) * ev(nsplit) * 4); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_lse, n) };
        let l_vo_r = b.bind_i32(((ev(qrow) * ev(hq) + ev(head)) * ev(nsplit) + ev(ns)) << 2);
        let big = b.c_i32(0x7fff_fff0);
        let l_vo = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(row_live), a: Opnd::Val(l_vo_r), b: Opnd::Val(big) });
        if lastc { let ni = b.bitcast_i32(neg_inf); b.buffer_store_c(r_l, Some(Opnd::Val(l_vo)), c0, 0, ni, amdgpu_dsl::MemTy::B32, false, amdgpu_dsl::CachePolicy { sc0: false, sc1: true, nt: false }) } else { b.buffer_store_f32(r_l, l_vo, c0, 0, neg_inf) };
    }
    b.end_if();
    if let Some(tk) = &tk {
        // ---- last-arriver combine: release, ticket, the last CTA of the (b, hk) merges ----
        b.fence_release_lite();
        b.barrier();
        let tid = b.let_(Op::WorkitemId);
        let c0l = b.c_i32(0);
        let c1l = b.c_i32(1);
        let is0 = b.cmp_lt_i32(tid, c1l);
        let one = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(is0), a: Opnd::Val(c1l), b: Opnd::Val(c0l) });
        let p_cnt = b.let_(Op::KernargPtr { offset: 208 });
        let gofs = b.bind_i32((ev(bat) * ev(hkv_n) + ev(hk)) << 2);
        // ONE lane takes the ticket: the other 255 lanes' offsets are out of range and dropped before L2 (an atomic by every lane - even
        // adding 0 - serializes 256 same-cache-line atomics per CTA: ~100 us at B16)
        let ncnt = { let v = b.bind_i32((ev(bat) + 1) * ev(hkv_n) * 4); b.read_first_lane(v) };
        let r_cnt = b.buffer_desc_dyn(p_cnt, ncnt);
        let bigc = b.c_i32(0x7fff_fff0);
        let cvo = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(is0), a: Opnd::Val(gofs), b: Opnd::Val(bigc) });
        let _ = one;
        let old = b.buffer_atomic_add_ret(r_cnt, cvo, c0l, 0, c1l);
        let tkb = tk.base as i32;
        let slot = amdgpu_tile::SlotIdx::new(0, 0);
        tk.at(&slot).write(&mut b, |b| {
            // thread 0 writes the ticket word, the others a scratch word
            let a = b.bind_i32(ev(tid).lt(1).select_i32(amdgpu_dsl::I32::c(tkb), amdgpu_dsl::I32::c(tkb + 4)));
            b.store_lds_b32(a, old);
        });
        b.wait_lds(0);
        b.barrier();
        let ticket = tk.at(&slot).read(&mut b, |b| { let a = b.c_i32(tkb); b.load_lds_b32(a) });
        let sm1 = b.bind_i32(ev(s_n) - 1);
        let is_last = b.let_(Op::CmpGeI32(Opnd::Val(ticket), Opnd::Val(sm1)));
        let any = b.any_lane(is_last);
        // the S CTAs of a (b, hk) share counter[b HK + hk]; exactly one sees ticket S - 1 (one atomic per CTA, the last resets the counter)
        b.wg_winner(any, &[2], "last arriver: the S split CTAs of (b, hk) take tickets 0..S-1 on one counter; only ticket S - 1 runs the combine");
        b.begin_if(Opnd::Val(any));
        {
            b.fence_acquire_lite();
            let zf = b.c_f32(0.0);
            let c0 = b.c_i32(0);
            let col = b.bind_i32(ev(tid).lshr(4));
            let d0 = b.bind_i32((ev(tid) & 15) << 3);
            let ohead = b.bind_i32(ev(hk) * ev(gsz) + ev(col).min(ev(gsz) - 1));
            let prow = b.bind_i32((ev(bat) * ev(hq) + ev(ohead)) * ev(nsplit));
            let r_po = { let n = b.bind_i32((ev(bat) + 1) * ev(hq) * ev(nsplit) * (4 * D)); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_po, n) };
            let r_l = { let n = b.bind_i32((ev(bat) + 1) * ev(hq) * ev(nsplit) * 4); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_lse, n) };
            let ninf = b.c_f32(f32::NEG_INFINITY);
            let mut st: Vec<(DType, Opnd)> = vec![(DType::I32, Opnd::I32(0)), (DType::F32, Opnd::Val(ninf)), (DType::F32, Opnd::Val(zf))];
            st.extend((0..8).map(|_| (DType::F32, Opnd::Val(zf))));
            let v = b.begin_loop(&st);
            let si = amdgpu_dsl::varx(v[0]);
            let row = b.bind_i32(ev(prow) + si.clone());
            let lo = b.bind_i32(ev(row) << 2);
            let sc1 = amdgpu_dsl::CachePolicy { sc0: false, sc1: true, nt: false };
            let x = b.buffer_load_c(r_l, Some(Opnd::Val(lo)), c0, 0, amdgpu_dsl::MemTy::B32, sc1);
            let x = b.bitcast_f32(x);
            let oo = b.bind_i32(ev(row) * (4 * D) + (ev(d0) << 2));
            let o0 = b.buffer_load_c(r_po, Some(Opnd::Val(oo)), c0, 0, amdgpu_dsl::MemTy::B128, sc1);
            let o1 = b.buffer_load_c(r_po, Some(Opnd::Val(oo)), c0, 16, amdgpu_dsl::MemTy::B128, sc1);
            // online merge in natural-log units: m' = max(m, x), scale = e^(m - m'), w = e^(x - m')
            let mn = b.max_f32(Opnd::Var(v[1]), x);
            let mbits = b.bitcast_i32(mn);
            let fin = b.bind_cond(ev(mbits).ne(0xff80_0000u32 as i32));
            let mref = b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(fin), a: Opnd::Val(mn), b: Opnd::Val(zf) });
            let l2e = b.c_f32(std::f32::consts::LOG2_E);
            let d1 = b.sub_f32(Opnd::Var(v[1]), mref);
            let t1 = b.mul_f32(d1, l2e);
            let sc = b.exp2_f32(t1);
            let d2 = b.sub_f32(x, mref);
            let t2 = b.mul_f32(d2, l2e);
            let w = b.exp2_f32(t2);
            let l1 = b.mul_f32(Opnd::Var(v[2]), sc);
            let l2 = b.add_f32(l1, w);
            for k in 0..8u8 {
                let f = if k < 4 { o0 } else { o1 };
                let ox = b.frag4_lane(f, k % 4);
                let ox = b.bitcast_f32(ox);
                let a1 = b.mul_f32(Opnd::Var(v[3 + k as usize]), sc);
                let a2 = b.let_(Op::FmaF32(Opnd::Val(w), Opnd::Val(ox), Opnd::Val(a1)));
                b.set_var(v[3 + k as usize], a2);
            }
            b.set_var(v[1], mn);
            b.set_var(v[2], l2);
            let n1 = b.bind_i32(si + 1);
            b.set_var(v[0], n1);
            let dn = b.bind_cond(ev(n1).ge(ev(nsplit)));
            b.break_if(dn);
            b.end_loop();
            let tiny = b.c_f32(1e-30);
            let den = b.max_f32(Opnd::Var(v[2]), tiny);
            let rinv = b.rcp_f32(den);
            let outv: Vec<ValueId> = (0..8usize).map(|k| b.mul_f32(Opnd::Var(v[3 + k]), rinv)).collect();
            let w4: Vec<ValueId> = (0..4).map(|i| b.cvt_pk_bf16_f32(outv[2 * i], outv[2 * i + 1])).collect();
            let f4 = b.pack4(w4[0], w4[1], w4[2], w4[3]);
            let live = b.bind_cond(ev(col).lt(ev(gsz)));
            let o_vo_r = b.bind_i32(((ev(bat) * ev(hq) + ev(ohead)) * D + ev(d0)) << 1);
            let bigo = b.c_i32(0x7fff_fff0);
            let o_vo = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(live), a: Opnd::Val(o_vo_r), b: Opnd::Val(bigo) });
            let p_out = b.let_(Op::KernargPtr { offset: 224 });
            let r_o = { let n = b.bind_i32((ev(bat) + 1) * ev(hq) * (2 * D)); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_out, n) };
            b.buffer_store_b128(r_o, o_vo, c0, 0, f4);
            // the counter back to 0 for the next launch (thread 0)
            let zc = b.c_i32(0);
            let cvo = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(is0), a: Opnd::Val(zc), b: Opnd::Val(bigo) });
            let r_c = { let n = b.bind_i32((ev(bat) + 1) * ev(hkv_n) * 4); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_cnt, n) };
            let cso = { let v = b.bind_i32(ev(gofs)); b.read_first_lane(v) };
            let zcf = b.bitcast_f32(zc);
            b.buffer_store_f32(r_c, cvo, cso, 0, zcf);
        }
        b.end_if();
    }
    if let Some(e) = &epi {
        // ---- fused combine: thread t = (column t / 16, 8 d at 8 (t % 16)) merges the 4 wave partials, bf16 out ----
        b.wait_lds(0);
        b.barrier();
        let zf = b.c_f32(0.0);
        let c0 = b.c_i32(0);
        let tid = b.let_(Op::WorkitemId);
        let col = b.bind_i32(ev(tid).lshr(4));
        let d0 = b.bind_i32((ev(tid) & 15) << 3);
        let (ls, os): (Vec<ValueId>, Vec<(ValueId, ValueId)>) = e.at(&amdgpu_tile::SlotIdx::new(0, 0)).read(&mut b, |b| {
            let ls = (0..4i32).map(|w| { let a = b.bind_i32((ev(col) << 2) + (EPI_L + 8192 * w)); b.load_lds_b32(a) }).collect();
            let os = (0..4i32)
                .map(|w| {
                    let a = b.bind_i32((ev(col) << 9) + (ev(d0) << 2) + (w << 13));
                    let a1 = b.bind_i32(ev(a) + 16);
                    (b.load_lds_b128(a), b.load_lds_b128(a1))
                })
                .collect();
            (ls, os)
        });
        let ls: Vec<ValueId> = ls.into_iter().map(|x| b.bitcast_f32(x)).collect();
        let mut mx = ls[0];
        for x in &ls[1..] {
            mx = b.max_f32(mx, *x);
        }
        let mbits = b.bitcast_i32(mx);
        let fin = b.bind_cond(ev(mbits).ne(0xff80_0000u32 as i32));
        let mm = b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(fin), a: Opnd::Val(mx), b: Opnd::Val(zf) });
        let c_l2e = b.c_f32(std::f32::consts::LOG2_E);
        let ws: Vec<ValueId> = ls.iter().map(|x| { let d = b.sub_f32(*x, mm); let t = b.mul_f32(d, c_l2e); b.exp2_f32(t) }).collect();
        let mut wsum = ws[0];
        for x in &ws[1..] {
            wsum = b.add_f32(wsum, *x);
        }
        let tiny = b.c_f32(1e-30);
        let den = b.max_f32(wsum, tiny);
        let rinv = b.rcp_f32(den);
        let mut outv: Vec<ValueId> = Vec::with_capacity(8);
        for k in 0..8u8 {
            let mut acc = zf;
            for w in 0..4usize {
                let f = if k < 4 { os[w].0 } else { os[w].1 };
                let x = b.frag4_lane(f, k % 4);
                let x = b.bitcast_f32(x);
                acc = b.let_(Op::FmaF32(Opnd::Val(ws[w]), Opnd::Val(x), Opnd::Val(acc)));
            }
            outv.push(b.mul_f32(acc, rinv));
        }
        let w: Vec<ValueId> = (0..4).map(|i| b.cvt_pk_bf16_f32(outv[2 * i], outv[2 * i + 1])).collect();
        let f4 = b.pack4(w[0], w[1], w[2], w[3]);
        // column col -> head hk G + col (live col < G)
        let live = b.bind_cond(ev(col).lt(ev(gsz)));
        let ohead = b.bind_i32(ev(hk) * ev(gsz) + ev(col).min(ev(gsz) - 1));
        let o_vo_r = b.bind_i32(((ev(bat) * ev(hq) + ev(ohead)) * D + ev(d0)) << 1);
        let bigo = b.c_i32(0x7fff_fff0);
        let o_vo = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(live), a: Opnd::Val(o_vo_r), b: Opnd::Val(bigo) });
        let r_o = { let n = b.bind_i32((ev(bat) + 1) * ev(hq) * (2 * D)); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_po, n) };
        b.buffer_store_b128(r_o, o_vo, c0, 0, f4);
    }
    b.finish()
}

/// The f32 value of a loop variable (after the loop).
fn l_v_val(b: &mut FnBuilder, v: VarId) -> ValueId {
    let z = b.c_f32(0.0);
    b.add_f32(Opnd::Var(v), z)
}