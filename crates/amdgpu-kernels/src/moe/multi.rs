//! Multi-block MoE sorting (docs/op-coverage-memtemplates.md T5 + T7): the same output format as the single-block `moe_sorting_e<E>_k<K>_b<BM>` (see `moe/mod.rs`), for large routed-pair counts: three kernels
//! sharing a workspace `ws` (u32 `[E][64]`, row e = the 64 per-block slots of expert e; at most G <= 64 blocks).
//!
//! 1. `moe_count_e<E>_k<K>`        grid G, 1024 threads: block g counts its slice `[g * P, min((g + 1) * P, N))` of the routed pairs per expert (LDS atomics) and writes `ws[e][g]`.
//!    ABI: topk_ids @0, ws @8, M @16 (i32), P @20 (i32, pairs per block).
//! 2. `moe_scan_e<E>_k<K>_b<BM>`   grid 1, 1024 threads: per expert the sum over the blocks, the padded prefix scan over experts, then `ws[e][g] = first position of expert e's pairs of block g`
//!    (padded offset of e + the pairs of the earlier blocks), plus `tile_start`, `num_valid_ids`, `sorted_expert_ids` and the padding / tail sentinels of `sorted_token_ids` / `sorted_weights`.
//!    ABI: ws @0, sorted_token_ids @8, sorted_weights @16, sorted_expert_ids @24, num_valid_ids @32, tile_start @40, M @48 (i32), G @52 (i32).
//! 3b. `moe_scatter_ord_e<E>_k<K>_b<BM>` (same ABI): the DETERMINISTIC variant: every expert's pairs end up in ascending pair index (token, slot) order -- a stable counting sort. Chunks of 1024 consecutive pairs are processed
//!    in order; inside a chunk a wave matches equal experts with `ballot` / `mbcnt` (rank among the earlier lanes with the same expert + the wave's count per expert), the per-expert counts of the 16 waves are
//!    prefix-summed in LDS (one thread per expert), and the expert's cursor advances by the chunk total.
//! 3. `moe_scatter_e<E>_k<K>_b<BM>` grid G, 1024 threads: block g places its pairs: position = `ws[e][g]` advanced by an LDS cursor (`ds_add_rtn_u32`), so the layout is deterministic across blocks;
//!    inside a block the order of one expert's pairs is the arrival order. ABI: topk_ids @0, topk_weights @8, sorted_token_ids @16, sorted_weights @24, ws @32, M @40 (i32), P @44 (i32).

use amdgpu_dsl::{AtomicOp, BinOp, CachePolicy, CmpPred, CmpTy, CompileProfile, DType, Ext, FnBuilder, Function, MemTy, Op, Opnd, ValueId};
use crate::lds_decl::tile_at;
use amdgpu_tile::{ev, SlotIdx};

const T: i32 = 1024;
const WS_ROW: i32 = 256; // bytes per expert row of the workspace (64 slots)

pub fn by_name(name: &str) -> Option<Function> {
    let (rest, kind) = if let Some(r) = name.strip_prefix("moe_scatter_ord_e") {
        (r, 4)
    } else if let Some(r) = name.strip_prefix("moe_count_e") {
        (r, 1)
    } else if let Some(r) = name.strip_prefix("moe_scan_e") {
        (r, 2)
    } else if let Some(r) = name.strip_prefix("moe_scatter_e") {
        (r, 3)
    } else {
        return None;
    };
    let (e, rest) = rest.split_once("_k")?;
    let (k, bm) = match rest.split_once("_b") {
        Some((k, bm)) => (k, Some(bm)),
        None => (rest, None),
    };
    let (e, k): (i64, i64) = (e.parse().ok()?, k.parse().ok()?);
    if !((8..=512).contains(&e) && (1..=16).contains(&k)) {
        return None;
    }
    match kind {
        1 if bm.is_none() => Some(moe_count(name, e, k)),
        3 | 4 => {
            let bm: i64 = bm?.parse().ok()?;
            [16, 32, 64, 128].contains(&bm).then(|| if kind == 3 { moe_scatter(name, e, k, bm) } else { moe_scatter_ordered(name, e, k, bm) })
        }
        2 => {
            let bm: i64 = bm?.parse().ok()?;
            [16, 32, 64, 128].contains(&bm).then(|| moe_scan(name, e, k, bm))
        }
        _ => None,
    }
}

/// The expert id of pair `i`, clamped to 0..=1023 for the verifier; 1023 (an unused slot) when `i` is not a valid pair of this block or the id is outside 0..E.
fn pair_expert(b: &mut FnBuilder, r_ids: ValueId, i: ValueId, in_slice: ValueId, e: i64) -> ValueId {
    let off = b.bind_i32(ev(i) << 2);
    let f = b.buffer_load_f32(r_ids, off, 0i32, 0);
    let id0 = b.bitcast_i32(f);
    let id1 = b.let_(Op::MinI32(Opnd::Val(id0), Opnd::I32(1023)));
    let id = b.bin(BinOp::MaxI32, id1, 0i32);
    let ge = b.cmp(CmpPred::Ge, CmpTy::I32, id0, 0);
    let lt = b.cmp(CmpPred::Lt, CmpTy::I32, id0, e as i32);
    let v1 = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(in_slice), a: Opnd::Val(id), b: Opnd::I32(1023) });
    let v2 = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(ge), a: Opnd::Val(v1), b: Opnd::I32(1023) });
    b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(lt), a: Opnd::Val(v2), b: Opnd::I32(1023) })
}

pub fn moe_count(name: &str, e: i64, k: i64) -> Function {
    let mut fb = FnBuilder::new(name, T as u32);
    fb.compile_profile(CompileProfile::Default);
    let b = &mut fb;
    b.arg("topk_ids", DType::Ptr, 0, 8);
    b.arg("ws", DType::Ptr, 8, 8);
    b.arg("m", DType::I32, 16, 4);
    b.arg("per", DType::I32, 20, 4);
    let mm = Ext::sym("M");
    b.sym_range("M", 1, 1 << 22);
    b.ka_is(16, mm.clone());
    b.arg_tensor("topk_ids", mm * Ext::c(4 * k));
    b.arg_tensor("ws", Ext::c(e * WS_ROW as i64));
    b.lds(4096);
    let t_a = tile_at(b, "A", 0, 4096);   // the per-expert counts: zeroed, counted by atomics (one version), read after the barrier
    let (p_ids, p_ws) = (b.let_(Op::KernargPtr { offset: 0 }), b.let_(Op::KernargPtr { offset: 8 }));
    let m_v = b.let_(Op::KernargI32 { offset: 16 });
    let per = b.let_(Op::KernargI32 { offset: 20 });
    let tid = b.let_(Op::WorkitemId);
    let bx = b.let_(Op::WorkgroupIdX);
    let n_el = b.bind_i32(ev(m_v) * k as i32);
    let r_ids = { let v = b.bind_i32(ev(n_el) << 2); let v = b.read_first_lane(v); b.buffer_desc_dyn(p_ids, v) };
    let r_ws = b.buffer_desc(p_ws, (e * WS_ROW as i64) as u32);
    let tid4 = b.bind_i32(ev(tid) << 2);
    let zero = b.c_i32(0);
    t_a.at(&SlotIdx::new(0, 0)).write(b, |b| b.store_lds(tid4, zero, MemTy::B32, false));
    b.wait_lds(0);
    b.barrier();
    let base = b.bind_i32(ev(bx) * ev(per));
    let end0 = b.bind_i32(ev(base) + ev(per));
    let end = b.let_(Op::MinI32(Opnd::Val(end0), Opnd::Val(n_el)));
    let it_count = b.bind_i32((ev(per) + (4 * T - 1)).lshr(12));
    b.loop_(&[(DType::I32, Opnd::I32(0))], |b, vars| {
        let it = vars[0];
        let done = b.let_(Op::CmpGeI32(Opnd::Var(it), Opnd::Val(it_count)));
        b.break_if(done);
        let off = b.let_(Op::MulI32(Opnd::Var(it), Opnd::I32(4 * T)));
        let mut eids = Vec::new();
        for u in 0..4 {
            let i = b.bind_i32(ev(base) + ev(off) + ev(tid) + u * T);
            let ins = b.let_(Op::CmpLtI32(Opnd::Val(i), Opnd::Val(end)));
            eids.push(pair_expert(b, r_ids, i, ins, e));
        }
        for eid in eids {
            let a = b.bind_i32(ev(eid) << 2);
            t_a.at(&SlotIdx::new(0, 0)).write(b, |b| b.atomic_lds(AtomicOp::AddU32, a, 1i32, false));
        }
        let nx = b.let_(Op::AddI32(Opnd::Var(it), Opnd::I32(1)));
        b.set_var(it, nx);
    });
    b.wait_lds(0);
    b.barrier();
    let cnt = t_a.at(&SlotIdx::new(0, 0)).read(b, |b| b.load_lds(tid4, MemTy::B32));
    b.wait_lds(0);
    let in_e = b.cmp(CmpPred::Lt, CmpTy::I32, tid, e as i32);
    let o0 = b.bind_i32(ev(tid) * WS_ROW + (ev(bx) << 2));
    let oob = b.c_i32(0x7fff_ff00);
    let o = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(in_e), a: Opnd::Val(o0), b: Opnd::Val(oob) });
    let cf = b.bitcast_f32(cnt);
    b.buffer_store_f32(r_ws, o, 0i32, 0, cf);
    fb.finish()
}

pub fn moe_scan(name: &str, e: i64, k: i64, bm: i64) -> Function {
    let mut fb = FnBuilder::new(name, T as u32);
    fb.compile_profile(CompileProfile::Default);
    let b = &mut fb;
    for (i, nm) in ["ws", "sorted_token_ids", "sorted_weights", "sorted_expert_ids", "num_valid_ids", "tile_start"].iter().enumerate() {
        b.arg(nm, DType::Ptr, 8 * i as u32, 8);
    }
    b.arg("m", DType::I32, 48, 4);
    b.arg("g", DType::I32, 52, 4);
    let mm = Ext::sym("M");
    b.sym_range("M", 1, 1 << 22);
    b.ka_is(48, mm.clone());
    b.grid_is(0, Ext::c(1));
    let maxp = mm.clone() * Ext::c(k) + Ext::c(e * bm - k);
    b.arg_tensor("ws", Ext::c(e * WS_ROW as i64));
    b.arg_tensor("sorted_token_ids", maxp.clone() * Ext::c(4));
    b.arg_tensor("sorted_weights", maxp.clone() * Ext::c(4));
    b.arg_tensor("sorted_expert_ids", maxp.ceil_div(bm) * Ext::c(4));
    b.arg_tensor("num_valid_ids", Ext::c(8));
    b.arg_tensor("tile_start", Ext::c(4 * (e + 1)));
    // LDS: A = counts, S0 / S1 = scan ping-pong, C = offsets; 1024 dwords each
    b.lds(4 * 4 * 1024);
    let (offa, offs0, offs1, offc) = (0i32, 4096i32, 8192i32, 12288i32);
    // declared tiles (see `moe_sorting`): A 0 (counts), S0 / S1 ping-pong (step j writes the other tile as version j + 1), C 0 (the exclusive offsets, published by the barrier before the padding loop)
    let t_a = tile_at(b, "A", offa as i64, 4096);
    let t_s0 = tile_at(b, "S0", offs0 as i64, 4096);
    let t_s1 = tile_at(b, "S1", offs1 as i64, 4096);
    let t_c = tile_at(b, "C", offc as i64, 4096);
    let ptr: Vec<ValueId> = (0..6).map(|i| b.let_(Op::KernargPtr { offset: 8 * i })).collect();
    let m_v = b.let_(Op::KernargI32 { offset: 48 });
    let g_v = b.let_(Op::KernargI32 { offset: 52 });
    let tid = b.let_(Op::WorkitemId);
    let n_el = b.bind_i32(ev(m_v) * k as i32);
    let maxp_v = b.bind_i32(ev(n_el) + (e * bm - k) as i32);
    let ntiles_max = b.bind_i32(ev(maxp_v).lshr(bm.trailing_zeros() as i32));
    let dsc = |b: &mut FnBuilder, p: ValueId, bytes: ValueId| { let v = b.read_first_lane(bytes); b.buffer_desc_dyn(p, v) };
    let r_ws = b.buffer_desc(ptr[0], (e * WS_ROW as i64) as u32);
    let sz_maxp = b.bind_i32(ev(maxp_v) << 2);
    let r_sid = dsc(b, ptr[1], sz_maxp);
    let r_sw = dsc(b, ptr[2], sz_maxp);
    let sz_t = b.bind_i32(ev(ntiles_max) << 2);
    let r_eid = dsc(b, ptr[3], sz_t);
    let r_nv = b.buffer_desc(ptr[4], 8);
    let r_ts = b.buffer_desc(ptr[5], (4 * (e + 1)) as u32);
    let oob = b.c_i32(0x7fff_ff00);
    let tid4 = b.bind_i32(ev(tid) << 2);
    let in_e = b.cmp(CmpPred::Lt, CmpTy::I32, tid, e as i32);
    // 1. this expert's 64 per-block counts (16 x b128, all loads in flight), masked past G
    let row = b.bind_i32(ev(tid) * WS_ROW);
    let mut counts: Vec<ValueId> = Vec::new();
    let mut frags = Vec::new();
    for q in 0..16i32 {
        let o0 = b.bind_i32(ev(row) + 16 * q);
        let o = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(in_e), a: Opnd::Val(o0), b: Opnd::Val(oob) });
        frags.push(b.buffer_load_c(r_ws, Some(Opnd::Val(o)), 0i32, 0, MemTy::B128, CachePolicy::NONE));
    }
    let zero = b.c_i32(0);
    for (q, f) in frags.iter().enumerate() {
        for d in 0..4u8 {
            let w = b.frag4_lane(Opnd::Val(*f), d);
            let wi = b.bitcast_i32(w);
            let gi = (q * 4 + d as usize) as i32;
            let ok = b.cmp(CmpPred::Lt, CmpTy::I32, g_v, gi + 1);      // gi < G  <=>  G >= gi + 1 ; written as !(G < gi + 1)
            let live = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(ok), a: Opnd::Val(zero), b: Opnd::Val(wi) });
            counts.push(live);
        }
    }
    let mut cnt = counts[0];
    for c in &counts[1..] {
        cnt = b.add_i32(cnt, *c);
    }
    let a_cnt = b.bind_i32(ev(tid4) + offa);
    t_a.at(&SlotIdx::new(0, 0)).write(b, |b| b.store_lds(a_cnt, cnt, MemTy::B32, false));
    // 2. padded counts and the inclusive scan over experts (Hillis-Steele in LDS)
    let padded0 = b.bind_i32(((ev(cnt) + (bm - 1) as i32) >> (bm.trailing_zeros() as i32)) << (bm.trailing_zeros() as i32));
    let padded = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(in_e), a: Opnd::Val(padded0), b: Opnd::I32(0) });
    let a_s0 = b.bind_i32(ev(tid4) + offs0);
    t_s0.at(&SlotIdx::new(0, 0)).write(b, |b| b.store_lds(a_s0, padded, MemTy::B32, false));
    b.wait_lds(0);
    b.barrier();
    let mut cur = offs0;
    let mut s = 1i32;
    let mut ver = 0i32;   // the version of the tile `cur`
    while s < 512 {
        let nxt = if cur == offs0 { offs1 } else { offs0 };
        let (cur_t, nxt_t) = if cur == offs0 { (&t_s0, &t_s1) } else { (&t_s1, &t_s0) };
        let a_own = b.bind_i32(ev(tid4) + cur);
        let own = cur_t.at(&SlotIdx::new(0, ver)).read(b, |b| b.load_lds(a_own, MemTy::B32));
        let prev_i = b.bind_i32(ev(tid) - s);
        let ok = b.cmp(CmpPred::Ge, CmpTy::I32, prev_i, 0);
        // (a max, not a select: provably inside the scan tile; masked by `ok` below)
        let pi = b.bind_i32(ev(prev_i).max(0));
        let a_pv = b.bind_i32((ev(pi) << 2) + cur);
        let pv = cur_t.at(&SlotIdx::new(0, ver)).read(b, |b| b.load_lds(a_pv, MemTy::B32));
        b.wait_lds(0);
        let add = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(ok), a: Opnd::Val(pv), b: Opnd::I32(0) });
        let sum = b.add_i32(own, add);
        let a_nx = b.bind_i32(ev(tid4) + nxt);
        nxt_t.at(&SlotIdx::new(0, ver + 1)).write(b, |b| b.store_lds(a_nx, sum, MemTy::B32, false));
        b.wait_lds(0);
        b.barrier();
        cur = nxt;
        ver += 1;
        s *= 2;
    }
    let cur_t = if cur == offs0 { &t_s0 } else { &t_s1 };
    let a_incl = b.bind_i32(ev(tid4) + cur);
    let incl = cur_t.at(&SlotIdx::new(0, ver)).read(b, |b| b.load_lds(a_incl, MemTy::B32));
    let a_tot = b.c_i32(cur + 4 * (e as i32 - 1));
    let total = cur_t.at(&SlotIdx::new(0, ver)).read(b, |b| b.load_lds(a_tot, MemTy::B32));
    b.wait_lds(0);
    let off_e = b.bind_i32(ev(incl) - ev(padded));
    let a_c = b.bind_i32(ev(tid4) + offc);
    t_c.at(&SlotIdx::new(0, 0)).write(b, |b| b.store_lds(a_c, off_e, MemTy::B32, false));
    // 3. ws[e][g] = off_e + the pairs of the earlier blocks (exclusive prefix over g), 16 x b128 stores
    let mut run = off_e;
    for q in 0..16usize {
        let mut ws4 = [0; 4];
        for d in 0..4usize {
            ws4[d] = run;
            run = b.add_i32(run, counts[q * 4 + d]);
        }
        let f = b.frag4(ws4[0], ws4[1], ws4[2], ws4[3]);
        let o0 = b.bind_i32(ev(row) + 16 * q as i32);
        let o = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(in_e), a: Opnd::Val(o0), b: Opnd::Val(oob) });
        b.buffer_store_c(r_ws, Some(Opnd::Val(o)), 0i32, 0, f, MemTy::B128, false, CachePolicy::NONE);
    }
    // 4. tile_start, num_valid_ids, padding entries, tail, expert ids (as in the single-block kernel)
    let bsh = bm.trailing_zeros() as i32;
    let tile0 = b.bind_i32(ev(off_e).lshr(bsh));
    let ts_o0 = b.bind_i32(ev(tid4));
    let ts_o = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(in_e), a: Opnd::Val(ts_o0), b: Opnd::Val(oob) });
    let tile0_f = b.bitcast_f32(tile0);
    b.buffer_store_f32(r_ts, ts_o, 0i32, 0, tile0_f);
    let is_e = b.cmp(CmpPred::Eq, CmpTy::I32, tid, e as i32);
    let tot_tiles = b.bind_i32(ev(total).lshr(bsh));
    let te_o = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(is_e), a: Opnd::I32(4 * e as i32), b: Opnd::Val(oob) });
    let tt_f = b.bitcast_f32(tot_tiles);
    b.buffer_store_f32(r_ts, te_o, 0i32, 0, tt_f);
    let is0 = b.cmp(CmpPred::Eq, CmpTy::I32, tid, 0);
    let nv_o = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(is0), a: Opnd::I32(0), b: Opnd::Val(oob) });
    let total_f = b.bitcast_f32(total);
    let m_f = b.bitcast_f32(m_v);
    b.buffer_store_f32(r_nv, nv_o, 0i32, 0, total_f);
    b.buffer_store_f32(r_nv, nv_o, 0i32, 4, m_f);
    let sent = b.bind_i32(((k as i32) << 24) + ev(m_v));
    let fzero = b.c_f32(0.0);
    let sent_f = b.bitcast_f32(sent);
    let pad_iters = ((e * bm) as i32 + T - 1) / T;
    // the exclusive offsets (C) are read by OTHER threads in the padding loop: their stores must complete before the barrier publishes them (the tile protocol found the missing wait: a barrier alone does
    // not wait for the DS write)
    b.wait_lds(0);
    b.barrier();
    b.loop_(&[(DType::I32, Opnd::I32(0))], |b, vars| {
        let it = vars[0];
        let done = b.let_(Op::CmpGeI32(Opnd::Var(it), Opnd::I32(pad_iters)));
        b.break_if(done);
        let q = b.let_(Op::MulI32(Opnd::Var(it), Opnd::I32(T)));
        let q = b.bind_i32(ev(q) + ev(tid));
        let ex = b.bind_i32(ev(q).lshr(bsh));
        let sl = b.bind_i32(ev(q) & (bm as i32 - 1));
        let exc1 = b.let_(Op::MinI32(Opnd::Val(ex), Opnd::I32(1023)));
        let exc = b.bin(BinOp::MaxI32, exc1, 0i32);
        let a_c = b.bind_i32((ev(exc) << 2) + offc);
        let a_n = b.bind_i32((ev(exc) << 2) + offa);
        let a_s = b.bind_i32((ev(exc) << 2) + cur);
        let oe = t_c.at(&SlotIdx::new(0, 0)).read(b, |b| b.load_lds(a_c, MemTy::B32));
        let ce = t_a.at(&SlotIdx::new(0, 0)).read(b, |b| b.load_lds(a_n, MemTy::B32));
        let ie = cur_t.at(&SlotIdx::new(0, ver)).read(b, |b| b.load_lds(a_s, MemTy::B32));
        b.wait_lds(0);
        let pe = b.bind_i32(ev(ie) - ev(oe));
        let pos = b.bind_i32(ev(oe) + ev(ce) + ev(sl));
        let lim = b.bind_i32(ev(oe) + ev(pe));
        let in_ex = b.cmp(CmpPred::Lt, CmpTy::I32, ex, e as i32);
        let in_p = b.cmp(CmpPred::Lt, CmpTy::I32, pos, lim);
        let o0 = b.bind_i32(ev(pos) << 2);
        let o1 = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(in_ex), a: Opnd::Val(o0), b: Opnd::Val(oob) });
        let o = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(in_p), a: Opnd::Val(o1), b: Opnd::Val(oob) });
        b.buffer_store_f32(r_sid, o, 0i32, 0, sent_f);
        b.buffer_store_f32(r_sw, o, 0i32, 0, fzero);
        let nx = b.let_(Op::AddI32(Opnd::Var(it), Opnd::I32(1)));
        b.set_var(it, nx);
    });
    let span = b.bind_i32(ev(maxp_v) - ev(total));
    b.loop_(&[(DType::I32, Opnd::I32(0))], |b, vars| {
        let r = vars[0];
        let done = b.let_(Op::CmpGeI32(Opnd::Var(r), Opnd::Val(span)));
        b.break_if(done);
        let r0 = b.let_(Op::AddI32(Opnd::Var(r), Opnd::Val(tid)));
        let p = b.bind_i32(ev(total) + ev(r0));
        let ok = b.cmp(CmpPred::Lt, CmpTy::I32, p, maxp_v);
        let o0 = b.bind_i32(ev(p) << 2);
        let o = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(ok), a: Opnd::Val(o0), b: Opnd::Val(oob) });
        b.buffer_store_f32(r_sid, o, 0i32, 0, sent_f);
        b.buffer_store_f32(r_sw, o, 0i32, 0, fzero);
        let nx = b.let_(Op::AddI32(Opnd::Var(r), Opnd::I32(T)));
        b.set_var(r, nx);
    });
    b.loop_(&[(DType::I32, Opnd::I32(0))], |b, vars| {
        let r = vars[0];
        let done = b.let_(Op::CmpGeI32(Opnd::Var(r), Opnd::Val(ntiles_max)));
        b.break_if(done);
        let r0 = b.let_(Op::AddI32(Opnd::Var(r), Opnd::Val(tid)));
        let j = b.bind_i32(ev(r0));
        let x = b.bind_i32(ev(j) << bsh);
        let mut pos = b.c_i32(0);
        for step in [512, 256, 128, 64, 32, 16, 8, 4, 2, 1] {
            let cand = b.bind_i32(ev(pos) + step);
            let ca = b.bind_i32(((ev(cand) - 1) << 2) + cur);
            let v = cur_t.at(&SlotIdx::new(0, ver)).read(b, |b| b.load_lds(ca, MemTy::B32));
            b.wait_lds(0);
            let le = b.cmp(CmpPred::Le, CmpTy::I32, v, x);
            pos = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(le), a: Opnd::Val(cand), b: Opnd::Val(pos) });
        }
        let ex = b.let_(Op::MinI32(Opnd::Val(pos), Opnd::I32(e as i32)));
        let ex_f = b.bitcast_f32(ex);
        let ok = b.cmp(CmpPred::Lt, CmpTy::I32, j, ntiles_max);
        let o0 = b.bind_i32(ev(j) << 2);
        let o = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(ok), a: Opnd::Val(o0), b: Opnd::Val(oob) });
        b.buffer_store_f32(r_eid, o, 0i32, 0, ex_f);
        let nx = b.let_(Op::AddI32(Opnd::Var(r), Opnd::I32(T)));
        b.set_var(r, nx);
    });
    fb.finish()
}

pub fn moe_scatter(name: &str, e: i64, k: i64, bm: i64) -> Function {
    let mut fb = FnBuilder::new(name, T as u32);
    fb.compile_profile(CompileProfile::Default);
    let b = &mut fb;
    for (i, nm) in ["topk_ids", "topk_weights", "sorted_token_ids", "sorted_weights", "ws"].iter().enumerate() {
        b.arg(nm, DType::Ptr, 8 * i as u32, 8);
    }
    b.arg("m", DType::I32, 40, 4);
    b.arg("per", DType::I32, 44, 4);
    let mm = Ext::sym("M");
    b.sym_range("M", 1, 1 << 22);
    b.ka_is(40, mm.clone());
    let maxp = mm.clone() * Ext::c(k) + Ext::c(e * bm - k);
    b.arg_tensor("topk_ids", mm.clone() * Ext::c(4 * k));
    b.arg_tensor("topk_weights", mm.clone() * Ext::c(4 * k));
    b.arg_tensor("sorted_token_ids", maxp.clone() * Ext::c(4));
    b.arg_tensor("sorted_weights", maxp * Ext::c(4));
    b.arg_tensor("ws", Ext::c(e * WS_ROW as i64));
    b.lds(4096);
    let t_cur = tile_at(b, "CUR", 0, 4096);   // the per-expert cursors: initialised from ws, advanced by returning atomics (one version)
    let ptr: Vec<ValueId> = (0..5).map(|i| b.let_(Op::KernargPtr { offset: 8 * i })).collect();
    let m_v = b.let_(Op::KernargI32 { offset: 40 });
    let per = b.let_(Op::KernargI32 { offset: 44 });
    let tid = b.let_(Op::WorkitemId);
    let bx = b.let_(Op::WorkgroupIdX);
    let n_el = b.bind_i32(ev(m_v) * k as i32);
    let sz_ids = b.bind_i32(ev(n_el) << 2);
    let r_ids = { let v = b.read_first_lane(sz_ids); b.buffer_desc_dyn(ptr[0], v) };
    let r_w = { let v = b.read_first_lane(sz_ids); b.buffer_desc_dyn(ptr[1], v) };
    let maxp_v = b.bind_i32(ev(n_el) + (e * bm - k) as i32);
    let sz_maxp = b.bind_i32(ev(maxp_v) << 2);
    let r_sid = { let v = b.read_first_lane(sz_maxp); b.buffer_desc_dyn(ptr[2], v) };
    let r_sw = { let v = b.read_first_lane(sz_maxp); b.buffer_desc_dyn(ptr[3], v) };
    let r_ws = b.buffer_desc(ptr[4], (e * WS_ROW as i64) as u32);
    let oob = b.c_i32(0x7fff_ff00);
    let tid4 = b.bind_i32(ev(tid) << 2);
    // cursors: this block's first position of every expert
    let in_e = b.cmp(CmpPred::Lt, CmpTy::I32, tid, e as i32);
    let o0 = b.bind_i32(ev(tid) * WS_ROW + (ev(bx) << 2));
    let o = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(in_e), a: Opnd::Val(o0), b: Opnd::Val(oob) });
    let base_f = b.buffer_load_f32(r_ws, o, 0i32, 0);
    let base_i = b.bitcast_i32(base_f);
    t_cur.at(&SlotIdx::new(0, 0)).write(b, |b| b.store_lds(tid4, base_i, MemTy::B32, false));
    b.wait_lds(0);
    b.barrier();
    let magic = ((1u64 << 32) + k as u64 - 1) / k as u64;
    let base = b.bind_i32(ev(bx) * ev(per));
    let end0 = b.bind_i32(ev(base) + ev(per));
    let end = b.let_(Op::MinI32(Opnd::Val(end0), Opnd::Val(n_el)));
    let it_count = b.bind_i32((ev(per) + (4 * T - 1)).lshr(12));
    b.loop_(&[(DType::I32, Opnd::I32(0))], |b, vars| {
        let it = vars[0];
        let done = b.let_(Op::CmpGeI32(Opnd::Var(it), Opnd::Val(it_count)));
        b.break_if(done);
        let off = b.let_(Op::MulI32(Opnd::Var(it), Opnd::I32(4 * T)));
        let mut items = Vec::new();
        for u in 0..4 {
            let i = b.bind_i32(ev(base) + ev(off) + ev(tid) + u * T);
            let ins = b.let_(Op::CmpLtI32(Opnd::Val(i), Opnd::Val(end)));
            let eid = pair_expert(b, r_ids, i, ins, e);
            let wo = b.bind_i32(ev(i) << 2);
            let wf = b.buffer_load_f32(r_w, wo, 0i32, 0);
            let a = b.bind_i32(ev(eid) << 2);
            let pos = t_cur.at(&SlotIdx::new(0, 0)).write(b, |b| b.atomic_lds(AtomicOp::AddU32, a, 1i32, true)).unwrap();
            items.push((i, eid, wf, pos));
        }
        b.wait_lds(0);
        for (i, eid, wf, pos) in items {
            let q = if k == 1 {
                i
            } else {
                let mg = b.c_i32(magic as u32 as i32);
                b.bin(BinOp::MulHiU32, i, mg)
            };
            let slot = b.bind_i32(ev(i) - ev(q) * k as i32);
            let packed = b.bind_i32((ev(slot) << 24) + ev(q));
            let valid = b.cmp(CmpPred::Ne, CmpTy::I32, eid, 1023);
            let o0 = b.bind_i32(ev(pos) << 2);
            let o = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(valid), a: Opnd::Val(o0), b: Opnd::Val(oob) });
            let pf = b.bitcast_f32(packed);
            b.buffer_store_f32(r_sid, o, 0i32, 0, pf);
            b.buffer_store_f32(r_sw, o, 0i32, 0, wf);
        }
        let nx = b.let_(Op::AddI32(Opnd::Var(it), Opnd::I32(1)));
        b.set_var(it, nx);
    });
    fb.finish()
}

/// The stable (token-order) scatter, see the module comment.
pub fn moe_scatter_ordered(name: &str, e: i64, k: i64, bm: i64) -> Function {
    let mut fb = FnBuilder::new(name, T as u32);
    fb.compile_profile(CompileProfile::Default);
    let b = &mut fb;
    for (i, nm) in ["topk_ids", "topk_weights", "sorted_token_ids", "sorted_weights", "ws"].iter().enumerate() {
        b.arg(nm, DType::Ptr, 8 * i as u32, 8);
    }
    b.arg("m", DType::I32, 40, 4);
    b.arg("per", DType::I32, 44, 4);
    let mm = Ext::sym("M");
    b.sym_range("M", 1, 1 << 22);
    b.ka_is(40, mm.clone());
    let maxp = mm.clone() * Ext::c(k) + Ext::c(e * bm - k);
    b.arg_tensor("topk_ids", mm.clone() * Ext::c(4 * k));
    b.arg_tensor("topk_weights", mm.clone() * Ext::c(4 * k));
    b.arg_tensor("sorted_token_ids", maxp.clone() * Ext::c(4));
    b.arg_tensor("sorted_weights", maxp * Ext::c(4));
    b.arg_tensor("ws", Ext::c(e * WS_ROW as i64));
    // LDS: WC[16 waves][1024] (per-wave expert counts, then their exclusive prefix over the waves), CURSOR[1024], BASE[1024], one dummy dword
    // the dummy sink word is the (junk) column of the sentinel expert 1023 of the last wave row, inside WC: the select between a real count slot and the sink then stays inside one tile
    let (wc, cursor, base_a, dummy) = (0i32, 65536i32, 69632i32, 65532i32);
    b.lds(73736);
    // declared tiles: WC (version = the chunk index: zeroed, counted, prefix-summed in place by the owning thread, read by the positions), CURSOR (initialised once, advanced in place), BASE (rewritten every chunk, version = the chunk index)
    let t_wc = tile_at(b, "WC", wc as i64, 65536);
    let t_cur = tile_at(b, "CURSOR", cursor as i64, 4096);
    let t_base = tile_at(b, "BASE", base_a as i64, 4096);
    let ptr: Vec<ValueId> = (0..5).map(|i| b.let_(Op::KernargPtr { offset: 8 * i })).collect();
    let m_v = b.let_(Op::KernargI32 { offset: 40 });
    let per = b.let_(Op::KernargI32 { offset: 44 });
    let tid = b.let_(Op::WorkitemId);
    let bx = b.let_(Op::WorkgroupIdX);
    let wave = b.bind_i32(ev(tid).lshr(6));
    let lane = b.bind_i32(ev(tid) & 63);
    let n_el = b.bind_i32(ev(m_v) * k as i32);
    let sz_ids = b.bind_i32(ev(n_el) << 2);
    let r_ids = { let v = b.read_first_lane(sz_ids); b.buffer_desc_dyn(ptr[0], v) };
    let r_w = { let v = b.read_first_lane(sz_ids); b.buffer_desc_dyn(ptr[1], v) };
    let maxp_v = b.bind_i32(ev(n_el) + (e * bm - k) as i32);
    let sz_maxp = b.bind_i32(ev(maxp_v) << 2);
    let r_sid = { let v = b.read_first_lane(sz_maxp); b.buffer_desc_dyn(ptr[2], v) };
    let r_sw = { let v = b.read_first_lane(sz_maxp); b.buffer_desc_dyn(ptr[3], v) };
    let r_ws = b.buffer_desc(ptr[4], (e * WS_ROW as i64) as u32);
    let oob = b.c_i32(0x7fff_ff00);
    let tid4 = b.bind_i32(ev(tid) << 2);
    let in_e = b.cmp(CmpPred::Lt, CmpTy::I32, tid, e as i32);
    // the block's first position of every expert -> CURSOR
    let o0 = b.bind_i32(ev(tid) * WS_ROW + (ev(bx) << 2));
    let o = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(in_e), a: Opnd::Val(o0), b: Opnd::Val(oob) });
    let base_f = b.buffer_load_f32(r_ws, o, 0i32, 0);
    let base_i = b.bitcast_i32(base_f);
    let a_cur = b.bind_i32(ev(tid4) + cursor);
    t_cur.at(&SlotIdx::new(0, 0)).write(b, |b| b.store_lds(a_cur, base_i, MemTy::B32, false));
    b.wait_lds(0);
    b.barrier();
    let magic = ((1u64 << 32) + k as u64 - 1) / k as u64;
    let base = b.bind_i32(ev(bx) * ev(per));
    let end0 = b.bind_i32(ev(base) + ev(per));
    let end = b.let_(Op::MinI32(Opnd::Val(end0), Opnd::Val(n_el)));
    let nchunks = b.bind_i32((ev(per) + (T - 1)).lshr(10));      // chunks of 1024 pairs in this block's slice
    let wc_col = b.bind_i32(ev(tid4) + wc);                      // thread e's column start: WC[0][e]; WC[w][e] = + 4096 w
    b.loop_(&[(DType::I32, Opnd::I32(0))], |b, vars| {
        let ch = vars[0];
        let done = b.let_(Op::CmpGeI32(Opnd::Var(ch), Opnd::Val(nchunks)));
        b.break_if(done);
        let i = b.let_(Op::MulI32(Opnd::Var(ch), Opnd::I32(T)));
        let i = b.bind_i32(ev(base) + ev(i) + ev(tid));
        let ins = b.let_(Op::CmpLtI32(Opnd::Val(i), Opnd::Val(end)));
        let eid = pair_expert(b, r_ids, i, ins, e);
        let wo = b.bind_i32(ev(i) << 2);
        let wf = b.buffer_load_f32(r_w, wo, 0i32, 0);
        // WC versions: 2 ch (zeroed, counted), 2 ch + 1 (the prefix over the waves)
        let chv = b.let_(Op::AddI32(Opnd::Var(ch), Opnd::I32(0)));
        let v_cnt = b.bind_i32(ev(chv) << 1);
        let v_pre = b.bind_i32((ev(chv) << 1) + 1);
        // [A] zero this thread's column of WC (all 16 waves' counts of expert `tid`)
        let zero = b.c_i32(0);
        for w in 0..16i32 {
            let a = b.bind_i32(ev(wc_col) + 4096 * w);
            t_wc.at(&SlotIdx::new(0, Opnd::Val(v_cnt))).write(b, |b| b.store_lds(a, zero, MemTy::B32, false));
        }
        b.wait_lds(0);
        b.barrier();
        // [B] the wave's match loop: rank among the earlier lanes with the same expert, count per expert
        let tr = b.cmp(CmpPred::Ge, CmpTy::I32, tid, 0);             // always true: the full lane mask
        let (full_lo, full_hi) = b.ballot(tr);
        let res = b.loop_(&[(DType::I32, Opnd::Val(full_lo)), (DType::I32, Opnd::Val(full_hi)), (DType::I32, Opnd::I32(0))], |b, v2| {
            let (rl, rh, rk) = (v2[0], v2[1], v2[2]);
            let any = b.let_(Op::Bin(BinOp::OrB32, Opnd::Var(rl), Opnd::Var(rh)));
            let fin = b.cmp(CmpPred::Eq, CmpTy::I32, any, 0);
            b.break_if(fin);
            let rlv = b.let_(Op::AddI32(Opnd::Var(rl), Opnd::I32(0)));
            let rhv = b.let_(Op::AddI32(Opnd::Var(rh), Opnd::I32(0)));
            let lo_ne = b.cmp(CmpPred::Ne, CmpTy::I32, rlv, 0);
            let f_lo = b.un(amdgpu_dsl::UnOp::Ffbl, rlv);
            let f_hi = b.un(amdgpu_dsl::UnOp::Ffbl, rhv);
            let f_hi32 = b.add_i32(f_hi, 32);
            let l0 = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(lo_ne), a: Opnd::Val(f_lo), b: Opnd::Val(f_hi32) });
            let l0b = b.bind_i32(ev(l0) << 2);
            let e0 = b.bpermute(l0b, eid);
            let eq = b.cmp(CmpPred::Eq, CmpTy::I32, eid, e0);
            let (eq_lo, eq_hi) = b.ballot(eq);
            let mb = b.mbcnt(eq_lo, eq_hi);
            let ranked = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(eq), a: Opnd::Val(mb), b: Opnd::Var(rk) });
            let c0 = b.bin(BinOp::BcntAdd, eq_hi, 0i32);
            let cnt = b.bin(BinOp::BcntAdd, eq_lo, c0);
            // the lanes of the group write the wave's count of this expert (every lane of the group the same value); the others write a dummy slot
            let e0a = b.let_(Op::MinI32(Opnd::Val(e0), Opnd::I32(1023)));
            let e0c = b.bin(BinOp::MaxI32, e0a, 0i32);
            let wa0 = b.bind_i32(((ev(wave) << 10) + ev(e0c)) * 4 + wc);
            let dm = b.c_i32(dummy);
            let wa = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(eq), a: Opnd::Val(wa0), b: Opnd::Val(dm) });
            t_wc.at(&SlotIdx::new(0, Opnd::Val(v_cnt))).write(b, |b| b.store_lds(wa, cnt, MemTy::B32, false));
            let neq_lo = b.let_(Op::XorI32(Opnd::Val(eq_lo), Opnd::I32(-1)));
            let neq_hi = b.let_(Op::XorI32(Opnd::Val(eq_hi), Opnd::I32(-1)));
            let nl = b.let_(Op::AndI32(Opnd::Var(rl), Opnd::Val(neq_lo)));
            let nh = b.let_(Op::AndI32(Opnd::Var(rh), Opnd::Val(neq_hi)));
            b.set_var(rl, nl);
            b.set_var(rh, nh);
            b.set_var(rk, ranked);
            rk
        });
        let rank = b.let_(Op::AddI32(Opnd::Var(res), Opnd::I32(0)));
        b.wait_lds(0);
        b.barrier();
        // [C] one thread per expert: exclusive prefix of the wave counts over the 16 waves (in place), the expert's chunk base and the advanced cursor
        let mut cs = Vec::new();
        for w in 0..16i32 {
            let a = b.bind_i32(ev(wc_col) + 4096 * w);
            cs.push(t_wc.at(&SlotIdx::new(0, Opnd::Val(v_cnt))).read(b, |b| b.load_lds(a, MemTy::B32)));
        }
        let ca = b.bind_i32(ev(tid4) + cursor);
        let curv = t_cur.at(&SlotIdx::new(0, 0)).read(b, |b| b.load_lds(ca, MemTy::B32));
        b.wait_lds(0);
        // the prefix rewrites the column words just read (the column is the thread's own, but the protocol engine cannot see column ownership): the reads complete and a barrier separates them from the
        // in-place writes, which are a new version
        b.barrier();
        let mut run = b.c_i32(0);
        for w in 0..16i32 {
            let a = b.bind_i32(ev(wc_col) + 4096 * w);
            t_wc.at(&SlotIdx::new(0, Opnd::Val(v_pre))).write(b, |b| b.store_lds(a, run, MemTy::B32, false));
            run = b.add_i32(run, cs[w as usize]);
        }
        let ba = b.bind_i32(ev(tid4) + base_a);
        t_base.at(&SlotIdx::new(0, Opnd::Var(ch))).write(b, |b| b.store_lds(ba, curv, MemTy::B32, false));
        let adv = b.add_i32(curv, run);
        t_cur.at(&SlotIdx::new(0, 0)).write(b, |b| b.store_lds(ca, adv, MemTy::B32, false));
        b.wait_lds(0);
        b.barrier();
        // [D] position = the expert's chunk base + the wave's exclusive prefix + the rank in the wave
        let a_b = b.bind_i32((ev(eid) << 2) + base_a);
        let a_w = b.bind_i32(((ev(wave) << 10) + ev(eid)) * 4 + wc);
        let bse = t_base.at(&SlotIdx::new(0, Opnd::Var(ch))).read(b, |b| b.load_lds(a_b, MemTy::B32));
        let wpre = t_wc.at(&SlotIdx::new(0, Opnd::Val(v_pre))).read(b, |b| b.load_lds(a_w, MemTy::B32));
        b.wait_lds(0);
        let pos = b.bind_i32(ev(bse) + ev(wpre) + ev(rank));
        let q = if k == 1 {
            i
        } else {
            let mg = b.c_i32(magic as u32 as i32);
            b.bin(BinOp::MulHiU32, i, mg)
        };
        let slot = b.bind_i32(ev(i) - ev(q) * k as i32);
        let packed = b.bind_i32((ev(slot) << 24) + ev(q));
        let valid = b.cmp(CmpPred::Ne, CmpTy::I32, eid, 1023);
        let o0 = b.bind_i32(ev(pos) << 2);
        let o = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(valid), a: Opnd::Val(o0), b: Opnd::Val(oob) });
        let pf = b.bitcast_f32(packed);
        b.buffer_store_f32(r_sid, o, 0i32, 0, pf);
        b.buffer_store_f32(r_sw, o, 0i32, 0, wf);
        b.barrier();
        let nx = b.let_(Op::AddI32(Opnd::Var(ch), Opnd::I32(1)));
        b.set_var(ch, nx);
    });
    fb.finish()
}
