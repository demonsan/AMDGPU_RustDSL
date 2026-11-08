//! MoE alignment / sorting (docs/op-coverage-memtemplates.md T5 + T7), the "oneshot" single-block kernel: a counting sort of the routed (token, slot) pairs by expert, each expert's group padded to a multiple of
//! `BM` (the grouped GEMM's M tile), in aiter's `moe_sorting` convention. One 1024-thread block; LDS histogram with `ds_add_u32`, LDS prefix scan (Hillis-Steele), positions from `ds_add_rtn_u32`.
//!
//! `moe_sorting_e<E>_k<K>_b<BM>`  (E experts 8..=512, K top-k 1..=16, BM in {16, 32, 64, 128}; M tokens at run time, M * K <= 2^22 elements -- the single block loops over them).
//! ABI: topk_ids @0 (i32 [M][K]; ids outside 0..E are ignored), topk_weights @8 (f32 [M][K]), sorted_token_ids @16 (u32 [max_padded]), sorted_weights @24 (f32 [max_padded]), sorted_expert_ids @32
//! (i32 [max_padded / BM]), num_valid_ids @40 (i32 [2]), tile_start @48 (i32 [E + 1]), M @56 (i32).  `max_padded = M * K + E * BM - K`.
//!
//! Outputs: `sorted_token_ids[p] = (slot << 24) | token` grouped by expert (ascending), each group padded with `(K << 24) | M`; `sorted_weights[p]` the weight (0 on padding); `sorted_expert_ids[j]` the expert of tile
//! j (E past the used tiles); `num_valid_ids = [padded_total, M]`; `tile_start[e]` = first tile of expert e, `tile_start[E]` = padded_total / BM. Order inside an expert's group = atomic arrival order (a multiset).

use amdgpu_dsl::{AtomicOp, CmpPred, CmpTy, CompileProfile, DType, Ext, FnBuilder, Function, MemTy, Op, Opnd, ValueId};
use amdgpu_tile::ev;

pub fn by_name(name: &str) -> Option<Function> {
    let rest = name.strip_prefix("moe_sorting_e")?;
    let (e, rest) = rest.split_once("_k")?;
    let (k, bm) = rest.split_once("_b")?;
    let (e, k, bm): (i64, i64, i64) = (e.parse().ok()?, k.parse().ok()?, bm.parse().ok()?);
    ((8..=512).contains(&e) && (1..=16).contains(&k) && [16, 32, 64, 128].contains(&bm)).then(|| moe_sorting(name, e, k, bm))
}

const T: i32 = 1024; // threads

pub fn moe_sorting(name: &str, e: i64, k: i64, bm: i64) -> Function {
    let mut fb = FnBuilder::new(name, T as u32);
    fb.compile_profile(CompileProfile::Default);
    let b = &mut fb;
    for (i, nm) in ["topk_ids", "topk_weights", "sorted_token_ids", "sorted_weights", "sorted_expert_ids", "num_valid_ids", "tile_start"].iter().enumerate() {
        b.arg(nm, DType::Ptr, 8 * i as u32, 8);
    }
    b.arg("m", DType::I32, 56, 4);
    let mm = Ext::sym("M");
    b.sym_range("M", 1, 1 << 22);
    b.ka_is(56, mm.clone());
    b.grid_is(0, Ext::c(1));
    let maxp = mm.clone() * Ext::c(k) + Ext::c(e * bm - k);
    b.arg_tensor("topk_ids", mm.clone() * Ext::c(4 * k));
    b.arg_tensor("topk_weights", mm.clone() * Ext::c(4 * k));
    b.arg_tensor("sorted_token_ids", maxp.clone() * Ext::c(4));
    b.arg_tensor("sorted_weights", maxp.clone() * Ext::c(4));
    b.arg_tensor("sorted_expert_ids", maxp.ceil_div(bm) * Ext::c(4));
    b.arg_tensor("num_valid_ids", Ext::c(8));
    b.arg_tensor("tile_start", Ext::c(4 * (e + 1)));
    // LDS: A = counts / padded counts, S0 / S1 = scan ping-pong, C = cursors; 1024 dwords each (entries >= E stay 0, entry 1023 is the dummy target of masked atomics)
    b.lds(4 * 4 * 1024);
    let (offa, offs0, offs1, offc) = (0i32, 4096i32, 8192i32, 12288i32);
    let ptr: Vec<ValueId> = (0..7).map(|i| b.let_(Op::KernargPtr { offset: 8 * i })).collect();
    let m_v = b.let_(Op::KernargI32 { offset: 56 });
    let tid = b.let_(Op::WorkitemId);
    let n_el = b.bind_i32(ev(m_v) * k as i32);                              // routed pairs
    let maxp_v = b.bind_i32(ev(n_el) + (e * bm - k) as i32);
    let ntiles_max = b.bind_i32(ev(maxp_v).lshr(bm.trailing_zeros() as i32));
    let dsc = |b: &mut FnBuilder, p: ValueId, bytes: ValueId| { let v = b.read_first_lane(bytes); b.buffer_desc_dyn(p, v) };
    let sz_ids = b.bind_i32(ev(n_el) << 2);
    let r_ids = dsc(b, ptr[0], sz_ids);
    let r_w = dsc(b, ptr[1], sz_ids);
    let sz_maxp = b.bind_i32(ev(maxp_v) << 2);
    let r_sid = dsc(b, ptr[2], sz_maxp);
    let r_sw = dsc(b, ptr[3], sz_maxp);
    let sz_t = b.bind_i32(ev(ntiles_max) << 2);
    let r_eid = dsc(b, ptr[4], sz_t);
    let r_nv = b.buffer_desc(ptr[5], 8);
    let r_ts = b.buffer_desc(ptr[6], (4 * (e + 1)) as u32);
    let oob = b.c_i32(0x7fff_ff00);
    let tid4 = b.bind_i32(ev(tid) << 2);
    let zero = b.c_i32(0);
    // 0. zero the counts (and the dummy slot)
    let a_cnt = b.bind_i32(ev(tid4) + offa);
    b.store_lds(a_cnt, zero, MemTy::B32, false);
    b.wait_lds(0);
    b.barrier();
    // 1. count the routed pairs per expert
    let magic = ((1u64 << 32) + k as u64 - 1) / k as u64;                    // ceil(2^32 / K): floor(i * magic / 2^32) = i / K exactly for i < 2^28
    let elems_per_iter = T * 4;                                              // 4 elements per thread per iteration
    let it_count = b.bind_i32((ev(n_el) + (elems_per_iter - 1)).lshr(12));    // ceil(n / 4096)
    let load_ids = |b: &mut FnBuilder, i: ValueId| -> (ValueId, ValueId) {
        // (expert id or 1023 when invalid, valid flag)
        let inb = b.let_(Op::CmpLtI32(Opnd::Val(i), Opnd::Val(n_el)));
        let off = b.bind_i32(ev(i) << 2);
        let f = b.buffer_load_f32(r_ids, off, 0i32, 0);
        let id = b.bitcast_i32(f);
        let ge = b.cmp(CmpPred::Ge, CmpTy::I32, id, 0);
        let lt = b.cmp(CmpPred::Lt, CmpTy::I32, id, e as i32);
        let v1 = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(inb), a: Opnd::Val(id), b: Opnd::I32(1023) });
        let v2 = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(ge), a: Opnd::Val(v1), b: Opnd::I32(1023) });
        let v3 = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(lt), a: Opnd::Val(v2), b: Opnd::I32(1023) });
        (v3, inb)
    };
    b.loop_(&[(DType::I32, Opnd::I32(0))], |b, vars| {
        let it = vars[0];
        let done = b.let_(Op::CmpGeI32(Opnd::Var(it), Opnd::Val(it_count)));
        b.break_if(done);
        let base = b.let_(Op::MulI32(Opnd::Var(it), Opnd::I32(elems_per_iter)));
        let mut eids = Vec::new();
        for u in 0..4 {
            let i = b.bind_i32(ev(base) + ev(tid) + u * T);
            eids.push(load_ids(b, i).0);
        }
        for eid in eids {
            let a = b.bind_i32((ev(eid) << 2) + offa);
            b.atomic_lds(AtomicOp::AddU32, a, 1i32, false);
        }
        let nx = b.let_(Op::AddI32(Opnd::Var(it), Opnd::I32(1)));
        b.set_var(it, nx);
    });
    b.wait_lds(0);
    b.barrier();
    // 2. padded counts and their inclusive prefix scan (Hillis-Steele in LDS)
    let a_cnt2 = b.bind_i32(ev(tid4) + offa);
    let cnt = b.load_lds(a_cnt2, MemTy::B32);
    b.wait_lds(0);
    let in_e = b.cmp(CmpPred::Lt, CmpTy::I32, tid, e as i32);
    let padded0 = b.bind_i32(((ev(cnt) + (bm - 1) as i32) >> (bm.trailing_zeros() as i32)) << (bm.trailing_zeros() as i32));
    let padded = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(in_e), a: Opnd::Val(padded0), b: Opnd::I32(0) });
    let a_s0 = b.bind_i32(ev(tid4) + offs0);
    b.store_lds(a_s0, padded, MemTy::B32, false);
    b.wait_lds(0);
    b.barrier();
    let mut cur = offs0;
    let mut s = 1i32;
    while s < 512 {
        let nxt = if cur == offs0 { offs1 } else { offs0 };
        let a_own = b.bind_i32(ev(tid4) + cur);
        let own = b.load_lds(a_own, MemTy::B32);
        let prev_i = b.bind_i32(ev(tid) - s);
        let ok = b.cmp(CmpPred::Ge, CmpTy::I32, prev_i, 0);
        let pi = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(ok), a: Opnd::Val(prev_i), b: Opnd::I32(0) });
        let a_pv = b.bind_i32((ev(pi) << 2) + cur);
        let pv = b.load_lds(a_pv, MemTy::B32);
        b.wait_lds(0);
        let add = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(ok), a: Opnd::Val(pv), b: Opnd::I32(0) });
        let sum = b.add_i32(own, add);
        let a_nx = b.bind_i32(ev(tid4) + nxt);
        b.store_lds(a_nx, sum, MemTy::B32, false);
        b.wait_lds(0);
        b.barrier();
        cur = nxt;
        s *= 2;
    }
    let a_incl = b.bind_i32(ev(tid4) + cur);
    let incl = b.load_lds(a_incl, MemTy::B32);
    let a_tot = b.c_i32(cur + 4 * (e as i32 - 1));
    let total = b.load_lds(a_tot, MemTy::B32);
    b.wait_lds(0);
    let off_e = b.bind_i32(ev(incl) - ev(padded));                                     // this expert's first padded position
    let a_c = b.bind_i32(ev(tid4) + offc);
    b.store_lds(a_c, off_e, MemTy::B32, false);
    // outputs that need only the scan: tile_start, num_valid_ids
    let tile0 = b.bind_i32(ev(off_e).lshr(bm.trailing_zeros() as i32));
    let ts_off = b.bind_i32(ev(tid4));
    let ts_ok = b.cmp(CmpPred::Lt, CmpTy::I32, tid, e as i32);
    let ts_o = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(ts_ok), a: Opnd::Val(ts_off), b: Opnd::Val(oob) });
    let tile0_f = b.bitcast_f32(tile0);
    b.buffer_store_f32(r_ts, ts_o, 0i32, 0, tile0_f);
    let is_e = b.cmp(CmpPred::Eq, CmpTy::I32, tid, e as i32);
    let tot_tiles = b.bind_i32(ev(total).lshr(bm.trailing_zeros() as i32));
    let te_o = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(is_e), a: Opnd::I32(4 * e as i32), b: Opnd::Val(oob) });
    let tt_f = b.bitcast_f32(tot_tiles);
    b.buffer_store_f32(r_ts, te_o, 0i32, 0, tt_f);
    let is0 = b.cmp(CmpPred::Eq, CmpTy::I32, tid, 0);
    let nv_o = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(is0), a: Opnd::I32(0), b: Opnd::Val(oob) });
    let total_f = b.bitcast_f32(total);
    let m_f = b.bitcast_f32(m_v);
    b.buffer_store_f32(r_nv, nv_o, 0i32, 0, total_f);
    b.buffer_store_f32(r_nv, nv_o, 0i32, 4, m_f);
    // padding entries of this expert's group: positions off_e + cnt .. off_e + padded (< BM of them), and the tail past the used part
    let sent = b.bind_i32(((k as i32) << 24) + ev(m_v));
    let fzero = b.c_f32(0.0);
    let sent_f = b.bitcast_f32(sent);
    let pad_n = b.bind_i32(ev(padded) - ev(cnt));                                      // padded - cnt (0 for experts >= E since padded = 0 and cnt = dummy count: clamp)
    let pad_n = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(in_e), a: Opnd::Val(pad_n), b: Opnd::I32(0) });
    let start = b.bind_i32(ev(off_e) + ev(cnt));
    b.loop_(&[(DType::I32, Opnd::I32(0))], |b, vars| {
        let r = vars[0];
        let done = b.let_(Op::CmpGeI32(Opnd::Var(r), Opnd::Val(pad_n)));
        b.break_if(done);
        let p = b.let_(Op::AddI32(Opnd::Val(start), Opnd::Var(r)));
        let o = b.bind_i32(ev(p) << 2);
        b.buffer_store_f32(r_sid, o, 0i32, 0, sent_f);
        b.buffer_store_f32(r_sw, o, 0i32, 0, fzero);
        let nx = b.let_(Op::AddI32(Opnd::Var(r), Opnd::I32(1)));
        b.set_var(r, nx);
    });
    // tail: positions total .. max_padded, strided over the threads
    b.loop_(&[(DType::I32, Opnd::I32(0))], |b, vars| {
        let r = vars[0];
        let r0 = b.let_(Op::AddI32(Opnd::Var(r), Opnd::Val(tid)));
        let p = b.bind_i32(ev(total) + ev(r0));
        let done = b.let_(Op::CmpGeI32(Opnd::Val(p), Opnd::Val(maxp_v)));
        b.break_if(done);
        let o = b.bind_i32(ev(p) << 2);
        b.buffer_store_f32(r_sid, o, 0i32, 0, sent_f);
        b.buffer_store_f32(r_sw, o, 0i32, 0, fzero);
        let nx = b.let_(Op::AddI32(Opnd::Var(r), Opnd::I32(T)));
        b.set_var(r, nx);
    });
    // sorted_expert_ids: the tiles of this expert, and E for the tiles past the used part
    let tiles_e = b.bind_i32(ev(padded).lshr(bm.trailing_zeros() as i32));
    let eid_f = b.bitcast_f32(tid);
    b.loop_(&[(DType::I32, Opnd::I32(0))], |b, vars| {
        let r = vars[0];
        let done = b.let_(Op::CmpGeI32(Opnd::Var(r), Opnd::Val(tiles_e)));
        b.break_if(done);
        let j = b.let_(Op::AddI32(Opnd::Val(tile0), Opnd::Var(r)));
        let o = b.bind_i32(ev(j) << 2);
        b.buffer_store_f32(r_eid, o, 0i32, 0, eid_f);
        let nx = b.let_(Op::AddI32(Opnd::Var(r), Opnd::I32(1)));
        b.set_var(r, nx);
    });
    let e_const = b.c_i32(e as i32);
    let e_f = b.bitcast_f32(e_const);
    b.loop_(&[(DType::I32, Opnd::I32(0))], |b, vars| {
        let r = vars[0];
        let r0 = b.let_(Op::AddI32(Opnd::Var(r), Opnd::Val(tid)));
        let j = b.bind_i32(ev(tot_tiles) + ev(r0));
        let done = b.let_(Op::CmpGeI32(Opnd::Val(j), Opnd::Val(ntiles_max)));
        b.break_if(done);
        let o = b.bind_i32(ev(j) << 2);
        b.buffer_store_f32(r_eid, o, 0i32, 0, e_f);
        let nx = b.let_(Op::AddI32(Opnd::Var(r), Opnd::I32(T)));
        b.set_var(r, nx);
    });
    b.wait_lds(0);
    b.barrier();
    // 3. scatter: the position of a pair = its expert's cursor, advanced atomically
    b.loop_(&[(DType::I32, Opnd::I32(0))], |b, vars| {
        let it = vars[0];
        let done = b.let_(Op::CmpGeI32(Opnd::Var(it), Opnd::Val(it_count)));
        b.break_if(done);
        let base = b.let_(Op::MulI32(Opnd::Var(it), Opnd::I32(elems_per_iter)));
        let mut items = Vec::new();
        for u in 0..4 {
            let i = b.bind_i32(ev(base) + ev(tid) + u * T);
            let (eid, inb) = load_ids(b, i);
            let wo = b.bind_i32(ev(i) << 2);
            let wf = b.buffer_load_f32(r_w, wo, 0i32, 0);
            let a = b.bind_i32((ev(eid) << 2) + offc);
            let pos = b.atomic_lds(AtomicOp::AddU32, a, 1i32, true).unwrap();
            items.push((i, eid, inb, wf, pos));
        }
        b.wait_lds(0);
        for (i, eid, inb, wf, pos) in items {
            // token = i / K, slot = i - token * K
            let q = if k == 1 {
                i
            } else {
                let mg = b.c_i32(magic as u32 as i32);
                b.bin(amdgpu_dsl::BinOp::MulHiU32, i, mg)
            };
            let slot = b.bind_i32(ev(i) - ev(q) * k as i32);
            let packed = b.bind_i32((ev(slot) << 24) + ev(q));
            let valid = b.cmp(CmpPred::Ne, CmpTy::I32, eid, 1023);
            let o0 = b.bind_i32(ev(pos) << 2);
            let o1 = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(valid), a: Opnd::Val(o0), b: Opnd::Val(oob) });
            let o = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(inb), a: Opnd::Val(o1), b: Opnd::Val(oob) });
            let pf = b.bitcast_f32(packed);
            b.buffer_store_f32(r_sid, o, 0i32, 0, pf);
            b.buffer_store_f32(r_sw, o, 0i32, 0, wf);
        }
        let nx = b.let_(Op::AddI32(Opnd::Var(it), Opnd::I32(1)));
        b.set_var(it, nx);
    });
    fb.finish()
}

fn zero_plus(b: &mut FnBuilder, v: i32) -> ValueId {
    b.c_i32(v)
}