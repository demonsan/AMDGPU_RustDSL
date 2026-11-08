//! MLA KV-cache write (aiter `concat_and_cache_mla`, `concat_and_cache_mla_seg`): the compressed KV `kv_c [T][R]` and the rotary part `k_pe [T][P]` of every token are concatenated into one cache entry at
//! `slot_mapping[t]` (a negative slot = padding token, nothing is written; a slot >= S is dropped too).
//!
//! * `concat_cache_mla_{bf16,fp8}_r<R>_p<P>[_i64]`: entry layout `[NB][BS][R + P]` = `[S][R + P]` (the block size does not matter): `entry = [kv_c | k_pe]`.
//! * `concat_cache_mla_seg_{bf16,fp8}_r<R>_p<P>_b<BS>[_i64]`: segmented block layout `[NB][BS * (R + P)]`: a block holds the BS `kv_c` rows (the "nope" segment) first, then the BS `k_pe` rows.
//!
//! bf16: a copy; fp8 (e4m3): `x / scale` with the static per-tensor `scale` kernarg. `_i64`: slot_mapping is int64 (low dword read). R, P multiples of 8. Caches of any size (the descriptors are moved to the
//! token's entry / block with a 64-bit `ptr_mad`). ABI: kv_c @0, k_pe @8, kv_cache @16, slot_mapping @24, T @32 (i32), S @36 (i32 slots), [scale @40 (f32, fp8)].
//! A lane moves one 16-byte chunk (8 elements) of the entry; `ceil(chunks / 64)` waves per token (a block holds 4 tokens when that is 1 wave, 2 for 2, else 1).

use amdgpu_dsl::{CachePolicy, CompileProfile, DType, Ext, FnBuilder, Function, MemTy, Op, Opnd};
use amdgpu_tile::prims::{bf16x8_to_f32, fp8_pack};
use amdgpu_tile::ev;

pub fn by_name(full: &str) -> Option<Function> {
    let (name, slot64) = match full.strip_suffix("_i64") {
        Some(n) => (n, true),
        None => (full, false),
    };
    let (rest, seg) = if let Some(r) = name.strip_prefix("concat_cache_mla_seg_") { (r, true) } else { (name.strip_prefix("concat_cache_mla_")?, false) };
    let (fp8, rest) = if let Some(r) = rest.strip_prefix("fp8_r") { (true, r) } else if let Some(r) = rest.strip_prefix("bf16_r") { (false, r) } else { return None };
    let (r, rest) = rest.split_once("_p")?;
    let (p, bs) = if seg {
        let (p, bs) = rest.split_once("_b")?;
        (p, Some(bs))
    } else {
        (rest, None)
    };
    let (r, p): (i64, i64) = (r.parse().ok()?, p.parse().ok()?);
    let bs: i64 = match bs {
        Some(b) => b.parse().ok()?,
        None => 1,
    };
    (r % 8 == 0 && p % 8 == 0 && (8..=4096).contains(&r) && (8..=512).contains(&p) && (!seg || [8, 16, 32, 64, 128].contains(&bs))).then(|| concat_cache_mla(full, r, p, bs, seg, fp8, slot64))
}

pub fn concat_cache_mla(name: &str, r: i64, p: i64, bs: i64, seg: bool, fp8: bool, slot64: bool) -> Function {
    let sw: i64 = if slot64 { 8 } else { 4 };
    let ent = r + p;                                    // elements per entry
    let nv = ent / 8;                                   // 16-byte input chunks per entry
    let nvr = r / 8;                                    // of them from kv_c
    let lt = (((nv + 63) / 64).min(4) * 64) as u32;     // lanes per token
    let tpb = if lt <= 128 { (256 / lt) as i64 } else { 1 };
    let threads = lt * tpb as u32;
    let iters = (nv + lt as i64 - 1) / lt as i64;
    let esz: i64 = if fp8 { 1 } else { 2 };             // cache bytes per element
    let ob = 8 * esz;                                   // output bytes per chunk
    let entry_bytes = ent * esz;
    let block_bytes = bs * entry_bytes;
    let bsh = bs.trailing_zeros() as i32;
    let mut fb = FnBuilder::new(name, threads);
    fb.compile_profile(CompileProfile::Default);
    let b = &mut fb;
    b.arg("kv_c", DType::Ptr, 0, 8);
    b.arg("k_pe", DType::Ptr, 8, 8);
    b.arg("kv_cache", DType::Ptr, 16, 8);
    b.arg("slot_mapping", DType::Ptr, 24, 8);
    b.arg("t", DType::I32, 32, 4);
    b.arg("s", DType::I32, 36, 4);
    if fp8 {
        b.arg("scale", DType::F32, 40, 4);
    }
    let tt = Ext::sym("T");
    // S slots; for the segmented layout whole blocks (S = NB * BS)
    let ss = if seg { Ext::sym("NB") * Ext::c(bs) } else { Ext::sym("S") };
    b.sym_range("T", 1, 1 << 24);
    if seg {
        b.sym_range("NB", 1, i32::MAX as i64 / bs);
    } else {
        b.sym_range("S", 1, i32::MAX as i64);
    }
    b.ka_is(32, tt.clone());
    b.ka_is(36, ss.clone());
    b.grid_is(0, tt.clone().ceil_div(tpb));
    b.arg_tensor("kv_c", tt.clone() * Ext::c(2 * r));
    b.arg_tensor("k_pe", tt.clone() * Ext::c(2 * p));
    b.arg_tensor("kv_cache", ss * Ext::c(entry_bytes));
    b.arg_tensor("slot_mapping", tt * Ext::c(sw));
    let ptrs: Vec<_> = (0..4).map(|i| b.let_(Op::KernargPtr { offset: 8 * i })).collect();
    let t_v = b.let_(Op::KernargI32 { offset: 32 });
    let s_v = b.let_(Op::KernargI32 { offset: 36 });
    let tid = b.let_(Op::WorkitemId);
    let bx = b.let_(Op::WorkgroupIdX);
    let (tok, lane) = if tpb == 1 {
        (bx, tid)
    } else {
        let sh = (lt as i64).trailing_zeros() as i32;
        let tk = b.bind_i32(ev(bx) * tpb as i32 + ev(tid).lshr(sh));
        let ln = b.bind_i32(ev(tid) & (lt as i32 - 1));
        (tk, ln)
    };
    let tok_ok = b.let_(Op::CmpLtI32(Opnd::Val(tok), Opnd::Val(t_v)));
    let r_c = { let v = b.bind_i32(ev(t_v) * (2 * r) as i32); let v = b.read_first_lane(v); b.buffer_desc_dyn(ptrs[0], v) };
    let r_p = { let v = b.bind_i32(ev(t_v) * (2 * p) as i32); let v = b.read_first_lane(v); b.buffer_desc_dyn(ptrs[1], v) };
    let r_sl = { let v = b.bind_i32(ev(t_v) * sw as i32); let v = b.read_first_lane(v); b.buffer_desc_dyn(ptrs[3], v) };
    let vslot = b.bind_i32(ev(tok) * sw as i32);
    let slotf = b.buffer_load_f32(r_sl, vslot, 0i32, 0);
    // the input chunks: chunk c < R / 8 from kv_c, else from k_pe (both loads issued, the other one is out of range: reads zeros); all before the slot is consumed
    let oob = b.c_i32(0x7fff_ff00);
    let mut loaded = Vec::new();
    for it in 0..iters {
        let c = b.bind_i32(ev(lane) + (it * lt as i64) as i32);
        let from_c = b.cmp(amdgpu_dsl::CmpPred::Lt, amdgpu_dsl::CmpTy::I32, c, nvr as i32);
        let oc = b.bind_i32((ev(tok) * (2 * r) as i32) + (ev(c) << 4));
        let oc = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(from_c), a: Opnd::Val(oc), b: Opnd::Val(oob) });
        let op_ = b.bind_i32((ev(tok) * (2 * p) as i32) + ((ev(c) - nvr as i32) << 4));
        let op_ = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(from_c), a: Opnd::Val(oob), b: Opnd::Val(op_) });
        let fc = b.buffer_load_c(r_c, Some(Opnd::Val(oc)), 0i32, 0, MemTy::B128, CachePolicy::NT);
        let fp = b.buffer_load_c(r_p, Some(Opnd::Val(op_)), 0i32, 0, MemTy::B128, CachePolicy::NT);
        let c_ok = b.let_(Op::CmpLtI32(Opnd::Val(c), Opnd::I32(nv as i32)));
        loaded.push((c, from_c, fc, fp, c_ok));
    }
    let slot = b.bitcast_i32(slotf);
    let slot = b.read_first_lane(slot);
    let slot = b.let_(Op::AddI32(Opnd::Val(slot), Opnd::I32(0)));
    let ok = b.bind_cond((ev(slot).ge(0).select_i32(1, 0) & ev(slot).lt(ev(s_v)).select_i32(1, 0)).eq(1));
    let sl = { let v = b.bind_i32(ev(slot).max(0).min(ev(s_v) - 1)); b.read_first_lane(v) };
    // the cache descriptor covers this token's entry (flat layout) or block (segmented), its base moved with a 64-bit ptr_mad
    let (unit, unit_bytes) = if seg {
        let blk = { let v = b.bind_i32(ev(sl).lshr(bsh).min(ev(s_v).lshr(bsh) - 1).max(0)); b.read_first_lane(v) };
        (blk, block_bytes)
    } else {
        (sl, entry_bytes)
    };
    let off_in_blk = b.bind_i32(ev(sl) & (bs as i32 - 1));
    let ub = b.c_i32(unit_bytes as i32);
    let pc = b.ptr_mad(ptrs[2], unit, ub);
    let r_cache = b.buffer_desc(pc, unit_bytes as u32);
    let inv = if fp8 {
        let sc = b.let_(Op::KernargF32 { offset: 40 });
        Some(b.rcp_f32(sc))
    } else {
        None
    };
    for (c, from_c, fc, fp, c_ok) in loaded {
        // the chunk's offset inside the descriptor's unit
        let dst = if seg {
            // nope chunk: off * R + 8 c (elements); pe chunk: BS * R + off * P + 8 (c - R / 8)
            let n_el = b.bind_i32(ev(off_in_blk) * r as i32 + (ev(c) << 3));
            let p_el = b.bind_i32(ev(off_in_blk) * p as i32 + ((ev(c) - nvr as i32) << 3) + (bs * r) as i32);
            let el = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(from_c), a: Opnd::Val(n_el), b: Opnd::Val(p_el) });
            b.bind_i32(ev(el) * esz as i32)
        } else {
            b.bind_i32(ev(c) * ob as i32)
        };
        let o1 = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(ok), a: Opnd::Val(dst), b: Opnd::Val(oob) });
        let o2 = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(c_ok), a: Opnd::Val(o1), b: Opnd::Val(oob) });
        let o = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(tok_ok), a: Opnd::Val(o2), b: Opnd::Val(oob) });
        // the chunk (the other load read zeros: both are combined by the same select as the offsets)
        let frag = b.let_(Op::Select { ty: DType::Frag4, cond: Opnd::Val(from_c), a: Opnd::Val(fc), b: Opnd::Val(fp) });
        if fp8 {
            let x = bf16x8_to_f32(b, frag);
            let words = fp8_pack(b, &x, inv.unwrap());
            let f2 = b.frag2(words[0], words[1]);
            b.buffer_store_c(r_cache, Some(Opnd::Val(o)), 0i32, 0, f2, MemTy::B64, false, CachePolicy::NT);
        } else {
            b.buffer_store_c(r_cache, Some(Opnd::Val(o)), 0i32, 0, frag, MemTy::B128, false, CachePolicy::NT);
        }
    }
    fb.finish()
}