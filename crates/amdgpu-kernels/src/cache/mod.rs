//! KV-cache write (docs/op-coverage-memtemplates.md T7), the "flash" layout: `k, v [T][Hk][D]` bf16 are scattered to `key_cache, value_cache [S][Hk][D]` (S = num_blocks * block_size slots) at the slot given by
//! `slot_mapping [T]` (i32; a negative slot = padding token, nothing is written). One token per 64 / 128 / 192 / 256 lanes (a block holds 4 / 2 / 1 / 1 tokens) copy the row (`Hk * D * 2` bytes) with one b128 per lane per 16 bytes of the row.
//!
//! * `reshape_cache_flash_bf16_h<Hk>_d<D>`: ABI k @0, v @8, key_cache @16, value_cache @24, slot_mapping @32, T @40 (i32), S @44 (i32, slots of the cache). Caches of any size
//!   (up to 2^31 slots): the cache descriptors are moved to the token's slot row with a 64-bit `ptr_mad` (num_records = one row).

use amdgpu_dsl::{CachePolicy, MemTy, CompileProfile, DType, Ext, FnBuilder, Function, Op, Opnd, ValueId};
use amdgpu_tile::prims::{abs_f32, bf16x8_to_f32, fp8_pack, fp8_scale, group_reduce_dpp, lane_reduce, RedOp};
use amdgpu_tile::ev;

pub fn by_name(full: &str) -> Option<Function> {
    // `_i64`: slot_mapping is an int64 tensor (vLLM / SGLang) instead of int32: the low dword of every entry is read (slots < 2^31; -1 is 0xffffffff)
    let (name, slot64) = match full.strip_suffix("_i64") {
        Some(n) => (n, true),
        None => (full, false),
    };
    if let Some(rest) = name.strip_prefix("reshape_cache_blocked_fp8_h") {
        // per-token (and per-head) dynamic fp8: h<Hk>_d<D>_b<BS>
        let (h, rest) = rest.split_once("_d")?;
        let (d, bs) = rest.split_once("_b")?;
        let (h, d, bs): (i64, i64, i64) = (h.parse().ok()?, d.parse().ok()?, bs.parse().ok()?);
        return ((1..=256).contains(&h) && [64, 128, 256].contains(&d) && [16, 32, 64, 128].contains(&bs) && 2 * h * d <= 65536).then(|| reshape_cache_blocked_fp8(full, h, d, bs, slot64));
    }
    if let Some(rest) = name.strip_prefix("reshape_cache_blocked_bf16_h") {
        // h<Hk>_d<D>_b<BS>
        let (h, rest) = rest.split_once("_d")?;
        let (d, bs) = rest.split_once("_b")?;
        let (h, d, bs): (i64, i64, i64) = (h.parse().ok()?, d.parse().ok()?, bs.parse().ok()?);
        return ((1..=256).contains(&h) && [64, 128, 256].contains(&d) && [8, 16, 32, 64, 128].contains(&bs) && 2 * h * d <= 65536).then(|| reshape_cache_blocked(full, h, d, bs, slot64));
    }
    // `reshape_cache_flash_fp8_h<Hk>_d<D>`: the cache is fp8 (e4m3), the new K / V values are divided by the per-tensor scales k_scale / v_scale (kernargs) on the way in
    let (rest, fp8) = match name.strip_prefix("reshape_cache_flash_fp8_h") {
        Some(r) => (r, true),
        None => (name.strip_prefix("reshape_cache_flash_bf16_h")?, false),
    };
    let (h, d) = rest.split_once("_d")?;
    let (h, d): (i64, i64) = (h.parse().ok()?, d.parse().ok()?);
    let row = 2 * h * d;
    ((1..=256).contains(&h) && d % 8 == 0 && (8..=512).contains(&d) && row % 16 == 0 && row <= 65536).then(|| reshape_cache_flash(full, h, d, fp8, slot64))
}

pub fn reshape_cache_flash(name: &str, h: i64, d: i64, fp8: bool, slot64: bool) -> Function {
    let sw: i64 = if slot64 { 8 } else { 4 };            // bytes per slot_mapping entry
    let row = 2 * h * d;                    // bytes per token row (bf16 input)
    let orow = if fp8 { row / 2 } else { row };   // bytes per cache row
    let nv = row / 16;                      // b128 vectors per row
    let lt = (((nv + 63) / 64).min(4) * 64) as u32;      // lanes per token
    let tpb = if lt <= 128 { (256 / lt) as i64 } else { 1 };   // tokens per block (lt is 64 / 128 / 256 / 192: 4 / 2 / 1 / 1)
    let threads = lt * tpb as u32;
    let iters = (nv + lt as i64 - 1) / lt as i64;
    let mut fb = FnBuilder::new(name, threads);
    fb.compile_profile(CompileProfile::Default);
    let b = &mut fb;
    b.arg("k", DType::Ptr, 0, 8);
    b.arg("v", DType::Ptr, 8, 8);
    b.arg("key_cache", DType::Ptr, 16, 8);
    b.arg("value_cache", DType::Ptr, 24, 8);
    b.arg("slot_mapping", DType::Ptr, 32, 8);
    b.arg("t", DType::I32, 40, 4);
    b.arg("s", DType::I32, 44, 4);
    if fp8 {
        b.arg("k_scale", DType::F32, 48, 4);
        b.arg("v_scale", DType::F32, 52, 4);
    }
    let tt = Ext::sym("T");
    let ss = Ext::sym("S");
    b.sym_range("T", 1, 1 << 24);
    b.sym_range("S", 1, i32::MAX as i64); // > 4 GiB caches: the descriptors are moved per slot row / block (64-bit ptr_mad)
    b.ka_is(40, tt.clone());
    b.ka_is(44, ss.clone());
    b.grid_is(0, tt.clone().ceil_div(tpb));
    b.arg_tensor("k", tt.clone() * Ext::c(row));
    b.arg_tensor("v", tt.clone() * Ext::c(row));
    b.arg_tensor("key_cache", ss.clone() * Ext::c(orow));
    b.arg_tensor("value_cache", ss * Ext::c(orow));
    b.arg_tensor("slot_mapping", tt * Ext::c(sw));
    let ptrs: Vec<_> = (0..5).map(|i| b.let_(Op::KernargPtr { offset: 8 * i })).collect();
    let t_v = b.let_(Op::KernargI32 { offset: 40 });
    let s_v = b.let_(Op::KernargI32 { offset: 44 });
    let tid = b.let_(Op::WorkitemId);
    let bx = b.let_(Op::WorkgroupIdX);
    // token of this lane group and the lane within the token's row
    let (tok, lane) = if tpb == 1 {
        (bx, tid)
    } else {
        let sh = (lt as i64).trailing_zeros() as i32;
        let tk = b.bind_i32(ev(bx) * tpb as i32 + ev(tid).lshr(sh));
        let ln = b.bind_i32(ev(tid) & (lt as i32 - 1));
        (tk, ln)
    };
    let tok_ok = b.let_(Op::CmpLtI32(Opnd::Val(tok), Opnd::Val(t_v)));
    let r_k = { let v = b.bind_i32(ev(t_v) * row as i32); let v = b.read_first_lane(v); b.buffer_desc_dyn(ptrs[0], v) };
    let r_v = { let v = b.bind_i32(ev(t_v) * row as i32); let v = b.read_first_lane(v); b.buffer_desc_dyn(ptrs[1], v) };
    let r_sl = { let v = b.bind_i32(ev(t_v) * sw as i32); let v = b.read_first_lane(v); b.buffer_desc_dyn(ptrs[4], v) };
    let vslot = b.bind_i32(ev(tok) * sw as i32);
    let slotf = b.buffer_load_f32(r_sl, vslot, 0i32, 0);
    let src = b.bind_i32(ev(tok) * row as i32 + (ev(lane) << 4));
    // the data loads do not depend on the slot: they are issued right behind the slot load (their latency overlaps the slot's), the stores come after the descriptor setup; lanes past the row (and padding tokens) store out of range
    let mut loaded = Vec::new();
    for it in 0..iters {
        let so = b.bind_i32(ev(src) + (it * lt as i64 * 16) as i32);
        let kv = b.buffer_load_c(r_k, Some(Opnd::Val(so)), 0i32, 0, MemTy::B128, CachePolicy::NT);
        let vv = b.buffer_load_c(r_v, Some(Opnd::Val(so)), 0i32, 0, MemTy::B128, CachePolicy::NT);
        let lane_ok = b.let_(Op::CmpLtI32(Opnd::Val(lane), Opnd::I32((nv - it * lt as i64) as i32)));
        loaded.push((it, kv, vv, lane_ok));
    }
    let slot = b.bitcast_i32(slotf);
    let slot = b.read_first_lane(slot);
    // a valid slot is in [0, S); the cache descriptors cover just this slot's row, their bases moved by a 64-bit ptr_mad (caches > 4 GiB).
    // Invalid slots (padding < 0, or >= S) write nothing: their stores are masked, the descriptor uses the clamped slot.
    let slot = b.let_(Op::AddI32(Opnd::Val(slot), Opnd::I32(0))); // signed (read_first_lane yields u32)
    let ok = b.bind_cond((ev(slot).ge(0).select_i32(1, 0) & ev(slot).lt(ev(s_v)).select_i32(1, 0)).eq(1));
    let sl = { let v = b.bind_i32(ev(slot).max(0).min(ev(s_v) - 1)); b.read_first_lane(v) };
    let rowc = b.c_i32(orow as i32);
    let pkc = b.ptr_mad(ptrs[2], sl, rowc);
    let pvc = b.ptr_mad(ptrs[3], sl, rowc);
    let r_kc = b.buffer_desc(pkc, orow as u32);
    let r_vc = b.buffer_desc(pvc, orow as u32);
    let ob = if fp8 { 8 } else { 16 };      // output bytes per lane and vector
    let dst = b.let_(Op::MulI32(Opnd::Val(lane), Opnd::I32(ob))); // the lane's chunk inside the slot row (I32)
    let (inv_k, inv_v) = if fp8 {
        let (ks, vs) = (b.let_(Op::KernargF32 { offset: 48 }), b.let_(Op::KernargF32 { offset: 52 }));
        (Some(b.rcp_f32(ks)), Some(b.rcp_f32(vs)))
    } else {
        (None, None)
    };
    for (it, kv, vv, lane_ok) in loaded {
        let dof = b.bind_i32(ev(dst) + (it * lt as i64 * ob as i64) as i32);
        let o1 = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(ok), a: Opnd::Val(dof), b: Opnd::I32(-16) });
        let o2 = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(lane_ok), a: Opnd::Val(o1), b: Opnd::I32(-16) });
        let o = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(tok_ok), a: Opnd::Val(o2), b: Opnd::I32(-16) });
        if fp8 {
            for (src, inv, rc) in [(kv, inv_k.unwrap(), r_kc), (vv, inv_v.unwrap(), r_vc)] {
                let f = bf16x8_to_f32(b, src);
                let words = fp8_pack(b, &f, inv);
                let f2 = b.frag2(words[0], words[1]);
                b.buffer_store_c(rc, Some(Opnd::Val(o)), 0i32, 0, f2, MemTy::B64, false, CachePolicy::NT);
            }
        } else {
            b.buffer_store_c(r_kc, Some(Opnd::Val(o)), 0i32, 0, kv, MemTy::B128, false, CachePolicy::NT);
            b.buffer_store_c(r_vc, Some(Opnd::Val(o)), 0i32, 0, vv, MemTy::B128, false, CachePolicy::NT);
        }
    }
    fb.finish()
}

/// KV-cache write into the vLLM / aiter blocked layout: `k, v [T][Hk][D]` bf16 go to `key_cache [NB][Hk][D/8][BS][8]` and `value_cache [NB][Hk][D][BS]` (bf16) at the slot of `slot_mapping[t]`
/// (`block = slot / BS`, `off = slot % BS`; a negative slot = padding token). The key side moves one 16-byte chunk (8 elements) per lane; the value side is transposed (the token index is the fastest dimension), so
/// each element is a 16-bit store. D in {64, 128, 256}, BS in {8 .. 128} (powers of two), caches of any size (per-block descriptors, 64-bit bases).
///
/// * `reshape_cache_blocked_bf16_h<Hk>_d<D>_b<BS>`: ABI k @0, v @8, key_cache @16, value_cache @24, slot_mapping @32, T @40 (i32), S @44 (i32: slots = NB * BS).
pub fn reshape_cache_blocked(name: &str, h: i64, d: i64, bs: i64, slot64: bool) -> Function {
    let sw: i64 = if slot64 { 8 } else { 4 };            // bytes per slot_mapping entry
    let row = 2 * h * d;
    let nv = row / 16;                                   // 16-byte chunks per token row
    let lt = (((nv + 63) / 64).min(4) * 64) as u32;      // lanes per token
    let tpb = if lt <= 128 { (256 / lt) as i64 } else { 1 };
    let threads = lt * tpb as u32;
    let iters = (nv + lt as i64 - 1) / lt as i64;
    let jd = d / 8;                                      // chunks per head (a power of two)
    let jsh = jd.trailing_zeros() as i32;
    let bsh = bs.trailing_zeros() as i32;
    let mut fb = FnBuilder::new(name, threads);
    fb.compile_profile(CompileProfile::Default);
    let b = &mut fb;
    b.arg("k", DType::Ptr, 0, 8);
    b.arg("v", DType::Ptr, 8, 8);
    b.arg("key_cache", DType::Ptr, 16, 8);
    b.arg("value_cache", DType::Ptr, 24, 8);
    b.arg("slot_mapping", DType::Ptr, 32, 8);
    b.arg("t", DType::I32, 40, 4);
    b.arg("s", DType::I32, 44, 4);
    let tt = Ext::sym("T");
    // S = NB BS slots (the ABI: whole blocks); > 4 GiB caches: the descriptors are moved per block (64-bit ptr_mad)
    let ss = Ext::sym("NB") * Ext::c(bs);
    b.sym_range("T", 1, 1 << 24);
    b.sym_range("NB", 1, i32::MAX as i64 / bs);
    b.ka_is(40, tt.clone());
    b.ka_is(44, ss.clone());
    b.grid_is(0, tt.clone().ceil_div(tpb));
    b.arg_tensor("k", tt.clone() * Ext::c(row));
    b.arg_tensor("v", tt.clone() * Ext::c(row));
    b.arg_tensor("key_cache", ss.clone() * Ext::c(row));
    b.arg_tensor("value_cache", ss * Ext::c(row));
    b.arg_tensor("slot_mapping", tt * Ext::c(sw));
    let ptrs: Vec<_> = (0..5).map(|i| b.let_(Op::KernargPtr { offset: 8 * i })).collect();
    let t_v = b.let_(Op::KernargI32 { offset: 40 });
    let s_v = b.let_(Op::KernargI32 { offset: 44 });
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
    let r_k = { let v = b.bind_i32(ev(t_v) * row as i32); let v = b.read_first_lane(v); b.buffer_desc_dyn(ptrs[0], v) };
    let r_v = { let v = b.bind_i32(ev(t_v) * row as i32); let v = b.read_first_lane(v); b.buffer_desc_dyn(ptrs[1], v) };
    let r_sl = { let v = b.bind_i32(ev(t_v) * sw as i32); let v = b.read_first_lane(v); b.buffer_desc_dyn(ptrs[4], v) };
    let vslot = b.bind_i32(ev(tok) * sw as i32);
    let slotf = b.buffer_load_f32(r_sl, vslot, 0i32, 0);
    // the data loads do not depend on the slot: issued right behind the slot load
    let mut loaded = Vec::new();
    for it in 0..iters {
        let c = b.bind_i32(ev(lane) + (it * lt as i64) as i32);      // chunk index within the token row
        let so = b.bind_i32((ev(tok) * row as i32) + (ev(c) << 4));
        let kv = b.buffer_load_c(r_k, Some(Opnd::Val(so)), 0i32, 0, MemTy::B128, CachePolicy::NT);
        let vv = b.buffer_load_c(r_v, Some(Opnd::Val(so)), 0i32, 0, MemTy::B128, CachePolicy::NT);
        let c_ok = b.let_(Op::CmpLtI32(Opnd::Val(c), Opnd::I32(nv as i32)));
        loaded.push((c, kv, vv, c_ok));
    }
    let slot = b.bitcast_i32(slotf);
    let slot = b.read_first_lane(slot);
    // a valid slot is in [0, S); the cache descriptors cover just the slot's block (BS rows), their bases moved by a 64-bit ptr_mad (caches
    // > 4 GiB). Invalid slots write nothing (masked stores); the descriptor uses the clamped slot.
    let slot = b.let_(Op::AddI32(Opnd::Val(slot), Opnd::I32(0))); // signed (read_first_lane yields u32)
    let ok = b.bind_cond((ev(slot).ge(0).select_i32(1, 0) & ev(slot).lt(ev(s_v)).select_i32(1, 0)).eq(1));
    let sl = { let v = b.bind_i32(ev(slot).max(0).min(ev(s_v) - 1)); b.read_first_lane(v) };
    // the block, clamped to the last whole block (S = NB BS by the ABI: a no-op for valid slots; it lets the verifier prove the block descriptor in bounds)
    let blk = { let v = b.bind_i32(ev(sl).lshr(bsh).min(ev(s_v).lshr(bsh) - 1).max(0)); b.read_first_lane(v) };
    let off = b.bind_i32(ev(sl) & (bs as i32 - 1));
    let blk_bytes = (bs * row) as u32;
    let bb = b.c_i32(blk_bytes as i32);
    let pkc = b.ptr_mad(ptrs[2], blk, bb);
    let pvc = b.ptr_mad(ptrs[3], blk, bb);
    let r_kc = b.buffer_desc(pkc, blk_bytes);
    let r_vc = b.buffer_desc(pvc, blk_bytes);
    let oob = b.c_i32(0x7fff_ff00);
    for (c, kv, vv, c_ok) in loaded {
        let head = b.bind_i32(ev(c).lshr(jsh));
        let j = b.bind_i32(ev(c) & (jd as i32 - 1));
        // key: chunk (head, j) of the slot's block (the descriptor base), row `off`: (head * (D/8) + j) * BS * 16 + off * 16
        let kd = b.bind_i32(((ev(head) * jd as i32 + ev(j)) * (bs as i32 * 16)) + (ev(off) << 4));
        let kd = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(ok), a: Opnd::Val(kd), b: Opnd::Val(oob) });
        let kd = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(c_ok), a: Opnd::Val(kd), b: Opnd::Val(oob) });
        let kd = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(tok_ok), a: Opnd::Val(kd), b: Opnd::Val(oob) });
        b.buffer_store_c(r_kc, Some(Opnd::Val(kd)), 0i32, 0, kv, MemTy::B128, false, CachePolicy::NT);
        // value: element (head, 8 j + i) of the slot's block: (head * D + 8 j + i) * BS * 2 + off * 2
        let vbase = b.bind_i32(((ev(head) * d as i32 + (ev(j) << 3)) * (bs as i32 * 2)) + (ev(off) << 1));
        let vbase = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(ok), a: Opnd::Val(vbase), b: Opnd::Val(oob) });
        let vbase = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(c_ok), a: Opnd::Val(vbase), b: Opnd::Val(oob) });
        let vbase = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(tok_ok), a: Opnd::Val(vbase), b: Opnd::Val(oob) });
        for m in 0..4u8 {
            let w = b.frag4_lane(Opnd::Val(vv), m);
            for hi in [false, true] {
                // the sentinel is past every descriptor; the element offsets (imm <= 7 * BS * 2 < 4096) are added to it
                let imm = (2 * m as i64 + hi as i64) * bs * 2;
                b.buffer_store_c(r_vc, Some(Opnd::Val(vbase)), 0i32, imm as u32, w, MemTy::U16, hi, CachePolicy::NONE); // 2-byte partial stores must NOT be non-temporal (measured 8x slower: no write combining)
            }
        }
    }
    fb.finish()
}

/// Blocked-layout KV-cache write with per-token (and per-head) DYNAMIC fp8 quantisation (aiter `reshape_and_cache_with_pertoken_quant`, non-asm layout, x = 16): for every (token, head) the scale is
/// `max(amax, floor) / 448` over the head's D elements, `q = fp8(x / scale)`; key cache `[NB][Hk][D/16][BS][16]` and value cache `[NB][Hk][D][BS]` (fp8 bytes), scales `k_scales, v_scales f32 [Hk][S]`
/// (`S = NB * BS` slots, index `head * S + slot`). `reshape_cache_blocked_fp8_h<Hk>_d<D>_b<BS>[_i64]`: ABI k @0, v @8, key_cache @16, value_cache @24, k_scales @32, v_scales @40, slot_mapping @48,
/// T @56 (i32), S @60 (i32). A negative slot = padding. The key side is one 8-byte store per lane and head chunk, the value side 8 single-byte stores (the transposed layout; slow like the bf16 one).
pub fn reshape_cache_blocked_fp8(name: &str, h: i64, d: i64, bs: i64, slot64: bool) -> Function {
    let sw: i64 = if slot64 { 8 } else { 4 };
    let row = 2 * h * d;                                 // input bytes per token row (bf16)
    let nv = row / 16;                                   // 16-byte chunks (8 elements) per token row
    let lt = (((nv + 63) / 64).min(4) * 64) as u32;      // lanes per token
    let tpb = if lt <= 128 { (256 / lt) as i64 } else { 1 };
    let threads = lt * tpb as u32;
    let iters = (nv + lt as i64 - 1) / lt as i64;
    let jd = d / 8;                                      // chunks per head (the lanes of a head: 8 / 16 / 32)
    let jsh = jd.trailing_zeros() as i32;
    let bsh = bs.trailing_zeros() as i32;
    let blk_bytes = (bs * h * d) as u32;                 // fp8 bytes of one block, per cache
    let mut fb = FnBuilder::new(name, threads);
    fb.compile_profile(CompileProfile::Default);
    let b = &mut fb;
    b.arg("k", DType::Ptr, 0, 8);
    b.arg("v", DType::Ptr, 8, 8);
    b.arg("key_cache", DType::Ptr, 16, 8);
    b.arg("value_cache", DType::Ptr, 24, 8);
    b.arg("k_scales", DType::Ptr, 32, 8);
    b.arg("v_scales", DType::Ptr, 40, 8);
    b.arg("slot_mapping", DType::Ptr, 48, 8);
    b.arg("t", DType::I32, 56, 4);
    b.arg("s", DType::I32, 60, 4);
    let tt = Ext::sym("T");
    let ss = Ext::sym("NB") * Ext::c(bs);
    b.sym_range("T", 1, 1 << 24);
    b.sym_range("NB", 1, (i32::MAX as i64 / 4 / h) / bs);
    b.ka_is(56, tt.clone());
    b.ka_is(60, ss.clone());
    b.grid_is(0, tt.clone().ceil_div(tpb));
    b.arg_tensor("k", tt.clone() * Ext::c(row));
    b.arg_tensor("v", tt.clone() * Ext::c(row));
    b.arg_tensor("key_cache", ss.clone() * Ext::c(h * d));
    b.arg_tensor("value_cache", ss.clone() * Ext::c(h * d));
    b.arg_tensor("k_scales", ss.clone() * Ext::c(4 * h));
    b.arg_tensor("v_scales", ss * Ext::c(4 * h));
    b.arg_tensor("slot_mapping", tt * Ext::c(sw));
    let ptrs: Vec<_> = (0..7).map(|i| b.let_(Op::KernargPtr { offset: 8 * i })).collect();
    let t_v = b.let_(Op::KernargI32 { offset: 56 });
    let s_v = b.let_(Op::KernargI32 { offset: 60 });
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
    let r_k = { let v = b.bind_i32(ev(t_v) * row as i32); let v = b.read_first_lane(v); b.buffer_desc_dyn(ptrs[0], v) };
    let r_v = { let v = b.bind_i32(ev(t_v) * row as i32); let v = b.read_first_lane(v); b.buffer_desc_dyn(ptrs[1], v) };
    let r_ks = { let v = b.bind_i32(ev(s_v) * (4 * h) as i32); let v = b.read_first_lane(v); b.buffer_desc_dyn(ptrs[4], v) };
    let r_vs = { let v = b.bind_i32(ev(s_v) * (4 * h) as i32); let v = b.read_first_lane(v); b.buffer_desc_dyn(ptrs[5], v) };
    let r_sl = { let v = b.bind_i32(ev(t_v) * sw as i32); let v = b.read_first_lane(v); b.buffer_desc_dyn(ptrs[6], v) };
    let vslot = b.bind_i32(ev(tok) * sw as i32);
    let slotf = b.buffer_load_f32(r_sl, vslot, 0i32, 0);
    let mut loaded = Vec::new();
    for it in 0..iters {
        let c = b.bind_i32(ev(lane) + (it * lt as i64) as i32);
        let so = b.bind_i32((ev(tok) * row as i32) + (ev(c) << 4));
        let kv = b.buffer_load_c(r_k, Some(Opnd::Val(so)), 0i32, 0, MemTy::B128, CachePolicy::NT);
        let vv = b.buffer_load_c(r_v, Some(Opnd::Val(so)), 0i32, 0, MemTy::B128, CachePolicy::NT);
        let c_ok = b.let_(Op::CmpLtI32(Opnd::Val(c), Opnd::I32(nv as i32)));
        loaded.push((c, kv, vv, c_ok));
    }
    let slot = b.bitcast_i32(slotf);
    let slot = b.read_first_lane(slot);
    let slot = b.let_(Op::AddI32(Opnd::Val(slot), Opnd::I32(0)));
    let ok = b.bind_cond((ev(slot).ge(0).select_i32(1, 0) & ev(slot).lt(ev(s_v)).select_i32(1, 0)).eq(1));
    let sl = { let v = b.bind_i32(ev(slot).max(0).min(ev(s_v) - 1)); b.read_first_lane(v) };
    let blk = { let v = b.bind_i32(ev(sl).lshr(bsh).min(ev(s_v).lshr(bsh) - 1).max(0)); b.read_first_lane(v) };
    let off = b.bind_i32(ev(sl) & (bs as i32 - 1));
    let bb = b.c_i32(blk_bytes as i32);
    let pkc = b.ptr_mad(ptrs[2], blk, bb);
    let pvc = b.ptr_mad(ptrs[3], blk, bb);
    let r_kc = b.buffer_desc(pkc, blk_bytes);
    let r_vc = b.buffer_desc(pvc, blk_bytes);
    let oob = b.c_i32(0x7fff_ff00);
    let zero = b.c_f32(0.0);
    let _ = zero;
    for (c, kv, vv, c_ok) in loaded {
        let head = b.bind_i32(ev(c).lshr(jsh));
        let j = b.bind_i32(ev(c) & (jd as i32 - 1));
        // the lanes of a head: the amax over its D elements, the scale, the quantised bytes
        let kx = bf16x8_to_f32(b, kv);
        let vx = bf16x8_to_f32(b, vv);
        let mut res: Vec<(ValueId, ValueId, ValueId)> = Vec::new();   // (scale, word0, word1) for k then v
        for x in [kx, vx] {
            let ab: Vec<ValueId> = x.iter().map(|&e| abs_f32(b, e)).collect();
            let m = lane_reduce(b, RedOp::Max, &ab);
            let amax = group_reduce_dpp(b, RedOp::Max, m, jd);
            let (sc, inv) = fp8_scale(b, amax);
            let words = fp8_pack(b, &x, inv);
            res.push((sc, words[0], words[1]));
        }
        let mask = |b: &mut FnBuilder, o: ValueId| -> ValueId {
            let o = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(ok), a: Opnd::Val(o), b: Opnd::Val(oob) });
            let o = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(c_ok), a: Opnd::Val(o), b: Opnd::Val(oob) });
            b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(tok_ok), a: Opnd::Val(o), b: Opnd::Val(oob) })
        };
        // key: element 8 j of the head -> x-group j / 2, byte 8 (j & 1) of its 16: ((head * (D/16) + j / 2) * BS + off) * 16 + 8 (j & 1)
        let kd = b.bind_i32(((ev(head) * (d / 16) as i32 + ev(j).lshr(1)) * bs as i32 + ev(off)) * 16 + ((ev(j) & 1) << 3));
        let kd = mask(b, kd);
        let f2 = b.frag2(res[0].1, res[0].2);
        b.buffer_store_c(r_kc, Some(Opnd::Val(kd)), 0i32, 0, f2, MemTy::B64, false, CachePolicy::NT);
        // value: byte (head, 8 j + i) of the block: (head * D + 8 j + i) * BS + off, one byte store each (the byte shifted down)
        let vb = b.bind_i32((ev(head) * d as i32 + (ev(j) << 3)) * bs as i32 + ev(off));
        let vb = mask(b, vb);
        for q in 0..8i64 {
            let w = if q < 4 { res[1].1 } else { res[1].2 };
            let sh = 8 * (q % 4) as i32;
            let byte = b.bind_i32(ev(w).lshr(sh));
            // the sentinel is past every descriptor; the element offsets (imm <= 7 * BS) are added to it
            b.buffer_store_c(r_vc, Some(Opnd::Val(vb)), 0i32, (q * bs) as u32, byte, MemTy::U8, false, CachePolicy::NONE);
        }
        // the scales: the head's first lane stores them (k_scales[head * S + slot], v_scales likewise)
        let lead = b.let_(Op::CmpLtI32(Opnd::Val(j), Opnd::I32(1)));
        let so = b.bind_i32((ev(head) * ev(s_v) + ev(sl)) << 2);
        let so = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(lead), a: Opnd::Val(so), b: Opnd::Val(oob) });
        let so = mask(b, so);
        b.buffer_store_f32(r_ks, so, 0i32, 0, res[0].0);
        b.buffer_store_f32(r_vs, so, 0i32, 0, res[1].0);
    }
    fb.finish()
}
