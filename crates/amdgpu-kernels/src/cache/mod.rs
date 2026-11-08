//! KV-cache write (docs/op-coverage-memtemplates.md T7), the "flash" layout: `k, v [T][Hk][D]` bf16 are scattered to `key_cache, value_cache [S][Hk][D]` (S = num_blocks * block_size slots) at the slot given by
//! `slot_mapping [T]` (i32; a negative slot = padding token, nothing is written). One token per 64 / 128 / 192 / 256 lanes (a block holds 4 / 2 / 1 / 1 tokens) copy the row (`Hk * D * 2` bytes) with one b128 per lane per 16 bytes of the row.
//!
//! * `reshape_cache_flash_bf16_h<Hk>_d<D>`: ABI k @0, v @8, key_cache @16, value_cache @24, slot_mapping @32, T @40 (i32), S @44 (i32, slots of the cache). Caches < 4 GiB (32-bit buffer offsets).

use amdgpu_dsl::{CachePolicy, MemTy, CompileProfile, DType, Ext, FnBuilder, Function, Op, Opnd};
use amdgpu_tile::ev;

pub fn by_name(name: &str) -> Option<Function> {
    if let Some(rest) = name.strip_prefix("reshape_cache_blocked_bf16_h") {
        // h<Hk>_d<D>_b<BS>
        let (h, rest) = rest.split_once("_d")?;
        let (d, bs) = rest.split_once("_b")?;
        let (h, d, bs): (i64, i64, i64) = (h.parse().ok()?, d.parse().ok()?, bs.parse().ok()?);
        return ((1..=256).contains(&h) && [64, 128, 256].contains(&d) && [8, 16, 32, 64, 128].contains(&bs) && 2 * h * d <= 65536).then(|| reshape_cache_blocked(name, h, d, bs));
    }
    let rest = name.strip_prefix("reshape_cache_flash_bf16_h")?;
    let (h, d) = rest.split_once("_d")?;
    let (h, d): (i64, i64) = (h.parse().ok()?, d.parse().ok()?);
    let row = 2 * h * d;
    ((1..=256).contains(&h) && d % 8 == 0 && (8..=512).contains(&d) && row % 16 == 0 && row <= 65536).then(|| reshape_cache_flash(name, h, d))
}

pub fn reshape_cache_flash(name: &str, h: i64, d: i64) -> Function {
    let row = 2 * h * d;                    // bytes per token row
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
    let tt = Ext::sym("T");
    let ss = Ext::sym("S");
    b.sym_range("T", 1, 1 << 24);
    b.sym_range("S", 1, 1 << 24);
    b.ka_is(40, tt.clone());
    b.ka_is(44, ss.clone());
    b.grid_is(0, tt.clone().ceil_div(tpb));
    b.arg_tensor("k", tt.clone() * Ext::c(row));
    b.arg_tensor("v", tt.clone() * Ext::c(row));
    b.arg_tensor("key_cache", ss.clone() * Ext::c(row));
    b.arg_tensor("value_cache", ss * Ext::c(row));
    b.arg_tensor("slot_mapping", tt * Ext::c(4));
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
    let r_kc = { let v = b.bind_i32(ev(s_v) * row as i32); let v = b.read_first_lane(v); b.buffer_desc_dyn(ptrs[2], v) };
    let r_vc = { let v = b.bind_i32(ev(s_v) * row as i32); let v = b.read_first_lane(v); b.buffer_desc_dyn(ptrs[3], v) };
    let r_sl = { let v = b.bind_i32(ev(t_v) << 2); let v = b.read_first_lane(v); b.buffer_desc_dyn(ptrs[4], v) };
    let vslot = b.bind_i32(ev(tok) << 2);
    let slotf = b.buffer_load_f32(r_sl, vslot, 0i32, 0);
    let slot = b.bitcast_i32(slotf);
    let slot = b.read_first_lane(slot);
    let ok = b.let_(Op::CmpGeI32(Opnd::Val(slot), Opnd::I32(0)));
    let src = b.bind_i32(ev(tok) * row as i32 + (ev(lane) << 4));
    let dst = b.bind_i32(ev(slot) * row as i32 + (ev(lane) << 4));
    // loads first (K and V of every iteration), then the stores; lanes past the row (and padding tokens) store out of range
    let mut loaded = Vec::new();
    for it in 0..iters {
        let so = b.bind_i32(ev(src) + (it * lt as i64 * 16) as i32);
        let kv = b.buffer_load_c(r_k, Some(Opnd::Val(so)), 0i32, 0, MemTy::B128, CachePolicy::NT);
        let vv = b.buffer_load_c(r_v, Some(Opnd::Val(so)), 0i32, 0, MemTy::B128, CachePolicy::NT);
        let lane_ok = b.let_(Op::CmpLtI32(Opnd::Val(lane), Opnd::I32((nv - it * lt as i64) as i32)));
        loaded.push((it, kv, vv, lane_ok));
    }
    for (it, kv, vv, lane_ok) in loaded {
        let dof = b.bind_i32(ev(dst) + (it * lt as i64 * 16) as i32);
        let o1 = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(ok), a: Opnd::Val(dof), b: Opnd::I32(-16) });
        let o2 = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(lane_ok), a: Opnd::Val(o1), b: Opnd::I32(-16) });
        let o = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(tok_ok), a: Opnd::Val(o2), b: Opnd::I32(-16) });
        b.buffer_store_c(r_kc, Some(Opnd::Val(o)), 0i32, 0, kv, MemTy::B128, false, CachePolicy::NT);
        b.buffer_store_c(r_vc, Some(Opnd::Val(o)), 0i32, 0, vv, MemTy::B128, false, CachePolicy::NT);
    }
    fb.finish()
}

/// KV-cache write into the vLLM / aiter blocked layout: `k, v [T][Hk][D]` bf16 go to `key_cache [NB][Hk][D/8][BS][8]` and `value_cache [NB][Hk][D][BS]` (bf16) at the slot of `slot_mapping[t]`
/// (`block = slot / BS`, `off = slot % BS`; a negative slot = padding token). The key side moves one 16-byte chunk (8 elements) per lane; the value side is transposed (the token index is the fastest dimension), so
/// each element is a 16-bit store. D in {64, 128, 256}, BS in {8 .. 128} (powers of two), caches < 4 GiB.
///
/// * `reshape_cache_blocked_bf16_h<Hk>_d<D>_b<BS>`: ABI k @0, v @8, key_cache @16, value_cache @24, slot_mapping @32, T @40 (i32), S @44 (i32: slots = NB * BS).
pub fn reshape_cache_blocked(name: &str, h: i64, d: i64, bs: i64) -> Function {
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
    let ss = Ext::sym("S");
    b.sym_range("T", 1, 1 << 24);
    b.sym_range("S", 1, 1 << 24);
    b.ka_is(40, tt.clone());
    b.ka_is(44, ss.clone());
    b.grid_is(0, tt.clone().ceil_div(tpb));
    b.arg_tensor("k", tt.clone() * Ext::c(row));
    b.arg_tensor("v", tt.clone() * Ext::c(row));
    b.arg_tensor("key_cache", ss.clone() * Ext::c(row));
    b.arg_tensor("value_cache", ss * Ext::c(row));
    b.arg_tensor("slot_mapping", tt * Ext::c(4));
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
    let r_kc = { let v = b.bind_i32(ev(s_v) * row as i32); let v = b.read_first_lane(v); b.buffer_desc_dyn(ptrs[2], v) };
    let r_vc = { let v = b.bind_i32(ev(s_v) * row as i32); let v = b.read_first_lane(v); b.buffer_desc_dyn(ptrs[3], v) };
    let r_sl = { let v = b.bind_i32(ev(t_v) << 2); let v = b.read_first_lane(v); b.buffer_desc_dyn(ptrs[4], v) };
    let vslot = b.bind_i32(ev(tok) << 2);
    let slotf = b.buffer_load_f32(r_sl, vslot, 0i32, 0);
    let slot = b.bitcast_i32(slotf);
    let slot = b.read_first_lane(slot);
    let ok = b.let_(Op::CmpGeI32(Opnd::Val(slot), Opnd::I32(0)));
    let blk = b.bind_i32(ev(slot).lshr(bsh));
    let off = b.bind_i32(ev(slot) & (bs as i32 - 1));
    let oob = b.c_i32(0x7fff_ff00);
    let mut loaded = Vec::new();
    for it in 0..iters {
        let c = b.bind_i32(ev(lane) + (it * lt as i64) as i32);      // chunk index within the token row
        let so = b.bind_i32((ev(tok) * row as i32) + (ev(c) << 4));
        let kv = b.buffer_load_c(r_k, Some(Opnd::Val(so)), 0i32, 0, MemTy::B128, CachePolicy::NT);
        let vv = b.buffer_load_c(r_v, Some(Opnd::Val(so)), 0i32, 0, MemTy::B128, CachePolicy::NT);
        let c_ok = b.let_(Op::CmpLtI32(Opnd::Val(c), Opnd::I32(nv as i32)));
        loaded.push((c, kv, vv, c_ok));
    }
    for (c, kv, vv, c_ok) in loaded {
        let head = b.bind_i32(ev(c).lshr(jsh));
        let j = b.bind_i32(ev(c) & (jd as i32 - 1));
        // key: chunk (head, j) of block blk, row `off`: ((blk * Hk + head) * (D/8) + j) * BS * 16 + off * 16
        let kd = b.bind_i32((((ev(blk) * h as i32 + ev(head)) * jd as i32 + ev(j)) * (bs as i32 * 16)) + (ev(off) << 4));
        let kd = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(ok), a: Opnd::Val(kd), b: Opnd::Val(oob) });
        let kd = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(c_ok), a: Opnd::Val(kd), b: Opnd::Val(oob) });
        let kd = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(tok_ok), a: Opnd::Val(kd), b: Opnd::Val(oob) });
        b.buffer_store_c(r_kc, Some(Opnd::Val(kd)), 0i32, 0, kv, MemTy::B128, false, CachePolicy::NONE);
        // value: element (head, 8 j + i): ((blk * Hk + head) * D + 8 j + i) * BS * 2 + off * 2
        let vbase = b.bind_i32((((ev(blk) * h as i32 + ev(head)) * d as i32 + (ev(j) << 3)) * (bs as i32 * 2)) + (ev(off) << 1));
        let vbase = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(ok), a: Opnd::Val(vbase), b: Opnd::Val(oob) });
        let vbase = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(c_ok), a: Opnd::Val(vbase), b: Opnd::Val(oob) });
        let vbase = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(tok_ok), a: Opnd::Val(vbase), b: Opnd::Val(oob) });
        for m in 0..4u8 {
            let w = b.frag4_lane(Opnd::Val(vv), m);
            for hi in [false, true] {
                // the sentinel is past every descriptor; the element offsets (imm <= 7 * BS * 2 < 4096) are added to it
                let imm = (2 * m as i64 + hi as i64) * bs * 2;
                b.buffer_store_c(r_vc, Some(Opnd::Val(vbase)), 0i32, imm as u32, w, MemTy::U16, hi, CachePolicy::NONE);
            }
        }
    }
    fb.finish()
}
