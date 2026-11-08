//! Causal 1-D convolution, the decode update (Mamba / gated-delta-net front end; aiter `causal_conv1d_update`, non-circular, one new token per sequence).
//!
//! `causal_conv1d_update_bf16_w<W>_s<SL>[_idx][_silu]`: for every sequence b and channel c: `x_vals = [state[SL - (W - 1) ..], x]`, `out = bias + sum_j weight[c][j] * x_vals[j]` (silu if `_silu`),
//! and the state window shifts left by one with x appended (`state[l] = state[l + 1]`, `state[SL - 1] = x`). W in 2..=4, SL in W - 1 ..= 8. A lane owns a channel; the state row of a channel is SL bf16 values.
//! ABI: x @0 (bf16 [B][dim]), weight @8 (bf16 [dim][W]), bias @16 (bf16 [dim]), state @24 (bf16 [SLOTS][dim][SL]), out @32 (bf16 [B][dim]); without `_idx`: dim @40, slots @44 (i32); with `_idx`:
//! state_indices @40 (i32 [B], the state slot of sequence b; a negative index = padding sequence, nothing is read or written), dim @48, slots @52. Grid: x = ceil(dim / 256) channel tiles, y = B (<= 65535).

use amdgpu_dsl::{CachePolicy, CmpPred, CmpTy, CompileProfile, DType, Ext, FnBuilder, Function, MemTy, Op, Opnd, ValueId};
use amdgpu_tile::prims::{act, Act};
use amdgpu_tile::ev;

pub fn by_name(name: &str) -> Option<Function> {
    let rest = name.strip_prefix("causal_conv1d_update_bf16_w")?;
    let (silu, rest) = match rest.strip_suffix("_silu") {
        Some(r) => (true, r),
        None => (false, rest),
    };
    let (idx, rest) = match rest.strip_suffix("_idx") {
        Some(r) => (true, r),
        None => (false, rest),
    };
    let (w, sl) = rest.split_once("_s")?;
    let (w, sl): (i64, i64) = (w.parse().ok()?, sl.parse().ok()?);
    ((2..=4).contains(&w) && (w - 1..=8).contains(&sl)).then(|| conv1d_update(name, w, sl, idx, silu))
}

fn bf16_lo_to_f32(b: &mut FnBuilder, v: ValueId) -> ValueId {
    let s = b.bind_i32(ev(v) << 16);
    b.bitcast_f32(s)
}

pub fn conv1d_update(name: &str, w: i64, sl: i64, idx: bool, silu: bool) -> Function {
    let mut fb = FnBuilder::new(name, 256);
    fb.compile_profile(CompileProfile::Default);
    let b = &mut fb;
    for (i, nm) in ["x", "weight", "bias", "state", "out"].iter().enumerate() {
        b.arg(nm, DType::Ptr, 8 * i as u32, 8);
    }
    let (idx_off, dim_off) = if idx {
        b.arg("state_indices", DType::Ptr, 40, 8);
        (Some(40u32), 48u32)
    } else {
        (None, 40u32)
    };
    b.arg("dim", DType::I32, dim_off, 4);
    b.arg("slots", DType::I32, dim_off + 4, 4);
    let dim = Ext::sym("DIM");
    let slots = Ext::sym("SLOTS");
    let bb = Ext::sym("B");
    b.sym_range("DIM", 1, 1 << 20);
    b.sym_range("SLOTS", 1, 1 << 20);
    b.sym_range("B", 1, 65535);
    b.ka_is(dim_off, dim.clone());
    b.ka_is(dim_off + 4, slots.clone());
    b.grid_is(0, dim.clone().ceil_div(256));
    b.grid_is(1, bb.clone());
    b.arg_tensor("x", bb.clone() * dim.clone() * Ext::c(2));
    b.arg_tensor("weight", dim.clone() * Ext::c(2 * w));
    b.arg_tensor("bias", dim.clone() * Ext::c(2));
    b.arg_tensor("state", slots.clone() * dim.clone() * Ext::c(2 * sl));
    b.arg_tensor("out", bb.clone() * dim.clone() * Ext::c(2));
    if idx {
        b.arg_tensor("state_indices", bb * Ext::c(4));
    }
    let ptr: Vec<ValueId> = (0..5).map(|i| b.let_(Op::KernargPtr { offset: 8 * i })).collect();
    let p_idx = idx_off.map(|o| b.let_(Op::KernargPtr { offset: o }));
    let dim_v = b.let_(Op::KernargI32 { offset: dim_off });
    let slots_v = b.let_(Op::KernargI32 { offset: dim_off + 4 });
    let tid = b.let_(Op::WorkitemId);
    let tile = b.let_(Op::WorkgroupIdX);
    let seq = b.let_(Op::WorkgroupIdY);
    let ch = b.bind_i32(ev(tile) * 256 + ev(tid));
    let ch_ok = b.let_(Op::CmpLtI32(Opnd::Val(ch), Opnd::Val(dim_v)));
    // the state slot of this sequence (padding: negative)
    let slot = if let Some(pi) = p_idx {
        let r_i = { let n = b.bind_i32(ev(seq) + 1); let n = b.bind_i32(ev(n) << 2); let n = b.read_first_lane(n); b.buffer_desc_dyn(pi, n) };
        let o = b.bind_i32(ev(seq) << 2);
        let f = b.buffer_load_f32(r_i, o, 0i32, 0);
        let v = b.bitcast_i32(f);
        b.read_first_lane(v)
    } else {
        b.let_(Op::AddI32(Opnd::Val(seq), Opnd::I32(0)))
    };
    let slot = b.let_(Op::AddI32(Opnd::Val(slot), Opnd::I32(0)));
    let slot_ok = b.cmp(CmpPred::Ge, CmpTy::I32, slot, 0);
    let dsz = |b: &mut FnBuilder, per: i64| { let v = b.bind_i32(ev(dim_v) * per as i32); b.read_first_lane(v) };
    let bsz = b.bind_i32(ev(dim_v) * 2);
    let r_x = { let n = b.bind_i32((ev(seq) + 1) * ev(bsz)); let n = b.read_first_lane(n); b.buffer_desc_dyn(ptr[0], n) };
    let r_out = { let n = b.bind_i32((ev(seq) + 1) * ev(bsz)); let n = b.read_first_lane(n); b.buffer_desc_dyn(ptr[4], n) };
    let r_w = { let n = dsz(b, 2 * w); b.buffer_desc_dyn(ptr[1], n) };
    let r_b = { let n = dsz(b, 2); b.buffer_desc_dyn(ptr[2], n) };
    let r_s = { let n = b.bind_i32(ev(slots_v) * ev(dim_v) * (2 * sl) as i32); let n = b.read_first_lane(n); b.buffer_desc_dyn(ptr[3], n) };
    let oob = b.c_i32(0x7fff_ff00);
    // loads: state row (SL u16), weights (W u16), bias, x
    let sbase = b.bind_i32((ev(slot) * ev(dim_v) + ev(ch)) * (2 * sl) as i32);
    let sbase_l = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(slot_ok), a: Opnd::Val(sbase), b: Opnd::Val(oob) });
    let mut st = Vec::new();
    for l in 0..sl {
        st.push(b.buffer_load_c(r_s, Some(Opnd::Val(sbase_l)), 0i32, (2 * l) as u32, MemTy::U16, CachePolicy::NONE));
    }
    let wbase = b.bind_i32(ev(ch) * (2 * w) as i32);
    let mut wv = Vec::new();
    for j in 0..w {
        wv.push(b.buffer_load_c(r_w, Some(Opnd::Val(wbase)), 0i32, (2 * j) as u32, MemTy::U16, CachePolicy::NONE));
    }
    let boff = b.bind_i32(ev(ch) << 1);
    let bias_raw = b.buffer_load_c(r_b, Some(Opnd::Val(boff)), 0i32, 0, MemTy::U16, CachePolicy::NONE);
    let xoff = b.bind_i32(ev(seq) * ev(bsz) + (ev(ch) << 1));
    let x_raw = b.buffer_load_c(r_x, Some(Opnd::Val(xoff)), 0i32, 0, MemTy::U16, CachePolicy::NONE);
    // x_vals = [state[SL - (W - 1) ..], x]
    let mut xv: Vec<ValueId> = Vec::new();
    for l in (sl - (w - 1))..sl {
        xv.push(bf16_lo_to_f32(b, st[l as usize]));
    }
    xv.push(bf16_lo_to_f32(b, x_raw));
    let mut acc = bf16_lo_to_f32(b, bias_raw);
    for j in 0..w as usize {
        let wf = bf16_lo_to_f32(b, wv[j]);
        let t = b.mul_f32(wf, xv[j]);
        acc = b.add_f32(acc, t);
    }
    if silu {
        acc = act(b, acc, Act::Silu);
    }
    // out (bf16 RNE of the f32), masked for padding sequences / channels past dim
    let pk = b.cvt_pk_bf16_f32(acc, acc);
    let ook = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(slot_ok), a: Opnd::Val(xoff), b: Opnd::Val(oob) });
    let ook = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(ch_ok), a: Opnd::Val(ook), b: Opnd::Val(oob) });
    b.buffer_store_c(r_out, Some(Opnd::Val(ook)), 0i32, 0, pk, MemTy::U16, false, CachePolicy::NONE);
    // the state shifts left by one, x appended
    let sst = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(ch_ok), a: Opnd::Val(sbase_l), b: Opnd::Val(oob) });
    for l in 0..sl {
        let v = if l + 1 < sl { st[(l + 1) as usize] } else { x_raw };
        b.buffer_store_c(r_s, Some(Opnd::Val(sst)), 0i32, (2 * l) as u32, v, MemTy::U16, false, CachePolicy::NONE);
    }
    fb.finish()
}