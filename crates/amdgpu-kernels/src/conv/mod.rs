//! Causal 1-D convolution, the decode update (Mamba / gated-delta-net front end; aiter `causal_conv1d_update`, non-circular, one new token per sequence).
//!
//! `causal_conv1d_update_bf16_w<W>_s<SL>[_idx][_silu]`: for every sequence b and channel c: `x_vals = [state[SL - (W - 1) ..], x]`, `out = bias + sum_j weight[c][j] * x_vals[j]` (silu if `_silu`),
//! and the state window shifts left by one with x appended (`state[l] = state[l + 1]`, `state[SL - 1] = x`). W in 2..=4, SL in W - 1 ..= 8. A lane owns a channel; the state row of a channel is SL bf16 values.
//! ABI: x @0 (bf16 [B][dim]), weight @8 (bf16 [dim][W]), bias @16 (bf16 [dim]), state @24 (bf16 [SLOTS][dim][SL]), out @32 (bf16 [B][dim]); without `_idx`: dim @40, slots @44 (i32); with `_idx`:
//! state_indices @40 (i32 [B], the state slot of sequence b; a negative index = padding sequence, nothing is read or written), dim @48, slots @52. Grid: x = ceil(dim / 256) channel tiles, y = B (<= 65535).

use amdgpu_dsl::{CachePolicy, CmpPred, CmpTy, CompileProfile, DType, Ext, FnBuilder, Function, MemTy, Op, Opnd, ValueId};
use amdgpu_tile::prims::{act, bf16x8_to_f32, Act};
use amdgpu_tile::ev;

pub fn by_name(name: &str) -> Option<Function> {
    if name.starts_with("causal_conv1d_fwd_") {
        return fwd_by_name(name);
    }
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

// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Prefill / varlen forward (aiter `causal_conv1d_fwd_split_qkv`; channel-last x [T][dim], W-1 = SL state values per channel).
//
// `causal_conv1d_fwd_bf16_w<W>[_silu]`: out [T][dim].   `causal_conv1d_fwd_split_bf16_w<W>[_silu]`: the feature range [0, k) goes to q_out [T][k], [k, 2k) to k_out [T][k], [2k, dim) to v_out [T][dim - 2k].
//   out[t][f] = bias[f] + sum_{i < W} weight[f][i] * x[t - (W - 1) + i][f]   (x before the sequence start: the initial state if the sequence has one, else 0), silu optional,
//   and after the sequence the state `conv_states[cache_idx][f][0 .. W-1)` = its last W-1 inputs (older initial-state values when the sequence is shorter than W-1).
// ABI: x @0 (bf16 [T][dim]), weight @8 (bf16 [dim][W]), bias @16 (bf16 [dim]), cu_seqlens @24 (i32 [B + 1]), cache_indices @32 (i32 [B], negative = skip the sequence), has_initial @40 (u8 [B]),
//   conv_states @48 (bf16 [slots][dim][W - 1]), then out @56 (or q @56, k @64, v @72), then dim (i32), [k_dim (i32)], ntok (i32), slots (i32).
// `_sched` suffix: grid (ceil(dim / 2048), NC, 1) and two more args (sched_seq @(slots + 8), sched_chunk @(slots + 16): i32 [NC] tables with the sequence and the chunk index of every non-empty chunk, built
// by the caller; nc @(slots + 24)): no block idles past a sequence end when the lengths are uneven. Without it:
// Grid: x = ceil(dim / 2048) feature tiles (a lane owns 8 consecutive features, dim a multiple of 8), y = B, z = ceil(max_seqlen / 8) token chunks (blocks past a sequence's end exit); a block runs 8
// tokens sequentially with the W-1 previous inputs in registers. All 8 loads of a block are issued before the first output (kept packed: a 16-token chunk needed more than 256 VGPRs).

const TM: i64 = 8;

pub fn fwd_by_name(name: &str) -> Option<Function> {
    let (rest, split) = if let Some(r) = name.strip_prefix("causal_conv1d_fwd_split_bf16_w") {
        (r, true)
    } else {
        (name.strip_prefix("causal_conv1d_fwd_bf16_w")?, false)
    };
    // `_sched`: a (sequence, chunk) schedule table instead of the (B, max_chunks) grid
    let (rest, sched) = match rest.strip_suffix("_sched") {
        Some(r) => (r, true),
        None => (rest, false),
    };
    let (rest, silu) = match rest.strip_suffix("_silu") {
        Some(r) => (r, true),
        None => (rest, false),
    };
    let w: i64 = rest.parse().ok()?;
    (2..=4).contains(&w).then(|| conv1d_fwd(name, w, split, silu, sched))
}

pub fn conv1d_fwd(name: &str, w: i64, split: bool, silu: bool, sched: bool) -> Function {
    let sl = w - 1;
    let mut fb = FnBuilder::new(name, 256);
    fb.compile_profile(CompileProfile::Default);
    let b = &mut fb;
    for (i, nm) in ["x", "weight", "bias", "cu_seqlens", "cache_indices", "has_initial", "conv_states"].iter().enumerate() {
        b.arg(nm, DType::Ptr, 8 * i as u32, 8);
    }
    let nout = if split { 3 } else { 1 };
    let outs = if split { ["q_out", "k_out", "v_out"] } else { ["out", "", ""] };
    for i in 0..nout {
        b.arg(outs[i], DType::Ptr, 56 + 8 * i as u32, 8);
    }
    let d_off = 56 + 8 * nout as u32;
    let (k_off, n_off) = if split { (Some(d_off + 4), d_off + 8) } else { (None, d_off + 4) };
    b.arg("dim", DType::I32, d_off, 4);
    if let Some(ko) = k_off {
        b.arg("k_dim", DType::I32, ko, 4);
    }
    b.arg("ntok", DType::I32, n_off, 4);
    b.arg("slots", DType::I32, n_off + 4, 4);
    if sched {
        b.arg("sched_seq", DType::Ptr, n_off + 8, 8);
        b.arg("sched_chunk", DType::Ptr, n_off + 16, 8);
        b.arg("nc", DType::I32, n_off + 24, 4);
    }
    let dim = Ext::sym("DIM");
    let ntok = Ext::sym("NTOK");
    let slots = Ext::sym("SLOTS");
    let nb = Ext::sym("B");
    let nz = Ext::sym("NZ");
    b.sym_range("DIM", 8, 1 << 20);
    b.sym_range("NTOK", 1, 1 << 26);
    b.sym_range("SLOTS", 1, 1 << 20);
    b.sym_range("B", 1, 65535);
    b.sym_range("NZ", 1, 65535);
    b.ka_is(d_off, dim.clone());
    b.ka_is(n_off, ntok.clone());
    b.ka_is(n_off + 4, slots.clone());
    b.grid_is(0, dim.clone().ceil_div(2048));
    if sched {
        // one grid y per scheduled (sequence, chunk): the sequence / chunk of every entry come from the two tables
        let ncs = Ext::sym("NC");
        b.sym_range("NC", 1, 1 << 24);
        b.ka_is(n_off + 24, ncs.clone());
        b.grid_is(1, ncs.clone());
        b.grid_is(2, Ext::c(1));
        b.arg_tensor("sched_seq", ncs.clone() * Ext::c(4));
        b.arg_tensor("sched_chunk", ncs * Ext::c(4));
    } else {
        b.grid_is(1, nb.clone());
        b.grid_is(2, nz);
    }
    b.arg_tensor("x", ntok.clone() * dim.clone() * Ext::c(2));
    b.arg_tensor("weight", dim.clone() * Ext::c(2 * w));
    b.arg_tensor("bias", dim.clone() * Ext::c(2));
    b.arg_tensor("cu_seqlens", (nb.clone() + Ext::c(1)) * Ext::c(4));
    b.arg_tensor("cache_indices", nb.clone() * Ext::c(4));
    b.arg_tensor("has_initial", nb);
    b.arg_tensor("conv_states", slots * dim.clone() * Ext::c(2 * sl));
    if split {
        for nm in outs {
            b.arg_tensor(nm, ntok.clone() * dim.clone() * Ext::c(2));
        }
    } else {
        b.arg_tensor("out", ntok * dim.clone() * Ext::c(2));
    }
    let ptr: Vec<ValueId> = (0..7 + nout).map(|i| b.let_(Op::KernargPtr { offset: 8 * i as u32 })).collect();
    let dim_v = b.let_(Op::KernargI32 { offset: d_off });
    let kdim_v = k_off.map(|o| b.let_(Op::KernargI32 { offset: o }));
    let ntok_v = b.let_(Op::KernargI32 { offset: n_off });
    let slots_v = b.let_(Op::KernargI32 { offset: n_off + 4 });
    let tid = b.let_(Op::WorkitemId);
    let tile = b.let_(Op::WorkgroupIdX);
    let (seq, chunk) = if sched {
        let flat = b.let_(Op::WorkgroupIdY);
        let (p_sq, p_ck) = (b.let_(Op::KernargPtr { offset: n_off + 8 }), b.let_(Op::KernargPtr { offset: n_off + 16 }));
        let nc_v = b.let_(Op::KernargI32 { offset: n_off + 24 });
        let nbs = b.bind_i32(ev(nc_v) << 2);
        let nbs = b.read_first_lane(nbs);
        let (r_sq, r_ck) = (b.buffer_desc_dyn(p_sq, nbs), b.buffer_desc_dyn(p_ck, nbs));
        let fo = b.bind_i32(ev(flat) << 2);
        let sf = b.buffer_load_f32(r_sq, fo, 0i32, 0);
        let cf = b.buffer_load_f32(r_ck, fo, 0i32, 0);
        let (s0, c0v) = (b.bitcast_i32(sf), b.bitcast_i32(cf));
        let (s1, c1) = (b.read_first_lane(s0), b.read_first_lane(c0v));
        (b.let_(Op::AddI32(Opnd::Val(s1), Opnd::I32(0))), b.let_(Op::AddI32(Opnd::Val(c1), Opnd::I32(0))))
    } else {
        (b.let_(Op::WorkgroupIdY), b.let_(Op::WorkgroupIdZ))
    };
    let feat = b.bind_i32((ev(tile) * 256 + ev(tid)) << 3);               // first feature of this lane
    let feat_ok = b.let_(Op::CmpLtI32(Opnd::Val(feat), Opnd::Val(dim_v)));
    let fok0 = feat_ok_i(b, feat_ok);
    let fok = fok0;
    // the sequence: start, length, state slot, has-initial (uniform)
    let one_scalar = |b: &mut FnBuilder, p: ValueId, n_bytes: ValueId, off: ValueId, ty: MemTy| -> ValueId {
        let r = b.buffer_desc_dyn(p, n_bytes);
        let v = b.buffer_load_c(r, Some(Opnd::Val(off)), 0i32, 0, ty, CachePolicy::NONE);
        let v = b.read_first_lane(v);
        b.let_(Op::AddI32(Opnd::Val(v), Opnd::I32(0)))
    };
    let nbytes_cu = b.bind_i32((ev(seq) + 2) << 2);
    let nbytes_cu = b.read_first_lane(nbytes_cu);
    let o_seq = b.bind_i32(ev(seq) << 2);
    let seq_start = one_scalar(b, ptr[3], nbytes_cu, o_seq, MemTy::B32);
    let o_seq1 = b.bind_i32((ev(seq) + 1) << 2);
    let seq_end = one_scalar(b, ptr[3], nbytes_cu, o_seq1, MemTy::B32);
    let seqlen = b.bind_i32(ev(seq_end) - ev(seq_start));
    let nbytes_ci = b.bind_i32((ev(seq) + 1) << 2);
    let nbytes_ci = b.read_first_lane(nbytes_ci);
    let cache_idx = one_scalar(b, ptr[4], nbytes_ci, o_seq, MemTy::B32);
    let nbytes_hi = b.bind_i32(ev(seq) + 1);
    let nbytes_hi = b.read_first_lane(nbytes_hi);
    let o_hi = b.bind_i32(ev(seq) + 0);
    let has_init = one_scalar(b, ptr[5], nbytes_hi, o_hi, MemTy::U8);
    let c0 = b.bind_i32(ev(chunk) * TM as i32);                        // first token of this chunk, relative to the sequence start
    let active = b.cmp(CmpPred::Lt, CmpTy::I32, c0, seqlen);
    let seq_ok = b.cmp(CmpPred::Ge, CmpTy::I32, cache_idx, 0);
    let act_i = sel_i(b, active);
    let fok = b.bind_i32(ev(fok0) & ev(act_i));                      // blocks past the end of their sequence touch no memory
    let oob = b.c_i32(0x7fff_ff00);
    let bytes_x = b.bind_i32(ev(ntok_v) * ev(dim_v) * 2);
    let bytes_x = b.read_first_lane(bytes_x);
    let r_x = b.buffer_desc_dyn(ptr[0], bytes_x);
    let r_out: Vec<ValueId> = (0..nout).map(|i| b.buffer_desc_dyn(ptr[7 + i], bytes_x)).collect();
    let wbytes = b.bind_i32(ev(dim_v) * (2 * w) as i32);
    let wbytes = b.read_first_lane(wbytes);
    let r_w = b.buffer_desc_dyn(ptr[1], wbytes);
    let bbytes = b.bind_i32(ev(dim_v) * 2);
    let bbytes = b.read_first_lane(bbytes);
    let r_b = b.buffer_desc_dyn(ptr[2], bbytes);
    let sbytes = b.bind_i32(ev(slots_v) * ev(dim_v) * (2 * sl) as i32);
    let sbytes = b.read_first_lane(sbytes);
    let r_s = b.buffer_desc_dyn(ptr[6], sbytes);
    // weights (8 features x W) and bias
    let wbase0 = b.bind_i32(ev(feat) * (2 * w) as i32);
    let fon = b.cmp(CmpPred::Ne, CmpTy::I32, fok, 0);
    let wbase = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(fon), a: Opnd::Val(wbase0), b: Opnd::Val(oob) });
    let mut wv: Vec<ValueId> = Vec::new();                               // wv[f * W + i]
    for c in 0..(8 * w * 2 / 16) {
        let o = b.bind_i32(ev(wbase) + 16 * c as i32);
        let fr = b.buffer_load_c(r_w, Some(Opnd::Val(o)), 0i32, 0, MemTy::B128, CachePolicy::NONE);
        wv.extend_from_slice(&bf16x8_to_f32(b, fr));
    }
    let bo0 = b.bind_i32(ev(feat) << 1);
    let bo = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(fon), a: Opnd::Val(bo0), b: Opnd::Val(oob) });
    let bias_fr = b.buffer_load_c(r_b, Some(Opnd::Val(bo)), 0i32, 0, MemTy::B128, CachePolicy::NONE);
    let bias = bf16x8_to_f32(b, bias_fr);
    // the initial state of the 8 features: [f][j] = element f * (W - 1) + j (zero when the sequence has none)
    let st_base = b.bind_i32((ev(cache_idx) * ev(dim_v) + ev(feat)) * (2 * sl) as i32);
    let sk_i = sel_i(b, seq_ok);
    let st_on = b.bind_i32(ev(sk_i) & ev(fok));
    let st_on_c = b.cmp(CmpPred::Ne, CmpTy::I32, st_on, 0);
    let st_off = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(st_on_c), a: Opnd::Val(st_base), b: Opnd::Val(oob) });
    let zero = b.c_f32(0.0);
    let hi_c = b.cmp(CmpPred::Ne, CmpTy::I32, has_init, 0);
    let mut st: Vec<ValueId> = Vec::new();
    for c in 0..(8 * sl * 2 / 16) {
        let o = b.bind_i32(ev(st_off) + 16 * c as i32);
        let fr = b.buffer_load_c(r_s, Some(Opnd::Val(o)), 0i32, 0, MemTy::B128, CachePolicy::NONE);
        for v in bf16x8_to_f32(b, fr) {
            st.push(b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(hi_c), a: Opnd::Val(v), b: Opnd::Val(zero) }));
        }
    }
    // the inputs: W - 1 previous (chunk 0: the initial state) and the TM of this chunk
    let tok0 = b.bind_i32(ev(seq_start) + ev(c0));
    let first = b.cmp(CmpPred::Eq, CmpTy::I32, chunk, 0);
    let xoff = |b: &mut FnBuilder, tok: ValueId, ok: ValueId| -> ValueId {
        let o = b.bind_i32(ev(tok) * ev(dim_v) * 2 + (ev(feat) << 1));
        b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(ok), a: Opnd::Val(o), b: Opnd::Val(oob) })
    };
    let lane_on = b.cmp(CmpPred::Ne, CmpTy::I32, fok, 0);
    let mut prev: Vec<[ValueId; 8]> = Vec::new();   // prev[i][f], oldest first
    for i in 0..sl {
        let p = b.bind_i32(ev(tok0) - (sl - i) as i32);
        let not_first = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(first), a: Opnd::I32(0), b: Opnd::I32(1) });
        let on = b.bind_i32(ev(not_first) & ev(fok));
        let on_c = b.cmp(CmpPred::Ne, CmpTy::I32, on, 0);
        let o = xoff(b, p, on_c);
        let fr = b.buffer_load_c(r_x, Some(Opnd::Val(o)), 0i32, 0, MemTy::B128, CachePolicy::NT);
        let xp = bf16x8_to_f32(b, fr);
        let mut a = [0; 8];
        for f in 0..8usize {
            let sv = st[f * sl as usize + i as usize];
            a[f] = b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(first), a: Opnd::Val(sv), b: Opnd::Val(xp[f]) });
        }
        prev.push(a);
    }
    let _ = lane_on;
    let mut newx: Vec<(ValueId, ValueId)> = Vec::new();   // (token valid, the packed bf16x8 input)
    for t in 0..TM {
        let tl = b.bind_i32(ev(c0) + t as i32);
        let valid = b.cmp(CmpPred::Lt, CmpTy::I32, tl, seqlen);
        let tk = b.bind_i32(ev(seq_start) + ev(tl));
        let sel_t1 = sel_i(b, valid);
        let on = b.bind_i32(ev(sel_t1) & ev(fok));
        let on_c = b.cmp(CmpPred::Ne, CmpTy::I32, on, 0);
        let o = xoff(b, tk, on_c);
        let fr = b.buffer_load_c(r_x, Some(Opnd::Val(o)), 0i32, 0, MemTy::B128, CachePolicy::NT);
        newx.push((valid, fr));
    }
    // outputs
    let q_ok = {
        let o = b.c_i32(0);
        let _ = o;
    };
    let _ = q_ok;
    let mut win = prev;
    for t in 0..TM as usize {
        let (valid, xfr) = (newx[t].0, newx[t].1);
        let xn = bf16x8_to_f32(b, xfr);
        let mut y = [0; 8];
        for f in 0..8usize {
            let mut acc = bias[f];
            for i in 0..sl as usize {
                let tm = b.mul_f32(wv[f * w as usize + i], win[i][f]);
                acc = b.add_f32(acc, tm);
            }
            let tm = b.mul_f32(wv[f * w as usize + sl as usize], xn[f]);
            acc = b.add_f32(acc, tm);
            if silu {
                acc = act(b, acc, Act::Silu);
            }
            y[f] = acc;
        }
        let packed = pack8(b, &y);
        let tl = b.bind_i32(ev(c0) + t as i32);
        let tk = b.bind_i32(ev(seq_start) + ev(tl));
        let sel_t2 = sel_i(b, valid);
        let on = b.bind_i32(ev(sel_t2) & ev(fok));
        let sq_t = sel_i(b, seq_ok);
        let on = b.bind_i32(ev(on) & ev(sq_t));
        let on_c = b.cmp(CmpPred::Ne, CmpTy::I32, on, 0);
        if split {
            // the feature range picks the output tensor (a group of 8 features never straddles a range: k_dim % 8 == 0)
            let kd = kdim_v.unwrap();
            let k2 = b.bind_i32(ev(kd) << 1);
            let in_q = b.cmp(CmpPred::Lt, CmpTy::I32, feat, kd);
            let in_k = b.cmp(CmpPred::Lt, CmpTy::I32, feat, k2);
            let vdim = b.bind_i32(ev(dim_v) - ev(k2));
            let _ = vdim;
            // the row pitch of q / k is k_dim, of v dim - 2 k_dim
            let pitch_q = kd;
            let pitch_v = b.bind_i32(ev(dim_v) - ev(k2));
            let f_in_k = b.bind_i32(ev(feat) - ev(kd));
            let f_in_v = b.bind_i32(ev(feat) - ev(k2));
            let oq = b.bind_i32(ev(tk) * ev(pitch_q) * 2 + (ev(feat) << 1));
            let ok_ = b.bind_i32(ev(tk) * ev(pitch_q) * 2 + (ev(f_in_k) << 1));
            let ov = b.bind_i32(ev(tk) * ev(pitch_v) * 2 + (ev(f_in_v) << 1));
            let sel_t3 = sel_i(b, in_q);
            let sel_q = b.bind_i32(ev(sel_t3) & ev(on));
            let kq = b.cmp(CmpPred::Ne, CmpTy::I32, sel_q, 0);
            let notq = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(in_q), a: Opnd::I32(0), b: Opnd::I32(1) });
            let sel_t4 = sel_i(b, in_k);
            let ink = b.bind_i32(ev(sel_t4) & ev(notq) & ev(on));
            let kk = b.cmp(CmpPred::Ne, CmpTy::I32, ink, 0);
            let notk = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(in_k), a: Opnd::I32(0), b: Opnd::I32(1) });
            let inv = b.bind_i32(ev(notk) & ev(on));
            let vv = b.cmp(CmpPred::Ne, CmpTy::I32, inv, 0);
            let oq = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(kq), a: Opnd::Val(oq), b: Opnd::Val(oob) });
            let ok_ = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(kk), a: Opnd::Val(ok_), b: Opnd::Val(oob) });
            let ov = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(vv), a: Opnd::Val(ov), b: Opnd::Val(oob) });
            b.buffer_store_c(r_out[0], Some(Opnd::Val(oq)), 0i32, 0, packed, MemTy::B128, false, CachePolicy::NT);
            b.buffer_store_c(r_out[1], Some(Opnd::Val(ok_)), 0i32, 0, packed, MemTy::B128, false, CachePolicy::NT);
            b.buffer_store_c(r_out[2], Some(Opnd::Val(ov)), 0i32, 0, packed, MemTy::B128, false, CachePolicy::NT);
        } else {
            let o = xoff(b, tk, on_c);
            b.buffer_store_c(r_out[0], Some(Opnd::Val(o)), 0i32, 0, packed, MemTy::B128, false, CachePolicy::NT);
        }
        // slide the window by the new input of every VALID token (the final window feeds nothing here: the state is written below from x directly)
        let mut nw: Vec<[ValueId; 8]> = Vec::new();
        for i in 0..sl as usize {
            let mut a = [0; 8];
            for f in 0..8usize {
                let nxt = if i + 1 < sl as usize { win[i + 1][f] } else { xn[f] };
                a[f] = b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(valid), a: Opnd::Val(nxt), b: Opnd::Val(win[i][f]) });
            }
            nw.push(a);
        }
        win = nw;
    }
    // the final state: written by the chunk that holds the last token (uniform): the last W-1 inputs of the sequence, older initial-state values when the sequence is shorter than W-1
    let last_chunk = b.bind_i32(ev(c0) + TM as i32);
    let is_last = b.cmp(CmpPred::Ge, CmpTy::I32, last_chunk, seqlen);
    b.if_(active, |b| {
        b.if_(is_last, |b| {
            let mut vals: Vec<ValueId> = Vec::new();    // [f][j] element f * sl + j
            let mut cols: Vec<[ValueId; 8]> = Vec::new();
            for j in 0..sl {
                let pos = b.bind_i32(ev(seqlen) - (sl - j) as i32);              // relative to the sequence start; < 0: from the initial state
                let have = b.cmp(CmpPred::Ge, CmpTy::I32, pos, 0);
                let tk = b.bind_i32(ev(seq_start) + ev(pos));
                let sel_t5 = sel_i(b, have);
                let on = b.bind_i32(ev(sel_t5) & ev(fok));
                let on_c = b.cmp(CmpPred::Ne, CmpTy::I32, on, 0);
                let o = xoff(b, tk, on_c);
                let fr = b.buffer_load_c(r_x, Some(Opnd::Val(o)), 0i32, 0, MemTy::B128, CachePolicy::NT);
                let xv = bf16x8_to_f32(b, fr);
                // the initial-state element j + seqlen (seqlen < W - 1 here): a select chain over its W - 1 possible values
                let src = b.bind_i32(ev(seqlen) + j as i32);
                let mut a = [0; 8];
                for f in 0..8usize {
                    let mut sv = zero;
                    for jj in 0..sl as usize {
                        let m = b.cmp(CmpPred::Eq, CmpTy::I32, src, jj as i32);
                        sv = b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(m), a: Opnd::Val(st[f * sl as usize + jj]), b: Opnd::Val(sv) });
                    }
                    a[f] = b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(have), a: Opnd::Val(xv[f]), b: Opnd::Val(sv) });
                }
                cols.push(a);
            }
            for f in 0..8usize {
                for j in 0..sl as usize {
                    vals.push(cols[j][f]);
                }
            }
            let words: Vec<ValueId> = (0..vals.len() / 2).map(|k| b.cvt_pk_bf16_f32(vals[2 * k], vals[2 * k + 1])).collect();
            let sel_t6 = sel_i(b, seq_ok);
            let wr = b.bind_i32(ev(sel_t6) & ev(fok));
            let wr_c = b.cmp(CmpPred::Ne, CmpTy::I32, wr, 0);
            let base = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(wr_c), a: Opnd::Val(st_base), b: Opnd::Val(oob) });
            for c in 0..words.len() / 4 {
                let f4 = b.frag4(words[4 * c], words[4 * c + 1], words[4 * c + 2], words[4 * c + 3]);
                let o = b.bind_i32(ev(base) + 16 * c as i32);
                let o = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(wr_c), a: Opnd::Val(o), b: Opnd::Val(oob) });
                b.buffer_store_c(r_s, Some(Opnd::Val(o)), 0i32, 0, f4, MemTy::B128, false, CachePolicy::NONE);
            }
        });
    });
    fb.finish()
}

fn feat_ok_i(b: &mut FnBuilder, c: ValueId) -> ValueId {
    b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(c), a: Opnd::I32(1), b: Opnd::I32(0) })
}
fn sel_i(b: &mut FnBuilder, c: ValueId) -> ValueId {
    b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(c), a: Opnd::I32(1), b: Opnd::I32(0) })
}

fn pack8(b: &mut FnBuilder, r: &[ValueId; 8]) -> ValueId {
    let w: Vec<ValueId> = (0..4).map(|k| b.cvt_pk_bf16_f32(r[2 * k], r[2 * k + 1])).collect();
    b.frag4(w[0], w[1], w[2], w[3])
}