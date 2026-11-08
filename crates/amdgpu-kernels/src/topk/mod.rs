//! Expert routing: softmax over the experts of a row, then the top-k probabilities and their indices (docs/op-coverage-memtemplates.md T6, wave-level).
//!
//! Each wave processes `RPW = 64 / L` rows at once (L = lanes per row = E / 8 rounded up to a power of two, >= 4): lane `l` holds the experts `8 (l % L) .. +7` (one b128). A selection round is ONE group
//! reduction: every element becomes an order-preserving key (the bf16 value in the high 16 bits, the expert id in the low 16 bits, arranged so that the larger float is the larger value and, among equal
//! values, the LOWER id), the group maximum of the keys is the winner, the winner is masked out. The softmax denominator is one more group reduction; the K probabilities are `exp(v_r - v_0) / sum`.
//! A group reduction = two DPP quad permutations + the quad values through the wave's private LDS scratch (all RPW groups share the one write / read, so the latency is amortised over RPW rows).
//!
//! * `topk_softmax_bf16_e<E>_k<K>`         `weights[t][r]` = softmax probability of the r-th largest expert, `indices[t][r]` its expert id (ties: the lowest id first).
//! * `topk_softmax_renorm_bf16_e<E>_k<K>`  the same with the K selected probabilities renormalised to sum to 1.
//!
//! E a multiple of 8, 32..=512; K in 1..=16. 4 waves per block, grid = ceil(T / (4 RPW)). ABI: gating @0 (bf16 [T][E]), weights @8 (f32 [T][K]), indices @16 (i32 [T][K]), T @24 (i32).

use amdgpu_dsl::{CachePolicy, CmpPred, CmpTy, CompileProfile, MemTy, DType, Ext, FnBuilder, Function, Op, Opnd, ValueId};
use amdgpu_tile::prims::{bf16x8_to_f32, group_reduce_dpp, lane_reduce, RedOp};
use amdgpu_tile::shared::PrivRef;
use amdgpu_tile::{ev, Lane, LdsPlan};

pub fn by_name(name: &str) -> Option<Function> {
    let (rest, renorm) = if let Some(r) = name.strip_prefix("topk_softmax_renorm_bf16_e") {
        (r, true)
    } else if let Some(r) = name.strip_prefix("topk_softmax_bf16_e") {
        (r, false)
    } else {
        return None;
    };
    let (e, k) = rest.split_once("_k")?;
    let (e, k): (i64, i64) = (e.parse().ok()?, k.parse().ok()?);
    (e % 8 == 0 && (32..=512).contains(&e) && (1..=16).contains(&k) && k <= e).then(|| topk_softmax(name, e, k, renorm))
}

fn sel_f32(b: &mut FnBuilder, cond: ValueId, x: ValueId, y: ValueId) -> ValueId {
    b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(cond), a: Opnd::Val(x), b: Opnd::Val(y) })
}

fn sel_i32(b: &mut FnBuilder, cond: ValueId, x: ValueId, y: ValueId) -> ValueId {
    b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(cond), a: Opnd::Val(x), b: Opnd::Val(y) })
}

/// Reduction over the `l` lanes of each aligned group of a wave (l a power of two in 4..=64); every lane of a group gets the group's result.
fn group_reduce(b: &mut FnBuilder, op: RedOp, x: ValueId, scratch: &PrivRef, off: i32, gen: i32, lane: &Lane, l: i64) -> ValueId {
    // the DPP all-reduce is the default here (+9..19 % measured); AMDC_DPP_REDUCE=0 selects the LDS exchange
    if std::env::var("AMDC_DPP_REDUCE").map(|v| v != "0").unwrap_or(true) {
        return group_reduce_dpp(b, op, x, l);
    }
    let comb = |b: &mut FnBuilder, p: ValueId, q: ValueId| match op {
        RedOp::Sum => b.add_f32(p, q),
        RedOp::Max => b.max_f32(p, q),
    };
    let mut v = x;
    for perm in [[1u8, 0, 3, 2], [2, 3, 0, 1]] {
        let y = b.dpp_quad_perm(v, perm);
        let y = b.bitcast_f32(y);
        v = comb(b, v, y);
    }
    if l == 4 {
        return v;
    }
    let quads = l / 4;                                  // quad values per group
    let base = scratch.base() + off;
    let wa = b.bind_i32(base.clone() + (lane.lane().lshr(2) << 2));
    let vi = b.bitcast_i32(v);
    scratch.write(b, gen, |b| b.store_lds_b32(wa, vi));
    b.wait_lds(0);
    let gshift = (l as u32).trailing_zeros() as i32;
    let parts: Vec<ValueId> = scratch.read_back(b, gen, |b| {
        let a0 = b.bind_i32(base.clone() + (lane.lane().lshr(gshift) * (4 * quads) as i32));
        if quads == 2 {
            vec![b.load_lds_b64(a0)]
        } else {
            (0..quads / 4)
                .map(|q| {
                    let a = b.bind_i32(ev(a0) + 16 * q as i32);
                    b.load_lds_b128(a)
                })
                .collect()
        }
    });
    let mut vals = Vec::new();
    for p in parts {
        if quads == 2 {
            for d in 0..2u8 {
                let w = b.frag2_lane(Opnd::Val(p), d);
                vals.push(b.bitcast_f32(w));
            }
        } else {
            for d in 0..4u8 {
                let w = b.frag4_lane(Opnd::Val(p), d);
                vals.push(b.bitcast_f32(w));
            }
        }
    }
    lane_reduce(b, op, &vals)
}

pub fn topk_softmax(name: &str, e: i64, k: i64, renorm: bool) -> Function {
    let l = ((e / 8) as u64).next_power_of_two() as i64;   // lanes per row (>= 4)
    let rpw = 64 / l;                                       // rows per wave
    let mut fb = FnBuilder::new(name, 256);
    fb.compile_profile(CompileProfile::Default);
    let b = &mut fb;
    b.arg("gating", DType::Ptr, 0, 8);
    b.arg("weights", DType::Ptr, 8, 8);
    b.arg("indices", DType::Ptr, 16, 8);
    b.arg("t", DType::I32, 24, 4);
    let mt = Ext::sym("MT");
    b.sym_range("MT", 1, 1 << 20);
    b.ka_is(24, mt.clone() * Ext::c(4 * rpw));
    b.grid_is(0, mt.clone());
    b.arg_tensor("gating", mt.clone() * Ext::c(4 * rpw * 2 * e));
    b.arg_tensor("weights", mt.clone() * Ext::c(4 * rpw * 4 * k));
    b.arg_tensor("indices", mt * Ext::c(4 * rpw * 4 * k));
    let (p_g, p_w, p_i) = (b.let_(Op::KernargPtr { offset: 0 }), b.let_(Op::KernargPtr { offset: 8 }), b.let_(Op::KernargPtr { offset: 16 }));
    let t_v = b.let_(Op::KernargI32 { offset: 24 });
    let tid = b.let_(Op::WorkitemId);
    let lane = Lane::from_tid(b, tid);
    let bx = b.let_(Op::WorkgroupIdX);
    let mut lds = LdsPlan::new();
    let red = lds.private("RED", 128, 4, &[]);
    lds.commit(b);
    let r_g = { let v = b.bind_i32(ev(t_v) * (2 * e) as i32); let v = b.read_first_lane(v); b.buffer_desc_dyn(p_g, v) };
    let r_w = { let v = b.bind_i32(ev(t_v) * (4 * k) as i32); let v = b.read_first_lane(v); b.buffer_desc_dyn(p_w, v) };
    let r_i = { let v = b.bind_i32(ev(t_v) * (4 * k) as i32); let v = b.read_first_lane(v); b.buffer_desc_dyn(p_i, v) };
    let wid = b.bind_i32(lane.wave());
    let me = red.mine(wid);
    let lsh = (l as u32).trailing_zeros() as i32;
    let row_in_wave = b.bind_i32(lane.lane().lshr(lsh));
    let lr = b.bind_i32(lane.lane() & (l - 1) as i32);       // lane within the row group
    let row = b.bind_i32((ev(bx) * 4 + lane.wave()) * rpw as i32 + ev(row_in_wave));
    let voff = b.bind_i32(ev(row) * (2 * e) as i32 + (ev(lr) << 4));
    let g = b.buffer_load_c(r_g, Some(Opnd::Val(voff)), 0i32, 0, MemTy::B128, CachePolicy::NT);
    b.wait_vm(0);
    let neg = b.c_f32(-3.0e38);
    let zero = b.c_f32(0.0);
    let l2e = b.c_f32(std::f32::consts::LOG2_E);
    let valid = if e < 8 * l {
        let el = b.bind_i32(ev(lr) << 3);
        Some(b.let_(Op::CmpLtI32(Opnd::Val(el), Opnd::I32(e as i32))))
    } else {
        None
    };
    // the 8 values (f32) and keys of this lane
    let xf = bf16x8_to_f32(b, g);
    let mut keys = [0; 8];
    for i in 0..8 {
        let xb = b.bitcast_i32(xf[i]);
        let idx = b.bind_i32((ev(lr) << 3) + i as i32);
        let isneg = b.let_(Op::CmpLtI32(Opnd::Val(xb), Opnd::I32(0)));
        let inv = b.bind_i32(ev(idx) * (-1) + 0xffff);
        let fld = sel_i32(b, isneg, idx, inv);               // negative values: a larger mantissa is a smaller float, so the id grows with it; positive: it shrinks
        let kb = b.bind_i32((ev(xb) & (0xffff_0000u32 as i32)) | ev(fld));
        let kf = b.bitcast_f32(kb);
        keys[i] = if let Some(c) = valid { sel_f32(b, c, kf, neg) } else { kf };
    }
    // round 0 winner = the maximum: its value is the softmax reference
    let mut cur = keys;
    let mut win: Vec<ValueId> = Vec::new();   // winner keys (f32 bit patterns)
    let w0 = {
        let m = lane_reduce(b, RedOp::Max, &cur);
        group_reduce(b, RedOp::Max, m, &me, 0, 0, &lane, l)
    };
    win.push(w0);
    let wb0 = b.bitcast_i32(w0);
    let m0b = b.bind_i32(ev(wb0) & (0xffff_0000u32 as i32));
    let m0 = b.bitcast_f32(m0b);
    // denominator: sum of exp(x - m0) over the valid experts of the group
    let mut ex = [0; 8];
    for i in 0..8 {
        let t = b.sub_f32(xf[i], m0);
        let u = b.mul_f32(t, l2e);
        let p = b.exp2_f32(u);
        ex[i] = if let Some(c) = valid { sel_f32(b, c, p, zero) } else { p };
    }
    let ls = lane_reduce(b, RedOp::Sum, &ex);
    let sum = group_reduce(b, RedOp::Sum, ls, &me, 64, 1, &lane, l);
    let inv_sum = b.rcp_f32(sum);
    for r in 1..k as i32 {
        // mask the previous winner, then the next group maximum
        let prev = win[(r - 1) as usize];
        for i in 0..8 {
            let hit = b.cmp(CmpPred::Eq, CmpTy::F32, cur[i], prev);
            cur[i] = sel_f32(b, hit, neg, cur[i]);
        }
        let m = lane_reduce(b, RedOp::Max, &cur);
        let off = if r % 2 == 0 { 0 } else { 64 };
        let w = group_reduce(b, RedOp::Max, m, &me, off, 1 + r, &lane, l);
        win.push(w);
    }
    // decode: id and probability of each winner
    let mut ws: Vec<ValueId> = Vec::new();
    let mut ids: Vec<ValueId> = Vec::new();
    for &w in &win {
        let wi = b.bitcast_i32(w);
        let vb = b.bind_i32(ev(wi) & (0xffff_0000u32 as i32));
        let v = b.bitcast_f32(vb);
        let fld = b.bind_i32(ev(wi) & 0xffff);
        let isneg = b.let_(Op::CmpLtI32(Opnd::Val(wi), Opnd::I32(0)));
        let inv = b.bind_i32(ev(fld) * (-1) + 0xffff);
        ids.push(sel_i32(b, isneg, fld, inv));
        let t = b.sub_f32(v, m0);
        let u = b.mul_f32(t, l2e);
        let p = b.exp2_f32(u);
        ws.push(b.mul_f32(p, inv_sum));
    }
    let scale = if renorm {
        let mut s = ws[0];
        for &w in &ws[1..] {
            s = b.add_f32(s, w);
        }
        Some(b.rcp_f32(s))
    } else {
        None
    };
    // the first lane of each row group stores the row's K results (other lanes, rows past T: out of range)
    let first = b.let_(Op::CmpLtI32(Opnd::Val(lr), Opnd::I32(1)));
    let in_t = b.let_(Op::CmpLtI32(Opnd::Val(row), Opnd::Val(t_v)));
    let wbase = b.bind_i32(ev(row) * (4 * k) as i32);
    let oob = b.c_i32(0x7fff_ff00);   // past every descriptor even with the immediate added
    let o1 = sel_i32(b, first, wbase, oob);
    let ok = sel_i32(b, in_t, o1, oob);
    for r in 0..k as usize {
        let w = if let Some(s) = scale { b.mul_f32(ws[r], s) } else { ws[r] };
        b.buffer_store_f32(r_w, ok, 0i32, 4 * r as u32, w);
        let ib = b.bitcast_f32(ids[r]);
        b.buffer_store_f32(r_i, ok, 0i32, 4 * r as u32, ib);
    }
    fb.finish()
}