//! Sampling from logits (docs/op-coverage-memtemplates.md T5 / T6), one 1024-thread block per row, the whole row register-resident (each thread owns `C = ceil(V / 8192)` consecutive 16-byte chunks, so a
//! thread's elements are contiguous in index order, which the cumulative-sum search needs).
//!
//! * `argmax_bf16_v<V>`  greedy: `out[b] = argmax_v logits[b][v]` (ties: the lowest index). ABI logits @0 (bf16 [B][V]), out @8 (i32 [B]), B @16 (i32).
//! * `sample_bf16_v<V>`  temperature sampling: `p = softmax(logits / T)`, `out[b]` = the smallest index whose cumulative probability reaches `u[b] * sum` (inverse-CDF / multinomial with the caller's uniform
//!   random number). ABI logits @0, u @8 (f32 [B], in [0, 1)), out @16 (i32 [B]), inv_temperature @24 (f32), B @28 (i32).
//!
//! V a multiple of 8, 8..=131072. A block reduction = DPP quads + the wave's LDS exchange, then one slot per wave and a barrier (a fresh slot per reduction, so no barrier is needed to reuse them).

use amdgpu_dsl::{CachePolicy, CmpPred, CmpTy, CompileProfile, DType, Ext, FnBuilder, Function, MemTy, Op, Opnd, ValueId};
use amdgpu_tile::prims::{bf16x8_to_f32, lane_reduce, RedOp};
use amdgpu_tile::ev;

const T: i32 = 1024;
const NEG: f32 = -3.0e38;

pub fn by_name(name: &str) -> Option<Function> {
    let (rest, sample) = if let Some(r) = name.strip_prefix("sample_bf16_v") {
        (r, true)
    } else if let Some(r) = name.strip_prefix("argmax_bf16_v") {
        (r, false)
    } else {
        return None;
    };
    let v: i64 = rest.parse().ok()?;
    (v % 8 == 0 && (8..=131072).contains(&v)).then(|| sampling(name, v, sample))
}

fn sel_f32(b: &mut FnBuilder, cond: ValueId, x: ValueId, y: ValueId) -> ValueId {
    b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(cond), a: Opnd::Val(x), b: Opnd::Val(y) })
}

/// Block-wide reduction of one f32 per thread (all `T` threads): LDS regions `ws` (16 waves x 64 B wave exchange, shared by all reductions) and `blk + 64 * round` (16 wave results).
fn block_reduce(b: &mut FnBuilder, op: RedOp, x: ValueId, wave: ValueId, lane: ValueId, tid_wave_base: i32, ws: i32, blk: i32, round: i32) -> ValueId {
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
    // wave exchange: the 16 quad values of the wave
    let wa = b.bind_i32(ev(wave) * 64 + (ev(lane).lshr(2) << 2) + ws + tid_wave_base);
    let vi = b.bitcast_i32(v);
    b.store_lds(wa, vi, MemTy::B32, false);
    b.wait_lds(0);
    let ra = b.bind_i32(ev(wave) * 64 + ws + tid_wave_base);
    let mut vals = Vec::new();
    for q in 0..4i32 {
        let a = b.bind_i32(ev(ra) + 16 * q);
        let f = b.load_lds(a, MemTy::B128);
        for d in 0..4u8 {
            let w = b.frag4_lane(Opnd::Val(f), d);
            vals.push(b.bitcast_f32(w));
        }
    }
    b.wait_lds(0);
    let wr = lane_reduce(b, op, &vals);
    // one slot per wave, then every thread reduces the 16 slots
    let sa = b.bind_i32(ev(wave) * 4 + blk + 64 * round);
    let wri = b.bitcast_i32(wr);
    b.store_lds(sa, wri, MemTy::B32, false);
    b.wait_lds(0);
    b.barrier();
    let mut vals2 = Vec::new();
    for q in 0..4i32 {
        let a = b.c_i32(blk + 64 * round + 16 * q);
        let f = b.load_lds(a, MemTy::B128);
        for d in 0..4u8 {
            let w = b.frag4_lane(Opnd::Val(f), d);
            vals2.push(b.bitcast_f32(w));
        }
    }
    b.wait_lds(0);
    lane_reduce(b, op, &vals2)
}

pub fn sampling(name: &str, v: i64, sample: bool) -> Function {
    let nchunk = v / 8;
    let c = ((nchunk + (T as i64) - 1) / T as i64) as i32;   // chunks per thread
    let mut fb = FnBuilder::new(name, T as u32);
    fb.compile_profile(CompileProfile::Default);
    let b = &mut fb;
    b.arg("logits", DType::Ptr, 0, 8);
    let (out_off, b_off) = if sample {
        b.arg("u", DType::Ptr, 8, 8);
        b.arg("out", DType::Ptr, 16, 8);
        b.arg("inv_t", DType::F32, 24, 4);
        (16, 28)
    } else {
        b.arg("out", DType::Ptr, 8, 8);
        (8, 16)
    };
    b.arg("b", DType::I32, b_off, 4);
    let bb = Ext::sym("B");
    b.sym_range("B", 1, 1 << 20);
    b.ka_is(b_off, bb.clone());
    b.grid_is(0, bb.clone());
    b.arg_tensor("logits", bb.clone() * Ext::c(2 * v));
    if sample {
        b.arg_tensor("u", bb.clone() * Ext::c(4));
    }
    b.arg_tensor("out", bb * Ext::c(4));
    // LDS: wave exchange 1024 B, 4 reduction slots (256 B), scan ping-pong 2 x 4096 B
    let (ws, blk, sc0, sc1) = (0i32, 1024i32, 1280i32, 5376i32);
    b.lds(9472);
    let p_l = b.let_(Op::KernargPtr { offset: 0 });
    let p_out = b.let_(Op::KernargPtr { offset: out_off });
    let p_u = if sample { Some(b.let_(Op::KernargPtr { offset: 8 })) } else { None };
    let inv_t = if sample { Some(b.let_(Op::KernargF32 { offset: 24 })) } else { None };
    let b_v = b.let_(Op::KernargI32 { offset: b_off });
    let tid = b.let_(Op::WorkitemId);
    let row = b.let_(Op::WorkgroupIdX);
    let wave = b.bind_i32(ev(tid).lshr(6));
    let lane = b.bind_i32(ev(tid) & 63);
    let r_l = { let n = b.bind_i32(ev(b_v) * (2 * v) as i32); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_l, n) };
    let r_o = { let n = b.bind_i32(ev(b_v) << 2); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_out, n) };
    // this thread's chunks: consecutive, ascending
    let first = b.bind_i32(ev(tid) * c);                         // first chunk index
    let rowb = b.bind_i32(ev(row) * (2 * v) as i32);
    let neg = b.c_f32(NEG);
    let zero = b.c_f32(0.0);
    let mut xs: Vec<[ValueId; 8]> = Vec::new();
    let mut oks: Vec<ValueId> = Vec::new();
    let mut frags = Vec::new();
    for ci in 0..c {
        let ch = b.bind_i32(ev(first) + ci);
        let ok = b.let_(Op::CmpLtI32(Opnd::Val(ch), Opnd::I32(nchunk as i32)));
        let off = b.bind_i32(ev(rowb) + (ev(ch) << 4));
        frags.push(b.buffer_load_c(r_l, Some(Opnd::Val(off)), 0i32, 0, MemTy::B128, CachePolicy::NT));
        oks.push(ok);
    }
    for ci in 0..c as usize {
        let mut x = bf16x8_to_f32(b, frags[ci]);
        if let Some(it) = inv_t {
            for k in 0..8 {
                x[k] = b.mul_f32(x[k], it);
            }
        }
        for k in 0..8 {
            x[k] = sel_f32(b, oks[ci], x[k], neg);
        }
        xs.push(x);
    }
    // the first (lowest-index) maximum of this thread
    let mut best = xs[0][0];
    let mut bidx = b.bind_i32(ev(first) << 3);
    for ci in 0..c as usize {
        for k in 0..8usize {
            if ci == 0 && k == 0 {
                continue;
            }
            let gt = b.cmp(CmpPred::Gt, CmpTy::F32, xs[ci][k], best);
            let idx = b.bind_i32((ev(first) << 3) + (ci * 8 + k) as i32);
            best = sel_f32(b, gt, xs[ci][k], best);
            bidx = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(gt), a: Opnd::Val(idx), b: Opnd::Val(bidx) });
        }
    }
    let m = block_reduce(b, RedOp::Max, best, wave, lane, 0, ws, blk, 0);
    // the lowest index among the threads holding the maximum: the maximum of the negated indices
    let big = b.c_f32(-1.0e9);
    let at_max = b.cmp(CmpPred::Eq, CmpTy::F32, best, m);
    let nidx_i = b.bind_i32(ev(bidx) * (-1));
    let nidx = b.un(amdgpu_dsl::UnOp::CvtF32I32, nidx_i);
    let cand = sel_f32(b, at_max, nidx, big);
    let min_neg = block_reduce(b, RedOp::Max, cand, wave, lane, 0, ws, blk, 1);
    let argmax_f = b.sub_f32(zero, min_neg);
    let argmax = b.un(amdgpu_dsl::UnOp::CvtI32F32, argmax_f);
    let result = if !sample {
        argmax
    } else {
        // softmax numerators (masked -> 0), the thread sums, the block prefix scan, the first element whose cumulative value reaches u * total
        let l2e = b.c_f32(std::f32::consts::LOG2_E);
        let mut es: Vec<[ValueId; 8]> = Vec::new();
        let mut st = zero;
        for ci in 0..c as usize {
            let mut e = [0; 8];
            for k in 0..8 {
                let t = b.sub_f32(xs[ci][k], m);
                let u = b.mul_f32(t, l2e);
                let p = b.exp2_f32(u);
                e[k] = sel_f32(b, oks[ci], p, zero);
                st = b.add_f32(st, e[k]);
            }
            es.push(e);
        }
        // inclusive scan of the thread sums (Hillis-Steele in LDS)
        let tid4 = b.bind_i32(ev(tid) << 2);
        let a0 = b.bind_i32(ev(tid4) + sc0);
        let sti = b.bitcast_i32(st);
        b.store_lds(a0, sti, MemTy::B32, false);
        b.wait_lds(0);
        b.barrier();
        let mut cur = sc0;
        let mut s = 1i32;
        while s < T {
            let nxt = if cur == sc0 { sc1 } else { sc0 };
            let a_own = b.bind_i32(ev(tid4) + cur);
            let own = b.load_lds(a_own, MemTy::B32);
            let prev_i = b.bind_i32(ev(tid) - s);
            let okp = b.cmp(CmpPred::Ge, CmpTy::I32, prev_i, 0);
            let pi = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(okp), a: Opnd::Val(prev_i), b: Opnd::I32(0) });
            let a_pv = b.bind_i32((ev(pi) << 2) + cur);
            let pv = b.load_lds(a_pv, MemTy::B32);
            b.wait_lds(0);
            let ownf = b.bitcast_f32(own);
            let pvf = b.bitcast_f32(pv);
            let addf = sel_f32(b, okp, pvf, zero);
            let sum = b.add_f32(ownf, addf);
            let sumi = b.bitcast_i32(sum);
            let a_nx = b.bind_i32(ev(tid4) + nxt);
            b.store_lds(a_nx, sumi, MemTy::B32, false);
            b.wait_lds(0);
            b.barrier();
            cur = nxt;
            s *= 2;
        }
        let a_inc = b.bind_i32(ev(tid4) + cur);
        let incl_i = b.load_lds(a_inc, MemTy::B32);
        let a_tot = b.c_i32(cur + 4 * (T - 1));
        let tot_i = b.load_lds(a_tot, MemTy::B32);
        b.wait_lds(0);
        let incl = b.bitcast_f32(incl_i);
        let total = b.bitcast_f32(tot_i);
        let excl = b.sub_f32(incl, st);
        // the caller's uniform number (clamped below 1) times the total
        let uo = b.bind_i32(ev(row) << 2);
        let r_u = { let n = b.bind_i32(ev(b_v) << 2); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_u.unwrap(), n) };
        let uf = b.buffer_load_f32(r_u, uo, 0i32, 0);
        let one_m = b.c_f32(0.99999994);
        let uc = b.bin(amdgpu_dsl::BinOp::MinF32, uf, one_m);
        let target = b.mul_f32(uc, total);
        // first element (in this thread's order) whose cumulative value reaches the target; -(index) so that a Max picks the lowest index
        let mut acc = excl;
        let mut found_neg = big;   // -1e9 = not found
        for ci in 0..c as usize {
            for k in 0..8usize {
                acc = b.add_f32(acc, es[ci][k]);
                let ge = b.cmp(CmpPred::Ge, CmpTy::F32, acc, target);
                let idx_i = b.bind_i32((ev(first) << 3) + (ci * 8 + k) as i32);
                let nf_i = b.bind_i32(ev(idx_i) * (-1));
                let nf = b.un(amdgpu_dsl::UnOp::CvtF32I32, nf_i);
                // every element from the first hit on satisfies `acc >= target`; the first hit has the largest negated index of the thread
                let cand_k = sel_f32(b, ge, nf, big);
                found_neg = b.max_f32(found_neg, cand_k);
            }
        }
        let mn = block_reduce(b, RedOp::Max, found_neg, wave, lane, 0, ws, blk, 2);
        // fallback (rounding: nothing reached the target): the argmax
        let half = b.c_f32(-0.5e9);
        let none = b.cmp(CmpPred::Lt, CmpTy::F32, mn, half);
        let pick_f = b.sub_f32(zero, mn);
        let pick = b.un(amdgpu_dsl::UnOp::CvtI32F32, pick_f);
        b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(none), a: Opnd::Val(argmax), b: Opnd::Val(pick) })
    };
    let is0 = b.let_(Op::CmpLtI32(Opnd::Val(tid), Opnd::I32(1)));
    let oo = b.bind_i32(ev(row) << 2);
    let oob = b.c_i32(0x7fff_ff00);
    let o = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(is0), a: Opnd::Val(oo), b: Opnd::Val(oob) });
    let rf = b.bitcast_f32(result);
    b.buffer_store_f32(r_o, o, 0i32, 0, rf);
    fb.finish()
}

