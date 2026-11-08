//! Sampling from logits (docs/op-coverage-memtemplates.md T5 / T6), one 1024-thread block per row, the whole row register-resident (each thread owns `C = ceil(V / 8192)` consecutive 16-byte chunks, so a
//! thread's elements are contiguous in index order, which the cumulative-sum search needs).
//!
//! * `argmax_bf16_v<V>`  greedy: `out[b] = argmax_v logits[b][v]` (ties: the lowest index). ABI logits @0 (bf16 [B][V]), out @8 (i32 [B]), B @16 (i32).
//! * `sample_bf16_v<V>`  temperature sampling: `p = softmax(logits / T)`, `out[b]` = the smallest index whose cumulative probability reaches `u[b] * sum` (inverse-CDF / multinomial with the caller's uniform
//!   random number). ABI logits @0, u @8 (f32 [B], in [0, 1)), out @16 (i32 [B]), inv_temperature @24 (f32), B @28 (i32).
//!
//! V a multiple of 8, 8..=131072. A block reduction = DPP quads + the wave's LDS exchange, then one slot per wave and a barrier (a fresh slot per reduction, so no barrier is needed to reuse them).

use amdgpu_dsl::{AtomicOp, CachePolicy, CmpPred, Dpp, DppAluOp, DppCtrl, CmpTy, CompileProfile, DType, Ext, FnBuilder, Function, MemTy, Op, Opnd, ValueId};
use amdgpu_tile::prims::{bf16x8_to_f32, dpp_reduce, lane_reduce, wave_reduce_dpp, RedOp};
use amdgpu_tile::ev;

const T: i32 = 1024;
const NEG: f32 = -3.0e38;

pub fn by_name(name: &str) -> Option<Function> {
    let (rest, sample, filtered) = if let Some(r) = name.strip_prefix("sample_topkp_bf16_v") {
        (r, true, true)
    } else if let Some(r) = name.strip_prefix("sample_bf16_v") {
        (r, true, false)
    } else if let Some(r) = name.strip_prefix("argmax_bf16_v") {
        (r, false, false)
    } else {
        return None;
    };
    let v: i64 = rest.parse().ok()?;
    (v % 8 == 0 && (8..=131072).contains(&v)).then(|| sampling(name, v, sample, filtered))
}

/// The order-preserving 16-bit key of an f32 that holds a bf16 value (raw bits flipped so that integer order = value order).
fn key_of2(b: &mut FnBuilder, xf: ValueId) -> ValueId {
    let bits = b.bitcast_i32(xf);
    let raw = b.bind_i32(ev(bits).lshr(16));
    let sign = b.bind_i32(ev(raw) & 0x8000);
    let neg = b.cmp(CmpPred::Ne, CmpTy::I32, sign, 0);
    let flip = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(neg), a: Opnd::I32(0xffff), b: Opnd::I32(0x8000) });
    b.let_(Op::XorI32(Opnd::Val(raw), Opnd::Val(flip)))
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
    let wr = if dpp_reduce() {
        wave_reduce_dpp(b, op, x)
    } else {
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
        lane_reduce(b, op, &vals)
    };
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

pub fn sampling(name: &str, v: i64, sample: bool, filtered: bool) -> Function {
    let nchunk = v / 8;
    let c = ((((nchunk + (T as i64) - 1) / T as i64) as i32 + 3) / 4) * 4;   // chunks per thread (a multiple of 4: the passes take 4 chunks per iteration)
    let mut fb = FnBuilder::new(name, T as u32);
    fb.compile_profile(CompileProfile::Default);
    let b = &mut fb;
    b.arg("logits", DType::Ptr, 0, 8);
    let (out_off, b_off) = if sample {
        b.arg("u", DType::Ptr, 8, 8);
        b.arg("out", DType::Ptr, 16, 8);
        b.arg("inv_t", DType::F32, 24, 4);
        if filtered {
            b.arg("top_p", DType::F32, 28, 4);
            b.arg("top_k", DType::I32, 32, 4);
            (16, 36)
        } else {
            (16, 28)
        }
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
    let (ws, blk, sc0, sc1) = (0i32, 1024i32, 1536i32, 5632i32);
    b.lds(12944);
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
    let rowb = b.bind_i32(ev(row) * (2 * v) as i32);
    let neg = b.c_f32(NEG);
    let zero = b.c_f32(0.0);
    let big = b.c_f32(-1.0e9);
    // the passes are runtime loops of 4 chunks per iteration: the row is read once per pass (2nd / 3rd read from L2); a 1024-thread block can hold <= 128 VGPRs per thread, so
    // the row cannot stay in registers for a 128 K vocabulary, and unrolled passes would have all their loads hoisted to the top
    let c4 = c / 4;                                                        // loop iterations
    // chunk of this thread: `coalesced` (pass 1: chunk = 1024 * ci + tid) or contiguous (passes 2 / 3: chunk = tid * C + ci, so a thread's elements ascend in index order)
    let chunk_of = |b: &mut FnBuilder, ci: ValueId, coalesced: bool| -> (ValueId, ValueId) {
        let ch = if coalesced { b.bind_i32((ev(ci) << 10) + ev(tid)) } else { b.bind_i32(ev(tid) * c + ev(ci)) };
        let ok = b.let_(Op::CmpLtI32(Opnd::Val(ch), Opnd::I32(nchunk as i32)));
        (ch, ok)
    };
    // ---- pass 1 (coalesced): the lowest-index maximum
    let res1 = b.loop_(&[(DType::F32, Opnd::F32(NEG)), (DType::I32, Opnd::I32(0)), (DType::I32, Opnd::I32(0))], |b, vars| {
        let (vb, vi, vit) = (vars[0], vars[1], vars[2]);
        let done = b.let_(Op::CmpGeI32(Opnd::Var(vit), Opnd::I32(c4)));
        b.break_if(done);
        let mut best = b.let_(Op::AddF32(Opnd::Var(vb), Opnd::F32(0.0)));
        let mut bidx = b.let_(Op::AddI32(Opnd::Var(vi), Opnd::I32(0)));
        let mut loaded = Vec::new();
        for u in 0..4i32 {
            let ci = b.let_(Op::AddI32(Opnd::Var(vit), Opnd::I32(0)));
            let ci = b.bind_i32(ev(ci) * 4 + u);
            let (ch, ok) = chunk_of(b, ci, true);
            let off = b.bind_i32(ev(rowb) + (ev(ch) << 4));
            let frag = b.buffer_load_c(r_l, Some(Opnd::Val(off)), 0i32, 0, MemTy::B128, CachePolicy::NT);
            loaded.push((ch, ok, frag));
        }
        for (ch, ok, frag) in loaded {
            let mut x = bf16x8_to_f32(b, frag);
            if let Some(it) = inv_t {
                for k in 0..8 {
                    x[k] = b.mul_f32(x[k], it);
                }
            }
            for k in 0..8usize {
                let xk = sel_f32(b, ok, x[k], neg);
                let gt = b.cmp(CmpPred::Gt, CmpTy::F32, xk, best);
                let idx = b.bind_i32((ev(ch) << 3) + k as i32);
                best = sel_f32(b, gt, xk, best);
                bidx = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(gt), a: Opnd::Val(idx), b: Opnd::Val(bidx) });
            }
        }
        b.set_var(vb, best);
        b.set_var(vi, bidx);
        let nx = b.let_(Op::AddI32(Opnd::Var(vit), Opnd::I32(1)));
        b.set_var(vit, nx);
        (vb, vi)
    });
    let best = b.let_(Op::AddF32(Opnd::Var(res1.0), Opnd::F32(0.0)));
    let bidx = b.let_(Op::AddI32(Opnd::Var(res1.1), Opnd::I32(0)));
    let m = block_reduce(b, RedOp::Max, best, wave, lane, 0, ws, blk, 0);
    // the lowest index among the threads holding the maximum: the maximum of the negated indices
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
        let l2e = b.c_f32(std::f32::consts::LOG2_E);
        // ---- top-k / top-p: the threshold as an order-preserving 16-bit KEY of the bf16 logit (no sort): a two-pass radix select (histogram of the high byte, then of the low byte
        // inside the selected bin), once on the element counts (top-k) and once on the exp mass (top-p); histograms in LDS (`ds_add_u32` / `ds_add_f32`)
        let thr: Option<ValueId> = if filtered {
            let top_p = b.let_(Op::KernargF32 { offset: 28 });
            let top_k = b.let_(Op::KernargI32 { offset: 32 });
            let (hist, scan_a, scan_b, resl) = (9728i32, 10768i32, 11808i32, 12848i32);
            let tid4 = b.bind_i32(ev(tid) << 2);
            let tcl = b.let_(Op::MinI32(Opnd::Val(tid), Opnd::I32(256)));     // threads >= 256 work on the dummy slot 256
            let tcl4 = b.bind_i32(ev(tcl) << 2);
            let zero_i = b.c_i32(0);
            // the key of an f32 that holds a bf16 value: raw bits, flipped so that integer order = value order
            let key_of = |b: &mut FnBuilder, xf: ValueId| -> ValueId {
                let bits = b.bitcast_i32(xf);
                let raw = b.bind_i32(ev(bits).lshr(16));
                let sign = b.bind_i32(ev(raw) & 0x8000);
                let neg = b.cmp(CmpPred::Ne, CmpTy::I32, sign, 0);
                let flip = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(neg), a: Opnd::I32(0xffff), b: Opnd::I32(0x8000) });
                b.let_(Op::XorI32(Opnd::Val(raw), Opnd::Val(flip)))
            };
            // one histogram pass over the row: the elements with key >= kmin (and, `second`, with high byte == bsel); bins by the high byte or the low byte; weight 1 or exp(x - m)
            let hist_pass = |b: &mut FnBuilder, second: bool, kmin: ValueId, bsel: ValueId, weighted: bool| {
                b.loop_(&[(DType::I32, Opnd::I32(0))], |b, vv| {
                    let vit = vv[0];
                    let done = b.let_(Op::CmpGeI32(Opnd::Var(vit), Opnd::I32(c4)));
                    b.break_if(done);
                    let mut loaded = Vec::new();
                    for u in 0..4i32 {
                        let ci = b.let_(Op::AddI32(Opnd::Var(vit), Opnd::I32(0)));
                        let ci = b.bind_i32(ev(ci) * 4 + u);
                        let (ch, ok) = chunk_of(b, ci, true);
                        let off = b.bind_i32(ev(rowb) + (ev(ch) << 4));
                        let frag = b.buffer_load_c(r_l, Some(Opnd::Val(off)), 0i32, 0, MemTy::B128, CachePolicy::NT);
                        loaded.push((ok, frag));
                    }
                    // the elements that do not count go to a per-lane sink bin (256 + lane): 64 lanes on one address would serialise the LDS atomic
                    let dm = b.bind_i32(256 + (ev(tid) & 63));
                    for (ok, frag) in loaded {
                        let x = bf16x8_to_f32(b, frag);
                        for k in 0..8usize {
                            let key = key_of(b, x[k]);
                            let ge = b.cmp(CmpPred::Ge, CmpTy::I32, key, kmin);
                            let hi = b.bind_i32(ev(key).lshr(8));
                            let bin = if second { b.bind_i32(ev(key) & 255) } else { hi };
                            // the dummy bin (256) takes everything that does not count
                            let b1 = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(ge), a: Opnd::Val(bin), b: Opnd::Val(dm) });
                            let b2 = if second {
                                let same = b.cmp(CmpPred::Eq, CmpTy::I32, hi, bsel);
                                b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(same), a: Opnd::Val(b1), b: Opnd::Val(dm) })
                            } else {
                                b1
                            };
                            let b3 = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(ok), a: Opnd::Val(b2), b: Opnd::Val(dm) });
                            let addr = b.bind_i32((ev(b3) << 2) + hist);
                            if weighted {
                                let xs = match inv_t { Some(it) => b.mul_f32(x[k], it), None => x[k] };
                                let t = b.sub_f32(xs, m);
                                let u = b.mul_f32(t, l2e);
                                let e = b.exp2_f32(u);
                                b.atomic_lds(AtomicOp::AddF32, addr, e, false);
                            } else {
                                b.atomic_lds(AtomicOp::AddU32, addr, 1i32, false);
                            }
                        }
                    }
                    let nx = b.let_(Op::AddI32(Opnd::Var(vit), Opnd::I32(1)));
                    b.set_var(vit, nx);
                });
            };
            // zero the 257 bins, barrier
            let clear = |b: &mut FnBuilder| {
                let a = b.bind_i32(ev(tcl4) + hist);
                b.store_lds(a, zero_i, MemTy::B32, false);
                b.wait_lds(0);
                b.barrier();
            };
            // descending cumulative scan over the 256 bins; returns (bin, cumulative before the bin) of the bin where the cumulative reaches `tgt` (bin 0 / 0 when it never does); `float`: the bins hold f32 sums.
            // Thread t < 256 owns bin 255 - t: a DPP inclusive scan inside each wave (row_shr 1 / 2 / 4 / 8, then row_bcast 15 / 31), the four wave totals through LDS, one barrier.
            let wt = resl + 32;                                     // 16 dwords: the wave totals
            let scan_find = |b: &mut FnBuilder, tgt: ValueId, float: bool| -> (ValueId, ValueId) {
                let rb = b.bind_i32(255 - ev(tcl));
                let a_h = b.bind_i32((ev(rb) << 2) + hist);
                let v0 = b.load_lds(a_h, MemTy::B32);
                b.wait_lds(0);
                let in256 = b.cmp(CmpPred::Lt, CmpTy::I32, tid, 256);
                let v0z = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(in256), a: Opnd::Val(v0), b: Opnd::I32(0) });
                let vty = if float { b.bitcast_f32(v0z) } else { v0z };
                let aop = if float { DppAluOp::AddF32 } else { DppAluOp::AddU32 };
                let mut v = vty;
                for c in [DppCtrl::RowShr(1), DppCtrl::RowShr(2), DppCtrl::RowShr(4), DppCtrl::RowShr(8)] {
                    v = b.dpp_alu(aop, Dpp { ctrl: c, row_mask: 0xf, bank_mask: 0xf, bound_zero: true }, v, v, v);
                }
                v = b.dpp_alu(aop, Dpp { ctrl: DppCtrl::RowBcast15, row_mask: 0xa, bank_mask: 0xf, bound_zero: false }, v, v, v);
                v = b.dpp_alu(aop, Dpp { ctrl: DppCtrl::RowBcast31, row_mask: 0xc, bank_mask: 0xf, bound_zero: false }, v, v, v);
                // the wave total (lane 63) -> LDS; defaults of the result slots (thread 0)
                let tot = b.read_lane(v, 63);
                let a_wt = b.bind_i32((ev(wave) << 2) + wt);
                b.store_lds(a_wt, tot, MemTy::B32, false);
                let is0 = b.cmp(CmpPred::Eq, CmpTy::I32, tid, 0);
                let dflt_a = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(is0), a: Opnd::I32(resl), b: Opnd::I32(resl + 8) });
                b.store_lds(dflt_a, zero_i, MemTy::B32, false);
                let dflt_b = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(is0), a: Opnd::I32(resl + 4), b: Opnd::I32(resl + 8) });
                b.store_lds(dflt_b, zero_i, MemTy::B32, false);
                b.wait_lds(0);
                b.barrier();
                let mut tots = Vec::new();
                for w in 0..3i32 {
                    let a = b.c_i32(wt + 4 * w);
                    tots.push(b.load_lds(a, MemTy::B32));
                }
                b.wait_lds(0);
                // this wave's offset = the sum of the totals of the lower waves (waves >= 4 are idle; their value is unused)
                let incl = if float {
                    let mut pre = b.c_f32(0.0);
                    for (w, t) in tots.iter().enumerate() {
                        let tf = b.bitcast_f32(*t);
                        let sum = b.add_f32(pre, tf);
                        let is_hi = b.cmp(CmpPred::Gt, CmpTy::I32, wave, w as i32);
                        pre = sel_f32(b, is_hi, sum, pre);
                    }
                    b.add_f32(v, pre)
                } else {
                    let mut pre = b.c_i32(0);
                    for (w, t) in tots.iter().enumerate() {
                        let sum = b.add_i32(pre, *t);
                        let is_hi = b.cmp(CmpPred::Gt, CmpTy::I32, wave, w as i32);
                        pre = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(is_hi), a: Opnd::Val(sum), b: Opnd::Val(pre) });
                    }
                    b.add_i32(v, pre)
                };
                // exclusive = inclusive - own value; the owner is the thread with exclusive < tgt <= inclusive
                let (hit, before) = if float {
                    let vf = b.bitcast_f32(v0z);
                    let exf = b.sub_f32(incl, vf);
                    let g1 = b.cmp(CmpPred::Lt, CmpTy::F32, exf, tgt);
                    let g2 = b.cmp(CmpPred::Ge, CmpTy::F32, incl, tgt);
                    let g12 = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(g1), a: Opnd::I32(1), b: Opnd::I32(0) });
                    let g12 = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(g2), a: Opnd::Val(g12), b: Opnd::I32(0) });
                    (b.cmp(CmpPred::Ne, CmpTy::I32, g12, 0), b.bitcast_i32(exf))
                } else {
                    let ex = b.bind_i32(ev(incl) - ev(v0z));
                    let g1 = b.cmp(CmpPred::Lt, CmpTy::I32, ex, tgt);
                    let g2 = b.cmp(CmpPred::Ge, CmpTy::I32, incl, tgt);
                    let g12 = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(g1), a: Opnd::I32(1), b: Opnd::I32(0) });
                    let g12 = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(g2), a: Opnd::Val(g12), b: Opnd::I32(0) });
                    (b.cmp(CmpPred::Ne, CmpTy::I32, g12, 0), ex)
                };
                let go = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(hit), a: Opnd::I32(1), b: Opnd::I32(0) });
                let go = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(in256), a: Opnd::Val(go), b: Opnd::I32(0) });
                let wr = b.cmp(CmpPred::Ne, CmpTy::I32, go, 0);
                let addr_bin = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(wr), a: Opnd::I32(resl), b: Opnd::I32(resl + 8) });
                b.store_lds(addr_bin, rb, MemTy::B32, false);
                let addr_bef = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(wr), a: Opnd::I32(resl + 4), b: Opnd::I32(resl + 8) });
                b.store_lds(addr_bef, before, MemTy::B32, false);
                b.wait_lds(0);
                b.barrier();
                let r_bin_a = b.c_i32(resl);
                let r_bef_a = b.c_i32(resl + 4);
                let bin = b.load_lds(r_bin_a, MemTy::B32);
                let bef = b.load_lds(r_bef_a, MemTy::B32);
                b.wait_lds(0);
                b.barrier();
                (bin, bef)
            };
            // the radix select: the largest key T (>= kmin) with weight(key >= T) >= tgt
            let select = |b: &mut FnBuilder, tgt: ValueId, float: bool, kmin: ValueId| -> ValueId {
                clear(b);
                hist_pass(b, false, kmin, zero_i, float);
                b.wait_lds(0);
                b.barrier();
                let (bin, bef) = scan_find(b, tgt, float);
                let tgt2 = if float {
                    let (bf, tf) = (b.bitcast_f32(bef), tgt);
                    b.sub_f32(tf, bf)
                } else {
                    b.bind_i32(ev(tgt) - ev(bef))
                };
                clear(b);
                hist_pass(b, true, kmin, bin, float);
                b.wait_lds(0);
                b.barrier();
                let (low, _b2) = scan_find(b, tgt2, float);
                let hi8 = b.bind_i32(ev(bin) << 8);
                let t = b.bind_i32(ev(hi8) + ev(low));
                b.bin(amdgpu_dsl::BinOp::MaxI32, t, kmin)
            };
            let k_on = b.let_(Op::CmpLtI32(Opnd::I32(0), Opnd::Val(top_k)));
            let (res_k, res_p) = (resl + 12, resl + 16);
            let ak = b.c_i32(res_k);
            let ap = b.c_i32(res_p);
            b.store_lds(ak, zero_i, MemTy::B32, false);
            b.wait_lds(0);
            // top-k (counts): the largest key whose count(key >= T) >= k; off: key 0 keeps everything (a uniform `if`: nothing runs when the filter is off)
            b.if_(k_on, |b| {
                let k_tgt = b.bin(amdgpu_dsl::BinOp::MaxI32, top_k, 1i32);
                let tk = select(b, k_tgt, false, zero_i);
                b.store_lds(ak, tk, MemTy::B32, false);
                b.wait_lds(0);
            });
            b.barrier();
            let thr_k = b.load_lds(ak, MemTy::B32);
            b.wait_lds(0);
            b.store_lds(ap, thr_k, MemTy::B32, false);
            b.wait_lds(0);
            // top-p (mass): over what top-k kept
            let one_c2 = b.c_f32(1.0);
            let p_on = b.cmp(CmpPred::Lt, CmpTy::F32, top_p, one_c2);
            b.if_(p_on, |b| {
                let r = b.loop_(&[(DType::F32, Opnd::F32(0.0)), (DType::I32, Opnd::I32(0))], |b, v3| {
                    let (vs, vit) = (v3[0], v3[1]);
                    let done = b.let_(Op::CmpGeI32(Opnd::Var(vit), Opnd::I32(c4)));
                    b.break_if(done);
                    let mut sm = b.let_(Op::AddF32(Opnd::Var(vs), Opnd::F32(0.0)));
                    let mut loaded = Vec::new();
                    for u in 0..4i32 {
                        let ci = b.let_(Op::AddI32(Opnd::Var(vit), Opnd::I32(0)));
                        let ci = b.bind_i32(ev(ci) * 4 + u);
                        let (ch, ok) = chunk_of(b, ci, true);
                        let off = b.bind_i32(ev(rowb) + (ev(ch) << 4));
                        let frag = b.buffer_load_c(r_l, Some(Opnd::Val(off)), 0i32, 0, MemTy::B128, CachePolicy::NT);
                        loaded.push((ok, frag));
                    }
                    for (ok, frag) in loaded {
                        let x = bf16x8_to_f32(b, frag);
                        for k in 0..8usize {
                            let key = key_of(b, x[k]);
                            let keep = b.cmp(CmpPred::Ge, CmpTy::I32, key, thr_k);
                            let xs = match inv_t { Some(it) => b.mul_f32(x[k], it), None => x[k] };
                            let t = b.sub_f32(xs, m);
                            let u = b.mul_f32(t, l2e);
                            let pe = b.exp2_f32(u);
                            let p1 = sel_f32(b, keep, pe, zero);
                            let p2 = sel_f32(b, ok, p1, zero);
                            sm = b.add_f32(sm, p2);
                        }
                    }
                    b.set_var(vs, sm);
                    let nx = b.let_(Op::AddI32(Opnd::Var(vit), Opnd::I32(1)));
                    b.set_var(vit, nx);
                    vs
                });
                let st = b.let_(Op::AddF32(Opnd::Var(r), Opnd::F32(0.0)));
                let sk = block_reduce(b, RedOp::Sum, st, wave, lane, 0, ws, blk, 3);
                b.barrier();
                let target_p = b.mul_f32(top_p, sk);
                let tp = select(b, target_p, true, thr_k);
                b.store_lds(ap, tp, MemTy::B32, false);
                b.wait_lds(0);
            });
            b.barrier();
            let thr_f = b.load_lds(ap, MemTy::B32);
            b.wait_lds(0);
            Some(thr_f)
        } else {
            None
        };
        // ---- pass 2 (contiguous): the thread sum of exp(x - m)
        let st_v = b.loop_(&[(DType::F32, Opnd::F32(0.0)), (DType::I32, Opnd::I32(0))], |b, vars| {
            let (vs, vit) = (vars[0], vars[1]);
            let done = b.let_(Op::CmpGeI32(Opnd::Var(vit), Opnd::I32(c4)));
            b.break_if(done);
            let mut acc = b.let_(Op::AddF32(Opnd::Var(vs), Opnd::F32(0.0)));
            let mut loaded = Vec::new();
            for u in 0..4i32 {
                let ci = b.let_(Op::AddI32(Opnd::Var(vit), Opnd::I32(0)));
                let ci = b.bind_i32(ev(ci) * 4 + u);
                let (ch, ok) = chunk_of(b, ci, false);
                let off = b.bind_i32(ev(rowb) + (ev(ch) << 4));
                let frag = b.buffer_load_c(r_l, Some(Opnd::Val(off)), 0i32, 0, MemTy::B128, CachePolicy::NT);
                loaded.push((ok, frag));
            }
            for (ok, frag) in loaded {
                let mut x = bf16x8_to_f32(b, frag);
                let keys: Vec<ValueId> = if thr.is_some() { (0..8).map(|k| key_of2(b, x[k])).collect() } else { Vec::new() };
                for k in 0..8usize {
                    if let Some(it) = inv_t {
                        x[k] = b.mul_f32(x[k], it);
                    }
                    let t = b.sub_f32(x[k], m);
                    let u = b.mul_f32(t, l2e);
                    let p = b.exp2_f32(u);
                    let mut pm = sel_f32(b, ok, p, zero);
                    if let Some(th) = thr {
                        let keep = b.cmp(CmpPred::Ge, CmpTy::I32, keys[k], th);
                        pm = sel_f32(b, keep, pm, zero);
                    }
                    acc = b.add_f32(acc, pm);
                }
            }
            b.set_var(vs, acc);
            let nx = b.let_(Op::AddI32(Opnd::Var(vit), Opnd::I32(1)));
            b.set_var(vit, nx);
            vs
        });
        let st = b.let_(Op::AddF32(Opnd::Var(st_v), Opnd::F32(0.0)));
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
        // ---- pass 3 (contiguous): the first element whose cumulative value reaches the target; -(index), so that a Max picks the lowest index
        let fnd = b.loop_(&[(DType::F32, Opnd::F32(-1.0e9)), (DType::F32, Opnd::F32(0.0)), (DType::I32, Opnd::I32(0))], |b, vars| {
            let (vf, va, vit) = (vars[0], vars[1], vars[2]);
            let done = b.let_(Op::CmpGeI32(Opnd::Var(vit), Opnd::I32(c4)));
            b.break_if(done);
            let mut found = b.let_(Op::AddF32(Opnd::Var(vf), Opnd::F32(0.0)));
            let ca = b.let_(Op::AddF32(Opnd::Var(va), Opnd::F32(0.0)));
            let mut acc = b.add_f32(ca, excl);
            let mut loaded = Vec::new();
            for u in 0..4i32 {
                let ci = b.let_(Op::AddI32(Opnd::Var(vit), Opnd::I32(0)));
                let ci = b.bind_i32(ev(ci) * 4 + u);
                let (ch, ok) = chunk_of(b, ci, false);
                let off = b.bind_i32(ev(rowb) + (ev(ch) << 4));
                let frag = b.buffer_load_c(r_l, Some(Opnd::Val(off)), 0i32, 0, MemTy::B128, CachePolicy::NT);
                loaded.push((ch, ok, frag));
            }
            for (ch, ok, frag) in loaded {
                let mut x = bf16x8_to_f32(b, frag);
                let keys: Vec<ValueId> = if thr.is_some() { (0..8).map(|k| key_of2(b, x[k])).collect() } else { Vec::new() };
                for k in 0..8usize {
                    if let Some(it) = inv_t {
                        x[k] = b.mul_f32(x[k], it);
                    }
                    let t = b.sub_f32(x[k], m);
                    let u = b.mul_f32(t, l2e);
                    let p = b.exp2_f32(u);
                    let mut pm = sel_f32(b, ok, p, zero);
                    let mut keep_c = None;
                    if let Some(th) = thr {
                        let keep = b.cmp(CmpPred::Ge, CmpTy::I32, keys[k], th);
                        pm = sel_f32(b, keep, pm, zero);
                        keep_c = Some(keep);
                    }
                    acc = b.add_f32(acc, pm);
                    let ge = b.cmp(CmpPred::Ge, CmpTy::F32, acc, target);
                    let idx_i = b.bind_i32(ev(ch) * (-8) - k as i32);
                    let nf = b.un(amdgpu_dsl::UnOp::CvtF32I32, idx_i);
                    let c1 = sel_f32(b, ge, nf, big);
                    let c1 = if let Some(kc) = keep_c { sel_f32(b, kc, c1, big) } else { c1 };
                    let cand_k = sel_f32(b, ok, c1, big);
                    found = b.max_f32(found, cand_k);
                }
            }
            let ncum = b.sub_f32(acc, excl);
            b.set_var(vf, found);
            b.set_var(va, ncum);
            let nx = b.let_(Op::AddI32(Opnd::Var(vit), Opnd::I32(1)));
            b.set_var(vit, nx);
            vf
        });
        let found_neg = b.let_(Op::AddF32(Opnd::Var(fnd), Opnd::F32(0.0)));
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

