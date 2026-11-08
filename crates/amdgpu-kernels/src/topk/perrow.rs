//! Per-row top-K of f32 logits (aiter `top_k_per_row_decode` / `top_k_per_row_prefill`, the sparse-attention indexer): the K largest values of each row, as column indices.
//!
//! * `topk_per_row_decode_f32_k<K>`:  ABI logits @0 (f32 [R][stride]), seq_lens @8 (i32 [B]), indices @16 (i32 [R][K]), stride @24 (i32 elements), next_n @28 (i32), rows @32 (i32, R = B * next_n).
//!   Grid x = B, y = next_n; the row `b * next_n + j` has `L = seq_lens[b] - next_n + j + 1` valid columns (0 .. L).
//! * `topk_per_row_prefill_f32_k<K>`: ABI logits @0, row_starts @8 (i32 [R]), row_ends @16 (i32 [R]), indices @24, stride @32, rows @36. Grid x = R; the valid columns are [start, end), the indices are absolute columns.
//!
//! Output: `indices[r][0 .. min(K, L))` = the columns of the K largest values (unordered; ties at the K-th value are taken lowest-slot-first by an LDS counter, i.e. an arbitrary subset), and `-1` after
//! them when `L < K`. A row of `L <= K` columns is returned whole (column order). K in 1..=2048.
//!
//! One block of 1024 threads per row: a four-pass radix select on the order-preserving 32-bit key of the f32 (8 bits per pass, LDS histogram atomics, the 256-bin descending scan of the sampling
//! kernels: DPP wave scan + 4 wave totals) finds the K-th largest key T and `rem` = how many elements equal to T are needed; a last pass emits every element > T (LDS counter) and the first
//! `rem` elements equal to T. The row is read five times (the first from HBM, the others mostly from L2).

use amdgpu_dsl::{AtomicOp, CachePolicy, CmpPred, CmpTy, CompileProfile, DType, Dpp, DppAluOp, DppCtrl, Ext, FnBuilder, Function, MemTy, Op, Opnd, ValueId};
use amdgpu_tile::ev;

pub fn by_name(name: &str) -> Option<Function> {
    let (rest, prefill) = if let Some(r) = name.strip_prefix("topk_per_row_decode_f32_k") {
        (r, false)
    } else if let Some(r) = name.strip_prefix("topk_per_row_prefill_f32_k") {
        (r, true)
    } else {
        return None;
    };
    let k: i64 = rest.parse().ok()?;
    (1..=2048).contains(&k).then(|| topk_per_row(name, k, prefill))
}

// LDS: histogram 320 slots (256 bins + a per-lane sink 256 + lane), wave totals, result slots, two counters, the emit sink (one slot per thread)
const HIST: i32 = 0;
const WT: i32 = 1280;
const RES: i32 = 1344;
const CNT: i32 = 1360;
const SINK: i32 = 1376;
const LDS: u32 = 1376 + 4096;

pub fn topk_per_row(name: &str, k: i64, prefill: bool) -> Function {
    let mut fb = FnBuilder::new(name, 1024);
    fb.compile_profile(CompileProfile::Default);
    let b = &mut fb;
    b.arg("logits", DType::Ptr, 0, 8);
    let (idx_off, st_off) = if prefill {
        b.arg("row_starts", DType::Ptr, 8, 8);
        b.arg("row_ends", DType::Ptr, 16, 8);
        b.arg("indices", DType::Ptr, 24, 8);
        b.arg("stride", DType::I32, 32, 4);
        b.arg("rows", DType::I32, 36, 4);
        (24u32, 32u32)
    } else {
        b.arg("seq_lens", DType::Ptr, 8, 8);
        b.arg("indices", DType::Ptr, 16, 8);
        b.arg("stride", DType::I32, 24, 4);
        b.arg("next_n", DType::I32, 28, 4);
        b.arg("rows", DType::I32, 32, 4);
        (16u32, 24u32)
    };
    let rows_off = if prefill { 36u32 } else { 32u32 };
    let rr = Ext::sym("R");
    let ss = Ext::sym("S");
    b.sym_range("R", 1, 1 << 24);
    b.sym_range("S", 1, 1 << 26);
    b.ka_is(rows_off, rr.clone());
    b.ka_is(st_off, ss.clone());
    if prefill {
        b.grid_is(0, rr.clone());
    } else {
        // rows = B * next_n, grid (B, next_n)
        let nn = Ext::sym("NN");
        b.sym_range("NN", 1, 16);
        b.ka_is(28, nn.clone());
        b.grid_is(1, nn);
    }
    b.arg_tensor("logits", rr.clone() * ss * Ext::c(4));
    b.arg_tensor("indices", rr.clone() * Ext::c(4 * k));
    if prefill {
        b.arg_tensor("row_starts", rr.clone() * Ext::c(4));
        b.arg_tensor("row_ends", rr * Ext::c(4));
    } else {
        b.arg_tensor("seq_lens", rr * Ext::c(4));
    }
    b.lds(LDS);
    let p_l = b.let_(Op::KernargPtr { offset: 0 });
    let p_i = b.let_(Op::KernargPtr { offset: idx_off });
    let stride_v = b.let_(Op::KernargI32 { offset: st_off });
    let rows_v = b.let_(Op::KernargI32 { offset: rows_off });
    let tid = b.let_(Op::WorkitemId);
    let bx = b.let_(Op::WorkgroupIdX);
    let wave = b.bind_i32(ev(tid).lshr(6));
    let zero_i = b.c_i32(0);
    let r_l = { let n = b.bind_i32(ev(rows_v) * ev(stride_v) * 4); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_l, n) };
    let r_o = { let n = b.bind_i32(ev(rows_v) * (4 * k) as i32); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_i, n) };
    // the row, its first column and its length (uniform)
    let (row, start, len) = if prefill {
        let (p_s, p_e) = (b.let_(Op::KernargPtr { offset: 8 }), b.let_(Op::KernargPtr { offset: 16 }));
        let n = b.bind_i32(ev(rows_v) << 2);
        let n = b.read_first_lane(n);
        let (r_s, r_e) = (b.buffer_desc_dyn(p_s, n), b.buffer_desc_dyn(p_e, n));
        let o = b.bind_i32(ev(bx) << 2);
        let sf = b.buffer_load_f32(r_s, o, 0i32, 0);
        let ef = b.buffer_load_f32(r_e, o, 0i32, 0);
        let s0 = b.bitcast_i32(sf);
        let e0 = b.bitcast_i32(ef);
        let s0 = b.read_first_lane(s0);
        let e0 = b.read_first_lane(e0);
        let s1 = b.let_(Op::AddI32(Opnd::Val(s0), Opnd::I32(0)));
        let e1 = b.let_(Op::AddI32(Opnd::Val(e0), Opnd::I32(0)));
        let l = b.bind_i32(ev(e1) - ev(s1));
        (bx, s1, l)
    } else {
        let by = b.let_(Op::WorkgroupIdY);
        let nn = b.let_(Op::KernargI32 { offset: 28 });
        let p_sl = b.let_(Op::KernargPtr { offset: 8 });
        let n = b.bind_i32(ev(rows_v) << 2);
        let n = b.read_first_lane(n);
        let r_sl = b.buffer_desc_dyn(p_sl, n);
        let o = b.bind_i32(ev(bx) << 2);
        let lf = b.buffer_load_f32(r_sl, o, 0i32, 0);
        let l0 = b.bitcast_i32(lf);
        let l0 = b.read_first_lane(l0);
        let l1 = b.let_(Op::AddI32(Opnd::Val(l0), Opnd::I32(0)));
        let l = b.bind_i32(ev(l1) - ev(nn) + ev(by) + 1);
        let rw = b.bind_i32(ev(bx) * ev(nn) + ev(by));
        (rw, zero_i, l)
    };
    // the length is clamped to [0, stride]
    let len = b.bin(amdgpu_dsl::BinOp::MaxI32, len, 0i32);
    let len = b.let_(Op::MinI32(Opnd::Val(len), Opnd::Val(stride_v)));
    let len = b.read_first_lane(len);
    let len = b.let_(Op::AddI32(Opnd::Val(len), Opnd::I32(0)));
    let rowb = b.bind_i32((ev(row) * ev(stride_v) + ev(start)) << 2);   // byte offset of the row's first valid column
    let orow = b.bind_i32(ev(row) * (4 * k) as i32);                  // byte offset of the row's output
    let tcl = b.let_(Op::MinI32(Opnd::Val(tid), Opnd::I32(256)));
    let tcl4 = b.bind_i32(ev(tcl) << 2);
    let sink_bin = b.bind_i32(256 + (ev(tid) & 63));
    let sink_e = b.bind_i32((ev(tid) << 2) + SINK);
    let iters = b.bind_i32((ev(len) + 4095) >> 12);
    let oob = b.c_i32(0x7fff_ff00);
    let kc = b.c_i32(k as i32);
    let long_row = b.cmp(CmpPred::Gt, CmpTy::I32, len, kc);
    let short_row = b.cmp(CmpPred::Le, CmpTy::I32, len, kc);
    let key_of = |b: &mut FnBuilder, x: ValueId| -> ValueId {
        let bits = b.bitcast_i32(x);
        let neg = b.cmp(CmpPred::Lt, CmpTy::I32, bits, 0);
        let m1 = b.c_i32(-1);
        let mn = b.c_i32(i32::MIN);
        let flip = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(neg), a: Opnd::Val(m1), b: Opnd::Val(mn) });
        b.let_(Op::XorI32(Opnd::Val(bits), Opnd::Val(flip)))
    };
    // ---- the short rows: the whole row in column order, -1 after it ----
    b.if_(short_row, |b| {
        for t in 0..((k + 1023) / 1024) {
            let p = b.bind_i32(ev(tid) + (t * 1024) as i32);
            let valid = b.cmp(CmpPred::Lt, CmpTy::I32, p, len);
            let col = b.bind_i32(ev(start) + ev(p));
            let m1 = b.c_i32(-1);
            let val = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(valid), a: Opnd::Val(col), b: Opnd::Val(m1) });
            let inb = b.cmp(CmpPred::Lt, CmpTy::I32, p, kc);
            let off = b.bind_i32(ev(orow) + (ev(p) << 2));
            let off = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(inb), a: Opnd::Val(off), b: Opnd::Val(oob) });
            let vf = b.bitcast_f32(val);
            b.buffer_store_f32(r_o, off, 0i32, 0, vf);
        }
    });
    // ---- the long rows: the radix select + the emit pass ----
    b.if_(long_row, |b| {
        // one histogram pass over the row: digit `s` (bits s .. s + 8) of the keys whose higher bits equal those of `thr`
        let hist_pass = |b: &mut FnBuilder, s: i32, thr: ValueId| {
            b.loop_(&[(DType::I32, Opnd::I32(0))], |b, vv| {
                let vit = vv[0];
                let done = b.let_(Op::CmpGeI32(Opnd::Var(vit), Opnd::Val(iters)));
                b.break_if(done);
                let mut loaded = Vec::new();
                for u in 0..4i32 {
                    let idx = b.bind_i32((ev(vit) << 12) + (u * 1024) as i32 + ev(tid));
                    let off = b.bind_i32(ev(rowb) + (ev(idx) << 2));
                    let x = b.buffer_load_f32(r_l, off, 0i32, 0);
                    let ok = b.cmp(CmpPred::Lt, CmpTy::I32, idx, len);
                    loaded.push((ok, x));
                }
                for (ok, x) in loaded {
                    let key = key_of(b, x);
                    let bin = b.bind_i32(ev(key).lshr(s) & 255);
                    let bsel = if s == 24 {
                        b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(ok), a: Opnd::Val(bin), b: Opnd::Val(sink_bin) })
                    } else {
                        let hi_k = b.bind_i32(ev(key).lshr(s + 8));
                        let hi_t = b.bind_i32(ev(thr).lshr(s + 8));
                        let same = b.cmp(CmpPred::Eq, CmpTy::I32, hi_k, hi_t);
                        let b1 = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(same), a: Opnd::Val(bin), b: Opnd::Val(sink_bin) });
                        b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(ok), a: Opnd::Val(b1), b: Opnd::Val(sink_bin) })
                    };
                    let addr = b.bind_i32((ev(bsel) << 2) + HIST);
                    b.atomic_lds(AtomicOp::AddU32, addr, 1i32, false);
                }
                let nx = b.let_(Op::AddI32(Opnd::Var(vit), Opnd::I32(1)));
                b.set_var(vit, nx);
            });
        };
        let clear = |b: &mut FnBuilder| {
            let a = b.bind_i32(ev(tcl4) + HIST);
            b.store_lds(a, zero_i, MemTy::B32, false);
            b.wait_lds(0);
            b.barrier();
        };
        // the descending scan of the 256 bins (thread t < 256 owns bin 255 - t): (bin, count of elements in higher bins) of the bin where the cumulative count reaches `tgt`
        let scan_find = |b: &mut FnBuilder, tgt: ValueId| -> (ValueId, ValueId) {
            let rb = b.bind_i32(255 - ev(tcl));
            let a_h = b.bind_i32((ev(rb) << 2) + HIST);
            let v0 = b.load_lds(a_h, MemTy::B32);
            b.wait_lds(0);
            let in256 = b.cmp(CmpPred::Lt, CmpTy::I32, tid, 256);
            let v0z = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(in256), a: Opnd::Val(v0), b: Opnd::I32(0) });
            let mut v = v0z;
            for c in [DppCtrl::RowShr(1), DppCtrl::RowShr(2), DppCtrl::RowShr(4), DppCtrl::RowShr(8)] {
                v = b.dpp_alu(DppAluOp::AddU32, Dpp { ctrl: c, row_mask: 0xf, bank_mask: 0xf, bound_zero: true }, v, v, v);
            }
            v = b.dpp_alu(DppAluOp::AddU32, Dpp { ctrl: DppCtrl::RowBcast15, row_mask: 0xa, bank_mask: 0xf, bound_zero: false }, v, v, v);
            v = b.dpp_alu(DppAluOp::AddU32, Dpp { ctrl: DppCtrl::RowBcast31, row_mask: 0xc, bank_mask: 0xf, bound_zero: false }, v, v, v);
            let tot = b.read_lane(v, 63);
            let a_wt = b.bind_i32((ev(wave) << 2) + WT);
            b.store_lds(a_wt, tot, MemTy::B32, false);
            let is0 = b.cmp(CmpPred::Eq, CmpTy::I32, tid, 0);
            let dflt_a = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(is0), a: Opnd::I32(RES), b: Opnd::I32(RES + 8) });
            b.store_lds(dflt_a, zero_i, MemTy::B32, false);
            let dflt_b = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(is0), a: Opnd::I32(RES + 4), b: Opnd::I32(RES + 8) });
            b.store_lds(dflt_b, zero_i, MemTy::B32, false);
            b.wait_lds(0);
            b.barrier();
            let mut tots = Vec::new();
            for w in 0..3i32 {
                let a = b.c_i32(WT + 4 * w);
                tots.push(b.load_lds(a, MemTy::B32));
            }
            b.wait_lds(0);
            let mut pre = b.c_i32(0);
            for (w, t) in tots.iter().enumerate() {
                let sum = b.add_i32(pre, *t);
                let is_hi = b.cmp(CmpPred::Gt, CmpTy::I32, wave, w as i32);
                pre = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(is_hi), a: Opnd::Val(sum), b: Opnd::Val(pre) });
            }
            let incl = b.add_i32(v, pre);
            let ex = b.bind_i32(ev(incl) - ev(v0z));
            let g1 = b.cmp(CmpPred::Lt, CmpTy::I32, ex, tgt);
            let g2 = b.cmp(CmpPred::Ge, CmpTy::I32, incl, tgt);
            let g12 = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(g1), a: Opnd::I32(1), b: Opnd::I32(0) });
            let g12 = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(g2), a: Opnd::Val(g12), b: Opnd::I32(0) });
            let go = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(in256), a: Opnd::Val(g12), b: Opnd::I32(0) });
            let wr = b.cmp(CmpPred::Ne, CmpTy::I32, go, 0);
            let addr_bin = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(wr), a: Opnd::I32(RES), b: Opnd::I32(RES + 8) });
            b.store_lds(addr_bin, rb, MemTy::B32, false);
            let addr_bef = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(wr), a: Opnd::I32(RES + 4), b: Opnd::I32(RES + 8) });
            b.store_lds(addr_bef, ex, MemTy::B32, false);
            b.wait_lds(0);
            b.barrier();
            let ra = b.c_i32(RES);
            let rb2 = b.c_i32(RES + 4);
            let bin = b.load_lds(ra, MemTy::B32);
            let bef = b.load_lds(rb2, MemTy::B32);
            b.wait_lds(0);
            b.barrier();
            (bin, bef)
        };
        // four passes: the K-th largest key `thr` and the number `rem` of elements equal to it that are needed
        let mut thr = b.c_i32(0);
        let mut rem = b.c_i32(k as i32);
        for s in [24, 16, 8, 0] {
            clear(b);
            hist_pass(b, s, thr);
            b.wait_lds(0);
            b.barrier();
            let (bin, bef) = scan_find(b, rem);
            let sh = b.bind_i32(ev(bin) << s);
            thr = b.bind_i32(ev(thr) | ev(sh));
            rem = b.bind_i32(ev(rem) - ev(bef));
        }
        // counters (thread 0, 1), then the emit pass
        let is2 = b.cmp(CmpPred::Lt, CmpTy::I32, tid, 2);
        let ca = b.bind_i32(CNT + (ev(tid) << 2));
        let ca = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(is2), a: Opnd::Val(ca), b: Opnd::Val(sink_e) });
        b.store_lds(ca, zero_i, MemTy::B32, false);
        b.wait_lds(0);
        b.barrier();
        let cgt = b.c_i32(CNT);
        let ceq = b.c_i32(CNT + 4);
        let kmr = b.bind_i32(k as i32 - ev(rem));                 // the first slot of the elements equal to thr
        b.loop_(&[(DType::I32, Opnd::I32(0))], |b, vv| {
            let vit = vv[0];
            let done = b.let_(Op::CmpGeI32(Opnd::Var(vit), Opnd::Val(iters)));
            b.break_if(done);
            let mut loaded = Vec::new();
            for u in 0..4i32 {
                let idx = b.bind_i32((ev(vit) << 12) + (u * 1024) as i32 + ev(tid));
                let off = b.bind_i32(ev(rowb) + (ev(idx) << 2));
                let x = b.buffer_load_f32(r_l, off, 0i32, 0);
                let ok = b.cmp(CmpPred::Lt, CmpTy::I32, idx, len);
                loaded.push((idx, ok, x));
            }
            for (idx, ok, x) in loaded {
                let key = key_of(b, x);
                let gt0 = b.cmp(CmpPred::Gt, CmpTy::U32, key, thr);
                let eq0 = b.cmp(CmpPred::Eq, CmpTy::I32, key, thr);
                let gt_i = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(gt0), a: Opnd::I32(1), b: Opnd::I32(0) });
                let eq_i = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(eq0), a: Opnd::I32(1), b: Opnd::I32(0) });
                let ok_i = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(ok), a: Opnd::I32(1), b: Opnd::I32(0) });
                let gt_ok = b.bind_i32(ev(gt_i) & ev(ok_i));
                let eq_ok = b.bind_i32(ev(eq_i) & ev(ok_i));
                let gt_c = b.cmp(CmpPred::Ne, CmpTy::I32, gt_ok, 0);
                let eq_c = b.cmp(CmpPred::Ne, CmpTy::I32, eq_ok, 0);
                let ag = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(gt_c), a: Opnd::Val(cgt), b: Opnd::Val(sink_e) });
                let ae = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(eq_c), a: Opnd::Val(ceq), b: Opnd::Val(sink_e) });
                let sg = b.atomic_lds(AtomicOp::AddU32, ag, 1i32, true).unwrap();
                let se = b.atomic_lds(AtomicOp::AddU32, ae, 1i32, true).unwrap();
                let eq_take = b.cmp(CmpPred::Lt, CmpTy::I32, se, rem);
                let eq_take_i = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(eq_take), a: Opnd::I32(1), b: Opnd::I32(0) });
                let eq_w = b.bind_i32(ev(eq_ok) & ev(eq_take_i));
                let pos_e = b.bind_i32(ev(kmr) + ev(se));
                let pos = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(gt_c), a: Opnd::Val(sg), b: Opnd::Val(pos_e) });
                let wr_i = b.bind_i32(ev(gt_ok) | ev(eq_w));
                let wr = b.cmp(CmpPred::Ne, CmpTy::I32, wr_i, 0);
                let o = b.bind_i32(ev(orow) + (ev(pos) << 2));
                let o = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(wr), a: Opnd::Val(o), b: Opnd::Val(oob) });
                let col = b.bind_i32(ev(start) + ev(idx));
                let cf = b.bitcast_f32(col);
                b.buffer_store_f32(r_o, o, 0i32, 0, cf);
            }
            let nx = b.let_(Op::AddI32(Opnd::Var(vit), Opnd::I32(1)));
            b.set_var(vit, nx);
        });
    });
    fb.finish()
}