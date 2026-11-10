//! Per-row top-K of f32 logits (aiter `top_k_per_row_decode` / `top_k_per_row_prefill`, the sparse-attention indexer): the K largest values of each row, as column indices.
//!
//! * `topk_per_row_decode_f32_k<K>`:  ABI logits @0 (f32 [R][stride]), seq_lens @8 (i32 [B]), indices @16 (i32 [R][K]), stride @24 (i32 elements), next_n @28 (i32), rows @32 (i32, R = B * next_n).
//!   Grid x = B, y = next_n; the row `b * next_n + j` has `L = seq_lens[b] - next_n + j + 1` valid columns (0 .. L).
//! * `topk_plain_f32_k<K>`: every row has `len` valid columns: ABI logits @0, indices @8, len @16, stride @20, rows @24 (i32); grid x = rows.
//! * `topk_per_row_prefill_f32_k<K>`: ABI logits @0, row_starts @8 (i32 [R]), row_ends @16 (i32 [R]), indices @24, stride @32, rows @36. Grid x = R; the valid columns are [start, end), the indices are absolute columns.
//!
//! Output: `indices[r][0 .. min(K, L))` = the columns of the K largest values (unordered; ties at the K-th value are taken lowest-slot-first by an LDS counter, i.e. an arbitrary subset), and `-1` after
//! them when `L < K`. A row of `L <= K` columns is returned whole (column order). K in 1..=2048.
//!
//! One block of 1024 threads per row: a four-pass radix select on the order-preserving 32-bit key of the f32 (8 bits per pass, LDS histogram atomics, the 256-bin descending scan of the sampling
//! kernels: DPP wave scan + 4 wave totals) finds the K-th largest key T and `rem` = how many elements equal to T are needed; a last pass emits every element > T (LDS counter) and the first
//! `rem` elements equal to T. The row is read five times (the first from HBM, the others mostly from L2).

use amdgpu_dsl::{AtomicOp, CachePolicy, CmpPred, CmpTy, CompileProfile, DType, Dpp, DppAluOp, DppCtrl, Ext, FnBuilder, Function, MemTy, Op, Opnd, ValueId};
use crate::lds_decl::tile_at;
use amdgpu_stage::templates::select::{DescScan, ScanThreads, SelectVer};
use amdgpu_tile::{ev, SlotIdx};

pub fn by_name(name: &str) -> Option<Function> {
    let (rest, prefill, plain) = if let Some(r) = name.strip_prefix("topk_per_row_decode_f32_k") {
        (r, false, false)
    } else if let Some(r) = name.strip_prefix("topk_per_row_prefill_f32_k") {
        (r, true, false)
    } else if let Some(r) = name.strip_prefix("topk_plain_f32_k") {
        (r, false, true)
    } else {
        return None;
    };
    let k: i64 = rest.parse().ok()?;
    (1..=2048).contains(&k).then(|| topk_per_row_mode(name, k, prefill, plain))
}

pub fn topk_per_row(name: &str, k: i64, prefill: bool) -> Function {
    topk_per_row_mode(name, k, prefill, false)
}

// LDS: histogram 320 slots (256 bins + a per-lane sink 256 + lane), wave totals, result slots, two counters, the emit sink (one slot per thread)
// (the histogram starts at 64: threads >= 256 read "bin -1" in the scan, which stays inside the segment; their value is masked)
const HIST: i32 = 64;
const WT: i32 = 1344;
const RES: i32 = 1408;
const CNT: i32 = 1424;
const SINK: i32 = 1440;
const LDS: u32 = 1440 + 4096;

pub fn topk_per_row_mode(name: &str, k: i64, prefill: bool, plain: bool) -> Function {
    let mut fb = FnBuilder::new(name, 1024);
    fb.compile_profile(CompileProfile::Default);
    let b = &mut fb;
    b.arg("logits", DType::Ptr, 0, 8);
    let (idx_off, st_off) = if plain {
        // every row has `len` valid columns (a row pitch of `stride`)
        b.arg("indices", DType::Ptr, 8, 8);
        b.arg("len", DType::I32, 16, 4);
        b.arg("stride", DType::I32, 20, 4);
        b.arg("rows", DType::I32, 24, 4);
        (8u32, 20u32)
    } else if prefill {
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
    let rows_off = if plain { 24u32 } else if prefill { 36u32 } else { 32u32 };
    let rr = Ext::sym("R");
    let ss = Ext::sym("S");
    b.sym_range("R", 1, 1 << 24);
    b.sym_range("S", 1, 1 << 26);
    b.ka_is(rows_off, rr.clone());
    b.ka_is(st_off, ss.clone());
    if prefill || plain {
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
    if plain {
    } else if prefill {
        b.arg_tensor("row_starts", rr.clone() * Ext::c(4));
        b.arg_tensor("row_ends", rr * Ext::c(4));
    } else {
        b.arg_tensor("seq_lens", rr * Ext::c(4));
    }
    b.lds(LDS);
    // declared tiles (every LDS access is inside a protocol marker: `lds-untracked` 0); the histogram tile starts one word early (the "bin -1" read of the threads >= 256, masked)
    let t_hist = tile_at(b, "HIST", (HIST - 4) as i64, 1284);
    let t_wt = tile_at(b, "WT", WT as i64, 64);
    let t_rb = tile_at(b, "RB", RES as i64, 12);
    let t_cs = tile_at(b, "CS", CNT as i64, 16 + 4096);   // the two counters + the emit sink (one atomic target: the counter or the thread's own sink slot)
    let ver = SelectVer::new();   // a fresh HIST version per histogram build, a fresh RB / WT version per scan_find (L2 template `select`)
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
    let (row, start, len) = if plain {
        let l0 = b.let_(Op::KernargI32 { offset: 16 });
        (bx, zero_i, l0)
    } else if prefill {
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
                let vit_v = b.let_(Op::AddI32(Opnd::Var(vit), Opnd::I32(0)));
                let mut loaded = Vec::new();
                for u in 0..4i32 {
                    let idx = b.bind_i32((ev(vit_v) << 12) + (u * 1024) as i32 + ev(tid));
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
                    t_hist.at(&SlotIdx::new(0, ver.hist())).write(b, |b| b.atomic_lds(AtomicOp::AddU32, addr, 1i32, false));
                }
                let nx = b.let_(Op::AddI32(Opnd::Var(vit), Opnd::I32(1)));
                b.set_var(vit, nx);
            });
        };
        // the histogram clear and the descending scan of the 256 bins (thread t < 256 owns bin 255 - t): (bin, count of elements in higher bins) of the bin where the cumulative count reaches `tgt`
        // (L2 template `select`, docs/l2-templates.md T9; the HIST tile starts one word early so the idle threads' bin -1 read stays inside it: no clamp)
        let th = ScanThreads { tid, tcl, tcl4, wave, zero: zero_i };
        let desc = DescScan { hist: &t_hist, wt: &t_wt, rb: &t_rb, hist_base: HIST, wt_base: WT, res_base: RES, clamp_bin: false, norm_hit: false };
        let clear = |b: &mut FnBuilder| desc.clear(b, &th, &ver);
        let scan_find = |b: &mut FnBuilder, tgt: ValueId| -> (ValueId, ValueId) { desc.scan_find(b, &th, &ver, tgt, false) };
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
        t_cs.at(&SlotIdx::new(0, 0)).write(b, |b| b.store_lds(ca, zero_i, MemTy::B32, false));
        b.wait_lds(0);
        b.barrier();
        let cgt = b.c_i32(CNT);
        let ceq = b.c_i32(CNT + 4);
        let kmr = b.bind_i32(k as i32 - ev(rem));                 // the first slot of the elements equal to thr
        b.loop_(&[(DType::I32, Opnd::I32(0))], |b, vv| {
            let vit = vv[0];
            let done = b.let_(Op::CmpGeI32(Opnd::Var(vit), Opnd::Val(iters)));
            b.break_if(done);
            let vit_v = b.let_(Op::AddI32(Opnd::Var(vit), Opnd::I32(0)));
            let mut loaded = Vec::new();
            for u in 0..4i32 {
                let idx = b.bind_i32((ev(vit_v) << 12) + (u * 1024) as i32 + ev(tid));
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
                // the returning counter atomics are read-modify-writes of the counter / sink words: produce (same version as the clear)
                let sg = t_cs.at(&SlotIdx::new(0, 0)).write(b, |b| b.atomic_lds(AtomicOp::AddU32, ag, 1i32, true)).unwrap();
                let se = t_cs.at(&SlotIdx::new(0, 0)).write(b, |b| b.atomic_lds(AtomicOp::AddU32, ae, 1i32, true)).unwrap();
                let (sg, se) = (b.bitcast_i32(sg), b.bitcast_i32(se));
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