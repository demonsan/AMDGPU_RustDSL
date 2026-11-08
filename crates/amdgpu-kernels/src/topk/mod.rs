//! Expert routing: softmax over the experts of a row, then the top-k probabilities and their indices (docs/op-coverage-memtemplates.md T6, wave-level).
//! One row per wave (4 rows per 256-thread block): lane `l` holds the experts `8 l .. 8 l + 7` (one b128), the softmax statistics and each of the K selection rounds are `wave_reduce`s
//! (round: wave max of the remaining values, then the smallest index among the lanes equal to it = max of the negated indices, then that element is masked out). Ties go to the lowest index.
//!
//! * `topk_softmax_bf16_e<E>_k<K>`         `weights[t][r]` = softmax probability of the r-th largest expert, `indices[t][r]` its expert id.
//! * `topk_softmax_renorm_bf16_e<E>_k<K>`  the same with the K selected probabilities renormalised to sum to 1.
//!
//! E a multiple of 8, 8..=512; K in 1..=16. ABI: gating @0 (bf16 [T][E]), weights @8 (f32 [T][K]), indices @16 (i32 [T][K]), T @24 (i32).

use amdgpu_dsl::{CmpPred, CmpTy, CompileProfile, DType, Ext, FnBuilder, Function, Op, Opnd, UnOp, ValueId};
use amdgpu_tile::prims::{bf16x8_to_f32, lane_reduce, wave_reduce, RedOp};
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
    (e % 8 == 0 && (8..=512).contains(&e) && (1..=16).contains(&k) && k <= e).then(|| topk_softmax(name, e, k, renorm))
}

fn sel_f32(b: &mut FnBuilder, cond: ValueId, x: ValueId, y: ValueId) -> ValueId {
    b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(cond), a: Opnd::Val(x), b: Opnd::Val(y) })
}

pub fn topk_softmax(name: &str, e: i64, k: i64, renorm: bool) -> Function {
    let mut fb = FnBuilder::new(name, 256);
    fb.compile_profile(CompileProfile::Default);
    let b = &mut fb;
    b.arg("gating", DType::Ptr, 0, 8);
    b.arg("weights", DType::Ptr, 8, 8);
    b.arg("indices", DType::Ptr, 16, 8);
    b.arg("t", DType::I32, 24, 4);
    let mt = Ext::sym("MT");
    b.sym_range("MT", 1, 1 << 20);
    b.ka_is(24, mt.clone() * Ext::c(4));
    b.grid_is(0, mt.clone());
    b.arg_tensor("gating", mt.clone() * Ext::c(4 * 2 * e));
    b.arg_tensor("weights", mt.clone() * Ext::c(4 * 4 * k));
    b.arg_tensor("indices", mt * Ext::c(4 * 4 * k));
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
    let row = b.bind_i32((ev(bx) << 2) + lane.wave());
    let wid = b.bind_i32(lane.wave());
    let me = red.mine(wid);
    let voff = b.bind_i32(ev(row) * (2 * e) as i32 + (lane.lane() << 4));
    let g = b.buffer_load_b128(r_g, voff, 0i32, 0);
    b.wait_vm(0);
    let neg = b.c_f32(-3.0e38);
    let zero = b.c_f32(0.0);
    let x0 = bf16x8_to_f32(b, g);
    let valid = if e < 512 {
        let el = b.bind_i32(lane.lane() << 3);
        Some(b.let_(Op::CmpLtI32(Opnd::Val(el), Opnd::I32(e as i32))))
    } else {
        None
    };
    let mut cur = x0;
    if let Some(c) = valid {
        for i in 0..8 {
            cur[i] = sel_f32(b, c, x0[i], neg);
        }
    }
    // softmax statistics
    let lm = lane_reduce(b, RedOp::Max, &cur);
    let m0 = wave_reduce(b, RedOp::Max, lm, &me, 0, 0, &lane);
    let l2e = b.c_f32(std::f32::consts::LOG2_E);
    let mut ex = [0; 8];
    for i in 0..8 {
        let t = b.sub_f32(cur[i], m0);
        let u = b.mul_f32(t, l2e);
        let p = b.exp2_f32(u);
        ex[i] = if let Some(c) = valid { sel_f32(b, c, p, zero) } else { p };
    }
    let ls = lane_reduce(b, RedOp::Sum, &ex);
    let sum = wave_reduce(b, RedOp::Sum, ls, &me, 64, 1, &lane);
    let inv = b.rcp_f32(sum);
    // the negated expert index of each element (as f32), lane-local
    let nidx: Vec<ValueId> = (0..8)
        .map(|i| {
            let ni = b.bind_i32(lane.lane() * (-8) - i as i32);
            b.un(UnOp::CvtF32I32, ni)
        })
        .collect();
    let big_neg = b.c_f32(-1.0e9);
    let mut ws: Vec<ValueId> = Vec::new();
    let mut idxs: Vec<ValueId> = Vec::new();
    for r in 0..k as i32 {
        let m = if r == 0 {
            m0
        } else {
            let lm = lane_reduce(b, RedOp::Max, &cur);
            wave_reduce(b, RedOp::Max, lm, &me, 0, 2 * r, &lane)
        };
        // the lowest index among the elements equal to the maximum: max of the negated indices
        let cand: Vec<ValueId> = (0..8)
            .map(|i| {
                let eq = b.cmp(CmpPred::Eq, CmpTy::F32, cur[i], m);
                sel_f32(b, eq, nidx[i], big_neg)
            })
            .collect();
        let lc = lane_reduce(b, RedOp::Max, &cand);
        let sel = wave_reduce(b, RedOp::Max, lc, &me, 64, 2 * r + 1, &lane);
        for i in 0..8 {
            let hit = b.cmp(CmpPred::Eq, CmpTy::F32, nidx[i], sel);
            cur[i] = sel_f32(b, hit, neg, cur[i]);
        }
        let t = b.sub_f32(m, m0);
        let u = b.mul_f32(t, l2e);
        let p = b.exp2_f32(u);
        ws.push(b.mul_f32(p, inv));
        let zf = b.c_f32(0.0);
        let idx_f = b.sub_f32(zf, sel);
        idxs.push(b.un(UnOp::CvtI32F32, idx_f));
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
    // lane 0 of each wave stores the row's K results (the other lanes store out of range)
    let lv = b.bind_i32(lane.lane());
    let l0 = b.let_(Op::CmpLtI32(Opnd::Val(lv), Opnd::I32(1)));
    let wbase = b.bind_i32(ev(row) * (4 * k) as i32);
    let ok = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(l0), a: Opnd::Val(wbase), b: Opnd::I32(-16) });
    for r in 0..k as usize {
        let w = if let Some(s) = scale { b.mul_f32(ws[r], s) } else { ws[r] };
        b.buffer_store_f32(r_w, ok, 0i32, 4 * r as u32, w);
        let ib = b.bitcast_f32(idxs[r]);
        b.buffer_store_f32(r_i, ok, 0i32, 4 * r as u32, ib);
    }
    fb.finish()
}