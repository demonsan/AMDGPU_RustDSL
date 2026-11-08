//! DeepSeek-V3 style expert gate (aiter `moe_fused_gate`, SGLang): `moe_fused_gate_bf16_e<E>_g<G>_tg<TG>_k<K>`.
//!
//! For every token row of `scores [M][E]` (bf16 logits): `s = sigmoid(x)` (NaN -> 0), `b = s + bias[E]`; an expert group (E / G consecutive experts) scores the sum of its two largest `b`; the
//! `G - TG` lowest-scoring groups are dropped (ties: the lowest group index goes first); the K largest `b` among the remaining experts are chosen (ties: the lowest expert index); the weights
//! are `s / sum(s over the chosen)`. Outputs `topk_weights f32 [M][K]`, `topk_ids i32 [M][K]` in selection order. (No fused shared experts, no padding rows.)
//!
//! One lane per (row, group): a lane holds the VPT = E / G experts of its group (VPT in {8, 16, 24, 32}), so a wave handles 64 / G rows and every reduction is a DPP all-reduce over the G lanes of
//! a row (G in {4, 8, 16, 32, 64}). ABI: scores @0, bias @8 (bf16 [E]), weights @16, ids @24, M @32 (i32). Block: 256 threads, grid = ceil(M / (4 * 64 / G)).

use amdgpu_dsl::{CachePolicy, CmpPred, CmpTy, CompileProfile, DType, DppAluOp, Ext, FnBuilder, Function, MemTy, Op, Opnd, ValueId};
use amdgpu_tile::prims::bf16x8_to_f32;
use amdgpu_tile::tileop::group_allreduce;
use amdgpu_tile::ev;

pub fn by_name(name: &str) -> Option<Function> {
    let rest = name.strip_prefix("moe_fused_gate_bf16_e")?;
    let (e, rest) = rest.split_once("_g")?;
    let (g, rest) = rest.split_once("_tg")?;
    let (tg, k) = rest.split_once("_k")?;
    let (e, g, tg, k): (i64, i64, i64, i64) = (e.parse().ok()?, g.parse().ok()?, tg.parse().ok()?, k.parse().ok()?);
    let vpt = if g > 0 { e / g } else { 0 };
    ([4, 8, 16, 32, 64].contains(&g) && e % g == 0 && [8, 16, 24, 32].contains(&vpt) && (1..=g).contains(&tg) && (1..=32).contains(&k) && k <= tg * vpt).then(|| moe_fused_gate(name, e, g, tg, k))
}

fn fsel(b: &mut FnBuilder, c: ValueId, x: ValueId, y: ValueId) -> ValueId {
    b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(c), a: Opnd::Val(x), b: Opnd::Val(y) })
}
fn isel(b: &mut FnBuilder, c: ValueId, x: ValueId, y: ValueId) -> ValueId {
    b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(c), a: Opnd::Val(x), b: Opnd::Val(y) })
}

pub fn moe_fused_gate(name: &str, e: i64, g: i64, tg: i64, k: i64) -> Function {
    let vpt = e / g;
    let rpw = 64 / g;                       // rows per wave
    let rpb = 4 * rpw;                      // rows per block
    let mut fb = FnBuilder::new(name, 256);
    fb.compile_profile(CompileProfile::Default);
    let b = &mut fb;
    b.arg("scores", DType::Ptr, 0, 8);
    b.arg("bias", DType::Ptr, 8, 8);
    b.arg("weights", DType::Ptr, 16, 8);
    b.arg("ids", DType::Ptr, 24, 8);
    b.arg("m", DType::I32, 32, 4);
    let mt = Ext::sym("MT");
    b.sym_range("MT", 1, 1 << 20);
    b.ka_is(32, mt.clone() * Ext::c(rpb));
    b.grid_is(0, mt.clone());
    b.arg_tensor("scores", mt.clone() * Ext::c(rpb * 2 * e));
    b.arg_tensor("bias", Ext::c(2 * e));
    b.arg_tensor("weights", mt.clone() * Ext::c(rpb * 4 * k));
    b.arg_tensor("ids", mt * Ext::c(rpb * 4 * k));
    let (p_s, p_b, p_w, p_i) = (b.let_(Op::KernargPtr { offset: 0 }), b.let_(Op::KernargPtr { offset: 8 }), b.let_(Op::KernargPtr { offset: 16 }), b.let_(Op::KernargPtr { offset: 24 }));
    let m_v = b.let_(Op::KernargI32 { offset: 32 });
    let tid = b.let_(Op::WorkitemId);
    let bx = b.let_(Op::WorkgroupIdX);
    let r_s = { let v = b.bind_i32(ev(m_v) * (2 * e) as i32); let v = b.read_first_lane(v); b.buffer_desc_dyn(p_s, v) };
    let r_w = { let v = b.bind_i32(ev(m_v) * (4 * k) as i32); let v = b.read_first_lane(v); b.buffer_desc_dyn(p_w, v) };
    let r_i = { let v = b.bind_i32(ev(m_v) * (4 * k) as i32); let v = b.read_first_lane(v); b.buffer_desc_dyn(p_i, v) };
    let r_b = b.buffer_desc(p_b, (2 * e) as u32);
    let lane = b.bind_i32(ev(tid) & 63);
    let wave = b.bind_i32(ev(tid).lshr(6));
    let lg = b.bind_i32(ev(lane) & (g as i32 - 1));                      // the group (lane) within the row
    let rw = b.bind_i32(ev(lane).lshr(g.trailing_zeros() as i32));        // the row within the wave
    let row = b.bind_i32(ev(bx) * rpb as i32 + ev(wave) * rpw as i32 + ev(rw));
    let row_ok = b.let_(Op::CmpLtI32(Opnd::Val(row), Opnd::Val(m_v)));
    // loads: this group's VPT scores and biases
    let base = b.bind_i32(ev(row) * (2 * e) as i32 + ev(lg) * (2 * vpt) as i32);
    let bbase = b.bind_i32(ev(lg) * (2 * vpt) as i32);
    let mut xs = Vec::new();
    let mut bs = Vec::new();
    for c in 0..vpt / 8 {
        let o = b.bind_i32(ev(base) + 16 * c as i32);
        xs.push(b.buffer_load_c(r_s, Some(Opnd::Val(o)), 0i32, 0, MemTy::B128, CachePolicy::NT));
        let ob = b.bind_i32(ev(bbase) + 16 * c as i32);
        bs.push(b.buffer_load_c(r_b, Some(Opnd::Val(ob)), 0i32, 0, MemTy::B128, CachePolicy::NONE));
    }
    let neg_log2e = b.c_f32(-std::f32::consts::LOG2_E);
    let one = b.c_f32(1.0);
    let zero = b.c_f32(0.0);
    let flt_max = b.c_f32(3.402_823_5e38);
    let neg_flt_max = b.c_f32(-3.402_823_5e38);
    let mut s: Vec<ValueId> = Vec::new();   // sigmoid scores
    let mut bi: Vec<ValueId> = Vec::new();  // biased scores
    for c in 0..(vpt / 8) as usize {
        let xf = bf16x8_to_f32(b, xs[c]);
        let bf = bf16x8_to_f32(b, bs[c]);
        for q in 0..8 {
            let t = b.mul_f32(xf[q], neg_log2e);
            let ex = b.exp2_f32(t);
            let d = b.add_f32(ex, one);
            let sg = b.rcp_f32(d);
            let isnum = b.cmp(CmpPred::Eq, CmpTy::F32, xf[q], xf[q]);
            let sg = fsel(b, isnum, sg, zero);
            s.push(sg);
            bi.push(b.add_f32(sg, bf[q]));
        }
    }
    // group score: the two largest biased scores of the group
    let (mut m1, mut m2) = (neg_flt_max, neg_flt_max);
    for i in 0..vpt as usize {
        let gt1 = b.cmp(CmpPred::Gt, CmpTy::F32, bi[i], m1);
        let gt2 = b.cmp(CmpPred::Gt, CmpTy::F32, bi[i], m2);
        let alt = fsel(b, gt2, bi[i], m2);
        m2 = fsel(b, gt1, m1, alt);
        m1 = fsel(b, gt1, bi[i], m1);
    }
    let gscore = b.add_f32(m1, m2);
    // drop the G - TG lowest groups (lowest index first on ties)
    let big = b.c_i32(i32::MAX);
    let mut excl = b.c_i32(0);
    for _ in 0..(g - tg) {
        let one_i = b.c_i32(1);
        let is_ex = b.cmp(CmpPred::Ne, CmpTy::I32, excl, 0);
        let cur = fsel(b, is_ex, flt_max, gscore);
        let mn = group_allreduce(b, DppAluOp::MinF32, cur, g);
        let at_min = b.cmp(CmpPred::Eq, CmpTy::F32, cur, mn);
        let key = isel(b, at_min, lg, big);
        let chosen = group_allreduce(b, DppAluOp::MinI32, key, g);
        let hit = b.cmp(CmpPred::Eq, CmpTy::I32, lg, chosen);
        excl = isel(b, hit, one_i, excl);
    }
    let excluded = b.cmp(CmpPred::Ne, CmpTy::I32, excl, 0);
    // K rounds: the largest biased score of the kept groups (lowest expert index on ties)
    let mut ids: Vec<ValueId> = Vec::new();
    let mut ss: Vec<ValueId> = Vec::new();
    let lgv = b.bind_i32(ev(lg) * vpt as i32);       // the first expert of this lane's group
    for _ in 0..k {
        let mut m = neg_flt_max;
        let mut j = b.c_i32(0);
        for i in 0..vpt as usize {
            let gt = b.cmp(CmpPred::Gt, CmpTy::F32, bi[i], m);
            m = fsel(b, gt, bi[i], m);
            let ic = b.c_i32(i as i32);
            j = isel(b, gt, ic, j);
        }
        m = fsel(b, excluded, neg_flt_max, m);
        let zi = b.c_i32(0);
        j = isel(b, excluded, zi, j);
        let mx = group_allreduce(b, DppAluOp::MaxF32, m, g);
        let ex = b.bind_i32(ev(lgv) + ev(j));
        let at_max = b.cmp(CmpPred::Eq, CmpTy::F32, m, mx);
        let key = isel(b, at_max, ex, big);
        let sel = group_allreduce(b, DppAluOp::MinI32, key, g);
        // the owner lane clears the element and contributes its sigmoid score
        let tgt = b.bind_i32(ev(sel) - ev(lgv));
        let mut acc = zero;
        for i in 0..vpt as usize {
            let hit = b.cmp(CmpPred::Eq, CmpTy::I32, tgt, i as i32);
            acc = fsel(b, hit, s[i], acc);
            bi[i] = fsel(b, hit, neg_flt_max, bi[i]);
        }
        let sv = group_allreduce(b, DppAluOp::AddF32, acc, g);
        ids.push(sel);
        ss.push(sv);
    }
    let mut sum = zero;
    for &v in &ss {
        sum = b.add_f32(sum, v);
    }
    let rs = b.rcp_f32(sum);
    let ws: Vec<ValueId> = ss.iter().map(|&v| b.mul_f32(v, rs)).collect();
    // store: lane lg writes the entries kk = lg + t * G
    let oob = b.c_i32(0x7fff_ff00);
    for t in 0..((k + g - 1) / g) {
        let kk = b.bind_i32(ev(lg) + (t * g) as i32);
        let mut wv = zero;
        let zi = b.c_i32(0);
        let mut iv = zi;
        for q in 0..k as usize {
            let hit = b.cmp(CmpPred::Eq, CmpTy::I32, kk, q as i32);
            wv = fsel(b, hit, ws[q], wv);
            iv = isel(b, hit, ids[q], iv);
        }
        let off = b.bind_i32(ev(row) * (4 * k) as i32 + (ev(kk) << 2));
        let kk_ok = b.cmp(CmpPred::Lt, CmpTy::I32, kk, k as i32);
        let o1 = isel(b, kk_ok, off, oob);
        let o2 = isel(b, row_ok, o1, oob);
        b.buffer_store_f32(r_w, o2, 0i32, 0, wv);
        let ivf = b.bitcast_f32(iv);
        b.buffer_store_f32(r_i, o2, 0i32, 0, ivf);
    }
    fb.finish()
}