//! Quantization kernels, one row per wave like `rows` (docs/op-coverage-memtemplates.md T2 / T3): lane `l` holds the elements `8 (l + 64 i) + k` of the row (one b128 per chunk `i`), fp8 (e4m3) values
//! are stored 8 bytes per lane and chunk. 4 waves per 256-thread block, grid = ceil(M / 4).
//!
//! * `quant_token_bf16_fp8_n<N>`   per-token dynamic: `scale[m] = max|x[m]| / 448`, `q = fp8(x / scale)`.    ABI x @0, q @8, scale @16 (f32 [M]), M @24 (i32)
//! * `quant_group128_bf16_fp8_n<N>` per-group (128 elements = 16 lanes) dynamic: `scale[m][g]` f32 [M][N / 128] (N a multiple of 128).  same ABI, scale [M][N/128]
//! * `situv2_mul_quant_fp8_d<D>`    aiter `situv2_and_mul_quant` (group_size == D): `y = beta tanh(g / beta) sigmoid(g) * linear_beta tanh(u / linear_beta)` with g = x[m][:D], u = x[m][D:], then per-token fp8. ABI + beta @28, linear_beta @32 (f32).
//! * `silu_mul_quant_fp8_d<D>`     `y = silu(x[m][:D]) * x[m][D:]` (bf16 `[M][2D]`) then per-token dynamic fp8 of y.        same ABI, q [M][D], scale [M]
//!
//! N (D) a multiple of 8, 8..=32768; chunks past the row end are masked in the statistics and their stores dropped by the buffer descriptor.
//!
//! Rows up to 8192 columns keep the whole row in registers (one pass over the data). Longer rows (`quant_kernel_long`: 8192 < N <= 32768, e.g. the Llama-8B MLP D = 14336) stream the row twice in groups of
//! `LONG_G` chunks per lane with the next group's loads in flight: pass 1 reduces the amax, pass 2 re-reads (an L2 hit: the row was read microseconds earlier) and quantizes; the group form (`Group128`) is one pass.

use amdgpu_dsl::{CachePolicy, MemTy, CompileProfile, DType, Ext, FnBuilder, Function, Op, Opnd, TriOp, UnOp, ValueId};
use amdgpu_tile::prims::{act, abs_f32, bf16x8_to_f32, fp8_pack, fp8_scale, lane_reduce, wave_reduce, Act, RedOp};
use amdgpu_tile::{ev, Lane, LdsPlan};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantKind {
    Token,
    Group128,
    SiluMulToken,
    /// SiTUv2 gate * up (aiter `situv2_and_mul_quant`), then per-token fp8; beta / linear_beta kernargs
    SituMulToken,
    /// per-token int8: `scale = max|x| / 127`, `q = clamp(rne(x / scale), -127, 127)`
    TokenInt8,
}

pub fn by_name(name: &str) -> Option<Function> {
    let (rest, k, mult) = if let Some(r) = name.strip_prefix("quant_token_bf16_int8_n") {
        (r, QuantKind::TokenInt8, 8)
    } else if let Some(r) = name.strip_prefix("quant_group128_bf16_fp8_n") {
        (r, QuantKind::Group128, 128)
    } else if let Some(r) = name.strip_prefix("quant_token_bf16_fp8_n") {
        (r, QuantKind::Token, 8)
    } else if let Some(r) = name.strip_prefix("situv2_mul_quant_fp8_d") {
        (r, QuantKind::SituMulToken, 8)
    } else if let Some(r) = name.strip_prefix("silu_mul_quant_fp8_d") {
        (r, QuantKind::SiluMulToken, 8)
    } else {
        return None;
    };
    let n: i64 = rest.parse().ok()?;
    (n % mult == 0 && (8..=32768).contains(&n)).then(|| quant_kernel(name, n, k))
}

/// The int8 row scale (`max(amax, 1e-12) / 127`) and its reciprocal.
fn int8_scale(b: &mut FnBuilder, amax: ValueId) -> (ValueId, ValueId) {
    let tiny = b.c_f32(1e-12);
    let a = b.max_f32(amax, tiny);
    let k = b.c_f32(1.0 / 127.0);
    let scale = b.mul_f32(a, k);
    let inv = b.rcp_f32(scale);
    (scale, inv)
}

/// `clamp(rne(y * inv), -127, 127)` as int8, four per dword in value order.
fn int8_pack(b: &mut FnBuilder, ys: &[ValueId; 8], inv: ValueId) -> Vec<ValueId> {
    let lo = b.c_i32(-127);
    let hi = b.c_i32(127);
    let q: Vec<ValueId> = ys
        .iter()
        .map(|&y| {
            let t = b.mul_f32(y, inv);
            let r = b.un(UnOp::RndneF32, t);
            let i = b.un(UnOp::CvtI32F32, r);
            b.tri(TriOp::Med3I32, i, lo, hi)
        })
        .collect();
    q.chunks(4).map(|c| b.bind_i32((ev(c[0]) & 255) + ((ev(c[1]) & 255) << 8) + ((ev(c[2]) & 255) << 16) + (ev(c[3]) << 24))).collect()
}

fn sel_f32(b: &mut FnBuilder, cond: ValueId, x: ValueId, y: ValueId) -> ValueId {
    b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(cond), a: Opnd::Val(x), b: Opnd::Val(y) })
}

/// Maximum over the 16 lanes of a group (lanes 16 g .. 16 g + 15): two DPP quad permutations, then the four quad values through the wave's private LDS scratch (`gen` distinguishes the uses).
fn group16_max(b: &mut FnBuilder, x: ValueId, scratch: &amdgpu_tile::shared::PrivRef, gen: i32, lane: &Lane) -> ValueId {
    let mut v = x;
    for perm in [[1u8, 0, 3, 2], [2, 3, 0, 1]] {
        let y = b.dpp_quad_perm(v, perm);
        let y = b.bitcast_f32(y);
        v = b.max_f32(v, y);
    }
    let base = scratch.base();
    let wa = b.bind_i32(base.clone() + (lane.lane().lshr(2) << 2));
    let vi = b.bitcast_i32(v);
    scratch.write(b, gen, |b| b.store_lds_b32(wa, vi));
    b.wait_lds(0);
    let part = scratch.read_back(b, gen, |b| {
        let a = b.bind_i32(base.clone() + (lane.lane().lshr(4) << 4));
        vec![b.load_lds_b128(a)]
    });
    let vals: Vec<ValueId> = (0..4u8)
        .map(|d| {
            let l = b.frag4_lane(Opnd::Val(part[0]), d);
            b.bitcast_f32(l)
        })
        .collect();
    lane_reduce(b, RedOp::Max, &vals)
}

/// Chunks (b128 per lane) per streamed group of the long-row form.
const LONG_G: usize = 8;

/// The values to quantize of one chunk (8 f32): the bf16 input, or `act(gate) * up` / the SiTUv2 form for the two-input kinds; the lanes past the row end (`valid` false) are zeroed (they must not enter any maximum).
fn chunk_y(b: &mut FnBuilder, situ: Option<(ValueId, ValueId, ValueId, ValueId)>, xa: ValueId, x2: Option<ValueId>, valid: Option<ValueId>, zero: ValueId) -> [ValueId; 8] {
    let a = bf16x8_to_f32(b, xa);
    let mut y = a;
    if let Some(x2) = x2 {
        let g = bf16x8_to_f32(b, x2);
        for k in 0..8 {
            if let Some((beta, lbeta, gm, um)) = situ {
                let one = b.c_f32(1.0);
                let two = b.c_f32(2.0);
                let nl = b.c_f32(-std::f32::consts::LOG2_E);
                let (eg, ig, iu) = (b.mul_f32(a[k], gm), b.mul_f32(a[k], nl), b.mul_f32(g[k], um));
                let (xg, xs, xu) = (b.exp2_f32(eg), b.exp2_f32(ig), b.exp2_f32(iu));
                let (dg, ds, du) = (b.add_f32(one, xg), b.add_f32(one, xs), b.add_f32(one, xu));
                let (rg, rs, ru) = (b.rcp_f32(dg), b.rcp_f32(ds), b.rcp_f32(du));
                // tanh = 2 r - 1; value = beta tanh_g sigmoid_g * linear_beta tanh_u
                let tg = b.let_(Op::FmaF32(Opnd::Val(two), Opnd::Val(rg), Opnd::F32(-1.0)));
                let tu = b.let_(Op::FmaF32(Opnd::Val(two), Opnd::Val(ru), Opnd::F32(-1.0)));
                let l = b.mul_f32(beta, tg);
                let l = b.mul_f32(l, rs);
                let r = b.mul_f32(lbeta, tu);
                y[k] = b.mul_f32(l, r);
            } else {
                let s = act(b, a[k], Act::Silu);
                y[k] = b.mul_f32(s, g[k]);
            }
        }
    }
    if let Some(c) = valid {
        for k in 0..8 {
            y[k] = sel_f32(b, c, y[k], zero);
        }
    }
    y
}

pub fn quant_kernel(name: &str, n: i64, kind: QuantKind) -> Function {
    if n > 8192 {
        return quant_kernel_long(name, n, kind);
    }
    let nld = ((n + 511) / 512) as usize;
    let mut fb = FnBuilder::new(name, 256);
    fb.compile_profile(CompileProfile::Default);
    let b = &mut fb;
    b.arg("x", DType::Ptr, 0, 8);
    b.arg("q", DType::Ptr, 8, 8);
    b.arg("scale", DType::Ptr, 16, 8);
    b.arg("m", DType::I32, 24, 4);
    if kind == QuantKind::SituMulToken {
        b.arg("beta", DType::F32, 28, 4);
        b.arg("linear_beta", DType::F32, 32, 4);
    }
    let mt = Ext::sym("MT");
    b.sym_range("MT", 1, 1 << 20);
    b.ka_is(24, mt.clone() * Ext::c(4));
    b.grid_is(0, mt.clone());
    let two_in = matches!(kind, QuantKind::SiluMulToken | QuantKind::SituMulToken);
    let in_row = if two_in { 4 * n } else { 2 * n };
    let n_scales = if kind == QuantKind::Group128 { n / 128 } else { 1 };
    b.arg_tensor("x", mt.clone() * Ext::c(4 * in_row));
    b.arg_tensor("q", mt.clone() * Ext::c(4 * n));
    b.arg_tensor("scale", mt * Ext::c(4 * 4 * n_scales));
    let (p_x, p_q, p_s) = (b.let_(Op::KernargPtr { offset: 0 }), b.let_(Op::KernargPtr { offset: 8 }), b.let_(Op::KernargPtr { offset: 16 }));
    let m_v = b.let_(Op::KernargI32 { offset: 24 });
    let tid = b.let_(Op::WorkitemId);
    let lane = Lane::from_tid(b, tid);
    let bx = b.let_(Op::WorkgroupIdX);
    let mut lds = LdsPlan::new();
    let red = lds.private("RED", 128, 4, &[]);
    lds.commit(b);
    let r_x = { let v = b.bind_i32(ev(m_v) * in_row as i32); let v = b.read_first_lane(v); b.buffer_desc_dyn(p_x, v) };
    let r_q = { let v = b.bind_i32(ev(m_v) * n as i32); let v = b.read_first_lane(v); b.buffer_desc_dyn(p_q, v) };
    let r_s = { let v = b.bind_i32(ev(m_v) * (4 * n_scales) as i32); let v = b.read_first_lane(v); b.buffer_desc_dyn(p_s, v) };
    let row = b.bind_i32((ev(bx) << 2) + lane.wave());
    let wid = b.bind_i32(lane.wave());
    let me = red.mine(wid);
    let ngrp = nld.div_ceil(4);
    let vx = b.bind_i32(ev(row) * in_row as i32 + (lane.lane() << 4));
    let gx: Vec<ValueId> = (0..ngrp).map(|g| if g == 0 { vx } else { b.bind_i32(ev(vx) + 4096 * g as i32) }).collect();
    let xs: Vec<ValueId> = (0..nld).map(|i| b.buffer_load_c(r_x, Some(Opnd::Val(gx[i / 4])), 0i32, (1024 * (i % 4)) as u32, MemTy::B128, CachePolicy::NT)).collect();
    let x2s: Option<Vec<ValueId>> = two_in.then(|| {
        let v2 = b.bind_i32(ev(vx) + (2 * n) as i32);
        let g2: Vec<ValueId> = (0..ngrp).map(|g| if g == 0 { v2 } else { b.bind_i32(ev(v2) + 4096 * g as i32) }).collect();
        (0..nld).map(|i| b.buffer_load_c(r_x, Some(Opnd::Val(g2[i / 4])), 0i32, (1024 * (i % 4)) as u32, MemTy::B128, CachePolicy::NT)).collect()
    });
    b.wait_vm(0);
    let valid: Vec<Option<ValueId>> = (0..nld)
        .map(|i| {
            if (i as i64 + 1) * 512 <= n {
                None
            } else {
                let e = b.bind_i32((lane.lane() << 3) + 512 * i as i32);
                Some(b.let_(Op::CmpLtI32(Opnd::Val(e), Opnd::I32(n as i32))))
            }
        })
        .collect();
    let zero = b.c_f32(0.0);
    // SiTUv2 constants: tanh(x / beta) = 2 / (1 + 2^(-2 log2e x / beta)) - 1
    let situ = (kind == QuantKind::SituMulToken).then(|| {
        let (beta, lbeta) = (b.let_(Op::KernargF32 { offset: 28 }), b.let_(Op::KernargF32 { offset: 32 }));
        let (rb, rl) = (b.rcp_f32(beta), b.rcp_f32(lbeta));
        let c = b.c_f32(-2.0 * std::f32::consts::LOG2_E);
        let (gm, um) = (b.mul_f32(c, rb), b.mul_f32(c, rl));
        (beta, lbeta, gm, um)
    });
    // the values to quantize (f32), 8 per chunk; the tail lanes of a partial chunk are zeroed (they must not enter any maximum)
    let mut ys: Vec<[ValueId; 8]> = Vec::with_capacity(nld);
    for i in 0..nld {
        let y = chunk_y(b, situ, xs[i], x2s.as_ref().map(|v| v[i]), valid[i], zero);
        ys.push(y);
    }
    let abs8 = |b: &mut FnBuilder, y: &[ValueId; 8]| -> Vec<ValueId> { y.iter().map(|&v| abs_f32(b, v)).collect() };
    let store_q = |b: &mut FnBuilder, i: usize, inv: ValueId, y: &[ValueId; 8]| {
        let words = if kind == QuantKind::TokenInt8 { int8_pack(b, y, inv) } else { fp8_pack(b, y, inv) };
        let f2 = b.frag2(words[0], words[1]);
        let base = b.bind_i32(ev(row) * n as i32 + (lane.lane() << 3) + 512 * i as i32);
        let voff = match valid[i] {
            None => base,
            Some(c) => b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(c), a: Opnd::Val(base), b: Opnd::I32(-16) }),
        };
        b.buffer_store_c(r_q, Some(Opnd::Val(voff)), 0i32, 0, f2, MemTy::B64, false, CachePolicy::NT);
    };
    match kind {
        QuantKind::Token | QuantKind::SiluMulToken | QuantKind::SituMulToken | QuantKind::TokenInt8 => {
            let mut part: Option<ValueId> = None;
            for i in 0..nld {
                let a = abs8(b, &ys[i]);
                let m = lane_reduce(b, RedOp::Max, &a);
                part = Some(match part { None => m, Some(p) => b.max_f32(p, m) });
            }
            let amax = wave_reduce(b, RedOp::Max, part.unwrap(), &me, 0, 0, &lane);
            let (s, inv) = if kind == QuantKind::TokenInt8 { int8_scale(b, amax) } else { fp8_scale(b, amax) };
            for i in 0..nld {
                store_q(b, i, inv, &ys[i]);
            }
            let vs = b.bind_i32(ev(row) << 2);
            b.buffer_store_f32(r_s, vs, 0i32, 0, s);
        }
        QuantKind::Group128 => {
            for i in 0..nld {
                let a = abs8(b, &ys[i]);
                let m = lane_reduce(b, RedOp::Max, &a);
                let gmax = group16_max(b, m, &me, i as i32, &lane);
                let (s, inv) = fp8_scale(b, gmax);
                store_q(b, i, inv, &ys[i]);
                // one scale per group: [row][4 i + lane / 16] (all 16 lanes of a group store the same value; tail groups are dropped)
                let g = b.bind_i32(ev(row) * (n / 128) as i32 + 4 * i as i32 + lane.lane().lshr(4));
                let off = b.bind_i32(ev(g) << 2);
                let off = match valid[i] {
                    None => off,
                    Some(c) => b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(c), a: Opnd::Val(off), b: Opnd::I32(-16) }),
                };
                b.buffer_store_f32(r_s, off, 0i32, 0, s);
            }
        }
    }
    fb.finish()
}
/// The long-row form (8192 < N <= 32768): one wave per row, the row streamed in groups of `LONG_G` chunks per lane with the next group's loads issued before the current group is processed (counted `vmcnt`).
/// Per-token kinds: pass 1 reduces the row amax (nothing is kept), pass 2 re-reads the row (the first group's loads are issued before the amax reduction) and quantizes; `Group128` is a single pass.
fn quant_kernel_long(name: &str, n: i64, kind: QuantKind) -> Function {
    let nld = ((n + 511) / 512) as usize;
    let ngroups = nld.div_ceil(LONG_G);
    let mut fb = FnBuilder::new(name, 256);
    fb.compile_profile(CompileProfile::Default);
    let b = &mut fb;
    b.arg("x", DType::Ptr, 0, 8);
    b.arg("q", DType::Ptr, 8, 8);
    b.arg("scale", DType::Ptr, 16, 8);
    b.arg("m", DType::I32, 24, 4);
    if kind == QuantKind::SituMulToken {
        b.arg("beta", DType::F32, 28, 4);
        b.arg("linear_beta", DType::F32, 32, 4);
    }
    let mt = Ext::sym("MT");
    b.sym_range("MT", 1, 1 << 20);
    b.ka_is(24, mt.clone() * Ext::c(4));
    b.grid_is(0, mt.clone());
    let two_in = matches!(kind, QuantKind::SiluMulToken | QuantKind::SituMulToken);
    let in_row = if two_in { 4 * n } else { 2 * n };
    let n_scales = if kind == QuantKind::Group128 { n / 128 } else { 1 };
    b.arg_tensor("x", mt.clone() * Ext::c(4 * in_row));
    b.arg_tensor("q", mt.clone() * Ext::c(4 * n));
    b.arg_tensor("scale", mt * Ext::c(4 * 4 * n_scales));
    let (p_x, p_q, p_s) = (b.let_(Op::KernargPtr { offset: 0 }), b.let_(Op::KernargPtr { offset: 8 }), b.let_(Op::KernargPtr { offset: 16 }));
    let m_v = b.let_(Op::KernargI32 { offset: 24 });
    let tid = b.let_(Op::WorkitemId);
    let lane = Lane::from_tid(b, tid);
    let bx = b.let_(Op::WorkgroupIdX);
    let mut lds = LdsPlan::new();
    let red = lds.private("RED", 128, 4, &[]);
    lds.commit(b);
    let r_x = { let v = b.bind_i32(ev(m_v) * in_row as i32); let v = b.read_first_lane(v); b.buffer_desc_dyn(p_x, v) };
    let r_q = { let v = b.bind_i32(ev(m_v) * n as i32); let v = b.read_first_lane(v); b.buffer_desc_dyn(p_q, v) };
    let r_s = { let v = b.bind_i32(ev(m_v) * (4 * n_scales) as i32); let v = b.read_first_lane(v); b.buffer_desc_dyn(p_s, v) };
    let row = b.bind_i32((ev(bx) << 2) + lane.wave());
    let wid = b.bind_i32(lane.wave());
    let me = red.mine(wid);
    let vx = b.bind_i32(ev(row) * in_row as i32 + (lane.lane() << 4));
    // chunk `i` of the row: byte offset vx + 1024 i (a base per 4 chunks + the immediate offset)
    let nb4 = nld.div_ceil(4);
    let gx: Vec<ValueId> = (0..nb4).map(|g| if g == 0 { vx } else { b.bind_i32(ev(vx) + 4096 * g as i32) }).collect();
    let v2 = two_in.then(|| b.bind_i32(ev(vx) + (2 * n) as i32));
    let g2: Vec<ValueId> = match v2 {
        Some(v2) => (0..nb4).map(|g| if g == 0 { v2 } else { b.bind_i32(ev(v2) + 4096 * g as i32) }).collect(),
        None => Vec::new(),
    };
    let valid: Vec<Option<ValueId>> = (0..nld)
        .map(|i| {
            if (i as i64 + 1) * 512 <= n {
                None
            } else {
                let e = b.bind_i32((lane.lane() << 3) + 512 * i as i32);
                Some(b.let_(Op::CmpLtI32(Opnd::Val(e), Opnd::I32(n as i32))))
            }
        })
        .collect();
    let zero = b.c_f32(0.0);
    let situ = (kind == QuantKind::SituMulToken).then(|| {
        let (beta, lbeta) = (b.let_(Op::KernargF32 { offset: 28 }), b.let_(Op::KernargF32 { offset: 32 }));
        let (rb, rl) = (b.rcp_f32(beta), b.rcp_f32(lbeta));
        let c = b.c_f32(-2.0 * std::f32::consts::LOG2_E);
        let (gm, um) = (b.mul_f32(c, rb), b.mul_f32(c, rl));
        (beta, lbeta, gm, um)
    });
    // the loads of group `g` (chunks g LONG_G ..): (first chunk, the b128 of each chunk, the second input's)
    let issue = |b: &mut FnBuilder, g: usize| -> (usize, Vec<ValueId>, Vec<ValueId>) {
        let (c0, c1) = (g * LONG_G, ((g + 1) * LONG_G).min(nld));
        let xs: Vec<ValueId> = (c0..c1).map(|i| b.buffer_load_c(r_x, Some(Opnd::Val(gx[i / 4])), 0i32, (1024 * (i % 4)) as u32, MemTy::B128, CachePolicy::NT)).collect();
        let x2s: Vec<ValueId> = if two_in { (c0..c1).map(|i| b.buffer_load_c(r_x, Some(Opnd::Val(g2[i / 4])), 0i32, (1024 * (i % 4)) as u32, MemTy::B128, CachePolicy::NT)).collect() } else { Vec::new() };
        (c0, xs, x2s)
    };
    // the loads one group issues (the counted wait of the previous group lets them stay in flight)
    let nloads = |g: usize| -> u8 { ((((g + 1) * LONG_G).min(nld) - g * LONG_G) * if two_in { 2 } else { 1 }) as u8 };
    let ys_of = |b: &mut FnBuilder, grp: &(usize, Vec<ValueId>, Vec<ValueId>)| -> Vec<(usize, [ValueId; 8])> {
        let (c0, xs, x2s) = grp;
        (0..xs.len()).map(|j| (c0 + j, chunk_y(b, situ, xs[j], if two_in { Some(x2s[j]) } else { None }, valid[c0 + j], zero))).collect()
    };
    let abs8 = |b: &mut FnBuilder, y: &[ValueId; 8]| -> Vec<ValueId> { y.iter().map(|&v| abs_f32(b, v)).collect() };
    let store_q = |b: &mut FnBuilder, i: usize, inv: ValueId, y: &[ValueId; 8]| {
        let words = if kind == QuantKind::TokenInt8 { int8_pack(b, y, inv) } else { fp8_pack(b, y, inv) };
        let f2 = b.frag2(words[0], words[1]);
        let base = b.bind_i32(ev(row) * n as i32 + (lane.lane() << 3) + 512 * i as i32);
        let voff = match valid[i] {
            None => base,
            Some(c) => b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(c), a: Opnd::Val(base), b: Opnd::I32(-16) }),
        };
        b.buffer_store_c(r_q, Some(Opnd::Val(voff)), 0i32, 0, f2, MemTy::B64, false, CachePolicy::NT);
    };
    // one streamed pass: the next group's loads before the current group's work; `prefetched` = the first group's loads were issued by the caller
    match kind {
        QuantKind::Group128 => {
            let mut cur = issue(b, 0);
            for g in 0..ngroups {
                let nxt = (g + 1 < ngroups).then(|| issue(b, g + 1));
                b.wait_vm(if g + 1 < ngroups { nloads(g + 1) } else { 0 });
                for (i, y) in ys_of(b, &cur) {
                    let a = abs8(b, &y);
                    let m = lane_reduce(b, RedOp::Max, &a);
                    let gmax = group16_max(b, m, &me, i as i32, &lane);
                    let (s, inv) = fp8_scale(b, gmax);
                    store_q(b, i, inv, &y);
                    let gi = b.bind_i32(ev(row) * (n / 128) as i32 + 4 * i as i32 + lane.lane().lshr(4));
                    let off = b.bind_i32(ev(gi) << 2);
                    let off = match valid[i] {
                        None => off,
                        Some(c) => b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(c), a: Opnd::Val(off), b: Opnd::I32(-16) }),
                    };
                    b.buffer_store_f32(r_s, off, 0i32, 0, s);
                }
                if let Some(nx) = nxt {
                    cur = nx;
                }
            }
        }
        _ => {
            // pass 1: the row amax
            let mut part: Option<ValueId> = None;
            let mut cur = issue(b, 0);
            for g in 0..ngroups {
                let nxt = (g + 1 < ngroups).then(|| issue(b, g + 1));
                b.wait_vm(if g + 1 < ngroups { nloads(g + 1) } else { 0 });
                for (_, y) in ys_of(b, &cur) {
                    let a = abs8(b, &y);
                    let m = lane_reduce(b, RedOp::Max, &a);
                    part = Some(match part { None => m, Some(p) => b.max_f32(p, m) });
                }
                if let Some(nx) = nxt {
                    cur = nx;
                }
            }
            // pass 2 starts loading before the amax reduction
            let mut cur = issue(b, 0);
            let amax = wave_reduce(b, RedOp::Max, part.unwrap(), &me, 0, 0, &lane);
            let (s, inv) = if kind == QuantKind::TokenInt8 { int8_scale(b, amax) } else { fp8_scale(b, amax) };
            for g in 0..ngroups {
                let nxt = (g + 1 < ngroups).then(|| issue(b, g + 1));
                // the previous group's stores are younger than this group's loads: they complete first (in-order vmcnt) - the wait is conservative
                b.wait_vm(if g + 1 < ngroups { nloads(g + 1) } else { 0 });
                for (i, y) in ys_of(b, &cur) {
                    store_q(b, i, inv, &y);
                }
                if let Some(nx) = nxt {
                    cur = nx;
                }
            }
            let vs = b.bind_i32(ev(row) << 2);
            b.buffer_store_f32(r_s, vs, 0i32, 0, s);
        }
    }
    fb.finish()
}