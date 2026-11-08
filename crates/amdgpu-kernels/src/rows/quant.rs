//! Quantization kernels, one row per wave like `rows` (docs/op-coverage-memtemplates.md T2 / T3): lane `l` holds the elements `8 (l + 64 i) + k` of the row (one b128 per chunk `i`), fp8 (e4m3) values
//! are stored 8 bytes per lane and chunk. 4 waves per 256-thread block, grid = ceil(M / 4).
//!
//! * `quant_token_bf16_fp8_n<N>`   per-token dynamic: `scale[m] = max|x[m]| / 448`, `q = fp8(x / scale)`.    ABI x @0, q @8, scale @16 (f32 [M]), M @24 (i32)
//! * `quant_group128_bf16_fp8_n<N>` per-group (128 elements = 16 lanes) dynamic: `scale[m][g]` f32 [M][N / 128] (N a multiple of 128).  same ABI, scale [M][N/128]
//! * `silu_mul_quant_fp8_d<D>`     `y = silu(x[m][:D]) * x[m][D:]` (bf16 `[M][2D]`) then per-token dynamic fp8 of y.        same ABI, q [M][D], scale [M]
//!
//! N (D) a multiple of 8, 8..=8192; chunks past the row end are masked in the statistics and their stores dropped by the buffer descriptor.

use amdgpu_dsl::{CachePolicy, MemTy, CompileProfile, DType, Ext, FnBuilder, Function, Op, Opnd, TriOp, UnOp, ValueId};
use amdgpu_tile::prims::{act, abs_f32, bf16x8_to_f32, fp8_pack, fp8_scale, lane_reduce, wave_reduce, Act, RedOp};
use amdgpu_tile::{ev, Lane, LdsPlan};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantKind {
    Token,
    Group128,
    SiluMulToken,
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
    } else if let Some(r) = name.strip_prefix("silu_mul_quant_fp8_d") {
        (r, QuantKind::SiluMulToken, 8)
    } else {
        return None;
    };
    let n: i64 = rest.parse().ok()?;
    (n % mult == 0 && (8..=8192).contains(&n)).then(|| quant_kernel(name, n, k))
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

pub fn quant_kernel(name: &str, n: i64, kind: QuantKind) -> Function {
    let nld = ((n + 511) / 512) as usize;
    let mut fb = FnBuilder::new(name, 256);
    fb.compile_profile(CompileProfile::Default);
    let b = &mut fb;
    b.arg("x", DType::Ptr, 0, 8);
    b.arg("q", DType::Ptr, 8, 8);
    b.arg("scale", DType::Ptr, 16, 8);
    b.arg("m", DType::I32, 24, 4);
    let mt = Ext::sym("MT");
    b.sym_range("MT", 1, 1 << 20);
    b.ka_is(24, mt.clone() * Ext::c(4));
    b.grid_is(0, mt.clone());
    let in_row = if kind == QuantKind::SiluMulToken { 4 * n } else { 2 * n };
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
    let x2s: Option<Vec<ValueId>> = (kind == QuantKind::SiluMulToken).then(|| {
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
    // the values to quantize (f32), 8 per chunk; the tail lanes of a partial chunk are zeroed (they must not enter any maximum)
    let mut ys: Vec<[ValueId; 8]> = Vec::with_capacity(nld);
    for i in 0..nld {
        let a = bf16x8_to_f32(b, xs[i]);
        let mut y = a;
        if let Some(x2) = &x2s {
            let g = bf16x8_to_f32(b, x2[i]);
            for k in 0..8 {
                let s = act(b, a[k], Act::Silu);
                y[k] = b.mul_f32(s, g[k]);
            }
        }
        if let Some(c) = valid[i] {
            for k in 0..8 {
                y[k] = sel_f32(b, c, y[k], zero);
            }
        }
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
        QuantKind::Token | QuantKind::SiluMulToken | QuantKind::TokenInt8 => {
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