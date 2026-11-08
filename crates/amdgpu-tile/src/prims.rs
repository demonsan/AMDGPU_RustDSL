//! Generic E / R / Q primitives (docs/fusion-stage-library-design.md §1, P2): elementwise maps on f32 register values, wave-wide row reductions,
//! and fp8 quantization. They emit IR directly (the stage builders wrap them in units); every function is shape-agnostic over a slice of the
//! lane's values.

use crate::shared::PrivRef;
use amdgpu_dsl::{ev, FnBuilder, Opnd, ValueId};

/// Activation of an elementwise map.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Act {
    Identity,
    Relu,
    /// x * sigmoid(x) = x / (1 + 2^(-x log2 e))
    Silu,
    /// 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))), tanh(u) = 1 - 2 / (1 + 2^(2 u log2 e))
    GeluTanh,
}

/// `act(x)` for one f32 value.
pub fn act(b: &mut FnBuilder, x: ValueId, a: Act) -> ValueId {
    match a {
        Act::Identity => x,
        Act::Relu => {
            let z = b.c_f32(0.0);
            b.max_f32(x, z)
        }
        Act::Silu => {
            let k = b.c_f32(-std::f32::consts::LOG2_E);
            let one = b.c_f32(1.0);
            let e = b.mul_f32(x, k);
            let p = b.exp2_f32(e);
            let d = b.add_f32(p, one);
            let r = b.rcp_f32(d);
            b.mul_f32(x, r)
        }
        Act::GeluTanh => {
            // u = sqrt(2/pi) x (1 + 0.044715 x^2); t = 1 - 2 / (1 + 2^(2 u log2 e)); y = 0.5 x (1 + t) = x - x / (1 + 2^(2 u log2 e))
            let x2 = b.mul_f32(x, x);
            let c1 = b.c_f32(0.044715);
            let one = b.c_f32(1.0);
            let t = b.mul_f32(x2, c1);
            let p = b.add_f32(t, one);
            let k = b.c_f32(2.0 * 0.797_884_56 * std::f32::consts::LOG2_E);
            let xp = b.mul_f32(x, p);
            let e = b.mul_f32(xp, k);
            let e2 = b.exp2_f32(e);
            let d = b.add_f32(e2, one);
            let r = b.rcp_f32(d);
            let q = b.mul_f32(x, r);
            b.sub_f32(x, q)
        }
    }
}

/// `x * scale + bias` (either may be absent).
pub fn affine(b: &mut FnBuilder, x: ValueId, scale: Option<ValueId>, bias: Option<ValueId>) -> ValueId {
    let y = match scale {
        Some(s) => b.mul_f32(x, s),
        None => x,
    };
    match bias {
        Some(c) => b.add_f32(y, c),
        None => y,
    }
}

/// The 8 bf16 values of a 16-byte fragment (Frag4 of 4 dwords, low half first) as f32 (`bf16 << 16`).
pub fn bf16x8_to_f32(b: &mut FnBuilder, frag4: ValueId) -> [ValueId; 8] {
    let mut out = [0; 8];
    for d in 0..4u8 {
        let l = b.frag4_lane(Opnd::Val(frag4), d);
        let w = b.bitcast_i32(l);
        let lo = b.bind_i32(ev(w) << 16);
        let hi = b.bind_i32(ev(w) & (0xffff_0000u32 as i32));
        out[2 * d as usize] = b.bitcast_f32(lo);
        out[2 * d as usize + 1] = b.bitcast_f32(hi);
    }
    out
}

/// Reduction operator of a row reduction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RedOp {
    Sum,
    Max,
}

fn combine(b: &mut FnBuilder, op: RedOp, x: ValueId, y: ValueId) -> ValueId {
    match op {
        RedOp::Sum => b.add_f32(x, y),
        RedOp::Max => b.max_f32(x, y),
    }
}

/// Lane-local reduction of `xs`.
pub fn lane_reduce(b: &mut FnBuilder, op: RedOp, xs: &[ValueId]) -> ValueId {
    let mut acc = xs[0];
    for &x in &xs[1..] {
        acc = combine(b, op, acc, x);
    }
    acc
}

/// Wave-wide reduction of one f32 per lane; every lane gets the result. Quads by two DPP quad permutations, then the 16 quad results through the
/// wave's private LDS scratch (`scratch`, 64 bytes at byte offset `off`, generation `gen`): every lane writes its quad's value to slot
/// `lane / 4`, then reads all 16 slots back (4 `ds_read_b128`) and combines them.
pub fn wave_reduce(b: &mut FnBuilder, op: RedOp, x: ValueId, scratch: &PrivRef, off: i32, gen: i32, lane: &crate::Lane) -> ValueId {
    let mut v = x;
    for perm in [[1u8, 0, 3, 2], [2, 3, 0, 1]] {
        let y = b.dpp_quad_perm(v, perm);
        let y = b.bitcast_f32(y);
        v = combine(b, op, v, y);
    }
    let base = scratch.base() + off;
    let wa = b.bind_i32(base.clone() + (lane.lane().lshr(2) << 2));
    let vi = b.bitcast_i32(v);
    scratch.write(b, gen, |b| b.store_lds_b32(wa, vi));
    b.wait_lds(0);
    let parts: Vec<ValueId> = scratch.read_back(b, gen, |b| {
        (0..4i32)
            .map(|q| {
                let a = b.bind_i32(base.clone() + 16 * q);
                b.load_lds_b128(a)
            })
            .collect()
    });
    let mut vals = Vec::with_capacity(16);
    for p in parts {
        for d in 0..4u8 {
            let l = b.frag4_lane(Opnd::Val(p), d);
            vals.push(b.bitcast_f32(l));
        }
    }
    lane_reduce(b, op, &vals)
}

/// Largest finite e4m3 (fp8 OCP) magnitude.
pub const FP8_E4M3_MAX: f32 = 448.0;

/// The fp8 (e4m3) row scale and its reciprocal from the absolute maximum (with `fp8_pack`: the values can be streamed between the two).
pub fn fp8_scale(b: &mut FnBuilder, amax: ValueId) -> (ValueId, ValueId) {
    let tiny = b.c_f32(1e-12);
    let a = b.max_f32(amax, tiny);
    let inv_max = b.c_f32(1.0 / FP8_E4M3_MAX);
    let scale = b.mul_f32(a, inv_max);
    let inv = b.rcp_f32(scale);
    (scale, inv)
}

/// `fp8(y * inv)` packed four per dword in value order.
pub fn fp8_pack(b: &mut FnBuilder, ys: &[ValueId], inv: ValueId) -> Vec<ValueId> {
    assert!(ys.len() % 4 == 0, "fp8_pack: a multiple of 4 values");
    ys.chunks(4)
        .map(|c| {
            let q: Vec<ValueId> = c.iter().map(|&y| b.mul_f32(y, inv)).collect();
            b.cvt_pk_fp8x4_f32(q[0], q[1], q[2], q[3])
        })
        .collect()
}

/// |x| for f32 (max(x, -x)).
pub fn abs_f32(b: &mut FnBuilder, x: ValueId) -> ValueId {
    let z = b.c_f32(0.0);
    let n = b.sub_f32(z, x);
    b.max_f32(x, n)
}

/// `1 / sqrt(x)` as `2^(-log2(x) / 2)`.
pub fn rsqrt_f32(b: &mut FnBuilder, x: ValueId) -> ValueId {
    let l = b.log2_f32(x);
    let h = b.c_f32(-0.5);
    let e = b.mul_f32(l, h);
    b.exp2_f32(e)
}
