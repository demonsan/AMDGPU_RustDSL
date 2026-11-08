//! RoPE (docs/op-coverage-memtemplates.md T7): `x [T][H][D]` bf16, `positions [T]` i32, `cos_sin [P][D]` bf16 (per position: cos in the first D/2 elements, sin in the last D/2), out-of-place `out [T][H][D]`.
//! One grid x per token; a block of `min(4, ceil(H * L / 64))` waves rotates `waves * 64 / L` heads at a time (L lanes per head row); all loads are issued before any compute. Positions >= P read zeros (cos = sin = 0).
//!
//! * `rope_neox_bf16_h<H>_d<D>`: rotate half (partner of element i is i + D/2), L = D/16 lanes per row, two b128 chunks per lane.
//! * `rope_gptj_bf16_h<H>_d<D>`: interleaved pairs (2i, 2i+1), L = D/8 lanes per row, one b128 chunk per lane.
//!
//! D in {64, 128, 256} (full rotary dimension), H in 1..=256. ABI: x @0, out @8, positions @16, cos_sin @24, T @32 (i32), P @36 (i32).

use amdgpu_dsl::{CachePolicy, MemTy, CompileProfile, DType, Ext, FnBuilder, Function, Op, Opnd, ValueId};
use amdgpu_tile::prims::{bf16x8_to_f32, group_reduce_dpp, lane_reduce, rsqrt_f32, RedOp};
use amdgpu_tile::ev;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeStyle {
    Neox,
    Gptj,
}

pub fn by_name(name: &str) -> Option<Function> {
    // `qknorm_rope_*`: a per-head RMSNorm (weight gamma [D], eps) of every head row, then the rotation (docs: fused_qk_norm_rope of aiter, Qwen3-style); ABI adds gamma @40 (bf16 [D]) and eps @48 (f32)
    let (name_s, norm) = match name.strip_prefix("qknorm_") {
        Some(n) => (n, true),
        None => (name, false),
    };
    let (rest, style) = if let Some(r) = name_s.strip_prefix("rope_neox_bf16_h") {
        (r, RopeStyle::Neox)
    } else if let Some(r) = name_s.strip_prefix("rope_gptj_bf16_h") {
        (r, RopeStyle::Gptj)
    } else {
        return None;
    };
    let (h, rest) = rest.split_once("_d")?;
    // `_r<R>`: partial rotation, only the first R elements of a head rotate (default: all D)
    let (d, r) = match rest.split_once("_r") {
        Some((d, r)) => (d, Some(r)),
        None => (rest, None),
    };
    let (h, d): (i64, i64) = (h.parse().ok()?, d.parse().ok()?);
    let r: i64 = match r {
        Some(r) => r.parse().ok()?,
        None => d,
    };
    ((1..=256).contains(&h) && [64, 128, 256].contains(&d) && r % 16 == 0 && (16..=d).contains(&r) && (r == d || r < d) && !(norm && r != d)).then(|| rope_kernel(name, h, d, style, r, norm))
}

fn bf16x4_to_f32(b: &mut FnBuilder, frag2: ValueId) -> [ValueId; 4] {
    let mut out = [0; 4];
    for d in 0..2u8 {
        let l = b.frag2_lane(Opnd::Val(frag2), d);
        let w = b.bitcast_i32(l);
        let lo = b.bind_i32(ev(w) << 16);
        let hi = b.bind_i32(ev(w) & (0xffff_0000u32 as i32));
        out[2 * d as usize] = b.bitcast_f32(lo);
        out[2 * d as usize + 1] = b.bitcast_f32(hi);
    }
    out
}

fn pack8(b: &mut FnBuilder, r: &[ValueId; 8]) -> ValueId {
    let w: Vec<ValueId> = (0..4).map(|k| b.cvt_pk_bf16_f32(r[2 * k], r[2 * k + 1])).collect();
    b.frag4(w[0], w[1], w[2], w[3])
}

pub fn rope_kernel(name: &str, h: i64, d: i64, style: RopeStyle, r: i64, norm: bool) -> Function {
    let l = if style == RopeStyle::Neox { d / 16 } else { d / 8 }; // lanes per head row
    let rows_per_wave = 64 / l;
    // waves per block: only as many as the heads need (a block of idle waves would waste occupancy), at most 4
    // a token with few lanes (H * L < 64, a power of two) leaves most of a wave idle: pack `tpw` tokens into every wave (H = 1, D = 64: 16 tokens per wave)
    let hl = h * l;
    let packed = hl <= 32 && (hl & (hl - 1)) == 0;
    let tpw = if packed { 64 / hl } else { 1 };
    let nwaves = if packed { 4 } else { ((h + rows_per_wave - 1) / rows_per_wave).min(4) };
    let heads_per_pass = if packed { h } else { nwaves * rows_per_wave };
    let passes = (h + heads_per_pass - 1) / heads_per_pass;
    let mut fb = FnBuilder::new(name, (64 * nwaves) as u32);
    fb.compile_profile(CompileProfile::Default);
    let b = &mut fb;
    b.arg("x", DType::Ptr, 0, 8);
    b.arg("out", DType::Ptr, 8, 8);
    b.arg("pos", DType::Ptr, 16, 8);
    b.arg("cos_sin", DType::Ptr, 24, 8);
    b.arg("t", DType::I32, 32, 4);
    b.arg("p", DType::I32, 36, 4);
    if norm {
        b.arg("gamma", DType::Ptr, 40, 8);
        b.arg("eps", DType::F32, 48, 4);
        b.arg_tensor("gamma", Ext::c(2 * d));
    }
    let tt = Ext::sym("T");
    let pp = Ext::sym("P");
    b.sym_range("T", 1, 1 << 24);
    b.sym_range("P", 1, 1 << 24);
    b.ka_is(32, tt.clone());
    b.ka_is(36, pp.clone());
    b.grid_is(0, if packed { tt.clone().ceil_div(4 * tpw) } else { tt.clone() });
    b.arg_tensor("x", tt.clone() * Ext::c(2 * h * d));
    b.arg_tensor("out", tt.clone() * Ext::c(2 * h * d));
    b.arg_tensor("pos", tt * Ext::c(4));
    b.arg_tensor("cos_sin", pp * Ext::c(2 * r));
    let (p_x, p_o, p_pos, p_cs) = (b.let_(Op::KernargPtr { offset: 0 }), b.let_(Op::KernargPtr { offset: 8 }), b.let_(Op::KernargPtr { offset: 16 }), b.let_(Op::KernargPtr { offset: 24 }));
    let t_v = b.let_(Op::KernargI32 { offset: 32 });
    let p_v = b.let_(Op::KernargI32 { offset: 36 });
    let tid = b.let_(Op::WorkitemId);
    let bx = b.let_(Op::WorkgroupIdX);
    let wave = b.bind_i32(ev(tid).lshr(6));
    let lane = b.bind_i32(ev(tid) & 63);
    // the token of this lane: the block id (uniform), or, packed, `tpw` tokens per wave, 4 waves per block (a vector value: every lane group has its own position / table row)
    let tok = if packed { b.bind_i32(ev(bx) * (4 * tpw) as i32 + ev(wave) * tpw as i32 + ev(lane).lshr(hl.trailing_zeros() as i32)) } else { bx };
    let r_x = { let v = b.bind_i32(ev(t_v) * (2 * h * d) as i32); let v = b.read_first_lane(v); b.buffer_desc_dyn(p_x, v) };
    let r_o = { let v = b.bind_i32(ev(t_v) * (2 * h * d) as i32); let v = b.read_first_lane(v); b.buffer_desc_dyn(p_o, v) };
    let r_pos = { let v = b.bind_i32(ev(t_v) << 2); let v = b.read_first_lane(v); b.buffer_desc_dyn(p_pos, v) };
    let r_cs = { let v = b.bind_i32(ev(p_v) * (2 * r) as i32); let v = b.read_first_lane(v); b.buffer_desc_dyn(p_cs, v) };
    // this token's position (uniform): positions[token], then the cos / sin row of the table
    let voff_pos = b.bind_i32(ev(tok) << 2);
    let posf = b.buffer_load_f32(r_pos, voff_pos, 0i32, 0);
    let posi = b.bitcast_i32(posf);
    let pos = if packed { posi } else { b.read_first_lane(posi) };
    let cs_row = b.bind_i32(ev(pos) * (2 * r) as i32);
    let lr = b.bind_i32(ev(lane) & (l - 1) as i32);       // lane within the head row (l is a power of two)
    let row_in_wave = b.bind_i32(ev(lane).lshr(l.trailing_zeros() as i32));
    // cos / sin: lane `lr`'s part of the row, shared by every head
    let cs_lane = b.bind_i32(ev(cs_row) + if style == RopeStyle::Neox { ev(lr) << 4 } else { ev(lr) << 3 });
    let cs_sin = b.bind_i32(ev(cs_lane) + r as i32);
    let (cos, sin): (Vec<ValueId>, Vec<ValueId>) = if style == RopeStyle::Neox {
        let c = b.buffer_load_b128(r_cs, cs_lane, 0i32, 0);
        let s = b.buffer_load_b128(r_cs, cs_sin, 0i32, 0);
        (bf16x8_to_f32(b, c).to_vec(), bf16x8_to_f32(b, s).to_vec())
    } else {
        let c = b.buffer_load_b64(r_cs, cs_lane, 0i32, 0);
        let s = b.buffer_load_b64(r_cs, cs_sin, 0i32, 0);
        (bf16x4_to_f32(b, c).to_vec(), bf16x4_to_f32(b, s).to_vec())
    };
    // the per-head RMSNorm weights (this lane's chunks) and constants
    let (gl, gh, eps_v) = if norm {
        let p_g = b.let_(Op::KernargPtr { offset: 40 });
        let r_g = b.buffer_desc(p_g, (2 * d) as u32);
        let go = b.bind_i32(ev(lr) << 4);
        let gl_raw = b.buffer_load_c(r_g, Some(Opnd::Val(go)), 0i32, 0, MemTy::B128, CachePolicy::NONE);
        let gh_raw = if style == RopeStyle::Neox {
            let goh = b.bind_i32(ev(go) + d as i32);
            Some(b.buffer_load_c(r_g, Some(Opnd::Val(goh)), 0i32, 0, MemTy::B128, CachePolicy::NONE))
        } else {
            None
        };
        let gl = bf16x8_to_f32(b, gl_raw);
        let gh = gh_raw.map(|g| bf16x8_to_f32(b, g));
        let eps = b.let_(Op::KernargF32 { offset: 48 });
        (Some(gl), gh, Some(eps))
    } else {
        (None, None, None)
    };
    let inv_d = b.c_f32(1.0 / d as f32);
    let partial = r < d;
    let rot = if partial {
        let lim = if style == RopeStyle::Neox { r / 16 } else { r / 8 };
        Some(b.let_(Op::CmpLtI32(Opnd::Val(lr), Opnd::I32(lim as i32))))
    } else {
        None
    };
    let (cos, sin) = if let Some(rot) = rot {
        let one = b.c_f32(1.0);
        let zero = b.c_f32(0.0);
        let sel = |b: &mut FnBuilder, v: &[ValueId], fill: ValueId| -> Vec<ValueId> {
            v.iter().map(|&x| b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(rot), a: Opnd::Val(x), b: Opnd::Val(fill) })).collect()
        };
        (sel(b, &cos, one), sel(b, &sin, zero))
    } else {
        (cos, sin)
    };
    // all the passes' loads first
    struct Pass { head_ok: ValueId, off: ValueId, offh: Option<ValueId>, lo: ValueId, hi: Option<ValueId> }
    let mut ps: Vec<Pass> = Vec::new();
    for ps_i in 0..passes {
        let head = if packed {
            b.bind_i32(ev(row_in_wave) & (h as i32 - 1))
        } else {
            b.bind_i32(ev(wave) * rows_per_wave as i32 + ev(row_in_wave) + (ps_i * heads_per_pass) as i32)
        };
        // (packed: a lane group whose token is past the last one stores nothing)
        let head_ok = if packed { b.let_(Op::CmpLtI32(Opnd::Val(tok), Opnd::Val(t_v))) } else { b.let_(Op::CmpLtI32(Opnd::Val(head), Opnd::I32(h as i32))) };
        let row_off = b.bind_i32(ev(tok) * (2 * h * d) as i32 + ev(head) * (2 * d) as i32);
        // the lane's chunk: lanes that rotate take elements 8 lr.. (and, NeoX, the partner R / 2 further); the pass-through lanes of a partial rotation take a pair of adjacent chunks past R (identity cos / sin)
        let off = if let (true, Some(rotc), true) = (partial, rot, style == RopeStyle::Neox) {
            let pass_off = b.bind_i32(ev(lr) * 32);   // 2R + 32 (lr - R/16) bytes = 32 lr
            let rot_off = b.bind_i32(ev(lr) << 4);
            let o = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(rotc), a: Opnd::Val(rot_off), b: Opnd::Val(pass_off) });
            b.bind_i32(ev(row_off) + ev(o))
        } else {
            b.bind_i32(ev(row_off) + (ev(lr) << 4))
        };
        let lo = b.buffer_load_c(r_x, Some(Opnd::Val(off)), 0i32, 0, MemTy::B128, CachePolicy::NT);
        let mut offh_v: Option<ValueId> = None;
        let hi = if style == RopeStyle::Neox {
            let delta = if let Some(rotc) = rot {
                let rd = b.c_i32(r as i32);
                let pd = b.c_i32(16);
                b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(rotc), a: Opnd::Val(rd), b: Opnd::Val(pd) })
            } else {
                b.c_i32(d as i32)
            };
            let offh = b.bind_i32(ev(off) + ev(delta));
            offh_v = Some(offh);
            Some(b.buffer_load_c(r_x, Some(Opnd::Val(offh)), 0i32, 0, MemTy::B128, CachePolicy::NT))
        } else {
            None
        };
        ps.push(Pass { head_ok, off, offh: offh_v, lo, hi });
    }
    for p in &ps {
        let mut xl = bf16x8_to_f32(b, p.lo);
        let mut xh_n: Option<[ValueId; 8]> = p.hi.map(|h| bf16x8_to_f32(b, h));
        if norm {
            // sum of squares of the head row (this lane's 8 or 16 elements, then the L lanes of the head), mean + eps, rsqrt, times gamma
            let mut sq: Vec<ValueId> = xl.iter().map(|&x| b.mul_f32(x, x)).collect();
            if let Some(xh) = &xh_n {
                sq.extend(xh.iter().map(|&x| b.mul_f32(x, x)));
            }
            let part = lane_reduce(b, RedOp::Sum, &sq);
            let tot = group_reduce_dpp(b, RedOp::Sum, part, l);
            let mean = b.mul_f32(tot, inv_d);
            let me = b.add_f32(mean, eps_v.unwrap());
            let inv = rsqrt_f32(b, me);
            let gl = gl.unwrap();
            for k in 0..8 {
                let t = b.mul_f32(xl[k], inv);
                xl[k] = b.mul_f32(t, gl[k]);
            }
            if let (Some(xh), Some(gh)) = (xh_n.as_mut(), gh.as_ref()) {
                for k in 0..8 {
                    let t = b.mul_f32(xh[k], inv);
                    xh[k] = b.mul_f32(t, gh[k]);
                }
            }
        }
        let (yl, yh);
        if style == RopeStyle::Neox {
            let xh = xh_n.unwrap();
            let mut a = [0; 8];
            let mut c = [0; 8];
            for k in 0..8 {
                let t1 = b.mul_f32(xl[k], cos[k]);
                let t2 = b.mul_f32(xh[k], sin[k]);
                a[k] = b.sub_f32(t1, t2);
                let t3 = b.mul_f32(xh[k], cos[k]);
                let t4 = b.mul_f32(xl[k], sin[k]);
                c[k] = b.add_f32(t3, t4);
            }
            yl = a;
            yh = Some(c);
        } else {
            let mut a = [0; 8];
            for j in 0..4 {
                let (x0, x1) = (xl[2 * j], xl[2 * j + 1]);
                let t1 = b.mul_f32(x0, cos[j]);
                let t2 = b.mul_f32(x1, sin[j]);
                a[2 * j] = b.sub_f32(t1, t2);
                let t3 = b.mul_f32(x1, cos[j]);
                let t4 = b.mul_f32(x0, sin[j]);
                a[2 * j + 1] = b.add_f32(t3, t4);
            }
            yl = a;
            yh = None;
        }
        let masked = |b: &mut FnBuilder, o: ValueId| b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(p.head_ok), a: Opnd::Val(o), b: Opnd::I32(-16) });
        let o_lo = masked(b, p.off);
        let f = pack8(b, &yl);
        b.buffer_store_c(r_o, Some(Opnd::Val(o_lo)), 0i32, 0, f, MemTy::B128, false, CachePolicy::NT);
        if let Some(yh) = yh {
            let o_hi = masked(b, p.offh.unwrap());
            let f = pack8(b, &yh);
            b.buffer_store_c(r_o, Some(Opnd::Val(o_hi)), 0i32, 0, f, MemTy::B128, false, CachePolicy::NT);
        }
    }
    fb.finish()
}