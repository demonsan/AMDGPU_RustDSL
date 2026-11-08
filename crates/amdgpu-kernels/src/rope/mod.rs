//! RoPE (docs/op-coverage-memtemplates.md T7): `x [T][H][D]` bf16, `positions [T]` i32, `cos_sin [P][D]` bf16 (per position: cos in the first D/2 elements, sin in the last D/2), out-of-place `out [T][H][D]`.
//! One grid x per token; a block of `min(4, ceil(H * L / 64))` waves rotates `waves * 64 / L` heads at a time (L lanes per head row); all loads are issued before any compute. Positions >= P read zeros (cos = sin = 0).
//!
//! * `rope_neox_bf16_h<H>_d<D>`: rotate half (partner of element i is i + D/2), L = D/16 lanes per row, two b128 chunks per lane.
//! * `rope_gptj_bf16_h<H>_d<D>`: interleaved pairs (2i, 2i+1), L = D/8 lanes per row, one b128 chunk per lane.
//!
//! D in {64, 128, 256} (full rotary dimension), H in 1..=256. ABI: x @0, out @8, positions @16, cos_sin @24, T @32 (i32), P @36 (i32).

use amdgpu_dsl::{CompileProfile, DType, Ext, FnBuilder, Function, Op, Opnd, ValueId};
use amdgpu_tile::prims::bf16x8_to_f32;
use amdgpu_tile::ev;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeStyle {
    Neox,
    Gptj,
}

pub fn by_name(name: &str) -> Option<Function> {
    let (rest, style) = if let Some(r) = name.strip_prefix("rope_neox_bf16_h") {
        (r, RopeStyle::Neox)
    } else if let Some(r) = name.strip_prefix("rope_gptj_bf16_h") {
        (r, RopeStyle::Gptj)
    } else {
        return None;
    };
    let (h, d) = rest.split_once("_d")?;
    let (h, d): (i64, i64) = (h.parse().ok()?, d.parse().ok()?);
    ((1..=256).contains(&h) && [64, 128, 256].contains(&d)).then(|| rope_kernel(name, h, d, style))
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

pub fn rope_kernel(name: &str, h: i64, d: i64, style: RopeStyle) -> Function {
    let l = if style == RopeStyle::Neox { d / 16 } else { d / 8 }; // lanes per head row
    let rows_per_wave = 64 / l;
    // waves per block: only as many as the heads need (a block of idle waves would waste occupancy), at most 4
    let nwaves = ((h + rows_per_wave - 1) / rows_per_wave).min(4);
    let heads_per_pass = nwaves * rows_per_wave;
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
    let tt = Ext::sym("T");
    let pp = Ext::sym("P");
    b.sym_range("T", 1, 1 << 24);
    b.sym_range("P", 1, 1 << 24);
    b.ka_is(32, tt.clone());
    b.ka_is(36, pp.clone());
    b.grid_is(0, tt.clone());
    b.arg_tensor("x", tt.clone() * Ext::c(2 * h * d));
    b.arg_tensor("out", tt.clone() * Ext::c(2 * h * d));
    b.arg_tensor("pos", tt * Ext::c(4));
    b.arg_tensor("cos_sin", pp * Ext::c(2 * d));
    let (p_x, p_o, p_pos, p_cs) = (b.let_(Op::KernargPtr { offset: 0 }), b.let_(Op::KernargPtr { offset: 8 }), b.let_(Op::KernargPtr { offset: 16 }), b.let_(Op::KernargPtr { offset: 24 }));
    let t_v = b.let_(Op::KernargI32 { offset: 32 });
    let p_v = b.let_(Op::KernargI32 { offset: 36 });
    let tid = b.let_(Op::WorkitemId);
    let tok = b.let_(Op::WorkgroupIdX);
    let r_x = { let v = b.bind_i32(ev(t_v) * (2 * h * d) as i32); let v = b.read_first_lane(v); b.buffer_desc_dyn(p_x, v) };
    let r_o = { let v = b.bind_i32(ev(t_v) * (2 * h * d) as i32); let v = b.read_first_lane(v); b.buffer_desc_dyn(p_o, v) };
    let r_pos = { let v = b.bind_i32(ev(t_v) << 2); let v = b.read_first_lane(v); b.buffer_desc_dyn(p_pos, v) };
    let r_cs = { let v = b.bind_i32(ev(p_v) * (2 * d) as i32); let v = b.read_first_lane(v); b.buffer_desc_dyn(p_cs, v) };
    // this token's position (uniform): positions[token], then the cos / sin row of the table
    let voff_pos = b.bind_i32(ev(tok) << 2);
    let posf = b.buffer_load_f32(r_pos, voff_pos, 0i32, 0);
    let posi = b.bitcast_i32(posf);
    let pos = b.read_first_lane(posi);
    let cs_row = b.bind_i32(ev(pos) * (2 * d) as i32);
    let wave = b.bind_i32(ev(tid).lshr(6));
    let lane = b.bind_i32(ev(tid) & 63);
    let lr = b.bind_i32(ev(lane) & (l - 1) as i32);       // lane within the head row (l is a power of two)
    let row_in_wave = b.bind_i32(ev(lane).lshr(l.trailing_zeros() as i32));
    // cos / sin: lane `lr`'s part of the row, shared by every head
    let cs_lane = b.bind_i32(ev(cs_row) + if style == RopeStyle::Neox { ev(lr) << 4 } else { ev(lr) << 3 });
    let cs_sin = b.bind_i32(ev(cs_lane) + d as i32);
    let (cos, sin): (Vec<ValueId>, Vec<ValueId>) = if style == RopeStyle::Neox {
        let c = b.buffer_load_b128(r_cs, cs_lane, 0i32, 0);
        let s = b.buffer_load_b128(r_cs, cs_sin, 0i32, 0);
        (bf16x8_to_f32(b, c).to_vec(), bf16x8_to_f32(b, s).to_vec())
    } else {
        let c = b.buffer_load_b64(r_cs, cs_lane, 0i32, 0);
        let s = b.buffer_load_b64(r_cs, cs_sin, 0i32, 0);
        (bf16x4_to_f32(b, c).to_vec(), bf16x4_to_f32(b, s).to_vec())
    };
    // all the passes' loads first
    struct Pass { head_ok: ValueId, off: ValueId, lo: ValueId, hi: Option<ValueId> }
    let mut ps: Vec<Pass> = Vec::new();
    for ps_i in 0..passes {
        let head = b.bind_i32(ev(wave) * rows_per_wave as i32 + ev(row_in_wave) + (ps_i * heads_per_pass) as i32);
        let head_ok = b.let_(Op::CmpLtI32(Opnd::Val(head), Opnd::I32(h as i32)));
        let row_off = b.bind_i32(ev(tok) * (2 * h * d) as i32 + ev(head) * (2 * d) as i32);
        let off = b.bind_i32(ev(row_off) + if style == RopeStyle::Neox { ev(lr) << 4 } else { ev(lr) << 4 });
        let lo = b.buffer_load_b128(r_x, off, 0i32, 0);
        let hi = if style == RopeStyle::Neox {
            let offh = b.bind_i32(ev(off) + d as i32);
            Some(b.buffer_load_b128(r_x, offh, 0i32, 0))
        } else {
            None
        };
        ps.push(Pass { head_ok, off, lo, hi });
    }
    for p in &ps {
        let xl = bf16x8_to_f32(b, p.lo);
        let (yl, yh);
        if style == RopeStyle::Neox {
            let xh = bf16x8_to_f32(b, p.hi.unwrap());
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
        b.buffer_store_b128(r_o, o_lo, 0i32, 0, f);
        if let Some(yh) = yh {
            let offh = b.bind_i32(ev(p.off) + d as i32);
            let o_hi = masked(b, offh);
            let f = pack8(b, &yh);
            b.buffer_store_b128(r_o, o_hi, 0i32, 0, f);
        }
    }
    fb.finish()
}