//! MXFP6 quantisation of bf16 activations (OCP MX, 32-element groups with an E8M0 scale), generic row-major layout:
//! * `quant_mxfp6_e2m3_bf16_k<K>`: fp6 e2m3 (max 7.5): shared exponent = floor(log2 amax) - 2.
//! * `quant_mxfp6_e3m2_bf16_k<K>`: fp6 e3m2 (max 28):  shared exponent = floor(log2 amax) - 4.
//!
//! x bf16 [M][K] -> q [M][K / 32][24 bytes] (each group of 32 elements = 32 six-bit codes, code j at bits [6 j, 6 j + 6) of its 192 bits) + scales e8m0 u8 [M][K / 32]; `q = fp6(x / 2^(e8m0 - 127))`
//! (the conversion saturates and rounds to nearest even). A lane quantises one group (4 b128 loads, 3 b64 stores); `v_cvt_scalef32_2xpk16_fp6_f32` takes the even elements of the group in one 16-dword source and the odd
//! ones in the other (it interleaves them). This is NOT the tiled / Hadamard-rotated layout of aiter's `quant_mxfp6_gemm` (that one is a GEMM-specific preprocessing). K % 256 == 0.
//! ABI: x @0, q @8, scale @16, M @24 (i32). Grid (ceil(K / 32 / 256), M), 256 threads.

use amdgpu_dsl::{CompileProfile, CvtOp, DType, Ext, FnBuilder, Function, MemTy, Op, Opnd, ValueId, CachePolicy};
use amdgpu_tile::prims::{abs_f32, bf16x8_to_f32};
use amdgpu_tile::ev;

pub fn by_name(name: &str) -> Option<Function> {
    let (rest, bf6) = if let Some(r) = name.strip_prefix("quant_mxfp6_e2m3_bf16_k") { (r, false) } else if let Some(r) = name.strip_prefix("quant_mxfp6_e3m2_bf16_k") { (r, true) } else { return None };
    let k: i64 = rest.parse().ok()?;
    (k % 256 == 0 && (256..=32768).contains(&k)).then(|| mxfp6(name, k, bf6))
}

fn mxfp6(name: &str, k: i64, bf6: bool) -> Function {
    let groups = k / 32;
    let mut b = FnBuilder::new(name, 256);
    b.compile_profile(CompileProfile::Default);
    b.arg("x", DType::Ptr, 0, 8);
    b.arg("q", DType::Ptr, 8, 8);
    b.arg("scale", DType::Ptr, 16, 8);
    b.arg("M", DType::I32, 24, 4);
    let mt = Ext::sym("M");
    b.sym_range("M", 1, 1 << 24);
    b.ka_is(24, mt.clone());
    b.grid_is(0, Ext::c((groups + 255) / 256));
    b.grid_is(1, mt.clone());
    b.arg_tensor("x", mt.clone() * Ext::c(2 * k));
    b.arg_tensor("q", mt.clone() * Ext::c(groups * 24));
    b.arg_tensor("scale", mt * Ext::c(groups));
    let kp = |b: &mut FnBuilder, o: u32| b.let_(Op::KernargPtr { offset: o });
    let (p_x, p_q, p_s) = (kp(&mut b, 0), kp(&mut b, 8), kp(&mut b, 16));
    let m_v = b.let_(Op::KernargI32 { offset: 24 });
    let tid = b.let_(Op::WorkitemId);
    let bx = b.let_(Op::WorkgroupIdX);
    let row = b.let_(Op::WorkgroupIdY);
    let g = b.bind_i32((ev(bx) << 8) + ev(tid));
    let g_ok = b.let_(Op::CmpLtI32(Opnd::Val(g), Opnd::I32(groups as i32)));
    let nx = { let v = b.bind_i32(ev(m_v) * (2 * k as i32)); b.read_first_lane(v) };
    let nq = { let v = b.bind_i32(ev(m_v) * (groups * 24) as i32); b.read_first_lane(v) };
    let ns = { let v = b.bind_i32(ev(m_v) * groups as i32); b.read_first_lane(v) };
    let (r_x, r_q, r_s) = (b.buffer_desc_dyn(p_x, nx), b.buffer_desc_dyn(p_q, nq), b.buffer_desc_dyn(p_s, ns));
    let xo = b.bind_i32(ev(row) * (2 * k as i32) + (ev(g) << 6));
    let xo = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(g_ok), a: Opnd::Val(xo), b: Opnd::I32(0x7fff_ff00) });
    let mut xs: Vec<ValueId> = Vec::with_capacity(32);
    for c in 0..4u32 {
        let f = b.buffer_load_c(r_x, Some(Opnd::Val(xo)), 0i32, 16 * c, MemTy::B128, CachePolicy::NT);
        xs.extend_from_slice(&bf16x8_to_f32(&mut b, f));
    }
    // the shared exponent: floor(log2 amax) - emax (the f32 exponent field), clamped to [0, 254]
    let mut amax = abs_f32(&mut b, xs[0]);
    for x in &xs[1..] {
        let a = abs_f32(&mut b, *x);
        amax = b.max_f32(amax, a);
    }
    let emax = if bf6 { 4 } else { 2 };
    let ab = b.bitcast_i32(amax);
    let e = b.bind_i32((ev(ab).lshr(23) & 0xff) - emax);
    let e = b.bind_i32(ev(e).max(0).min(254));
    let sbits = b.bind_i32(ev(e).max(1) << 23);
    let sf = b.bitcast_f32(sbits);
    // even elements in the first source, odd ones in the second (the instruction interleaves them)
    let mut ev_parts: Vec<Opnd> = Vec::new();
    let mut od_parts: Vec<Opnd> = Vec::new();
    for j in 0..16usize {
        let (a_, b_) = (b.bitcast_i32(xs[2 * j]), b.bitcast_i32(xs[2 * j + 1]));
        ev_parts.push(Opnd::Val(a_));
        od_parts.push(Opnd::Val(b_));
    }
    let fa = b.frag_cat(&ev_parts);
    let fb = b.frag_cat(&od_parts);
    let op = if bf6 { CvtOp::Scale2xPk16Bf6F32 } else { CvtOp::Scale2xPk16Fp6F32 };
    let codes = b.cvt(op, &[Opnd::Val(fa), Opnd::Val(fb), Opnd::Val(sf)], 0);
    // 192 bits = dwords 0 .. 6: three b64 stores at 24 g
    let base = b.bind_i32(ev(row) * (groups * 24) as i32 + ev(g) * 24);
    let base = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(g_ok), a: Opnd::Val(base), b: Opnd::I32(0x7fff_ff00) });
    for p in 0..3u8 {
        let part = b.frag_slice(codes, 2 * p, 2);
        b.buffer_store_c(r_q, Some(Opnd::Val(base)), 0i32, 8 * p as u32, part, MemTy::B64, false, CachePolicy::NT);
    }
    let so = b.bind_i32(ev(row) * groups as i32 + ev(g));
    let so = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(g_ok), a: Opnd::Val(so), b: Opnd::I32(0x7fff_ff00) });
    b.buffer_store_c(r_s, Some(Opnd::Val(so)), 0i32, 0, e, MemTy::U8, false, CachePolicy::NT);
    b.finish()
}