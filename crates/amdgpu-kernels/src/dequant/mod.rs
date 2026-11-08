//! fp8 (e4m3) -> bf16 dequantization (docs/op-coverage-memtemplates.md T1 / T3): `out = bf16(q * scale)`, the scale multiply fused in the hardware conversion (`v_cvt_scalef32_pk_bf16_fp8`).
//! A lane converts 8 elements (one b64 load of fp8 bytes, one b128 store of bf16), non-temporal streaming accesses.
//!
//! * `dequant_fp8_bf16_static`          one f32 scale for the tensor. ABI q @0 (fp8 [n]), out @8 (bf16 [n]), scale @16 (f32), n @20 (i32; a multiple of 8). 256 threads = 2048 elements per block.
//! * `dequant_fp8_bf16_token_n<N>`      one f32 scale per row: scale [M]. ABI q @0 (fp8 [M][N]), out @8, scale @16 (ptr), M @24 (i32).
//! * `dequant_fp8_bf16_group128_n<N>`   one f32 scale per 128 elements: scale [M][N / 128] (N a multiple of 128). Same ABI.
//!
//! Row kernels: N a multiple of 8, 8..=65536, grid x = column tiles of `threads * 8` elements, grid y = rows (M <= 65535), tensors < 4 GiB.

use amdgpu_dsl::{CachePolicy, CompileProfile, CvtOp, DType, Ext, FnBuilder, Function, MemTy, Op, Opnd, ValueId};
use amdgpu_tile::ev;

pub fn by_name(name: &str) -> Option<Function> {
    if name == "dequant_fp8_bf16_static" {
        return Some(dequant_static(name));
    }
    let (rest, group) = if let Some(r) = name.strip_prefix("dequant_fp8_bf16_group128_n") {
        (r, true)
    } else if let Some(r) = name.strip_prefix("dequant_fp8_bf16_token_n") {
        (r, false)
    } else {
        return None;
    };
    let n: i64 = rest.parse().ok()?;
    (n % 8 == 0 && (8..=65536).contains(&n) && (!group || n % 128 == 0)).then(|| dequant_rows(name, n, group))
}

/// 8 fp8 bytes (a Frag2 of two dwords) -> 4 dwords of bf16 pairs as one Frag4, times the f32 `scale`.
/// (`v_cvt_scalef32_pk_bf16_fp8` is NOT used: its scale operand is an E8M0-style power of two -- measured: a scale of 1.63 acts as 1.0 -- so a general f32 scale is a multiply after the plain fp8 -> f32 conversion.)
fn dequant8(b: &mut FnBuilder, q: ValueId, scale: ValueId) -> ValueId {
    let mut w = Vec::new();
    for d in 0..2u8 {
        let dw = b.frag2_lane(Opnd::Val(q), d);
        for sel in 0..2u8 {
            let f2 = b.cvt(CvtOp::PkF32Fp8, &[Opnd::Val(dw)], sel);
            let (a, c) = (b.frag2_lane(Opnd::Val(f2), 0), b.frag2_lane(Opnd::Val(f2), 1));
            let (a, c) = (b.bitcast_f32(a), b.bitcast_f32(c));
            let (a, c) = (b.mul_f32(a, scale), b.mul_f32(c, scale));
            w.push(b.cvt_pk_bf16_f32(a, c));
        }
    }
    b.frag4(w[0], w[1], w[2], w[3])
}
pub fn dequant_static(name: &str) -> Function {
    let mut fb = FnBuilder::new(name, 256);
    fb.compile_profile(CompileProfile::Default);
    let b = &mut fb;
    b.arg("q", DType::Ptr, 0, 8);
    b.arg("out", DType::Ptr, 8, 8);
    b.arg("scale", DType::F32, 16, 4);
    b.arg("n", DType::I32, 20, 4);
    let n = Ext::sym("N");
    b.sym_range("N", 8, 1 << 30);
    b.ka_is(20, n.clone());
    b.grid_is(0, n.clone().ceil_div(2048));
    b.arg_tensor("q", n.clone());
    b.arg_tensor("out", n * Ext::c(2));
    let (p_q, p_o) = (b.let_(Op::KernargPtr { offset: 0 }), b.let_(Op::KernargPtr { offset: 8 }));
    let scale = b.let_(Op::KernargF32 { offset: 16 });
    let n_v = b.let_(Op::KernargI32 { offset: 20 });
    let tid = b.let_(Op::WorkitemId);
    let bx = b.let_(Op::WorkgroupIdX);
    let r_q = { let v = b.read_first_lane(n_v); b.buffer_desc_dyn(p_q, v) };
    let r_o = { let v = b.bind_i32(ev(n_v) << 1); let v = b.read_first_lane(v); b.buffer_desc_dyn(p_o, v) };
    let qo = b.bind_i32((ev(bx) << 11) + (ev(tid) << 3));
    let q = b.buffer_load_c(r_q, Some(Opnd::Val(qo)), 0i32, 0, MemTy::B64, CachePolicy::NT);
    let o = dequant8(b, q, scale);
    let oo = b.bind_i32((ev(bx) << 12) + (ev(tid) << 4));
    b.buffer_store_c(r_o, Some(Opnd::Val(oo)), 0i32, 0, o, MemTy::B128, false, CachePolicy::NT);
    fb.finish()
}

pub fn dequant_rows(name: &str, n: i64, group: bool) -> Function {
    let d8 = n / 8;
    let threads = (((d8 + 63) / 64).min(4) * 64) as u32;
    let tiles = (d8 + threads as i64 - 1) / threads as i64;
    let ns = if group { n / 128 } else { 1 };
    let mut fb = FnBuilder::new(name, threads);
    fb.compile_profile(CompileProfile::Default);
    let b = &mut fb;
    b.arg("q", DType::Ptr, 0, 8);
    b.arg("out", DType::Ptr, 8, 8);
    b.arg("scale", DType::Ptr, 16, 8);
    b.arg("m", DType::I32, 24, 4);
    let mt = Ext::sym("MT");
    b.sym_range("MT", 1, 65535);
    b.ka_is(24, mt.clone());
    b.grid_is(0, Ext::c(tiles));
    b.grid_is(1, mt.clone());
    b.arg_tensor("q", mt.clone() * Ext::c(n));
    b.arg_tensor("out", mt.clone() * Ext::c(2 * n));
    b.arg_tensor("scale", mt * Ext::c(4 * ns));
    let (p_q, p_o, p_s) = (b.let_(Op::KernargPtr { offset: 0 }), b.let_(Op::KernargPtr { offset: 8 }), b.let_(Op::KernargPtr { offset: 16 }));
    let m_v = b.let_(Op::KernargI32 { offset: 24 });
    let tid = b.let_(Op::WorkitemId);
    let tile = b.let_(Op::WorkgroupIdX);
    let row = b.let_(Op::WorkgroupIdY);
    let r_q = { let v = b.bind_i32(ev(m_v) * n as i32); let v = b.read_first_lane(v); b.buffer_desc_dyn(p_q, v) };
    let r_o = { let v = b.bind_i32(ev(m_v) * (2 * n) as i32); let v = b.read_first_lane(v); b.buffer_desc_dyn(p_o, v) };
    let r_s = { let v = b.bind_i32(ev(m_v) * (4 * ns) as i32); let v = b.read_first_lane(v); b.buffer_desc_dyn(p_s, v) };
    let col8 = b.bind_i32(ev(tile) * threads as i32 + ev(tid));
    let valid = b.let_(Op::CmpLtI32(Opnd::Val(col8), Opnd::I32(d8 as i32)));
    let qo = b.bind_i32(ev(row) * n as i32 + (ev(col8) << 3));
    let q = b.buffer_load_c(r_q, Some(Opnd::Val(qo)), 0i32, 0, MemTy::B64, CachePolicy::NT);
    // the scale: one per row (uniform), or one per 16 lanes' worth of columns (128 elements)
    let so = if group { b.bind_i32((ev(row) * (n / 128) as i32 + ev(col8).lshr(4)) << 2) } else { b.bind_i32(ev(row) << 2) };
    let sf = b.buffer_load_f32(r_s, so, 0i32, 0);
    let o = dequant8(b, q, sf);
    let oo = b.bind_i32(ev(row) * (2 * n) as i32 + (ev(col8) << 4));
    let oo = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(valid), a: Opnd::Val(oo), b: Opnd::I32(-16) });
    b.buffer_store_c(r_o, Some(Opnd::Val(oo)), 0i32, 0, o, MemTy::B128, false, CachePolicy::NT);
    fb.finish()
}