//! Streaming elementwise kernels (docs/op-coverage-memtemplates.md T1): one b128 access per lane and input, one tile per block (no grid-stride loop), buffer descriptors for the tails
//! (out-of-range loads read zeros, out-of-range stores are dropped), occupancy first. Row-structured ones (`act_mul`: `out[m][d] = act(x[m][d]) * x[m][D + d]`) run a row per block row and
//! column tiles of `threads * 8` elements on grid y; the flat ones (`add_bf16`) run `n / 8` lanes over the whole array.
//!
//! Names: `silu_mul_bf16_d<D>`, `gelu_tanh_mul_bf16_d<D>` (D a multiple of 8, 8..=65536; M <= 65535 and the input / output tensors < 4 GiB), `add_bf16` (flat, n a multiple of 8).
//! ABI: `act_mul`: x @0 (`[M][2D]` bf16), out @8 (`[M][D]` bf16), M @16 (i32). `add_bf16`: a @0, b @8, out @16, n @24 (i32, elements).

use amdgpu_dsl::{CompileProfile, DType, Ext, FnBuilder, Function, Op, Opnd, TriOp, ValueId};
use amdgpu_tile::prims::{act, bf16x8_to_f32, Act};
use amdgpu_tile::ev;

pub fn by_name(name: &str) -> Option<Function> {
    match name {
        "add_bf16" => return Some(add_bf16(name)),
        "cvt_f32_bf16" => return Some(cvt_f32_bf16(name)),
        "cvt_bf16_f32" => return Some(cvt_bf16_f32(name)),
        "quant_bf16_fp8_static" => return Some(quant_bf16_fp8_static(name)),
        _ => {}
    }
    let (rest, a) = if let Some(r) = name.strip_prefix("silu_mul_bf16_d") {
        (r, Act::Silu)
    } else if let Some(r) = name.strip_prefix("gelu_tanh_mul_bf16_d") {
        (r, Act::GeluTanh)
    } else {
        return None;
    };
    let d: i64 = rest.parse().ok()?;
    (d % 8 == 0 && (8..=65536).contains(&d)).then(|| act_mul_bf16(name, d, a))
}

/// Eight f32 values -> 4 packed-bf16 dwords as one Frag4.
fn pack_bf16x8(b: &mut FnBuilder, r: &[ValueId; 8]) -> ValueId {
    let w: Vec<ValueId> = (0..4).map(|k| b.cvt_pk_bf16_f32(r[2 * k], r[2 * k + 1])).collect();
    b.frag4(w[0], w[1], w[2], w[3])
}

/// `out[m][d] = act(x[m][d]) * x[m][D + d]` for bf16 `x [M][2D]`, `out [M][D]`; grid x = the column tiles of `threads * 8` elements (fastest: consecutive blocks read consecutive memory), grid y = the rows (M <= 65535).
pub fn act_mul_bf16(name: &str, d: i64, a: Act) -> Function {
    let d8 = d / 8;
    let threads = (((d8 + 63) / 64).min(4) * 64) as u32;
    let tiles = (d8 + threads as i64 - 1) / threads as i64;
    let mut fb = FnBuilder::new(name, threads);
    fb.compile_profile(CompileProfile::Default);
    let b = &mut fb;
    b.arg("x", DType::Ptr, 0, 8);
    b.arg("out", DType::Ptr, 8, 8);
    b.arg("m", DType::I32, 16, 4);
    let mt = Ext::sym("MT");
    b.sym_range("MT", 1, 65535);
    b.ka_is(16, mt.clone());
    b.grid_is(0, Ext::c(tiles));
    b.grid_is(1, mt.clone());
    b.arg_tensor("x", mt.clone() * Ext::c(4 * d));
    b.arg_tensor("out", mt * Ext::c(2 * d));
    let (p_x, p_o) = (b.let_(Op::KernargPtr { offset: 0 }), b.let_(Op::KernargPtr { offset: 8 }));
    let m_v = b.let_(Op::KernargI32 { offset: 16 });
    let tid = b.let_(Op::WorkitemId);
    let tile = b.let_(Op::WorkgroupIdX);
    let row = b.let_(Op::WorkgroupIdY);
    // descriptors: num_records = the tensor bytes (loads past the end read zeros, stores past the end are dropped)
    let r_x = { let v = b.bind_i32(ev(m_v) * (4 * d) as i32); let v = b.read_first_lane(v); b.buffer_desc_dyn(p_x, v) };
    let r_o = { let v = b.bind_i32(ev(m_v) * (2 * d) as i32); let v = b.read_first_lane(v); b.buffer_desc_dyn(p_o, v) };
    let col8 = b.bind_i32(ev(tile) * threads as i32 + ev(tid));
    let valid = b.let_(Op::CmpLtI32(Opnd::Val(col8), Opnd::I32(d8 as i32)));
    let off1 = b.bind_i32(ev(row) * (4 * d) as i32 + (ev(col8) << 4));
    let off2 = b.bind_i32(ev(off1) + (2 * d) as i32);
    let offo = b.bind_i32(ev(row) * (2 * d) as i32 + (ev(col8) << 4));
    // lanes past the row end store to an offset outside every descriptor (dropped by the hardware)
    let offo = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(valid), a: Opnd::Val(offo), b: Opnd::I32(-16) });
    let x1 = b.buffer_load_b128(r_x, off1, 0i32, 0);
    let x2 = b.buffer_load_b128(r_x, off2, 0i32, 0);
    let (f1, f2) = (bf16x8_to_f32(b, x1), bf16x8_to_f32(b, x2));
    let mut r = [0; 8];
    for i in 0..8 {
        let y = act(b, f1[i], a);
        r[i] = b.mul_f32(y, f2[i]);
    }
    let o = pack_bf16x8(b, &r);
    b.buffer_store_b128(r_o, offo, 0i32, 0, o);
    fb.finish()
}

/// `out[i] = a[i] + b[i]` (bf16, rounded to nearest even from the f32 sum) over `n` elements (n a multiple of 8); 256 threads, one b128 per lane and input.
pub fn add_bf16(name: &str) -> Function {
    let mut fb = FnBuilder::new(name, 256);
    fb.compile_profile(CompileProfile::Default);
    let b = &mut fb;
    b.arg("a", DType::Ptr, 0, 8);
    b.arg("b", DType::Ptr, 8, 8);
    b.arg("out", DType::Ptr, 16, 8);
    b.arg("n", DType::I32, 24, 4);
    let n = Ext::sym("N");
    b.sym_range("N", 8, 1 << 30);
    b.ka_is(24, n.clone());
    b.grid_is(0, n.clone().ceil_div(2048));
    b.arg_tensor("a", n.clone() * Ext::c(2));
    b.arg_tensor("b", n.clone() * Ext::c(2));
    b.arg_tensor("out", n * Ext::c(2));
    let (p_a, p_b, p_o) = (b.let_(Op::KernargPtr { offset: 0 }), b.let_(Op::KernargPtr { offset: 8 }), b.let_(Op::KernargPtr { offset: 16 }));
    let n_v = b.let_(Op::KernargI32 { offset: 24 });
    let tid = b.let_(Op::WorkitemId);
    let bx = b.let_(Op::WorkgroupIdX);
    let nb = b.bind_i32(ev(n_v) << 1);
    let nb = b.read_first_lane(nb);
    let (r_a, r_b, r_o) = (b.buffer_desc_dyn(p_a, nb), b.buffer_desc_dyn(p_b, nb), b.buffer_desc_dyn(p_o, nb));
    let off = b.bind_i32((ev(bx) << 12) + (ev(tid) << 4));
    let xa = b.buffer_load_b128(r_a, off, 0i32, 0);
    let xb = b.buffer_load_b128(r_b, off, 0i32, 0);
    let (fa, fbv) = (bf16x8_to_f32(b, xa), bf16x8_to_f32(b, xb));
    let mut r = [0; 8];
    for i in 0..8 {
        r[i] = b.add_f32(fa[i], fbv[i]);
    }
    let o = pack_bf16x8(b, &r);
    b.buffer_store_b128(r_o, off, 0i32, 0, o);
    fb.finish()
}

/// `out[i] = bf16(x[i])` for f32 `x` (n a multiple of 4): a lane converts two 4-element vectors (b128 loads, b64 stores, both perfectly coalesced), 256 threads = 2048 elements per block. ABI x @0, out @8, n @16 (i32).
pub fn cvt_f32_bf16(name: &str) -> Function {
    let mut fb = FnBuilder::new(name, 256);
    fb.compile_profile(CompileProfile::Default);
    let b = &mut fb;
    b.arg("x", DType::Ptr, 0, 8);
    b.arg("out", DType::Ptr, 8, 8);
    b.arg("n", DType::I32, 16, 4);
    let n = Ext::sym("N");
    b.sym_range("N", 4, 1 << 29);
    b.ka_is(16, n.clone());
    b.grid_is(0, n.clone().ceil_div(2048));
    b.arg_tensor("x", n.clone() * Ext::c(4));
    b.arg_tensor("out", n * Ext::c(2));
    let (p_x, p_o) = (b.let_(Op::KernargPtr { offset: 0 }), b.let_(Op::KernargPtr { offset: 8 }));
    let n_v = b.let_(Op::KernargI32 { offset: 16 });
    let tid = b.let_(Op::WorkitemId);
    let bx = b.let_(Op::WorkgroupIdX);
    let r_x = { let v = b.bind_i32(ev(n_v) << 2); let v = b.read_first_lane(v); b.buffer_desc_dyn(p_x, v) };
    let r_o = { let v = b.bind_i32(ev(n_v) << 1); let v = b.read_first_lane(v); b.buffer_desc_dyn(p_o, v) };
    let v0 = b.bind_i32((ev(bx) << 9) + ev(tid));          // vector index (4 elements)
    let o0 = b.bind_i32(ev(v0) << 4);
    let o1 = b.bind_i32(ev(o0) + 4096);
    let (x0, x1) = (b.buffer_load_b128(r_x, o0, 0i32, 0), b.buffer_load_b128(r_x, o1, 0i32, 0));
    for (k, x) in [(0i32, x0), (1, x1)] {
        let w: Vec<ValueId> = (0..2u8).map(|j| {
            let a = b.frag4_lane(Opnd::Val(x), 2 * j);
            let c = b.frag4_lane(Opnd::Val(x), 2 * j + 1);
            b.cvt_pk_bf16_f32(a, c)
        }).collect();
        let f = b.frag2(w[0], w[1]);
        let off = b.bind_i32((ev(v0) << 3) + 2048 * k);
        b.buffer_store_b64(r_o, off, 0i32, 0, f);
    }
    fb.finish()
}

/// `out[i] = f32(x[i])` for bf16 `x` (n a multiple of 4). ABI x @0, out @8, n @16 (i32).
pub fn cvt_bf16_f32(name: &str) -> Function {
    let mut fb = FnBuilder::new(name, 256);
    fb.compile_profile(CompileProfile::Default);
    let b = &mut fb;
    b.arg("x", DType::Ptr, 0, 8);
    b.arg("out", DType::Ptr, 8, 8);
    b.arg("n", DType::I32, 16, 4);
    let n = Ext::sym("N");
    b.sym_range("N", 4, 1 << 29);
    b.ka_is(16, n.clone());
    b.grid_is(0, n.clone().ceil_div(2048));
    b.arg_tensor("x", n.clone() * Ext::c(2));
    b.arg_tensor("out", n * Ext::c(4));
    let (p_x, p_o) = (b.let_(Op::KernargPtr { offset: 0 }), b.let_(Op::KernargPtr { offset: 8 }));
    let n_v = b.let_(Op::KernargI32 { offset: 16 });
    let tid = b.let_(Op::WorkitemId);
    let bx = b.let_(Op::WorkgroupIdX);
    let r_x = { let v = b.bind_i32(ev(n_v) << 1); let v = b.read_first_lane(v); b.buffer_desc_dyn(p_x, v) };
    let r_o = { let v = b.bind_i32(ev(n_v) << 2); let v = b.read_first_lane(v); b.buffer_desc_dyn(p_o, v) };
    let v0 = b.bind_i32((ev(bx) << 9) + ev(tid));
    let o0 = b.bind_i32(ev(v0) << 3);
    let o1 = b.bind_i32(ev(o0) + 2048);
    let (x0, x1) = (b.buffer_load_b64(r_x, o0, 0i32, 0), b.buffer_load_b64(r_x, o1, 0i32, 0));
    for (k, x) in [(0i32, x0), (1, x1)] {
        let mut f = [0; 4];
        for d in 0..2u8 {
            let l = b.frag2_lane(Opnd::Val(x), d);
            let w = b.bitcast_i32(l);
            let lo = b.bind_i32(ev(w) << 16);
            let hi = b.bind_i32(ev(w) & (0xffff_0000u32 as i32));
            f[2 * d as usize] = b.bitcast_f32(lo);
            f[2 * d as usize + 1] = b.bitcast_f32(hi);
        }
        let out = b.frag4(f[0], f[1], f[2], f[3]);
        let off = b.bind_i32((ev(v0) << 4) + 4096 * k);
        b.buffer_store_b128(r_o, off, 0i32, 0, out);
    }
    fb.finish()
}

/// Static per-tensor fp8 (e4m3) quantization: `out[i] = fp8(clamp(x[i] / scale, -448, 448))` for bf16 `x` (n a multiple of 8); a lane converts 8 elements (b128 load, b64 store). ABI x @0, out @8, scale @16 (f32), n @20 (i32).
pub fn quant_bf16_fp8_static(name: &str) -> Function {
    let mut fb = FnBuilder::new(name, 256);
    fb.compile_profile(CompileProfile::Default);
    let b = &mut fb;
    b.arg("x", DType::Ptr, 0, 8);
    b.arg("out", DType::Ptr, 8, 8);
    b.arg("scale", DType::F32, 16, 4);
    b.arg("n", DType::I32, 20, 4);
    let n = Ext::sym("N");
    b.sym_range("N", 8, 1 << 30);
    b.ka_is(20, n.clone());
    b.grid_is(0, n.clone().ceil_div(2048));
    b.arg_tensor("x", n.clone() * Ext::c(2));
    b.arg_tensor("out", n);
    let (p_x, p_o) = (b.let_(Op::KernargPtr { offset: 0 }), b.let_(Op::KernargPtr { offset: 8 }));
    let scale = b.let_(Op::KernargF32 { offset: 16 });
    let n_v = b.let_(Op::KernargI32 { offset: 20 });
    let tid = b.let_(Op::WorkitemId);
    let bx = b.let_(Op::WorkgroupIdX);
    let r_x = { let v = b.bind_i32(ev(n_v) << 1); let v = b.read_first_lane(v); b.buffer_desc_dyn(p_x, v) };
    let r_o = { let v = b.read_first_lane(n_v); b.buffer_desc_dyn(p_o, v) };
    let inv = b.rcp_f32(scale);
    let lo = b.c_f32(-448.0);
    let hi = b.c_f32(448.0);
    let off = b.bind_i32((ev(bx) << 12) + (ev(tid) << 4));
    let x = b.buffer_load_b128(r_x, off, 0i32, 0);
    let f = bf16x8_to_f32(b, x);
    let mut q = [0; 8];
    for k in 0..8 {
        let y = b.mul_f32(f[k], inv);
        q[k] = b.tri(TriOp::Med3F32, y, lo, hi);
    }
    let words = [b.cvt_pk_fp8x4_f32(q[0], q[1], q[2], q[3]), b.cvt_pk_fp8x4_f32(q[4], q[5], q[6], q[7])];
    let f2 = b.frag2(words[0], words[1]);
    let oo = b.bind_i32((ev(bx) << 11) + (ev(tid) << 3));
    b.buffer_store_b64(r_o, oo, 0i32, 0, f2);
    fb.finish()
}
