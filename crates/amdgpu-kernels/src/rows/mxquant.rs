//! MX quantization of bf16 activations into the layouts our MX GEMMs read (coverage: quantization category).
//!
//! * `quant_mxfp4_bf16_k<K>`: x bf16 [M, K] -> q fp4 (e2m1) [M, K / 2] (two values per byte, low nibble first) + 1 x 32 E8M0 scales in the
//!   SHUFFLED `gemm_a4w4_mxfp4` layout [M / 16][KT / 2][64 lanes][4 bytes] (KT = K / 256; lane 16 g + r: groups g, 4 + g of tile 2 p, then
//!   of tile 2 p + 1). Scale rule = aiter's `dynamic_mxfp4_quant`: amax rounded at mantissa bit 21 (+0x200000, truncate), e8m0 = exponent
//!   of that - 2 (fp4 max 6 = 1.5 2^2), clamp [0, 254]; q = fp4(x / 2^(e8m0 - 127)) (hardware RNE, saturating).
//!   One wave per (16-row block, tile pair): lane (r = l % 16, g = l / 16) quantizes groups {16 p + g + 4 j} of row 16 rb + r.
//!   M % 16 == 0, K % 512 == 0. Grid (ceil(K / 2048), M / 16) x 256 threads (wave w: tile pair min(4 x + w, KT / 2 - 1)).
//! * `quant_mxfp8_bf16_k<K>`: x bf16 [M, K] -> q fp8 e4m3 [M, K] + 1 x 128 E8M0 scales [K / 128][M] (the `gemm_mxfp8_bs_v4` A-scale
//!   layout, K-major). e8m0 = ceil(log2(amax / 448)) + 127 (no saturation), clamp [0, 254]. A 16-lane group per (row, 128-wide group): lane
//!   j reads 8 values (16 bytes; the 16 lanes read the group's 256 contiguous bytes), amax by a 16-lane DPP all-reduce; a wave = 4 rows, a
//!   CTA = 16 rows: grid (K / 128, M / 16) x 256 threads. M % 16 == 0, K % 128 == 0.
//!
//! ABI (both): x @0, q @8, scale @16, M @24 (i32).
use amdgpu_dsl::{CompileProfile, CvtOp, DType, Ext, FnBuilder, Function, MemTy, Op, Opnd, ValueId};
use amdgpu_tile::prims::{abs_f32, bf16x8_to_f32, group_reduce_dpp, RedOp};
use amdgpu_tile::{ev, Lane};

pub fn by_name(name: &str) -> Option<Function> {
    if let Some(r) = name.strip_prefix("quant_mxfp4_bf16_k") {
        let k: i64 = r.parse().ok()?;
        return (k % 512 == 0 && (512..=32768).contains(&k)).then(|| mxfp4(name, k));
    }
    if let Some(r) = name.strip_prefix("quant_mxfp8_bf16_k") {
        let k: i64 = r.parse().ok()?;
        return (k % 512 == 0 && (512..=32768).contains(&k)).then(|| mxfp8(name, k));
    }
    None
}

fn abi(b: &mut FnBuilder, rows_mult: i64, k: i64, q_bytes_per_row: i64, scale_bytes_per_row: i64) {
    for (n, o) in [("x", 0u32), ("q", 8), ("scale", 16)] {
        b.arg(n, DType::Ptr, o, 8);
    }
    b.arg("M", DType::I32, 24, 4);
    b.sym_range("MB", 1, 1 << 14);
    let m = Ext::sym("MB") * Ext::c(rows_mult);
    b.ka_is(24, m.clone());
    b.arg_tensor("x", m.clone() * Ext::c(2 * k));
    b.arg_tensor("q", m.clone() * Ext::c(q_bytes_per_row));
    b.arg_tensor("scale", m * Ext::c(scale_bytes_per_row));
}

/// max |x| over a list of f32 values
fn amax_of(b: &mut FnBuilder, xs: &[ValueId]) -> ValueId {
    let mut m = abs_f32(b, xs[0]);
    for x in &xs[1..] {
        let a = abs_f32(b, *x);
        m = b.max_f32(m, a);
    }
    m
}

/// One MXFP4 group of 32 f32 values: the E8M0 scale byte (`e`, an i32 in 0..=254) and the 16 packed fp4 bytes (a Frag4). Scale rule = aiter's `dynamic_mxfp4_quant`: amax rounded at mantissa bit 21
/// (+0x200000, truncate), e8m0 = exponent - 2 (fp4 max 6 = 1.5 2^2), clamp [0, 254]; q = fp4(x / 2^(e8m0 - 127)) (hardware RNE, saturating). Shared with the fused `rows/mxnorm.rs`.
pub(crate) fn mxfp4_group(b: &mut FnBuilder, xs: &[ValueId]) -> (ValueId, ValueId) {
    assert_eq!(xs.len(), 32);
    let amax = amax_of(b, xs);
    // aiter: (bits + 0x200000) & 0xff800000, exponent - 2
    let ab = b.bitcast_i32(amax);
    let rnd = b.bind_i32((ev(ab) + 0x20_0000) & (0xff80_0000u32 as i32));
    let e = b.bind_i32((ev(rnd).lshr(23) & 0xff) - 2);
    let e = b.bind_i32(ev(e).max(0).min(254));
    // the scale as an f32 with that exponent (e = 0: the smallest normal, 2^-126, for the division only)
    let sbits = b.bind_i32(ev(e).max(1) << 23);
    let sf = b.bitcast_f32(sbits);
    // 32 values -> 16 bytes: v_cvt_scalef32_pk_fp4_f32 packs two values per byte (sel = byte)
    let mut words: Vec<ValueId> = Vec::with_capacity(4);
    for w in 0..4usize {
        // v_cvt_scalef32_pk_fp4_f32 writes byte `sel` of the previous word (chained)
        let mut acc = b.c_i32(0);
        for byte in 0..4usize {
            let i = 8 * w + 2 * byte;
            acc = b.cvt(CvtOp::ScalePkFp4F32, &[Opnd::Val(xs[i]), Opnd::Val(xs[i + 1]), Opnd::Val(sf), Opnd::Val(acc)], byte as u8);
        }
        words.push(acc);
    }
    let f4 = b.frag4(words[0], words[1], words[2], words[3]);
    (e, f4)
}

fn mxfp4(name: &str, k: i64) -> Function {
    let mut b = FnBuilder::new(name, 256);
    b.compile_profile(CompileProfile::Gemm);
    let kt = k / 256;
    abi(&mut b, 16, k, k / 2, k / 32);
    b.grid_is(0, Ext::c((k / 512 + 3) / 4));
    b.grid_is(1, Ext::sym("MB"));
    let kp = |b: &mut FnBuilder, o: u32| b.let_(Op::KernargPtr { offset: o });
    let (p_x, p_q, p_s) = (kp(&mut b, 0), kp(&mut b, 8), kp(&mut b, 16));
    let m_v = b.let_(Op::KernargI32 { offset: 24 });
    let x0 = b.let_(Op::WorkgroupIdX);
    let rb = b.let_(Op::WorkgroupIdY);
    let lane = Lane::new(&mut b);
    let r = b.bind_i32(lane.lane() & 15);
    let g = b.bind_i32(lane.lane().lshr(4));
    // tile pair of this wave (waves past the last pair of K % 2048 != 0 redo the last pair: same values)
    let p = { let v = b.bind_i32(((ev(x0) << 2) + lane.wave()).min(kt as i32 / 2 - 1)); b.read_first_lane(v) };
    let row = b.bind_i32((ev(rb) << 4) + ev(r));
    let nx = { let v = b.bind_i32(ev(m_v) * (2 * k as i32)); b.read_first_lane(v) };
    let nq = { let v = b.bind_i32(ev(m_v) * (k as i32 / 2)); b.read_first_lane(v) };
    let ns = { let v = b.bind_i32(ev(m_v) * (k as i32 / 32)); b.read_first_lane(v) };
    let (r_x, r_q, r_s) = (b.buffer_desc_dyn(p_x, nx), b.buffer_desc_dyn(p_q, nq), b.buffer_desc_dyn(p_s, ns));
    let c0 = b.c_i32(0);
    let mut sbytes: Vec<ValueId> = Vec::with_capacity(4);
    for j in 0..4i32 {
        // group 16 p + g + 4 j of the row: 32 bf16 = 64 bytes at (row K + 32 group) 2
        let grp_i = b.bind_i32((ev(p) << 4) + ev(g) + 4 * j);
        let xo = b.bind_i32((ev(row) * (k as i32) + (ev(grp_i) << 5)) << 1);
        let mut xs: Vec<ValueId> = Vec::with_capacity(32);
        for c in 0..4u32 {
            let f = b.buffer_load_b128(r_x, xo, c0, 16 * c);
            xs.extend(bf16x8_to_f32(&mut b, f));
        }
        let (e, f4) = mxfp4_group(&mut b, &xs);
        sbytes.push(e);
        let qo = b.bind_i32(ev(row) * (k as i32 / 2) + (ev(grp_i) << 4));
        b.buffer_store_b128(r_q, qo, c0, 0, f4);
    }
    // the lane's shuffled scale dword: bytes {g, 4 + g} of tile 2 p, {8 + g, 12 + g} = j 0..3
    let w = b.bind_i32(ev(sbytes[0]) + (ev(sbytes[1]) << 8) + (ev(sbytes[2]) << 16) + (ev(sbytes[3]) << 24));
    let so = b.bind_i32((((ev(rb) * (kt as i32 / 2)) + ev(p)) * 64 + lane.lane()) << 2);
    let wf = b.bitcast_f32(w);
    b.buffer_store_f32(r_s, so, c0, 0, wf);
    b.finish()
}

fn mxfp8(name: &str, k: i64) -> Function {
    let mut b = FnBuilder::new(name, 256);
    b.compile_profile(CompileProfile::Gemm);
    abi(&mut b, 16, k, k, k / 128);
    b.grid_is(0, Ext::c(k / 128));
    b.grid_is(1, Ext::sym("MB"));
    let kp = |b: &mut FnBuilder, o: u32| b.let_(Op::KernargPtr { offset: o });
    let (p_x, p_q, p_s) = (kp(&mut b, 0), kp(&mut b, 8), kp(&mut b, 16));
    let m_v = b.let_(Op::KernargI32 { offset: 24 });
    let kb = b.let_(Op::WorkgroupIdX);
    let mb = b.let_(Op::WorkgroupIdY);
    let lane = Lane::new(&mut b);
    let j = b.bind_i32(lane.lane() & 15);
    let row = b.bind_i32((ev(mb) << 4) + (lane.wave() << 2) + lane.lane().lshr(4));
    let nx = { let v = b.bind_i32(ev(m_v) * (2 * k as i32)); b.read_first_lane(v) };
    let nq = { let v = b.bind_i32(ev(m_v) * (k as i32)); b.read_first_lane(v) };
    let ns = { let v = b.bind_i32(ev(m_v) * (k as i32 / 128)); b.read_first_lane(v) };
    let (r_x, r_q, r_s) = (b.buffer_desc_dyn(p_x, nx), b.buffer_desc_dyn(p_q, nq), b.buffer_desc_dyn(p_s, ns));
    let c0 = b.c_i32(0);
    let col = b.bind_i32((ev(kb) << 7) + (ev(j) << 3));
    let xo = b.bind_i32((ev(row) * (k as i32) + ev(col)) << 1);
    let f = b.buffer_load_b128(r_x, xo, c0, 0);
    let xs = bf16x8_to_f32(&mut b, f);
    let am = amax_of(&mut b, &xs);
    let amax = group_reduce_dpp(&mut b, RedOp::Max, am, 16);
    // e8m0 = ceil(log2(amax / 448)) + 127: the exponent of amax / 448, + 1 when its mantissa is not zero
    let c448 = b.c_f32(1.0 / 448.0);
    let t = b.mul_f32(amax, c448);
    let tb = b.bitcast_i32(t);
    let ex = b.bind_i32(ev(tb).lshr(23) & 0xff);
    let up = b.bind_i32((ev(tb) & 0x7f_ffff).ne(0).select_i32(1, 0));
    let e = b.bind_i32((ev(ex) + ev(up)).max(0).min(254));
    let sbits = b.bind_i32(ev(e).max(1) << 23);
    let sf = b.bitcast_f32(sbits);
    let mut words: Vec<ValueId> = Vec::with_capacity(2);
    for w in 0..2usize {
        let i = 4 * w;
        let lo = b.c_i32(0);
        let a = b.cvt(CvtOp::ScalePkFp8F32, &[Opnd::Val(xs[i]), Opnd::Val(xs[i + 1]), Opnd::Val(sf), Opnd::Val(lo)], 0);
        let a = b.cvt(CvtOp::ScalePkFp8F32, &[Opnd::Val(xs[i + 2]), Opnd::Val(xs[i + 3]), Opnd::Val(sf), Opnd::Val(a)], 1);
        words.push(a);
    }
    let f2 = b.frag2(words[0], words[1]);
    let qo = b.bind_i32(ev(row) * (k as i32) + ev(col));
    b.buffer_store_b64(r_q, qo, c0, 0, f2);
    // scale [K / 128][M]: byte (kb M + row), written by lane j = 0 of the group
    let big = b.c_i32(0x7fff_fff0);
    let so = b.bind_i32(ev(kb) * ev(m_v) + ev(row));
    let first = b.bind_cond(ev(j).eq(0));
    let so = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(first), a: Opnd::Val(so), b: Opnd::Val(big) });
    b.buffer_store_n(r_s, Some(Opnd::Val(so)), c0, 0, e, MemTy::U8, false);
    b.finish()
}