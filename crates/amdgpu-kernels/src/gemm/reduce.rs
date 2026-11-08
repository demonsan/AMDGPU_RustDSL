//! Split-K reduction of the GEMM's fp32 partials (`GemmConfig::split_k`): C[m][n] (bf16) = (sum_s W[s][m][n]) (sa[m] sb[n] with row / column
//! scales). One block per row and 1024 columns (256 threads x 4 consecutive columns: one 16-byte load per split, one 8-byte bf16 store).
//!
//! ABI: 0 ptr_ws (fp32 [S][MP][N], MP = the slice rows: M rounded up to the GEMM CTA), 8 ptr_c (bf16 [M][N]), 16 ptr_sa (fp32 [M]),
//! 24 ptr_sb (fp32 [N]), 32 M, 36 N, 40 S, 44 MP; grid (N / 1024, M).
use amdgpu_dsl::{CompileProfile, DType, Ext, FnBuilder, Function, Op, Opnd, ValueId};
use amdgpu_tile::ev;

/// `splits` partial slices (compile time), with (`rowcol`) or without the per-token x per-channel scales.
pub fn gemm_splitk_reduce(name: &str, splits: i64, rowcol: bool) -> Function {
    let mut b = FnBuilder::new(name, 256);
    b.compile_profile(CompileProfile::Gemm);
    for (n, o) in [("ptr_ws", 0u32), ("ptr_c", 8), ("ptr_sa", 16), ("ptr_sb", 24)] {
        b.arg(n, DType::Ptr, o, 8);
    }
    for (n, o) in [("m", 32u32), ("n", 36), ("s", 40), ("mp", 44)] {
        b.arg(n, DType::I32, o, 4);
    }
    let (mm, nb) = (Ext::sym("M"), Ext::sym("NB"));
    b.sym_range("M", 1, 1 << 14);
    b.sym_range("NB", 1, 16);
    let n_ext = nb.clone() * Ext::c(1024);
    b.ka_is(32, mm.clone());
    b.ka_is(36, n_ext.clone());
    b.ka_is(40, Ext::c(splits));
    b.ka_is(44, mm.clone());
    b.grid_is(0, nb);
    b.grid_is(1, mm.clone());
    b.arg_tensor("ptr_ws", Ext::c(4 * splits) * mm.clone() * n_ext.clone());
    b.arg_tensor("ptr_c", Ext::c(2) * mm.clone() * n_ext.clone());
    if rowcol {
        b.arg_tensor("ptr_sa", Ext::c(4) * mm.clone());
        b.arg_tensor("ptr_sb", Ext::c(4) * n_ext);
    }
    let (p_ws, p_c, p_sa, p_sb) = (b.let_(Op::KernargPtr { offset: 0 }), b.let_(Op::KernargPtr { offset: 8 }), b.let_(Op::KernargPtr { offset: 16 }), b.let_(Op::KernargPtr { offset: 24 }));
    let m_v = b.let_(Op::KernargI32 { offset: 32 });
    let n_v = b.let_(Op::KernargI32 { offset: 36 });
    let tid = b.let_(Op::WorkitemId);
    let bx = b.let_(Op::WorkgroupIdX);
    let row = b.let_(Op::WorkgroupIdY);
    let mp_v = b.let_(Op::KernargI32 { offset: 44 });
    let mn = { let v = b.bind_i32(ev(m_v) * ev(n_v)); b.read_first_lane(v) };
    let mpn = { let v = b.bind_i32(ev(mp_v) * ev(n_v)); b.read_first_lane(v) };
    let r_ws = { let v = b.bind_i32(ev(mpn) * (4 * splits as i32)); let v = b.read_first_lane(v); b.buffer_desc_dyn(p_ws, v) };
    let r_c = { let v = b.bind_i32(ev(mn) << 1); let v = b.read_first_lane(v); b.buffer_desc_dyn(p_c, v) };
    // the lane's 4 columns: n = 1024 bx + 4 tid of row `row`
    let col = b.bind_i32((ev(bx) << 10) + (ev(tid) << 2));
    let e0 = b.bind_i32(ev(row) * ev(n_v) + ev(col));
    let v_ws = b.bind_i32(ev(e0) << 2);
    let mn4 = { let v = b.bind_i32(ev(mpn) << 2); b.read_first_lane(v) };
    let parts: Vec<ValueId> = (0..splits)
        .map(|s| {
            let so = if s == 0 { Opnd::I32(0) } else { Opnd::Val({ let v = b.bind_i32(ev(mn4) * (s as i32)); b.read_first_lane(v) }) };
            b.buffer_load_b128(r_ws, v_ws, so, 0)
        })
        .collect();
    let scl: Option<(ValueId, ValueId)> = rowcol.then(|| {
        let r_sa = { let v = b.bind_i32(ev(m_v) << 2); let v = b.read_first_lane(v); b.buffer_desc_dyn(p_sa, v) };
        let r_sb = { let v = b.bind_i32(ev(n_v) << 2); let v = b.read_first_lane(v); b.buffer_desc_dyn(p_sb, v) };
        let ra = b.bind_i32(ev(row) << 2);
        let sa = b.buffer_load_f32(r_sa, ra, 0i32, 0);
        let rb = b.bind_i32(ev(col) << 2);
        (sa, b.buffer_load_b128(r_sb, rb, 0i32, 0))
    });
    let mut y = Vec::with_capacity(4);
    for r in 0..4u8 {
        let mut acc: Option<ValueId> = None;
        for p in &parts {
            let l = b.frag4_lane(Opnd::Val(*p), r);
            let f = b.bitcast_f32(l);
            acc = Some(match acc { None => f, Some(a) => b.add_f32(a, f) });
        }
        let mut v = acc.expect("splits >= 1");
        if let Some((sa, sb)) = scl {
            let cl = b.frag4_lane(Opnd::Val(sb), r);
            let c = b.bitcast_f32(cl);
            let s = b.mul_f32(sa, c);
            v = b.mul_f32(v, s);
        }
        y.push(v);
    }
    let w0 = b.cvt_pk_bf16_f32(y[0], y[1]);
    let w1 = b.cvt_pk_bf16_f32(y[2], y[3]);
    let f2 = b.frag2(w0, w1);
    let vc = b.bind_i32(ev(e0) << 1);
    b.buffer_store_b64(r_c, vc, 0i32, 0, f2);
    b.finish()
}

/// MoE stage-2 sum over the routes: out[t][n] (bf16) = sum_k W[t][k][n] (fp32 [M, topk, N]). Grid (N / 1024, M), 256 threads x 4 columns.
/// ABI: 0 ptr_ws, 8 ptr_out, 16 M, 20 N.
pub fn moe_sum_bf16(name: &str, topk: i64) -> Function {
    let mut b = FnBuilder::new(name, 256);
    b.compile_profile(CompileProfile::Gemm);
    b.arg("ptr_ws", DType::Ptr, 0, 8);
    b.arg("ptr_out", DType::Ptr, 8, 8);
    b.arg("m", DType::I32, 16, 4);
    b.arg("n", DType::I32, 20, 4);
    let (mm, nb) = (Ext::sym("M"), Ext::sym("NB"));
    b.sym_range("M", 1, 1 << 20);
    b.sym_range("NB", 1, 16);
    let n_ext = nb.clone() * Ext::c(1024);
    b.ka_is(16, mm.clone());
    b.ka_is(20, n_ext.clone());
    b.grid_is(0, nb);
    b.grid_is(1, mm.clone());
    b.arg_tensor("ptr_ws", Ext::c(4 * topk) * mm.clone() * n_ext.clone());
    b.arg_tensor("ptr_out", Ext::c(2) * mm * n_ext);
    let (p_ws, p_o) = (b.let_(Op::KernargPtr { offset: 0 }), b.let_(Op::KernargPtr { offset: 8 }));
    let m_v = b.let_(Op::KernargI32 { offset: 16 });
    let n_v = b.let_(Op::KernargI32 { offset: 20 });
    let tid = b.let_(Op::WorkitemId);
    let bx = b.let_(Op::WorkgroupIdX);
    let row = b.let_(Op::WorkgroupIdY);
    let mn = { let v = b.bind_i32(ev(m_v) * ev(n_v)); b.read_first_lane(v) };
    let r_ws = { let v = b.bind_i32(ev(mn) * (4 * topk as i32)); let v = b.read_first_lane(v); b.buffer_desc_dyn(p_ws, v) };
    let r_o = { let v = b.bind_i32(ev(mn) << 1); let v = b.read_first_lane(v); b.buffer_desc_dyn(p_o, v) };
    let col = b.bind_i32((ev(bx) << 10) + (ev(tid) << 2));
    let base = b.bind_i32((ev(row) * (topk as i32) * ev(n_v) + ev(col)) << 2);
    let n4 = { let v = b.bind_i32(ev(n_v) << 2); b.read_first_lane(v) };
    let parts: Vec<ValueId> = (0..topk)
        .map(|s| {
            let so = if s == 0 { Opnd::I32(0) } else { Opnd::Val({ let v = b.bind_i32(ev(n4) * (s as i32)); b.read_first_lane(v) }) };
            b.buffer_load_b128(r_ws, base, so, 0)
        })
        .collect();
    let y: Vec<ValueId> = (0..4u8)
        .map(|r| {
            let mut acc: Option<ValueId> = None;
            for p in &parts {
                let l = b.frag4_lane(Opnd::Val(*p), r);
                let f = b.bitcast_f32(l);
                acc = Some(match acc { None => f, Some(a) => b.add_f32(a, f) });
            }
            acc.expect("topk >= 1")
        })
        .collect();
    let w0 = b.cvt_pk_bf16_f32(y[0], y[1]);
    let w1 = b.cvt_pk_bf16_f32(y[2], y[3]);
    let f2 = b.frag2(w0, w1);
    let vo = b.bind_i32((ev(row) * ev(n_v) + ev(col)) << 1);
    b.buffer_store_b64(r_o, vo, 0i32, 0, f2);
    b.finish()
}