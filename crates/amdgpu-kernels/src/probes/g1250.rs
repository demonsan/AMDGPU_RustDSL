//! gfx1250 G1 probes (docs/gfx1250-plan.md): small wave32 kernels for the gfx1250 emitter / encoder / LLVM tail. They compile for gfx950 too.

use amdgpu_dsl::{DType, Ext, FnBuilder, Function, MemTy, Op, Opnd};

/// `out[i] = a[i] + b[i]` (f32), one element per work-item, 256 work-items per workgroup: global loads / stores, the workgroup id, the
/// workitem id, 64-bit address arithmetic.
pub fn probe_g1250_add() -> Function {
    let mut b = FnBuilder::new("probe_g1250_add", 256);
    b.arg("a", DType::Ptr, 0, 8);
    b.arg("b", DType::Ptr, 8, 8);
    b.arg("out", DType::Ptr, 16, 8);
    b.grid_is(0, Ext::sym("N"));
    for n in ["a", "b", "out"] {
        b.arg_tensor(n, Ext::sym("N") * Ext::c(256 * 4));
    }
    let pa = b.let_(Op::KernargPtr { offset: 0 });
    let pb = b.let_(Op::KernargPtr { offset: 8 });
    let po = b.let_(Op::KernargPtr { offset: 16 });
    let tid = b.let_(Op::WorkitemId);
    let wg = b.let_(Op::WorkgroupIdX);
    let w8 = b.let_(Op::ShlI32(Opnd::Val(wg), Opnd::I32(8)));
    let i = b.let_(Op::AddI32(Opnd::Val(w8), Opnd::Val(tid)));
    let off = b.let_(Op::ShlI32(Opnd::Val(i), Opnd::I32(2)));
    let qa = b.let_(Op::PtrAdd { base: Opnd::Val(pa), byte_off: Opnd::Val(off) });
    let qb = b.let_(Op::PtrAdd { base: Opnd::Val(pb), byte_off: Opnd::Val(off) });
    let qo = b.let_(Op::PtrAdd { base: Opnd::Val(po), byte_off: Opnd::Val(off) });
    let x = b.load_global(qa, 0, MemTy::B32);
    let y = b.load_global(qb, 0, MemTy::B32);
    let xf = b.let_(Op::BitcastF32(Opnd::Val(x)));
    let yf = b.let_(Op::BitcastF32(Opnd::Val(y)));
    let s = b.let_(Op::AddF32(Opnd::Val(xf), Opnd::Val(yf)));
    b.store_global(qo, 0, s, MemTy::B32, false);
    b.finish()
}

/// Transcendentals and scalar arithmetic: `out[tid] = exp2(x) * rsq(y) + sqrt(z)` from three kernarg floats, 64 work-items.
pub fn probe_g1250_math() -> Function {
    let mut b = FnBuilder::new("probe_g1250_math", 64);
    b.arg("out", DType::Ptr, 0, 8);
    b.arg("x", DType::F32, 8, 4);
    b.arg("y", DType::F32, 12, 4);
    b.arg("z", DType::F32, 16, 4);
    b.arg_tensor("out", Ext::c(64 * 4));
    let po = b.let_(Op::KernargPtr { offset: 0 });
    let x = b.let_(Op::KernargF32 { offset: 8 });
    let y = b.let_(Op::KernargF32 { offset: 12 });
    let z = b.let_(Op::KernargF32 { offset: 16 });
    let tid = b.let_(Op::WorkitemId);
    let e = b.exp2_f32(x);
    let r = b.un(amdgpu_dsl::UnOp::RsqF32, y);
    let q = b.un(amdgpu_dsl::UnOp::SqrtF32, z);
    let t = b.mul_f32(e, r);
    let v = b.add_f32(t, q);
    let tf = b.un(amdgpu_dsl::UnOp::CvtF32I32, tid);
    let w = b.add_f32(v, tf);
    let off = b.let_(Op::ShlI32(Opnd::Val(tid), Opnd::I32(2)));
    let qo = b.let_(Op::PtrAdd { base: Opnd::Val(po), byte_off: Opnd::Val(off) });
    b.store_global(qo, 0, w, MemTy::B32, false);
    b.finish()
}

/// TDM: the 12 descriptor words (groups 0 and 1) come from the kernarg segment (offset 16..64, built by the host), one
/// `tensor_load_to_lds`, its completion wait, the workgroup barrier, then every work-item copies one LDS dword to `out`. `raw`: the same
/// load as three verbatim instruction words (exact-replay form).
pub fn probe_g1250_tdm_with(raw: Option<[u32; 3]>) -> Function {
    let mut b = FnBuilder::new(if raw.is_some() { "probe_g1250_tdm_raw" } else { "probe_g1250_tdm" }, 64);
    b.arg("out", DType::Ptr, 0, 8);
    for k in 0..12u32 {
        b.arg(&format!("d{k}"), DType::I32, 16 + 4 * k, 4);
    }
    b.arg_tensor("out", Ext::c(64 * 4));
    b.lds(4096);
    let po = b.let_(Op::KernargPtr { offset: 0 });
    let w: Vec<Opnd> = (0..12u32).map(|k| Opnd::Val(b.let_(Op::KernargI32 { offset: 16 + 4 * k }))).collect();
    let g0 = b.let_(Op::RsrcCat { words: [w[0], w[1], w[2], w[3]] });
    let g1 = b.sgpr_cat8([w[4], w[5], w[6], w[7], w[8], w[9], w[10], w[11]]);
    match raw {
        Some(r) => b.tensor_op_raw(true, Opnd::Val(g0), Opnd::Val(g1), None, amdgpu_dsl::TdmCpol::default(), r),
        None => b.tensor_load_to_lds(Opnd::Val(g0), Opnd::Val(g1), None, amdgpu_dsl::TdmCpol::default()),
    }
    b.wait_tensor(0);
    b.barrier();
    let tid = b.let_(Op::WorkitemId);
    let a = b.let_(Op::ShlI32(Opnd::Val(tid), Opnd::I32(2)));
    let x = b.load_lds(a, MemTy::B32);
    let qo = b.let_(Op::PtrAdd { base: Opnd::Val(po), byte_off: Opnd::Val(a) });
    b.store_global(qo, 0, x, MemTy::B32, false);
    b.finish()
}

pub fn probe_g1250_tdm() -> Function {
    probe_g1250_tdm_with(None)
}
