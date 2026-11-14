//! gpu_verify: ring counters (an offset cycling through a fixed set of slots) and their shadows are proven from the IR (no assume).

use amdgpu_dsl::{CmpPred, CmpTy, DType, FnBuilder, Op, Opnd};

/// ring = 0, 16640, 33280, 0, ... (`r += r >= 33280 ? -33280 : 16640`); shadow = 64 + ring. LDS reads at ring + 4 and shadow + 8.
fn ring_kernel(step: i32) -> amdgpu_dsl::Function {
    let mut b = FnBuilder::new("probe_gv_ring", 64);
    let v = b.begin_loop(&[(DType::I32, Opnd::I32(0)), (DType::I32, Opnd::I32(64)), (DType::I32, Opnd::I32(0))]);
    let (r, s, i) = (v[0], v[1], v[2]);
    let a = b.let_(Op::AddI32(Opnd::Var(r), Opnd::I32(4)));
    let _x = b.load_lds_b32(a);
    let a2 = b.let_(Op::AddI32(Opnd::Var(s), Opnd::I32(8)));
    let _y = b.load_lds_b32(a2);
    let c = b.let_(Op::Cmp { pred: CmpPred::Ge, ty: CmpTy::U32, a: Opnd::Var(r), b: Opnd::I32(33280) });
    let d = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(c), a: Opnd::I32(-33280), b: Opnd::I32(step) });
    let r2 = b.let_(Op::AddI32(Opnd::Val(d), Opnd::Var(r)));
    let s2 = b.let_(Op::AddI32(Opnd::Val(d), Opnd::Var(s)));
    b.set_var(r, r2);
    b.set_var(s, s2);
    let i2 = b.let_(Op::AddI32(Opnd::Var(i), Opnd::I32(1)));
    b.set_var(i, i2);
    let done = b.let_(Op::Cmp { pred: CmpPred::Ge, ty: CmpTy::I32, a: Opnd::Val(i2), b: Opnd::I32(8) });
    b.break_if(done);
    b.end_loop();
    let mut f = b.finish();
    f.lds_bytes = 33280 + 64 + 16;
    f
}

fn lds_oob(f: &amdgpu_dsl::Function) -> Vec<String> {
    amdgpu_codegen::gpu_verify::verify_ir(f).errors().iter().filter(|e| e.code == "GV-LDS-OOB").map(|e| e.msg.clone()).collect()
}

#[test]
fn a_ring_counter_and_its_shadow_are_proven_in_bounds() {
    let e = lds_oob(&ring_kernel(16640));
    assert!(e.is_empty(), "{e:?}");
}

#[test]
fn a_drifting_counter_is_not_a_ring() {
    // 16641: the values never repeat within the enumeration bound -> no ring fact -> the reads are unbounded
    let e = lds_oob(&ring_kernel(16641));
    assert!(!e.is_empty());
}
