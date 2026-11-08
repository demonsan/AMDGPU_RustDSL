//! IR-verifier rules of the P1 tile ops (docs/op-coverage-tileops.md).
use amdgpu_dsl::{CmpPred, CmpTy, CvtOp, DType, FnBuilder, Op, Opnd};

/// The scalef32 conversions use only the exponent of the scale (measured: 1.63 acts as 1.0): a constant non-power-of-two scale is rejected.
#[test]
fn a_constant_non_power_of_two_cvt_scale_is_rejected() {
    for (scale, ok) in [(1.63f32, false), (3.0, false), (2.0, true), (0.25, true)] {
        let mut b = FnBuilder::new("cvt_scale", 64);
        let tid = b.let_(Op::WorkitemId);
        let s = b.c_f32(scale);
        let _ = b.cvt(CvtOp::ScalePkF32Fp8, &[Opnd::Val(tid), Opnd::Val(s)], 0);
        let _ = b.cvt(CvtOp::ScalePkFp8F32, &[Opnd::F32(1.0), Opnd::F32(2.0), Opnd::F32(scale), Opnd::I32(0)], 0);
        let f = b.finish();
        let errs = amdgpu_dsl::verify(&f);
        let hits = errs.iter().filter(|e| e.contains("not a power of two")).count();
        assert_eq!(hits == 0, ok, "scale {scale}: {errs:?}");
        if !ok {
            assert_eq!(hits, 2, "both the value and the immediate scale are checked: {errs:?}");
        }
        let _ = DType::F32;
    }
}

/// Pair-adjacent unpack (sched's packed-math layernorm): `bf16x8_to_f32_pairs` feeds `v_pk_mul_f32` / `v_pk_add_f32` without a single
/// register-to-register move.
#[test]
fn bf16_pairs_feed_packed_f32_ops_without_moves() {
    let f = amdgpu_kernels::kernel_by_name("probe_pkpair").unwrap();
    let m = amdgpu_compiler::compile_mir(&f).unwrap();
    let moves = m
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .filter(|i| i.op == amdgpu_compiler::mir::MOp::VMovB32 && matches!(i.srcs.first(), Some(amdgpu_compiler::mir::Src::Reg(_))))
        .count();
    let pk = m.blocks.iter().flat_map(|b| &b.insts).filter(|i| matches!(i.op, amdgpu_compiler::mir::MOp::VPkMulF32 | amdgpu_compiler::mir::MOp::VPkAddF32)).count();
    assert_eq!(pk, 6);
    // the frag4 of the four loaded words and frag2_dup copy; the unpack -> pair path itself adds none
    assert!(moves <= 6, "{moves} register moves");
}

/// kernel-opt: `frag4_lane(var_slice4(v, 1), 2)` must read dword 4 + 2 = 6 of the Frag8 variable (it read dword 2: the lane accessor lost
/// the view's offset).
#[test]
fn a_lane_of_a_slice_view_keeps_the_slice_offset() {
    use amdgpu_dsl::MemTy;
    let mut b = FnBuilder::new("slice_lane", 64);
    b.arg("p", DType::Ptr, 0, 8);
    b.arg_tensor("p", amdgpu_dsl::Ext::Const(4096));
    let p = b.let_(Op::KernargPtr { offset: 0 });
    let z = b.frag_zero(DType::Frag8);
    let vars = b.begin_loop(&[(DType::Frag8, Opnd::Val(z)), (DType::I32, Opnd::I32(0))]);
    let sl = b.var_slice4(vars[0], 1);
    let x = b.frag4_lane(sl, 2);
    b.store_global(p, 0, x, MemTy::B32, false);
    let i2 = b.add_i32(Opnd::Var(vars[1]), 1);
    b.set_var(vars[1], i2);
    let done = b.cmp(CmpPred::Ge, CmpTy::I32, i2, 1);
    b.break_if(done);
    b.end_loop();
    let m = amdgpu_compiler::compile_mir(&b.finish()).unwrap();
    let st = m.blocks.iter().flat_map(|b| &b.insts).find(|i| format!("{:?}", i.op).starts_with("GlobalStore")).expect("store");
    let r = match st.srcs[1] { amdgpu_compiler::mir::Src::Reg(r) => r, _ => panic!("store value") };
    let (base, off) = m.alias.get(&r).map(|a| (a.0, a.1)).expect("the stored dword is a view of the variable");
    assert_eq!(m.vreg_class(base).dwords(), 8);
    assert_eq!(off, 6, "slice 1, lane 2 = dword 6");
}
