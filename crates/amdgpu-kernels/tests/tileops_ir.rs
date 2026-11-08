//! IR-verifier rules of the P1 tile ops (docs/op-coverage-tileops.md).
use amdgpu_dsl::{CvtOp, DType, FnBuilder, Op, Opnd};

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
