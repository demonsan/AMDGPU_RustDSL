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
