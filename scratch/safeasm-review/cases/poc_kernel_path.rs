// POC part 2: the same phase written with the existing DSL -> lower -> regalloc -> emit path, with stores of
// both results so the two variants can be compared on hardware as data (the safe Seq has no store ops).
use amdgpu_codegen::lower;
use amdgpu_dsl::{DType, FnBuilder, Op};
fn main() {
    let mut b = FnBuilder::new("pv_phase_dsl", 64);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let c0 = b.c_i32(0);
    let (c1, c2, c3, c4) = (b.c_i32(1), b.c_i32(2), b.c_i32(3), b.c_i32(4));
    let pa = b.frag8(c0, c0, c0, c0, c0, c0, c0, c0);
    let pb = b.frag8(c0, c0, c0, c0, c0, c0, c0, c0);
    let v0 = b.frag8(c0, c0, c0, c0, c0, c0, c0, c0);
    let v1 = b.frag8(c0, c0, c0, c0, c0, c0, c0, c0);
    let z16 = b.frag_zero(DType::Frag16);
    let o1 = b.mfma_scale_fp8(pa, v0, z16, c1, c2);
    let o2 = b.mfma_scale_fp8(pb, v1, o1, c1, c2);
    let a_rs = b.frag4(c0, c0, c0, c0);
    let z4 = b.frag_zero(DType::Frag4);
    let r1 = b.smfmac_fp8_16x16x128(a_rs, pa, c3, z4);
    let r2 = b.smfmac_fp8_16x16x128(a_rs, pb, c3, r1);
    let r3 = b.smfmac_fp8_16x16x128(a_rs, pa, c4, r2);
    let r4 = b.smfmac_fp8_16x16x128(a_rs, pb, c4, r3);
    let lu = b.frag4_lane(r4, 2);
    let lf = b.bitcast_f32(lu);
    let zf = b.c_f32(0.0);
    let l_new = b.add_f32(zf, lf);
    let o_lane = b.frag16_lane(o2, 0);
    let of = b.bitcast_f32(o_lane);
    b.store_f32(o, 0, l_new);
    b.store_f32(o, 4, of);
    let m = lower::lower(&b.finish()).unwrap();
    let alloc = amdgpu_codegen::regalloc::allocate(&m).unwrap();
    println!("=== KERNEL_PATH ===");
    println!("{}", amdgpu_codegen::emit::emit(&m, &alloc));
}
