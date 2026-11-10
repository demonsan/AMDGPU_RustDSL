// (NEW, safe path) End-to-end: embed the const-drop block (real matrix result read too early) in a DSL kernel, compile,
// print the asm; the job assembles it and runs it on GPU (expected stale 0 instead of 64).
use amdgpu_dsl::{DType, FnBuilder, Op};
use amdgpu_codegen::asm::block_to_asm;
use amdgpu_safeasm::*;
fn build() -> AsmBlock {
    let mut s = Seq::new();
    let mut a = s.vgpr::<8>();
    let mut b = s.vgpr::<8>();
    let ones = s.splat(0x38383838u32 as i32);
    let l0 = s.lane_mut::<8, 0>(&mut a); s.v_mov_b32(l0, &ones);
    let l1 = s.lane_mut::<8, 1>(&mut a); s.v_mov_b32(l1, &ones);
    let l2 = s.lane_mut::<8, 2>(&mut a); s.v_mov_b32(l2, &ones);
    let l3 = s.lane_mut::<8, 3>(&mut a); s.v_mov_b32(l3, &ones);
    let l4 = s.lane_mut::<8, 4>(&mut a); s.v_mov_b32(l4, &ones);
    let l5 = s.lane_mut::<8, 5>(&mut a); s.v_mov_b32(l5, &ones);
    let l6 = s.lane_mut::<8, 6>(&mut a); s.v_mov_b32(l6, &ones);
    let l7 = s.lane_mut::<8, 7>(&mut a); s.v_mov_b32(l7, &ones);
    let m0 = s.lane_mut::<8, 0>(&mut b); s.v_mov_b32(m0, &ones);
    let m1 = s.lane_mut::<8, 1>(&mut b); s.v_mov_b32(m1, &ones);
    let m2 = s.lane_mut::<8, 2>(&mut b); s.v_mov_b32(m2, &ones);
    let m3 = s.lane_mut::<8, 3>(&mut b); s.v_mov_b32(m3, &ones);
    let m4 = s.lane_mut::<8, 4>(&mut b); s.v_mov_b32(m4, &ones);
    let m5 = s.lane_mut::<8, 5>(&mut b); s.v_mov_b32(m5, &ones);
    let m6 = s.lane_mut::<8, 6>(&mut b); s.v_mov_b32(m6, &ones);
    let m7 = s.lane_mut::<8, 7>(&mut b); s.v_mov_b32(m7, &ones);
    let sa = s.splat(127);
    let sb = s.splat(127);
    let o = s.vgpr::<16>();
    let (ra, rb, rsa, rsb) = (s.ready(&a), s.ready(&b), s.ready(&sa), s.ready(&sb));
    let t = s.v_mfma_scale_f32_32x32x64_f8f6f4_zero(o, ra, rb, rsa, rsb);
    for _ in 0..19 { let _p = s.splat(0x38383838u32 as i32); }
    let o = t.retire(&mut s);
    let d0 = s.lane::<16, 0>(&o);
    let mut x = s.vgpr::<1>();
    s.v_add_u32(&mut x, d0, &sa);
    s.output(x);
    s.finish().unwrap()
}
fn main() {
    // standalone block asm (for inspection)
    match block_to_asm(&build()) {
        Ok(asm) => {
            println!("=== BLOCK_ASM ===");
            println!("{asm}");
        }
        Err(e) => println!("block_to_asm Err: {e}"),
    }
    let mut f = FnBuilder::new("const_drop_kernel", 64);
    f.arg("o", DType::Ptr, 0, 8);
    let o = f.let_(Op::KernargPtr { offset: 0 });
    let out = f.asm(build(), &[]);
    let of = f.bitcast_f32(out);
    f.store_f32(o, 0, of);
    let func = f.finish();
    match amdgpu_codegen::compile(&func) {
        Ok(r) => {
            println!("=== KERNEL_ASM ===");
            println!("{}", r.asm);
        }
        Err(e) => println!("compile Err: {e}"),
    }
}
