// ATTACK (a): read an in-flight MFMA D by dropping the mandatory-retire token with `let _ =` / mem::forget
// and then reading D through the plain `Operand` path (which does not require `Ready`).
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let a = s.zeroed::<8>();
    let b = s.zeroed::<8>();
    let sa = s.splat(127);
    let sb = s.splat(127);
    let mut o = s.vgpr::<16>();
    let (ra, rb, rsa, rsb) = (s.ready(&a), s.ready(&b), s.ready(&sa), s.ready(&sb));
    let t = s.v_mfma_scale_f32_32x32x64_f8f6f4_zero(&mut o, ra, rb, rsa, rsb);
    let _ = t;                        // must_use is only a warning -> borrows released with NO 20-state pad
    let d0 = s.lane::<16, 0>(&o);      // raw VALU read of the matrix result: no Ready needed
    let mut x = s.vgpr::<1>();
    s.v_add_f32(&mut x, &d0, &d0);
    let asm = s.to_asm().unwrap();
    let nops = asm.matches("s_nop").count();
    println!("CASE pass_read_drop: compiled; emitted s_nop count = {nops}");
    for l in asm.lines().filter(|l| l.contains("mfma") || l.contains("v_add") || l.contains("s_nop") || l.contains("v_mov_b32_e32 v0, 0")) {
        println!("  {l}");
    }
}
