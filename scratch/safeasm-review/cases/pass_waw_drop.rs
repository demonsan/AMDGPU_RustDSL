// ATTACK (b2): overwrite the accumulator D of an in-flight MFMA (WAW) after dropping the token.
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
    let _ = t;
    let mut o0 = s.lane_mut::<16, 0>(&mut o);
    let z = s.splat(0);
    s.v_mov_b32(&mut o0, &z);
    println!("CASE pass_waw_drop: compiled (WAW on an in-flight matrix D), asm:");
    for l in s.to_asm().unwrap().lines().filter(|l| l.contains("mfma") || l.contains("v_mov") || l.contains("s_nop")) {
        println!("  {l}");
    }
}
