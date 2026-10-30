// ATTACK (b): overwrite an operand of an in-flight MFMA (WAR) by dropping the token first.
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let a = s.zeroed::<8>();
    let mut b = s.zeroed::<8>();
    let sa = s.splat(127);
    let sb = s.splat(127);
    let mut o = s.vgpr::<16>();
    let (ra, rb, rsa, rsb) = (s.ready(&a), s.ready(&b), s.ready(&sa), s.ready(&sb));
    let t = s.v_mfma_scale_f32_32x32x64_f8f6f4_zero(&mut o, ra, rb, rsa, rsb);
    let _ = t;                         // release operand borrows without retiring
    let b0 = s.lane_mut::<8, 0>(&mut b);       // WAR: overwrite a dword the MFMA is still reading
    let z = s.splat(0);
    s.v_mov_b32(b0, &z);
    let asm = s.to_asm().unwrap();
    println!("CASE pass_war_drop: compiled; matrix->overwrite distance (no s_nop expected):");
    for l in asm.lines().filter(|l| l.contains("mfma") || l.contains("v_mov") || l.contains("s_nop")) {
        println!("  {l}");
    }
}
