// (defense) F2: dropping the token then overwriting an operand must be reported by the poison flag AND the verifier.
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let a = s.zeroed::<8>();
    let mut b = s.zeroed::<8>();
    let sa = s.splat(1);
    let sb = s.splat(2);
    let o = s.vgpr::<16>();
    let (ra, rb, rsa, rsb) = (s.ready(&a), s.ready(&b), s.ready(&sa), s.ready(&sb));
    let t = s.v_mfma_scale_f32_32x32x64_f8f6f4_zero(o, ra, rb, rsa, rsb);
    drop(t);                              // poison + operand borrows released
    let z = s.splat(0);
    let b0 = s.lane_mut::<8, 0>(&mut b);  // WAR: overwrite an operand still being read
    s.v_mov_b32(b0, &z);
    let e = s.finish().unwrap_err();
    println!("CASE pass_war_poison: finish() refused: {e}");
    assert!(e.contains("unretired matrix result") && e.contains("WAR"), "{e}");
}
