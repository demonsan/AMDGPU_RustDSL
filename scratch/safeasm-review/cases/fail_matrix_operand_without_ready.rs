// (c) forget the 2-wait rule: use a plain register instead of a `Ready` proof.
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let a = s.zeroed::<8>();
    let mut o = s.vgpr::<16>();
    let sa = s.splat(1);
    let b = s.zeroed::<8>();
    let _ = s.v_mfma_scale_f32_32x32x64_f8f6f4_zero(&mut o, &a, &b, s.ready(&sa), s.ready(&sa)); // expected: mismatched types (Ready expected)
}
