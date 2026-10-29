// error-contains: mismatched types
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let a4 = s.zeroed::<4>(); // a 4-dword fragment where the scaled MFMA needs 8
    let b = s.zeroed::<8>();
    let sa = s.splat(1);
    let mut o = s.vgpr::<16>();
    let (ra, rb, rsa) = (s.ready(&a4), s.ready(&b), s.ready(&sa));
    let _t = s.v_mfma_scale_f32_32x32x64_f8f6f4_zero(&mut o, ra, rb, rsa, rsa);
}