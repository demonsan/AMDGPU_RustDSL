// error-contains: mismatched types
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let (a, b, c) = (s.zeroed::<8>(), s.zeroed::<8>(), s.zeroed::<16>());
    let sa = s.splat(127);
    let mut dst = s.vgpr::<16>();
    let (ra, rb, rc, rsa) = (s.ready(&a), s.ready(&b), s.ready(&c), s.ready(&sa));
    // the destination must be OWNED by the token (moved in); a borrowed register is refused
    let _t = s.v_mfma_scale_f32_32x32x64_f8f6f4_c(&mut dst, rc, ra, rb, rsa, rsa);
}