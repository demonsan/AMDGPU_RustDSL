// (a) read an in-flight MFMA D WITHOUT dropping the token.
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
    let d0 = s.lane::<16, 0>(&o);            // expected: cannot borrow `o` as immutable ...
    let _ = (t, d0);
}
