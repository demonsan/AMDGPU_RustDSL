// (b) overwrite an operand of an in-flight MFMA without dropping the token.
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
    let mut b0 = s.lane_mut::<8, 0>(&mut b); // expected: cannot borrow `b` as mutable ...
    let _ = (t, &mut b0);
}
