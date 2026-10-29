// error-contains: cannot borrow `o` as immutable because it is also borrowed as mutable
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let a = s.zeroed::<8>();
    let b = s.zeroed::<8>();
    let sa = s.splat(1);
    let mut o = s.vgpr::<16>();
    let (ra, rb, rsa) = (s.ready(&a), s.ready(&b), s.ready(&sa));
    let t = s.v_mfma_scale_f32_32x32x64_f8f6f4_zero(&mut o, ra, rb, rsa, rsa);
    let _lane = s.lane::<16, 0>(&o); // reading the MFMA result before the 20 wait states: refused
    t.retire(&mut s);
}