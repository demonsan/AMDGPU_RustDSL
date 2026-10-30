// error-contains: cannot borrow `a` as mutable because it is also borrowed as immutable
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let mut a = s.zeroed::<8>();
    let b = s.zeroed::<8>();
    let sa = s.splat(1);
    let o = s.vgpr::<16>();
    let (ra, rb, rsa) = (s.ready(&a), s.ready(&b), s.ready(&sa));
    let t = s.v_mfma_scale_f32_32x32x64_f8f6f4_zero(o, ra, rb, rsa, rsa);
    let _w = s.lane_mut::<8, 0>(&mut a); // overwriting an operand the MFMA is still reading: refused
    let _o = t.retire(&mut s);
}