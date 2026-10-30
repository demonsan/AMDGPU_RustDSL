// (F1/F4 regression) token drop/read and chain after zero still rejected.
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let (a, b, sa, sb) = (s.zeroed::<8>(), s.zeroed::<8>(), s.splat(1), s.splat(2));
    let o = s.vgpr::<16>();
    let (ra, rb, rsa, rsb) = (s.ready(&a), s.ready(&b), s.ready(&sa), s.ready(&sb));
    let t = s.v_mfma_scale_f32_32x32x64_f8f6f4_zero(o, ra, rb, rsa, rsb);
    let _ = t;
    let lane = s.lane::<16, 0>(&o);
    let mut x = s.vgpr::<1>();
    s.v_add_f32(&mut x, lane, &sa);
}
