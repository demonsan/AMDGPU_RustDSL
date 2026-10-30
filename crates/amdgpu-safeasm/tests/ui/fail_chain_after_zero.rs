// error-contains: is not an iterator
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let a = s.zeroed::<8>();
    let b = s.zeroed::<8>();
    let sa = s.splat(1);
    let o = s.vgpr::<16>();
    let (ra, rb, rsa) = (s.ready(&a), s.ready(&b), s.ready(&sa));
    let t = s.v_mfma_scale_f32_32x32x64_f8f6f4_zero(o, ra, rb, rsa, rsa);
    // the `_zero` form is a different opcode: only `retire` (full latency) is available, `chain` is not defined for it
    let _t = t.chain(&mut s, ra, rb, rsa, rsa);
}