// error-contains: borrow of moved value: `o1`
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let (a, b) = (s.zeroed::<8>(), s.zeroed::<8>());
    let sa = s.splat(127);
    let o1 = s.vgpr::<16>();
    let o2 = s.vgpr::<16>();
    let (ra, rb, rsa) = (s.ready(&a), s.ready(&b), s.ready(&sa));
    let t1 = s.v_mfma_scale_f32_32x32x64_f8f6f4_zero(o1, ra, rb, rsa, rsa); // o1 moves into the in-flight token
    let rc = s.ready(&o1); // using the in-flight result as the C operand of the next MFMA: refused
    let _t2 = s.v_mfma_scale_f32_32x32x64_f8f6f4_c(o2, rc, ra, rb, rsa, rsa);
    let _ = t1.retire(&mut s);
}