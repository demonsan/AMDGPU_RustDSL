// error-contains: no 32-bit literal
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let v = s.splat(1);
    let mut d = s.vgpr::<1>();
    s.v_mul_lo_u32(&mut d, Lit(123456), &v);
}