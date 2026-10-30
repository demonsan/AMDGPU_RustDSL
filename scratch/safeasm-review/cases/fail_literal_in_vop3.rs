// (h) 32-bit literal in a VOP3 form.
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let x = s.splat(1);
    let mut d = s.vgpr::<1>();
    s.v_mul_lo_u32(&mut d, &x, &Lit(12345));   // expected: gfx9 VOP3 ... no 32-bit literal
}
