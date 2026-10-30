// (e) two scalar sources in one VOP3 instruction (constant bus).
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let a = s.sgpr::<1>();
    let b = s.sgpr::<1>();
    let mut d = s.vgpr::<1>();
    s.v_mul_lo_u32(&mut d, &a, &b);          // expected: constant bus limit 1
}
