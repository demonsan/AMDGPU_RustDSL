// error-contains: constant bus
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let s1 = s.sgpr::<1>();
    let s2 = s.sgpr::<1>();
    let mut d = s.vgpr::<1>();
    s.v_mul_lo_u32(&mut d, &s1, &s2); // gfx9: one scalar source per VALU instruction
}