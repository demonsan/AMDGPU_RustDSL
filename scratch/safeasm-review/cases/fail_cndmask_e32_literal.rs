// (e) v_cndmask_b32_e32 reads vcc through the constant bus: a literal data operand is illegal.
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let x = s.splat(1);
    let y = s.splat(2);
    let mut d = s.vgpr::<1>();
    let v = s.vcc();
    let live = s.v_cmp_lt_i32(v, &x, &y);
    s.v_cndmask_b32(live, &mut d, &Lit(7), &y); // expected: &Lit: VgprOnly is not satisfied
}
