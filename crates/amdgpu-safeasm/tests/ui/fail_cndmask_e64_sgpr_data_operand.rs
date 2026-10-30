// error-contains: SGPR-pair mask through the constant bus
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let sg = s.sgpr::<1>();
    let v = s.splat(1);
    let mut m = s.sgpr::<2>();
    s.v_cmp_lt_i32_e64(&mut m, &v, &v);
    let mut d = s.vgpr::<1>();
    s.v_cndmask_b32_e64(&mut d, &sg, &v, &m);
}