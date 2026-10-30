// error-contains: mismatched types
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let x = s.splat(1);
    let d = s.vgpr::<1>();
    let t = s.v_exp_f32(d, &x);
    let mut y = s.vgpr::<1>();
    s.v_add_f32(&mut y, &x, t); // the exp result is not readable before settle()
}