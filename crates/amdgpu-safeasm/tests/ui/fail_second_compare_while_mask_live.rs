// error-contains: mismatched types
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let v = s.splat(1);
    let vcc = s.vcc().unwrap();
    let live = s.v_cmp_lt_i32(vcc, &v, &v);
    let _again = s.v_cmp_lt_i32(live, &v, &v); // vcc still holds a live mask: needs Vcc<Free>
}