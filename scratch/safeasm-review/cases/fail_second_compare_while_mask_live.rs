// (f) a second compare while a vcc mask is live.
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let x = s.splat(1);
    let y = s.splat(2);
    let v = s.vcc();
    let live = s.v_cmp_lt_i32(v, &x, &y);
    let _ = s.v_cmp_lt_i32(live, &x, &y);   // expected: expected Vcc<Free>, found Vcc<Live>
}
