// error-contains: mismatched types
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let v = s.splat(7);
    let mut sg = s.sgpr::<1>();
    s.v_readfirstlane_b32(&mut sg, &v); // needs the 2 wait states: s.ready(&v)
}