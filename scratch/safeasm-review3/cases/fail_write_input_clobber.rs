// (F8) The round-2 hardware reproducer: write an input port. `Seq::input` now returns `In<N>` (read-only) - the write must
// not type-check. Expected: `&mut In<1>: Dst<1>` is not satisfied.
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let x = s.input::<1>();
    let y = s.splat(1);
    s.v_add_u32(&mut x, &y, &y);
    let _ = x;
}
