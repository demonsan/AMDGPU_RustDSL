// error-contains: mismatched types
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let addr = s.splat(0);
    let d = s.vgpr::<4>();
    let pending = s.ds_read_b128::<0, _>(d, &addr);
    let mut sg = s.sgpr::<1>();
    // a Pending load is not a readable register
    let _ = s.v_readfirstlane_b32(&mut sg, pending);
}