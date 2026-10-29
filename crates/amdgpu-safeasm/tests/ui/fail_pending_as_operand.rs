// error-contains: mismatched types
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let addr = s.splat(0);
    let pending = s.ds_read_b128(s.vgpr::<4>(), &addr, 0);
    let mut sg = s.sgpr::<1>();
    // a Pending load is not a readable register
    let _ = s.v_readfirstlane_b32(&mut sg, pending);
}