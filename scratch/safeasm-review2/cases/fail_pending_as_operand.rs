// (1e) A Pending token is not a readable register. Expected: mismatched types.
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let addr = s.splat(0);
    let d = s.vgpr::<4>();
    let p = s.ds_read_b128(d, &addr, 0);
    let mut sg = s.sgpr::<1>();
    let _ = s.v_readfirstlane_b32(&mut sg, p);
}
