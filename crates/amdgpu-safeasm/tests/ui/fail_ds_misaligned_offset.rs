// error-contains: ds offset must be aligned to the access size
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let addr = s.splat(0);
    let d = s.vgpr::<4>();
    let _p = s.ds_read_b128::<8, _>(d, &addr); // b128 needs a 16-byte aligned offset
}