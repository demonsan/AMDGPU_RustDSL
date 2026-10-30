// error-contains: Dst<2>
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let addr = s.splat(0);
    let d = s.vgpr::<4>(); // ds_read_b64_tr_b8 fills 2 dwords
    let _p = s.ds_read_b64_tr_b8::<0, _>(d, &addr);
}