// error-contains: mismatched types
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let d = s.vgpr::<1>();
    let data = s.splat(1);
    let scale = s.splat(2);
    let pw = s.partial(d);
    let pw = s.v_cvt_scalef32_pk_fp8_bf16_lo(pw, &data, &scale);
    let _pw = s.v_cvt_scalef32_pk_fp8_bf16_lo(pw, &data, &scale); // the low half is already written
}