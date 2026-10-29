// error-contains: mismatched types
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let a = s.zeroed::<8>();
    let b = s.zeroed::<8>();
    let sa = s.splat(1);
    let o = s.vgpr::<16>();
    let rb = s.ready(&b);
    let rsa = s.ready(&sa);
    // a plain `&Vgpr` is not accepted by a matrix instruction: only `Ready` (the 2-wait VALU->matrix rule)
    let _t = s.v_mfma_scale_f32_32x32x64_f8f6f4_zero(o, &a, rb, rsa, rsa);
}