// (F8 variants) `In<N>` must not be usable as a matrix/normal write destination through any trait.
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let x = s.input::<4>();
    let y = s.zeroed::<4>();
    // smfmac takes its accumulator by value from an owned Vgpr; an In port cannot move there
    let r = s.ready(&y);
    let _t = s.v_smfmac_f32_16x16x128_fp8_fp8(x, r, r, r);
}
