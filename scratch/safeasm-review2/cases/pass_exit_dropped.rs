// (defense) exit(): dropping the Exit without output_exit must be refused by the end-of-sequence verifier rule.
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let a = s.zeroed::<8>();
    let b = s.zeroed::<8>();
    let sa = s.splat(1);
    let sb = s.splat(2);
    let o = s.vgpr::<16>();
    let (ra, rb, rsa, rsb) = (s.ready(&a), s.ready(&b), s.ready(&sa), s.ready(&sb));
    let t = s.v_mfma_scale_f32_32x32x64_f8f6f4_zero(o, ra, rb, rsa, rsb);
    let e = t.exit();                    // not exported
    drop(e);
    println!("CASE pass_exit_dropped: finish() = {:?}", s.finish().err());
}
