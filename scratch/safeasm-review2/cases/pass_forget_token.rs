// (defense) `mem::forget` skips the token Drop (no poison flag), so only the independent verifier can catch the abandoned
// matrix write. Check that finish() still refuses.
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
    std::mem::forget(t);                  // Drop never runs: poison stays empty
    let e = s.finish().unwrap_err();
    println!("CASE pass_forget_token: finish() refused without the poison flag: {e}");
    assert!(e.contains("unretired matrix result") || e.contains("in flight"), "{e}");
}
