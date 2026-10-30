// (1b) F1 closed? Alias the accumulator BEFORE the matrix op and then move the register into the token: the alias keeps the
// register borrowed, so the move must be refused. Expected: cannot move out while borrowed.
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let (a, b, sa, sb) = (s.zeroed::<8>(), s.zeroed::<8>(), s.splat(1), s.splat(2));
    let o = s.vgpr::<16>();
    let lane = s.lane::<16, 0>(&o);     // borrow of `o`
    let (ra, rb, rsa, rsb) = (s.ready(&a), s.ready(&b), s.ready(&sa), s.ready(&sb));
    let t = s.v_mfma_scale_f32_32x32x64_f8f6f4_zero(o, ra, rb, rsa, rsb); // move while `lane` borrows `o`
    let _ = (t, lane);
}
