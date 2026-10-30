// (1c) F1 closed? Move the accumulator out of an Option slot while an alias exists (safe Rust cannot bypass the borrow).
// Expected: cannot move out of `slot` because it is borrowed.
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let (a, b, sa, sb) = (s.zeroed::<8>(), s.zeroed::<8>(), s.splat(1), s.splat(2));
    let mut slot: Option<Vgpr<16>> = Some(s.vgpr::<16>());
    let lane = s.lane::<16, 0>(slot.as_ref().unwrap());
    let (ra, rb, rsa, rsb) = (s.ready(&a), s.ready(&b), s.ready(&sa), s.ready(&sb));
    let o = slot.take().unwrap();       // move while borrowed
    let t = s.v_mfma_scale_f32_32x32x64_f8f6f4_zero(o, ra, rb, rsa, rsb);
    let _ = (t, lane);
}
