// (F8 variants) A mutable view of an input port must not exist either.
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let mut x = s.input::<4>();
    let lane = s.lane_mut::<4, 0>(&mut x);
    let y = s.splat(1);
    s.v_mov_b32(lane, &y);
}
