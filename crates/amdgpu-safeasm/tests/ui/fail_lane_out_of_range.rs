// error-contains: lane index out of range
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let v = s.zeroed::<4>();
    let _l = s.lane::<4, 4>(&v);
}