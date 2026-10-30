// (1d) A Pending load owns its destination: use it before `wait` must be refused. Expected: borrow of moved value.
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let addr = s.splat(0);
    let dst = s.vgpr::<4>();
    let p = s.ds_read_b128(dst, &addr, 0);
    let r = s.ready(&dst);
    let _v = p.wait(&mut s);
    let _ = r;
}
