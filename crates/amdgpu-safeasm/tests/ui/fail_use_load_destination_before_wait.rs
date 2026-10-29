// error-contains: borrow of moved value: `dst`
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let addr = s.splat(0);
    let dst = s.vgpr::<4>();
    let pending = s.ds_read_b128(dst, &addr, 0); // the destination register moves into the Pending token
    let _r = s.ready(&dst); // using it before `wait`: refused
    let _v = pending.wait(&mut s);
}