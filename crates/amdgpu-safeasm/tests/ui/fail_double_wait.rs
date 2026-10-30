// error-contains: use of moved value: `pending`
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let addr = s.splat(0);
    let d = s.vgpr::<4>();
    let pending = s.ds_read_b128::<0>(d, &addr);
    let _a = pending.wait(&mut s);
    let _b = pending.wait(&mut s); // a load can be waited for once; the token is consumed
}