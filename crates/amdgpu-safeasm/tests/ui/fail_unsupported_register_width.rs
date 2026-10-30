// error-contains: not satisfied
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let _v = s.vgpr::<3>(); // no 3-dword register tuples exist
}