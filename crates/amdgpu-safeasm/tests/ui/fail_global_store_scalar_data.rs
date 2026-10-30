// error-contains: Readable<1>
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let addr = s.vgpr::<2>();
    let sg = s.sgpr::<1>();
    s.global_store_dword(&addr, &sg, 0); // store data must be a VGPR
}