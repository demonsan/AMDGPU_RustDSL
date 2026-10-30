// error-contains: Dst<1>
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let mut i = s.input::<1>(); // input ports are read-only handles
    s.v_mov_b32(&mut i, Imm::<1>::new()); // writing the caller's register: refused
}