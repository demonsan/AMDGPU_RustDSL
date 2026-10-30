// (h2) inline constant outside -16..=64.
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let mut d = s.vgpr::<1>();
    let imm = Imm::<65>::new();               // expected: inline constants are -16..=64
    s.v_add_u32(&mut d, &imm, &d);
}
