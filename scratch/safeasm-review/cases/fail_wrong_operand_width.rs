// (g) wrong-width fragment/operand: an 8-dword fragment is not a valid e32 VALU operand.
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let v8 = s.vgpr::<8>();
    let mut d = s.vgpr::<1>();
    s.v_add_u32(&mut d, &v8, &d);        // expected: &Vgpr<8> is not an Operand (only Vgpr<1>/Sub<1>/Sgpr<1>)
}
