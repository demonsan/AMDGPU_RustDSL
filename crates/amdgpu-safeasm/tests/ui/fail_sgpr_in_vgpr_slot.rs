// error-contains: VgprOnly
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let sg = s.sgpr::<1>(); // v_cndmask_b32_e32: vcc already uses the constant bus, an SGPR source is illegal
    let v = s.splat(1);
    let mut d = s.vgpr::<1>();
    let vcc = s.vcc().unwrap();
    let live = s.v_cmp_lt_i32(vcc, &v, &v);
    let _ = s.v_cndmask_b32(live, &mut d, &sg, &v);
}