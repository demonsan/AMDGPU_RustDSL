// error-contains: VgprOnly
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let sg = s.sgpr::<1>();
    let v = s.splat(1);
    let vcc = s.vcc().unwrap();
    // e32 compare: src1 must be a VGPR (put the SGPR in src0, or use the _e64 form)
    let _ = s.v_cmp_lt_i32(vcc, &v, &sg);
}