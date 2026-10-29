// ATTACK (A3): `Mfma` does not encode WHICH opcode it is, so `_zero(...).chain(...)` emits a *different*
// opcode back-to-back with no wait, although the hardware interlock is only guaranteed for the same opcode.
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let a = s.zeroed::<8>();
    let b = s.zeroed::<8>();
    let sa = s.splat(127);
    let sb = s.splat(127);
    let mut o = s.vgpr::<16>();
    let (ra, rb, rsa, rsb) = (s.ready(&a), s.ready(&b), s.ready(&sa), s.ready(&sb));
    let t = s.v_mfma_scale_f32_32x32x64_f8f6f4_zero(&mut o, ra, rb, rsa, rsb); // opcode: ...Fp8Z
    let (ca, cb, csa, csb) = (s.ready(&a), s.ready(&b), s.ready(&sa), s.ready(&sb));
    let t = t.chain(s, ca, cb, csa, csb);                                       // opcode: ...Fp8 (different!)
    t.retire(s);
    let asm = s.to_asm().unwrap();
    println!("CASE pass_chain_after_zero: compiled; MFMA lines:");
    for l in asm.lines().filter(|l| l.contains("v_mfma_scale") || l.contains("s_nop")) {
        println!("  {l}");
    }
}
