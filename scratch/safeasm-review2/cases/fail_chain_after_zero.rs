// (3) F4 closed? Chain across opcodes: the `_zero` token has no `chain` (only the accumulate opcode does). Expected: no method.
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let (a, b, sa, sb) = (s.zeroed::<8>(), s.zeroed::<8>(), s.splat(1), s.splat(2));
    let o = s.vgpr::<16>();
    let (ra, rb, rsa, rsb) = (s.ready(&a), s.ready(&b), s.ready(&sa), s.ready(&sb));
    let t = s.v_mfma_scale_f32_32x32x64_f8f6f4_zero(o, ra, rb, rsa, rsb);
    let _t2 = t.chain(&mut s, ra, rb, rsa, rsb);   // expected: no method `chain` for Mfma<_, OP_MFMA_SCALE_ZERO>
}
