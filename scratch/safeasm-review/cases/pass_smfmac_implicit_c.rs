// ATTACK (d/a3): the SMFMAC implicit C. Can the C accumulator be written by VALU with no padding?
// (the type system calls make_ready_pub on the accumulator, so this is expected to PAD - verify the asm.)
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let a = s.zeroed::<4>();
    let b = s.zeroed::<8>();
    let idx = s.splat(3);
    let mut acc = s.vgpr::<4>();
    let z = s.splat(0);
    for i in 0..4u8 {
        let mut l = s.lane_mut::<4, i>(&mut acc);
        s.v_mov_b32(&mut l, &z);        // C written by VALU immediately before the SMFMAC
    }
    let (ra, rb, ri) = (s.ready(&a), s.ready(&b), s.ready(&idx));
    let t = s.v_smfmac_f32_16x16x128_fp8_fp8(&mut acc, ra, rb, ri);
    t.retire(&mut s);
    let asm = s.to_asm().unwrap();
    let after_mov = asm.lines().skip_while(|l| !l.contains("v_mov_b32_e32 v")).take_while(|l| !l.contains("smfmac")).collect::<Vec<_>>();
    let nops = after_mov.iter().filter(|l| l.contains("s_nop")).count();
    println!("CASE pass_smfmac_implicit_c: compiled; s_nop between the last C-zeroing VALU and the SMFMAC = {nops} (>=1 means the 2-state rule is applied)");
    for l in asm.lines().filter(|l| l.contains("smfmac") || l.contains("s_nop")) {
        println!("  {l}");
    }
}
