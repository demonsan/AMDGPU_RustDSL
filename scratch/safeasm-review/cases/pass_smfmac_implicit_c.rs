// ATTACK (d): SMFMAC implicit C written by VALU immediately before the SMFMAC - verify the builder pads 2 states.
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let a = s.zeroed::<4>();
    let b = s.zeroed::<8>();
    let idx = s.splat(3);
    let mut acc = s.vgpr::<4>();
    let z = s.splat(0);
    let l0 = s.lane_mut::<4, 0>(&mut acc);
    s.v_mov_b32(l0, &z);
    let l1 = s.lane_mut::<4, 1>(&mut acc);
    s.v_mov_b32(l1, &z);
    let l2 = s.lane_mut::<4, 2>(&mut acc);
    s.v_mov_b32(l2, &z);
    let l3 = s.lane_mut::<4, 3>(&mut acc);
    s.v_mov_b32(l3, &z);
    let (ra, rb, ri) = (s.ready(&a), s.ready(&b), s.ready(&idx));
    let t = s.v_smfmac_f32_16x16x128_fp8_fp8(&mut acc, ra, rb, ri);
    t.retire(&mut s);
    let asm = s.to_asm().unwrap();
    println!("CASE pass_smfmac_implicit_c: compiled; asm (look for s_nop before the first smfmac):");
    let mut seen_c_zero = false;
    for l in asm.lines() {
        if l.contains("smfmac") && !seen_c_zero { println!("  [smfmac] {l}"); seen_c_zero = true; }
        else if l.contains("s_nop") || l.contains("v_mov_b32_e32") { if !seen_c_zero { println!("  {l}"); } }
    }
}
