// (NEW, safe path, REAL hazard) The verifier's clock counts materialised-constant moves (`splat`) as wait states, but the
// compiler's lowering REMOVES them through the constant cache (only the first use of a value materialises a move). Put 19
// unused splats between an MFMA and its retire: Seq/verifier believe 19 states already separate the write from the reader and
// pad only 1, while the emitted stream contains *no* instruction there -> the matrix result is read 1 state after the MFMA
// (hardware needs 20). This case prints the emitted stream; `pass_const_drop_kernel.rs` runs it on hardware.
use amdgpu_codegen::asm::block_to_asm;
use amdgpu_safeasm::*;
fn build() -> AsmBlock {
    let mut s = Seq::new();
    // nonzero operands: A = B = fp8 1.0, scales 127 -> MFMA result = 64 per lane in dword 0
    let mut a = s.vgpr::<8>();
    let mut b = s.vgpr::<8>();
    let ones = s.splat(0x38383838u32 as i32);
    let l0 = s.lane_mut::<8, 0>(&mut a); s.v_mov_b32(l0, &ones);
    let l1 = s.lane_mut::<8, 1>(&mut a); s.v_mov_b32(l1, &ones);
    let l2 = s.lane_mut::<8, 2>(&mut a); s.v_mov_b32(l2, &ones);
    let l3 = s.lane_mut::<8, 3>(&mut a); s.v_mov_b32(l3, &ones);
    let l4 = s.lane_mut::<8, 4>(&mut a); s.v_mov_b32(l4, &ones);
    let l5 = s.lane_mut::<8, 5>(&mut a); s.v_mov_b32(l5, &ones);
    let l6 = s.lane_mut::<8, 6>(&mut a); s.v_mov_b32(l6, &ones);
    let l7 = s.lane_mut::<8, 7>(&mut a); s.v_mov_b32(l7, &ones);
    let m0 = s.lane_mut::<8, 0>(&mut b); s.v_mov_b32(m0, &ones);
    let m1 = s.lane_mut::<8, 1>(&mut b); s.v_mov_b32(m1, &ones);
    let m2 = s.lane_mut::<8, 2>(&mut b); s.v_mov_b32(m2, &ones);
    let m3 = s.lane_mut::<8, 3>(&mut b); s.v_mov_b32(m3, &ones);
    let m4 = s.lane_mut::<8, 4>(&mut b); s.v_mov_b32(m4, &ones);
    let m5 = s.lane_mut::<8, 5>(&mut b); s.v_mov_b32(m5, &ones);
    let m6 = s.lane_mut::<8, 6>(&mut b); s.v_mov_b32(m6, &ones);
    let m7 = s.lane_mut::<8, 7>(&mut b); s.v_mov_b32(m7, &ones);
    let sa = s.splat(127);
    let sb = s.splat(127);
    let o = s.vgpr::<16>();
    let (ra, rb, rsa, rsb) = (s.ready(&a), s.ready(&b), s.ready(&sa), s.ready(&sb));
    let t = s.v_mfma_scale_f32_32x32x64_f8f6f4_zero(o, ra, rb, rsa, rsb);
    // 19 unused splats of an already-materialised constant: each counts as a wait state for Seq/verifier, every one is
    // removed by the lowering's constant cache.
    for _ in 0..19 { let _p = s.splat(0x38383838u32 as i32); }
    let o = t.retire(&mut s);            // believes 19 states are already there: pads only 1
    let d0 = s.lane::<16, 0>(&o);
    let mut x = s.vgpr::<1>();
    s.v_add_u32(&mut x, d0, &sa);
    s.output(x);
    s.finish().expect("verifier accepts: it counts the 19 constant moves as wait states")
}
fn main() {
    let blk = build();
    println!("CASE pass_const_drop_matrix: finish() OK with the 19 constant moves as padding");
    match block_to_asm(&blk) {
        Ok(asm) => {
            let lines: Vec<&str> = asm.lines().collect();
            let mf = lines.iter().position(|l| l.contains("v_mfma_scale")).unwrap();
            let rd = lines.iter().position(|l| l.contains("v_add_u32")).unwrap();
            println!("  emitted asm MFMA -> reader ({} lines):", rd - mf);
            for l in &lines[mf..=rd] { println!("    {}", l.trim()); }
            let states: i64 = lines[mf + 1..rd].iter().map(|l| { let t = l.trim(); if t.starts_with("s_nop") { t.split_whitespace().nth(1).and_then(|v| v.parse::<i64>().ok()).unwrap_or(0) + 1 } else { 1 } }).sum();
            println!("  real wait states between the MFMA and the reader = {states} (the verifier requires 20)");
        }
        Err(e) => println!("  block_to_asm Err: {e}"),
    }
}
