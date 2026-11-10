// (F11) Constant-cache folding must be refused when it would shorten a wait distance: the round-2 "19 dropped splats"
// block must no longer pass finish() with an under-padded matrix read (or the folded stream must verify).
use amdgpu_codegen::asm::block_to_asm;
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
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
    for _ in 0..19 { let _p = s.splat(0x38383838u32 as i32); }
    let res = t.retire(&mut s);
    let d0 = s.lane::<16, 0>(&res);
    let mut x = s.vgpr::<1>();
    s.v_add_u32(&mut x, d0, &sa);
    s.output(x);
    match s.finish() {
        Ok(blk) => {
            println!("CASE pass_const_drop_f11: finish Ok - folding kept the stream valid; checking asm distance");
            if let Ok(asm) = block_to_asm(&blk) {
                let lines: Vec<&str> = asm.lines().collect();
                let mf = lines.iter().position(|l| l.contains("v_mfma")).unwrap_or(0);
                let rd = lines.iter().position(|l| l.contains("v_add_u32")).unwrap_or(lines.len() - 1);
                let states: i64 = lines[mf + 1..rd].iter().map(|l| { let t = l.trim(); if t.starts_with("s_nop") { t.split_whitespace().nth(1).and_then(|v| v.parse::<i64>().ok()).unwrap_or(0) + 1 } else { 1 } }).sum();
                println!("  emitted MFMA->reader wait states = {states} (needs 20; folding must not go below)");
            }
        }
        Err(e) => println!("CASE pass_const_drop_f11: finish Err (folding rejected): {}", e.chars().take(300).collect::<String>()),
    }
}
