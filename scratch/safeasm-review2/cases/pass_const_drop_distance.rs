// (NEW, safe path) `Seq` counts a materialised constant (`splat`, `VMovB32 imm`) as one wait state when computing `Ready`
// padding, but the compiler's asm lowering DROPS such constant moves (const cache), so the emitted instruction stream can be
// one state shorter than the verifier's model. Build exactly that: VALU write -> splat -> ready(matrix idx operand).
use amdgpu_compiler::asm::block_to_asm;
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let a = s.zeroed::<4>();
    let b = s.zeroed::<8>();
    let mut idx = s.vgpr::<1>();
    let one = s.splat(1);                // VMovB32 imm 1: counted as one state by Seq/verifier
    s.v_add_u32(&mut idx, &one, &one);   // idx = 2 (VALU write, needs 2 states before the matrix read)
    let constant_pad = s.splat(7);       // one state in the model, but lowering drops it as a const
    let ri = s.ready(&idx);              // have = 1 state -> pads only one more state
    let acc = s.zeroed::<4>();
    let ra = s.ready(&a);
    let rb = s.ready(&b);
    let t = s.v_smfmac_f32_16x16x128_fp8_fp8(acc, ra, rb, ri);
    let r = t.retire(&mut s);
    s.output(r);
    let blk = s.finish().unwrap();
    println!("CASE pass_const_drop_distance: finish() OK (the verifier sees the constant move as one state)");
    let _ = constant_pad;
    match block_to_asm(&blk) {
        Ok(asm) => {
            let lines: Vec<&str> = asm.lines().collect();
            let write = lines.iter().position(|l| l.contains("v_add_u32") && l.contains(&", 1, 1")).unwrap_or(0);
            let read = lines.iter().position(|l| l.contains("v_smfmac")).unwrap_or(0);
            println!("  emitted asm between the idx VALU write and the SMFMAC idx read:");
            for l in &lines[write..=read] { println!("    {}", l.trim()); }
            let states: usize = lines[write + 1..read].iter().map(|l| if l.trim().starts_with("s_nop") { 1 } else { 1 }).count();
            println!("  real instructions between = {states} (the model/verifier counted one more, the dropped const move)");
        }
        Err(e) => println!("  block_to_asm Err: {e}"),
    }
}
