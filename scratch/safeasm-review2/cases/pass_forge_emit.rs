// (4b) Forged encoding-illegal instructions that pass (or fail) the hazard verifier, then handed to the compiler's public
// `block_to_asm` / `block_to_mfunction` / `compile` path: does an illegal sequence escape as valid-looking asm?
use amdgpu_compiler::asm::{block_to_asm, block_to_mfunction};
use amdgpu_safeasm::*;
fn base() -> AsmBlock {
    let mut s = Seq::new();
    let a = s.zeroed::<8>();
    let b = s.zeroed::<8>();
    let sa = s.splat(1);
    let sb = s.splat(2);
    let o = s.vgpr::<16>();
    let (ra, rb, rsa, rsb) = (s.ready(&a), s.ready(&b), s.ready(&sa), s.ready(&sb));
    let o = s.v_mfma_scale_f32_32x32x64_f8f6f4_zero(o, ra, rb, rsa, rsb).retire(&mut s);
    let lane = s.lane::<16, 0>(&o);
    let mut x = s.vgpr::<1>();
    s.v_add_f32(&mut x, lane, &sa);
    s.output(x);
    s.finish().unwrap()
}
fn main() {
    let b = base();
    let v16 = b.regs().iter().position(|c| *c == AClass::V16).unwrap() as u32;
    let v1 = b.regs().iter().position(|c| *c == AClass::V1).unwrap() as u32;
    let mut insts = b.insts().to_vec();
    // 3 sources on a 2-source VALU op; readfirstlane with a VGPR destination; matrix op with a V1 destination
    insts.push(AInst { op: "VAddF32", dst: Some(v1), srcs: vec![ASrc::Reg(v1), ASrc::Reg(v1), ASrc::Reg(v1)], imm: 0 });
    insts.push(AInst { op: "VReadFirstLane", dst: Some(v1), srcs: vec![ASrc::Reg(v1)], imm: 0 });
    insts.push(AInst { op: "VMfmaScale32x32x64Fp8Z", dst: Some(v1), srcs: vec![ASrc::Reg(v1), ASrc::Reg(v1), ASrc::Reg(v1), ASrc::Reg(v1)], imm: 0 });
    insts.push(AInst { op: "VMovB32", dst: Some(v16), srcs: vec![ASrc::Reg(v1)], imm: 0 });
    let f = forge_for_tests(&b, insts);
    println!("CASE pass_forge_emit: verify = {:?}", f.verify().err().map(|e| e.join("; ")));
    match block_to_asm(&f) {
        Ok(asm) => {
            let bad: Vec<&str> = asm.lines().filter(|l| l.contains("v_add_f32") || l.contains("v_readfirstlane") || l.contains("v_mfma") || l.contains("v_mov_b32")).collect();
            println!("  block_to_asm OK; suspect lines:");
            for l in bad { println!("    {}", l.trim()); }
        }
        Err(e) => println!("  block_to_asm Err: {e}"),
    }
    match block_to_mfunction(&f, "forged") {
        Ok(m) => {
            let n: usize = m.blocks.iter().map(|b| b.insts.len()).sum();
            println!("  block_to_mfunction OK ({n} MIR insts) - the compiler accepted a forged, encoding-illegal block");
        }
        Err(e) => println!("  block_to_mfunction Err: {e}"),
    }
}
