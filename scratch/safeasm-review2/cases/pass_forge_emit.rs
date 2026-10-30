// (4b, fixed) Encoding-illegal but HAZARD-CLEAN forged instructions: the verifier only models hazards, so this should pass
// verify(); what do the compiler's public block_to_asm / block_to_mfunction do with it?
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
    let v1 = b.regs().iter().position(|c| *c == AClass::V1).unwrap() as u32;
    let mut insts = b.insts().to_vec();
    insts.push(AInst { op: "SNop", dst: None, srcs: vec![], imm: 19 }); // 20 states
    // 1) a 3-source e32 VALU op (the template has 2 operands)
    let mut i1 = insts.clone();
    i1.push(AInst { op: "VAddF32", dst: Some(v1), srcs: vec![ASrc::Reg(v1), ASrc::Reg(v1), ASrc::Reg(v1)], imm: 0 });
    let f1 = forge_for_tests(&b, i1);
    println!("CASE pass_forge_emit: 3-source VAddF32: verify = {:?}", f1.verify().err().map(|e| e.join("; ")));
    println!("  block_to_asm = {:?}", block_to_asm(&f1).map(|a| a.lines().filter(|l| l.contains("v_add")).map(|l| l.trim().to_string()).collect::<Vec<_>>()).map_err(|e| e.chars().take(160).collect::<String>()));
    println!("  block_to_mfunction = {:?}", block_to_mfunction(&f1, "forged").map(|m| m.blocks.iter().map(|b| b.insts.len()).sum::<usize>()).map_err(|e| e.chars().take(160).collect::<String>()));
    // 2) v_readfirstlane with a VGPR destination
    let mut i2 = insts.clone();
    i2.push(AInst { op: "VReadFirstLane", dst: Some(v1), srcs: vec![ASrc::Reg(v1)], imm: 0 });
    let f2 = forge_for_tests(&b, i2);
    println!("CASE pass_forge_emit: rfl with VGPR dst: verify = {:?}", f2.verify().err().map(|e| e.join("; ")));
    println!("  block_to_asm = {:?}", block_to_asm(&f2).map(|a| a.lines().filter(|l| l.contains("v_readfirstlane")).map(|l| l.trim().to_string()).collect::<Vec<_>>()).map_err(|e| e.chars().take(160).collect::<String>()));
    println!("  block_to_mfunction = {:?}", block_to_mfunction(&f2, "forged").map(|m| m.blocks.iter().map(|b| b.insts.len()).sum::<usize>()).map_err(|e| e.chars().take(160).collect::<String>()));
    // 3) negative SNop: does the verifier panic (debug overflow) / wrap (release)?
    let mut i3 = insts.clone();
    i3.push(AInst { op: "SNop", dst: None, srcs: vec![], imm: -1 });
    let f3 = forge_for_tests(&b, i3);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f3.verify()));
    match r {
        Ok(v) => println!("CASE pass_forge_emit: negative SNop: verify returned {:?}", v.err()),
        Err(_) => println!("CASE pass_forge_emit: ** verifier PANICKED on a forged negative SNop (debug overflow at the slot add) **"),
    }
}
