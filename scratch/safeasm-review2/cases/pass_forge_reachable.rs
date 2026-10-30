// (4a) `forge_for_tests` is `pub` (only #[doc(hidden)]) in a normal build: the verifier's reachability question.
// Prints whether an external, non-test crate can forge an AsmBlock and what the verifier says about it.
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
    // reachable in a plain binary (this file is compiled exactly like a user crate, no cfg(test))
    let forged = forge_for_tests(&b, b.insts().to_vec());
    println!("CASE pass_forge_reachable: forge_for_tests is callable from a normal (non-test) build: block `{}` with {} insts", forged.name(), forged.insts().len());
    // 1) forged sequence WITHOUT the 20-state retire padding: verifier must reject it
    let acc = b.insts().iter().find(|i| i.op == "VMfmaScale32x32x64Fp8Z").unwrap().dst.unwrap();
    let mut no_pad: Vec<AInst> = b.insts().iter().filter(|i| i.op != "SNop").cloned().collect();
    no_pad.push(AInst { op: "VAddF32", dst: Some(0), srcs: vec![ASrc::Reg(acc), ASrc::Reg(acc)], imm: 0 });
    let v = forge_for_tests(&b, no_pad).verify();
    println!("  forged (no retire pad, immediate reader): verify = {:?}", v.err().map(|e| e.join("; ")));
    // 2) forged encoding-illegal instruction that the hazard model does not look at: 3 sources on a 2-source VALU op,
    //    and a matrix op with a 1-dword destination. Does verify() accept them?
    let v16 = b.regs().iter().position(|c| *c == AClass::V16).unwrap() as u32;
    let v1 = b.regs().iter().position(|c| *c == AClass::V1).unwrap() as u32;
    let mut illegal_encoding = b.insts().to_vec();
    illegal_encoding.push(AInst { op: "VAddF32", dst: Some(v1), srcs: vec![ASrc::Reg(v1), ASrc::Reg(v1), ASrc::Reg(v1)], imm: 0 });
    illegal_encoding.push(AInst { op: "VReadFirstLane", dst: Some(v1), srcs: vec![ASrc::Reg(v1)], imm: 0 });
    illegal_encoding.push(AInst { op: "VMfmaScale32x32x64Fp8Z", dst: Some(v1), srcs: vec![ASrc::Reg(v1), ASrc::Reg(v1), ASrc::Reg(v1), ASrc::Reg(v1)], imm: 0 });
    let f2 = forge_for_tests(&b, illegal_encoding);
    println!("  forged (encoding-illegal ops, hazard-clean): verify = {:?}", f2.verify().err().map(|e| e.join("; ")));
    let _ = v16;
}
