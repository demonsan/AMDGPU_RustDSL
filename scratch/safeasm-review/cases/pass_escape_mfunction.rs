// ATTACK (l): the compiler crate is a public dependency and `to_mfunction` hands out a mutable MFunction,
// so all type guarantees can be bypassed by pushing raw MIR (raw MOp escape hatch).
use amdgpu_compiler::mir::{MFunction, MInst, MOp, RClass};
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let mut x = s.splat(1);
    let mut m: MFunction = s.to_mfunction("escape");
    // append a raw VALU op that was never checked by any safeasm type/borrow rule
    let d = m.vregs.len() as u32;
    m.vregs.push(RClass::V1);
    let a = m.vregs.len() as u32;
    m.vregs.push(RClass::V1);
    m.blocks[0].insts.push(MInst::new(MOp::VAddU32).def(d).srcreg(a).srcreg(a));
    let alloc = amdgpu_compiler::regalloc::allocate(&m).unwrap();
    let asm = amdgpu_compiler::emit::emit(&m, &alloc);
    let _ = &mut x;
    println!("CASE pass_escape_mfunction: compiled; raw MIR injected after the safe sequence; asm:");
    for l in asm.lines().filter(|l| l.contains("v_add") || l.contains("v_mov")) {
        println!("  {l}");
    }
}
