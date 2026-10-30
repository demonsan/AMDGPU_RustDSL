// (F8 via the bridge) An in/out block (rowsum_var_block has a mutable leading port) must not be embeddable through `Op::Asm`;
// only `Stmt::VarAsm` may bind a mutable port. Check both DSL verify and lowering.
use amdgpu_compiler::compile;
use amdgpu_dsl::{DType, FnBuilder, Op, Opnd};
use amdgpu_safeasm::examples::rowsum_var_block;
use amdgpu_safeasm::*;
fn main() {
    let blk = rowsum_var_block();
    println!("CASE pass_inout_via_asm: block ports = {:?}", blk.inputs().iter().map(|p| (p.class, p.mutable)).collect::<Vec<_>>());
    let mut f = FnBuilder::new("inout_asm", 64);
    f.arg("o", DType::Ptr, 0, 8);
    let o = f.let_(Op::KernargPtr { offset: 0 });
    let c0 = f.c_i32(0);
    let (c3, c4) = (f.c_i32(3), f.c_i32(4));
    let a_rs = f.frag4(c0, c0, c0, c0);
    let pa = f.frag8(c0, c0, c0, c0, c0, c0, c0, c0);
    let pb = f.frag8(c0, c0, c0, c0, c0, c0, c0, c0);
    let _r = f.asm(blk, &[Opnd::Val(a_rs), Opnd::Val(pa), Opnd::Val(pb), Opnd::Val(c3), Opnd::Val(c4)]);
    f.store_f32(o, 0, c0);
    let func = f.finish();
    println!("  DSL verify = {:?}", amdgpu_dsl::verify(&func));
    println!("  compile    = {:?}", compile(&func).map(|c| format!("Ok({} vgpr)", c.vgpr)).map_err(|e| e.chars().take(240).collect::<String>()));
}
