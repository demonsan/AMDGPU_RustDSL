// (F8 via the bridge, complete binding) Bind ALL ports of an in/out block through `Op::Asm` (the mutable leading port as a Frag4
// value): what does DSL verify say, and does the lowering reject the mutable port?
use amdgpu_compiler::compile;
use amdgpu_dsl::{DType, FnBuilder, Op, Opnd};
use amdgpu_safeasm::examples::rowsum_var_block;
use amdgpu_safeasm::*;
fn main() {
    let blk = rowsum_var_block();
    println!("CASE pass_inout_via_asm6: ports = {:?}", blk.inputs().iter().map(|p| (p.class, p.mutable)).collect::<Vec<_>>());
    let mut f = FnBuilder::new("inout_asm6", 64);
    f.arg("o", DType::Ptr, 0, 8);
    let o = f.let_(Op::KernargPtr { offset: 0 });
    let c0 = f.c_i32(0);
    let (c3, c4) = (f.c_i32(3), f.c_i32(4));
    let acc = f.frag_zero(DType::Frag4);
    let a_rs = f.frag4(c0, c0, c0, c0);
    let pa = f.frag8(c0, c0, c0, c0, c0, c0, c0, c0);
    let pb = f.frag8(c0, c0, c0, c0, c0, c0, c0, c0);
    let _r = f.asm(blk, &[Opnd::Val(acc), Opnd::Val(a_rs), Opnd::Val(pa), Opnd::Val(pb), Opnd::Val(c3), Opnd::Val(c4)]);
    f.store_f32(o, 0, c0);
    let func = f.finish();
    println!("  DSL verify = {:?}", amdgpu_dsl::verify(&func));
    println!("  compile    = {:?}", compile(&func).map(|c| format!("Ok({} vgpr, {} sgpr)", c.vgpr, c.sgpr)).map_err(|e| e.chars().take(300).collect::<String>()));
}
