// (F8 positive) Reading an input port is fine and must not clobber the caller's value: kernel stores v (7.0) before and
// after embedding a block that reads v as an input. Hardware check in the job (expect 7.0 both).
use amdgpu_codegen::compile;
use amdgpu_dsl::{DType, FnBuilder, Op, Opnd};
use amdgpu_safeasm::*;
fn reader_block() -> AsmBlock {
    let mut s = Seq::new();
    let x = s.input::<1>();
    let mut out = s.vgpr::<1>();
    s.v_add_u32(&mut out, &x, &x);
    s.output(out);
    s.finish().expect("read-only input block")
}
fn main() {
    println!("CASE pass_input_after_block:");
    let mut f = FnBuilder::new("input_after_block", 64);
    f.arg("o", DType::Ptr, 0, 8);
    let o = f.let_(Op::KernargPtr { offset: 0 });
    let three = f.c_f32(3.0);
    let four = f.c_f32(4.0);
    let v = f.add_f32(three, four);       // 7.0, a real register
    f.store_f32(o, 0, v);                 // before
    let _r = f.asm(reader_block(), &[Opnd::Val(v)]);
    f.store_f32(o, 4, v);                 // after: must still be 7.0
    let func = f.finish();
    println!("  DSL verify = {:?}", amdgpu_dsl::verify(&func));
    match compile(&func) {
        Ok(c) => {
            println!("  compiled OK");
            for l in c.asm.lines().filter(|l| l.contains("global_store") || l.contains("v_add_u32")) {
                println!("    {}", l.trim());
            }
            println!("=== KERNEL_ASM ===");
            println!("{}", c.asm);
        }
        Err(e) => println!("  compile Err: {e}"),
    }
}
