// (NEW, safe path) The `input` contract says the sequence must not WRITE an input port, but `input::<N>()` returns a normal
// owned Vgpr, so writing it type-checks, `finish()` verifies it (the verifier has no input-write rule), and the DSL lowering
// maps the port straight onto the caller's register: the caller's still-live value is silently clobbered by the asm block.
// This case builds the block, embeds it with `b.asm`, and prints the asm around the constant register.
use amdgpu_dsl::{DType, FnBuilder, Op, Opnd};
use amdgpu_safeasm::*;
fn input_clobber_block() -> AsmBlock {
    let mut s = Seq::new();
    let mut x = s.input::<1>();          // read-only contract... but it is an owned Vgpr
    let one = s.splat(1);
    s.v_add_u32(&mut x, &one, &one);     // writes the input port register (x = 2)
    let mut out = s.vgpr::<1>();
    s.v_add_u32(&mut out, &one, &one);   // out = 2
    s.output(out);
    s.finish().expect("verifier does not notice that an input port was written")
}
fn main() {
    let b = input_clobber_block();
    println!("CASE pass_input_write: finish() accepted a block that writes its input port");
    println!("  block insts: {:?}", b.insts().iter().map(|i| (i.op, i.dst, i.srcs.len())).collect::<Vec<_>>());
    // DSL: `v` is computed and used BEFORE and AFTER the asm block
    let mut f = FnBuilder::new("input_clobber", 64);
    f.arg("o", DType::Ptr, 0, 8);
    let o = f.let_(Op::KernargPtr { offset: 0 });
    let three = f.c_f32(3.0);
    let four = f.c_f32(4.0);
    let v = f.add_f32(three, four);      // v = 7.0 in a real register
    f.store_f32(o, 0, v);                // before asm
    let out = f.asm(b, &[Opnd::Val(v)]); // the block writes the register holding v
    f.store_f32(o, 4, v);                // after asm: must still be 7.0
    let func = f.finish();
    let errs = amdgpu_dsl::verify(&func);
    println!("  DSL verify errors: {errs:?}");
    match amdgpu_codegen::compile(&func) {
        Ok(r) => {
            println!("  compiled OK; lines touching the input register (look for a write to the register that later stores v):");
            for l in r.asm.lines().filter(|l| l.contains("v_add_f32") || l.contains("v_add_u32") || l.contains("global_store") || l.contains("v_mov")) {
                println!("    {}", l.trim());
            }
            println!("=== KERNEL_ASM ===");
            println!("{}", r.asm);
        }
        Err(e) => println!("  compile Err: {e}"),
    }
    let _ = out;
}
