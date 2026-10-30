// (NEW, safe path) `output` does not check that the register was ever written: a verified block can export an undefined register.
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let x = s.vgpr::<4>();               // contents undefined, never written
    s.output(x);
    match s.finish() {
        Ok(b) => println!("CASE pass_output_undefined: finish() accepted a block that exports a never-written register ({} insts)", b.insts().len()),
        Err(e) => println!("CASE pass_output_undefined: refused: {e}"),
    }
}
