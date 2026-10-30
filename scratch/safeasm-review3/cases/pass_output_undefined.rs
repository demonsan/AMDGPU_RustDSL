// (F9) An output that is never written must be refused by finish().
use amdgpu_safeasm::*;
fn main() {
    let mut s = Seq::new();
    let x = s.vgpr::<4>();
    s.output(x);
    println!("CASE pass_output_undefined: finish = {:?}", s.finish().err());
}
