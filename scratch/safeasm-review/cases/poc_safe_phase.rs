// POC part 1: the safe-DSL PV + row-sum phase, emitted as assembly (own backend, register-allocated).
use amdgpu_safeasm::examples::pv_rowsum_phase;
use amdgpu_safeasm::Seq;
fn main() {
    let mut s = Seq::new();
    pv_rowsum_phase(&mut s);
    println!("=== SAFE_ASM ===");
    println!("{}", s.to_asm().unwrap());
}
