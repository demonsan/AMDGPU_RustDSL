// (F12) `AsmBlock` must not be constructible outside the crate.
use amdgpu_safeasm::*;
fn main() {
    let b = AsmBlock { name: "x".into(), regs: vec![], alias: vec![], insts: vec![], inputs: vec![], outputs: vec![], exit_in_flight: vec![] };
    let _ = b;
}
