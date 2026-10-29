// error-contains: inline constants are -16..=64
use amdgpu_safeasm::*;
fn main() {
    let _i = Imm::<100>::new();
}