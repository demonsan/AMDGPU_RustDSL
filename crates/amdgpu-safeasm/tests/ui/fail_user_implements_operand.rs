// error-contains: Sealed
use amdgpu_safeasm::*;
struct Mine;
impl Operand for Mine {
    const BUS: u8 = 0;
    const VOP3_OK: bool = true;
    fn to_src(&self) -> amdgpu_compiler::mir::Src { unimplemented!() }
}
fn main() {}