// error-contains: Sealed
use amdgpu_safeasm::*;
struct Mine;
impl Operand for Mine {
    const BUS: u8 = 0;
    const VOP3_OK: bool = true;
    fn to_src(&self) -> ASrc { ASrc::Imm(0) }
}
fn main() {}