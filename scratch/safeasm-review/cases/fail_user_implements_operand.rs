// (l2) sealed operand trait: an external operand kind cannot be invented.
use amdgpu_safeasm::*;
struct Mine;
impl Operand for Mine { const BUS: u8 = 0; const VOP3_OK: bool = true; fn to_src(&self) -> amdgpu_codegen::mir::Src { amdgpu_codegen::mir::Src::Imm(0) } }
fn main() { let _ = Mine; }
