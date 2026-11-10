//! gfx950 per-instruction tables: MOp attributes (`op_attrs`), assembly emission (`emit`) and the LLVM MIR tables (`llvm/`).
//! The plain target data is `amdgpu_target::gfx950`.

pub mod emit;
pub mod op_attrs;
