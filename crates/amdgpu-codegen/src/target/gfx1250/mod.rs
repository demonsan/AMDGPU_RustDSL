//! gfx1250 (MI450, CDNA5): the per-instruction tables and the pipeline of the gfx1250 target, built in stages (docs/gfx1250-plan.md):
//! G0 the target plumbing (`--target gfx1250`), G1 the wave32 MIR ops, encoding and both emitters, G2 waits / hazards, G3 TDM / async LDS /
//! split barriers, G5 the hang-safety verifier.

use amdgpu_dsl::Function;

/// Compiles `f` for gfx1250. G0: the target is selectable; code generation arrives with G1.
pub fn compile(f: &Function) -> Result<crate::Compiled, String> {
    Err(format!(
        "{}: gfx1250 code generation is not available yet (stage G1 of docs/gfx1250-plan.md); the target description is amdgpu_target::gfx1250::GFX1250",
        f.name
    ))
}
