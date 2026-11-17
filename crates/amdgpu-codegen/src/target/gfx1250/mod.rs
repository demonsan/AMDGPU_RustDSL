//! gfx1250 (MI450, CDNA5): the per-instruction tables and the pipeline of the gfx1250 target, built in stages (docs/gfx1250-plan.md):
//! G0 the target plumbing (`--target gfx1250`), G1 the wave32 MIR ops, encoding and both emitters, G2 waits / hazards, G3 TDM / async LDS /
//! split barriers, G5 the hang-safety verifier.

use amdgpu_dsl::Function;

pub mod emit;

/// Compiles `f` for gfx1250 (G1: the shared lowering / scheduling / allocation, then the gfx1250 emitter; wave32 kernels without
/// divergent control flow, v0..v255, no AGPRs; the waits are the gfx950 ones mapped conservatively onto the split counters).
pub fn compile(f: &Function) -> Result<crate::Compiled, String> {
    if f.workgroup_size > 1024 {
        return Err(format!("{}: workgroup size {} (gfx1250: at most 1024 work-items)", f.name, f.workgroup_size));
    }
    let (m, alloc) = crate::lower_and_allocate(f)?;
    let asm = emit::emit(&m, &alloc)?;
    Ok(crate::Compiled { asm, vgpr: alloc.max_v, sgpr: alloc.max_s })
}
