//! Backend for the AMDGPU Rust DSL: lowering, register allocation, emission.

pub mod emit;
pub mod lower;
pub mod mir;
pub mod regalloc;

pub struct Compiled {
    pub asm: String,
    pub vgpr: u32,
    pub sgpr: u32,
}

/// Lowers, verifies, allocates and emits a kernel as gfx950 assembly text.
pub fn compile(f: &amdgpu_dsl::Function) -> Result<Compiled, String> {
    let m = lower::lower(f)?;
    let errs = mir::verify(&m);
    if !errs.is_empty() {
        return Err(format!("MIR verification failed:\n  {}", errs.join("\n  ")));
    }
    let alloc = regalloc::allocate(&m)?;
    let asm = emit::emit(&m, &alloc);
    Ok(Compiled { asm, vgpr: alloc.max_v, sgpr: alloc.max_s })
}

/// All kernel definitions currently shipped with the DSL.
pub fn kernel_names() -> &'static [&'static str] {
    &["attention_fwd", "probe_store"]
}

pub fn kernel_by_name(name: &str) -> Option<amdgpu_dsl::Function> {
    match name {
        "attention_fwd" => Some(amdgpu_dsl::kernels::attention_fwd()),
        "probe_store" => Some(amdgpu_dsl::kernels::probe_store()),
        _ => None,
    }
}
