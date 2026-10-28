//! Backend for the AMDGPU Rust DSL: lowering, register allocation, emission.

pub mod emit;
pub mod hazards;
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
    let mut m = lower::lower(f)?;
    hazards::fix_value_hazards(&mut m);
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
    &["attention_fwd", "attn_debug", "probe_store", "probe_math", "probe_mfma", "probe_lds", "probe_mfma_wide", "probe_mfma_fp8", "probe_mfma_scale", "probe_roles", "probe_waves2", "probe_lds_own", "probe_lds_self2"]
}

pub fn kernel_by_name(name: &str) -> Option<amdgpu_dsl::Function> {
    match name {
        "attention_fwd" => Some(amdgpu_dsl::kernels::attention_fwd()),
        "attn_debug" => Some(amdgpu_dsl::kernels::attn_debug()),
        "probe_store" => Some(amdgpu_dsl::kernels::probe_store()),
        "probe_math" => Some(amdgpu_dsl::kernels::probe_math()),
        "probe_mfma" => Some(amdgpu_dsl::kernels::probe_mfma()),
        "probe_lds" => Some(amdgpu_dsl::kernels::probe_lds()),
        "probe_mfma_wide" => Some(amdgpu_dsl::kernels::probe_mfma_wide()),
        "probe_mfma_fp8" => Some(amdgpu_dsl::kernels::probe_mfma_fp8()),
        "probe_mfma_scale" => Some(amdgpu_dsl::kernels::probe_mfma_scale()),
        "probe_roles" => Some(amdgpu_dsl::kernels::probe_roles()),
        "probe_waves2" => Some(amdgpu_dsl::kernels::probe_waves2()),
        "probe_lds_own" => Some(amdgpu_dsl::kernels::probe_lds_own()),
        "probe_lds_self2" => Some(amdgpu_dsl::kernels::probe_lds_self2()),
        _ => None,
    }
}