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
    &["attention_fwd", "attn_debug", "probe_store", "probe_math", "probe_mfma", "probe_lds", "probe_mfma_wide", "probe_mfma_fp8", "probe_mfma_scale", "probe_roles", "probe_waves2", "probe_lds_self2", "probe_roles_loop", "probe_fmha_datapath", "probe_ab_loops", "probe_nested_acc", "probe_loop_mfma", "probe_fmha_core", "probe_pack", "probe_fp8_matmul", "probe_fp8_load", "probe_fp8_min", "probe_fp8_minload", "probe_fp8_softmax", "probe_fp8_fmha_tile", "probe_fp8_fmha_2tile", "probe_fp8_fmha_online", "probe_fp8_fmha_oreg", "probe_fp8_fmha_multiwave", "probe_fp8_fmha_pingpong", "probe_fp8_fmha_abloops", "probe_lds_tr_b8"]
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
        "probe_lds_self2" => Some(amdgpu_dsl::kernels::probe_lds_self2()),
        "probe_roles_loop" => Some(amdgpu_dsl::kernels::probe_roles_loop()),
        "probe_fmha_datapath" => Some(amdgpu_dsl::kernels::probe_fmha_datapath()),
        "probe_ab_loops" => Some(amdgpu_dsl::kernels::probe_ab_loops()),
        "probe_nested_acc" => Some(amdgpu_dsl::kernels::probe_nested_acc()),
        "probe_loop_mfma" => Some(amdgpu_dsl::kernels::probe_loop_mfma()),
        "probe_fmha_core" => Some(amdgpu_dsl::kernels::probe_fmha_core()),
        "probe_pack" => Some(amdgpu_dsl::kernels::probe_pack()),
        "probe_fp8_matmul" => Some(amdgpu_dsl::kernels::probe_fp8_matmul()),
        "probe_fp8_load" => Some(amdgpu_dsl::kernels::probe_fp8_load()),
        "probe_fp8_min" => Some(amdgpu_dsl::kernels::probe_fp8_min()),
        "probe_fp8_minload" => Some(amdgpu_dsl::kernels::probe_fp8_minload()),
        "probe_fp8_softmax" => Some(amdgpu_dsl::kernels::probe_fp8_softmax()),
        "probe_fp8_fmha_tile" => Some(amdgpu_dsl::kernels::probe_fp8_fmha_tile()),
        "probe_fp8_fmha_2tile" => Some(amdgpu_dsl::kernels::probe_fp8_fmha_2tile()),
        "probe_fp8_fmha_online" => Some(amdgpu_dsl::kernels::probe_fp8_fmha_online()),
        "probe_fp8_fmha_oreg" => Some(amdgpu_dsl::kernels::probe_fp8_fmha_oreg()),
        "probe_fp8_fmha_multiwave" => Some(amdgpu_dsl::kernels::probe_fp8_fmha_multiwave()),
        "probe_fp8_fmha_pingpong" => Some(amdgpu_dsl::kernels::probe_fp8_fmha_pingpong()),
        "probe_fp8_fmha_abloops" => Some(amdgpu_dsl::kernels::probe_fp8_fmha_abloops()),
        "probe_lds_tr_b8" => Some(amdgpu_dsl::kernels::probe_lds_tr_b8()),
        _ => None,
    }
}