//! Backend for the AMDGPU Rust DSL: lowering, register allocation, emission.

pub mod backend;
pub mod emit;
pub mod hazards;
pub mod lower;
pub mod mir;
pub mod regalloc;
pub mod sched;

pub struct Compiled {
    pub asm: String,
    pub vgpr: u32,
    pub sgpr: u32,
}

/// Lowers, verifies, allocates and emits a kernel as gfx950 assembly text with our own tail.
pub fn compile(f: &amdgpu_dsl::Function) -> Result<Compiled, String> {
    compile_with(f, &backend::OwnTail)
}

/// Same pipeline, with a selectable tail backend (OwnTail or LlvmTail).
pub fn compile_with(f: &amdgpu_dsl::Function, tail: &dyn backend::TailBackend) -> Result<Compiled, String> {
    let (m, alloc) = lower_and_allocate(f)?;
    let asm = tail.finish(&m, &alloc)?;
    Ok(Compiled { asm, vgpr: alloc.max_v, sgpr: alloc.max_s })
}

/// The final (allocated, scheduled, hazard-padded) MIR of a kernel, for model reports.
pub fn compile_mir(f: &amdgpu_dsl::Function) -> Result<mir::MFunction, String> {
    lower_and_allocate(f).map(|(m, _)| m)
}

/// lower -> [schedule] -> hazards -> verify -> register allocation. With the scheduler enabled
/// (`sched::Options::from_env`) a result that does not fit the register file is retried with a
/// lower pressure limit and finally with the unscheduled code.
fn lower_and_allocate(f: &amdgpu_dsl::Function) -> Result<(mir::MFunction, regalloc::Alloc), String> {
    let base = lower::lower(f)?;
    let sched_opts = match sched::Options::from_env() {
        Some(o) if o.auto => sched::auto_plan(&base, &o),
        other => other,
    };
    if let Some(opts) = sched_opts {
        let mut last_err = String::new();
        for cut in [0u32, 16, 32, 48] {
            let mut o = opts.clone();
            o.max_vgprs = opts.max_vgprs.saturating_sub(cut);
            let mut m = base.clone();
            let attempt = (|| -> Result<regalloc::Alloc, String> {
                sched::run(&mut m, &o)?;
                hazards::fix_value_hazards_opts(&mut m, true);
                let errs = mir::verify(&m);
                if !errs.is_empty() {
                    return Err(format!("MIR verification failed:\n  {}", errs.join("\n  ")));
                }
                let a = regalloc::allocate(&m)?;
                hazards::pad_physical_war(&mut m, &a);
                Ok(a)
            })();
            match attempt {
                Ok(a) => return Ok((m, a)),
                Err(e) if e.contains("out of VGPRs") => last_err = e,
                Err(e) => return Err(e),
            }
            if !opts.reorder {
                break;
            }
        }
        eprintln!("sched: falling back to the unscheduled code ({})", last_err.chars().take(120).collect::<String>());
    }
    let mut m = base;
    hazards::fix_value_hazards(&mut m);
    let errs = mir::verify(&m);
    if !errs.is_empty() {
        return Err(format!("MIR verification failed:\n  {}", errs.join("\n  ")));
    }
    let alloc = regalloc::allocate(&m)?;
    Ok((m, alloc))
}

/// Lower, schedule and allocate, then export post-RA MIR (no llc involved).
pub fn export_mir(f: &amdgpu_dsl::Function, tail: &backend::LlvmTail) -> Result<String, String> {
    let (m, alloc) = lower_and_allocate(f)?;
    tail.export_mir(&m, &alloc)
}

/// All kernel definitions currently shipped with the DSL.
pub fn kernel_names() -> &'static [&'static str] {
    &["attention_fwd", "attn_debug", "probe_store", "probe_math", "probe_mfma", "probe_lds", "probe_mfma_wide", "probe_mfma_fp8", "probe_mfma_scale", "probe_roles", "probe_waves2", "probe_lds_self2", "probe_roles_loop", "probe_fmha_datapath", "probe_ab_loops", "probe_nested_acc", "probe_loop_mfma", "probe_fmha_core", "probe_pack", "probe_fp8_matmul", "probe_fp8_load", "probe_fp8_min", "probe_fp8_minload", "probe_fp8_softmax", "probe_fp8_fmha_tile", "probe_fp8_fmha_2tile", "probe_fp8_fmha_online", "probe_fp8_fmha_oreg", "probe_fp8_fmha_multiwave", "probe_fp8_fmha_pingpong", "probe_fp8_fmha_abloops", "probe_lds_tr_b8", "probe_fp8_matmul_tr", "probe_fp8_fmha_tr", "probe_fp8_fmha_trab", "probe_fp8_fmha_hd128", "probe_fp8_fmha_hd128b", "probe_fp8_fmha_hd128c", "probe_fp8_matmul_blk", "fmha_fwd_hd128_mxfp8", "fmha_fwd_hd128_mxfp8_t", "fmha_fwd_hd128_mxfp8_pp", "fmha_fwd_hd128_mxfp8_sv", "probe_dma", "probe_dma_tile", "probe_dma_seq", "sched_valu_mul", "sched_valu_max3", "sched_valu_exp", "sched_mfma_chain", "sched_mfma_valu16", "sched_mfma_valu4", "sched_diamond"]
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
        "probe_fp8_matmul_tr" => Some(amdgpu_dsl::kernels::probe_fp8_matmul_tr()),
        "probe_fp8_fmha_tr" => Some(amdgpu_dsl::kernels::probe_fp8_fmha_tr()),
        "probe_fp8_fmha_trab" => Some(amdgpu_dsl::kernels::probe_fp8_fmha_trab()),
        "probe_fp8_fmha_hd128" => Some(amdgpu_dsl::kernels::probe_fp8_fmha_hd128()),
        "probe_fp8_fmha_hd128b" => Some(amdgpu_dsl::kernels::probe_fp8_fmha_hd128b()),
        "probe_fp8_fmha_hd128c" => Some(amdgpu_dsl::kernels::probe_fp8_fmha_hd128c()),
        "probe_fp8_matmul_blk" => Some(amdgpu_dsl::kernels::probe_fp8_matmul_blk()),
        "fmha_fwd_hd128_mxfp8" => Some(amdgpu_dsl::kernels::fmha_fwd_hd128_mxfp8()),
        "fmha_fwd_hd128_mxfp8_t" => Some(amdgpu_dsl::kernels::fmha_fwd_hd128_mxfp8_t()),
        "fmha_fwd_hd128_mxfp8_pp" => Some(amdgpu_dsl::kernels::fmha_fwd_hd128_mxfp8_pp()),
        "fmha_fwd_hd128_mxfp8_sv" => Some(amdgpu_dsl::kernels::fmha_fwd_hd128_mxfp8_sv()),
        "probe_dma" => Some(amdgpu_dsl::kernels::probe_dma()),
        "probe_dma_tile" => Some(amdgpu_dsl::kernels::probe_dma_tile()),
        "probe_dma_seq" => Some(amdgpu_dsl::kernels::probe_dma_seq()),
        "sched_valu_mul" => Some(amdgpu_dsl::kernels::sched_valu_mul()),
        "sched_valu_max3" => Some(amdgpu_dsl::kernels::sched_valu_max3()),
        "sched_valu_exp" => Some(amdgpu_dsl::kernels::sched_valu_exp()),
        "sched_mfma_chain" => Some(amdgpu_dsl::kernels::sched_mfma_chain()),
        "sched_mfma_valu16" => Some(amdgpu_dsl::kernels::sched_mfma_valu16()),
        "sched_mfma_valu4" => Some(amdgpu_dsl::kernels::sched_mfma_valu4()),
        "sched_diamond" => Some(amdgpu_dsl::kernels::sched_diamond()),
        _ => None,
    }
}