//! Backend for the AMDGPU Rust DSL: lowering, register allocation, emission.

pub mod asm;
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

/// The final MIR and register assignment of a kernel (what the tail backends receive).
pub fn compile_mir_alloc(f: &amdgpu_dsl::Function) -> Result<(mir::MFunction, regalloc::Alloc), String> {
    lower_and_allocate(f)
}

/// The final (allocated, scheduled, hazard-padded) MIR of a kernel, for model reports.
pub fn compile_mir(f: &amdgpu_dsl::Function) -> Result<mir::MFunction, String> {
    lower_and_allocate(f).map(|(m, _)| m)
}

/// lower -> [schedule] -> hazards -> verify -> register allocation. With the scheduler enabled
/// (`sched::Options::from_env`) a result that does not fit the register file is retried with a
/// lower pressure limit and finally with the unscheduled code.
fn lower_and_allocate(f: &amdgpu_dsl::Function) -> Result<(mir::MFunction, regalloc::Alloc), String> {
    // Register pressure: when allocation fails, rematerialise loop-invariant constants next to their loop uses (remat1 -> remat3)
    // and try again. A kernel that allocates at the first attempt is unaffected.
    let mut last = String::new();
    for extra in [&[][..], &["remat1"][..], &["remat2"][..], &["remat3"][..], &["remat4"][..], &["imm", "ldsspill8"][..], &["imm", "ldsspill16"][..], &["imm", "ldsspill24"][..], &["imm", "ldsspill32"][..], &["imm", "ldsspill48"][..], &["imm", "ldsspill64"][..]] {
        match lower_and_allocate_with(f, extra) {
            Ok(r) => {
                if r.0.lds_bytes > 160 * 1024 {
                    return Err(format!("LDS spill needs {} bytes of LDS (limit 163840)", r.0.lds_bytes));
                }
                return Ok(r);
            }
            Err(e) if e.contains("out of VGPRs") => last = e,
            Err(e) => return Err(e),
        }
    }
    Err(last)
}

fn lower_and_allocate_with(f: &amdgpu_dsl::Function, extra: &[&str]) -> Result<(mir::MFunction, regalloc::Alloc), String> {
    // The vendor-structure kernels need the register savings of `scalarize` (wave-uniform integer chains in SGPRs) to fit
    // (svscale / the 3-slot swizzled ring); `sgpr` and `addrfold` miscompile them (wrong results, see docs/status.md round 47).
    let base = if (f.name.starts_with("fmha_fwd_hd128_mxfp8_vs") || f.name.ends_with("_kvsplit") || (f.name.ends_with("_g2") || f.name.ends_with("_g3") || f.name.ends_with("_g3o") || f.name.ends_with("_t128"))) && std::env::var("AMDC_LOWER_OPTS").map_or(true, |v| v.is_empty()) {
        // measured on vs128 (15/15, 301.9-304.0 vs 280.6 TFLOP/s): retarget2 (cvt results written straight into the MFMA fragment
        // lanes), imm and sgpr cut ~230 copies/VALU and free 12 VGPRs; addrfold and shl are slower and stay off. `imm` also gives
        // the vs128/vsm builds the register slack they need (docs/status.md rounds 49-50).
        let mut v: Vec<&str> = vec!["dce", "cmpfold", "scalarize", "retarget2", "imm", "sgpr", "zeroinit"];
        // `_t128` (4-wave blocks, one wave per SIMD): addrfold frees the DMA address temporaries so the full scheduler fits (416 -> 494 TFLOP/s)
        if f.name.ends_with("_t128") {
            v.push("addrfold");
        }
        v.extend_from_slice(extra);
        lower::with_opts(&v, || lower::lower(f))?
    } else if f.name.starts_with("fmha_fwd_hd128_mxfp8") && std::env::var("AMDC_LOWER_OPTS").map_or(true, |v| v.is_empty()) {
        // the FMHA kernels zero their accumulators in place (bit-exact on all 15 shapes, -160 v_mov on _t, docs/status.md)
        let mut v: Vec<&str> = vec!["zeroinit"];
        v.extend_from_slice(extra);
        lower::lower_with_extra(f, &v)?
    } else {
        lower::lower_with_extra(f, extra)?
    };
    let sched_opts = match sched::Options::from_env() {
        Some(o) if o.auto => sched::auto_plan(&base, &o),
        other => other,
    };
    // The FMHA kernels default to the wait-recomputing pass (partial vmcnt/lgkmcnt waits, ALU across barriers, ds_read hoisting): +9% on the
    // transposed kernel, bit-identical results (measured, docs/status.md round 44). AMDC_SCHED=0 turns it off.
    let sched_opts = match sched_opts {
        None if std::env::var("AMDC_SCHED").ok().as_deref() != Some("0")
            && (f.name.starts_with("fmha_fwd_hd128_mxfp8_t") || f.name.starts_with("fmha_fwd_hd128_mxfp8_sv") || f.name.starts_with("fmha_fwd_hd128_mxfp8_vs") || (f.name.ends_with("_g2") || f.name.ends_with("_g3") || f.name.ends_with("_g3o") || f.name.ends_with("_t128"))) =>
        {
            // "ld" preset: scoreboard (partial) waits + ds_read hoisting, no instruction reordering. Measured best on the
            // merged tree (docs/status.md round 45): 286.5 TFLOP/s vs 285.1 with rewait,xbar; bit-identical results.
            let mut o = sched::Options::default();
            // `_t128`: ALU may cross the single barrier per tile and the pressure limit 248 lets the list scheduler take a schedule at vgpr 256
            // (measured 489 -> 505 TFLOP/s on B1H8S2048; limit 246/248 only, 244/250 give the old schedule: a heuristic knob, re-measure on changes)
            if f.name.ends_with("_t128") {
                o.alu_across_barrier = true;
                o.max_vgprs = 248;
            }
            // `_vs128` keeps the full list scheduler (reordering): 299.1 vs 287.9 TFLOP/s with the `ld` preset (docs/status.md round 49)
            if !f.name.ends_with("_vs128") && !f.name.ends_with("_kvsplit") && !f.name.ends_with("_vsm") && !f.name.ends_with("_t128") && !(f.name.ends_with("_g2") || f.name.ends_with("_g3") || f.name.ends_with("_g3o")) {
                o.reorder = false;
                o.alu_across_barrier = true;
                o.ldshoist = true; // hoist ds_read above independent work (docs/scheduling.md); AMDC_SCHED=0 / 1 / auto / OPTS override
            }
            Some(o)
        }
        other => other,
    };
    if let Some(opts) = sched_opts {
        let mut last_err = String::new();
        // (pressure cut, hoist distance cap): first shrink the LDS-read hoisting (it lengthens live ranges and fragments
        // the register file), then the pressure limit, and only then give up the scheduler.
        let mut attempts: Vec<(u32, Option<usize>)> = vec![(0, None)];
        if opts.ldshoist {
            attempts.extend([(0, Some(24)), (0, Some(8)), (0, Some(0))]);
        }
        if opts.reorder {
            attempts.extend([(16, Some(0)), (32, Some(0)), (48, Some(0))]); // the pressure limit only matters for the reordering scheduler
        }
        for (cut, hoist) in attempts {
            let mut o = opts.clone();
            o.max_vgprs = opts.max_vgprs.saturating_sub(cut);
            if let Some(h) = hoist {
                o.hoist_dist = h;
                o.ldshoist = h > 0 && opts.ldshoist;
            }
            let mut m = base.clone();
            let attempt = (|| -> Result<regalloc::Alloc, String> {
                sched::run(&mut m, &o)?;
                let vcc_fill = f.name.starts_with("fmha_fwd_hd128_mxfp8_vs") && std::env::var("AMDC_VCC_FILL").map_or(false, |v| !v.is_empty() && v != "0");
                hazards::with_vcc_fill(vcc_fill, || hazards::fix_value_hazards_opts(&mut m, true));
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
        }
        eprintln!("sched: falling back to the unscheduled code ({})", last_err.chars().take(if std::env::var("AMDC_SCHED_VERBOSE").is_ok() { 6000 } else { 120 }).collect::<String>());
    }
    let mut m = base;
    hazards::fix_value_hazards(&mut m);
    let errs = mir::verify(&m);
    if !errs.is_empty() {
        return Err(format!("MIR verification failed:\n  {}", errs.join("\n  ")));
    }
    let alloc = regalloc::allocate(&m)?;
    // Register reuse (especially the hole-filling packing) can place a new value in a register an in-flight MFMA still reads:
    // pad the physical write-after-read hazard on every path, not only for scheduled code (AMDC_NO_PHYS_WAR=1 disables, testing).
    if std::env::var("AMDC_NO_PHYS_WAR").map(|v| v.is_empty() || v == "0").unwrap_or(true) {
        hazards::pad_physical_war(&mut m, &alloc);
    }
    Ok((m, alloc))
}

/// Lower, schedule and allocate, then export post-RA MIR (no llc involved).
pub fn export_mir(f: &amdgpu_dsl::Function, tail: &backend::LlvmTail) -> Result<String, String> {
    let (m, alloc) = lower_and_allocate(f)?;
    tail.export_mir(&m, &alloc)
}

/// Does the kernel contain own-backend-only MOps (the LDS-DMA loads)? The LLVM tail cannot express them.
pub fn uses_own_only_ops(f: &amdgpu_dsl::Function) -> bool {
    match lower::lower_with_extra(f, &[]) {
        Ok(m) => m.blocks.iter().any(|b| b.insts.iter().any(|i| matches!(i.op, mir::MOp::GlobalLoadLdsB128 | mir::MOp::GlobalLoadLdsB32))),
        Err(_) => false,
    }
}

/// Backend policy: uto (the CLI default) is llvm-tail unless the kernel uses own-backend-only ops, then own;
/// explicit own / llvm-tail are returned unchanged. The second value is a note to print when auto switched.
pub fn resolve_backend(name: &str, f: &amdgpu_dsl::Function) -> (String, Option<String>) {
    if name != "auto" {
        return (name.to_string(), None);
    }
    if uses_own_only_ops(f) {
        ("own".to_string(), Some(format!("kernel {} uses own-backend-only ops (DMA); emitting with --backend=own", f.name)))
    } else {
        ("llvm-tail".to_string(), None)
    }
}

/// All kernel definitions currently shipped with the DSL.
pub fn kernel_names() -> &'static [&'static str] {
    &["attention_fwd", "attn_debug", "probe_store", "probe_math", "probe_mfma", "probe_lds", "probe_mfma_wide", "probe_mfma_fp8", "probe_mfma_scale", "probe_roles", "probe_waves2", "probe_lds_self2", "probe_roles_loop", "probe_fmha_datapath", "probe_ab_loops", "probe_nested_acc", "probe_loop_mfma", "probe_fmha_core", "probe_pack", "probe_fp8_matmul", "probe_fp8_load", "probe_fp8_min", "probe_fp8_minload", "probe_fp8_softmax", "probe_fp8_fmha_tile", "probe_fp8_fmha_2tile", "probe_fp8_fmha_online", "probe_fp8_fmha_oreg", "probe_fp8_fmha_multiwave", "probe_fp8_fmha_pingpong", "probe_fp8_fmha_abloops", "probe_lds_tr_b8", "probe_fp8_matmul_tr", "probe_fp8_fmha_tr", "probe_fp8_fmha_trab", "probe_fp8_fmha_hd128", "probe_fp8_fmha_hd128b", "probe_fp8_fmha_hd128c", "probe_fp8_matmul_blk", "fmha_fwd_hd128_mxfp8", "fmha_fwd_hd128_mxfp8_t", "fmha_fwd_hd128_mxfp8_pp", "fmha_fwd_hd128_mxfp8_sv", "sched_ovl_both", "sched_ovl_mfma", "sched_ovl_valu", "sched_ovl_vfirst", "sched_coissue_both", "sched_coissue_mfma", "sched_coissue_valu", "sched_coissue_cboth", "sched_coissue_cvalu", "sched_valu_mul", "sched_valu_max3", "sched_valu_exp", "sched_mfma_chain", "sched_mfma_valu16", "sched_mfma_valu4", "sched_diamond", "sched_occ_chain", "probe_log_store4", "fmha_fwd_hd128_mxfp8_kvsplit_combine", "fmha_fwd_hd128_mxfp8_kvsplit_combine4"]
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
        "fmha_fwd_hd128_mxfp8_vs" => Some(amdgpu_dsl::kernels::fmha_fwd_hd128_mxfp8_vs()),
        "fmha_fwd_hd128_mxfp8_kvsplit" => Some(amdgpu_dsl::kernels::fmha_fwd_hd128_mxfp8_kvsplit()),
        "fmha_fwd_hd128_mxfp8_kvsplit_combine4h" => Some(amdgpu_dsl::kernels::fmha_fwd_hd128_mxfp8_kvsplit_combine4h()),
        "fmha_fwd_hd128_mxfp8_kvsplit_combine4" => Some(amdgpu_dsl::kernels::fmha_fwd_hd128_mxfp8_kvsplit_combine4()),
        "fmha_fwd_hd128_mxfp8_kvsplit_combine" => Some(amdgpu_dsl::kernels::fmha_fwd_hd128_mxfp8_kvsplit_combine()),
        "fmha_fwd_hd128_mxfp8_vs128" => Some(amdgpu_dsl::kernels::fmha_fwd_hd128_mxfp8_vs128()),
        "fmha_fwd_hd128_mxfp8_vsm" => Some(amdgpu_dsl::kernels::fmha_fwd_hd128_mxfp8_vsm()),
        "probe_dma" => Some(amdgpu_dsl::kernels::probe_dma()),
        "probe_safeasm_e2e" => Some(amdgpu_dsl::kernels::probe_safeasm_e2e()),
        "probe_log_store4" => Some(amdgpu_dsl::kernels::probe_log_store4()),
        "probe_dma_tile" => Some(amdgpu_dsl::kernels::probe_dma_tile()),
        "probe_dma_seq" => Some(amdgpu_dsl::kernels::probe_dma_seq()),
        "probe_dma_kread" => Some(amdgpu_dsl::kernels::probe_dma_kread()),
        "probe_dma_map" => Some(amdgpu_dsl::kernels::probe_dma_map()),
        "probe_smfmac_ones" => Some(amdgpu_dsl::kernels::probe_smfmac_ones()),
        "sched_valu_mul" => Some(amdgpu_dsl::kernels::sched_valu_mul()),
        "sched_valu_max3" => Some(amdgpu_dsl::kernels::sched_valu_max3()),
        "sched_valu_exp" => Some(amdgpu_dsl::kernels::sched_valu_exp()),
        "sched_mfma_chain" => Some(amdgpu_dsl::kernels::sched_mfma_chain()),
        "sched_mfma_valu16" => Some(amdgpu_dsl::kernels::sched_mfma_valu16()),
        "sched_coissue_both" => Some(amdgpu_dsl::kernels::sched_coissue_both()),
        "fmha_fwd_hd128_mxfp8_t128" => Some(amdgpu_dsl::kernels::fmha_fwd_hd128_mxfp8_t128()),
        "fmha_fwd_hd128_mxfp8_g3o" => Some(amdgpu_dsl::kernels::fmha_fwd_hd128_mxfp8_g3o()),
        "fmha_fwd_hd128_mxfp8_g3" => Some(amdgpu_dsl::kernels::fmha_fwd_hd128_mxfp8_g3()),
        "fmha_fwd_hd128_mxfp8_g2" => Some(amdgpu_dsl::kernels::fmha_fwd_hd128_mxfp8_g2()),
        "sched_ovl_both" => Some(amdgpu_dsl::kernels::sched_ovl_both()),
        "sched_ovl_mfma" => Some(amdgpu_dsl::kernels::sched_ovl_mfma()),
        "sched_ovl_valu" => Some(amdgpu_dsl::kernels::sched_ovl_valu()),
        "sched_ovl_vfirst" => Some(amdgpu_dsl::kernels::sched_ovl_vfirst()),
        "sched_coissue_mfma" => Some(amdgpu_dsl::kernels::sched_coissue_mfma()),
        "sched_coissue_valu" => Some(amdgpu_dsl::kernels::sched_coissue_valu()),
        "sched_coissue_cboth" => Some(amdgpu_dsl::kernels::sched_coissue_cboth()),
        "sched_coissue_cvalu" => Some(amdgpu_dsl::kernels::sched_coissue_cvalu()),
        "sched_mfma_valu4" => Some(amdgpu_dsl::kernels::sched_mfma_valu4()),
        "sched_diamond" => Some(amdgpu_dsl::kernels::sched_diamond()),
        "sched_occ_chain" => Some(amdgpu_dsl::kernels::sched_occ_chain()),
        _ => None,
    }
}