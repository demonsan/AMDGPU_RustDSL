//! Backend for the AMDGPU Rust DSL: lowering, register allocation, emission.

#![forbid(unsafe_code)]

pub mod agpr;
pub mod asm;
pub mod asm_check;
pub mod pin_check;
pub mod backend;
pub mod emit;
pub mod gpu_protocol;
pub use gpu_protocol as gpu_tiles;
pub mod gpu_verify;
pub mod hostgen;
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

/// The name the compile policy (lowering options, scheduler preset) is derived from: the kernel's own name, unless the kernel declares an explicit
/// [`amdgpu_dsl::CompileProfile`], which maps to the name of the shipped kernel whose policy it requests (so a profile gets EXACTLY that kernel's policy).
pub fn policy_name(f: &amdgpu_dsl::Function) -> &str {
    use amdgpu_dsl::CompileProfile as P;
    match f.compile_profile {
        None => &f.name,
        Some(P::Vs128) => "fmha_fwd_hd128_mxfp8_vs128",
        Some(P::T128) => "fmha_fwd_hd128_mxfp8_t128",
        Some(P::KvSplit) => "fmha_fwd_hd128_mxfp8_kvsplit",
        Some(P::Combine) => "fmha_fwd_hd128_mxfp8_kvsplit_combine8h",
        Some(P::Default) | Some(P::Gemm) => "",
    }
}

/// Lowers, verifies, allocates and emits a kernel as gfx950 assembly text with our own tail.
pub fn compile(f: &amdgpu_dsl::Function) -> Result<Compiled, String> {
    compile_with(f, &backend::OwnTail)
}

/// Same pipeline, with a selectable tail backend (OwnTail or LlvmTail).
pub fn compile_with(f: &amdgpu_dsl::Function, tail: &dyn backend::TailBackend) -> Result<Compiled, String> {
    let (m, alloc) = lower_and_allocate(f)?;
    gpu_verify::enforce(f, &m, &alloc)?;
    let asm = tail.finish(&m, &alloc)?;
    Ok(Compiled { asm, vgpr: alloc.max_v, sgpr: alloc.max_s })
}

/// The final MIR and register assignment of a kernel (what the tail backends receive).
pub fn compile_mir_alloc(f: &amdgpu_dsl::Function) -> Result<(mir::MFunction, regalloc::Alloc), String> {
    lower_and_allocate(f)
}

/// The final (allocated, scheduled, hazard-padded) MIR of a kernel, for model reports.
pub fn compile_mir(f: &amdgpu_dsl::Function) -> Result<mir::MFunction, String> {
    sched::order::PIN_VMWAIT_KERNEL.store(policy_name(f).ends_with("_vs6"), std::sync::atomic::Ordering::Relaxed);
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

/// M0 reuse for LDS-DMA loads is part of the GEMM profile (AMDC_M0_REUSE=0 turns it off for A/B runs); the FMHA kernels keep their shipped output.
fn m0_reuse_on(f: &amdgpu_dsl::Function) -> bool {
    f.compile_profile == Some(amdgpu_dsl::CompileProfile::Gemm) && std::env::var("AMDC_M0_REUSE").map_or(true, |v| v != "0")
}

fn lower_and_allocate_with(f: &amdgpu_dsl::Function, extra: &[&str]) -> Result<(mir::MFunction, regalloc::Alloc), String> {
    let pname = policy_name(f);
    // per-kernel lowering options (Function::lower_opts) are appended to the caller's extras
    let mut extra_all: Vec<&str> = extra.to_vec();
    extra_all.extend(f.lower_opts.iter().map(|s| s.as_str()));
    let extra: &[&str] = &extra_all;
    // The vendor-structure kernels need the register savings of `scalarize` (wave-uniform integer chains in SGPRs) to fit
    // (svscale / the 3-slot swizzled ring); `sgpr` and `addrfold` miscompile them (wrong results, see docs/status.md round 47).
    let base = if (pname.starts_with("fmha_fwd_hd128_mxfp8_vs") || pname.ends_with("_kvsplit") || (pname.ends_with("_g2") || pname.ends_with("_g3") || pname.ends_with("_g3o") || pname.ends_with("_t128"))) && std::env::var("AMDC_LOWER_OPTS").map_or(true, |v| v.is_empty()) {
        // measured on vs128 (15/15, 301.9-304.0 vs 280.6 TFLOP/s): retarget2 (cvt results written straight into the MFMA fragment
        // lanes), imm and sgpr cut ~230 copies/VALU and free 12 VGPRs; addrfold and shl are slower and stay off. `imm` also gives
        // the vs128/vsm builds the register slack they need (docs/status.md rounds 49-50).
        let mut v: Vec<&str> = vec!["dce", "cmpfold", "scalarize", "retarget2", "imm", "sgpr", "zeroinit"];
        // `_t128` (4-wave blocks, one wave per SIMD): addrfold frees the DMA address temporaries so the full scheduler fits (416 -> 494 TFLOP/s)
        if pname.ends_with("_t128") {
            v.push("addrfold");
        }
        if std::env::var("FMHA_OPT").map_or(false, |x| x.split(|c: char| c == ',' || c == ' ').any(|t| t == "saddrlds")) {
            v.push("saddrlds");
        }
        if std::env::var("FMHA_OPT").map_or(false, |x| x.split(|c: char| c == ',' || c == ' ').any(|t| t == "pmdead")) {
            v.push("pmdead");
        }
        v.extend_from_slice(extra);
        lower::with_opts(&v, || lower::lower(f))?
    } else if f.compile_profile == Some(amdgpu_dsl::CompileProfile::Gemm) && std::env::var("AMDC_LOWER_OPTS").map_or(true, |v| v.is_empty()) {
        // CompileProfile::Gemm: partial lgkmcnt waits keep the prefetched fragment reads in flight (`lgkmn`)
        // + scalarize / scalarize2 / waveuni: the wave-uniform DMA address chains (tile origins with clamps/selects, slot bases, M0, soffsets) in SALU
        let mut v: Vec<&str> = vec!["dce", "cmpfold", "lgkmn", "zeroinit", "scalarize", "scalarize2", "waveuni"];
        v.extend_from_slice(extra);
        lower::with_opts(&v, || lower::lower(f))?
    } else if pname.starts_with("fmha_fwd_hd128_mxfp8") && std::env::var("AMDC_LOWER_OPTS").map_or(true, |v| v.is_empty()) {
        // the FMHA kernels zero their accumulators in place (bit-exact on all 15 shapes, -160 v_mov on _t, docs/status.md)
        let mut v: Vec<&str> = vec!["zeroinit"];
        v.extend_from_slice(extra);
        lower::lower_with_extra(f, &v)?
    } else {
        lower::lower_with_extra(f, extra)?
    };
    // AGPR requests (DSL pin_agpr / gpr): move the vregs to the AGPR file and repair the instructions that cannot touch it
    let base = {
        let mut b = base;
        agpr::assign(&mut b)?;
        b
    };
    let sched_opts = match sched::Options::from_env() {
        Some(o) if o.auto => sched::auto_plan(&base, &o),
        other => other,
    };
    // The FMHA kernels default to the wait-recomputing pass (partial vmcnt/lgkmcnt waits, ALU across barriers, ds_read hoisting): +9% on the
    // transposed kernel, bit-identical results (measured, docs/status.md round 44). AMDC_SCHED=0 turns it off.
    let sched_opts = match sched_opts {
        // GEMM profile / explicit Function::sched_opts: the scoreboard wait pass (+ ds_read hoisting, no reordering) with the kernel's own tokens on top
        None if std::env::var("AMDC_SCHED").ok().as_deref() != Some("0") && (f.compile_profile == Some(amdgpu_dsl::CompileProfile::Gemm) || f.sched_opts.is_some()) => {
            let mut o = sched::Options::default();
            if f.compile_profile == Some(amdgpu_dsl::CompileProfile::Gemm) {
                o.reorder = false;
                o.ldshoist = true;
            }
            if let Some(s) = &f.sched_opts {
                o.apply_tokens(s);
            }
            Some(o)
        }
        None if std::env::var("AMDC_SCHED").ok().as_deref() != Some("0")
            && (pname.starts_with("fmha_fwd_hd128_mxfp8_t") || pname.starts_with("fmha_fwd_hd128_mxfp8_sv") || pname.starts_with("fmha_fwd_hd128_mxfp8_vs") || (pname.ends_with("_g2") || pname.ends_with("_g3") || pname.ends_with("_g3o") || pname.ends_with("_t128"))) =>
        {
            // "ld" preset: scoreboard (partial) waits + ds_read hoisting, no instruction reordering. Measured best on the
            // merged tree (docs/status.md round 45): 286.5 TFLOP/s vs 285.1 with rewait,xbar; bit-identical results.
            let mut o = sched::Options::default();
            // `_t128`: ALU may cross the single barrier per tile and the pressure limit 248 lets the list scheduler take a schedule at vgpr 256
            // (measured 489 -> 505 TFLOP/s on B1H8S2048; limit 246/248 only, 244/250 give the old schedule: a heuristic knob, re-measure on changes)
            if pname.ends_with("_t128") {
                o.alu_across_barrier = true;
                o.max_vgprs = 248;
            }
            // `_vs128` keeps the full list scheduler (reordering): 299.1 vs 287.9 TFLOP/s with the `ld` preset (docs/status.md round 49)
            if !pname.ends_with("_vs128") && !pname.ends_with("_vs5") && !pname.ends_with("_vs6") && !pname.ends_with("_vs9") && !pname.ends_with("_kvsplit") && !pname.ends_with("_vsm") && !pname.ends_with("_t128") && !(pname.ends_with("_g2") || pname.ends_with("_g3") || pname.ends_with("_g3o")) {
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
                if m0_reuse_on(f) {
                    sched::m0::reuse(&mut m);
                }
                let pe = pin_check::check(&m);
                if !pe.is_empty() {
                    return Err(format!("pinned (safe-asm) blocks fail the final MIR check: {}", pe.join("; ")));
                }
                let vcc_fill = pname.starts_with("fmha_fwd_hd128_mxfp8_vs") && std::env::var("AMDC_VCC_FILL").map_or(false, |v| !v.is_empty() && v != "0");
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
    sched::phase::strip(&mut m); // scheduler directives are meaningless without the scheduler
    if m0_reuse_on(f) {
        sched::m0::reuse(&mut m);
    }
    let pe = pin_check::check(&m);
    if !pe.is_empty() {
        return Err(format!("pinned (safe-asm) blocks fail the final MIR check: {}", pe.join("; ")));
    }
    hazards::fix_value_hazards(&mut m);
    let errs = mir::verify(&m);
    if !errs.is_empty() {
        return Err(format!("MIR verification failed:\n  {}", errs.join("\n  ")));
    }
    // the unscheduled path trusted its lowering-time waits; passes after lowering (LDS spill, ...) can invalidate them: re-check on every path
    sched::waits::verify_waits(&m).map_err(|e| format!("wait verification failed: {e}"))?;
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
        Ok(m) => m.blocks.iter().any(|b| b.insts.iter().any(|i| matches!(i.op, mir::MOp::GlobalLoadLdsB128 | mir::MOp::GlobalLoadLdsB32 | mir::MOp::GlobalLoadLdsB128S | mir::MOp::GlobalLoadLdsB32S | mir::MOp::BufferLoadDwordX4Lds | mir::MOp::BufferLoadDwordX1Lds) || (!llvm_tail_has_rows() && matches!(i.op, mir::MOp::BufferWbl2 | mir::MOp::BufferInv | mir::MOp::GlobalAtomicAddRet)))),
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
    &["probe_agpr_reload4_ref", "probe_agpr_reload4", "probe_agpr_reload_ref", "probe_agpr_reload", "probe_agpr_m16_ref", "probe_agpr_m16_mix", "probe_agpr_ref", "probe_agpr_acc", "probe_agpr_full", "probe_agpr_pinned", "attention_fwd", "attn_debug", "probe_store", "probe_math", "probe_mfma", "probe_lds", "gv_audit_region_trust", "probe_mfma_wide", "probe_mfma_fp8", "probe_mfma_scale", "probe_roles", "probe_waves2", "probe_lds_self2", "probe_roles_loop", "probe_fmha_datapath", "probe_ab_loops", "probe_nested_acc", "probe_loop_mfma", "probe_fmha_core", "probe_pack", "probe_fp8_matmul", "probe_fp8_load", "probe_fp8_min", "probe_fp8_minload", "probe_fp8_softmax", "probe_fp8_fmha_tile", "probe_fp8_fmha_2tile", "probe_fp8_fmha_online", "probe_fp8_fmha_oreg", "probe_fp8_fmha_multiwave", "probe_fp8_fmha_pingpong", "probe_fp8_fmha_abloops", "probe_lds_tr_b8", "probe_fp8_matmul_tr", "probe_fp8_fmha_tr", "probe_fp8_fmha_trab", "probe_fp8_fmha_hd128", "probe_fp8_fmha_hd128b", "probe_fp8_fmha_hd128c", "probe_fp8_matmul_blk", "fmha_fwd_hd128_mxfp8", "fmha_fwd_hd128_mxfp8_t", "fmha_fwd_hd128_mxfp8_pp", "fmha_fwd_hd128_mxfp8_sv", "sched_ovl_both", "sched_ovl_mfma", "sched_ovl_valu", "sched_ovl_vfirst", "sched_coissue_both", "sched_coissue_mfma", "sched_coissue_valu", "sched_coissue_cboth", "sched_coissue_cvalu", "sched_valu_mul", "sched_valu_max3", "sched_valu_exp", "sched_mfma_chain", "sched_mfma_valu16", "sched_mfma_valu4", "sched_diamond", "sched_occ_chain", "probe_log_store4", "probe_llvm_kernarg", "probe_llvm_ids", "probe_llvm_roles", "probe_llvm_lds", "probe_llvm_branch", "probe_llvm_loop", "probe_llvm_atomic", "probe_llvm_maskge", "probe_llvm_k1", "probe_llvm_k2", "probe_buffer_rw", "probe_mfma_cform", "probe_store_x2", "probe_last_arriver", "probe_last_arriver_lite", "probe_last_arriver_nofence", "probe_last_arriver_plain", "probe_last_arriver_plain_fenced", "probe_la8", "probe_la8_lite", "probe_la8_plain_fenced", "probe_la8_plain", "probe_la8_nodrain", "probe_la_verify", "fmha_fwd_hd128_mxfp8_kvsplit_combine", "fmha_fwd_hd128_mxfp8_kvsplit_combine4", "probe_mfma_unscaled", "probe_expr_ids", "probe_expr_kernarg"]
}

pub fn kernel_by_name(name: &str) -> Option<amdgpu_dsl::Function> {
    match name {
        "probe_agpr_m16_ref" => Some(amdgpu_dsl::kernels::probe_agpr_m16_ref()),
        "probe_agpr_m16_mix" => Some(amdgpu_dsl::kernels::probe_agpr_m16_mix()),
        "probe_agpr_reload_ref" => Some(amdgpu_dsl::kernels::probe_agpr_reload_ref()),
        "probe_agpr_reload" => Some(amdgpu_dsl::kernels::probe_agpr_reload()),
        "probe_agpr_reload4_ref" => Some(amdgpu_dsl::kernels::probe_agpr_reload4_ref()),
        "probe_agpr_reload4" => Some(amdgpu_dsl::kernels::probe_agpr_reload4()),
        "probe_dma_voff_spill" => Some(amdgpu_dsl::kernels::probe_dma_voff_spill()),
        "probe_agpr_ref" => Some(amdgpu_dsl::kernels::probe_agpr_ref()),
        "probe_agpr_acc" => Some(amdgpu_dsl::kernels::probe_agpr_acc()),
        "probe_agpr_full" => Some(amdgpu_dsl::kernels::probe_agpr_full()),
        "probe_agpr_pinned" => Some(amdgpu_dsl::kernels::probe_agpr_pinned()),
        "attention_fwd" => Some(amdgpu_dsl::kernels::attention_fwd()),
        "attn_debug" => Some(amdgpu_dsl::kernels::attn_debug()),
        "probe_store" => Some(amdgpu_dsl::kernels::probe_store()),
        "probe_math" => Some(amdgpu_dsl::kernels::probe_math()),
        "probe_mfma" => Some(amdgpu_dsl::kernels::probe_mfma()),
        "probe_lds" => Some(amdgpu_dsl::kernels::probe_lds()),
        "gv_audit_region_trust" => Some(amdgpu_dsl::kernels::gv_audit_region_trust()),
        "probe_lds_oob" => Some(amdgpu_dsl::kernels::probe_lds_oob()),
        "gv_attack_false_contract" => Some(amdgpu_dsl::kernels::gv_attack_false_contract()),
        "gv_attack_assume_loop" => Some(amdgpu_dsl::kernels::gv_attack_assume_loop()),
        "gv_attack_assume_trusted" => Some(amdgpu_dsl::kernels::gv_attack_assume_trusted()),
        "gv_attack_lds_const" => Some(amdgpu_dsl::kernels::gv_attack_lds_const()),
        "gv_attack_desc_over" => Some(amdgpu_dsl::kernels::gv_attack_desc_over()),
        "gv_attack_clamp_path" => Some(amdgpu_dsl::kernels::gv_attack_clamp_path()),
        "probe_expr_ids" => Some(amdgpu_dsl::kernels::probe_expr_ids()),
        "probe_expr_kernarg" => Some(amdgpu_dsl::kernels::probe_expr_kernarg()),
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
        "fmha_fwd_hd128_mxfp8_kvsplit_combine16h" => Some(amdgpu_dsl::kernels::fmha_fwd_hd128_mxfp8_kvsplit_combine16h()),
        "fmha_fwd_hd128_mxfp8_kvsplit_combine8h" => Some(amdgpu_dsl::kernels::fmha_fwd_hd128_mxfp8_kvsplit_combine8h()),
        "gemm_mxfp8_bs_v0" => Some(amdgpu_dsl::gemm_mxfp8::gemm_mxfp8_bs_v0()),
        "gemm_mxfp8_bs_v1" => Some(amdgpu_dsl::gemm_mxfp8::gemm_mxfp8_bs_v1()),
        n if n.starts_with("gemm_mxfp8_bs_v") && amdgpu_kernels::gemm_mxfp8::by_name(n).is_some() => amdgpu_kernels::gemm_mxfp8::by_name(n),
        "fmha_fwd_hd128_mxfp8_dsl_vs128" => Some(amdgpu_dsl::fmha_tile::fmha_fwd_hd128_mxfp8_dsl_vs128()),
        "fmha_fwd_hd128_mxfp8_dsl_t128" => Some(amdgpu_dsl::fmha_tile::fmha_fwd_hd128_mxfp8_dsl_t128()),
        "fmha_fwd_hd128_mxfp8_dsl_kvsplit" => Some(amdgpu_dsl::fmha_tile::fmha_fwd_hd128_mxfp8_dsl_kvsplit()),
        "fmha_fwd_hd128_mxfp8_dsl_kvsplit_combine8h" => Some(amdgpu_dsl::fmha_tile::fmha_fwd_hd128_mxfp8_dsl_kvsplit_combine8h()),
        // P3: the same four kernels on the layout layer (amdgpu-layout + amdgpu-isa + amdgpu-tile)
        "fmha_fwd_hd128_mxfp8_v3_vs128" => Some(amdgpu_tile::fmha::fmha_fwd_hd128_mxfp8_v3_vs128()),
        "fmha_fwd_hd128_mxfp8_v3_t128" => Some(amdgpu_tile::fmha::fmha_fwd_hd128_mxfp8_v3_t128()),
        "fmha_fwd_hd128_mxfp8_v3_kvsplit" => Some(amdgpu_tile::fmha::fmha_fwd_hd128_mxfp8_v3_kvsplit()),
        "fmha_fwd_hd128_mxfp8_v3_kvsplit_combine8h" => Some(amdgpu_tile::fmha::fmha_fwd_hd128_mxfp8_v3_kvsplit_combine8h()),
        "fmha_fwd_hd128_mxfp8_kvsplit_combine4h" => Some(amdgpu_dsl::kernels::fmha_fwd_hd128_mxfp8_kvsplit_combine4h()),
        "fmha_fwd_hd128_mxfp8_kvsplit_combine4" => Some(amdgpu_dsl::kernels::fmha_fwd_hd128_mxfp8_kvsplit_combine4()),
        "fmha_fwd_hd128_mxfp8_kvsplit_combine" => Some(amdgpu_dsl::kernels::fmha_fwd_hd128_mxfp8_kvsplit_combine()),
        "fmha_fwd_hd128_mxfp8_vs9" => Some(amdgpu_dsl::kernels::fmha_fwd_hd128_mxfp8_vs9()),
        "fmha_fwd_hd128_mxfp8_vs6" => Some(amdgpu_dsl::kernels::fmha_fwd_hd128_mxfp8_vs6()),
        "fmha_fwd_hd128_mxfp8_vs5" => Some(amdgpu_dsl::kernels::fmha_fwd_hd128_mxfp8_vs5()),
        "fmha_fwd_hd128_mxfp8_vs128" => Some(amdgpu_dsl::kernels::fmha_fwd_hd128_mxfp8_vs128()),
        "fmha_fwd_hd128_mxfp8_vsm" => Some(amdgpu_dsl::kernels::fmha_fwd_hd128_mxfp8_vsm()),
        "probe_dma" => Some(amdgpu_dsl::kernels::probe_dma()),
        "probe_safeasm_e2e" => Some(amdgpu_dsl::kernels::probe_safeasm_e2e()),
        "probe_log_store4" => Some(amdgpu_dsl::kernels::probe_log_store4()),
        "probe_llvm_kernarg" => Some(amdgpu_dsl::kernels::probe_llvm_kernarg()),
        "probe_llvm_ids" => Some(amdgpu_dsl::kernels::probe_llvm_ids()),
        "probe_llvm_roles" => Some(amdgpu_dsl::kernels::probe_llvm_roles()),
        "probe_llvm_lds" => Some(amdgpu_dsl::kernels::probe_llvm_lds()),
        "probe_llvm_branch" => Some(amdgpu_dsl::kernels::probe_llvm_branch()),
        "probe_llvm_loop" => Some(amdgpu_dsl::kernels::probe_llvm_loop()),
        "probe_llvm_atomic" => Some(amdgpu_dsl::kernels::probe_llvm_atomic()),
        "probe_llvm_maskge" => Some(amdgpu_dsl::kernels::probe_llvm_maskge()),
        "probe_llvm_k1" => Some(amdgpu_dsl::kernels::probe_llvm_k1()),
        "probe_llvm_k2" => Some(amdgpu_dsl::kernels::probe_llvm_k2()),
        "probe_mfma_unscaled" => Some(amdgpu_dsl::kernels::probe_mfma_unscaled()),
        "probe_buffer_rw" => Some(amdgpu_dsl::kernels::probe_buffer_rw()),
        "probe_buffer_dyn" => Some(amdgpu_dsl::kernels::probe_buffer_dyn()),
        "probe_buffer_lds" => Some(amdgpu_dsl::kernels::probe_buffer_lds()),
        "probe_mfma_cform" => Some(amdgpu_dsl::kernels::probe_mfma_cform()),
        "probe_store_x2" => Some(amdgpu_dsl::kernels::probe_store_x2()),
        "probe_cvtchain_dsl" => Some(amdgpu_dsl::kernels::probe_cvtchain_dsl()),
        "probe_cvtchain_asm" => Some(amdgpu_dsl::kernels::probe_cvtchain_asm()),
        "probe_la8" => Some(amdgpu_dsl::kernels::probe_la8()),
        "probe_la8_lite" => Some(amdgpu_dsl::kernels::probe_la8_lite()),
        "probe_la8_plain_fenced" => Some(amdgpu_dsl::kernels::probe_la8_plain_fenced()),
        "probe_la8_plain" => Some(amdgpu_dsl::kernels::probe_la8_plain()),
        "probe_la8_nodrain" => Some(amdgpu_dsl::kernels::probe_la8_nodrain()),
        "probe_la_verify" => Some(amdgpu_dsl::kernels::probe_la_verify()),
        "probe_last_arriver" => Some(amdgpu_dsl::kernels::probe_last_arriver()),
        "probe_last_arriver_lite" => Some(amdgpu_dsl::kernels::probe_last_arriver_lite()),
        "probe_last_arriver_plain" => Some(amdgpu_dsl::kernels::probe_last_arriver_plain()),
        "probe_last_arriver_plain_fenced" => Some(amdgpu_dsl::kernels::probe_last_arriver_plain_fenced()),
        "probe_last_arriver_nofence" => Some(amdgpu_dsl::kernels::probe_last_arriver_nofence()),
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
/// The atomic/fence ops of the last-arriver pattern have rows in the LLVM tables only when `llvm_tail_has_rows` says so (set by the table probe).
fn llvm_tail_has_rows() -> bool {
    true
}
