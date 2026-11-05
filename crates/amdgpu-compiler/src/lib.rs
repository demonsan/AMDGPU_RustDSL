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
pub mod functional_proofs;
pub mod ir_print;
pub mod hazards;
pub mod lower;
pub mod mir;
pub mod op_attrs;
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

/// M0 reuse for LDS-DMA loads is part of the GEMM profile; the FMHA kernels keep their shipped output.
fn m0_reuse_on(f: &amdgpu_dsl::Function) -> bool {
    f.compile_profile == Some(amdgpu_dsl::CompileProfile::Gemm)
}

fn lower_and_allocate_with(f: &amdgpu_dsl::Function, extra: &[&str]) -> Result<(mir::MFunction, regalloc::Alloc), String> {
    let pname = policy_name(f);
    // per-kernel lowering options (Function::lower_opts) are appended to the caller's extras
    let mut extra_all: Vec<&str> = extra.to_vec();
    extra_all.extend(f.lower_opts.iter().map(|s| s.as_str()));
    let extra: &[&str] = &extra_all;
    // The vendor-structure kernels need the register savings of `scalarize` (wave-uniform integer chains in SGPRs) to fit
    // (svscale / the 3-slot swizzled ring); `sgpr` and `addrfold` miscompile them (wrong results, see docs/archive/status.md round 47).
    let base = if (pname.starts_with("fmha_fwd_hd128_mxfp8_vs") || pname.ends_with("_kvsplit") || (pname.ends_with("_g2") || pname.ends_with("_g3") || pname.ends_with("_g3o") || pname.ends_with("_t128"))) {
        // measured on vs128 (15/15, 301.9-304.0 vs 280.6 TFLOP/s): retarget2 (cvt results written straight into the MFMA fragment
        // lanes), imm and sgpr cut ~230 copies/VALU and free 12 VGPRs; addrfold and shl are slower and stay off. `imm` also gives
        // the vs128/vsm builds the register slack they need (docs/archive/status.md rounds 49-50).
        let mut v: Vec<&str> = vec!["dce", "cmpfold", "scalarize", "retarget2", "imm", "sgpr", "zeroinit"];
        // `_t128` (4-wave blocks, one wave per SIMD): addrfold frees the DMA address temporaries so the full scheduler fits (416 -> 494 TFLOP/s)
        if pname.ends_with("_t128") {
            v.push("addrfold");
        }
        // v3 `_t128`: the SADDR DMA form (scalar tensor base + per-lane 32-bit offset: no 64-bit pointer add per DMA) measured +1.8..+2.1% on the
        // B2H8S2048 layout median (two same-run A/Bs, docs/dsl-layout-refactor-design.md); on _vs128 / _kvsplit it is -0.2..-1.4%: not adopted there
        // v3 `_t128`: `edgepad` (MFMA hazard windows carried across blocks, no block-end padding) and `salubr` (wave-uniform branch conditions on SCC):
        // same-run A/B, B2H8S2048 layout median +3.3% together (on `_vs128` / `_kvsplit` -0.7..-1.5%: not used there, docs/dsl-layout-refactor-design.md
        // §9.4); `saddrlds` (the SADDR DMA form: +1.8..+2.1%; -0.2..-1.4% on _vs128 / _kvsplit)
        if pname.ends_with("_t128") && f.name.starts_with("fmha_fwd_hd128_mxfp8_v3_") {
            v.extend_from_slice(&["saddrlds", "edgepad", "salubr"]);
        }
        v.extend_from_slice(extra);
        lower::with_opts(&v, || lower::lower(f))?
    } else if f.compile_profile == Some(amdgpu_dsl::CompileProfile::Gemm) {
        // CompileProfile::Gemm: partial lgkmcnt waits keep the prefetched fragment reads in flight (`lgkmn`)
        // + scalarize / scalarize2 / waveuni: the wave-uniform DMA address chains (tile origins with clamps/selects, slot bases, M0, soffsets) in SALU
        // + kabatch (kernarg loads batched into s_load_dwordx4) and imm (inline constants / literals as VALU operands instead of v_mov materialisations)
        let mut v: Vec<&str> = vec!["dce", "cmpfold", "lgkmn", "zeroinit", "scalarize", "scalarize2", "waveuni", "edgepad", "kabatch", "imm"];
        v.extend_from_slice(extra);
        lower::with_opts(&v, || lower::lower(f))?
    } else if pname.starts_with("fmha_fwd_hd128_mxfp8") {
        // the FMHA kernels zero their accumulators in place (bit-exact on all 15 shapes, -160 v_mov on _t, docs/archive/status.md)
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
    let sched_opts = sched::Options::from_env();
    // The FMHA kernels default to the wait-recomputing pass (partial vmcnt/lgkmcnt waits, ALU across barriers, ds_read hoisting): +9% on the
    // transposed kernel, bit-identical results (measured, docs/archive/status.md round 44). AMDC_SCHED=0 turns it off.
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
            // merged tree (docs/archive/status.md round 45): 286.5 TFLOP/s vs 285.1 with rewait,xbar; bit-identical results.
            let mut o = sched::Options::default();
            // `_t128`: ALU may cross the single barrier per tile and the pressure limit 248 lets the list scheduler take a schedule at vgpr 256
            // (measured 489 -> 505 TFLOP/s on B1H8S2048; limit 246/248 only, 244/250 give the old schedule: a heuristic knob, re-measure on changes)
            if pname.ends_with("_t128") {
                o.alu_across_barrier = true;
                o.max_vgprs = 248;
            }
            // `_vs128` keeps the full list scheduler (reordering): 299.1 vs 287.9 TFLOP/s with the `ld` preset (docs/archive/status.md round 49)
            if !pname.ends_with("_vs128") && !pname.ends_with("_vs5") && !pname.ends_with("_vs6") && !pname.ends_with("_vs9") && !pname.ends_with("_kvsplit") && !pname.ends_with("_vsm") && !pname.ends_with("_t128") && !(pname.ends_with("_g2") || pname.ends_with("_g3") || pname.ends_with("_g3o")) {
                o.reorder = false;
                o.alu_across_barrier = true;
                o.ldshoist = true; // hoist ds_read above independent work (docs/scheduling.md); AMDC_SCHED=0 / 1 / auto / OPTS override
            }
            Some(o)
        }
        other => other,
    };
    // experiment switch (applied ON TOP of the kernel's own scheduler options, i.e. the shipped preset): AMDC_SCHED_ADD=<kernel name substring>:<tokens> (e.g.
    // `fmha_fwd_hd128_mxfp8_v3_vs128:ldahead=4,ldslack=8`; several entries separated by `;`) appends scheduler tokens to the options the kernel would use anyway
    let sched_opts = match (std::env::var("AMDC_SCHED_ADD").ok().filter(|v| !v.is_empty()), sched_opts) {
        (Some(spec), Some(mut o)) => {
            for ent in spec.split(';') {
                if let Some((pat, toks)) = ent.split_once(':') {
                    if pat == "*" || pname.contains(pat) || f.name.contains(pat) {
                        o.apply_tokens(toks);
                    }
                }
            }
            Some(o)
        }
        (_, so) => so,
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
    // pad the physical write-after-read hazard on every path, not only for scheduled code.
    hazards::pad_physical_war(&mut m, &alloc);
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
        Ok(m) => m.blocks.iter().any(|b| {
            b.insts.iter().any(|i| match op_attrs::attrs(i.op).backend {
                op_attrs::Backend::OwnOnly => true,
                op_attrs::Backend::OwnOnlyIfNoRows => !llvm_tail_has_rows(),
                op_attrs::Backend::Llvm | op_attrs::Backend::NoRow => false,
            })
        }),
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

/// The atomic/fence ops of the last-arriver pattern have rows in the LLVM tables only when `llvm_tail_has_rows` says so (set by the table probe).
fn llvm_tail_has_rows() -> bool {
    true
}
