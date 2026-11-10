//! Backend for the AMDGPU Rust DSL: lowering, register allocation, emission.

#![forbid(unsafe_code)]

pub mod agpr;
pub mod asm;
pub mod asm_check;
pub mod pin_check;
pub mod backend;
pub use target::gfx950::emit;
pub mod gpu_protocol;
pub use gpu_protocol as gpu_tiles;
pub mod gpu_verify;
pub mod functional_proofs;
pub mod ir_print;
pub mod diag;
pub mod hazards;
pub mod hzfill;
pub mod pass;
pub mod target;
pub mod lower;
pub mod mir;
pub use target::gfx950::op_attrs;
pub mod regalloc;
pub mod regplan;
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
    regalloc::check_workgroup_budget(&m, &alloc)?;
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
    for extra in pass::OUTER_LADDER {
        match lower_and_allocate_with(f, extra) {
            Ok(r) => {
                if r.0.lds_bytes > 160 * 1024 {
                    return Err(format!("LDS spill needs {} bytes of LDS (limit 163840)", r.0.lds_bytes));
                }
                return Ok(r);
            }
            Err(e) if pass::retryable(&e) => last = e,
            Err(e) => return Err(e),
        }
    }
    Err(last)
}

/// M0 reuse for LDS-DMA loads is part of the GEMM profile; the FMHA kernels keep their shipped output.
fn m0_reuse_on(f: &amdgpu_dsl::Function) -> bool {
    f.compile_policy().m0_reuse
}

fn lower_and_allocate_with(f: &amdgpu_dsl::Function, extra: &[&str]) -> Result<(mir::MFunction, regalloc::Alloc), String> {
    let pname = policy_name(f);
    // per-kernel lowering options (Function::lower_opts) are appended to the caller's extras
    let mut extra_all: Vec<&str> = extra.to_vec();
    extra_all.extend(f.lower_opts.iter().map(|s| s.as_str()));
    let extra: &[&str] = &extra_all;
    // the kernel's declared compile policy (amdgpu_dsl::CompileProfile::policy): its lowering tokens, then the retry ladder's and the kernel's own
    let pol = f.compile_policy();
    let mut v: Vec<&str> = pol.lower.to_vec();
    v.extend_from_slice(extra);
    let base = match pol.lower_base {
        amdgpu_dsl::LowerBase::Replace => lower::with_opts(&v, || lower::lower(f))?,
        amdgpu_dsl::LowerBase::Inherit => lower::lower_with_extra(f, &v)?,
    };
    // AGPR requests (DSL pin_agpr / gpr): move the vregs to the AGPR file and repair the instructions that cannot touch it
    let base = {
        let mut b = base;
        agpr::assign(&mut b)?;
        b
    };
    let sched_opts = scheduler_options(f).map(|mut o| {
        // weave regions: the strategies declared by name, mapped onto the region ids lowering stamped (one id per scope instance; an undeclared region inherits its parent's strategy)
        if !declared_regions(f).is_empty() {
            o.region_of = region_strategies(f, &base);
        }
        o
    });
    // experiment switch (applied ON TOP of the kernel's own scheduler options, i.e. the shipped preset): AMDC_SCHED_ADD=<kernel name substring>:<tokens> (e.g.
    // `fmha_fwd_hd128_mxfp8_v3_vs128:ldsink=2`; several entries separated by `;`) appends scheduler tokens to the options the kernel would use anyway
    let sched_opts = match (std::env::var("AMDC_SCHED_ADD").ok().filter(|v| !v.is_empty()), sched_opts) {
        (Some(spec), Some(mut o)) => {
            for ent in spec.split([';', '|']) {
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
    // `hzfill` (hzfill.rs): from the kernel's scheduler options, or AMDC_SCHED_ADD=<kernel substring>:hzfill (also for kernels without a scheduler preset)
    let hz_opts = sched_opts.clone();
    let hzfill = sched_opts.as_ref().map_or(false, |o| o.any_region(|r| r.hzfill))
        || std::env::var("AMDC_SCHED_ADD").ok().map_or(false, |spec| {
            spec.split(';').filter_map(|e| e.split_once(':')).any(|(pat, toks)| (pat == "*" || pname.contains(pat) || f.name.contains(pat)) && toks.split(',').any(|t| t.trim() == "hzfill"))
        });
    let m0_reuse = m0_reuse_on(f);
    if let Some(opts) = sched_opts {
        let mut last_err = String::new();
        for (cut, hoist) in pass::inner_ladder(&opts, base.workgroup_size) {
            let mut o = opts.clone();
            o.max_vgprs = opts.max_vgprs.saturating_sub(cut);
            if let Some(h) = hoist {
                o.hoist_dist = h;
                o.ldshoist = h > 0 && opts.ldshoist;
            }
            let cx = pass::PassCtx { f, path: pass::Path::Scheduled, sched: Some(&o), hz_opts: hz_opts.as_ref(), hzfill, m0_reuse };
            let mut s = pass::MirState { m: base.clone(), alloc: None };
            match pass::run_body(pass::SCHEDULED, &mut s, &cx) {
                Ok(()) => return Ok((s.m, s.alloc.expect("allocated"))),
                Err(e) if pass::retryable(&e) => last_err = e,
                Err(e) => return Err(e),
            }
        }
        eprintln!("sched: falling back to the unscheduled code ({})", last_err.chars().take(if std::env::var("AMDC_SCHED_VERBOSE").is_ok() { 6000 } else { 120 }).collect::<String>());
    }
    let cx = pass::PassCtx { f, path: pass::Path::Unscheduled, sched: None, hz_opts: hz_opts.as_ref(), hzfill, m0_reuse };
    let mut s = pass::MirState { m: base, alloc: None };
    pass::run_body(pass::UNSCHEDULED, &mut s, &cx)?;
    Ok((s.m, s.alloc.expect("allocated")))
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
                op_attrs::Backend::Llvm | op_attrs::Backend::NoRow => false,
            })
        }),
        Err(_) => false,
    }
}

/// The scheduler options a kernel compiles with. `AMDC_SCHED` (1 / ld) overrides everything; `AMDC_SCHED=0` turns the scheduler off. Otherwise the kernel's PROFILE PRESET
/// applies - the GEMM profile (scoreboard waits + ds_read hoisting, no reordering) or the `_t128` profile (ALU across its barrier, the pressure limit 248) - and the kernel's own `Function::sched_opts` tokens are applied ON TOP of
/// that preset (never instead of it: a kernel that adds a token keeps its profile's options). A kernel without a preset that sets tokens gets the default options plus them.
pub fn scheduler_options(f: &amdgpu_dsl::Function) -> Option<sched::Options> {
    let mut o = scheduler_options_base(f);
    let decls = declared_regions(f);
    if !decls.is_empty() && !matches!(std::env::var("AMDC_SCHED").ok().as_deref(), Some("0")) {
        // per-region strategies: the scheduler runs for them even on a kernel without a preset (the kernel default then is `rewait`: waits only)
        let base = o.get_or_insert_with(|| {
            let mut d = sched::Options::default();
            d.reorder = false;
            d
        });
        base.regions = decls.iter().map(|r| sched::region::RegionSpec { name: r.name.clone(), authored: r.authored, tokens: r.tokens.clone() }).collect();
    }
    o
}

/// The kernel's declared region strategies plus the experiment switch `AMDC_SCHED_REGION=<region name>:<authored|free>[:<tokens>][;...] (`|` also separates entries, for tools that split on `;`)` (appended, a later entry of the same name wins):
/// per-region A/B tests without editing the kernel builder, e.g. `AMDC_SCHED_REGION=m_phase:authored:ldsink=1;window:free:reorder`.
fn declared_regions(f: &amdgpu_dsl::Function) -> Vec<amdgpu_dsl::RegionDecl> {
    let mut out = f.regions.clone();
    if let Some(spec) = std::env::var("AMDC_SCHED_REGION").ok().filter(|v| !v.is_empty()) {
        for ent in spec.split([';', '|']) {
            let mut it = ent.splitn(3, ':');
            let (Some(name), Some(mode)) = (it.next(), it.next()) else { continue };
            let d = amdgpu_dsl::RegionDecl { name: name.trim().to_string(), authored: mode.trim() == "authored", tokens: it.next().unwrap_or("").trim().to_string() };
            out.retain(|r| r.name != d.name);
            out.push(d);
        }
    }
    out
}

/// `MInst::region` id (1-based index into `MFunction::reg_hints.regions`, 0 = none) -> strategy id (1-based index into `Function::regions`, 0 = the kernel default): the nearest declared
/// region on the way up the nesting.
pub fn region_strategies(f: &amdgpu_dsl::Function, m: &mir::MFunction) -> Vec<u16> {
    let regs = &m.reg_hints.regions;
    let decls = declared_regions(f);
    let mut out = vec![0u16; regs.len() + 1];
    for (k, (name, parent)) in regs.iter().enumerate() {
        let own = decls.iter().position(|r| &r.name == name).map(|i| (i + 1) as u16);
        // parents are lowered before their children: out[parent] is final
        out[k + 1] = own.unwrap_or(if *parent == 0 { 0 } else { out[*parent as usize] });
    }
    out
}

fn scheduler_options_base(f: &amdgpu_dsl::Function) -> Option<sched::Options> {
    use amdgpu_dsl::SchedPreset as P;
    if let Some(o) = sched::Options::from_env() {
        return Some(o);
    }
    if std::env::var("AMDC_SCHED").ok().as_deref() == Some("0") {
        return None;
    }
    let preset = match f.compile_policy().sched_preset {
        // scoreboard (partial) waits + ds_read hoisting, no instruction reordering (docs/scheduling.md; measured best on the GEMM)
        P::Gemm => {
            let mut o = sched::Options::default();
            o.reorder = false;
            o.ldshoist = true;
            Some(o)
        }
        // `_t128`: ALU may cross the single barrier per tile and the pressure limit 248 lets the list scheduler take a schedule at vgpr 256 (measured 489 -> 505 TFLOP/s on B1H8S2048;
        // limit 246 / 248 only, 244 / 250 give the old schedule: a heuristic knob, re-measure on changes)
        P::T128 => {
            let mut o = sched::Options::default();
            o.alu_across_barrier = true;
            o.max_vgprs = 248;
            Some(o)
        }
        P::None => None,
    };
    match (preset, &f.sched_opts) {
        (Some(mut o), Some(s)) => {
            o.apply_tokens(s);
            Some(o)
        }
        (Some(o), None) => Some(o),
        (None, Some(s)) => {
            let mut o = sched::Options::default();
            o.apply_tokens(s);
            Some(o)
        }
        (None, None) => None,
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

