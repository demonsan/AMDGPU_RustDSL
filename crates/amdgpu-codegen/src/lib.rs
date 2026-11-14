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
pub mod fxhash;
pub mod gpu_verify;
pub mod functional_proofs;
pub mod ir_print;
pub mod diag;
pub mod hazards;
pub mod hzfill;
pub mod bweave;
pub mod lower_cf;
pub mod cfg_thread;
pub mod exact;
pub mod pass;
pub mod target;
pub mod lower;
pub mod mir;
pub use target::gfx950::op_attrs;
pub mod regalloc;
pub mod regplan;
pub mod sched;
pub mod timing;

pub struct Compiled {
    pub asm: String,
    pub vgpr: u32,
    pub sgpr: u32,
}

/// What a compile does besides producing assembly. `verify` runs the GPU-semantics verifier (`gpu_verify::enforce`) on the allocated MIR: a pure check (the assembly is the same either way), so a JIT may
/// run it offline per shape bucket and compile with `verify: false` (docs/compile-time.md).
#[derive(Clone, Debug)]
pub struct CompileOptions {
    pub verify: bool,
}

impl Default for CompileOptions {
    fn default() -> Self {
        CompileOptions { verify: true }
    }
}

/// Lowers, verifies, allocates and emits a kernel as gfx950 assembly text with our own tail.
pub fn compile(f: &amdgpu_dsl::Function) -> Result<Compiled, String> {
    compile_with(f, &backend::OwnTail)
}

/// Same pipeline, with a selectable tail backend (OwnTail or LlvmTail).
pub fn compile_with(f: &amdgpu_dsl::Function, tail: &dyn backend::TailBackend) -> Result<Compiled, String> {
    compile_with_options(f, tail, &CompileOptions::default())
}

/// `compile_with` with explicit options.
pub fn compile_with_options(f: &amdgpu_dsl::Function, tail: &dyn backend::TailBackend, o: &CompileOptions) -> Result<Compiled, String> {
    compile_with_mir(f, tail, o).map(|r| r.0)
}

/// `compile_with_options` that also returns the final MIR and register assignment (a caller that verifies the same compile does not compile twice).
pub fn compile_with_mir(f: &amdgpu_dsl::Function, tail: &dyn backend::TailBackend, o: &CompileOptions) -> Result<(Compiled, mir::MFunction, regalloc::Alloc), String> {
    let (m, alloc) = lower_and_allocate(f)?;
    timing::time("check-workgroup-budget", || regalloc::check_workgroup_budget(&m, &alloc))?;
    if o.verify {
        timing::time("verify-gpu", || gpu_verify::enforce(f, &m, &alloc))?;
    }
    let asm = timing::time("emit-tail", || tail.finish(&m, &alloc))?;
    let c = Compiled { asm, vgpr: alloc.max_v, sgpr: alloc.max_s };
    Ok((c, m, alloc))
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
/// The inner-ladder index meaning "the per-phase fallback rung" / "the unscheduled body" in a ladder hint and in [`last_rung`].
pub const RUNG_FREE_PHASES: u8 = 254;
pub const RUNG_UNSCHEDULED: u8 = 255;

thread_local! {
    static LAST_RUNG: std::cell::Cell<(u8, u8)> = const { std::cell::Cell::new((0, 0)) };
}

/// The (outer, inner) retry-ladder rung the last compile of this thread succeeded at (inner: the index in `pass::inner_ladder`, `RUNG_FREE_PHASES`, `RUNG_UNSCHEDULED`; 0 for a kernel without a scheduler): what
/// `Function::ladder_hint` records.
pub fn last_rung() -> (u8, u8) {
    LAST_RUNG.with(|c| c.get())
}

fn lower_and_allocate(f: &amdgpu_dsl::Function) -> Result<(mir::MFunction, regalloc::Alloc), String> {
    // Register pressure: when allocation fails, rematerialise loop-invariant constants next to their loop uses (remat1 -> remat3)
    // and try again. A kernel that allocates at the first attempt is unaffected.
    let mut last = String::new();
    let hint = f.ladder_hint;
    for (oi, extra) in pass::OUTER_LADDER.iter().enumerate() {
        // a hinted kernel starts at its known rung: the earlier outer rungs are skipped, and the inner ladder of the hinted outer rung starts at the hinted inner rung
        if hint.map_or(false, |h| (oi as u8) < h.0) {
            continue;
        }
        let inner_start = hint.filter(|h| h.0 as usize == oi).map_or(0, |h| h.1);
        match lower_and_allocate_with(f, extra, inner_start) {
            Ok((m, a, rung)) => {
                if m.lds_bytes > 160 * 1024 {
                    return Err(format!("LDS spill needs {} bytes of LDS (limit 163840)", m.lds_bytes));
                }
                LAST_RUNG.with(|c| c.set((oi as u8, rung)));
                return Ok((m, a));
            }
            Err(e) if pass::retryable(&e) => last = e,
            Err(e) => return Err(e),
        }
    }
    Err(last)
}

/// Experiment switch `AMDC_LOWER_ADD=<tokens>`: comma-separated lowering tokens appended to every non-exact kernel on top of its policy
/// (`name-substring:tokens` entries, separated by `;`, restrict them to matching kernels) - for measuring a token before it becomes a default.
pub fn lower_add_tokens(f: &amdgpu_dsl::Function) -> Vec<String> {
    match std::env::var("AMDC_LOWER_ADD") {
        Ok(spec) if !spec.is_empty() && !f.exact => spec
            .split(';')
            .filter_map(|e| match e.split_once(':') {
                Some((k, toks)) if f.name.contains(k) => Some(toks.to_string()),
                Some(_) => None,
                None => Some(e.to_string()),
            })
            .flat_map(|toks| toks.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect::<Vec<_>>())
            .collect(),
        _ => Vec::new(),
    }
}

/// M0 reuse for LDS-DMA loads is part of the GEMM profile; the FMHA kernels keep their shipped output.
fn m0_reuse_on(f: &amdgpu_dsl::Function) -> bool {
    // the GEMM profile's policy, or the lowering token `m0reuse` on any kernel (W3 B2: one M0 write per group of LDS-DMA issues)
    f.compile_policy().m0_reuse || f.lower_opts.iter().any(|t| t == "m0reuse") || lower_add_tokens(f).iter().any(|t| t == "m0reuse")
}

fn lower_and_allocate_with(f: &amdgpu_dsl::Function, extra: &[&str], inner_start: u8) -> Result<(mir::MFunction, regalloc::Alloc, u8), String> {
    // per-kernel lowering options (Function::lower_opts) are appended to the caller's extras
    let mut extra_all: Vec<&str> = extra.to_vec();
    extra_all.extend(f.lower_opts.iter().map(|s| s.as_str()));
    let lower_add = lower_add_tokens(f);
    extra_all.extend(lower_add.iter().map(|s| s.as_str()));
    // exact mode: operands in the form the instruction takes (inline / literal constants, SGPR sources), never via a copy
    if f.exact {
        // edgepad: MFMA windows still open at a block end are carried into the successors (checked there), not padded at every block end
        // saddr: a per-lane address over an SGPR-pair pointer is the global saddr form (`global_load v, vOff, s[..]`)
        extra_all.extend(["imm", "sgpr", "sgpr2", "salu32", "cmpneg", "edgepad", "saddr"]);
    }
    // default: MFMA hazard windows open at a block end are checked in the successors (`edgepad`), not padded at every block end;
    // a kernel opts out with `noedgepad`
    if !extra_all.contains(&"noedgepad") && !f.compile_policy().lower.contains(&"noedgepad") {
        extra_all.push("edgepad");
    }
    let extra: &[&str] = &extra_all;
    // the kernel's declared compile policy (amdgpu_dsl::CompileProfile::policy): its lowering tokens, then the retry ladder's and the kernel's own
    let pol = f.compile_policy();
    let mut v: Vec<&str> = pol.lower.to_vec();
    v.extend_from_slice(extra);
    let base = timing::time("lower-mir", || match pol.lower_base {
        amdgpu_dsl::LowerBase::Replace => lower::with_opts(&v, || lower::lower(f)),
        amdgpu_dsl::LowerBase::Inherit => lower::lower_with_extra(f, &v),
    })?;
    // AGPR requests (DSL pin_agpr / gpr): move the vregs to the AGPR file and repair the instructions that cannot touch it
    let base = {
        let mut b = base;
        timing::time("assign-agpr", || agpr::assign(&mut b))?;
        b
    };
    // exact authored mode: no scheduling, no allocation, no inserted waits / nops; the author's code is verified (docs/exact-mode.md)
    if f.exact {
        return exact::finalize(f, base).map(|(m, a)| (m, a, 0));
    }
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
                    if pat == "*" || f.name.contains(pat) {
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
            spec.split(';').filter_map(|e| e.split_once(':')).any(|(pat, toks)| (pat == "*" || f.name.contains(pat)) && toks.split(',').any(|t| t.trim() == "hzfill"))
        });
    let m0_reuse = m0_reuse_on(f);
    if let Some(opts) = sched_opts {
        let mut last_err = String::new();
        for (ri, (cut, hoist)) in pass::inner_ladder(&opts, base.workgroup_size).into_iter().enumerate() {
            if inner_start >= RUNG_FREE_PHASES || (ri as u8) < inner_start {
                continue;
            }
            let mut o = opts.clone();
            o.max_vgprs = opts.max_vgprs.saturating_sub(cut);
            if let Some(h) = hoist {
                o.hoist_dist = h;
                o.ldshoist = h > 0 && opts.ldshoist;
            }
            let cx = pass::PassCtx { f, path: pass::Path::Scheduled, sched: Some(&o), hz_opts: hz_opts.as_ref(), hzfill, m0_reuse };
            let mut s = pass::MirState::new(base.clone());
            match pass::run_body(pass::SCHEDULED, &mut s, &cx) {
                Ok(()) => return Ok((s.m, s.alloc.expect("allocated"), ri as u8)),
                Err(e) if pass::retryable(&e) => last_err = e,
                Err(e) => return Err(e),
            }
        }
        // W3 B6c: before the whole kernel falls back, keep the schedule everywhere except the back-end-ordered (Free) plan phases, whose
        // regions keep their emitted order (waits still recomputed); docs/w3-b5-b6.md section 4
        if let Some(o) = free_phases_unordered(f, &opts).filter(|_| inner_start != RUNG_UNSCHEDULED) {
            let cx = pass::PassCtx { f, path: pass::Path::Scheduled, sched: Some(&o), hz_opts: hz_opts.as_ref(), hzfill, m0_reuse };
            let mut s = pass::MirState::new(base.clone());
            match pass::run_body(pass::SCHEDULED, &mut s, &cx) {
                Ok(()) => {
                    if std::env::var("AMDC_SCHED_VERBOSE").is_ok() {
                        eprintln!("sched: {}: the Free phases keep their order (allocation failed with them scheduled)", f.name);
                    }
                    return Ok((s.m, s.alloc.expect("allocated"), RUNG_FREE_PHASES));
                }
                Err(e) if pass::retryable(&e) => last_err = e,
                Err(e) => return Err(e),
            }
        }
        if inner_start != RUNG_UNSCHEDULED {
            eprintln!("sched: falling back to the unscheduled code ({})", last_err.chars().take(if std::env::var("AMDC_SCHED_VERBOSE").is_ok() { 6000 } else { 120 }).collect::<String>());
        }
    }
    let cx = pass::PassCtx { f, path: pass::Path::Unscheduled, sched: None, hz_opts: hz_opts.as_ref(), hzfill, m0_reuse };
    let mut s = pass::MirState::new(base);
    pass::run_body(pass::UNSCHEDULED, &mut s, &cx)?;
    Ok((s.m, s.alloc.expect("allocated"), if hz_opts.is_some() { RUNG_UNSCHEDULED } else { 0 }))
}

/// The scheduler options with every region of a Free plan phase set to its emitted order (`rewait`), or None when no region belongs
/// to a Free phase (B6c's per-phase fallback rung).
pub fn free_phases_unordered(f: &amdgpu_dsl::Function, opts: &sched::Options) -> Option<sched::Options> {
    let p = f.plan.as_ref()?;
    let mut o = opts.clone();
    let mut any = false;
    for r in o.regions.iter_mut() {
        if p.phases.iter().any(|ph| ph.name == r.name && !matches!(ph.order, amdgpu_dsl::plan::OrderSpec::Authored)) {
            r.tokens = if r.tokens.trim().is_empty() { "rewait".to_string() } else { format!("{},rewait", r.tokens) };
            any = true;
        }
    }
    any.then_some(o)
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
                op_attrs::Backend::OwnOnly | op_attrs::Backend::Both => true,
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
        // W3 B6a: the VGPR budget of a back-end-ordered (Free) plan phase is the scheduler's pressure limit inside its region, unless the
        // region's own tokens set one (docs/w3-b5-b6.md section 4)
        if let Some(p) = &f.plan {
            for r in base.regions.iter_mut() {
                let budget = p.phases.iter().find(|ph| ph.name == r.name && !matches!(ph.order, amdgpu_dsl::plan::OrderSpec::Authored)).and_then(|ph| ph.budget.vgpr);
                if let Some(v) = budget {
                    if !r.tokens.split(',').any(|t| t.trim().starts_with("limit=")) {
                        r.tokens = if r.tokens.trim().is_empty() { format!("limit={v}") } else { format!("{},limit={v}", r.tokens) };
                    }
                }
            }
        }
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
    // exact mode: the author's instructions are printed by the own emitter (the LLVM tail has no rows for the exact-only forms)
    if f.exact {
        return ("own".to_string(), Some(format!("kernel {} is exact; emitting with --backend=own", f.name)));
    }
    if uses_own_only_ops(f) {
        ("own".to_string(), Some(format!("kernel {} uses own-backend-only ops (DMA); emitting with --backend=own", f.name)))
    } else {
        ("llvm-tail".to_string(), None)
    }
}

