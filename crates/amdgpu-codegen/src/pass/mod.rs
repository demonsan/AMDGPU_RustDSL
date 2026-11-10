//! The pass manager (docs/pass-catalog.md section 5): named passes with their stage, target dependence, gate and evidence, the gfx950 pipeline
//! as ordered lists, and the two retry ladders as data. `lower_and_allocate` (lib.rs) runs these lists; `amdc passes --target gfx950` prints them.
//!
//! The MIR bodies keep today's two forms exactly (docs/pass-catalog.md section 5.3 item 3): the SCHEDULED body (one inner-ladder rung) and the
//! UNSCHEDULED fallback differ in the scheduler passes, the `strict` flag of `hazard-value` / `hz-fill`, where `verify-waits` runs and the
//! launch-budget check inside the rung.

use crate::mir::MFunction;
use crate::regalloc::Alloc;
use crate::{hazards, hzfill, mir, pin_check, regalloc, sched};

/// What backs a pass's correctness.
#[derive(Clone, Copy, Debug)]
pub enum Evidence {
    /// a cargo test (file under crates/*/tests or a module's unit tests)
    Test(&'static str),
    /// a hardware probe (crates/amdgpu-kernels/src/probes, tools/*probe*.sh)
    Probe(&'static str),
    /// compared against the pinned LLVM (tools/hazard_oracle.sh, tools/llvm_compare.sh, ...)
    Oracle(&'static str),
    /// a measured note under docs/
    Doc(&'static str),
    /// no specific evidence; covered indirectly (stated)
    Indirect(&'static str),
}

/// How a pass depends on the target (docs/pass-catalog.md section 2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetDep {
    Generic,
    /// generic algorithm over target parameters (amdgpu_target)
    Param,
    /// rules or instructions of the target itself
    Only,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Source,
    Lower,
    Mir,
    Sched,
    Ra,
    Post,
    Verify,
    Emit,
}

/// One pass's description.
#[derive(Clone, Copy, Debug)]
pub struct PassInfo {
    pub name: &'static str,
    pub stage: Stage,
    pub dep: TargetDep,
    /// when it runs
    pub gate: &'static str,
    pub evidence: &'static [Evidence],
}

/// Which MIR body is running.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Path {
    Scheduled,
    Unscheduled,
}

/// What the MIR passes read besides the MIR.
pub struct PassCtx<'a> {
    pub f: &'a amdgpu_dsl::Function,
    pub path: Path,
    /// this rung's scheduler options (scheduled path)
    pub sched: Option<&'a sched::Options>,
    /// the kernel's scheduler options as `hz-fill` reads them
    pub hz_opts: Option<&'a sched::Options>,
    pub hzfill: bool,
    pub m0_reuse: bool,
}

/// The state the MIR passes transform: the MIR and, after `regalloc`, the assignment.
pub struct MirState {
    pub m: MFunction,
    pub alloc: Option<Alloc>,
    /// the scheduler passes' counters (printed with the `stats` token)
    pub sched_stats: sched::Stats,
}

impl MirState {
    pub fn new(m: MFunction) -> MirState {
        MirState { m, alloc: None, sched_stats: sched::Stats::default() }
    }
}

impl MirState {
    fn alloc(&self) -> &Alloc {
        self.alloc.as_ref().expect("a post-RA pass before regalloc")
    }
}

pub type RunFn = fn(&mut MirState, &PassCtx) -> Result<(), String>;

/// A runnable MIR pass.
#[derive(Clone, Copy)]
pub struct Pass {
    pub info: PassInfo,
    pub run: RunFn,
}

pub(crate) const fn info(name: &'static str, stage: Stage, dep: TargetDep, gate: &'static str, evidence: &'static [Evidence]) -> PassInfo {
    PassInfo { name, stage, dep, gate, evidence }
}

use Evidence::*;
use Stage::*;
use TargetDep::*;

// ---------------------------------------------------------------------------------------------------------------- source / lowering (descriptions)

/// The source-IR and lowering steps (run by `lower::lower` / `agpr::assign` inside the outer ladder; listed for `amdc passes`).
pub const FRONT: &[PassInfo] = &[
    info("verify-ir-structure", Source, Generic, "always", &[Test("crates/amdgpu-codegen/tests/bringup_guards.rs"), Test("crates/amdgpu-codegen/tests/uniform_select.rs")]),
    info("strip-any-lane", Source, Generic, "when any_lane markers are present", &[Indirect("every kernel using any_lane; probe_anylane")]),
    info("analysis-uniform-values", Source, Param, "lowering token scalarize", &[Test("crates/amdgpu-codegen/tests/uniform_select.rs"), Test("crates/amdgpu-codegen/tests/scalarize2.rs")]),
    info("lower-mir", Lower, Only, "always; tokens = policy + ladder rung + Function::lower_opts", &[Test("crates/amdgpu-kernels/tests/mir_invariants.rs"), Test("crates/amdgpu-kernels/tests/tail_invariance.rs"), Oracle("tools/llvm_compare.sh"), Probe("crates/amdgpu-kernels/src/probes/tileops.rs")]),
    info("assign-agpr", Mir, Only, "always (no-op without AGPR marks)", &[Test("crates/amdgpu-kernels/tests/agpr.rs"), Probe("tools/agpr_probe.sh"), Doc("docs/agpr.md")]),
];

// ---------------------------------------------------------------------------------------------------------------- MIR passes

// the scheduler, step by step (exactly `sched::run`, which stays as the one-call form for its unit tests)
fn so<'a>(cx: &'a PassCtx) -> &'a sched::Options {
    cx.sched.expect("the scheduled path has options")
}
fn run_cfg_cleanup(s: &mut MirState, cx: &PassCtx) -> Result<(), String> {
    if so(cx).cfg_cleanup {
        sched::cfg::cleanup(&mut s.m);
    }
    Ok(())
}
fn run_strip_auto(s: &mut MirState, _: &PassCtx) -> Result<(), String> {
    sched::waits::strip_auto_waits(&mut s.m);
    Ok(())
}
fn run_dead_selects(s: &mut MirState, _: &PassCtx) -> Result<(), String> {
    sched::cfg::dce_dead_selects(&mut s.m);
    Ok(())
}
fn run_ldshoist(s: &mut MirState, cx: &PassCtx) -> Result<(), String> {
    let o = so(cx);
    if o.any_region(|r| r.ldshoist) {
        let n = sched::hoist::run(&mut s.m, o);
        if o.stats {
            eprintln!("ldshoist: {n} loads hoisted");
        }
    }
    Ok(())
}
fn run_slotfill(s: &mut MirState, cx: &PassCtx) -> Result<(), String> {
    let o = so(cx);
    if o.any_region(|r| r.slotfill) {
        let n = sched::slotfill::run(&mut s.m, o)?;
        if o.stats {
            eprintln!("slotfill: {n} instructions moved");
        }
    }
    Ok(())
}
fn run_phase(s: &mut MirState, cx: &PassCtx) -> Result<(), String> {
    // `sched_group_barrier` phases (blocks with directives; the directives are removed)
    let o = so(cx);
    let phased = sched::phase::run(&mut s.m, o)?;
    if o.stats && phased > 0 {
        eprintln!("phase: {phased} instructions moved");
    }
    Ok(())
}
fn run_list(s: &mut MirState, cx: &PassCtx) -> Result<(), String> {
    sched::schedule_blocks(&mut s.m, so(cx), &mut s.sched_stats)
}
fn run_wsink(s: &mut MirState, cx: &PassCtx) -> Result<(), String> {
    let o = so(cx);
    if o.any_region(|r| r.waitsink) {
        let n = sched::ldahead::sink_explicit_waits(&mut s.m, o);
        if o.stats {
            eprintln!("wsink: {n} explicit waits moved");
        }
    }
    Ok(())
}
fn run_ldahead(s: &mut MirState, cx: &PassCtx) -> Result<(), String> {
    let o = so(cx);
    if o.any_region(|r| r.ldahead32 > 0 || r.ldsink > 0) {
        let n = sched::ldahead::run(&mut s.m, o);
        if o.stats {
            eprintln!("ldahead: {n} blocks changed");
        }
    }
    Ok(())
}
fn run_insert_waits(s: &mut MirState, _: &PassCtx) -> Result<(), String> {
    s.sched_stats.waits_inserted = sched::waits::insert_waits(&mut s.m);
    Ok(())
}
fn run_sched_verify_waits(s: &mut MirState, cx: &PassCtx) -> Result<(), String> {
    let o = so(cx);
    if o.verify {
        sched::waits::verify_waits(&s.m).map_err(|e| format!("wait verification failed: {e}"))?;
    }
    if o.stats {
        eprintln!("sched: {:?}", s.sched_stats);
    }
    Ok(())
}
fn run_phase_strip(s: &mut MirState, _: &PassCtx) -> Result<(), String> {
    // scheduler directives are meaningless without the scheduler
    sched::phase::strip(&mut s.m);
    Ok(())
}
fn run_m0_reuse(s: &mut MirState, cx: &PassCtx) -> Result<(), String> {
    if cx.m0_reuse {
        sched::m0::reuse(&mut s.m);
    }
    Ok(())
}
fn run_m0_explicit(s: &mut MirState, _: &PassCtx) -> Result<(), String> {
    sched::m0::explicit(&mut s.m);
    Ok(())
}
fn run_pin_check(s: &mut MirState, _: &PassCtx) -> Result<(), String> {
    let pe = pin_check::check(&s.m);
    if !pe.is_empty() {
        return Err(format!("pinned (safe-asm) blocks fail the final MIR check: {}", pe.join("; ")));
    }
    Ok(())
}
fn run_hazard_value(s: &mut MirState, cx: &PassCtx) -> Result<(), String> {
    match cx.path {
        Path::Scheduled => hazards::fix_value_hazards_opts(&mut s.m, true),
        Path::Unscheduled => hazards::fix_value_hazards(&mut s.m),
    }
    Ok(())
}
fn run_mir_verify(s: &mut MirState, _: &PassCtx) -> Result<(), String> {
    let errs = mir::verify(&s.m);
    if !errs.is_empty() {
        return Err(format!("MIR verification failed:\n  {}", errs.join("\n  ")));
    }
    Ok(())
}
fn run_verify_waits(s: &mut MirState, _: &PassCtx) -> Result<(), String> {
    // the unscheduled path trusted its lowering-time waits; passes after lowering (LDS spill, ...) can invalidate them: re-check on every path
    sched::waits::verify_waits(&s.m).map_err(|e| format!("wait verification failed: {e}"))
}
fn run_regalloc(s: &mut MirState, _: &PassCtx) -> Result<(), String> {
    s.alloc = Some(regalloc::allocate(&s.m)?);
    Ok(())
}
fn run_budget_rung(s: &mut MirState, _: &PassCtx) -> Result<(), String> {
    // over the workgroup's launch budget: like running out of VGPRs (try the next, tighter attempt)
    regalloc::check_workgroup_budget(&s.m, s.alloc()).map_err(|e| format!("out of VGPRs for the launch: {e}"))
}
fn run_inflight_waw(s: &mut MirState, _: &PassCtx) -> Result<(), String> {
    let a = s.alloc.take().expect("allocated");
    hazards::pad_inflight_waw(&mut s.m, &a);
    s.alloc = Some(a);
    Ok(())
}
fn run_phys_war(s: &mut MirState, _: &PassCtx) -> Result<(), String> {
    let a = s.alloc.take().expect("allocated");
    hazards::pad_physical_war(&mut s.m, &a);
    s.alloc = Some(a);
    Ok(())
}
fn run_hzfill(s: &mut MirState, cx: &PassCtx) -> Result<(), String> {
    if cx.hzfill {
        let a = s.alloc.take().expect("allocated");
        let st = hzfill::fill_with(&mut s.m, &a, cx.path == Path::Scheduled, cx.hz_opts);
        s.alloc = Some(a);
        if std::env::var("AMDC_SCHED_VERBOSE").is_ok() {
            eprintln!("hzfill {}: s_nop wait states {} -> {} ({} moves, {} rejected)", cx.f.name, st.states_before, st.states_after, st.moved, st.rejected);
        }
    }
    Ok(())
}
fn run_m0_pad(s: &mut MirState, _: &PassCtx) -> Result<(), String> {
    sched::m0::pad(&mut s.m);
    Ok(())
}
fn run_valu_sgpr(s: &mut MirState, _: &PassCtx) -> Result<(), String> {
    let a = s.alloc.take().expect("allocated");
    hazards::pad_valu_sgpr(&mut s.m, &a);
    s.alloc = Some(a);
    Ok(())
}

const SCHED_CFG: Pass = Pass { info: info("sched-cfg-cleanup", Sched, Generic, "scheduler (not token nocfg)", &[Indirect("sched_kernels.rs: every kernel schedules and verifies")]), run: run_cfg_cleanup };
const SCHED_STRIP: Pass = Pass { info: info("waits-strip-auto", Sched, Generic, "scheduler", &[Test("crates/amdgpu-codegen/tests/wait_operands.rs")]), run: run_strip_auto };
const SCHED_DSEL: Pass = Pass { info: info("sched-dce-dead-selects", Sched, Generic, "scheduler", &[Indirect("sched_kernels.rs; md5 manifest")]), run: run_dead_selects };
const SCHED_HOIST: Pass = Pass { info: info("sched-ldshoist", Sched, Param, "token ldshoist (GEMM preset; ladder rungs cap hoist=N)", &[Test("crates/amdgpu-kernels/tests/sched_kernels.rs"), Test("crates/amdgpu-kernels/tests/sched_preset.rs"), Doc("docs/scheduling.md")]), run: run_ldshoist };
const SCHED_SLOT: Pass = Pass { info: info("sched-slotfill", Sched, Only, "token slotfill", &[Test("crates/amdgpu-codegen/tests/slotfill.rs"), Doc("docs/slotfill.md")]), run: run_slotfill };
const SCHED_PHASE: Pass = Pass { info: info("sched-phase", Sched, Param, "sched_group_barrier markers (always consumed)", &[Test("crates/amdgpu-codegen/tests/sched_phase.rs")]), run: run_phase };
const SCHED_LIST: Pass = Pass {
    info: info("sched-list", Sched, Param, "tokens reorder / rewait / xbar / limit=N (per region)", &[Test("crates/amdgpu-codegen/src/sched/tests.rs"), Test("crates/amdgpu-kernels/tests/sched_kernels.rs"), Test("crates/amdgpu-codegen/tests/sched_regions.rs"), Doc("docs/scheduling.md"), Probe("tools/sched_mb/")]),
    run: run_list,
};
const SCHED_WSINK: Pass = Pass { info: info("sched-wsink", Sched, Only, "token wsink", &[Test("crates/amdgpu-codegen/tests/ldahead.rs"), Doc("docs/lgkm-lookahead.md")]), run: run_wsink };
const SCHED_LDAHEAD: Pass = Pass { info: info("sched-ldahead", Sched, Only, "tokens ldsink=D / ldahead32=D", &[Test("crates/amdgpu-codegen/tests/ldahead.rs"), Doc("docs/lgkm-lookahead.md")]), run: run_ldahead };
const SCHED_WAITS: Pass = Pass { info: info("waitcnt-recompute", Sched, Only, "scheduler", &[Test("crates/amdgpu-codegen/tests/wait_operands.rs"), Test("crates/amdgpu-codegen/tests/inflight_waw.rs"), Doc("docs/scheduling.md")]), run: run_insert_waits };
const SCHED_VERIFY: Pass = Pass { info: info("verify-waits", Verify, Only, "scheduler (not token noverify)", &[Test("crates/amdgpu-codegen/tests/wait_operands.rs"), Test("crates/amdgpu-kernels/tests/gpu_semantics.rs")]), run: run_sched_verify_waits };
const PHASE_STRIP: Pass = Pass { info: info("sched-phase-strip", Mir, Generic, "unscheduled path", &[Test("crates/amdgpu-codegen/tests/sched_phase.rs")]), run: run_phase_strip };
const M0_REUSE: Pass = Pass { info: info("m0-reuse", Mir, Only, "policy m0_reuse (GEMM profile)", &[Indirect("md5 manifest; dma_hazards.rs mentions M0")]), run: run_m0_reuse };
const M0_EXPLICIT: Pass = Pass { info: info("m0-explicit", Mir, Only, "always", &[Indirect("md5 manifest; dispatch_compile.rs (M0 dangling-use regression)")]), run: run_m0_explicit };
const PIN_CHECK: Pass = Pass { info: info("verify-pinned-blocks", Verify, Only, "always", &[Test("crates/amdgpu-codegen/tests/pin_check.rs"), Test("crates/amdgpu-kernels/tests/safeasm.rs")]), run: run_pin_check };
const HAZARD_VALUE: Pass = Pass {
    info: info("hazard-value", Mir, Only, "always; strict (block-end padding) on the scheduled path", &[Test("crates/amdgpu-codegen/tests/hazard_classes.rs"), Test("crates/amdgpu-codegen/tests/hazard_oracle.rs"), Oracle("tools/hazard_oracle.sh"), Probe("tools/mfma_dep.sh")]),
    run: run_hazard_value,
};
const MIR_VERIFY: Pass = Pass { info: info("verify-mir-structure", Verify, Param, "always", &[Test("crates/amdgpu-kernels/tests/sched_kernels.rs"), Test("crates/amdgpu-codegen/tests/smfmac_hazards.rs")]), run: run_mir_verify };
const VERIFY_WAITS: Pass = Pass { info: info("verify-waits", Verify, Only, "unscheduled path (after verify-mir-structure)", &[Test("crates/amdgpu-codegen/tests/wait_operands.rs"), Test("crates/amdgpu-kernels/tests/gpu_semantics.rs")]), run: run_verify_waits };
const REGALLOC: Pass = Pass {
    info: info("regalloc", Ra, Param, "always (scan -> scan-top -> two-phase; regplan-check when the kernel has register marks)", &[Test("crates/amdgpu-codegen/tests/ra_reaching.rs"), Test("crates/amdgpu-kernels/tests/register_accounting.rs"), Test("crates/amdgpu-codegen/tests/regplan.rs")]),
    run: run_regalloc,
};
const BUDGET_RUNG: Pass = Pass { info: info("check-workgroup-budget", Verify, Only, "scheduled path (retryable: the next rung)", &[Test("crates/amdgpu-kernels/tests/register_accounting.rs")]), run: run_budget_rung };
const INFLIGHT_WAW: Pass = Pass { info: info("hazard-inflight-waw", Post, Only, "always", &[Test("crates/amdgpu-codegen/tests/inflight_waw.rs")]), run: run_inflight_waw };
const PHYS_WAR: Pass = Pass { info: info("hazard-phys-war", Post, Only, "always", &[Test("crates/amdgpu-codegen/tests/phys_waw.rs"), Test("crates/amdgpu-kernels/tests/agpr.rs")]), run: run_phys_war };
const HZFILL: Pass = Pass { info: info("hz-fill", Post, Only, "scheduler token hzfill (strict on the scheduled path)", &[Test("crates/amdgpu-kernels/tests/hzfill.rs")]), run: run_hzfill };
const M0_PAD: Pass = Pass { info: info("m0-pad", Post, Only, "always", &[Indirect("md5 manifest")]), run: run_m0_pad };
const VALU_SGPR: Pass = Pass {
    info: info("hazard-valu-sgpr", Post, Only, "always (not switchable)", &[Test("crates/amdgpu-kernels/tests/hazard_valu_sgpr.rs"), Probe("probe_valu_sgpr"), Oracle("tools/hazard_oracle.sh"), Doc("docs/trust-inventory.md")]),
    run: run_valu_sgpr,
};

/// The scheduled body (one inner-ladder rung, on a clone of the post-AGPR MIR).
pub const SCHEDULED: &[Pass] = &[SCHED_CFG, SCHED_STRIP, SCHED_DSEL, SCHED_HOIST, SCHED_SLOT, SCHED_PHASE, SCHED_LIST, SCHED_WSINK, SCHED_LDAHEAD, SCHED_WAITS, SCHED_VERIFY, M0_REUSE, M0_EXPLICIT, PIN_CHECK, HAZARD_VALUE, MIR_VERIFY, REGALLOC, BUDGET_RUNG, INFLIGHT_WAW, PHYS_WAR, HZFILL, M0_PAD, VALU_SGPR];

/// The unscheduled body (no scheduler, or every inner rung failed).
pub const UNSCHEDULED: &[Pass] = &[PHASE_STRIP, M0_REUSE, M0_EXPLICIT, PIN_CHECK, HAZARD_VALUE, MIR_VERIFY, VERIFY_WAITS, REGALLOC, INFLIGHT_WAW, PHYS_WAR, HZFILL, M0_PAD, VALU_SGPR];

/// The final checks and the tail (compile_with).
pub const BACK: &[PassInfo] = &[
    info("check-workgroup-budget", Verify, Only, "always (hard error)", &[Test("crates/amdgpu-kernels/tests/register_accounting.rs")]),
    info("verify-gpu", Verify, Param, "AMDC_GPU_VERIFY (default: MIR register / wait errors fail)", &[Test("crates/amdgpu-kernels/tests/gpu_semantics.rs"), Test("crates/amdgpu-kernels/tests/trust_inventory.rs"), Doc("docs/gpu-semantics-verifier.md")]),
    info("emit-own | emit-llvm-tail", Emit, Only, "--backend own | llvm-tail | auto", &[Test("crates/amdgpu-kernels/tests/asm_assembles.rs"), Test("crates/amdgpu-kernels/tests/tail_invariance.rs"), Test("crates/amdc/tests/md5_manifest.rs"), Oracle("tools/llvm_conformance.sh")]),
];

/// Runs a body.
pub fn run_body(body: &[Pass], s: &mut MirState, cx: &PassCtx) -> Result<(), String> {
    for p in body {
        (p.run)(s, cx)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------------------------- retry ladders

/// The outer ladder: lowering tokens added per rung; retried when the error contains `out of VGPRs`.
pub const OUTER_LADDER: &[&[&str]] = &[
    &[],
    &["remat1"],
    &["remat2"],
    &["remat3"],
    &["remat4"],
    &["imm", "ldsspill8"],
    &["imm", "ldsspill16"],
    &["imm", "ldsspill24"],
    &["imm", "ldsspill32"],
    &["imm", "ldsspill48"],
    &["imm", "ldsspill64"],
];

/// The inner ladder's rungs for the given scheduler options: (pressure cut, hoist distance cap). First shrink the LDS-read hoisting (it lengthens
/// live ranges and fragments the register file), then the pressure limit, then the launch budget of a big workgroup; when all fail, the unscheduled body.
pub fn inner_ladder(opts: &sched::Options, workgroup_size: u32) -> Vec<(u32, Option<usize>)> {
    let mut attempts: Vec<(u32, Option<usize>)> = vec![(0, None)];
    if opts.ldshoist {
        attempts.extend([(0, Some(24)), (0, Some(8)), (0, Some(0))]);
    }
    if opts.reorder {
        attempts.extend([(16, Some(0)), (32, Some(0)), (48, Some(0))]); // the pressure limit only matters for the reordering scheduler
        // a big workgroup: the schedule must also fit the launch budget (ceil(waves / 4) waves per SIMD share 512 registers); a schedule that
        // hoists later loads above earlier work can blow it. Retry against that budget before giving up the scheduler.
        let launch = sched::launch_vgpr_budget(workgroup_size);
        if launch < opts.max_vgprs {
            let c = opts.max_vgprs - launch;
            attempts.extend([(c, Some(0)), (c + 16, Some(0)), (c + 32, Some(0))]);
        }
    }
    attempts
}

/// The retryable error class of both ladders.
pub fn retryable(e: &str) -> bool {
    e.contains("out of VGPRs")
}

/// The text `amdc passes --target <t>` prints.
pub fn describe(target: &str, evidence: bool) -> Result<String, String> {
    if target != crate::target::T.name {
        return Err(format!("no pipeline for target {target} (implemented: {}; gfx1250 is a specification only, docs/target-gfx1250.md)", crate::target::T.name));
    }
    let mut s = format!("pipeline {target}\n");
    let mut n = 0;
    let mut line = |s: &mut String, part: &str, p: &PassInfo| {
        n += 1;
        let ev: Vec<String> = p.evidence.iter().map(|e| match e {
            Test(x) => format!("test:{x}"),
            Probe(x) => format!("probe:{x}"),
            Oracle(x) => format!("oracle:{x}"),
            Doc(x) => format!("doc:{x}"),
            Indirect(x) => format!("indirect:{x}"),
        }).collect();
        let dep = match p.dep {
            Generic => "generic".to_string(),
            Param => format!("{target}-param"),
            Only => format!("{target}-only"),
        };
        s.push_str(&format!("  {n:>3}  {part:<11} {:<7} {:<26} {:<14} {}\n", format!("{:?}", p.stage).to_lowercase(), p.name, dep, p.gate));
        if evidence {
            for e in ev {
                s.push_str(&format!("{:>48}{e}\n", ""));
            }
        }
    };
    for p in FRONT {
        line(&mut s, "front", p);
        if p.name == "lower-mir" {
            for c in crate::lower::CLEANUP {
                line(&mut s, "front", &c.info);
            }
        }
    }
    for p in SCHEDULED {
        line(&mut s, "scheduled", &p.info);
    }
    for p in UNSCHEDULED {
        line(&mut s, "unscheduled", &p.info);
    }
    for p in BACK {
        line(&mut s, "back", p);
    }
    s.push_str("retry ladders\n");
    let rungs: Vec<String> = OUTER_LADDER.iter().map(|r| if r.is_empty() { "-".to_string() } else { r.join("+") }).collect();
    s.push_str(&format!("  outer (lowering tokens, from lower-mir; retry on `out of VGPRs`): {}\n", rungs.join(", ")));
    s.push_str("  inner (scheduled body on a copy; (pressure cut, hoist cap)): (0,-) [+ ldshoist: (0,24) (0,8) (0,0)] [+ reorder: (16,0) (32,0) (48,0) + launch budget c: (c,0) (c+16,0) (c+32,0)]; then the unscheduled body\n");
    Ok(s)
}

#[cfg(test)]
mod tests {
    #[test]
    fn every_pass_has_evidence() {
        for p in super::FRONT.iter().chain(crate::lower::CLEANUP.iter().map(|p| &p.info)).chain(super::SCHEDULED.iter().map(|p| &p.info)).chain(super::UNSCHEDULED.iter().map(|p| &p.info)).chain(super::BACK) {
            assert!(!p.evidence.is_empty(), "{} has no evidence entry", p.name);
        }
    }

    #[test]
    fn evidence_paths_exist() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        for p in super::FRONT.iter().chain(crate::lower::CLEANUP.iter().map(|p| &p.info)).chain(super::SCHEDULED.iter().map(|p| &p.info)).chain(super::UNSCHEDULED.iter().map(|p| &p.info)).chain(super::BACK) {
            for e in p.evidence {
                let path = match e {
                    super::Evidence::Test(x) | super::Evidence::Doc(x) | super::Evidence::Oracle(x) | super::Evidence::Probe(x) => *x,
                    super::Evidence::Indirect(_) => continue,
                };
                if path.contains('/') {
                    assert!(root.join(path).exists(), "{}: evidence {path} does not exist", p.name);
                }
            }
        }
    }
}
