//! gfx1250 (MI450, CDNA5): the per-instruction tables and the pipeline of the gfx1250 target, built in stages (docs/gfx1250-plan.md):
//! G0 the target plumbing (`--target gfx1250`), G1 the wave32 MIR ops, encoding and both emitters, G2 waits / hazards, G3 TDM / async LDS /
//! split barriers, G5 the hang-safety verifier.

use amdgpu_dsl::Function;

pub mod check;
pub mod cse;
pub mod dse;
pub mod vopd;
pub mod emit;
pub mod hazards;
pub mod matorder;
pub mod msb;
pub mod waits;

thread_local! {
    static XNACK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static PRELOAD: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// The kernarg dwords the hardware preloads into s[2 ..] for the kernel being compiled (`.amdhsa_user_sgpr_kernarg_preload_length`; LLVM
/// does the same for `inreg` kernel arguments on gfx1250, no compatibility prologue): the kernel's kernarg reads at those offsets become
/// SGPR copies instead of scalar loads. Up to 16 dwords (the user SGPRs are s[0:1] + the preload).
pub fn preload_n() -> u32 {
    PRELOAD.with(|c| c.get())
}

/// Whether the kernel runs with XNACK (translation-fault replay) enabled: `kd` fact `xnack` (default off, as our code objects declare it).
/// The address-translation counter (XCNT) only matters then: a replayed memory operation re-reads its operand registers, so they may not
/// be overwritten before its translation is done. LLVM's SIInsertWaitcnts tracks XCNT and waits on it only when
/// `ST.hasWaitXcnt() && ST.isXNACKEnabled()` (the VMEM_GROUP / SMEM_GROUP events are generated only with XNACK).
pub fn xnack() -> bool {
    XNACK.with(|c| c.get())
}

/// Compiles `f` for gfx1250 (G1: the shared lowering / scheduling / allocation, then the gfx1250 emitter; wave32 kernels without
/// divergent control flow, v0..v255, no AGPRs; the waits are the gfx950 ones mapped conservatively onto the split counters).
pub fn compile(f: &Function) -> Result<crate::Compiled, String> {
    compile_with(f, "own")
}

/// The pinned gfx1250 llc directory (backend.gfx1250.lock): `ROCM_LLVM_BIN_GFX1250`, else the MI450 nodes' build.
pub fn llvm_bin() -> std::path::PathBuf {
    std::env::var("ROCM_LLVM_BIN_GFX1250").map(std::path::PathBuf::from).unwrap_or_else(|_| "/data/blyu/llvm-tools/rocm-10.2.0/bin".into())
}

/// Compiles `f` for gfx1250 with the `own` tail (the gfx1250 emitter) or the `llvm-tail` (the allocated MIR through the pinned gfx1250 llc
/// from `gcn-create-vopd`: LLVM computes the waits, hazards and the kernel prologue; our auto waits are not exported).
pub fn compile_with(f: &Function, backend: &str) -> Result<crate::Compiled, String> {
    if f.workgroup_size > 1024 {
        return Err(format!("{}: workgroup size {} (gfx1250: at most 1024 work-items)", f.name, f.workgroup_size));
    }
    let xn = f.kd.iter().rev().find(|(k, _)| k == "xnack").map_or(false, |(_, v)| *v != 0);
    let old = XNACK.with(|c| c.replace(xn));
    // the kernarg preload: the author's (exact: kd `user_sgpr_kernarg_preload_length`, read through `KernargPreload`) or, for the own
    // tail, every kernarg dword up to 16
    let pre = if f.exact || backend != "own" || f.kd.iter().any(|(k, _)| k == "user_sgpr_kernarg_preload_length") { 0 } else { f.kernarg_size.div_ceil(4).min(16) };
    let oldp = PRELOAD.with(|c| c.replace(pre));
    let mut r = compile_inner(f, backend);
    // the preloaded dwords are reserved SGPRs: a kernel that runs out of SGPRs compiles again without the preload (scalar loads instead)
    if pre > 0 && matches!(&r, Err(e) if e.contains("out of SGPRs")) {
        PRELOAD.with(|c| c.set(0));
        r = compile_inner(f, backend);
    }
    PRELOAD.with(|c| c.set(oldp));
    XNACK.with(|c| c.set(old));
    r
}

fn compile_inner(f: &Function, backend: &str) -> Result<crate::Compiled, String> {
    let (mut m, alloc) = crate::lower_and_allocate(f)?;
    // dead writes the allocation exposed (descriptor words rewritten before use; `dse.rs`)
    if backend == "own" && !f.exact {
        dse::dse(&mut m, &alloc);
        dse::tdm_first(&mut m, &alloc);
    }
    // G5: the hang-safety rules are compile errors (docs/gfx1250-hang-rules.md): the kernel IR here, the final instruction text below
    crate::gpu_verify_1250::enforce(f, &m, &alloc)?;
    // the LDS tile protocol (gpu_protocol: completion incl. the tensor operations' TENSORcnt, publication by barriers, slot reuse) of the
    // kernel's declared tiles / rings
    let proto = crate::gpu_verify::verify_ir_protocol(f);
    let errs: Vec<String> = proto.errors().iter().map(|e| format!("[{}] {}", e.code, e.msg)).collect();
    if !errs.is_empty() {
        return Err(format!("{}: tile protocol: {} violation(s)\n  {}", f.name, errs.len(), errs.join("\n  ")));
    }
    // the global / tensor bounds (GV-GLB-*, GV-TDM-*): every global access and every tensor tile must lie inside a declared kernarg extent (a null / unrelated descriptor base is a device hang)
    if !f.exact {
        let g = crate::gpu_verify::analyse(f)?;
        let errs: Vec<String> = g.errors().iter().filter(|e| e.code.starts_with("GV-GLB") || e.code.starts_with("GV-TDM") || e.code.starts_with("GV-LDS")).map(|e| format!("[{}] {}", e.code, e.msg)).collect();
        if !errs.is_empty() {
            return Err(format!("{}: gfx1250 global / tensor bounds: {} violation(s)\n  {}", f.name, errs.len(), errs.join("\n  ")));
        }
    }
    // the compiler's waits, checked with the gfx1250 counters themselves (independent of the shared pass's model)
    if backend == "own" && !m.exact {
        let plan = waits::plan(&m, &alloc);
        let errs = check::verify_planned(&m, &alloc, &plan);
        // (`AMDC_WAIT_CHECK_WARN`: report and continue, to inspect the code of a failing kernel; never for code that runs)
        if !errs.is_empty() && std::env::var("AMDC_WAIT_CHECK_WARN").is_ok() {
            eprintln!("{}: gfx1250 wait self-check (warning only): {} missing wait(s)\n  {}", f.name, errs.len(), errs.join("\n  "));
        } else if !errs.is_empty() {
            return Err(format!("{}: gfx1250 wait self-check: {} missing wait(s) in the compiler's own output\n  {}", f.name, errs.len(), errs.join("\n  ")));
        }
    }
    // VOPD dual issue (opt-in `AMDC_VOPD`): the pairs, before the hazard plan the emitter computes
    vopd::select(&m, &alloc);
    let asm = match backend {
        "own" => emit::emit_kd(&m, &alloc, &f.kd)?,
        "llvm-tail" => {
            let own_only: Vec<String> = m
                .blocks
                .iter()
                .flat_map(|b| b.insts.iter())
                .filter(|i| matches!(i.op, crate::mir::MOp::TensorLoadToLds | crate::mir::MOp::TensorStoreFromLds | crate::mir::MOp::SWaitTensor | crate::mir::MOp::SWaitCtr | crate::mir::MOp::SBarrierSignal | crate::mir::MOp::SBarrierWait | crate::mir::MOp::SSetVgprMsb))
                .map(|i| format!("{:?}", i.op))
                .collect();
            if !f.kd.is_empty() {
                return Err(format!("{}: author kernel-descriptor directives (`kd`) are an own-tail (exact) feature", f.name));
            }
            if !own_only.is_empty() {
                return Err(format!("{}: the gfx1250 LLVM tail has no rows for {own_only:?} (use the own tail)", f.name));
            }
            let mut t = crate::backend::LlvmTail::new();
            t.tools = crate::backend::LlvmTools { bin: llvm_bin() };
            t.drop_auto_waits = true;
            let mir = t.export_mir(&m, &alloc)?;
            let asm = t.run_llc(&m.name, &mir, crate::backend::llvm20::START_PASS_PREFERENCE[0])?;
            crate::backend::declare_unnatural_args(&asm, &m)
        }
        b => return Err(format!("unknown backend `{b}` (own, llvm-tail)")),
    };
    // the relaxed scheduling mode is allowed only as the exact-mode statement the exact verifier models (check.rs); any other write of the
    // mode register stays a GV-1250-SCHED-MODE error
    let modeled = f.exact && f.stmts().iter().any(|s| matches!(s, amdgpu_dsl::Stmt::SetSchedMode { .. }));
    if modeled {
        crate::gpu_verify_1250::with_rules_disabled(&["GV-1250-SCHED-MODE"], || crate::gpu_verify_1250::enforce_asm(&f.name, &asm))?;
    } else {
        crate::gpu_verify_1250::enforce_asm(&f.name, &asm)?;
    }
    Ok(crate::Compiled { asm, vgpr: alloc.max_v, sgpr: alloc.max_s })
}

/// Test hook: the ABI-input check of `f` compiled with the own tail's preload, with the start sequence's scratch pair `pair` (None: the
/// emitter's choice). The mutation tests pass the pair of the 2026-11-18 hang.
pub fn start_sequence_check(f: &Function, pair: Option<u32>) -> Result<(), String> {
    crate::target::with_target(crate::target::by_name("gfx1250").ok_or("gfx1250")?, || {
        let pre = f.kernarg_size.div_ceil(4).min(16);
        let oldp = PRELOAD.with(|c| c.replace(pre));
        let r = crate::lower_and_allocate(f).and_then(|(m, a)| emit::abi_inputs_intact(&m, &a, pair.unwrap_or((2 + pre).next_multiple_of(2).max(6)), pre));
        PRELOAD.with(|c| c.set(oldp));
        r
    })
}
