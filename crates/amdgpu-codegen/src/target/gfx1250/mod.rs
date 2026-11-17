//! gfx1250 (MI450, CDNA5): the per-instruction tables and the pipeline of the gfx1250 target, built in stages (docs/gfx1250-plan.md):
//! G0 the target plumbing (`--target gfx1250`), G1 the wave32 MIR ops, encoding and both emitters, G2 waits / hazards, G3 TDM / async LDS /
//! split barriers, G5 the hang-safety verifier.

use amdgpu_dsl::Function;

pub mod check;
pub mod emit;
pub mod hazards;
pub mod waits;

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
    let (m, alloc) = crate::lower_and_allocate(f)?;
    // G5: the hang-safety rules are compile errors (docs/gfx1250-hang-rules.md): the kernel IR here, the final instruction text below
    crate::gpu_verify_1250::enforce(f, &m, &alloc)?;
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
