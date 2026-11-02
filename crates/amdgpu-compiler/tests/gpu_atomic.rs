//! Atomics and buffer descriptors through `gpu_verify`.
//!
//! Atom is a read-modify-write: the compiler checks it like a *store* (against the logical extent, never the
//! zeroed slack), and the returned old value is a VMEM result that must be waited for on **every** control-flow
//! path (path-sensitive `verify_waits`). Buffer descriptors must not overstate their allocation (this file
//! duplicates the descriptor cases for the atomic/base combination; the main descriptor cases live in
//! `gpu_semantics.rs`).

use amdgpu_compiler::gpu_verify::{verify_ir, verify_mir, Report, Severity};
use amdgpu_compiler::mir::{Block, MFunction, MInst, MOp, RClass, Term};
use amdgpu_compiler::regalloc::Alloc;
use amdgpu_dsl::{DType, Ext, FnBuilder, Function, Op};

fn has(rep: &Report, code: &str) -> bool {
    rep.findings.iter().any(|f| f.code == code && f.severity == Severity::Error)
}

fn errors(rep: &Report) -> String {
    rep.errors().iter().map(|e| format!("[{}] {}", e.code, e.msg)).collect::<Vec<_>>().join("\n")
}

/// `out[i] += 1` through `global_atomic_add` (device scope, returning the old value).
/// `contract`: none | ext(height) | slack(logical, alloc); `off` is a constant byte offset.
fn atomic_kernel(contract: &str, off: u32) -> Function {
    let mut b = FnBuilder::new("atomic_probe", 64);
    b.arg("p", DType::Ptr, 0, 8);
    b.arg("n", DType::I32, 8, 4);
    match contract {
        "ext64" => b.arg_tensor("p", Ext::c(64)),
        "ext8_alloc64" => b.arg_tensor_slack("p", Ext::c(8), Ext::c(64)),
        _ => {}
    }
    let p = b.let_(Op::KernargPtr { offset: 0 });
    let v = b.c_i32(1);
    b.atomic_add_u32(p, off, v);
    b.finish()
}

#[test]
fn atomic_inside_the_declared_extent_is_proven() {
    let r = verify_ir(&atomic_kernel("ext64", 0));
    assert!(r.errors().is_empty(), "{}", errors(&r));
    assert_eq!(r.proven.get("global"), Some(&1), "the atomic must count as a proven global access");
}

#[test]
fn atomic_without_a_declared_extent_is_rejected() {
    let r = verify_ir(&atomic_kernel("none", 0));
    assert!(has(&r, "GV-GLB-NOEXT"), "{}", r.render());
}

#[test]
fn atomic_past_the_declared_extent_is_rejected() {
    let r = verify_ir(&atomic_kernel("ext64", 64));
    assert!(has(&r, "GV-GLB-OOB"), "{}", r.render());
}

#[test]
fn atomic_is_checked_against_the_logical_extent_not_the_allocated_slack() {
    // logical 8 bytes, allocated 64: an atomic at 32 is inside the allocation but outside the tensor.
    let r = verify_ir(&atomic_kernel("ext8_alloc64", 32));
    assert!(has(&r, "GV-GLB-OOB"), "{}", r.render());
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// MIR: the returned old value is a VMEM result (vmcnt), so it needs a wait on every path

fn mfunc(blocks: Vec<(Vec<MInst>, Term)>, classes: Vec<RClass>) -> MFunction {
    let mut bs = Vec::new();
    for (i, (insts, term)) in blocks.into_iter().enumerate() {
        let mut b = Block::new(i as u32);
        b.insts = insts;
        b.term = term;
        bs.push(b);
    }
    MFunction {
        name: "gv".into(),
        args: Vec::new(),
        kernarg_size: 0,
        workgroup_size: 64,
        lds_bytes: 0,
        blocks: bs,
        vregs: classes,
        loops: Vec::new(),
        alias: Default::default(),
        elide_fallthrough: false,
    }
}

fn alloc(phys: Vec<u32>, classes: &[RClass]) -> Alloc {
    Alloc { phys, max_v: 64, max_s: 24, scalar: classes.iter().map(|c| !c.is_vector()).collect() }
}

fn dummy() -> Function {
    FnBuilder::new("dummy", 64).finish()
}

/// 0 = V2 base, 1 = V1 value, 2 = V1 old value (atomic result), 3 = V1 consumer of the result.
/// The consumer sits after a branch; `wait` decides whether that path drains the VMEM queue first.
fn atomic_result_cfg(wait_on_taken: bool, wait_on_fall: bool) -> Report {
    let classes = vec![RClass::V2, RClass::V1, RClass::V1, RClass::V1];
    let atomic = MInst::new(MOp::GlobalAtomicAddRet).def(2).srcreg(0).srcreg(1);
    let mut then_arm = Vec::new();
    if wait_on_taken {
        then_arm.push(MInst::new(MOp::SWaitVm).imm(0));
    }
    let mut else_arm = Vec::new();
    if wait_on_fall {
        else_arm.push(MInst::new(MOp::SWaitVm).imm(0));
    }
    let m = mfunc(
        vec![
            (vec![atomic], Term::CondGoto { on: Pred::Vccnz, target: 1, fall: 2 }),
            (then_arm, Term::Goto(3)),
            (else_arm, Term::Goto(3)),
            (vec![MInst::new(MOp::VMovB32).def(3).srcreg(2)], Term::Endpgm),
        ],
        classes,
    );
    verify_mir(&dummy(), &m, &alloc(vec![2, 4, 6, 8], &[RClass::V2, RClass::V1, RClass::V1, RClass::V1]))
}

#[test]
fn a_wait_on_every_path_after_an_atomic_is_accepted() {
    let r = atomic_result_cfg(true, true);
    assert!(r.errors().is_empty(), "{}", errors(&r));
}

#[test]
fn using_the_atomic_result_before_a_wait_is_rejected() {
    let r = atomic_result_cfg(true, false);
    assert!(has(&r, "GV-WAIT"), "{}", r.render());
    let r = atomic_result_cfg(false, true);
    assert!(has(&r, "GV-WAIT"), "{}", r.render());
}
