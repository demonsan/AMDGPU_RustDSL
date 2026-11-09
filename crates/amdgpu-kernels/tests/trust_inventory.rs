//! The trust surface of the GPU-semantics verifier made explicit (docs/trust-inventory.md): every escape is either an error, a counted
//! ledger entry, or a counted "bounds-checked, not race-checked" access - never silent. Each test pins one entry of the inventory.

use amdgpu_compiler::gpu_verify::{verify_ir, Severity};
use amdgpu_dsl::{CmpPred, CmpTy, DType, Ext, FnBuilder, Function, Op, Opnd};
use amdgpu_kernels::kernel_by_name;

fn no_errors(r: &amdgpu_compiler::gpu_verify::Report) -> bool {
    r.findings.iter().all(|f| f.severity != Severity::Error)
}

/// Wave 0 writes an LDS word, wave 1 reads it, no barrier and no tile markers: a race the protocol engine cannot see (it only checks
/// marked tiles). It must be COUNTED as untracked LDS, not pass silently.
fn unmarked_lds_race() -> Function {
    let mut b = FnBuilder::new("trust_lds_untracked", 128);
    b.lds(256);
    let t = b.let_(Op::WorkitemId);
    let wid = b.let_(Op::LshrI32(Opnd::Val(t), Opnd::I32(6)));
    let w0 = b.cmp(CmpPred::Lt, CmpTy::I32, wid, 1);
    let a = b.c_i32(0);
    let v = b.c_i32(7);
    let any = b.any_lane(w0);
    b.begin_if(Opnd::Val(any));
    b.store_lds(a, v, amdgpu_dsl::MemTy::B32, false);
    b.end_if();
    let _r = b.load_lds(a, amdgpu_dsl::MemTy::B32);
    b.finish()
}

#[test]
fn unmarked_lds_accesses_are_counted_not_silently_trusted() {
    let r = verify_ir(&unmarked_lds_race());
    assert!(no_errors(&r), "{}", r.render());
    assert!(r.unchecked.get("lds-untracked").copied().unwrap_or(0) >= 2, "{:?}", r.unchecked);
}

/// Every workgroup stores to the same global word: in bounds, but the cross-workgroup disjointness of global stores is trusted. Counted.
fn same_address_global_stores() -> Function {
    let mut b = FnBuilder::new("trust_global_untracked", 64);
    b.arg("out", DType::Ptr, 0, 8);
    b.grid_is(0, Ext::sym("N")); // N workgroups (an undefined axis the kernel never reads counts as one workgroup)
    b.arg_tensor("out", Ext::c(4));
    let p = b.let_(Op::KernargPtr { offset: 0 });
    let v = b.c_i32(1);
    b.store_global(p, 0, v, amdgpu_dsl::MemTy::B32, false);
    b.finish()
}

#[test]
fn global_stores_outside_regions_are_counted() {
    let r = verify_ir(&same_address_global_stores());
    assert!(no_errors(&r), "{}", r.render());
    assert_eq!(r.unchecked.get("global-write-untracked").copied(), Some(1), "{:?}", r.unchecked);
}

#[test]
fn safe_asm_blocks_are_in_the_ledger() {
    let f = kernel_by_name("probe_safeasm_e2e").expect("registered");
    let r = verify_ir(&f);
    assert_eq!(r.ledger().asm_opaque, 1, "{}", r.render());
    assert!(r.ledger().total() >= 1);
}

#[test]
fn a_false_trusted_assume_is_counted_not_proven() {
    // gv_attack_assume_trusted: a loop counter claimed to stay 0 although the loop runs further; the verifier cannot refute the trusted claim,
    // so the certificate must carry it in the ledger (never as "proven")
    let f = kernel_by_name("gv_attack_assume_trusted").expect("registered");
    let r = verify_ir(&f);
    assert!(r.ledger().assume_trusted >= 1, "{}", r.render());
}

/// `out[wg * stride + lane * 4]` (one dword per lane, grid x = N workgroups, 64 lanes): with stride >= 256 no two workgroups share a byte.
fn per_wg_stores(stride: i32, use_wg: bool) -> Function {
    let mut b = FnBuilder::new("trust_gw", 64);
    b.arg("out", DType::Ptr, 0, 8);
    b.grid_is(0, Ext::sym("N"));
    b.arg_tensor("out", Ext::sym("N") * Ext::c(1024));
    let p = b.let_(Op::KernargPtr { offset: 0 });
    let lane = b.let_(Op::WorkitemId);
    let wg = b.let_(Op::WorkgroupIdX);
    let l4 = b.let_(Op::ShlI32(Opnd::Val(lane), Opnd::I32(2)));
    let off = if use_wg {
        let w = b.let_(Op::MulI32(Opnd::Val(wg), Opnd::I32(stride)));
        b.let_(Op::AddI32(Opnd::Val(w), Opnd::Val(l4)))
    } else {
        l4
    };
    let q = b.let_(Op::PtrAdd { base: Opnd::Val(p), byte_off: Opnd::Val(off) });
    let v = b.c_i32(1);
    b.store_global(q, 0, v, amdgpu_dsl::MemTy::B32, false);
    b.finish()
}

#[test]
fn disjoint_per_workgroup_stores_are_proven() {
    let r = verify_ir(&per_wg_stores(256, true));
    assert!(no_errors(&r), "{}", r.render());
    assert_eq!(r.proven.get("global-write-disjoint").copied(), Some(1), "{:?} {:?}", r.proven, r.unchecked);
    assert!(r.unchecked.get("global-write-untracked").is_none(), "{:?}", r.unchecked);
}

#[test]
fn overlapping_workgroup_stores_are_not_proven() {
    // stride 128 < 64 lanes x 4 bytes: workgroup w's upper half is workgroup w+1's lower half
    let r = verify_ir(&per_wg_stores(128, true));
    assert!(r.proven.get("global-write-disjoint").is_none(), "{:?}", r.proven);
    assert_eq!(r.unchecked.get("global-write-untracked").copied(), Some(1));
}

#[test]
fn stores_ignoring_a_multi_workgroup_axis_are_not_proven() {
    // every workgroup of the N along x writes the same 256 bytes
    let r = verify_ir(&per_wg_stores(256, false));
    assert!(r.proven.get("global-write-disjoint").is_none(), "{:?}", r.proven);
    assert_eq!(r.unchecked.get("global-write-untracked").copied(), Some(1));
}
