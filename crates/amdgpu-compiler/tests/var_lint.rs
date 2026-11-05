//! Lint for loop-variable re-reads after `set_var` in the same iteration (see `amdgpu_dsl::lint`), plus a fixed
//! description of what it must and must not report.

use amdgpu_compiler::{kernel_by_name, kernel_names};
use amdgpu_dsl::{lint, DType, FnBuilder, Op, Opnd};

/// Kernels that already re-read loop variables after `set_var` (validated on hardware as they are; the lowering updates
/// the register in place). New kernels must be clean or opt out with `FnBuilder::allow_var_rereads()`.
const KNOWN_REREADERS: &[&str] = &[
];

#[test]
fn new_kernels_do_not_reread_loop_variables_after_set() {
    let mut hits = Vec::new();
    let (mut probe_hits, mut known_hits) = (0, 0);
    for k in kernel_names() {
        let f = kernel_by_name(k).unwrap();
        let w = lint(&f);
        if k.starts_with("probe_") {
            probe_hits += w.len();
        } else if KNOWN_REREADERS.contains(&k) {
            known_hits += w.len();
        } else {
            hits.extend(w.into_iter().map(|w| format!("{k}: {w}")));
        }
    }
    println!("re-read reports: {probe_hits} in probes, {known_hits} in known production kernels");
    assert!(hits.is_empty(), "{}", hits.join("\n"));
}
fn counter_kernel(read_after_set: bool, opt_out: bool) -> amdgpu_dsl::Function {
    let mut b = FnBuilder::new("t", 64);
    b.arg("out", DType::Ptr, 0, 8);
    if opt_out {
        b.allow_var_rereads();
    }
    let c0 = b.c_i32(0);
    let c1 = b.c_i32(1);
    let c10 = b.c_i32(10);
    let vars = b.begin_loop(&[(DType::I32, Opnd::Val(c0)), (DType::I32, Opnd::Val(c0))]);
    let (x, i) = (vars[0], vars[1]);
    if !read_after_set {
        // common case: read first (value from the iteration start), then set
        let _y = b.add_i32(Opnd::Var(x), Opnd::Val(c1));
    }
    let nx = b.add_i32(Opnd::Var(i), Opnd::Val(c1));
    b.set_var(x, Opnd::Val(nx));
    if read_after_set {
        let _r = b.add_i32(Opnd::Var(x), Opnd::Val(c10));
    }
    // loop condition / back edge on SSA values, not on the variable
    let inext = b.add_i32(Opnd::Var(i), Opnd::Val(c1));
    b.set_var(i, Opnd::Val(inext));
    let more = b.cmp_lt_i32(Opnd::Val(inext), Opnd::Val(c10));
    let nm = b.let_(Op::NotI1(Opnd::Val(more)));
    b.break_if(Opnd::Val(nm));
    b.end_loop();
    b.finish()
}

#[test]
fn lint_reports_reads_after_set_only() {
    assert!(lint(&counter_kernel(false, false)).is_empty(), "read-before-set / cross-iteration reads must not be reported");
    let w = lint(&counter_kernel(true, false));
    assert_eq!(w.len(), 1, "{w:?}");
    assert!(w[0].contains("var0"));
    assert!(lint(&counter_kernel(true, true)).is_empty(), "allow_var_rereads opts out");
}