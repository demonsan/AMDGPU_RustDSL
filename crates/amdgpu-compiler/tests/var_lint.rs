//! Lint for loop-variable re-reads after `set_var` in the same iteration (see `amdgpu_dsl::lint`).

use amdgpu_compiler::{kernel_by_name, kernel_names};
use amdgpu_dsl::{lint, DType, FnBuilder, Opnd, Stmt};

#[test]
fn shipped_kernels_do_not_reread_loop_variables_after_set() {
    let mut hits = Vec::new();
    for k in kernel_names() {
        let f = kernel_by_name(k).unwrap();
        for w in lint(&f) {
            hits.push(format!("{k}: {w}"));
        }
    }
    assert!(hits.is_empty(), "{}", hits.join("\n"));
}