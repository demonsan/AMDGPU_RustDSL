//! Printing and structural equality for the DSL IR (`amdc print-ir --stage dsl`).
//!
//! The printed form is the complete `irv1` text (`ir_text.rs`, docs/ir-text.md): it parses back to the same `Function`. Structural equality
//! is defined on an alpha-canonical form: ValueIds / VarIds are renumbered by first occurrence (in every statement, op operand and
//! annotation), so two builders that allocate different ids but construct the same program are equal and hash equal. Function names are
//! intentionally not part of the structural identity.

use crate::Function;
use std::fmt::Write;
use std::hash::{Hash, Hasher};

fn canon_function(f: &Function, include_contract: bool) -> String {
    let g = crate::ir_text::alpha_canonical(f);
    let mut s = String::new();
    // Signature/ABI/contract/tiles are structural, but the function *name* is not.
    let _ = writeln!(s, "kernarg={} wg={} lds={}", g.kernarg_size, g.workgroup_size, g.lds_bytes);
    let _ = writeln!(s, "args={:?}", g.args);
    if include_contract {
        let _ = writeln!(s, "contract={:?}", g.contract);
    }
    let _ = writeln!(s, "value_tys={:?}", g.value_tys);
    let _ = writeln!(s, "var_tys={:?}", g.var_tys);
    let _ = writeln!(s, "profile={:?} lower={:?} sched={:?}", g.compile_profile, g.lower_opts, g.sched_opts);
    let _ = writeln!(s, "agpr={:?}", g.agpr);
    if !g.reg_plan.is_empty() {
        let _ = writeln!(s, "reg_plan={:?}", g.reg_plan);
    }
    if !g.regions.is_empty() {
        let _ = writeln!(s, "regions={:?}", g.regions);
    }
    for st in g.stmts().iter() {
        let _ = writeln!(s, "{st:?}");
    }
    s
}

/// Structural equality over the DSL IR (alpha-equivalent ValueIds/VarIds; function name ignored).
pub fn dsl_structural_eq(a: &Function, b: &Function) -> bool {
    canon_function(a, true) == canon_function(b, true)
}

/// Body/ABI structural equality ignoring the declared contract metadata. Useful when comparing a
/// builder refactor before the same annotations are added to both versions (e.g. the expr-layer
/// probes vs their older manual twins).
pub fn dsl_body_structural_eq(a: &Function, b: &Function) -> bool {
    canon_function(a, false) == canon_function(b, false)
}

/// Structural hash consistent with [`dsl_structural_eq`].
pub fn dsl_structural_hash(f: &Function) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    canon_function(f, true).hash(&mut h);
    h.finish()
}

/// Structural hash of the body/ABI without contract metadata.
pub fn dsl_body_structural_hash(f: &Function) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    canon_function(f, false).hash(&mut h);
    h.finish()
}

/// Canonical form without contract metadata.
pub fn canonical_dsl_body(f: &Function) -> String {
    canon_function(f, false)
}

/// The DSL IR text (`irv1`: complete and parseable, see `ir_text::parse`).
pub fn print_dsl(f: &Function) -> String {
    crate::ir_text::print(f)
}
