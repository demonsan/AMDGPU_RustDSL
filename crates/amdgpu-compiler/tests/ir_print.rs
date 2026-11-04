//! Item D: readable IR printing + alpha-equivalent structural equality.
use amdgpu_compiler::ir_print::{canonical_dsl_body, dsl_body_structural_eq, dsl_body_structural_hash, dsl_structural_eq, dsl_structural_hash, print_dsl, print_mir};
use amdgpu_compiler::kernel_by_name;

#[test]
fn expr_layer_twins_are_structurally_equal() {
    for (manual, expr) in [("probe_llvm_ids", "probe_expr_ids"), ("probe_llvm_kernarg", "probe_expr_kernarg")] {
        let a = kernel_by_name(manual).unwrap();
        let b = kernel_by_name(expr).unwrap();
        if !dsl_body_structural_eq(&a, &b) {
            let ca = canonical_dsl_body(&a); let cb = canonical_dsl_body(&b);
            let first = ca.lines().zip(cb.lines()).enumerate().find(|(_, (x, y))| x != y);
            panic!("{expr} body not structurally equal to {manual}; first diff: {first:?}");
        }
        assert_eq!(dsl_body_structural_hash(&a), dsl_body_structural_hash(&b), "{expr} body hash");
        // the canonical form is alpha-renamed and does not contain the function name
        let ca = canonical_dsl_body(&a);
        assert!(!ca.contains(manual), "canonical form must not include the kernel name");
        assert_eq!(ca, canonical_dsl_body(&b));
    }
}

#[test]
fn structural_hash_detects_a_changed_operation() {
    let mut f = kernel_by_name("probe_expr_ids").expect("probe_expr_ids");
    let before = dsl_structural_hash(&f);
    let mut changed = false;
    for st in &mut f.stmts {
        if let amdgpu_dsl::Stmt::Let { op: amdgpu_dsl::Op::ConstI32(v), .. } = st {
            *v = v.wrapping_add(1);
            changed = true;
            break;
        }
    }
    assert!(changed, "probe must contain a ConstI32");
    assert_ne!(before, dsl_structural_hash(&f), "hash must change");
    let g = kernel_by_name("probe_expr_ids").unwrap();
    assert!(!dsl_structural_eq(&f, &g), "mutated function must not be equal");
}

#[test]
fn dsl_printer_is_structured_and_shows_types_operands_and_markers() {
    for name in ["probe_expr_ids", "probe_expr_kernarg", "fmha_fwd_hd128_mxfp8_v3_vs128"] {
        let f = kernel_by_name(name).unwrap();
        let text = print_dsl(&f);
        assert!(text.starts_with(&format!("dsl @{name}")), "{name}: header");
        assert!(text.contains("kernarg_size="), "{name}: abi");
        assert!(text.contains("  body:"), "{name}: body");
        assert!(text.contains("%v"), "{name}: value names");
        if name.starts_with("fmha") {
            assert!(text.contains("contract:"), "{name}: contract");
            assert!(text.contains("tile "), "{name}: tiles");
        }
    }
    let text = print_dsl(&kernel_by_name("probe_expr_ids").unwrap());
    assert!(text.lines().any(|l| l.trim_start().starts_with("%v") && l.contains(": ")), "a typed Let line");
}

#[test]
fn mir_printer_shows_blocks_ops_and_registers() {
    let f = kernel_by_name("probe_expr_kernarg").unwrap();
    let m = amdgpu_compiler::lower::lower(&f).unwrap();
    let text = print_mir(&m, None);
    assert!(text.starts_with("mir @probe_expr_kernarg"), "header");
    assert!(text.contains("  b0:"), "block 0");
    assert!(text.contains(":V"), "virtual register class");
    let (ma, a) = amdgpu_compiler::compile_mir_alloc(&f).unwrap();
    let ra = print_mir(&ma, Some(&a));
    assert!(ra.contains(":V") && (ra.contains(" v") || ra.contains(" s")), "physical registers");
}
