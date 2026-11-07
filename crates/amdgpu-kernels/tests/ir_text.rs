//! DSL IR text round trip (docs/ir-text.md): every registered kernel and the shipped set print to `irv1`, parse back to an identical
//! `Function` (text fixpoint + field-for-field equality), and compile to identical assembly.
use amdgpu_dsl::ir_text::{parse, print};

fn all_kernels() -> Vec<String> {
    let mut v: Vec<String> = amdgpu_kernels::kernel_names().iter().map(|s| s.to_string()).collect();
    for k in [
        "fmha_fwd_hd128_mxfp8_v3_vs128",
        "fmha_fwd_hd128_mxfp8_v3_t128",
        "fmha_fwd_hd128_mxfp8_v3_kvsplit",
        "fmha_fwd_hd128_mxfp8_v3_kvsplit_combine8h",
        "gemm_mxfp8_bs_v4",
        "mlp_fp8_ws_fused",
        "mlp_fp8_ws_g1",
        "mlp_fp8_ws_g2",
        // stage library P2 (generic E / R / Q stages)
        "mlp_fp8_ws_g1q_relu",
        "mlp_fp8_ws_g1q_silu",
        "rmsnorm_quant_fp8_n1024",
        "rmsnorm_quant_fp8_n4096",
        "rmsnorm_quant_fp8_n8192",
    ] {
        v.push(k.to_string());
    }
    v
}

#[test]
fn every_kernel_round_trips_exactly() {
    for k in all_kernels() {
        let f = amdgpu_kernels::kernel_by_name(&k).unwrap();
        let t = print(&f);
        let g = parse(&t).unwrap_or_else(|e| panic!("{k}: {e}"));
        assert_eq!(print(&g), t, "{k}: text fixpoint");
        assert_eq!(format!("{g:?}"), format!("{f:?}"), "{k}: Function equality");
        assert!(amdgpu_dsl::print::dsl_structural_eq(&f, &g), "{k}: structural equality");
    }
}

#[test]
fn round_tripped_kernels_emit_identical_assembly() {
    for k in all_kernels() {
        let f = amdgpu_kernels::kernel_by_name(&k).unwrap();
        let g = parse(&print(&f)).unwrap();
        let a = amdgpu_compiler::compile_with(&f, &amdgpu_compiler::backend::OwnTail).map(|c| c.asm);
        let b = amdgpu_compiler::compile_with(&g, &amdgpu_compiler::backend::OwnTail).map(|c| c.asm);
        assert_eq!(a, b, "{k}: assembly differs after the round trip");
    }
}

#[test]
fn edits_and_errors() {
    let f = amdgpu_kernels::kernel_by_name("probe_expr_kernarg").unwrap();
    let t = print(&f);
    // an edit: change a constant -> a different but valid function
    let line = t.lines().find(|l| l.contains("ConstI32(4)")).expect("a ConstI32(4) line");
    let edited = t.replacen(line, &line.replace("ConstI32(4)", "ConstI32(8)"), 1);
    let g = parse(&edited).expect("edited text parses");
    assert!(!amdgpu_dsl::print::dsl_structural_eq(&f, &g));
    // a typo in an op name: located error
    let bad = t.replacen("ConstI32(4)", "ConstI3(4)", 1);
    let e = parse(&bad).unwrap_err().0;
    assert!(e.contains("line ") && e.contains("ConstI3"), "{e}");
    // a wrong type annotation
    let bad = t.replacen(": I32 = ConstI32(4)", ": F32 = ConstI32(4)", 1);
    let e = parse(&bad).unwrap_err().0;
    assert!(e.contains("declared type"), "{e}");
    // a use before definition (the verifier)
    let bad = t.replacen("%v7: I32 = ConstI32(4)", "%v7: I32 = AddI32(Val(99), Val(6))", 1);
    assert!(parse(&bad).is_err());
    // unbalanced braces
    let e = parse(&t[..t.len() - 3]).unwrap_err().0;
    assert!(e.contains("`}`") || e.contains("end of input"), "{e}");
}

#[test]
fn alpha_equivalence_ignores_id_allocation() {
    // renumber every value id by +1000 through the text: still structurally equal, but not identical
    let f = amdgpu_kernels::kernel_by_name("probe_expr_ids").unwrap();
    let mut g = amdgpu_dsl::ir_text::alpha_canonical(&f);
    assert!(amdgpu_dsl::print::dsl_structural_eq(&f, &g));
    g.name = "other".into();
    assert!(amdgpu_dsl::print::dsl_structural_eq(&f, &g), "the name is not structural");
}

#[test]
fn weaver_annotations_stages_and_safe_asm_round_trip() {
    use amdgpu_dsl::{DType, FnBuilder, Op, Opnd, RegRef, StageContract};
    let mut b = FnBuilder::new("weave_probe", 64);
    b.arg("out", DType::Ptr, 0, 8);
    let out = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);
    b.declare_region("w", true, "rewait");
    let v = b.stage("s", StageContract::default(), |b| {
        b.weave_region("w", |b| {
            let c = b.c_i32(3);
            b.add_i32(Opnd::Val(tid), Opnd::Val(c))
        })
    });
    b.pin_vgpr_at(RegRef::Val(v), 8);
    b.reg_separate(&[RegRef::Val(v), RegRef::Val(tid)]);
    b.reg_cap("w", 64);
    let f32v = b.let_(Op::BitcastF32(Opnd::Val(v)));
    b.store_f32(Opnd::Val(out), 0, Opnd::Val(f32v));
    let f = b.finish();
    let t = amdgpu_dsl::ir_text::print(&f);
    assert!(t.contains("region_decl RegionDecl") && t.contains("reg_plan PinV") && t.contains("RegionMark") && t.contains("stage \"s\""), "{t}");
    let g = amdgpu_dsl::ir_text::parse(&t).unwrap_or_else(|e| panic!("{e}\n{t}"));
    assert_eq!(format!("{g:?}"), format!("{f:?}"));
    // safe-asm blocks decode from their Debug spelling
    for blk in [amdgpu_safeasm::examples::rowsum_var_block(), amdgpu_safeasm::examples::cvt_chain_block(), amdgpu_safeasm::examples::global_probe_block()] {
        let back: amdgpu_safeasm::AsmBlock = amdgpu_dsl::ir_text::parse_value(&format!("{blk:#?}")).unwrap();
        assert_eq!(back, blk);
    }
}
