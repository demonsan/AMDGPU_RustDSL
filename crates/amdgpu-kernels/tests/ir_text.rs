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
        // batch 2: streaming elementwise
        "silu_mul_bf16_d4096",
        "silu_mul_bf16_d24",
        "gelu_tanh_mul_bf16_d4096",
        "add_bf16",
        "rmsnorm_bf16_n4096",
        "layernorm_bf16_n776",
        "softmax_bf16_n1000",
        "add_rmsnorm_bf16_n4096",
        "quant_token_bf16_fp8_n4096",
        "quant_group128_bf16_fp8_n640",
        "silu_mul_quant_fp8_d4096",
        "cvt_f32_bf16",
        "cvt_bf16_f32",
        "quant_bf16_fp8_static",
        "rope_neox_bf16_h32_d128",
        "rope_gptj_bf16_h8_d256",
        "reshape_cache_flash_bf16_h8_d128",
        "topk_softmax_bf16_e128_k8",
        "reshape_cache_blocked_fp8_h8_d128_b16_i64",
        "reshape_cache_flash_bf16_h8_d128_i64",
        "causal_conv1d_fwd_split_bf16_w4_silu",
        "causal_conv1d_fwd_bf16_w2",
        "topk_per_row_decode_f32_k2048",
        "topk_per_row_prefill_f32_k128",
        "moe_fused_gate_bf16_e256_g8_tg4_k8",
        "qknorm_rope_neox_bf16_h8_d128",
        "add_rmsnorm_fp8_n1000",
        "causal_conv1d_update_bf16_w4_s3_idx_silu",
        "gated_rmsnorm_fp8_n128",
        "gated_rmsnorm_bf16_n1000",
        "reshape_cache_flash_fp8_h8_d128",
        "swiglu_oai_mul_bf16_d4096",
        "silu_mul_limit_bf16_d4096",
        "relu2_bf16",
        "moe_reduce_bf16_k8_d7168",
        "moe_combine_bf16_k8_d4096",
        "sample_topkp_bf16_v32000",
        "moe_scan_e128_k8_b32",
        "moe_scatter_ord_e128_k8_b32",
        "sample_bf16_v32000",
        "argmax_bf16_v128256",
        "moe_sorting_e128_k8_b32",
        "reshape_cache_blocked_bf16_h8_d128_b16",
        "quant_token_bf16_int8_n4096",
        "dequant_fp8_bf16_token_n1000",
        "dequant_fp8_bf16_group128_n4096",
        "dequant_fp8_bf16_static",
        "topk_softmax_renorm_bf16_e256_k8",
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
