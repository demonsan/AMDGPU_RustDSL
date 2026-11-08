//! The kernel registry: every kernel of the repository by name (`amdc emit <name>`, the gates, the tests), and the list `amdc list` prints.

/// All kernel definitions currently shipped with the DSL.
pub fn kernel_names() -> &'static [&'static str] {
    &["probe_agpr_ref", "probe_agpr_acc", "probe_agpr_full", "probe_agpr_pinned", "probe_store", "probe_math", "probe_mfma", "probe_lds", "gv_audit_region_trust", "probe_mfma_scale", "probe_lds_self2", "probe_roles_loop", "probe_lds_tr_b8", "probe_log_store4", "probe_llvm_kernarg", "probe_llvm_ids", "probe_llvm_roles", "probe_llvm_lds", "probe_llvm_branch", "probe_llvm_loop", "probe_llvm_atomic", "probe_llvm_maskge", "probe_llvm_k1", "probe_llvm_k2", "probe_buffer_rw", "probe_mfma_cform", "probe_store_x2", "probe_last_arriver", "probe_last_arriver_lite", "probe_last_arriver_nofence", "probe_last_arriver_plain", "probe_last_arriver_plain_fenced", "probe_la8", "probe_la8_lite", "probe_la8_plain_fenced", "probe_la8_plain", "probe_la8_nodrain", "probe_la_verify", "probe_mfma_unscaled", "probe_expr_ids", "probe_expr_kernarg", "probe_valu_p1", "probe_mem_p1", "probe_mfma_p1", "probe_f8f6f4_p1"]
}

pub fn kernel_by_name(name: &str) -> Option<amdgpu_dsl::Function> {
    match name {
        "probe_dma_voff_spill" => Some(crate::probes::probe_dma_voff_spill()),
        "probe_stamp" => Some(crate::probes::probe_stamp()),
        "probe_valu_p1" => Some(crate::probes::tileops::probe_valu_p1()),
        "probe_mem_p1" => Some(crate::probes::tileops::probe_mem_p1()),
        "probe_mfma_p1" => Some(crate::probes::tileops::probe_mfma_p1()),
        "probe_f8f6f4_p1" => Some(crate::probes::tileops::probe_f8f6f4_p1()),
        "probe_agpr_ref" => Some(crate::probes::probe_agpr_ref()),
        "probe_agpr_acc" => Some(crate::probes::probe_agpr_acc()),
        "probe_agpr_full" => Some(crate::probes::probe_agpr_full()),
        "probe_agpr_pinned" => Some(crate::probes::probe_agpr_pinned()),
        "probe_store" => Some(crate::probes::probe_store()),
        "probe_math" => Some(crate::probes::probe_math()),
        "probe_mfma" => Some(crate::probes::probe_mfma()),
        "probe_lds" => Some(crate::probes::probe_lds()),
        "gv_audit_region_trust" => Some(crate::verifier_cases::gv_audit_region_trust()),
        "probe_lds_oob" => Some(crate::verifier_cases::probe_lds_oob()),
        "gv_attack_false_contract" => Some(crate::verifier_cases::gv_attack_false_contract()),
        "gv_attack_assume_loop" => Some(crate::verifier_cases::gv_attack_assume_loop()),
        "gv_attack_assume_trusted" => Some(crate::verifier_cases::gv_attack_assume_trusted()),
        "gv_attack_lds_const" => Some(crate::verifier_cases::gv_attack_lds_const()),
        "gv_attack_desc_over" => Some(crate::verifier_cases::gv_attack_desc_over()),
        "gv_attack_clamp_path" => Some(crate::verifier_cases::gv_attack_clamp_path()),
        "probe_expr_ids" => Some(crate::probes::probe_expr_ids()),
        "probe_expr_kernarg" => Some(crate::probes::probe_expr_kernarg()),
        "probe_mfma_scale" => Some(crate::probes::probe_mfma_scale()),
        "probe_lds_self2" => Some(crate::probes::probe_lds_self2()),
        "probe_roles_loop" => Some(crate::probes::probe_roles_loop()),
        "probe_lds_tr_b8" => Some(crate::probes::probe_lds_tr_b8()),
        n if (n.starts_with("gemm_mxfp8_bs_v") || n.starts_with("gemm_a8w8_") || n.starts_with("gemm_a16w16_") || n.starts_with("gemm_a4w4_") || n.starts_with("gemm_splitk_")) && crate::gemm::by_name(n).is_some() => crate::gemm::by_name(n),
        n if n.starts_with("handoff_probe_") => crate::probes::handoff::by_name(n),
        n if n.starts_with("mlp_fp8_") => crate::mlp::by_name(n),
        // P2 (stage library): RMSNorm + per-token fp8 quantization from the generic E / R / Q stages; N in the name (multiple of 512)
        n if n.starts_with("rmsnorm_quant_fp8_n") => crate::norm::by_name(n),
        // batch 2: expert routing, RoPE, KV-cache write
        n if n.starts_with("topk_softmax_bf16_e") || n.starts_with("topk_softmax_renorm_bf16_e") => crate::topk::by_name(n),
        n if n.starts_with("rope_neox_bf16_h") || n.starts_with("rope_gptj_bf16_h") => crate::rope::by_name(n),
        n if n.starts_with("reshape_cache_flash_bf16_h") => crate::cache::by_name(n),
        // batch 2 (docs/op-coverage-memtemplates.md): row kernels, streaming elementwise
        n if n.starts_with("quant_token_bf16_fp8_n") || n.starts_with("quant_group128_bf16_fp8_n") || n.starts_with("silu_mul_quant_fp8_d") => crate::rows::quant::by_name(n),
        n if n.starts_with("rmsnorm_bf16_n") || n.starts_with("add_rmsnorm_bf16_n") || n.starts_with("layernorm_bf16_n") || n.starts_with("softmax_bf16_n") => crate::rows::by_name(n),
        n if n.starts_with("silu_mul_bf16_d") || n.starts_with("gelu_tanh_mul_bf16_d") || n == "add_bf16" || n == "cvt_f32_bf16" || n == "cvt_bf16_f32" || n == "quant_bf16_fp8_static" => crate::eltwise::by_name(n),
        // P3: the same four kernels on the layout layer (amdgpu-layout + amdgpu-isa + amdgpu-tile)
        "fmha_fwd_hd128_mxfp8_v3_vs128" => Some(crate::fmha::fmha_fwd_hd128_mxfp8_v3_vs128()),
        "fmha_fwd_hd128_mxfp8_v3_t128" => Some(crate::fmha::fmha_fwd_hd128_mxfp8_v3_t128()),
        "fmha_fwd_hd128_bf16" => Some(crate::fmha::bf16::fmha_fwd_hd128_bf16("fmha_fwd_hd128_bf16", false)),
        "fmha_fwd_hd128_bf16_causal" => Some(crate::fmha::bf16::fmha_fwd_hd128_bf16("fmha_fwd_hd128_bf16_causal", true)),
        n if n.starts_with("fmha_fwd_hd") && n.contains("_bf16") => {
            // fmha_fwd_hd<64|128|256>_bf16[_causal][_swa][_group]
            let d = n["fmha_fwd_hd".len()..].split('_').next().and_then(|s| s.parse::<i32>().ok())?;
            matches!(d, 64 | 128 | 256).then(|| crate::fmha::bf16::fmha_fwd_bf16(n, crate::fmha::bf16::AttnCfg { d, causal: n.contains("_causal"), swa: n.contains("_swa"), group: n.contains("_group") }))
        }
        "fmha_fwd_hd128_mxfp8_v3_kvsplit" => Some(crate::fmha::fmha_fwd_hd128_mxfp8_v3_kvsplit()),
        "fmha_fwd_hd128_mxfp8_v3_kvsplit_combine8h" => Some(crate::fmha::fmha_fwd_hd128_mxfp8_v3_kvsplit_combine8h()),
        // scheduler-token A/B variants of the shipped kernels (same code + `Function::sched_opts` tokens on top of the kernel's own preset; no env knobs):
        "probe_dma" => Some(crate::probes::probe_dma()),
        "probe_safeasm_e2e" => Some(crate::probes::probe_safeasm_e2e()),
        "probe_log_store4" => Some(crate::probes::probe_log_store4()),
        "probe_llvm_kernarg" => Some(crate::probes::probe_llvm_kernarg()),
        "probe_llvm_ids" => Some(crate::probes::probe_llvm_ids()),
        "probe_llvm_roles" => Some(crate::probes::probe_llvm_roles()),
        "probe_llvm_lds" => Some(crate::probes::probe_llvm_lds()),
        "probe_llvm_branch" => Some(crate::probes::probe_llvm_branch()),
        "probe_llvm_loop" => Some(crate::probes::probe_llvm_loop()),
        "probe_llvm_atomic" => Some(crate::probes::probe_llvm_atomic()),
        "probe_llvm_maskge" => Some(crate::probes::probe_llvm_maskge()),
        "probe_llvm_k1" => Some(crate::probes::probe_llvm_k1()),
        "probe_llvm_k2" => Some(crate::probes::probe_llvm_k2()),
        "probe_mfma_unscaled" => Some(crate::probes::probe_mfma_unscaled()),
        "probe_mfma_scale_opsel" => Some(crate::probes::probe_mfma_scale_opsel()),
        "probe_buffer_rw" => Some(crate::probes::probe_buffer_rw()),
        "probe_buffer_dyn" => Some(crate::probes::probe_buffer_dyn()),
        "probe_buffer_lds" => Some(crate::probes::probe_buffer_lds()),
        "probe_mfma_cform" => Some(crate::probes::probe_mfma_cform()),
        "probe_store_x2" => Some(crate::probes::probe_store_x2()),
        "probe_cvtchain_dsl" => Some(crate::probes::probe_cvtchain_dsl()),
        "probe_cvtchain_asm" => Some(crate::probes::probe_cvtchain_asm()),
        "probe_la8" => Some(crate::probes::probe_la8()),
        "probe_la8_lite" => Some(crate::probes::probe_la8_lite()),
        "probe_la8_plain_fenced" => Some(crate::probes::probe_la8_plain_fenced()),
        "probe_la8_plain" => Some(crate::probes::probe_la8_plain()),
        "probe_la8_nodrain" => Some(crate::probes::probe_la8_nodrain()),
        "probe_la_verify" => Some(crate::probes::probe_la_verify()),
        "probe_last_arriver" => Some(crate::probes::probe_last_arriver()),
        "probe_last_arriver_lite" => Some(crate::probes::probe_last_arriver_lite()),
        "probe_last_arriver_plain" => Some(crate::probes::probe_last_arriver_plain()),
        "probe_last_arriver_plain_fenced" => Some(crate::probes::probe_last_arriver_plain_fenced()),
        "probe_last_arriver_nofence" => Some(crate::probes::probe_last_arriver_nofence()),
        "probe_dma_tile" => Some(crate::probes::probe_dma_tile()),
        "probe_dma_seq" => Some(crate::probes::probe_dma_seq()),
        "probe_dma_kread" => Some(crate::probes::probe_dma_kread()),
        "probe_dma_map" => Some(crate::probes::probe_dma_map()),
        "probe_smfmac_ones" => Some(crate::probes::probe_smfmac_ones()),
        "probe_smfmac_gen" => Some(crate::probes::probe_smfmac_gen()),
        "probe_lds_tr_b8_gen" => Some(crate::probes::probe_lds_tr_b8_gen()),
        _ => None,
    }
}
