//! The kernel registry: every kernel of the repository by name (`amdc emit <name>`, the gates, the tests), and the list `amdc list` prints.

/// All kernel definitions currently shipped with the DSL.
pub fn kernel_names() -> &'static [&'static str] {
    &["probe_agpr_ref", "probe_agpr_acc", "probe_agpr_full", "probe_agpr_pinned", "probe_store", "probe_math", "probe_mfma", "probe_lds", "gv_audit_region_trust", "probe_mfma_scale", "probe_lds_self2", "probe_roles_loop", "probe_lds_tr_b8", "probe_log_store4", "probe_llvm_kernarg", "probe_llvm_ids", "probe_llvm_roles", "probe_llvm_lds", "probe_llvm_branch", "probe_llvm_loop", "probe_llvm_atomic", "probe_llvm_maskge", "probe_llvm_k1", "probe_llvm_k2", "probe_buffer_rw", "probe_mfma_cform", "probe_store_x2", "probe_last_arriver", "probe_last_arriver_lite", "probe_last_arriver_nofence", "probe_last_arriver_plain", "probe_last_arriver_plain_fenced", "probe_la8", "probe_la8_lite", "probe_la8_plain_fenced", "probe_la8_plain", "probe_la8_nodrain", "probe_la_verify", "probe_mfma_unscaled", "probe_expr_ids", "probe_expr_kernarg"]
}

pub fn kernel_by_name(name: &str) -> Option<amdgpu_dsl::Function> {
    match name {
        "probe_dma_voff_spill" => Some(crate::probes::probe_dma_voff_spill()),
        "probe_stamp" => Some(crate::probes::probe_stamp()),
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
        n if n.starts_with("gemm_mxfp8_bs_v") && crate::gemm::by_name(n).is_some() => crate::gemm::by_name(n),
        n if n.starts_with("handoff_probe_") => crate::probes::handoff::by_name(n),
        n if n.starts_with("mlp_fp8_") => crate::mlp::by_name(n),
        // P3: the same four kernels on the layout layer (amdgpu-layout + amdgpu-isa + amdgpu-tile)
        "fmha_fwd_hd128_mxfp8_v3_vs128" => Some(crate::fmha::fmha_fwd_hd128_mxfp8_v3_vs128()),
        "fmha_fwd_hd128_mxfp8_v4w_proto" => Some(crate::fmha::fmha_fwd_hd128_mxfp8_v4w_proto()),
        // scheduler-token variants of the 4-wave prototype (the full list scheduler cannot allocate it: sched falls back to the unscheduled code): `rewait` = authored order with
        // counted waits, `hoist=24` = ds_reads hoisted within the register budget, `ovl` = independent VALU spread over the MFMAs of each barrier-free stretch (docs/lgkm-lookahead.md)
        "fmha_fwd_hd128_mxfp8_v4w_proto_rw" | "fmha_fwd_hd128_mxfp8_v4w_proto_sched" => {
            let mut f = crate::fmha::fmha_fwd_hd128_mxfp8_v4w_proto();
            f.sched_opts = Some(if name.ends_with("_rw") { "rewait" } else { "rewait,hoist=24,ovl" }.to_string());
            f.name = name.to_string();
            Some(f)
        }
        "fmha_fwd_hd128_mxfp8_v3_t128" => Some(crate::fmha::fmha_fwd_hd128_mxfp8_v3_t128()),
        "fmha_fwd_hd128_mxfp8_v3_kvsplit" => Some(crate::fmha::fmha_fwd_hd128_mxfp8_v3_kvsplit()),
        "fmha_fwd_hd128_mxfp8_v3_kvsplit_combine8h" => Some(crate::fmha::fmha_fwd_hd128_mxfp8_v3_kvsplit_combine8h()),
        // scheduler-token A/B variants of the shipped kernels (same code + `Function::sched_opts` tokens on top of the kernel's own preset; no env knobs):
        // probe_ovl_w<waves>_m<mfma>_v<valu>: single-wave MFMA / VALU overlap (fmha-4w design)
        n if n.starts_with("probe_ovl_w") => {
            let p: Vec<u32> = n["probe_ovl_".len()..].split('_').filter_map(|x| x[1..].parse().ok()).collect();
            (p.len() == 3).then(|| crate::probes::probe_ovl(n, p[0], p[1], p[2], 200))
        }
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
