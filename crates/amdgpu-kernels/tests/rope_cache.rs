//! Batch 2 casts / static fp8 quant / RoPE / KV-cache write (docs/op-coverage-membound.md): name grammar, strict verification and assembly.
use amdgpu_compiler::gpu_verify::analyse;
use amdgpu_kernels::kernel_by_name;

const KERNELS: &[&str] = &[
    "cvt_f32_bf16", "cvt_bf16_f32", "quant_bf16_fp8_static",
    "rope_neox_bf16_h32_d128", "rope_gptj_bf16_h32_d128", "rope_neox_bf16_h40_d64", "rope_gptj_bf16_h8_d256", "rope_neox_bf16_h1_d64",
    "reshape_cache_flash_bf16_h8_d128", "reshape_cache_flash_bf16_h5_d64", "reshape_cache_flash_bf16_h16_d128",
    "quant_token_bf16_int8_n4096", "quant_token_bf16_int8_n1000", "dequant_fp8_bf16_static", "dequant_fp8_bf16_token_n4096", "dequant_fp8_bf16_group128_n4096", "dequant_fp8_bf16_token_n1000",
    "reshape_cache_blocked_bf16_h8_d128_b16", "reshape_cache_blocked_bf16_h8_d64_b64", "reshape_cache_blocked_bf16_h5_d128_b8",
    "moe_sorting_e128_k8_b32", "moe_sorting_e64_k2_b16", "moe_sorting_e256_k8_b128", "moe_sorting_e8_k1_b64", "moe_sorting_e385_k6_b32",
    "rope_neox_bf16_h32_d128_r64", "rope_gptj_bf16_h32_d128_r64", "rope_neox_bf16_h8_d256_r32",
    "argmax_bf16_v32000", "argmax_bf16_v128256", "argmax_bf16_v8", "sample_bf16_v32000", "sample_bf16_v128256", "sample_bf16_v1000",
    "moe_count_e128_k8", "moe_scan_e128_k8_b32", "moe_scatter_e128_k8_b32", "moe_scatter_ord_e128_k8_b32", "moe_count_e256_k8", "moe_scan_e256_k8_b128", "moe_scatter_ord_e256_k8_b128",
    "sample_topkp_bf16_v32000", "sample_topkp_bf16_v128256", "sample_topkp_bf16_v1000",
    "rope_neox_bf16_h1_d64", "rope_neox_bf16_h2_d128", "rope_gptj_bf16_h1_d64", "rope_neox_bf16_h4_d64_r32",
    "moe_reduce_bf16_k8_d7168", "moe_reduce_bf16_k6_d2048", "moe_combine_bf16_k8_d4096", "moe_combine_bf16_k2_d24",
    "silu_mul_limit_bf16_d4096", "swiglu_oai_mul_bf16_d4096", "silu_mul_limit_bf16_d24", "silu_bf16", "gelu_tanh_bf16", "gelu_fast_bf16", "relu2_bf16", "sigmoid_bf16",
    "gated_rmsnorm_bf16_n128", "gated_rmsnorm_fp8_n128", "gated_rmsnorm_bf16_n1000", "gated_rmsnorm_fp8_n2048", "reshape_cache_flash_fp8_h8_d128", "reshape_cache_flash_fp8_h5_d64",
    "qknorm_rope_neox_bf16_h8_d128", "qknorm_rope_gptj_bf16_h4_d256", "qknorm_rope_neox_bf16_h1_d64", "add_rmsnorm_fp8_n1000", "add_rmsnorm_fp8_n128",
    "causal_conv1d_update_bf16_w4_s3", "causal_conv1d_update_bf16_w4_s3_idx_silu", "causal_conv1d_update_bf16_w2_s1",
    "moe_fused_gate_bf16_e256_g8_tg4_k8", "moe_fused_gate_bf16_e128_g4_tg2_k6", "moe_fused_gate_bf16_e256_g16_tg4_k8", "moe_fused_gate_bf16_e64_g8_tg3_k4",
    "topk_per_row_decode_f32_k2048", "topk_per_row_decode_f32_k64", "topk_per_row_prefill_f32_k128", "topk_per_row_decode_f32_k1",
    "causal_conv1d_fwd_bf16_w4", "causal_conv1d_fwd_bf16_w4_silu", "causal_conv1d_fwd_split_bf16_w4_silu", "causal_conv1d_fwd_bf16_w2", "causal_conv1d_fwd_bf16_w3_silu",
    "reshape_cache_blocked_fp8_h8_d128_b16", "reshape_cache_blocked_fp8_h8_d128_b16_i64", "reshape_cache_blocked_fp8_h4_d64_b64", "reshape_cache_flash_bf16_h8_d128_i64", "reshape_cache_flash_fp8_h8_d128_i64", "reshape_cache_blocked_bf16_h8_d128_b16_i64",
    "topk_softmax_bf16_e128_k8", "topk_softmax_renorm_bf16_e256_k8", "topk_softmax_bf16_e64_k2", "topk_softmax_bf16_e512_k4", "topk_softmax_bf16_e32_k1",
];

#[test]
fn the_casts_rope_and_cache_kernels_verify_strict_and_compile() {
    for k in KERNELS {
        let f = kernel_by_name(k).unwrap_or_else(|| panic!("{k} is registered"));
        let r = analyse(&f).unwrap();
        assert!(r.errors().is_empty(), "{k}: {}", r.render());
        let c = amdgpu_compiler::compile(&f).unwrap_or_else(|e| panic!("{k}: {e}"));
        // 256-thread blocks may use up to 256 VGPRs; the heavier fused kernels (a lane owns 8 features x several tokens) are allowed 200, everything else stays small
        let limit = if k.starts_with("causal_conv1d_fwd") || k.starts_with("gated_rmsnorm_fp8") || k.starts_with("add_rmsnorm_fp8") { 200 } else { 128 };
        assert!(c.vgpr <= limit, "{k}: a streaming kernel stays small (occupancy): {} VGPR (limit {limit})", c.vgpr);
    }
}

#[test]
fn the_name_grammar_rejects_what_the_kernels_do_not_support() {
    for bad in ["rope_neox_bf16_h0_d128", "rope_neox_bf16_h8_d96", "rope_gptj_bf16_h257_d64", "rope_neox_bf16_hx_d64", "reshape_cache_flash_bf16_h8_d12", "reshape_cache_flash_bf16_h0_d64", "reshape_cache_flash_bf16_h8", "cvt_f32_bf16x", "topk_softmax_bf16_e24_k2", "topk_softmax_bf16_e128_k0", "topk_softmax_bf16_e128_k17", "topk_softmax_bf16_e520_k4", "topk_softmax_bf16_e128", "dequant_fp8_bf16_group128_n1000", "dequant_fp8_bf16_token_n7", "dequant_fp8_bf16_x", "quant_token_bf16_int8_n12", "reshape_cache_blocked_fp8_h8_d128_b8", "reshape_cache_blocked_fp8_h8_d96_b16", "reshape_cache_blocked_fp8_h8_d128", "reshape_cache_flash_bf16_h8_d128_i32", "reshape_cache_flash_bf16_h8_d128_i64x", "causal_conv1d_fwd_bf16_w5", "causal_conv1d_fwd_bf16_w1", "causal_conv1d_fwd_bf16_wx", "causal_conv1d_fwd_bf16_w4_relu", "topk_per_row_decode_f32_k0", "topk_per_row_decode_f32_k2049", "topk_per_row_prefill_f32_kx", "topk_per_row_f32_k64", "moe_fused_gate_bf16_e256_g8_tg9_k8", "moe_fused_gate_bf16_e256_g6_tg4_k8", "moe_fused_gate_bf16_e100_g4_tg2_k4", "moe_fused_gate_bf16_e256_g8_tg1_k33", "moe_fused_gate_bf16_e256_g8_tg4", "moe_fused_gate_bf16_e256_g2_tg1_k4", "qknorm_rope_neox_bf16_h8_d128_r64", "qknorm_rope_neox_bf16_h8_d96", "add_rmsnorm_fp8_n12", "causal_conv1d_update_bf16_w5_s4", "causal_conv1d_update_bf16_w4_s2", "causal_conv1d_update_bf16_w4_s9", "gated_rmsnorm_bf16_n12", "gated_rmsnorm_fp8_n9000", "reshape_cache_flash_fp8_h8_d12", "reshape_cache_flash_fp8_h0_d64", "silu_mul_limit_bf16_d7", "swiglu_oai_mul_bf16_d0", "relu2_bf16x", "swiglu_oai_mul_bf16_dx", "reshape_cache_flash_bf16_h8_d128_s32", "moe_reduce_bf16_k0_d64", "moe_reduce_bf16_k17_d64", "moe_combine_bf16_k4_d12", "moe_combine_bf16_k4_dx", "sample_topkp_bf16_v7", "sample_topkp_bf16_v131080", "moe_count_e128_k8_b32", "moe_scan_e128_k8", "moe_scatter_e128_k8", "moe_scatter_ord_e600_k8_b32", "moe_scan_e128_k8_b24", "sample_bf16_v7", "sample_bf16_v131080", "argmax_bf16_v0", "argmax_bf16_vx", "rope_neox_bf16_h8_d128_r24", "rope_neox_bf16_h8_d128_r144", "rope_neox_bf16_h8_d128_r0", "rope_gptj_bf16_h8_d128_rx", "moe_sorting_e4_k2_b32", "moe_sorting_e128_k0_b32", "moe_sorting_e128_k17_b32", "moe_sorting_e128_k8_b24", "moe_sorting_e600_k8_b32", "reshape_cache_blocked_bf16_h8_d96_b16", "reshape_cache_blocked_bf16_h8_d128_b24", "reshape_cache_blocked_bf16_h8_d128"] {
        assert!(kernel_by_name(bad).is_none(), "{bad}");
    }
}

#[test]
fn block_shapes_follow_the_work() {
    // RoPE: waves = min(4, ceil(H * L / 64)); the cache write: 4 / 2 / 1 tokens per 256-thread block by row size
    assert_eq!(kernel_by_name("rope_neox_bf16_h8_d128").unwrap().workgroup_size, 64);
    assert_eq!(kernel_by_name("rope_neox_bf16_h32_d128").unwrap().workgroup_size, 256);
    assert_eq!(kernel_by_name("reshape_cache_flash_bf16_h8_d128").unwrap().workgroup_size, 256);
    assert_eq!(kernel_by_name("reshape_cache_flash_bf16_h5_d64").unwrap().workgroup_size, 256);
}