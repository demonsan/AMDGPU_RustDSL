//! Batch 2 casts / static fp8 quant / RoPE / KV-cache write (docs/op-coverage-membound.md): name grammar, strict verification and assembly.
use amdgpu_compiler::gpu_verify::analyse;
use amdgpu_kernels::kernel_by_name;

const KERNELS: &[&str] = &[
    "cvt_f32_bf16", "cvt_bf16_f32", "quant_bf16_fp8_static",
    "rope_neox_bf16_h32_d128", "rope_gptj_bf16_h32_d128", "rope_neox_bf16_h40_d64", "rope_gptj_bf16_h8_d256", "rope_neox_bf16_h1_d64",
    "reshape_cache_flash_bf16_h8_d128", "reshape_cache_flash_bf16_h5_d64", "reshape_cache_flash_bf16_h16_d128",
    "quant_token_bf16_int8_n4096", "quant_token_bf16_int8_n1000", "dequant_fp8_bf16_static", "dequant_fp8_bf16_token_n4096", "dequant_fp8_bf16_group128_n4096", "dequant_fp8_bf16_token_n1000",
    "topk_softmax_bf16_e128_k8", "topk_softmax_renorm_bf16_e256_k8", "topk_softmax_bf16_e64_k2", "topk_softmax_bf16_e512_k4", "topk_softmax_bf16_e32_k1",
];

#[test]
fn the_casts_rope_and_cache_kernels_verify_strict_and_compile() {
    for k in KERNELS {
        let f = kernel_by_name(k).unwrap_or_else(|| panic!("{k} is registered"));
        let r = analyse(&f).unwrap();
        assert!(r.errors().is_empty(), "{k}: {}", r.render());
        let c = amdgpu_compiler::compile(&f).unwrap_or_else(|e| panic!("{k}: {e}"));
        assert!(c.vgpr <= 128, "{k}: a streaming kernel stays small (occupancy): {} VGPR", c.vgpr);
    }
}

#[test]
fn the_name_grammar_rejects_what_the_kernels_do_not_support() {
    for bad in ["rope_neox_bf16_h0_d128", "rope_neox_bf16_h8_d96", "rope_gptj_bf16_h257_d64", "rope_neox_bf16_hx_d64", "reshape_cache_flash_bf16_h8_d12", "reshape_cache_flash_bf16_h0_d64", "reshape_cache_flash_bf16_h8", "cvt_f32_bf16x", "topk_softmax_bf16_e24_k2", "topk_softmax_bf16_e128_k0", "topk_softmax_bf16_e128_k17", "topk_softmax_bf16_e520_k4", "topk_softmax_bf16_e128", "dequant_fp8_bf16_group128_n1000", "dequant_fp8_bf16_token_n7", "dequant_fp8_bf16_x", "quant_token_bf16_int8_n12"] {
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