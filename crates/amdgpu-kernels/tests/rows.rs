//! Batch 2 row and quantization kernels (docs/op-coverage-membound.md): name grammar, strict GPU-semantics verification and assembly (one b128 load per chunk, buffer accesses hardware-guarded).
use amdgpu_codegen::gpu_verify::analyse;
use amdgpu_kernels::kernel_by_name;

const KERNELS: &[&str] = &[
    "rmsnorm_bf16_n4096", "rmsnorm_bf16_n1000", "rmsnorm_bf16_n200", "add_rmsnorm_bf16_n4096", "layernorm_bf16_n4096", "layernorm_bf16_n776", "layernorm_bf16_n8", "softmax_bf16_n4096", "softmax_bf16_n1000", "softmax_bf16_n8192",
    "quant_token_bf16_fp8_n4096", "quant_token_bf16_fp8_n1000", "quant_group128_bf16_fp8_n4096", "quant_group128_bf16_fp8_n640", "silu_mul_quant_fp8_d4096", "silu_mul_quant_fp8_d1000",
];

#[test]
fn the_row_and_quant_kernels_verify_strict_and_compile() {
    for k in KERNELS {
        let f = kernel_by_name(k).unwrap_or_else(|| panic!("{k} is registered"));
        let r = analyse(&f).unwrap();
        assert!(r.errors().is_empty(), "{k}: {}", r.render());
        let c = amdgpu_codegen::compile(&f).unwrap_or_else(|e| panic!("{k}: {e}"));
        assert!(c.vgpr <= 200, "{k}: {} VGPR", c.vgpr);
        assert!(c.asm.contains("buffer_load_dwordx4"), "{k}: b128 loads");
    }
}

#[test]
fn the_name_grammar_rejects_what_the_kernels_do_not_support() {
    for bad in ["rmsnorm_bf16_n0", "rmsnorm_bf16_n12", "rmsnorm_bf16_n8200", "layernorm_bf16_nx", "softmax_bf16_n9", "quant_group128_bf16_fp8_n100", "quant_group128_bf16_fp8_n8200", "silu_mul_quant_fp8_d7", "quant_token_bf16_fp8_n12"] {
        assert!(kernel_by_name(bad).is_none(), "{bad}");
    }
}

#[test]
fn short_rows_get_several_rows_per_wave() {
    use amdgpu_kernels::rows::rows_per_wave;
    assert_eq!((rows_per_wave(200), rows_per_wave(776), rows_per_wave(1024), rows_per_wave(8192)), (8, 2, 1, 1));
    // short rows: g lanes per row (a power of two >= the chunk count, at least 4), 64 / g rows side by side
    use amdgpu_kernels::rows::group_lanes;
    assert_eq!((group_lanes(8), group_lanes(64), group_lanes(200), group_lanes(256), group_lanes(504), group_lanes(512)), (4, 8, 32, 32, 64, 64));
    assert_eq!(rows_per_wave(8), 16);
}