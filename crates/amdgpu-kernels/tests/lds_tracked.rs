//! The LDS accesses of the sampling / top-k / MoE-sorting / MXFP4-norm kernels are all inside declared tile markers (docs/trust-inventory.md: `lds-untracked`): the protocol engine checks their cross-wave order
//! and visibility, not only their bounds. A new unmarked LDS access in these families fails here.
use amdgpu_compiler::gpu_verify::{verify_ir, Severity};
use amdgpu_kernels::kernel_by_name;

#[test]
fn the_reduction_and_sorting_kernels_have_no_untracked_lds_accesses() {
    let names = [
        "argmax_bf16_v32000",
        "argmax_bf16_v8",
        "sample_bf16_v32000",
        "sample_topkp_bf16_v32000",
        "sample_topkp_bf16_v128256",
        "topk_per_row_decode_f32_k128",
        "topk_per_row_prefill_f32_k128",
        "topk_per_row_decode_f32_k2048",
        "topk_plain_f32_k8",
        "moe_sorting_e128_k8_b32",
        "moe_sorting_e256_k8_b128",
        "moe_sorting_e8_k1_b16",
        "rmsnorm_quant_mxfp4_bf16_k4096",
    ];
    for n in names {
        let f = kernel_by_name(n).unwrap_or_else(|| panic!("{n} is not registered"));
        let r = verify_ir(&f);
        assert!(r.findings.iter().all(|x| x.severity != Severity::Error), "{n}: {}", r.render());
        assert_eq!(r.unchecked.get("lds-untracked").copied().unwrap_or(0), 0, "{n}: {:?}", r.unchecked);
    }
}