//! The LDS accesses of the sampling / top-k / MoE-sorting / MXFP4-norm kernels are all inside declared tile markers (docs/trust-inventory.md: `lds-untracked`): the protocol engine checks their cross-wave order
//! and visibility, not only their bounds. A new unmarked LDS access in these families fails here.
use amdgpu_codegen::gpu_verify::{verify_ir, Severity};
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
        "moe_count_e128_k8",
        "moe_scan_e128_k8_b32",
        "moe_scan_e256_k8_b128",
        "moe_scatter_e128_k8_b32",
        "moe_scatter_ord_e128_k8_b32",
        "moe_scatter_ord_e256_k8_b128",
        "rmsnorm_quant_mxfp4_bf16_k4096",
    ];
    for n in names {
        let f = kernel_by_name(n).unwrap_or_else(|| panic!("{n} is not registered"));
        let r = verify_ir(&f);
        assert!(r.findings.iter().all(|x| x.severity != Severity::Error), "{n}: {}", r.render());
        assert_eq!(r.unchecked.get("lds-untracked").copied().unwrap_or(0), 0, "{n}: {:?}", r.unchecked);
    }
}

/// The race the tile protocol found in `moe_sorting`: the padding loop reads other threads' cursor words. Without the publishing barrier the verifier must reject the kernel (`GV-TILE-STALE`), with it the
/// shipped kernel verifies clean: the race-detection claim of the tile markers, evidenced on a real kernel.
#[test]
fn moe_sorting_without_the_cursor_barrier_is_rejected() {
    use amdgpu_kernels::moe::{moe_sorting, moe_sorting_unpublished};
    let bad = verify_ir(&moe_sorting_unpublished("moe_sorting_e128_k8_b32_unpublished", 128, 8, 32));
    let errs: Vec<_> = bad.findings.iter().filter(|x| x.severity == Severity::Error).collect();
    assert!(!errs.is_empty(), "the unsynchronised cursor read must be an error: {}", bad.render());
    assert!(bad.render().contains("GV-TILE-STALE"), "expected GV-TILE-STALE: {}", bad.render());
    let good = verify_ir(&moe_sorting("moe_sorting_e128_k8_b32", 128, 8, 32));
    assert!(good.findings.iter().all(|x| x.severity != Severity::Error), "{}", good.render());
}
