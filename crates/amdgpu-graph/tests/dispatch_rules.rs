//! The dispatch rules that replaced Python-side choices keep the Python behaviour they replaced (the former `amdgpu_jit.select_path` cases
//! of python/tests/test_unit.py).

use amdgpu_graph::dispatch::{gemm_kernels, mxfp8_prefill_path};

#[test]
fn mxfp8_prefill_path_matches_the_former_python_rule() {
    assert_eq!(mxfp8_prefill_path(4, 8, 2048, 2048), ("vs128", 0));
    assert_eq!(mxfp8_prefill_path(1, 8, 8192, 8192).0, "vs128");
    assert_eq!(mxfp8_prefill_path(2, 8, 2048, 2048), ("vs128", 0));
    assert_eq!(mxfp8_prefill_path(4, 8, 512, 512), ("t128", 0));
    assert_eq!(mxfp8_prefill_path(1, 8, 1024, 1024), ("t128", 0));
    assert_eq!(mxfp8_prefill_path(1, 4, 8192, 8192), ("kvsplit", 2));
    assert_eq!(mxfp8_prefill_path(1, 8, 2048, 2048), ("kvsplit", 4));
}

#[test]
fn gemm_kernels_follow_the_family_rules() {
    assert_eq!(gemm_kernels("bf16", 4096, 4096, 4096, 256), Some(("gemm_a16w16_bf16".into(), 1, None)));
    assert_eq!(gemm_kernels("bf16", 128, 4096, 8192, 256), Some(("gemm_a16w16_bf16_64_splitk".into(), 4, Some("gemm_splitk_reduce_s4".into()))));
    assert_eq!(gemm_kernels("fp8_rowcol", 64, 1024, 2048, 256), Some(("gemm_a8w8_rowcol_fp8_64_splitk".into(), 4, Some("gemm_splitk_reduce_s4_rc".into()))));
    // a short K runs unsplit (the split needs whole 256-element K pairs per part and pays a reduce): 64x1024x512 is `_64` S1 for both kinds
    assert_eq!(gemm_kernels("fp8_rowcol", 64, 1024, 512, 256).map(|x| (x.0, x.1)), Some(("gemm_a8w8_rowcol_fp8_64".into(), 1)));
    assert_eq!(gemm_kernels("bf16", 64, 1024, 512, 256).map(|x| (x.0, x.1)), Some(("gemm_a16w16_bf16_64".into(), 1)));
    assert_eq!(gemm_kernels("bf16", 64, 1024, 1024, 256).map(|x| x.1), Some(4));
    assert_eq!(gemm_kernels("fp8_blockscale", 256, 7168, 4096, 256).map(|x| x.0), Some("gemm_a8w8_blockscale_64".into()));
    // the reduce covers N in 1024-column blocks
    assert_eq!(gemm_kernels("bf16", 64, 1000, 512, 256).map(|x| x.1), Some(1));
    assert_eq!(gemm_kernels("fp4", 64, 1024, 512, 256), None);
}