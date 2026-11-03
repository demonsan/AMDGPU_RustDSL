//! The MXFP8 blockscale GEMM (`amdgpu-kernels::gemm_mxfp8`): every variant builds (its build-time layout proofs - DMA source = layout of the LDS
//! destination for sampled threads and issues - run inside the builder and panic on a mismatch), compiles, and the shipped variant passes the strict
//! GPU-semantics verification with a small trust ledger.
use amdgpu_compiler::gpu_verify::{analyse, Severity};

const VARIANTS: &[&str] = &["gemm_mxfp8_bs_v2", "gemm_mxfp8_bs_v3", "gemm_mxfp8_bs_v4"];

#[test]
fn gemm_variants_build_with_their_layout_proofs_and_compile() {
    for name in VARIANTS {
        let f = amdgpu_compiler::kernel_by_name(name).unwrap_or_else(|| panic!("unknown kernel {name}"));
        amdgpu_compiler::compile(&f).unwrap_or_else(|e| panic!("{name}: compile failed: {e}"));
    }
}

#[test]
fn gemm_v3_passes_strict_verification() {
    let f = amdgpu_compiler::kernel_by_name("gemm_mxfp8_bs_v3").expect("v3");
    let rep = analyse(&f).expect("analyse");
    let errs: Vec<String> = rep.errors().iter().map(|e| format!("[{}] {}", e.code, e.msg)).collect();
    assert!(errs.is_empty(), "{}", errs.join("\n"));
    let warns = rep.findings.iter().filter(|x| x.severity == Severity::Warning).count();
    assert_eq!(warns, 0, "{}", rep.render());
    // the trust ledger: the pipeline tile counter + the two XCD-raster permutation bounds
    let trusted = rep.findings.iter().filter(|x| x.code == "GV-ASSUME" && x.msg.contains("TRUSTED")).count();
    assert!(trusted <= 3, "trusted assumes: {trusted}\n{}", rep.render());
}
