//! The MXFP8 blockscale GEMM (`amdgpu-kernels::gemm_mxfp8`): every variant builds (its build-time layout proofs - DMA source = layout of the LDS
//! destination for sampled threads and issues - run inside the builder and panic on a mismatch), compiles, and the shipped variant passes the strict
//! GPU-semantics verification with a small trust ledger.
use amdgpu_compiler::gpu_verify::{analyse, Severity};

const VARIANTS: &[&str] = &["gemm_mxfp8_bs_v4"];

#[test]
fn gemm_variants_build_with_their_layout_proofs_and_compile() {
    for name in VARIANTS {
        let f = amdgpu_compiler::kernel_by_name(name).unwrap_or_else(|| panic!("unknown kernel {name}"));
        amdgpu_compiler::compile(&f).unwrap_or_else(|e| panic!("{name}: compile failed: {e}"));
    }
}

#[test]
fn gemm_v4_passes_strict_verification() {
    for name in ["gemm_mxfp8_bs_v4"] {
        let f = amdgpu_compiler::kernel_by_name(name).expect("kernel");
        let rep = analyse(&f).expect("analyse");
        let errs: Vec<String> = rep.errors().iter().map(|e| format!("[{}] {}", e.code, e.msg)).collect();
        assert!(errs.is_empty(), "{name}:\n{}", errs.join("\n"));
        let warns = rep.findings.iter().filter(|x| x.severity == Severity::Warn).count();
        assert_eq!(warns, 0, "{name}: {}", rep.render());
        // the trust ledger: the pipeline tile counter + the two XCD-raster permutation bounds
        let l = rep.ledger();
        assert!(l.total() <= 3 && l.tile_trust == 0 && l.region_trust == 0, "{name}: ledger {l:?}");
    }
}