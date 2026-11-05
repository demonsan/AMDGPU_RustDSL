//! Backend auto-selection: kernels with own-only (DMA) ops get the own backend under `auto`, everything else the LLVM
//! tail; explicit choices are untouched.

use amdgpu_compiler::backend::{LlvmTail, LlvmTools};
use amdgpu_compiler::{compile_with, kernel_by_name, kernel_names, resolve_backend, uses_own_only_ops};

const OWN_ONLY_BY_NAME: &[&str] = &["probe_smfmac_gen", "probe_lds_tr_b8_gen", "probe_stamp", "probe_dma", "probe_dma_tile", "probe_dma_seq", "probe_dma_kread", "probe_dma_map", "probe_buffer_lds", "probe_buffer_dyn", "probe_dma_voff_spill", "fmha_fwd_hd128_mxfp8_vs", "fmha_fwd_hd128_mxfp8_vs128"];

#[test]
fn dma_kernels_get_own_everything_else_llvm_tail() {
    for k in OWN_ONLY_BY_NAME {
        let Some(f) = kernel_by_name(k) else { continue };
        let (b, note) = resolve_backend("auto", &f);
        // some probes only emit DMA under particular FMHA_OPT strings: the policy follows what the MIR really contains
        assert_eq!(b == "own", uses_own_only_ops(&f), "{k}");
        assert_eq!(note.is_some(), b == "own", "{k}");
        if k.starts_with("fmha_fwd_hd128_mxfp8_vs") {
            assert_eq!(b, "own", "{k}: the vendor-structure kernels always use DMA");
            assert!(note.unwrap().contains("own-backend-only"));
        }
    }
    for k in kernel_names() {
        let f = kernel_by_name(k).unwrap();
        let (b, note) = resolve_backend("auto", &f);
        assert_eq!(b, "llvm-tail", "{k} must stay on the LLVM tail");
        assert!(note.is_none());
    }
    // explicit choices are never rewritten
    let f = kernel_by_name("fmha_fwd_hd128_mxfp8_vs").unwrap();
    assert_eq!(resolve_backend("llvm-tail", &f).0, "llvm-tail");
    assert_eq!(resolve_backend("own", &f).0, "own");
}

#[test]
fn auto_output_for_a_non_dma_kernel_is_the_llvm_tail_output() {
    let t = LlvmTools::locate();
    if !std::process::Command::new(t.tool("llc")).arg("--version").output().map(|o| o.status.success()).unwrap_or(false) {
        eprintln!("SKIPPED: llc not runnable");
        return;
    }
    let f = kernel_by_name("probe_store").unwrap();
    let (b, _) = resolve_backend("auto", &f);
    assert_eq!(b, "llvm-tail");
    let tail = LlvmTail::new();
    let a = compile_with(&f, &tail).unwrap().asm;
    let b2 = compile_with(&f, &tail).unwrap().asm;
    assert_eq!(a, b2, "deterministic");
}