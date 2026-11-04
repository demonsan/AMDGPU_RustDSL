//! E3: symbolic execution of the BUILT `gemm_mxfp8_bs_v4` Function (actual DSL IR, not a replica).
use amdgpu_kernels::gemm_mxfp8::{gemm_mxfp8_bs_v4, GemmConfig};
use amdgpu_kernels::ir_proof::{check_trace, prove_ir};

fn run(drop: Option<i64>) -> amdgpu_kernels::ir_proof::Trace {
    if let Some(kb) = drop {
        std::env::set_var("GEMM_PROOF_DROP_KB", kb.to_string());
    } else {
        std::env::remove_var("GEMM_PROOF_DROP_KB");
    }
    let f = gemm_mxfp8_bs_v4();
    std::env::remove_var("GEMM_PROOF_DROP_KB");
    let cfg = GemmConfig::v4();
    // one CTA, 3 K tiles (the pipeline's u2 loop is unrolled by the executor)
    prove_ir(&f, &cfg, 256, 256, 384).expect("IR symbolic execution")
}

#[test]
fn actual_v4_ir_proves_every_output_k_block_and_store() {
    let t = run(None);
    check_trace(&t, &GemmConfig::v4(), 256, 256, 384, None).expect("E3 theorem on actual IR");
    println!("E3 actual-IR trace: insts={} stores={} contributions={}", t.executed_insts, t.stores.len(), t.c_count.len());
}

#[test]
fn actual_kernel_tile1_bug_has_a_concrete_counterexample() {
    let t = run(Some(1));
    let e = check_trace(&t, &GemmConfig::v4(), 256, 256, 384, Some(1)).expect_err("dropped tile must leave a hole");
    assert!(e.contains("C[0,0]") && e.contains("K block 1"), "counterexample: {e}");
    println!("E3 actual-kernel counterexample: {e}");
}
