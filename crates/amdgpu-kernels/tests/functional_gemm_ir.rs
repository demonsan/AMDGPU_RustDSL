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
    println!("E3 trace: insts={} stores={} c_count={} contrib_vars={} unknown={:?} bstores={} no_origin={} lds_so={} lds_ro={} store={:?} store_orig={} left={} p1left={} p1so={} p2so={} read={:?} epi_read={:?}", t.executed_insts, t.stores.len(), t.c_count.len(), t.contrib.len(), t.unknown_ops, t.buffer_stores, t.buffer_store_no_origin, t.lds_store_origins, t.lds_read_origins, t.sample_store, t.sample_store_origin, t.lds_origins_left, t.pass1_origin_left, t.pass1_lds_store_origins, t.pass2_lds_store_origins, t.sample_read, t.sample_epi_read);
    check_trace(&t, &GemmConfig::v4(), 256, 256, 384, None).expect("E3 theorem on actual IR");
}

#[test]
fn actual_kernel_tile1_bug_has_a_concrete_counterexample() {
    let t = run(Some(1));
    println!("E3 bug trace: insts={} stores={} c_count={} contrib_vars={} unknown={:?} bstores={} no_origin={} lds_so={} lds_ro={} store={:?} store_orig={} left={} p1left={} p1so={} p2so={} read={:?} epi_read={:?}", t.executed_insts, t.stores.len(), t.c_count.len(), t.contrib.len(), t.unknown_ops, t.buffer_stores, t.buffer_store_no_origin, t.lds_store_origins, t.lds_read_origins, t.sample_store, t.sample_store_origin, t.lds_origins_left, t.pass1_origin_left, t.pass1_lds_store_origins, t.pass2_lds_store_origins, t.sample_read, t.sample_epi_read);
    let e = check_trace(&t, &GemmConfig::v4(), 256, 256, 384, Some(1)).expect_err("dropped tile must leave a hole");
    assert!(e.contains("C[0,0]") && e.contains("K block 1"), "counterexample: {e}");
    println!("E3 actual-kernel counterexample: {e}");
}
