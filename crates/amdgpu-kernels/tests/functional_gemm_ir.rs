//! E3: symbolic execution of the BUILT `gemm_mxfp8_bs_v4` Function (actual DSL IR, not a replica).
use amdgpu_kernels::gemm_mxfp8::{gemm_mxfp8_bs, GemmConfig};
use amdgpu_kernels::ir_proof::{check_trace, prove_ir};

fn run(drop: Option<i64>) -> amdgpu_kernels::ir_proof::Trace {
    if let Some(kb) = drop {
        std::env::set_var("GEMM_PROOF_DROP_KB", kb.to_string());
    } else {
        std::env::remove_var("GEMM_PROOF_DROP_KB");
    }
    // v4 with the registered direct-C epilogue (`gemm_mxfp8_bs_v4_directc` config): the LDS
    // epilogue is a byte-transposition path whose provenance is a separate proof obligation; the
    // K/DMA/MFMA/scale program below is the real v4 IR. See docs/functional-proofs.md.
    let cfg = GemmConfig { lds_epilogue: false, ..GemmConfig::v4() };
    let f = gemm_mxfp8_bs("proof_gemm_v4_directc", cfg);
    std::env::remove_var("GEMM_PROOF_DROP_KB");
    // one CTA, 3 K tiles (the pipeline's u2 loop is unrolled by the executor)
    prove_ir(&f, &cfg, 256, 256, 384).expect("IR symbolic execution")
}

#[test]
fn actual_v4_ir_proves_every_output_k_block_and_store() {
    let t = run(None);
    println!("E3 trace: insts={} stores={} c_count={} contrib_vars={} unknown={:?} bstores={} no_origin={} lds_so={} lds_ro={} store={:?} store_orig={} left={} p1left={} p1so={} p2so={} read={:?} epi_read={:?}", t.executed_insts, t.stores.len(), t.c_count.len(), t.contrib.len(), t.unknown_ops, t.buffer_stores, t.buffer_store_no_origin, t.lds_store_origins, t.lds_read_origins, t.sample_store, t.sample_store_origin, t.lds_origins_left, t.pass1_origin_left, t.pass1_lds_store_origins, t.pass2_lds_store_origins, t.sample_read, t.sample_epi_read);
    let mut kbs: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
    for &(_, _, kb) in t.c_count.keys() { kbs.insert(kb); }
    let mut hist: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
    for v in t.contrib.values() { *hist.entry(v.len()).or_insert(0) += 1; }
    println!("E3 kb keys={kbs:?} contrib_hist={hist:?} mfma={} no_tags={} dma={} lds={} tags={}", t.mfma_seen, t.mfma_no_tags, t.dma_writes, t.lds_entries, t.tag_entries);
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
