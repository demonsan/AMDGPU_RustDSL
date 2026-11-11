//! `amdgpu_graph::compose_facts` is generated from the kernels' plans (`amdc compose-facts --out crates/amdgpu-graph/src/compose_facts.rs`).

use amdgpu_kernels::{compose_facts_gen, reachable, registry};

#[test]
fn the_compose_facts_match_the_kernels_plans() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../amdgpu-graph/src/compose_facts.rs");
    let have = std::fs::read_to_string(path).unwrap_or_default().replace("\r\n", "\n");
    assert_eq!(have, compose_facts_gen::source(), "crates/amdgpu-graph/src/compose_facts.rs is stale: a kernel plan changed: run `bash tools/regen_compose_facts.sh` (= `amdc compose-facts --out crates/amdgpu-graph/src/compose_facts.rs`) and commit the file");
}

#[test]
fn every_dispatched_kernel_with_a_plan_has_facts_and_a_whole_number_of_waves() {
    let mut with_plan = 0;
    for n in reachable::dispatched_names() {
        if let Some(f) = registry::kernel_by_name(&n) {
            if f.plan.is_some() {
                let facts = amdgpu_graph::compose::facts(&n).unwrap_or_else(|| panic!("`{n}` has a plan but no compose fact"));
                assert_eq!(facts.threads % 64, 0, "{n}");
                with_plan += 1;
            }
        }
    }
    assert!(with_plan > 20, "{with_plan}");
}

#[test]
fn every_gemm_search_candidate_is_a_registered_kernel() {
    use amdgpu_graph::compose_search::gemm_candidates;
    let mut n = 0;
    for kind in ["bf16", "i8_rowcol", "fp8_rowcol", "fp8_blockscale"] {
        for (m, nn, k) in [(4096u64, 4096u64, 8192u64), (1024, 4096, 8192), (256, 7168, 4096), (128, 4096, 8192), (64, 8192, 8192), (32, 4096, 14336)] {
            for c in gemm_candidates(kind, m, nn, k, 256) {
                assert!(registry::kernel_by_name(&c.kernel).is_some(), "{kind}: candidate kernel `{}` is not registered", c.kernel);
                if let Some(r) = &c.reduce {
                    assert!(registry::kernel_by_name(r).is_some(), "{kind}: reduce kernel `{r}` is not registered");
                }
                n += 1;
            }
        }
    }
    assert!(n > 40, "{n}");
}

#[test]
fn the_recorded_gemm_rule_is_among_the_search_candidates() {
    use amdgpu_graph::compose_search::gemm_candidates;
    for kind in ["bf16", "i8_rowcol", "fp8_rowcol", "fp8_blockscale"] {
        for (m, n, k) in [(4096u64, 4096u64, 8192u64), (1024, 4096, 8192), (256, 7168, 4096), (128, 4096, 8192), (64, 8192, 8192)] {
            let (kernel, s, _) = amdgpu_graph::dispatch::gemm_kernels(kind, m, n, k, 256).unwrap();
            let c = gemm_candidates(kind, m, n, k, 256);
            assert!(c.iter().any(|x| x.kernel == kernel && x.splits == s), "{kind} {m}x{n}x{k}: the rule's `{kernel}` S{s} is not a search candidate: {:?}", c.iter().map(|x| (&x.kernel, x.splits)).collect::<Vec<_>>());
        }
    }
}