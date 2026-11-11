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