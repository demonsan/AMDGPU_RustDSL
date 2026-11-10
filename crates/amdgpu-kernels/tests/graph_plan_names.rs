//! Every kernel name the L1 fusion rules plan for the golden op graphs (`tests/golden/graph_plans`, docs/l1-graph.md §7.1) is a registered
//! kernel: the rules' supported sets mirror the families' `by_name` parsers, and this test keeps the mirror honest.

use amdgpu_graph::fusion::{plan, ModelGraph};
use amdgpu_kernels::kernel_by_name;

#[test]
fn planned_kernels_are_registered() {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/graph_plans");
    let mut names = std::collections::BTreeSet::new();
    for e in std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()) {
        let n = e.file_name().to_string_lossy().to_string();
        if !n.ends_with(".json") || n.ends_with(".plan.json") {
            continue;
        }
        let g = ModelGraph::from_json_str(&std::fs::read_to_string(e.path()).unwrap()).unwrap();
        for l in plan(&g).launches {
            names.extend(l.kernels);
        }
    }
    assert!(names.len() >= 15, "too few planned kernels: {names:?}");
    let missing: Vec<&String> = names.iter().filter(|n| kernel_by_name(n).is_none()).collect();
    assert!(missing.is_empty(), "planned kernels that do not build: {missing:?}");
}