//! The RMSNorm + quant op graph and its generated skeleton (P3, design section 4.4).
use amdgpu_kernels::norm::graph::graph;
use amdgpu_graph::opgraph::skeleton::generate;
use amdgpu_graph::opgraph::*;

fn file(rel: &str) -> String {
    format!("{}/src/norm/{rel}", env!("CARGO_MANIFEST_DIR"))
}

fn up_to_date(rel: &str, actual: &str) {
    let path = file(rel);
    if std::env::var("UPDATE_SKELETONS").map_or(false, |v| !v.is_empty()) {
        std::fs::write(&path, actual).unwrap();
        return;
    }
    let have = std::fs::read_to_string(&path).unwrap_or_default().replace("\r\n", "\n");
    assert_eq!(have, actual, "{rel} is stale: regenerate with UPDATE_SKELETONS=1");
}

#[test]
fn the_graph_json_and_the_skeleton_are_up_to_date() {
    let g = graph();
    g.validate().expect("valid");
    up_to_date("rmsnorm.graph.json", &g.to_json_string());
    assert_eq!(Graph::from_json_str(&g.to_json_string()).unwrap(), g);
    let ids: Vec<&str> = g.nodes.iter().map(|n| n.id.as_str()).collect();
    up_to_date("skeleton_rmsnorm.rs", &generate(&g, "rmsnorm", &ids, "rmsnorm.graph.json", amdgpu_kernels::stage_registry::registry()).unwrap());
}

#[test]
fn every_hand_over_stays_in_registers() {
    let g = graph();
    let ids: Vec<&str> = g.nodes.iter().map(|n| n.id.as_str()).collect();
    let r = g.kernel_report(&ids).unwrap();
    assert_eq!(r.stages, vec!["load", "rms", "quant", "store"]);
    assert!(r.internal.iter().all(|e| e.class == Class::C0), "{:?}", r.internal);
    assert_eq!(r.internal.len(), 5);
}

#[test]
fn every_node_names_a_registry_stage_of_its_primitive() {
    let reg = amdgpu_kernels::stage_registry::registry();
    for n in &graph().nodes {
        let d = reg.iter().find(|d| d.key == n.stage).unwrap_or_else(|| panic!("node {}: stage key {} is not in the registry", n.id, n.stage));
        assert_eq!(n.prim.code(), format!("{:?}", d.prim), "node {}", n.id);
    }
}