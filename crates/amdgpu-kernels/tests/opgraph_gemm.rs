//! The GEMM v4 op graph and its generated pipelined skeleton (P3): the committed graph JSON and skeleton source are what the generator produces now (`UPDATE_SKELETONS=1` rewrites them).
use amdgpu_kernels::gemm::graph::graph;
use amdgpu_graph::opgraph::skeleton::generate;
use amdgpu_graph::opgraph::*;

fn file(rel: &str) -> String {
    format!("{}/src/gemm/{rel}", env!("CARGO_MANIFEST_DIR"))
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
fn the_graph_json_and_the_pipelined_skeleton_are_up_to_date() {
    let g = graph();
    g.validate().expect("valid");
    up_to_date("gemm_v4.graph.json", &g.to_json_string());
    assert_eq!(Graph::from_json_str(&g.to_json_string()).unwrap(), g);
    let ids: Vec<&str> = g.nodes.iter().map(|n| n.id.as_str()).collect();
    up_to_date("skeleton_v4.rs", &generate(&g, "v4", &ids, "gemm_v4.graph.json").unwrap());
}

#[test]
fn the_tile_ring_is_a_block_scope_edge_and_the_accumulators_stay_in_registers() {
    let g = graph();
    let ids: Vec<&str> = g.nodes.iter().map(|n| n.id.as_str()).collect();
    let r = g.kernel_report(&ids).unwrap();
    let cls = |t: &str| r.internal.iter().find(|e| e.tensor == t).unwrap().class;
    assert_eq!(cls("tiles"), Class::C3);
    assert_eq!(cls("acc"), Class::C0);
    assert_eq!(r.stages, vec!["ab_stage", "tile", "epilogue"]);
}

#[test]
fn every_node_names_a_registry_stage_of_its_primitive() {
    let reg = amdgpu_stage::stage::registry();
    for n in &graph().nodes {
        let d = reg.iter().find(|d| d.key == n.stage).unwrap_or_else(|| panic!("node {}: stage key {} is not in the registry", n.id, n.stage));
        assert_eq!(n.prim.code(), format!("{:?}", d.prim), "node {}", n.id);
    }
}