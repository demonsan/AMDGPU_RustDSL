//! The MLP op graph and its generated skeletons (P3): the committed graph JSON and skeleton sources are what the generator produces now (`UPDATE_SKELETONS=1` rewrites them), and the cuts of the
//! graph give the fused kernel (one kernel) and the unfused pair (C4 at `h`).
use amdgpu_kernels::mlp::graph::{graph, graph_unfused};
use amdgpu_tile::opgraph::skeleton::generate;
use amdgpu_tile::opgraph::*;

fn file(rel: &str) -> String {
    format!("{}/src/mlp/{rel}", env!("CARGO_MANIFEST_DIR"))
}

fn up_to_date(rel: &str, actual: &str) {
    let path = file(rel);
    if std::env::var("UPDATE_SKELETONS").is_ok() {
        std::fs::write(&path, actual).unwrap();
        return;
    }
    let have = std::fs::read_to_string(&path).unwrap_or_default().replace("\r\n", "\n");
    assert_eq!(have, actual, "{rel} is stale: regenerate with UPDATE_SKELETONS=1");
}

fn ids(g: &Graph) -> Vec<&str> {
    g.nodes.iter().map(|n| n.id.as_str()).collect()
}

#[test]
fn the_graph_json_and_the_fused_skeleton_are_up_to_date() {
    let g = graph();
    g.validate().expect("valid");
    up_to_date("mlp_fp8.graph.json", &g.to_json_string());
    assert_eq!(Graph::from_json_str(&g.to_json_string()).unwrap(), g, "the JSON round-trips");
    up_to_date("skeleton_fused.rs", &generate(&g, "fused", &ids(&g), "mlp_fp8.graph.json").unwrap());
}

#[test]
fn the_fused_kernel_keeps_h_on_chip_at_class_c0() {
    let g = graph();
    let r = g.kernel_report(&ids(&g)).unwrap();
    let h = r.internal.iter().find(|e| e.tensor == "h").expect("h is on chip");
    assert_eq!(h.class, Class::C0, "the P hand-off is a register rename (sigma absorbed by the pre-permuted W2)");
    assert_eq!(r.stages, vec!["w1_image", "w2_image", "x_first", "gemm1_act", "x_next", "gemm2"], "dependence order, loop-carried edge ignored");
}

#[test]
fn cutting_h_gives_the_unfused_pair() {
    let g = graph();
    let cut = Cut {
        kernels: vec![
            KernelCut { name: "g1".into(), nodes: vec!["w1_image".into(), "x_first".into(), "x_next".into(), "gemm1_act".into()] },
            KernelCut { name: "g2".into(), nodes: vec!["w2_image".into(), "gemm2".into()] },
        ],
    };
    let r = g.evaluate_cut(&cut, Budget::default()).unwrap();
    assert_eq!(r.cut_edges.len(), 1);
    assert_eq!((r.cut_edges[0].tensor.as_str(), r.cut_edges[0].class), ("h", Class::C4));
    assert_eq!(r.kernels[0].outputs, vec!["h"]);
    assert!(r.kernels[1].inputs.contains(&"h".to_string()) && r.kernels[1].inputs.contains(&"w2".to_string()), "{:?}", r.kernels[1].inputs);
    let text = r.text(&g);
    assert!(text.contains("cut h gemm1_act -> gemm2 : C4"), "{text}");
}

#[test]
fn every_node_names_a_registry_stage_of_its_primitive() {
    let g = graph();
    let reg = amdgpu_tile::stage::registry();
    for n in &g.nodes {
        let d = reg.iter().find(|d| d.key == n.stage).unwrap_or_else(|| panic!("node {}: stage key {} is not in the registry", n.id, n.stage));
        assert_eq!(n.prim.code(), format!("{:?}", d.prim), "node {}: primitive differs from the registry's", n.id);
    }
}

const G1: &[&str] = &["w1_image", "x_first", "gemm1_act", "x_next"];
const G2: &[&str] = &["w2_image", "h_first", "h_next", "gemm2"];

#[test]
fn the_unfused_graph_json_and_both_cut_skeletons_are_up_to_date() {
    let g = graph_unfused();
    g.validate().expect("valid");
    up_to_date("mlp_fp8_unfused.graph.json", &g.to_json_string());
    assert_eq!(Graph::from_json_str(&g.to_json_string()).unwrap(), g);
    up_to_date("skeleton_g1.rs", &generate(&g, "g1", G1, "mlp_fp8_unfused.graph.json").unwrap());
    up_to_date("skeleton_g2.rs", &generate(&g, "g2", G2, "mlp_fp8_unfused.graph.json").unwrap());
}

#[test]
fn the_unfused_pair_is_two_kernels_joined_by_a_c4_edge() {
    let g = graph_unfused();
    let cut = Cut { kernels: vec![KernelCut { name: "g1".into(), nodes: G1.iter().map(|s| s.to_string()).collect() }, KernelCut { name: "g2".into(), nodes: G2.iter().map(|s| s.to_string()).collect() }] };
    let r = g.evaluate_cut(&cut, Budget::default()).unwrap();
    assert_eq!(r.cut_edges.len(), 1);
    assert_eq!((r.cut_edges[0].tensor.as_str(), r.cut_edges[0].class), ("h", Class::C4));
    assert_eq!(r.kernels[0].stages, vec!["w1_image", "x_first", "gemm1_act", "x_next"]);
    assert_eq!(r.kernels[1].stages, vec!["w2_image", "h_first", "h_next", "gemm2"]);
}
