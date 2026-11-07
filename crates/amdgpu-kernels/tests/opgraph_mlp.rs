//! The MLP op graph and its generated skeletons (P3): the committed graph JSON and skeleton sources are what the generator produces now (`UPDATE_SKELETONS=1` rewrites them), and the cuts of the
//! graph give the fused kernel (one kernel) and the unfused pair (C4 at `h`).
use amdgpu_kernels::mlp::graph::graph;
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