//! The FMHA op graphs and their generated dispatched skeletons (P3): the committed JSON / Rust files are what the generator produces now (`UPDATE_SKELETONS=1` rewrites them).
use amdgpu_kernels::fmha::graph::{graph_kvsplit, graph_t128, graph_vs128};
use amdgpu_graph::opgraph::skeleton::generate;
use amdgpu_graph::opgraph::*;

fn file(rel: &str) -> String {
    format!("{}/src/fmha/{rel}", env!("CARGO_MANIFEST_DIR"))
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

fn all() -> Vec<(&'static str, Graph)> {
    vec![("vs128", graph_vs128()), ("kvsplit", graph_kvsplit()), ("t128", graph_t128())]
}

#[test]
fn the_graph_jsons_and_the_dispatched_skeletons_are_up_to_date() {
    for (k, g) in all() {
        g.validate().unwrap_or_else(|e| panic!("{k}: {e}"));
        let json = format!("{k}.graph.json");
        up_to_date(&json, &g.to_json_string());
        assert_eq!(Graph::from_json_str(&g.to_json_string()).unwrap(), g, "{k}");
        let ids: Vec<&str> = g.nodes.iter().map(|n| n.id.as_str()).collect();
        up_to_date(&format!("skeleton_{k}.rs"), &generate(&g, k, &ids, &json).unwrap());
    }
}

#[test]
fn s_to_p_is_c0_and_the_epilogue_hand_off_is_a_wave_private_transpose() {
    for (k, g) in all() {
        let ids: Vec<&str> = g.nodes.iter().map(|n| n.id.as_str()).collect();
        let r = g.kernel_report(&ids).unwrap();
        let cls = |t: &str| r.internal.iter().find(|e| e.tensor == t).unwrap().class;
        assert_eq!(cls("s"), Class::C0, "{k}");
        assert_eq!(cls("p"), Class::C0, "{k}");
        assert_eq!(cls("kv"), Class::C3, "{k}: the ring is a block-scope LDS tile");
        assert_eq!(cls("o"), Class::C2, "{k}");
        assert_eq!(r.stages, vec!["kv_stage", "softmax", if k == "t128" { "qk" } else { "pv" }, if k == "t128" { "pv" } else { "qk" }, r.stages[4].as_str()], "{k}: dependence order");
    }
}

#[test]
fn every_node_names_a_registry_stage_of_its_primitive() {
    let reg = amdgpu_stage::stage::registry();
    for (k, g) in all() {
        for n in &g.nodes {
            let d = reg.iter().find(|d| d.key == n.stage).unwrap_or_else(|| panic!("{k}/{}: stage key {} is not in the registry", n.id, n.stage));
            assert_eq!(n.prim.code(), format!("{:?}", d.prim), "{k}/{}", n.id);
        }
    }
}