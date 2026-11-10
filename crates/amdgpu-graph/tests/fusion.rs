//! The L1 launch plans of the golden op graphs (`tests/golden/graph_plans/<name>.json` -> `<name>.plan.json`, docs/l1-graph.md §7.2), plus
//! the rule decisions the negative graphs must show. Regenerate a plan after an intended rule change with
//! `amdc graph plan tests/golden/graph_plans/<name>.json > tests/golden/graph_plans/<name>.plan.json`.

use amdgpu_graph::fusion::{plan, ModelGraph, Plan};
use std::path::PathBuf;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/graph_plans")
}

fn graphs() -> Vec<(String, String)> {
    let mut v: Vec<(String, String)> = std::fs::read_dir(dir())
        .expect("tests/golden/graph_plans")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| n.ends_with(".json") && !n.ends_with(".plan.json"))
        .map(|n| (n.trim_end_matches(".json").to_string(), std::fs::read_to_string(dir().join(&n)).unwrap()))
        .collect();
    v.sort();
    v
}

fn plan_of(name: &str) -> Plan {
    let text = std::fs::read_to_string(dir().join(format!("{name}.json"))).unwrap();
    plan(&ModelGraph::from_json_str(&text).unwrap_or_else(|e| panic!("{name}: {e}")))
}

#[test]
fn golden_plans_match() {
    let mut bad = Vec::new();
    for (name, text) in graphs() {
        let got = plan(&ModelGraph::from_json_str(&text).unwrap_or_else(|e| panic!("{name}: {e}"))).to_json().to_string_pretty();
        let want = std::fs::read_to_string(dir().join(format!("{name}.plan.json"))).unwrap_or_default();
        if got != want.replace("\r\n", "\n") {
            bad.push(format!("{name}: plan differs (regenerate with `amdc graph plan tests/golden/graph_plans/{name}.json`):\n{got}"));
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

fn rules(p: &Plan) -> Vec<&str> {
    p.launches.iter().map(|l| l.rule).collect()
}

#[test]
fn fused_rules_fire() {
    for (g, r, k) in [
        ("rmsnorm_quant_fp8", "rmsnorm+quant_token.fp8", "rmsnorm_quant_fp8_n4096"),
        ("rmsnorm_quant_mxfp4", "rmsnorm+quant_mx.fp4", "rmsnorm_quant_mxfp4_bf16_k4096"),
        ("add_rmsnorm", "add+rmsnorm", "add_rmsnorm_bf16_n7168"),
        ("add_rmsnorm_quant_fp8", "add+rmsnorm+quant_token.fp8", "add_rmsnorm_fp8_n4096"),
        ("silu_mul_quant_fp8", "act_mul.silu+quant_token.fp8", "silu_mul_quant_fp8_d4096"),
        ("qknorm_rope", "qknorm+rope", "qknorm_rope_neox_bf16_h32_d128"),
        ("qknorm_rope_kvcache_fp8", "qknorm+rope+kvcache", "qknorm_rope_kvcache_neox_fp8_h8_d128_i64"),
    ] {
        let p = plan_of(g);
        assert_eq!(rules(&p), vec![r], "{g}");
        assert_eq!(p.launches[0].kernels, vec![k.to_string()], "{g}");
        assert!(p.errors.is_empty(), "{g}: {:?}", p.errors);
    }
    let p = plan_of("decoder_layer");
    assert_eq!(rules(&p), vec!["add+rmsnorm+quant_token.fp8", "act_mul.silu+quant_token.fp8"]);
}

#[test]
fn negatives_fall_back_with_a_reason() {
    let p = plan_of("neg_interior_consumer");
    assert_eq!(rules(&p), vec!["rmsnorm", "quant_token", "quant_token"]);
    assert!(p.rejected.iter().any(|r| r.rule == "rmsnorm+quant_token.fp8" && r.why.contains("also read by op quant8")), "{:?}", p.rejected);
    let p = plan_of("neg_interior_output");
    assert_eq!(rules(&p), vec!["rmsnorm", "quant_token"]);
    assert!(p.rejected.iter().any(|r| r.why.contains("graph output")), "{:?}", p.rejected);
    let p = plan_of("neg_unsupported_width");
    assert_eq!(p.launches.iter().map(|l| l.kernels[0].as_str()).collect::<Vec<_>>(), vec!["rmsnorm_bf16_n4104", "quant_token_bf16_fp8_n4104"]);
    assert!(p.rejected.iter().any(|r| r.why.contains("4104")), "{:?}", p.rejected);
    let p = plan_of("neg_fp8_cache_no_scale");
    assert!(p.rejected.iter().any(|r| r.rule == "qknorm+rope+kvcache" && r.why.contains("k_scale")), "{:?}", p.rejected);
    assert_eq!(rules(&p), vec!["qknorm+rope"]);
    assert_eq!(p.errors.len(), 1, "the fp8 cache write alone needs its scales too: {:?}", p.errors);
    let p = plan_of("neg_add_alone");
    assert!(p.launches.is_empty() && p.errors.len() == 1, "{:?}", p);
}

#[test]
fn per_shape_choices_come_from_dispatch() {
    let p = plan_of("moe_topk");
    let bm = amdgpu_graph::dispatch::moe_block_m(256, 8, 128);
    let moe = p.launches.iter().find(|l| l.rule == "moe_experts").expect("moe launch");
    assert_eq!(moe.kernels[0], format!("moe_sorting_e128_k8_b{bm}"));
    let p = plan_of("attention_prefill");
    let (k, s) = amdgpu_graph::dispatch::bf16_prefill_kernel(1, 8, 2048, 2048, true, 256);
    assert_eq!(p.launches[0].kernels[0], k);
    assert_eq!(p.launches[0].kernels.len(), if s > 1 { 2 } else { 1 });
}