//! The op graph of RMSNorm + per-token fp8 quantization (design section 4.4): load -> rms (R) -> quant (E + R + Q) -> store. One wave per row, straight-line (no loop): the stages are the
//! `norm` kernel's `Node::Stage` regions, the edges their hand-overs (all register-resident, C0). `N` is a stage parameter (here 1024); the generated skeleton is the same for every N.

use amdgpu_graph::opgraph::{Class, Edge, Graph, Node, Phase, Prim, Resources, Scope, Store, Val};

fn node(id: &str, prim: Prim, stage: &str, params: Vec<(&str, Val)>, vgprs: u32) -> Node {
    Node { id: id.into(), prim, stage: stage.into(), params: params.into_iter().map(|(k, v)| (k.to_string(), v)).collect(), resources: Resources { vgprs, agprs: 0, lds_bytes: 0, barriers: 0 }, roles: vec!["all 4 waves: one row each".into()], barrier_phase: 0, phase: Phase::Prologue, hook: None }
}

fn edge(name: &str, from: Option<&str>, to: &[&str], dtype: &str, shape: &[i64], layout: &str, scope: Scope, store: Store, hint: Class, abi: Option<(u32, u32)>) -> Edge {
    Edge { name: name.into(), from: from.map(|s| s.into()), to: to.iter().map(|s| s.to_string()).collect(), dtype: dtype.into(), shape: shape.to_vec(), layout: layout.into(), producer: None, consumer: None, scope, store, class_hint: Some(hint), abi, carried: false }
}

pub fn graph() -> Graph {
    let n = 1024i64;
    let w = Scope::Wave;
    Graph {
        name: "rmsnorm_quant_fp8".into(),
        attrs: vec![("threads".into(), Val::Int(256))],
        nodes: vec![
            node("load", Prim::M, "move.row_load", vec![("n", Val::Int(n)), ("dtype", Val::Str("bf16".into()))], 64),
            node("rms", Prim::R, "reduce.row_sumsq", vec![("op", Val::Str("sum x^2 / n".into()))], 80),
            node("quant", Prim::Q, "quant.fp8_row", vec![("format", Val::Str("e4m3".into())), ("granularity", Val::Str("per-token".into())), ("scale", Val::Str("f32".into()))], 96),
            node("store", Prim::M, "move.row_store", vec![("dtype", Val::Str("fp8_e4m3 + f32 scale".into()))], 24),
        ],
        edges: vec![
            edge("x", None, &["load"], "bf16", &[-1, n], "row-major", Scope::None, Store::Global, Class::C4, Some((0, 8))),
            edge("gamma", None, &["load"], "bf16", &[n], "vector", Scope::None, Store::Global, Class::C4, Some((8, 8))),
            edge("x_rows", Some("load"), &["rms", "quant"], "bf16", &[1, n], "packed bf16, 8 per 16-byte fragment (lane l holds elements 8 (l + 64 i) + k)", w, Store::Vgpr, Class::C0, None),
            edge("gamma_rows", Some("load"), &["quant"], "bf16", &[1, n], "packed bf16 fragments", w, Store::Vgpr, Class::C0, None),
            edge("inv_rms", Some("rms"), &["quant"], "f32", &[1], "per-row scalar (wave)", w, Store::Vgpr, Class::C0, None),
            edge("q_rows", Some("quant"), &["store"], "fp8", &[1, n], "fp8 words in accumulator-like order (4 per dword)", w, Store::Vgpr, Class::C0, None),
            edge("scale_row", Some("quant"), &["store"], "f32", &[1], "per-row scalar (wave)", w, Store::Vgpr, Class::C0, None),
            edge("q", Some("store"), &[], "fp8", &[-1, n], "row-major", Scope::None, Store::Global, Class::C4, Some((16, 8))),
            edge("scale", Some("store"), &[], "f32", &[-1], "vector", Scope::None, Store::Global, Class::C4, Some((24, 8))),
        ],
        groups: Vec::new(),
    }
}