//! The op graphs of the shipped FMHA forward kernels (design section 4.1): the input of the skeleton generator. Pipelined kernels with a DISPATCHED run (the two-group `_vs128` / `_kvsplit` run the
//! pipeline inside `Roles::each` / `versioned` branches; `_t128` is one group): the graph says which stage runs in which pipeline hook, which stages one weaver interleaves (weave groups) and what the
//! boundaries cost (S -> P is C0, O -> epilogue a wave-private LDS transpose, C2).

use amdgpu_tile::opgraph::{Class, Edge, Graph, Group, Node, Phase, Prim, Resources, Scope, Store, Val};

fn node(id: &str, prim: Prim, stage: &str, phase: Phase, hook: Option<&str>, params: Vec<(&str, Val)>, res: Resources, roles: &str) -> Node {
    Node { id: id.into(), prim, stage: stage.into(), params: params.into_iter().map(|(k, v)| (k.to_string(), v)).collect(), resources: res, roles: vec![roles.into()], barrier_phase: 0, phase, hook: hook.map(|h| h.to_string()) }
}

#[allow(clippy::too_many_arguments)]
fn edge(name: &str, from: Option<&str>, to: &[&str], dtype: &str, shape: &[i64], layout: &str, scope: Scope, store: Store, hint: Class, carried: bool) -> Edge {
    Edge { name: name.into(), from: from.map(|s| s.into()), to: to.iter().map(|s| s.to_string()).collect(), dtype: dtype.into(), shape: shape.to_vec(), layout: layout.into(), producer: None, consumer: None, scope, store, class_hint: Some(hint), abi: None, carried }
}

fn res(vgprs: u32, agprs: u32, lds_bytes: u32, barriers: u32) -> Resources {
    Resources { vgprs, agprs, lds_bytes, barriers }
}

/// shared edges: the inputs and the on-chip flow of one tile
fn flow(ring_slots: i64, out_stage: &str) -> Vec<Edge> {
    vec![
        edge("q", None, &["qk"], "fp8", &[-1, 128], "row-major (Q rows in registers)", Scope::None, Store::Global, Class::C4, false),
        edge("kv_in", None, &["kv_stage"], "fp8", &[-1, 128], "K / K-scale / V rows (range-checked buffer descriptors)", Scope::None, Store::Global, Class::C4, false),
        edge("kv", Some("kv_stage"), &["qk", "pv"], "fp8", &[128, 128], &format!("LDS ring ({ring_slots} slots), swizzled [key, d]; V read transposed"), Scope::Block, Store::Lds("ring".into()), Class::C3, false),
        edge("s", Some("qk"), &["softmax"], "f32", &[128, 128], "S^T accumulators (the next tile's, produced one phase ahead)", Scope::Wave, Store::Agpr, Class::C0, true),
        edge("p", Some("softmax"), &["pv"], "fp8", &[128, 128], "P words (pknorm + scalef32), the B operand of PV; K permutation absorbed by the V transposed read", Scope::Wave, Store::Vgpr, Class::C0, false),
        edge("o", Some("pv"), &[out_stage], "f32", &[128, 128], "O accumulators", Scope::Wave, Store::Agpr, Class::C2, false),
    ]
}

fn kv_stage(slots: i64) -> Node {
    node("kv_stage", Prim::M, "move.kv_stage", Phase::Loop, Some("stage"), vec![("slots", Val::Int(slots))], res(24, 0, 98_304, 1), "producer waves")
}

pub fn graph_vs128() -> Graph {
    let r = "two groups of 4 waves, offset by one phase";
    let mut edges = flow(2, "out");
    edges.push(edge("out", Some("out"), &[], "bf16", &[-1, 128], "row-major", Scope::None, Store::Global, Class::C4, false));
    Graph {
        name: "fmha_vs128".into(),
        attrs: vec![("threads".into(), Val::Int(512)), ("pipeline".into(), Val::Str("TwoPhaseOffset".into())), ("dispatch".into(), Val::Str("roles + versions".into()))],
        nodes: vec![
            kv_stage(2),
            node("softmax", Prim::O, "online.softmax", Phase::Loop, Some("v_phase"), vec![("recipe", Val::Str("vendor two-word reference, lazy rollback".into()))], res(96, 0, 0, 0), r),
            node("pv", Prim::A, "mma.pv", Phase::Loop, Some("m_phase"), vec![("atom", Val::Str("mfma_scale_32x32x64_f8".into()))], res(64, 128, 0, 0), r),
            node("qk", Prim::A, "mma.qk", Phase::Loop, Some("m_phase"), vec![("atom", Val::Str("mfma_scale_32x32x64_f8".into())), ("scale", Val::Str("K-scale byte select".into()))], res(64, 128, 0, 0), r),
            node("out", Prim::E, "map.store_bf16", Phase::Epilogue, None, vec![("n", Val::Int(128)), ("via", Val::Str("wave-private LDS transpose".into()))], res(64, 128, 32_768, 0), r),
        ],
        edges,
        groups: vec![Group { name: "m_phase".into(), label: "m_phase".into(), nodes: vec!["pv".into(), "qk".into()] }],
    }
}

pub fn graph_kvsplit() -> Graph {
    let r = "two groups of 4 waves, offset by one phase";
    let mut edges = flow(2, "partial");
    edges.push(edge("o_part", Some("partial"), &[], "f32", &[-1, 128], "partial O rows", Scope::None, Store::Global, Class::C4, false));
    edges.push(edge("lse", Some("partial"), &[], "f32", &[-1], "log-sum-exp per row", Scope::None, Store::Global, Class::C4, false));
    Graph {
        name: "fmha_kvsplit".into(),
        attrs: vec![("threads".into(), Val::Int(512)), ("pipeline".into(), Val::Str("TwoPhaseOffset".into())), ("dispatch".into(), Val::Str("roles + versions".into()))],
        nodes: vec![
            kv_stage(2),
            node("softmax", Prim::O, "online.softmax", Phase::Loop, Some("v_phase"), vec![("recipe", Val::Str("vendor two-word reference, lazy rollback".into()))], res(96, 0, 0, 0), r),
            node("pv", Prim::A, "mma.pv", Phase::Loop, Some("m_phase"), vec![("atom", Val::Str("mfma_scale_32x32x64_f8".into()))], res(64, 128, 0, 0), r),
            node("qk", Prim::A, "mma.qk", Phase::Loop, Some("m_phase"), vec![("atom", Val::Str("mfma_scale_32x32x64_f8".into()))], res(64, 128, 0, 0), r),
            node("partial", Prim::E, "map.epilogue_chain", Phase::Epilogue, None, vec![("ops", Val::Str("partial O + log-sum-exp".into()))], res(64, 128, 32_768, 0), r),
        ],
        edges,
        groups: vec![Group { name: "m_phase".into(), label: "m_phase".into(), nodes: vec!["pv".into(), "qk".into()] }],
    }
}

pub fn graph_t128() -> Graph {
    let r = "one group of 4 waves";
    let mut edges = flow(3, "out");
    edges.push(edge("out", Some("out"), &[], "bf16", &[-1, 128], "row-major", Scope::None, Store::Global, Class::C4, false));
    Graph {
        name: "fmha_t128".into(),
        attrs: vec![("threads".into(), Val::Int(256)), ("pipeline".into(), Val::Str("OneWindow".into())), ("dispatch".into(), Val::Str("versions".into()))],
        nodes: vec![
            kv_stage(3),
            node("softmax", Prim::O, "online.softmax", Phase::Loop, Some("window"), vec![("recipe", Val::Str("vendor two-word reference, lazy rollback".into()))], res(96, 0, 0, 0), r),
            node("qk", Prim::A, "mma.qk", Phase::Loop, Some("window"), vec![("atom", Val::Str("mfma_scale_32x32x64_f8".into()))], res(64, 128, 0, 0), r),
            node("pv", Prim::A, "mma.pv", Phase::Loop, Some("window"), vec![("atom", Val::Str("mfma_scale_32x32x64_f8".into()))], res(64, 128, 0, 0), r),
            node("out", Prim::E, "map.store_bf16", Phase::Epilogue, None, vec![("n", Val::Int(128))], res(64, 128, 0, 0), r),
        ],
        edges,
        groups: vec![Group { name: "window_weave".into(), label: "window".into(), nodes: vec!["softmax".into(), "qk".into()] }],
    }
}