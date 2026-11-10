//! The op graph of the shipped MXFP8 blockscale GEMM (`gemm_mxfp8_bs_v4`, design section 4.2): the input of the skeleton generator. One pipelined kernel: the A / B tile copies (`move.ab_stage`, the
//! pipeline's `stage` hook), the register-tile MMAs (`mma.blockscale_tile`, the `m_phase` hook) and the bf16 epilogue after the pipeline.

use amdgpu_graph::opgraph::{Class, Edge, Graph, Node, Phase, Prim, Resources, Scope, Store, Val};

fn node(id: &str, prim: Prim, stage: &str, phase: Phase, hook: Option<&str>, params: Vec<(&str, Val)>, res: Resources) -> Node {
    Node { id: id.into(), prim, stage: stage.into(), params: params.into_iter().map(|(k, v)| (k.to_string(), v)).collect(), resources: res, roles: vec!["4 waves (2 x 2), all the same program".into()], barrier_phase: 0, phase, hook: hook.map(|h| h.to_string()) }
}

fn edge(name: &str, from: Option<&str>, to: &[&str], dtype: &str, shape: &[i64], layout: &str, scope: Scope, store: Store, hint: Class, abi: Option<(u32, u32)>) -> Edge {
    Edge { name: name.into(), from: from.map(|s| s.into()), to: to.iter().map(|s| s.to_string()).collect(), dtype: dtype.into(), shape: shape.to_vec(), layout: layout.into(), producer: None, consumer: None, scope, store, class_hint: Some(hint), abi, carried: false }
}

pub fn graph() -> Graph {
    let nodes = vec![
        node("ab_stage", Prim::M, "move.ab_stage", Phase::Loop, Some("stage"), vec![("slots", Val::Int(2))], Resources { vgprs: 16, agprs: 0, lds_bytes: 140_000, barriers: 1 }),
        node(
            "tile",
            Prim::A,
            "mma.blockscale_tile",
            Phase::Loop,
            Some("m_phase"),
            vec![("atom", Val::Str("mfma_scale_16x16x128_f8".into())), ("wave_tile", Val::Str("128x128".into())), ("staged", Val::Bool(true))],
            Resources { vgprs: 192, agprs: 256, lds_bytes: 0, barriers: 0 },
        ),
        node("epilogue", Prim::E, "map.store_bf16", Phase::Epilogue, None, vec![("n", Val::Int(256))], Resources { vgprs: 64, agprs: 256, lds_bytes: 0, barriers: 0 }),
    ];
    let edges = vec![
        edge("a", None, &["ab_stage"], "fp8", &[-1, -1], "row-major (padded image in LDS)", Scope::None, Store::Global, Class::C4, Some((0, 8))),
        edge("b", None, &["ab_stage"], "fp8", &[-1, -1], "preshuffled", Scope::None, Store::Global, Class::C4, Some((8, 8))),
        edge("sfa", None, &["ab_stage"], "e8m0", &[-1, -1], "per-32 K block scales (transposed in the LDS panel)", Scope::None, Store::Global, Class::C4, Some((64, 8))),
        edge("sfb", None, &["ab_stage"], "e8m0", &[-1, -1], "per-128 block scales", Scope::None, Store::Global, Class::C4, Some((72, 8))),
        edge("tiles", Some("ab_stage"), &["tile"], "fp8", &[256, 128], "LDS ring (2 slots), padded A image + preshuffled B", Scope::Block, Store::Lds("ring".into()), Class::C3, None),
        edge("acc", Some("tile"), &["epilogue"], "f32", &[256, 256], "C accumulators", Scope::Wave, Store::Agpr, Class::C0, None),
        edge("c", Some("epilogue"), &[], "bf16", &[-1, -1], "row-major", Scope::None, Store::Global, Class::C4, Some((16, 8))),
    ];
    Graph {
        name: "gemm_v4".into(),
        attrs: vec![
            ("threads".into(), Val::Int(256)),
            ("pipeline".into(), Val::Str("RegStaged".into())),
            ("hooks".into(), Val::List(vec![Val::Str("prologue_stores".into()), Val::Str("pre_sync".into()), Val::Str("post_wait".into())])),
        ],
        nodes,
        edges,
        groups: Vec::new(),
    }
}