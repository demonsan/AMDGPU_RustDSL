//! The op graph (amdgpu_graph::opgraph): JSON round trip, computed boundary classes, cuts (C4 edges, scope / budget checks), the review text.
use amdgpu_graph::opgraph::json::Json;
use amdgpu_graph::opgraph::*;

fn node(id: &str, prim: Prim, stage: &str, vgprs: u32) -> Node {
    Node { id: id.into(), prim, stage: stage.into(), params: vec![("k".into(), Val::Int(256)), ("act".into(), Val::Str("relu".into()))], resources: Resources { vgprs, agprs: 0, lds_bytes: 1024, barriers: 0 }, roles: vec!["all".into()], barrier_phase: 0, phase: Phase::Loop, hook: None }
}

fn frag(name: &str, table: Vec<i64>) -> FragLayout {
    FragLayout { name: name.into(), threads: 64, values: 8, elem_bits: 32, table }
}
fn identity() -> Vec<i64> {
    (0..64 * 8).collect()
}
/// lane t holds what lane t ^ 1 held (a quad-local swap)
fn swapped() -> Vec<i64> {
    (0..64 * 8).map(|i| { let (t, v) = (i % 64, i / 64); (t ^ 1) + 64 * v }).collect()
}

fn edge(name: &str, from: Option<&str>, to: &[&str], hint: Option<Class>) -> Edge {
    Edge { name: name.into(), from: from.map(|s| s.into()), to: to.iter().map(|s| s.to_string()).collect(), dtype: "f32".into(), shape: vec![32, 64], layout: "row-major".into(), producer: None, consumer: None, scope: Scope::Wave, store: Store::Vgpr, class_hint: hint, abi: None, carried: false }
}

/// x -> A -> h -> E -> g -> B -> y
fn chain() -> Graph {
    let mut e1 = edge("h", Some("A"), &["E"], None);
    e1.producer = Some(frag("h", identity()));
    e1.consumer = Some(frag("h", identity()));
    let mut x = edge("x", None, &["A"], None);
    x.abi = Some((0, 8));
    x.class_hint = Some(Class::C4);
    let mut y = edge("y", Some("B"), &[], None);
    y.abi = Some((24, 8));
    Graph { name: "chain".into(), attrs: vec![("threads".into(), Val::Int(256))], nodes: vec![node("A", Prim::A, "gemm", 100), node("E", Prim::E, "map", 20), node("B", Prim::A, "gemm", 100)], edges: vec![x, e1, edge("g", Some("E"), &["B"], Some(Class::C0)), y], groups: Vec::new() }
}

#[test]
fn json_round_trips() {
    let g = chain();
    let s = g.to_json_string();
    let back = Graph::from_json_str(&s).expect("parses");
    assert_eq!(back, g);
    assert_eq!(back.to_json_string(), s, "the text form is stable");
    assert!(Json::parse("{\"a\": [1, 2, {\"b\": \"x\\ny\"}], \"c\": true, \"d\": null}").is_ok());
    assert!(Json::parse("{\"a\": 1,}").is_err());
}

#[test]
fn the_class_is_computed_from_the_layouts() {
    let g = chain();
    assert_eq!(g.class(g.edge("h").unwrap()).unwrap(), Class::C0);
    let mut e = g.edge("h").unwrap().clone();
    e.consumer = Some(frag("h", swapped()));
    let c = g.class(&e).unwrap();
    assert!(c > Class::C0 && c < Class::C3, "a lane swap inside a quad is a cross-lane class, got {c:?}");
    e.class_hint = Some(Class::C0);
    assert!(g.class(&e).unwrap_err().contains("declared class C0"), "a hint that disagrees with the layouts is an error");
    let mut bare = edge("z", Some("A"), &["E"], None);
    bare.class_hint = None;
    assert!(g.class(&bare).is_err(), "no layouts and no hint");
}

#[test]
fn a_single_kernel_cut_keeps_every_edge_on_chip() {
    let g = chain();
    let cut = Cut { kernels: vec![KernelCut { name: "all".into(), nodes: vec!["B".into(), "A".into(), "E".into()] }] };
    let r = g.evaluate_cut(&cut, Budget::default()).expect("fits");
    assert_eq!(r.kernels.len(), 1);
    assert_eq!(r.kernels[0].stages, vec!["A", "E", "B"], "dependence order, whatever the cut lists");
    assert!(r.cut_edges.is_empty());
    assert_eq!(r.kernels[0].internal.iter().map(|e| (e.tensor.as_str(), e.class)).collect::<Vec<_>>(), vec![("h", Class::C0), ("g", Class::C0)]);
    assert_eq!(r.kernels[0].inputs, vec!["x"]);
    assert_eq!(r.kernels[0].outputs, vec!["y"]);
    assert_eq!(r.kernels[0].resources.vgprs, 100, "the largest stage footprint");
    let text = r.text(&g);
    assert!(text.contains("on-chip h A -> E : C0 (wave)") && text.contains("kernel all"), "{text}");
}

#[test]
fn cutting_an_edge_makes_it_c4_and_splits_the_kernels() {
    let g = chain();
    let cut = Cut { kernels: vec![KernelCut { name: "k1".into(), nodes: vec!["A".into(), "E".into()] }, KernelCut { name: "k2".into(), nodes: vec!["B".into()] }] };
    let r = g.evaluate_cut(&cut, Budget::default()).unwrap();
    assert_eq!(r.cut_edges.len(), 1);
    assert_eq!((r.cut_edges[0].tensor.as_str(), r.cut_edges[0].class), ("g", Class::C4));
    assert_eq!(r.kernels[0].outputs, vec!["g"]);
    assert_eq!(r.kernels[1].inputs, vec!["g"]);
}

#[test]
fn scope_and_budget_and_coverage_are_checked() {
    let mut g = chain();
    let one = Cut { kernels: vec![KernelCut { name: "all".into(), nodes: vec!["A".into(), "E".into(), "B".into()] }] };
    // a tensor that may not stay on chip must be cut
    g.edges[2].scope = Scope::None;
    assert!(g.evaluate_cut(&one, Budget::default()).unwrap_err().contains("may not stay on chip"));
    g.edges[2].scope = Scope::Grid;
    assert!(g.evaluate_cut(&one, Budget::default()).unwrap_err().contains("grid-level"));
    g.edges[2].scope = Scope::Wave;
    // block scope: at least C3
    g.edges[2].scope = Scope::Block;
    let r = g.evaluate_cut(&one, Budget::default()).unwrap();
    assert_eq!(r.kernels[0].internal[1].class, Class::C3);
    g.edges[2].scope = Scope::Wave;
    // budget
    assert!(g.evaluate_cut(&one, Budget { vgprs: 50, ..Budget::default() }).unwrap_err().contains("exceeds the budget"));
    // coverage
    let missing = Cut { kernels: vec![KernelCut { name: "k".into(), nodes: vec!["A".into(), "E".into()] }] };
    assert!(g.evaluate_cut(&missing, Budget::default()).unwrap_err().contains("in no kernel"));
    let twice = Cut { kernels: vec![KernelCut { name: "k".into(), nodes: vec!["A".into(), "E".into(), "B".into()] }, KernelCut { name: "j".into(), nodes: vec!["A".into()] }] };
    assert!(g.evaluate_cut(&twice, Budget::default()).unwrap_err().contains("two kernels"));
}

#[test]
fn structural_errors() {
    let mut g = chain();
    g.edges.push(edge("h", Some("A"), &["E"], Some(Class::C0)));
    assert!(g.validate().unwrap_err().contains("duplicate tensor"));
    let mut g = chain();
    g.edges.push(edge("back", Some("B"), &["A"], Some(Class::C0)));
    assert!(g.validate().unwrap_err().contains("cycle"));
    let mut g = chain();
    g.edges[1].to = vec!["nope".into()];
    assert!(g.validate().unwrap_err().contains("unknown consumer"));
}

// ---------------------------------------------------------------- skeleton generation: weave groups and cut-edge slots

use amdgpu_graph::opgraph::skeleton::generate;

fn two_stage_graph() -> Graph {
    // the FMHA shape: v_phase = softmax (sm), m_phase = pv + qk woven by one weaver; qk produces the S of the NEXT iteration (loop-carried edge s)
    let mut g = chain();
    g.nodes = vec![node("sm", Prim::O, "online.softmax", 40), node("pv", Prim::A, "mma.pv", 100), node("qk", Prim::A, "mma.qk", 100)];
    g.edges = vec![
        { let mut e = edge("x", None, &["qk"], Some(Class::C4)); e.abi = Some((0, 8)); e },
        { let mut e = edge("s", Some("qk"), &["sm"], Some(Class::C0)); e.carried = true; e },
        edge("p", Some("sm"), &["pv"], Some(Class::C0)),
        { let mut e = edge("o", Some("pv"), &[], Some(Class::C4)); e.abi = Some((8, 8)); e },
    ];
    g.groups = vec![Group { name: "m_phase".into(), label: "m_phase".into(), nodes: vec!["pv".into(), "qk".into()] }];
    g
}

#[test]
fn a_weave_group_is_one_slot_with_a_default_composition_and_member_constructors() {
    let mut g = two_stage_graph();
    g.validate().unwrap();
    let text = generate(&g, "attn", &["sm", "pv", "qk"], "attn.graph.json").unwrap();
    assert!(text.contains("fn m_phase(&mut self, b: &mut FnBuilder, cx: &mut Self::Cx) {"), "{text}");
    assert!(text.contains("amdgpu_stage::stage::compose(\"m_phase\", vec![self.pv_stage(&*cx), self.qk_stage(&*cx)])"), "{text}");
    assert!(text.contains("fn qk_stage<'a>(&'a self, cx: &'a Self::Cx) -> amdgpu_stage::stage::StageInst<'a>;"), "{text}");
    assert!(!text.contains("fn qk(&mut self") && !text.contains("fn pv(&mut self"), "grouped nodes have no slots of their own");
    assert!(text.find("slots.sm(b, &mut cx);").unwrap() < text.find("slots.m_phase(b, &mut cx);").unwrap(), "{text}");
    assert_eq!(text, generate(&g, "attn", &["sm", "pv", "qk"], "attn.graph.json").unwrap(), "deterministic");
    assert_eq!(Graph::from_json_str(&g.to_json_string()).unwrap(), g, "groups survive the JSON round trip");
    // the members must be consecutive in the dependence order
    let mut far = g.clone();
    far.nodes.push(node("post", Prim::E, "map.epilogue_chain", 10));
    far.edges.push(edge("q", Some("pv"), &["post"], Some(Class::C0)));
    far.edges.push(edge("r", Some("post"), &["qk"], Some(Class::C0)));
    assert!(generate(&far, "attn", &["sm", "pv", "qk", "post"], "attn.graph.json").unwrap_err().contains("not consecutive"));
    // structural errors
    g.groups[0].nodes = vec!["pv".into()];
    assert!(g.validate().unwrap_err().contains("at least two"));
    g.groups[0].nodes = vec!["pv".into(), "nope".into()];
    assert!(g.validate().unwrap_err().contains("unknown node"));
}

#[test]
fn cut_edges_become_store_and_load_slots() {
    let g = two_stage_graph();
    // kernel a = sm (p leaves, the carried s arrives), kernel b = the m_phase group (p arrives, s leaves)
    let a = generate(&g, "a", &["sm"], "attn.graph.json").unwrap();
    assert!(a.contains("fn store_p(&mut self, _b: &mut FnBuilder, _cx: &mut Self::Cx) {") && a.contains("cut out: p to pv") && a.contains("cut in : s from qk"), "{a}");
    assert!(a.find("slots.load_s(b, &mut cx);").unwrap() < a.find("slots.sm(b, &mut cx);").unwrap(), "the load precedes its first consumer");
    assert!(a.find("slots.sm(b, &mut cx);").unwrap() < a.find("slots.store_p(b, &mut cx);").unwrap(), "the store follows its producer");
    assert!(a.contains("todo!(\"slot load_s") && a.contains("a `move.row_frags` stage"), "the default body names the registry stage: {a}");
    let b = generate(&g, "b", &["pv", "qk"], "attn.graph.json").unwrap();
    assert!(b.contains("fn load_p(") && b.contains("fn store_s(") && b.contains("fn m_phase("), "{b}");
    assert!(b.find("slots.load_p(b, &mut cx);").unwrap() < b.find("slots.m_phase(b, &mut cx);").unwrap());
    assert!(b.find("slots.m_phase(b, &mut cx);").unwrap() < b.find("slots.store_s(b, &mut cx);").unwrap());
    // a group split by the cut is an error
    assert!(generate(&g, "c", &["sm", "pv"], "attn.graph.json").unwrap_err().contains("split by the cut"));
}
