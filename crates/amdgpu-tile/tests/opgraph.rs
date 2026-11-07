//! The op graph (amdgpu_tile::opgraph): JSON round trip, computed boundary classes, cuts (C4 edges, scope / budget checks), the review text.
use amdgpu_tile::opgraph::json::Json;
use amdgpu_tile::opgraph::*;

fn node(id: &str, prim: Prim, stage: &str, vgprs: u32) -> Node {
    Node { id: id.into(), prim, stage: stage.into(), params: vec![("k".into(), Val::Int(256)), ("act".into(), Val::Str("relu".into()))], resources: Resources { vgprs, agprs: 0, lds_bytes: 1024, barriers: 0 }, roles: vec!["all".into()], barrier_phase: 0, phase: Phase::Loop }
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
    Graph { name: "chain".into(), attrs: vec![("threads".into(), Val::Int(256))], nodes: vec![node("A", Prim::A, "gemm", 100), node("E", Prim::E, "map", 20), node("B", Prim::A, "gemm", 100)], edges: vec![x, e1, edge("g", Some("E"), &["B"], Some(Class::C0)), y] }
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