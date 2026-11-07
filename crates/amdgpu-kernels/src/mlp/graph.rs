//! The op graph of the MLP family (`Y = fp8(relu(alpha X W1)) W2`, design section 4.3): the input of the skeleton generator. Kernels = cuts: the fused kernel is the whole graph, the unfused pair
//! (`mlp_fp8_ws_g1` / `_g2`) is the cut at the `h` edge (C4: H through HBM).

use super::{p_boundary, ALPHA, K1, N1, N2, W_IMG, W_PITCH};
use amdgpu_tile::opgraph::{Class, Edge, FragLayout, Graph, Node, Phase, Prim, Resources, Scope, Store, Val};

fn node(id: &str, prim: Prim, stage: &str, phase: Phase, params: Vec<(&str, Val)>, res: Resources) -> Node {
    Node { id: id.into(), prim, stage: stage.into(), params: params.into_iter().map(|(k, v)| (k.to_string(), v)).collect(), resources: res, roles: vec!["all 4 waves: same program".into()], barrier_phase: 0, phase }
}

fn edge(name: &str, from: Option<&str>, to: &[&str], dtype: &str, shape: &[i64], layout: &str, scope: Scope, store: Store, hint: Option<Class>) -> Edge {
    Edge { name: name.into(), from: from.map(|s| s.into()), to: to.iter().map(|s| s.to_string()).collect(), dtype: dtype.into(), shape: shape.to_vec(), layout: layout.into(), producer: None, consumer: None, scope, store, class_hint: hint, abi: None, carried: false }
}

/// The `h` hand-off (the four 64-wide n1 blocks of `p_boundary` side by side): lane t, value `32 jb + v`.
fn h_layouts() -> (FragLayout, FragLayout) {
    let cat = |producer: bool| {
        let frags: Vec<_> = (0..N1 / 64).map(|jb| { let (p, c) = p_boundary(jb); if producer { p.live[0].clone() } else { c.live[0].clone() } }).collect();
        let table: Vec<i64> = (0..64 * 128).map(|i| { let (t, v) = (i % 64, i / 64); let (jb, vv) = ((v / 32) as usize, v % 32); frags[jb].table[(t + 64 * vv) as usize] }).collect();
        FragLayout { name: "h".into(), threads: 64, values: 128, elem_bits: 8, table }
    };
    (cat(true), cat(false))
}

pub fn graph() -> Graph {
    let mm = |atom_extra: Vec<(&'static str, Val)>| -> Vec<(&'static str, Val)> {
        let mut p = vec![("atom", Val::Str("mfma_scale_32x32x64_f8".into())), ("n", Val::Int(N1)), ("k", Val::Int(K1))];
        p.extend(atom_extra);
        p
    };
    let image = |id: &str| node(id, Prim::M, "move.lds_image", Phase::Prologue, vec![("bytes", Val::Int(W_IMG)), ("pitch", Val::Int(W_PITCH))], Resources { vgprs: 8, agprs: 0, lds_bytes: W_IMG as u32, barriers: 1 });
    let rows = |id: &str, phase: Phase| node(id, Prim::M, "move.row_frags", phase, vec![("k", Val::Int(K1))], Resources { vgprs: 32, agprs: 0, lds_bytes: 0, barriers: 0 });
    let nodes = vec![
        image("w1_image"),
        image("w2_image"),
        rows("x_first", Phase::Prologue),
        node("gemm1_act", Prim::A, "mma.rows_x_lds", Phase::Loop, [mm(vec![("act", Val::Str("map.relu_pack_fp8".into())), ("alpha", Val::Str(format!("{ALPHA}")))]), vec![]].concat(), Resources { vgprs: 192, agprs: 0, lds_bytes: W_IMG as u32, barriers: 0 }),
        rows("x_next", Phase::Loop),
        node("gemm2", Prim::A, "mma.rows_x_lds", Phase::Loop, [mm(vec![("epilogue", Val::Str("map.store_bf16".into()))]), vec![]].concat(), Resources { vgprs: 160, agprs: 0, lds_bytes: W_IMG as u32, barriers: 0 }),
    ];
    let mut e_in = edge("in", None, &["x_first", "x_next"], "fp8", &[-1, K1], "row-major", Scope::None, Store::Global, Some(Class::C4));
    e_in.abi = Some((0, 8));
    let mut e_w1 = edge("w1", None, &["w1_image"], "fp8", &[N1, K1], "padded rows (pitch 272)", Scope::None, Store::Global, Some(Class::C4));
    e_w1.abi = Some((8, 8));
    let mut e_w2 = edge("w2", None, &["w2_image"], "fp8", &[N2, N1], "padded rows, key-permuted (sigma)", Scope::None, Store::Global, Some(Class::C4));
    e_w2.abi = Some((16, 8));
    let e_w1l = edge("w1_lds", Some("w1_image"), &["gemm1_act"], "fp8", &[N1, K1], "padded rows (pitch 272)", Scope::Block, Store::Lds("W1".into()), Some(Class::C3));
    let e_w2l = edge("w2_lds", Some("w2_image"), &["gemm2"], "fp8", &[N2, N1], "padded rows, key-permuted (sigma)", Scope::Block, Store::Lds("W2".into()), Some(Class::C3));
    let e_x0 = edge("x0", Some("x_first"), &["gemm1_act"], "fp8", &[32, K1], "B operand fragments", Scope::Wave, Store::Vgpr, Some(Class::C0));
    let mut e_xn = edge("xn", Some("x_next"), &["gemm1_act"], "fp8", &[32, K1], "B operand fragments", Scope::Wave, Store::Vgpr, Some(Class::C0));
    e_xn.carried = true;
    let mut e_h = edge("h", Some("gemm1_act"), &["gemm2"], "fp8", &[32, N1], "P words in accumulator order (sigma absorbed by the pre-permuted W2)", Scope::Wave, Store::Vgpr, None);
    let (p, c) = h_layouts();
    e_h.producer = Some(p);
    e_h.consumer = Some(c);
    let mut e_out = edge("out", Some("gemm2"), &[], "bf16", &[-1, N2], "row-major", Scope::None, Store::Global, Some(Class::C4));
    e_out.abi = Some((24, 8));
    Graph { name: "mlp_fp8".into(), attrs: vec![("threads".into(), Val::Int(256))], nodes, edges: vec![e_in, e_w1, e_w2, e_w1l, e_w2l, e_x0, e_xn, e_h, e_out], groups: Vec::new() }
}