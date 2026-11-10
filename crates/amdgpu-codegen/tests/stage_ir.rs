//! Node::Stage: structure (tree / flat round trip, nesting rules), transparency to lowering, and the GV-STAGE edge checks.
use amdgpu_codegen::gpu_verify::{verify_ir, Severity};
use amdgpu_dsl::{FnBuilder, Function, Node, Opnd, StageContract, StageVal, Stmt};

fn sv(name: &str, v: u32, table: Vec<i64>) -> StageVal {
    StageVal { name: name.into(), values: vec![v], table, elem_bits: 32, storage: "vgpr".into(), layout: String::new() }
}

/// two stages: s1 computes x = 3 + 4 and hands it over as `X` (table t_out); s2 consumes `X` (table t_in) and multiplies.
fn build(t_out: Vec<i64>, t_in: Vec<i64>, declare_in: bool, phase2: u32, extra_use: bool) -> Function {
    let mut b = FnBuilder::new("stages", 64);
    let c3 = b.c_i32(3);
    b.stage_begin("s1", StageContract { roles: vec!["all".into()], ..Default::default() });
    let c4 = b.c_i32(4);
    let x = b.add_i32(c3, c4);
    let w = b.add_i32(x, c4); // not handed over
    b.stage_live_out(sv("X", x, t_out));
    b.stage_end();
    let live_in = if declare_in { vec![sv("X", x, t_in)] } else { vec![] };
    b.stage("s2", StageContract { live_in, barrier_phase: phase2, roles: vec!["all".into()], ..Default::default() }, |b| {
        let y = b.add_i32(x, c3);
        if extra_use {
            let _ = b.add_i32(y, w); // w belongs to s1 and is not handed over (pooled constants such as c4 are exempt)
        }
    });
    b.finish()
}

fn stage_errors(f: &Function) -> Vec<String> {
    verify_ir(f).findings.iter().filter(|x| x.code == "GV-STAGE" && x.severity == Severity::Error).map(|x| x.msg.clone()).collect()
}

#[test]
fn stage_tree_and_flat() {
    let f = build(vec![1, 2], vec![1, 2], true, 0, false);
    assert!(matches!(f.body().iter().find(|n| matches!(n, Node::Stage { .. })), Some(Node::Stage { name, .. }) if name == "s1"));
    let flat = f.stmts();
    assert_eq!(flat.iter().filter(|s| matches!(s, Stmt::StageBegin { .. })).count(), 2);
    assert_eq!(flat.iter().filter(|s| matches!(s, Stmt::StageEnd)).count(), 2);
    assert!(Function::parse_flat(flat.to_vec()).is_ok());
    assert!(stage_errors(&f).is_empty(), "{:?}", stage_errors(&f));
}

#[test]
fn stage_edge_errors() {
    // layout mismatch at the edge
    assert!(stage_errors(&build(vec![1, 2], vec![2, 1], true, 0, false)).iter().any(|m| m.contains("layout")));
    // consumer reads a producer value without declaring it
    assert!(stage_errors(&build(vec![1, 2], vec![1, 2], false, 0, false)).iter().any(|m| m.contains("without declaring")));
    assert!(stage_errors(&build(vec![1, 2], vec![1, 2], true, 0, true)).iter().any(|m| m.contains("without declaring")));
    // barrier phase
    assert!(stage_errors(&build(vec![1, 2], vec![1, 2], true, 1, false)).iter().any(|m| m.contains("barrier phase")));
}

#[test]
#[should_panic(expected = "inside an open tile region")]
fn stage_inside_tile_region_panics() {
    let mut b = FnBuilder::new("bad_stage", 64);
    b.tile_produce("T", Opnd::I32(0), Opnd::I32(0));
    b.stage_begin("s", StageContract::default());
}

/// P0 extensions: TV layout / storage at the edge, scalar hand-over (owner), in-flight tolerances, barrier count.
fn build2(layout_in: &str, storage_in: &str, owner_in: &str, lgkm_in: Option<u32>, barriers_decl: Option<u32>) -> Function {
    use amdgpu_dsl::{StageOutstanding, StageResources, StageScalar};
    let mut b = FnBuilder::new("stages2", 64);
    let c3 = b.c_i32(3);
    let mut x = 0;
    let mut m = 0;
    b.stage("p", StageContract { roles: vec!["all".into()], ..Default::default() }, |b| {
        let c4 = b.c_i32(4);
        x = b.add_i32(c3, c4);
        m = b.add_i32(x, c4);
        let _ = b.load_lds_b32(Opnd::Val(c3)); // one LDS read left in flight
        b.stage_live_out(StageVal { name: "X".into(), values: vec![x], table: vec![1], elem_bits: 32, storage: "agpr".into(), layout: "(64,1):(1,0)".into() });
    });
    // scalars_out is set on the closed stage: re-open the contract through a second producer stage
    b.stage("p2", StageContract { roles: vec!["all".into()], barrier_phase: 0, scalars_out: vec![StageScalar { name: "M".into(), values: vec![m], owner: "row".into() }], live_in: vec![], ..Default::default() }, |_b| {});
    let c = StageContract {
        live_in: vec![StageVal { name: "X".into(), values: vec![x], table: vec![1], elem_bits: 32, storage: storage_in.into(), layout: layout_in.into() }],
        scalars_in: vec![StageScalar { name: "M".into(), values: vec![m], owner: owner_in.into() }],
        outstanding: StageOutstanding { lgkm_in, ..Default::default() },
        resources: StageResources { barriers: barriers_decl, ..Default::default() },
        roles: vec!["all".into()],
        ..Default::default()
    };
    b.stage("c", c, |b| {
        let y = b.add_i32(x, m);
        let _ = b.add_i32(y, c3);
        b.barrier();
    });
    b.finish()
}

#[test]
fn stage_contract_p0_extensions() {
    let ok = build2("(64,1):(1,0)", "vgpr", "row", Some(1), Some(1));
    assert!(stage_errors(&ok).is_empty(), "{:?}", stage_errors(&ok));
    assert!(stage_errors(&build2("(64,1):(2,0)", "vgpr", "row", Some(1), Some(1))).iter().any(|m| m.contains("TV layout")));
    assert!(stage_errors(&build2("(64,1):(1,0)", "lds:T", "row", Some(1), Some(1))).iter().any(|m| m.contains("storage")));
    assert!(stage_errors(&build2("(64,1):(1,0)", "vgpr", "wave", Some(1), Some(1))).iter().any(|m| m.contains("scalar `M`")));
    assert!(stage_errors(&build2("(64,1):(1,0)", "vgpr", "row", Some(0), Some(1))).iter().any(|m| m.contains("in flight at the entry")));
    assert!(stage_errors(&build2("(64,1):(1,0)", "vgpr", "row", Some(1), Some(0))).iter().any(|m| m.contains("barriers inside")));
}