//! W3 B6a (docs/w3-b5-b6.md section 4): the VGPR budget of a back-end-ordered (Free) plan phase becomes the list scheduler's pressure
//! limit inside the phase's region; authored phases and regions with their own `limit=` keep theirs.

use amdgpu_dsl::plan::*;
use amdgpu_dsl::{DType, FnBuilder, Function};

fn kernel(order: OrderSpec, tokens: &str) -> Function {
    let mut b = FnBuilder::new("probe_budget", 64);
    b.arg("o", DType::Ptr, 0x00, 8);
    b.declare_region("w", false, tokens);
    let mut f = b.finish();
    let mut p = f.plan.clone().unwrap_or_default();
    p.phases.retain(|ph| ph.name != "w");
    let id = p.phases.iter().map(|ph| ph.id + 1).max().unwrap_or(0);
    p.phases.push(Phase { id, name: "w".into(), parent: None, kind: PhaseKind::Straight, order, budget: Budget { vgpr: Some(96), ..Default::default() } });
    f.plan = Some(p);
    f
}

fn tokens(f: &Function) -> String {
    let o = amdgpu_codegen::scheduler_options(f).expect("a declared region runs the scheduler");
    o.regions.iter().find(|r| r.name == "w").expect("region w").tokens.clone()
}

#[test]
fn a_free_phase_budget_is_the_regions_pressure_limit() {
    assert_eq!(tokens(&kernel(OrderSpec::Free, "reorder")), "reorder,limit=96");
    assert_eq!(tokens(&kernel(OrderSpec::Free, "")), "limit=96");
}

#[test]
fn authored_phases_and_explicit_limits_are_kept() {
    assert_eq!(tokens(&kernel(OrderSpec::Authored, "reorder")), "reorder");
    assert_eq!(tokens(&kernel(OrderSpec::Free, "reorder,limit=200")), "reorder,limit=200");
}
