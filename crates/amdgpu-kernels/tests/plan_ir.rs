//! The kernel plan (docs/two-weavers.md) travels with the Function: irv1 prints and parses it, verify checks it, and a
//! function without a plan prints exactly as before.

use amdgpu_dsl::plan::*;
use amdgpu_dsl::{FnBuilder, RegRef};

fn small() -> amdgpu_dsl::Function {
    let mut b = FnBuilder::new("plan_probe", 64);
    b.weave_region_begin("main");
    b.weave_region_end();
    b.finish()
}

fn a_plan() -> KernelPlan {
    KernelPlan {
        phases: vec![
            Phase { id: 0, name: "main".into(), parent: None, kind: PhaseKind::Straight, order: OrderSpec::Authored, budget: Budget::default() },
            Phase {
                id: 1,
                name: "loop".into(),
                parent: Some(0),
                kind: PhaseKind::LoopBody { var: None, trip: Some(8) },
                order: OrderSpec::Partial(vec![(0, 1)]),
                budget: Budget { vgpr: Some(128), agpr: Some(64), sgpr: None, lds_bytes: Some(65536), inflight: vec![("vmcnt".into(), 6)], m0: M0Use::Owned },
            },
        ],
        units: vec![
            UnitDecl { id: 0, name: "dma".into(), phase: 1, class: UnitClass::Dma, anchor: false, iter: 1, latency: 500 },
            UnitDecl { id: 1, name: "mma".into(), phase: 1, class: UnitClass::Mfma, anchor: false, iter: 0, latency: 64 },
        ],
        deps: vec![Dep { from: 0, to: 1, kind: DepKind::Memory("lds".into()), distance: 1 }],
        handoffs: vec![HandOff { value: RegRef::Var(amdgpu_dsl::VarId(0)), from: 0, to: 1, via: Via::Reg { file: RegFileKind::Agpr, pin: Some(0) }, layout: Some("acc16x16".into()) }],
        sync: vec![SyncRule::Counted { producer: 0, consumer: 1, counter: "vmcnt".into(), slack: 2 }, SyncRule::Barrier { after: 0, before: 1 }],
        hints: Hints { interleave: vec![(1, vec![0])], priority: vec![(1, 2)], prefetch_depth: vec![(1, 2)], tokens: vec![(1, "ldsink=1".into())] },
    }
}

#[test]
fn a_function_without_a_plan_prints_no_plan_lines() {
    let t = amdgpu_dsl::ir_text::print(&small());
    assert!(!t.lines().any(|l| l.starts_with("plan_")), "{t}");
}

#[test]
fn the_plan_round_trips_through_irv1() {
    let mut f = small();
    f.plan = Some(a_plan());
    let t = amdgpu_dsl::ir_text::print(&f);
    assert!(t.contains("plan_phase") && t.contains("plan_unit") && t.contains("plan_dep") && t.contains("plan_handoff") && t.contains("plan_sync") && t.contains("plan_hints"), "{t}");
    let g = amdgpu_dsl::ir_text::parse(&t).unwrap_or_else(|e| panic!("{e}\n{t}"));
    assert_eq!(g.plan, f.plan);
    assert_eq!(amdgpu_dsl::ir_text::print(&g), t);
}

#[test]
fn verify_rejects_a_plan_with_unknown_ids() {
    let mut p = a_plan();
    assert!(verify(&p).is_empty(), "{:?}", verify(&p));
    p.deps.push(Dep { from: 0, to: 9, kind: DepKind::Data, distance: 0 });
    p.units.push(UnitDecl { id: 1, name: "dup".into(), phase: 7, class: UnitClass::Valu, anchor: false, iter: 0, latency: 8 });
    let e = verify(&p);
    assert!(e.iter().any(|x| x.contains("unknown unit")), "{e:?}");
    assert!(e.iter().any(|x| x.contains("unit id 1 twice")), "{e:?}");
    assert!(e.iter().any(|x| x.contains("unknown phase 7")), "{e:?}");
    let mut f = small();
    f.plan = Some(p);
    assert!(amdgpu_dsl::verify(&f).iter().any(|x| x.starts_with("plan:")));
}
