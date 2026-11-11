//! The kernel plan the front end derives (docs/two-weavers.md 6.1): phases / units / dependences / interleave hints from the Weaver, order / tokens / budgets / loop kinds / CTA shape
//! from the declarations at `finish`.

use amdgpu_dsl::plan::{verify, DepKind, OrderSpec, PhaseKind};
use amdgpu_dsl::{DType, FnBuilder, Op, Opnd};
use amdgpu_stage::weave::{Units, Weaver};

/// a -> b (b uses a's value) and an independent c
fn three(b: &mut FnBuilder) -> Units {
    let mut u = Units::new();
    let (_, x) = u.capture(b, "a", |b| b.c_i32(7));
    u.capture(b, "b", |b| b.let_(Op::AddI32(Opnd::Val(x), Opnd::I32(1))));
    u.capture(b, "c", |b| b.c_f32(1.0));
    u
}

#[test]
fn the_weaver_and_finish_derive_a_consistent_plan() {
    let mut fb = FnBuilder::new("plan_t", 256);
    fb.declare_region("hot", false, "ldsink=1");
    fb.reg_cap("hot", 96);
    let u = three(&mut fb);
    Weaver::new(u).phase("hot").seq().emit(&mut fb);
    // a woven loop body with a spread (skeleton a, fillers c b)
    let _v = fb.begin_loop(&[(DType::I32, Opnd::I32(0))]);
    let mut u2 = Units::new();
    let ids: Vec<usize> = (0..3).map(|i| u2.capture(&mut fb, format!("l{i}"), |b| b.c_i32(i)).0).collect();
    Weaver::new(u2).phase("body").spread(&ids[..1], &ids[1..], 1).emit(&mut fb);
    fb.end_loop();
    let f = fb.finish();
    let p = f.plan.as_ref().expect("a woven kernel has a plan");
    assert!(verify(p).is_empty(), "{:?}", verify(p));

    // CTA shape from the workgroup size
    let cta = p.cta.as_ref().unwrap();
    assert_eq!((cta.threads, cta.waves), (256, 4));

    // phases: `hot` is declared free-order with a token and a register cap; `body` sits in a loop
    let hot = p.phases.iter().find(|x| x.name == "hot").unwrap();
    assert_eq!(hot.order, OrderSpec::Free);
    assert_eq!(hot.budget.vgpr, Some(96));
    assert!(p.hints.tokens.iter().any(|(id, t)| *id == hot.id && t == "ldsink=1"));
    let body = p.phases.iter().find(|x| x.name == "body").unwrap();
    assert!(matches!(body.kind, PhaseKind::LoopBody { .. }), "{:?}", body.kind);
    assert_eq!(body.order, OrderSpec::Authored);

    // units in emission order with the derived dependence a -> b (a value flows)
    let hot_units: Vec<_> = p.units.iter().filter(|u| u.phase == hot.id).collect();
    assert_eq!(hot_units.iter().map(|u| u.name.as_str()).collect::<Vec<_>>(), ["a", "b", "c"]);
    assert!(p.deps.iter().any(|d| d.from == hot_units[0].id && d.to == hot_units[1].id && d.kind == DepKind::Data));
    assert!(!p.deps.iter().any(|d| d.to == hot_units[2].id), "c is independent");

    // the spread's interleave hint: skeleton l0 with its fillers l1, l2 (one per skeleton unit, the rest after the last)
    let body_units: Vec<_> = p.units.iter().filter(|u| u.phase == body.id).collect();
    assert!(p.hints.interleave.iter().any(|(s, fs)| *s == body_units[0].id && fs.len() == 2), "{:?}", p.hints.interleave);
}

#[test]
fn a_kernel_that_never_weaves_has_no_plan() {
    let mut fb = FnBuilder::new("plain", 64);
    fb.c_i32(1);
    assert!(fb.finish().plan.is_none());
}