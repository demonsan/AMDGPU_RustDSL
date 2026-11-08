//! weave2: captured units (derived dependencies / facts), statement-granularity split, loop rotation.

use amdgpu_dsl::{FnBuilder, Function};
use amdgpu_tile::weave::{rotate, Hint, Part, RotStage, UnitClass, Units, Weaver};

fn direct() -> Function {
    let mut b = FnBuilder::new("w2", 64);
    let x = b.c_f32(1.0);
    let y = b.c_f32(2.0);
    let s = b.add_f32(x, y);
    let p = b.mul_f32(s, s);
    let _ = b.add_f32(p, x);
    b.finish()
}

#[test]
fn capture_in_order_equals_direct_build_and_derives_deps() {
    let mut b = FnBuilder::new("w2", 64);
    let mut u = Units::new();
    let (ia, (x, y)) = u.capture(&mut b, "a", |b| (b.c_f32(1.0), b.c_f32(2.0)));
    let (ib, s) = u.capture(&mut b, "b", |b| b.add_f32(x, y));
    let (ic, p) = u.capture(&mut b, "c", |b| b.mul_f32(s, s));
    let (id, _) = u.capture(&mut b, "d", |b| b.add_f32(p, x));
    assert!(u.unit(ib).deps.contains(&ia));
    assert_eq!(u.unit(ic).deps, vec![ib]);
    assert!(u.unit(id).deps.contains(&ic) && u.unit(id).deps.contains(&ia));
    assert_eq!(u.unit(ib).class, UnitClass::Valu);
    Weaver::new(u).seq().emit(&mut b);
    assert!(amdgpu_dsl::print::dsl_structural_eq(&b.finish(), &direct()));
}

#[test]
#[should_panic(expected = "placed before its dependency")]
fn derived_deps_are_enforced() {
    let mut b = FnBuilder::new("w2", 64);
    let mut u = Units::new();
    let (_, x) = u.capture(&mut b, "a", |b| b.c_f32(1.0));
    let _ = u.capture(&mut b, "b", |b| b.add_f32(x, x));
    Weaver::new(u).order(vec![1, 0]).emit(&mut b);
}

#[test]
fn independent_captures_can_be_reordered_and_barriers_anchor() {
    let mut b = FnBuilder::new("w2", 64);
    let mut u = Units::new();
    let _ = u.capture(&mut b, "a", |b| b.c_f32(1.0));
    let _ = u.capture(&mut b, "b", |b| b.c_f32(2.0));
    let (ib, ()) = u.anchor(&mut b, "bar", |b| b.barrier());
    assert_eq!(u.unit(ib).hint, Hint::Anchor);
    assert!(u.unit(1).deps.is_empty(), "two constants are independent");
    Weaver::new(u).order(vec![1, 0, 2]).emit(&mut b);
}

#[test]
fn fine_splits_into_statements_with_derived_deps() {
    let mut b = FnBuilder::new("w2", 64);
    let mut u = Units::new();
    let _ = u.capture(&mut b, "all", |b| {
        let x = b.c_f32(1.0);
        let y = b.c_f32(2.0);
        let s = b.add_f32(x, y);
        b.mul_f32(s, s)
    });
    let w = Weaver::new(u).fine();
    assert_eq!(w.units().len(), 4);
    assert!(w.units().unit(1).deps.is_empty());
    assert!(w.units().unit(2).deps.contains(&0) && w.units().unit(2).deps.contains(&1));
    // the two constants swapped: still a legal order
    w.order(vec![1, 0, 2, 3]).emit(&mut b);
}

#[test]
fn rotate_keeps_parts_by_iteration_offset() {
    let parts = [Part { name: "next", iter: 1, anchor: false }, Part { name: "cur", iter: 0, anchor: false }];
    for (stage, want) in [(RotStage::Prologue, vec![0]), (RotStage::Steady, vec![0, 1]), (RotStage::Epilogue, vec![1])] {
        let mut b = FnBuilder::new("rot", 64);
        let mut built = Vec::new();
        rotate(&mut b, &parts, stage, None, &mut |b, i| {
            built.push(i);
            let _ = b.c_f32(i as f32);
        });
        assert_eq!(built, want, "{stage:?}");
    }
}