//! The weaver: explicit orders are checked (permutation + topological order) before anything is emitted; `spread` places fillers between skeleton units.

use amdgpu_dsl::{FnBuilder, Op, Opnd};
use amdgpu_tile::weave::{check_order, Hint, Out, UnitClass, Units, Weaver};

fn three(b: &mut FnBuilder) -> (Units<'static>, Out<amdgpu_dsl::ValueId>) {
    let _ = b;
    let mut u = Units::new();
    let x: Out<amdgpu_dsl::ValueId> = Out::new();
    let (x1, x2) = (x.clone(), x.clone());
    let a = u.add("a", UnitClass::Valu, Hint::Anchor, &[], move |b| x1.set(b.c_i32(7)));
    let c = u.add("b", UnitClass::Valu, Hint::Filler, &[a], move |b| {
        let v = x2.get();
        let _ = b.let_(Op::AddI32(Opnd::Val(v), Opnd::I32(1)));
    });
    u.add("c", UnitClass::Mfma, Hint::Skeleton, &[c], |_b| {});
    (u, x)
}

#[test]
fn seq_and_explicit_orders_emit() {
    let mut fb = FnBuilder::new("weave_t", 64);
    let (u, x) = three(&mut fb);
    Weaver::new(u).phase("t").order(vec![0, 1, 2]).emit(&mut fb);
    let _ = x.get();
}

#[test]
#[should_panic(expected = "placed before its dependency")]
fn non_topological_order_is_rejected() {
    let mut fb = FnBuilder::new("weave_t", 64);
    let (u, _) = three(&mut fb);
    Weaver::new(u).phase("t").order(vec![1, 0, 2]).emit(&mut fb);
}

#[test]
#[should_panic(expected = "listed twice")]
fn duplicate_is_rejected() {
    let mut fb = FnBuilder::new("weave_t", 64);
    let (u, _) = three(&mut fb);
    check_order(&u, &[0, 0, 2], "t");
}

#[test]
fn spread_places_fillers_after_skeleton_units() {
    let mut u = Units::new();
    let s0 = u.add("s0", UnitClass::Mfma, Hint::Skeleton, &[], |_b| {});
    let s1 = u.add("s1", UnitClass::Mfma, Hint::Skeleton, &[], |_b| {});
    let f0 = u.add("f0", UnitClass::Dma, Hint::Filler, &[], |_b| {});
    let f1 = u.add("f1", UnitClass::Dma, Hint::Filler, &[], |_b| {});
    let f2 = u.add("f2", UnitClass::Dma, Hint::Filler, &[], |_b| {});
    let w = Weaver::new(u).spread(&[s0, s1], &[f0, f1, f2], 1);
    assert_eq!(w.plan(), vec![s0, f0, s1, f1, f2]);
}
