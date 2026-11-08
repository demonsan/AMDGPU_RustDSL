//! The weaver: explicit orders are checked (permutation + topological order) before anything is emitted; `spread` places fillers between skeleton units.

use amdgpu_dsl::{FnBuilder, Op, Opnd};
use amdgpu_tile::weave::{check_order, Units, Weaver};

/// a -> b (b uses a's value) and an independent c
fn three(b: &mut FnBuilder) -> Units {
    let mut u = Units::new();
    let (_, x) = u.capture(b, "a", |b| b.c_i32(7));
    u.capture(b, "b", |b| b.let_(Op::AddI32(Opnd::Val(x), Opnd::I32(1))));
    u.capture(b, "c", |b| b.c_f32(1.0));
    u
}

#[test]
fn seq_and_explicit_orders_emit() {
    let mut fb = FnBuilder::new("weave_t", 64);
    let u = three(&mut fb);
    Weaver::new(u).phase("t").order(vec![2, 0, 1]).emit(&mut fb);
}

#[test]
#[should_panic(expected = "placed before its dependency")]
fn non_topological_order_is_rejected() {
    let mut fb = FnBuilder::new("weave_t", 64);
    let u = three(&mut fb);
    Weaver::new(u).phase("t").order(vec![1, 0, 2]).emit(&mut fb);
}

#[test]
#[should_panic(expected = "listed twice")]
fn duplicate_is_rejected() {
    let mut fb = FnBuilder::new("weave_t", 64);
    let u = three(&mut fb);
    check_order(&u, &[0, 0, 2], "t");
}

#[test]
fn spread_places_fillers_after_skeleton_units() {
    let mut fb = FnBuilder::new("weave_t", 64);
    let mut u = Units::new();
    let ids: Vec<usize> = (0..5).map(|i| u.capture(&mut fb, format!("u{i}"), |b| b.c_i32(i)).0).collect();
    let (s0, s1, f0, f1, f2) = (ids[0], ids[1], ids[2], ids[3], ids[4]);
    let w = Weaver::new(u).spread(&[s0, s1], &[f0, f1, f2], 1);
    assert_eq!(w.plan(), vec![s0, f0, s1, f1, f2]);
}