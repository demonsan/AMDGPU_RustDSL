//! `weave_auto` (amdgpu_tile::weave_auto): the cost-based weave order over captured units is a topological order, deterministic, better than the declaration order in its own model when reads can be
//! hoisted, and keeps barriers and everything ordered by the statement facts where they are.
use amdgpu_dsl::{DType, FnBuilder, Op, ValueId};
use amdgpu_tile::weave::{check_order, Units, Weaver};
use amdgpu_tile::weave_auto::{auto_order, simulate, Model};

fn base() -> (FnBuilder, ValueId, ValueId) {
    let mut b = FnBuilder::new("auto", 64);
    b.arg("out", DType::Ptr, 0, 8);
    let out = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);
    let _ = out;
    let c4 = b.c_i32(4);
    let addr = b.let_(Op::MulI32(amdgpu_dsl::Opnd::Val(tid), amdgpu_dsl::Opnd::Val(c4)));
    (b, tid, addr)
}

/// 4 blocks: a read unit (one `ds_read`) then a compute unit (a dependent VALU chain) consuming it - declared read, compute, read, compute, ...
fn blocks(b: &mut FnBuilder, addr: ValueId) -> Units {
    let mut u = Units::new();
    for i in 0..4 {
        let (_, r) = u.capture(b, format!("read{i}"), |b| b.load_lds_b32(addr));
        u.capture(b, format!("use{i}"), |b| {
            let mut x = b.add_i32(r, 1);
            for _ in 0..12 {
                x = b.add_i32(x, 3);
            }
            x
        });
    }
    u
}

#[test]
fn the_auto_order_is_topological_deterministic_and_beats_the_declaration_order_in_its_model() {
    let (mut b, _tid, addr) = base();
    let u = blocks(&mut b, addr);
    let m = Model::gfx950();
    let a = auto_order(&u, &m);
    check_order(&u, &a, "auto");
    assert_eq!(a, auto_order(&u, &m), "deterministic");
    let decl: Vec<usize> = (0..u.len()).collect();
    let (c_auto, _) = simulate(&u, &a, &m);
    let (c_decl, _) = simulate(&u, &decl, &m);
    assert!(c_auto < c_decl, "reads hoisted ahead of their consumers hide the LDS latency: auto {c_auto} vs declaration {c_decl} ({a:?})");
}

#[test]
fn a_barrier_is_never_crossed_by_memory_operations() {
    let (mut b, _tid, addr) = base();
    let mut u = Units::new();
    let (_, r0) = u.capture(&mut b, "read0", |b| b.load_lds_b32(addr));
    u.capture(&mut b, "use0", |b| b.add_i32(r0, 1));
    let (bar, _) = u.anchor(&mut b, "barrier", |b| b.barrier());
    let (_, r1) = u.capture(&mut b, "read1", |b| b.load_lds_b32(addr));
    u.capture(&mut b, "use1", |b| b.add_i32(r1, 1));
    let a = auto_order(&u, &Model::gfx950());
    check_order(&u, &a, "auto");
    let pos = |i: usize| a.iter().position(|&x| x == i).unwrap();
    assert!(pos(0) < pos(bar), "a read before the barrier stays before it: {a:?}");
    assert!(pos(3) > pos(bar), "a read after the barrier stays after it: {a:?}");
}

#[test]
fn an_explicit_order_is_authoritative_and_tiny_weaves_keep_their_declaration_order() {
    let (mut b, _tid, addr) = base();
    let mut u = Units::new();
    u.capture(&mut b, "a", |b| b.load_lds_b32(addr));
    u.capture(&mut b, "b", |b| b.c_i32(7));
    assert_eq!(auto_order(&u, &Model::gfx950()), vec![0, 1]);
    // `Weaver::order` is explicit: the plan is exactly what was given
    let w = Weaver::new(u).phase("p").order(vec![0, 1]);
    assert_eq!(w.plan(), vec![0, 1]);
}