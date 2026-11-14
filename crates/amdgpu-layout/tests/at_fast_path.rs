//! `Layout::at` has an integer fast path for static layouts (`at_flat`); it must agree with the symbolic `call` on every index, hierarchical and flat layouts alike.

use amdgpu_layout::layout::{make_layout, Layout};
use amdgpu_layout::{Dyns, Ix};

fn check(l: &Layout) {
    let n = match l.size().c() {
        Some(n) => n,
        None => return,
    };
    for i in 0..(n + 3).min(5000) {
        let slow = l.call(&Ix::C(i), &Dyns::new()).st();
        assert_eq!(l.at(i), slow, "layout {l} at {i}");
    }
}

#[test]
fn the_static_fast_path_agrees_with_the_symbolic_call() {
    check(&make_layout(7, 3));
    check(&make_layout(1, 0));
    check(&Layout::tuple(vec![make_layout(4, 1), make_layout(8, 4)]));
    check(&Layout::tuple(vec![Layout::tuple(vec![make_layout(2, 1), make_layout(3, 2)]), make_layout(5, 7), Layout::tuple(vec![make_layout(2, 64), make_layout(4, 128)])]));
    check(&Layout::tuple(vec![make_layout(32, 64), Layout::tuple(vec![make_layout(2, 4), make_layout(4, 1), make_layout(2, 32)])]));
}
