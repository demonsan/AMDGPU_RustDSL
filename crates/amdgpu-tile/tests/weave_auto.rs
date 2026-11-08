//! `weave_auto` (amdgpu_tile::weave_auto): the cost-based weave order is a topological order, deterministic, never worse than the declaration order in its own model, and respects fences, anchors
//! and the register guard.
use amdgpu_tile::weave::{check_order, Hint, RegProfile, UnitClass, Units};
use amdgpu_tile::weave_auto::{auto_order, simulate, Model, UnitCost};

/// setup; for each of 4 blocks: a read unit (4 ds_read_b128) then an MMA unit (2 MFMAs) that consumes it - declared read, mma, read, mma, ...
fn blocks<'a>(regs: u32) -> Units<'a> {
    let mut u = Units::new();
    let s = u.add("setup", UnitClass::Setup, Hint::Anchor, &[], |_b| {});
    u.cost(s, UnitCost { salu: 2, ..Default::default() });
    for i in 0..4 {
        let r = u.add(format!("read{i}"), UnitClass::LdsRead, Hint::LatencySensitive, &[s], |_b| {});
        u.cost(r, UnitCost { ds_b128: 4, ..Default::default() });
        u.facts(r, 144, RegProfile { live_in: 0, live_out: regs, temps: 0, agpr_ok: false });
        let m = u.add(format!("mma{i}"), UnitClass::Mfma, Hint::Skeleton, &[r], |_b| {});
        u.cost(m, UnitCost { mfma: 2, ..Default::default() });
        u.facts(m, 84, RegProfile { live_in: regs, live_out: 0, temps: 0, agpr_ok: true });
    }
    u
}

#[test]
fn the_auto_order_is_topological_deterministic_and_beats_the_declaration_order_in_its_model() {
    let u = blocks(16);
    let m = Model::gfx950();
    let a = auto_order(&u, &m);
    check_order(&u, &a, "auto");
    assert_eq!(a, auto_order(&u, &m), "deterministic");
    let decl: Vec<usize> = (0..u.len()).collect();
    let (c_auto, _) = simulate(&u, &a, &m);
    let (c_decl, _) = simulate(&u, &decl, &m);
    assert!(c_auto < c_decl, "reads hoisted ahead of the MFMAs hide their latency: auto {c_auto} vs declaration {c_decl} ({a:?})");
}

#[test]
fn the_register_guard_limits_how_far_reads_run_ahead() {
    // each read holds 60 dwords until its MMA: hoisting all four reads would hold 240 dwords at once
    let u = blocks(60);
    let mut m = Model::gfx950();
    m.reg_slack = 0;
    let a = auto_order(&u, &m);
    check_order(&u, &a, "auto");
    let decl: Vec<usize> = (0..u.len()).collect();
    let (_, p_auto) = simulate(&u, &a, &m);
    let (_, p_decl) = simulate(&u, &decl, &m);
    assert!(p_auto <= p_decl, "peak {p_auto} must not exceed the declaration order's {p_decl}");
    // with a generous slack the same units are free to run further ahead (and be faster)
    m.reg_slack = 1000;
    let b = auto_order(&u, &m);
    let (c_free, _) = simulate(&u, &b, &m);
    let (c_tight, _) = simulate(&u, &a, &{ let mut t = Model::gfx950(); t.reg_slack = 0; t });
    assert!(c_free <= c_tight);
}

#[test]
fn a_fence_unit_is_never_crossed_and_anchors_keep_their_places() {
    let mut u = Units::new();
    let s = u.add("setup", UnitClass::Setup, Hint::Anchor, &[], |_b| {});
    let r0 = u.add("read0", UnitClass::LdsRead, Hint::LatencySensitive, &[s], |_b| {});
    u.cost(r0, UnitCost { ds_b128: 4, ..Default::default() });
    let m0 = u.add("mma0", UnitClass::Mfma, Hint::Skeleton, &[r0], |_b| {});
    u.cost(m0, UnitCost { mfma: 2, ..Default::default() });
    let br = u.add("branch", UnitClass::Branch, Hint::Anchor, &[], |_b| {});
    u.cost(br, UnitCost { valu: 8, fence: true, ..Default::default() });
    let r1 = u.add("read1", UnitClass::LdsRead, Hint::LatencySensitive, &[s], |_b| {});
    u.cost(r1, UnitCost { ds_b128: 4, ..Default::default() });
    let m1 = u.add("mma1", UnitClass::Mfma, Hint::Skeleton, &[r1], |_b| {});
    u.cost(m1, UnitCost { mfma: 2, ..Default::default() });
    let a = auto_order(&u, &Model::gfx950());
    check_order(&u, &a, "auto");
    let pos = |i: usize| a.iter().position(|&x| x == i).unwrap();
    assert!(pos(r0) < pos(br) && pos(m0) < pos(br), "units before the fence stay before it: {a:?}");
    assert!(pos(r1) > pos(br) && pos(m1) > pos(br), "units after the fence stay after it: {a:?}");
    assert!(pos(s) < pos(br), "anchors keep their relative order: {a:?}");
}

#[test]
fn tiny_weaves_keep_their_declaration_order() {
    let mut u = Units::new();
    u.add("a", UnitClass::Valu, Hint::Filler, &[], |_b| {});
    u.add("b", UnitClass::Valu, Hint::Filler, &[0], |_b| {});
    assert_eq!(auto_order(&u, &Model::gfx950()), vec![0, 1]);
}