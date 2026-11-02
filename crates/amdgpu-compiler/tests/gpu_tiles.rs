//! gpu_verify v1.1: LDS tile lifetime / versioning. A two-role (producer wave / consumer wave) ring protocol, one positive case and one negative case per rule.
use amdgpu_compiler::gpu_verify::{verify_ir, Report, Severity};
use amdgpu_dsl::{DType, FnBuilder, Function, Op, Opnd};

#[derive(Clone, Copy)]
struct Opt {
    /// a compile-time trip-count bound (the loop exits once it+1 >= k) instead of the kernarg bound
    trip: Option<i32>,
    /// consumer slot = (min(it, cap)) & mask: equals it & mask up to `cap`, then sticks (a protocol that is only right for the first iterations)
    cap: Option<i32>,
    /// producer slot = (nxt & mask) + ((nxt >> 8) & 1): ordinary double buffering until iteration 255 (kernel-occ's affine_period_256 attack)
    deep: bool,
    /// ring index mask (default slots - 1)
    mask: Option<i32>,
    /// version transform applied to the produced AND the expected version: 0 none, 1 ^256, 2 <<8, 3 *256, 4 <<24
    vt: u8,
    slots: u32,
    prologue_barrier: bool,
    producer_wait: bool,
    consume_ver_off: i32,
    write_extra: i32,
    consumer_wait: bool,
}

fn good() -> Opt {
    Opt { trip: None, cap: None, deep: false, mask: None, vt: 0, slots: 2, prologue_barrier: true, producer_wait: true, consume_ver_off: 0, write_extra: 0, consumer_wait: true }
}

fn has(rep: &Report, code: &str) -> bool {
    rep.findings.iter().any(|f| f.code == code && f.severity == Severity::Error)
}

fn errors(rep: &Report) -> String {
    rep.errors().iter().map(|e| format!("[{}] {}", e.code, e.msg)).collect::<Vec<_>>().join("\n")
}

fn vtrans(b: &mut FnBuilder, vt: u8, v: amdgpu_dsl::ValueId) -> amdgpu_dsl::ValueId {
    match vt {
        1 => b.let_(Op::XorI32(Opnd::Val(v), Opnd::I32(256))),
        2 => b.let_(Op::ShlI32(Opnd::Val(v), Opnd::I32(8))),
        3 => b.mul_i32(v, Opnd::I32(256)),
        4 => b.let_(Op::ShlI32(Opnd::Val(v), Opnd::I32(24))),
        _ => v,
    }
}

/// role P (waves with tid < 64) produces slot (it+1)&mask version it+1 while role C reads slot it&mask version it, one barrier per iteration.
fn proto(o: Opt) -> Function {
    let mut b = FnBuilder::new("proto", 128);
    b.lds(8192);
    b.arg("n", DType::I32, 0, 4);
    b.lds_ring("buf", 0, 1024, o.slots);
    let n = b.let_(Op::KernargI32 { offset: 0 });
    let t = b.let_(Op::WorkitemId);
    let c64 = b.c_i32(64);
    let is_p = b.cmp_lt_i32(t, c64);
    let c63 = b.c_i32(63);
    let lane = b.let_(Op::AndI32(Opnd::Val(t), Opnd::Val(c63)));
    let c16 = b.c_i32(16);
    let lane16 = b.mul_i32(lane, c16);
    let z = b.c_i32(0);
    let v = b.frag4(z, z, z, z);
    // prologue: P writes slot 0 version 0
    b.begin_if(is_p);
    b.tile_produce("buf", Opnd::I32(0), Opnd::I32(0));
    let a0 = b.add_i32(lane16, Opnd::I32(o.write_extra));
    b.store_lds_b128(a0, v);
    b.tile_end();
    b.end_if();
    if o.producer_wait {
        b.wait_lds(0);
    }
    if o.prologue_barrier {
        b.barrier();
    }
    let vars = b.begin_loop(&[(DType::I32, Opnd::I32(0))]);
    let it = vars[0];
    let nxt = b.add_i32(it, Opnd::I32(1));
    let mask = Opnd::I32(o.mask.unwrap_or(o.slots as i32 - 1));
    let slot_n = if o.deep {
        let lo = b.let_(Op::AndI32(Opnd::Val(nxt), mask));
        let sh = b.let_(Op::LshrI32(Opnd::Val(nxt), Opnd::I32(8)));
        let hi = b.let_(Op::AndI32(Opnd::Val(sh), Opnd::I32(1)));
        b.add_i32(lo, hi)
    } else {
        b.let_(Op::AndI32(Opnd::Val(nxt), mask))
    };
    let slot_c = match o.cap {
        Some(cap) => {
            let m = b.let_(Op::MinI32(Opnd::Var(it), Opnd::I32(cap)));
            b.let_(Op::AndI32(Opnd::Val(m), mask))
        }
        None => b.let_(Op::AndI32(Opnd::Var(it), mask)),
    };
    let c1024 = b.c_i32(1024);
    b.begin_if(is_p);
    let pv = vtrans(&mut b, o.vt, nxt);
    b.tile_produce("buf", slot_n, pv);
    let base_n = b.mul_i32(slot_n, c1024);
    let wa = b.add_i32(base_n, lane16);
    let wa = b.add_i32(wa, Opnd::I32(o.write_extra));
    b.store_lds_b128(wa, v);
    b.tile_end();
    b.else_();
    let ver = b.add_i32(it, Opnd::I32(o.consume_ver_off));
    let cvv = vtrans(&mut b, o.vt, ver);
    b.tile_consume("buf", slot_c, cvv);
    let base_c = b.mul_i32(slot_c, c1024);
    let ra = b.add_i32(base_c, lane16);
    let _r = b.load_lds_b128(ra);
    b.tile_end();
    b.end_if();
    if o.consumer_wait {
        b.wait_lds(0);
    }
    b.barrier();
    b.set_var(it, nxt);
    let stop = match o.trip {
        Some(k) => b.let_(Op::CmpGeI32(Opnd::Val(nxt), Opnd::I32(k))),
        None => b.let_(Op::CmpGeI32(Opnd::Val(nxt), Opnd::Val(n))),
    };
    b.break_if(Opnd::Val(stop));
    b.end_loop();
    b.finish()
}

#[test]
fn a_correct_double_buffer_protocol_is_proven() {
    let r = verify_ir(&proto(good()));
    assert!(r.errors().is_empty(), "{}", errors(&r));
    assert!(r.proven.get("tile-reads").copied().unwrap_or(0) > 0, "{}", r.render());
    assert!(r.proven.get("tile-cover").copied().unwrap_or(0) > 0, "{}", r.render());
}

#[test]
fn a_single_buffer_overwritten_by_the_next_iteration_is_a_race() {
    // one slot: the producer writes version it+1 into the slot the consumer is reading (version it) in the same phase
    let r = verify_ir(&proto(Opt { slots: 1, ..good() }));
    assert!(has(&r, "GV-TILE-RACE") || has(&r, "GV-TILE-STALE"), "{}", r.render());
    assert!(has(&r, "GV-TILE-RACE"), "the overlapping write must be reported as a race: {}", r.render());
}

#[test]
fn reading_before_the_publishing_barrier_is_rejected() {
    // the prologue barrier is missing: the producer's version 0 is only published at the end of the first iteration's phase
    let r = verify_ir(&proto(Opt { prologue_barrier: false, ..good() }));
    assert!(has(&r, "GV-TILE-STALE"), "{}", r.render());
    assert!(r.errors().iter().any(|e| e.msg.contains("published at phase")), "{}", r.render());
}

#[test]
fn a_write_that_is_never_completed_is_not_published() {
    // no wait_lds(0) before the barrier: the producing wave's DS write is not known complete when the barrier is crossed
    let r = verify_ir(&proto(Opt { producer_wait: false, consumer_wait: false, ..good() }));
    assert!(has(&r, "GV-TILE-STALE"), "{}", r.render());
}

#[test]
fn a_consumer_expecting_the_wrong_version_sees_a_stale_slot() {
    // expects the NEXT version (it+1) while only version `it` is published
    let r = verify_ir(&proto(Opt { consume_ver_off: 1, ..good() }));
    assert!(has(&r, "GV-TILE-STALE"), "{}", r.render());
    // expects the PREVIOUS version (it-1): the slot already holds version `it`
    let r = verify_ir(&proto(Opt { consume_ver_off: -1, ..good() }));
    assert!(has(&r, "GV-TILE-STALE"), "{}", r.render());
}

#[test]
fn an_access_outside_its_marked_tile_escapes() {
    let r = verify_ir(&proto(Opt { write_extra: 4096, ..good() }));
    assert!(has(&r, "GV-TILE-ESCAPE"), "{}", r.render());
}

#[test]
fn overlapping_live_tiles_are_rejected_unless_aliased() {
    let build = |alias: bool| {
        let mut b = FnBuilder::new("ovl", 64);
        b.lds(8192);
        b.lds_tile("a", 0, 2048);
        b.lds_tile("b", 1024, 2048);
        if alias {
            b.lds_alias("a", "b");
        }
        b.finish()
    };
    let r = verify_ir(&build(false));
    assert!(has(&r, "GV-TILE-OVERLAP"), "{}", r.render());
    let r = verify_ir(&build(true));
    assert!(!has(&r, "GV-TILE-OVERLAP"), "{}", r.render());
    // a tile past the end of the declared segment
    let mut b = FnBuilder::new("big", 64);
    b.lds(1024);
    b.lds_tile("t", 512, 1024);
    assert!(has(&verify_ir(&b.finish()), "GV-TILE-OVERLAP"));
}

#[test]
fn a_barrier_on_one_role_only_is_rejected() {
    // role C skips a barrier the producer executes: the barrier count differs (GV-BAR) and the protocol phases no longer line up
    let mut b = FnBuilder::new("onebar", 128);
    b.lds(4096);
    b.lds_tile("t", 0, 1024);
    let t = b.let_(Op::WorkitemId);
    let c64 = b.c_i32(64);
    let is_p = b.cmp_lt_i32(t, c64);
    let z = b.c_i32(0);
    let v = b.frag4(z, z, z, z);
    b.begin_if(is_p);
    b.tile_produce("t", Opnd::I32(0), Opnd::I32(0));
    b.store_lds_b128(Opnd::I32(0), v);
    b.tile_end();
    b.wait_lds(0);
    b.barrier();
    b.else_();
    b.tile_consume("t", Opnd::I32(0), Opnd::I32(0));
    let _r = b.load_lds_b128(Opnd::I32(0));
    b.tile_end();
    b.end_if();
    let r = verify_ir(&b.finish());
    assert!(has(&r, "GV-BAR"), "{}", r.render());
    assert!(has(&r, "GV-TILE-STALE"), "the consumer never crosses the publishing barrier: {}", r.render());
}

#[test]
fn a_non_constant_slot_expression_is_reported() {
    let mut b = FnBuilder::new("idx", 128);
    b.lds(4096);
    b.arg("k", DType::I32, 0, 4);
    b.lds_ring("t", 0, 1024, 2);
    let k = b.let_(Op::KernargI32 { offset: 0 });
    let z = b.c_i32(0);
    let v = b.frag4(z, z, z, z);
    b.tile_produce("t", k, Opnd::I32(0));
    b.store_lds_b128(Opnd::I32(0), v);
    b.tile_end();
    let r = verify_ir(&b.finish());
    assert!(has(&r, "GV-TILE-INDEX"), "{}", r.render());
}
#[test]
fn the_shipped_cross_wave_exchange_probe_proves_its_tile_protocol() {
    let f = amdgpu_compiler::kernel_by_name("probe_roles_loop").unwrap();
    let r = verify_ir(&f);
    let tile_errors: Vec<_> = r.errors().into_iter().filter(|e| e.code.starts_with("GV-TILE")).collect();
    assert!(tile_errors.is_empty(), "{}", r.render());
    assert!(r.proven.get("tile-reads").copied().unwrap_or(0) > 0, "{}", r.render());
}

#[test]
fn removing_the_separating_barrier_of_the_shipped_probe_is_caught() {
    // mutate the shipped kernel: drop the SECOND barrier of each iteration (the one that separates the partner reads from the next iteration's writes)
    let mut f = amdgpu_compiler::kernel_by_name("probe_roles_loop").unwrap();
    let mut seen = 0;
    f.stmts.retain(|s| {
        if matches!(s, amdgpu_dsl::Stmt::Barrier) {
            seen += 1;
            return seen != 2;
        }
        true
    });
    let r = verify_ir(&f);
    assert!(r.errors().iter().any(|e| e.code == "GV-TILE-RACE" || e.code == "GV-TILE-STALE"), "{}", r.render());
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// the unroll gate: the simulator covers a bounded number of iterations, so the markers must be periodic in the loop counter (or explicitly trusted)

#[test]
fn a_protocol_that_is_right_only_for_the_first_iterations_is_rejected() {
    // consumer slot = min(it, 4) & 1: identical to it & 1 for it <= 4 (the old 1..=4 unroll saw nothing), then it sticks while the producer keeps alternating
    let r = verify_ir(&proto(Opt { cap: Some(4), ..good() }));
    assert!(has(&r, "GV-TILE-UNROLL-LIMIT"), "{}", r.render());
    assert!(has(&r, "GV-TILE-RACE") || has(&r, "GV-TILE-STALE"), "the simulation (10 iterations) also sees the late failure: {}", r.render());
}

#[test]
fn a_non_periodic_marker_beyond_the_simulated_iterations_is_gated_not_certified() {
    // min(it, 40) & 1: indistinguishable from it & 1 within the simulated iterations; only the periodicity gate can catch it
    let r = verify_ir(&proto(Opt { cap: Some(40), ..good() }));
    assert!(has(&r, "GV-TILE-UNROLL-LIMIT"), "{}", r.render());
    assert!(r.errors().iter().any(|e| e.msg.contains("NOT covered") && e.msg.contains("Min")), "{}", r.render());
}

#[test]
fn a_periodic_protocol_is_clean_for_any_trip_count_and_for_a_constant_bound() {
    for trip in [None, Some(3), Some(4)] {
        let r = verify_ir(&proto(Opt { trip, ..good() }));
        assert!(r.errors().is_empty(), "trip {trip:?}: {}", errors(&r));
    }
    // a 3-slot ring (period 3) with a runtime trip count
    let r = verify_ir(&proto(Opt { slots: 4, ..good() }));
    assert!(r.errors().is_empty(), "{}", errors(&r));
}

#[test]
fn tile_trust_periodic_is_an_explicit_listed_escape() {
    let mut f = proto(Opt { cap: Some(40), ..good() });
    f.contract.tile_trust = Some("test: the cap never binds in the launches this kernel is used for".to_string());
    let r = verify_ir(&f);
    assert!(!has(&r, "GV-TILE-UNROLL-LIMIT"), "{}", r.render());
    assert!(r.findings.iter().any(|x| x.code == "GV-ASSUME" && x.msg.contains("tile_trust_periodic")), "{}", r.render());
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// horizon vs period: a marker whose slot has a period longer than the simulated horizon must not be certified

#[test]
fn a_marker_period_longer_than_the_horizon_is_rejected_unless_trusted() {
    // slot = (nxt & 1) + ((nxt >> 8) & 1): affine / bitwise, ordinary double buffering for 255 iterations, then the extra bit moves the producer onto the consumer's slot
    let r = verify_ir(&proto(Opt { slots: 4, mask: Some(1), deep: true, ..good() }));
    assert!(has(&r, "GV-TILE-UNROLL-LIMIT"), "{}", r.render());
    assert!(r.errors().iter().any(|e| e.msg.contains("period of up to 512")), "{}", r.render());
    // the explicit, counted trust entry is the only way through
    let mut f = proto(Opt { slots: 4, mask: Some(1), deep: true, ..good() });
    f.contract.tile_trust = Some("test: launches never reach 256 iterations".to_string());
    let r = verify_ir(&f);
    assert!(!has(&r, "GV-TILE-UNROLL-LIMIT"), "{}", r.render());
    assert!(r.findings.iter().any(|x| x.code == "GV-ASSUME" && x.msg.contains("tile_trust_periodic")), "{}", r.render());
}

#[test]
fn a_true_period_two_ring_reports_its_period_and_is_clean_for_any_trip_count() {
    let r = verify_ir(&proto(good()));
    assert!(r.errors().is_empty(), "{}", errors(&r));
    assert!(r.findings.iter().any(|x| x.code == "GV-TILE-PERIOD" && x.msg.contains("dividing 2 ")), "{}", r.render());
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// wave-uniform runtime ifs: BOTH outcomes are explored

fn uniform_if_kernel(want: i32, in_loop: bool) -> Function {
    let mut b = FnBuilder::new("uif", 128);
    b.lds(4096);
    b.arg("flag", DType::I32, 0, 4);
    b.lds_tile("t", 0, 1024);
    let flag = b.let_(Op::KernargI32 { offset: 0 });
    let t = b.let_(Op::WorkitemId);
    let c64 = b.c_i32(64);
    let is_p = b.cmp_lt_i32(t, c64);
    let z = b.c_i32(0);
    let v = b.frag4(z, z, z, z);
    let go = b.cmp_lt_i32(z, flag);
    let lane = b.let_(Op::AndI32(Opnd::Val(t), Opnd::I32(63)));
    let c16 = b.c_i32(16);
    let a = b.mul_i32(lane, c16);
    b.begin_if(is_p);
    b.tile_produce("t", Opnd::I32(0), Opnd::I32(0));
    b.store_lds_b128(a, v);
    b.tile_end();
    b.end_if();
    b.wait_lds(0);
    b.barrier();
    let vars = if in_loop { Some(b.begin_loop(&[(DType::I32, Opnd::I32(0))])) } else { None };
    b.begin_if(go); // wave-uniform, runtime condition
    b.begin_if(is_p); // not the producer role: the consumer role
    b.else_();
    b.tile_consume("t", Opnd::I32(0), Opnd::I32(want));
    let _r = b.load_lds_b128(a);
    b.tile_end();
    b.end_if();
    b.end_if();
    b.wait_lds(0);
    if let Some(vs) = vars {
        b.barrier();
        let nx = b.add_i32(vs[0], Opnd::I32(1));
        b.set_var(vs[0], nx);
        let stop = b.let_(Op::CmpGeI32(Opnd::Val(nx), Opnd::I32(3)));
        b.break_if(Opnd::Val(stop));
        b.end_loop();
    }
    b.finish()
}

#[test]
fn both_outcomes_of_a_wave_uniform_runtime_if_are_explored() {
    // the read sits in the `then` arm of a runtime if: the correct version passes, the wrong one is found - so that arm really is simulated
    let r = verify_ir(&uniform_if_kernel(0, false));
    assert!(r.errors().is_empty(), "{}", errors(&r));
    assert!(r.proven.get("tile-reads").copied().unwrap_or(0) > 0, "{}", r.render());
    let r = verify_ir(&uniform_if_kernel(1, false));
    assert!(has(&r, "GV-TILE-STALE"), "{}", r.render());
}

#[test]
fn a_runtime_if_inside_a_loop_with_protocol_events_is_reported_not_skipped() {
    let r = verify_ir(&uniform_if_kernel(0, true));
    assert!(has(&r, "GV-TILE-UNSUPPORTED"), "{}", r.render());
    assert!(r.errors().iter().any(|e| e.msg.contains("INSIDE a loop")), "{}", r.render());
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// the shipped FMHA kernels: the lifetime simulation really runs, and is not vacuous (mutation test)

#[test]
fn vs128_runs_the_lifetime_simulation_and_removed_barriers_are_noticed() {
    let f = amdgpu_compiler::kernel_by_name("fmha_fwd_hd128_mxfp8_vs128").unwrap();
    let r = verify_ir(&f);
    assert!(r.errors().is_empty(), "{}", errors(&r));
    assert!(r.proven.get("tile-reads").copied().unwrap_or(0) > 1000, "{}", r.render());
    let barriers: Vec<usize> = f.stmts.iter().enumerate().filter(|(_, s)| matches!(s, amdgpu_dsl::Stmt::Barrier)).map(|(i, _)| i).collect();
    assert!(barriers.len() >= 6, "{} barriers", barriers.len());
    // remove every barrier in turn: the markers must notice most of them (a barrier that protects no marked tile is legitimately silent, e.g. the last one)
    let mut undetected: Vec<usize> = Vec::new();
    for (k, &stmt) in barriers.iter().enumerate() {
        let mut g = f.clone();
        g.stmts.remove(stmt);
        let m = verify_ir(&g);
        if !m.errors().iter().any(|e| e.code == "GV-BAR" || e.code.starts_with("GV-TILE")) {
            undetected.push(k);
        }
    }
    eprintln!("barriers: {}, removal undetected for barrier indices {:?} (stmts {:?})", barriers.len(), undetected, undetected.iter().map(|&k| barriers[k]).collect::<Vec<_>>());
    assert!(undetected.len() * 10 <= barriers.len() * 3, "too many removed barriers go unnoticed: {:?}", undetected);
}
// ---------------------------------------------------------------------------------------------------------------------------------------------
// private (per-wave) LDS regions: own-wave lifetime, symbolic own-slice proof

#[derive(Clone, Copy)]
struct Priv {
    /// read the slice of wave `wid ^ 1` instead of the own one
    other_wave: bool,
    own_wait: bool,
    read_ver: i32,
    overlap_shared: bool,
    untagged_poke: bool,
}

fn private_kernel(p: Priv) -> Function {
    let mut b = FnBuilder::new("priv", 128);
    b.lds(8192);
    b.lds_private("scr", 0, 1024, 2);
    if p.overlap_shared {
        b.lds_tile("shared", 1024, 1024); // inside wave 1's slice
    }
    let t = b.let_(Op::WorkitemId);
    let wid = b.let_(Op::LshrI32(Opnd::Val(t), Opnd::I32(6)));
    let lane = b.let_(Op::AndI32(Opnd::Val(t), Opnd::I32(63)));
    let c1024 = b.c_i32(1024);
    let c16 = b.c_i32(16);
    let base = b.mul_i32(wid, c1024);
    let l16 = b.mul_i32(lane, c16);
    let a = b.add_i32(base, l16);
    let z = b.c_i32(0);
    let v = b.frag4(z, z, z, z);
    b.tile_produce("scr", wid, Opnd::I32(0));
    b.store_lds_b128(a, v);
    b.tile_end();
    if p.own_wait {
        b.wait_lds(0);
    }
    // a barrier between produce and consume is NOT needed for own-wave data, but must be harmless
    b.barrier();
    let ra = if p.other_wave {
        let o = b.let_(Op::XorI32(Opnd::Val(wid), Opnd::I32(1)));
        let ob = b.mul_i32(o, c1024);
        b.add_i32(ob, l16)
    } else {
        a
    };
    b.tile_consume("scr", wid, Opnd::I32(p.read_ver));
    let _r = b.load_lds_b128(ra);
    b.tile_end();
    if p.untagged_poke {
        let _q = b.load_lds_b128(l16); // untagged, lands in wave 0's slice
    }
    b.finish()
}

fn good_priv() -> Priv {
    Priv { other_wave: false, own_wait: true, read_ver: 0, overlap_shared: false, untagged_poke: false }
}

#[test]
fn a_private_per_wave_scratch_is_proven_without_cross_wave_publication() {
    let r = verify_ir(&private_kernel(good_priv()));
    assert!(r.errors().is_empty(), "{}", errors(&r));
    assert!(r.proven.get("tile-reads").copied().unwrap_or(0) > 0, "{}", r.render());
    assert!(r.proven.get("tile-cover").copied().unwrap_or(0) >= 2, "{}", r.render());
}

#[test]
fn reading_another_waves_private_slice_is_rejected() {
    let r = verify_ir(&private_kernel(Priv { other_wave: true, ..good_priv() }));
    assert!(has(&r, "GV-TILE-PRIVATE-ESCAPE"), "{}", r.render());
}

#[test]
fn reading_a_private_slice_before_the_own_wait_is_rejected() {
    let r = verify_ir(&private_kernel(Priv { own_wait: false, ..good_priv() }));
    assert!(has(&r, "GV-TILE-STALE"), "{}", r.render());
    assert!(r.errors().iter().any(|e| e.msg.contains("not completed by this wave")), "{}", r.render());
}

#[test]
fn reading_the_wrong_private_version_is_stale() {
    let r = verify_ir(&private_kernel(Priv { read_ver: 1, ..good_priv() }));
    assert!(has(&r, "GV-TILE-STALE"), "{}", r.render());
}

#[test]
fn a_private_region_overlapping_a_shared_tile_is_rejected() {
    let r = verify_ir(&private_kernel(Priv { overlap_shared: true, ..good_priv() }));
    assert!(has(&r, "GV-TILE-OVERLAP"), "{}", r.render());
}

#[test]
fn an_unmarked_access_that_may_touch_a_private_region_is_rejected() {
    let r = verify_ir(&private_kernel(Priv { untagged_poke: true, ..good_priv() }));
    assert!(has(&r, "GV-TILE-PRIVATE-ESCAPE"), "{}", r.render());
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// the period claim must be a TRUE period (kernel-occ audit-of-record)

#[test]
fn a_version_xored_with_a_wide_constant_has_a_long_period_and_is_rejected() {
    // version = nxt ^ 256: the relation between neighbouring versions changes every 512 iterations
    let r = verify_ir(&proto(Opt { vt: 1, ..good() }));
    assert!(has(&r, "GV-TILE-UNROLL-LIMIT"), "{}", r.render());
    assert!(r.errors().iter().any(|e| e.msg.contains("period of up to 512")), "{}", r.render());
}

#[test]
fn a_huge_multiplier_or_shift_that_can_wrap_the_version_is_rejected() {
    let r = verify_ir(&proto(Opt { vt: 4, ..good() })); // << 24: wraps after 256 iterations
    assert!(has(&r, "GV-TILE-UNROLL-LIMIT"), "{}", r.render());
    assert!(r.errors().iter().any(|e| e.msg.contains("wrap")), "{}", r.render());
}

#[test]
fn affine_scalings_report_a_verified_period_not_a_made_up_one() {
    // version = nxt << 8 and nxt * 256 advance by a constant 256 per iteration: truly period 1 in the relative sense; the claim is verified on the simulated sequence
    for vt in [2u8, 3] {
        let r = verify_ir(&proto(Opt { vt, ..good() }));
        assert!(r.errors().is_empty(), "vt {vt}: {}", errors(&r));
        assert!(r.findings.iter().any(|x| x.code == "GV-TILE-PERIOD" && x.msg.contains("VERIFIED")), "{}", r.render());
    }
}

#[test]
fn a_genuine_period_two_ring_still_reports_a_verified_true_period() {
    let r = verify_ir(&proto(good()));
    assert!(r.errors().is_empty(), "{}", errors(&r));
    assert!(r.findings.iter().any(|x| x.code == "GV-TILE-PERIOD" && x.msg.contains("dividing 2 ") && x.msg.contains("VERIFIED")), "{}", r.render());
}