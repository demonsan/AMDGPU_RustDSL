//! gpu_verify v4 protocol model: the four axes (arrival / completion / visibility / generation) and the four obligations (O1 complete-before-publish, O2 publish-before-read,
//! O3 read-before-overwrite, O4 generation-consistency) on LDS (legacy GV-TILE-* names with GV-PROTO-* aliases) and on GLOBAL regions (GV-PROTO-* names), the opt-in fused
//! KV-split combine (last-arriver) pattern, and the counted trust ledger. See docs/gpu-semantic-protocol.md.
use amdgpu_compiler::gpu_verify::{verify_ir, Ledger, Report, Severity};
use amdgpu_dsl::{DType, Ext, FnBuilder, Function, Op, Opnd, Stmt};

fn has(r: &Report, code: &str) -> bool {
    r.findings.iter().any(|f| f.severity == Severity::Error && (f.code == code || f.alias == code))
}
fn errors(r: &Report) -> String {
    r.errors().iter().map(|e| format!("[{}|{}] {}", e.code, e.alias, e.msg)).collect::<Vec<_>>().join("\n")
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// LDS instance: legacy names + protocol aliases

fn lds_pair(read_gen: i32, wait: bool, barrier: bool) -> Function {
    let mut b = FnBuilder::new("lds_pair", 128);
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
    b.end_if();
    if wait {
        b.wait_lds(0);
    }
    if barrier {
        b.barrier();
    }
    b.begin_if(is_p);
    b.else_();
    b.tile_consume("t", Opnd::I32(0), Opnd::I32(read_gen));
    let _r = b.load_lds_b128(Opnd::I32(0));
    b.tile_end();
    b.end_if();
    b.finish()
}

#[test]
fn the_lds_instance_keeps_its_legacy_names_and_exposes_the_protocol_names() {
    let r = verify_ir(&lds_pair(0, true, true));
    assert!(r.errors().is_empty(), "{}", errors(&r));
    // wrong generation: O4
    let r = verify_ir(&lds_pair(1, true, true));
    assert!(r.findings.iter().any(|f| f.code == "GV-TILE-STALE" && f.alias == "GV-PROTO-STALE-GEN"), "{}", r.render());
    // no barrier: nothing publishes the data: O2
    let r = verify_ir(&lds_pair(0, true, false));
    assert!(r.findings.iter().any(|f| f.code == "GV-TILE-STALE" && f.alias == "GV-PROTO-UNPUBLISHED"), "{}", r.render());
    // no wait: the write is never complete: O1
    let r = verify_ir(&lds_pair(0, false, true));
    assert!(r.findings.iter().any(|f| f.code == "GV-TILE-STALE" && f.alias == "GV-PROTO-NO-COMPLETION"), "{}", r.render());
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// global-memory regions, intra-workgroup: produce -> complete -> release -> barrier (arrival) -> acquire -> consume

#[derive(Clone, Copy)]
struct G {
    wait: bool,
    release: Option<bool>,
    acquire: Option<bool>,
    read_gen: i32,
}

fn good() -> G {
    G { wait: true, release: Some(true), acquire: Some(true), read_gen: 0 }
}

fn greg(o: G) -> Function {
    let mut b = FnBuilder::new("greg", 128);
    b.arg("buf", DType::Ptr, 0, 8);
    b.arg_tensor("buf", Ext::c(8192));
    b.buf_region("r", "buf", Ext::c(0), Ext::c(8192));
    let p = b.let_(Op::KernargPtr { offset: 0 });
    let t = b.let_(Op::WorkitemId);
    let c64 = b.c_i32(64);
    let is_p = b.cmp_lt_i32(t, c64);
    let lane = b.let_(Op::AndI32(Opnd::Val(t), Opnd::I32(63)));
    let off = b.mul_i32(lane, Opnd::I32(4));
    let q = b.ptr_add(p, off);
    b.begin_if(is_p);
    b.buf_produce("r", Opnd::I32(0));
    b.store_f32(q, 0, Opnd::F32(1.0));
    b.tile_end();
    b.end_if();
    if o.wait {
        b.wait_vm(0);
    }
    match o.release {
        Some(true) => b.fence_release(),
        Some(false) => b.fence_release_lite(),
        None => {}
    }
    b.barrier();
    match o.acquire {
        Some(true) => b.fence_acquire(),
        Some(false) => b.fence_acquire_lite(),
        None => {}
    }
    b.begin_if(is_p);
    b.else_();
    b.buf_consume("r", Opnd::I32(o.read_gen));
    let _x = b.load_f32(q, 0);
    b.tile_end();
    b.end_if();
    b.finish()
}

#[test]
fn a_global_region_produced_released_published_acquired_and_consumed_is_proven() {
    let r = verify_ir(&greg(good()));
    assert!(r.errors().is_empty(), "{}", errors(&r));
    assert!(r.proven.get("tile-reads").copied().unwrap_or(0) > 0, "{}", r.render());
    assert!(r.proven.get("region-cover").copied().unwrap_or(0) >= 2, "the accesses must be proven inside the declared region: {}", r.render());
}

#[test]
fn reading_a_global_region_that_was_waited_but_never_released_is_not_visible() {
    let r = verify_ir(&greg(G { release: None, ..good() }));
    assert!(has(&r, "GV-PROTO-NOT-VISIBLE"), "{}", r.render());
}

#[test]
fn a_global_write_that_is_never_completed_has_no_completion() {
    let r = verify_ir(&greg(G { wait: false, release: None, ..good() }));
    assert!(has(&r, "GV-PROTO-NO-COMPLETION"), "{}", r.render());
}

#[test]
fn reading_before_the_acquire_is_not_visible() {
    let r = verify_ir(&greg(G { acquire: None, ..good() }));
    assert!(has(&r, "GV-PROTO-NOT-VISIBLE"), "{}", r.render());
    // a lite acquire (no invalidate) with plain loads does not make the lines visible either
    let r = verify_ir(&greg(G { acquire: Some(false), ..good() }));
    assert!(has(&r, "GV-PROTO-NOT-VISIBLE"), "{}", r.render());
}

#[test]
fn a_wrong_generation_is_stale_gen() {
    let r = verify_ir(&greg(G { read_gen: 1, ..good() }));
    assert!(has(&r, "GV-PROTO-STALE-GEN"), "{}", r.render());
}

/// loop: the producer writes generation it+1 while (`overlap`) / after (`!overlap`) the consumer reads generation it
fn greg_loop(overlap: bool) -> Function {
    let mut b = FnBuilder::new("greg_loop", 128);
    b.arg("buf", DType::Ptr, 0, 8);
    b.arg_tensor("buf", Ext::c(8192));
    b.buf_region("r", "buf", Ext::c(0), Ext::c(8192));
    let p = b.let_(Op::KernargPtr { offset: 0 });
    let t = b.let_(Op::WorkitemId);
    let c64 = b.c_i32(64);
    let is_p = b.cmp_lt_i32(t, c64);
    let lane = b.let_(Op::AndI32(Opnd::Val(t), Opnd::I32(63)));
    let off = b.mul_i32(lane, Opnd::I32(4));
    let q = b.ptr_add(p, off);
    let publish = |b: &mut FnBuilder| {
        b.wait_vm(0);
        b.fence_release();
        b.barrier();
        b.fence_acquire();
    };
    // prologue: generation 0
    b.begin_if(is_p);
    b.buf_produce("r", Opnd::I32(0));
    b.store_f32(q, 0, Opnd::F32(1.0));
    b.tile_end();
    b.end_if();
    publish(&mut b);
    let vars = b.begin_loop(&[(DType::I32, Opnd::I32(0))]);
    let it = vars[0];
    let nxt = b.add_i32(it, Opnd::I32(1));
    if overlap {
        b.begin_if(is_p);
        b.buf_produce("r", nxt);
        b.store_f32(q, 0, Opnd::F32(2.0));
        b.tile_end();
        b.else_();
        b.buf_consume("r", it);
        let _x = b.load_f32(q, 0);
        b.tile_end();
        b.end_if();
        publish(&mut b);
    } else {
        b.begin_if(is_p);
        b.else_();
        b.buf_consume("r", it);
        let _x = b.load_f32(q, 0);
        b.tile_end();
        b.end_if();
        b.wait_vm(0); // the reads are complete ...
        b.barrier(); // ... before the next generation may be written
        b.begin_if(is_p);
        b.buf_produce("r", nxt);
        b.store_f32(q, 0, Opnd::F32(2.0));
        b.tile_end();
        b.end_if();
        publish(&mut b);
    }
    b.set_var(it, nxt);
    let stop = b.let_(Op::CmpGeI32(Opnd::Val(nxt), Opnd::I32(3)));
    b.break_if(Opnd::Val(stop));
    b.end_loop();
    b.finish()
}

#[test]
fn a_second_generation_written_while_the_consumer_is_live_is_a_race() {
    let r = verify_ir(&greg_loop(true));
    assert!(has(&r, "GV-PROTO-RACE"), "{}", r.render());
}

#[test]
fn the_next_generation_after_the_consumers_reads_completed_is_clean() {
    let r = verify_ir(&greg_loop(false));
    assert!(r.errors().is_empty(), "{}", errors(&r));
    assert!(r.proven.get("tile-reads").copied().unwrap_or(0) > 0, "{}", r.render());
}

#[test]
fn an_access_outside_the_declared_global_region_escapes() {
    let mut b = FnBuilder::new("gesc", 64);
    b.arg("buf", DType::Ptr, 0, 8);
    b.arg_tensor("buf", Ext::c(8192));
    b.buf_region("r", "buf", Ext::c(0), Ext::c(1024));
    let p = b.let_(Op::KernargPtr { offset: 0 });
    b.buf_produce("r", Opnd::I32(0));
    b.store_f32(p, 4096, Opnd::F32(1.0)); // inside the allocation, outside the region
    b.tile_end();
    let r = verify_ir(&b.finish());
    assert!(has(&r, "GV-TILE-ESCAPE"), "{}", r.render());
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// the opt-in fused combine (last-arriver) pattern, on the shipped probe kernels

fn probe(name: &str) -> Function {
    amdgpu_compiler::kernel_by_name(name).unwrap_or_else(|| panic!("unknown kernel {name}"))
}

fn without(f: &Function, pred: &dyn Fn(&Stmt) -> bool) -> Function {
    let mut g = f.clone();
    let i = g.stmts.iter().position(|s| pred(s)).expect("statement to remove");
    g.stmts.remove(i);
    g
}

#[test]
fn the_full_and_the_lite_fused_combine_patterns_are_proven() {
    for name in ["probe_last_arriver", "probe_last_arriver_lite", "probe_last_arriver_plain_fenced"] {
        let r = verify_ir(&probe(name));
        assert!(r.errors().is_empty(), "{name}: {}", errors(&r));
        assert_eq!(r.proven.get("last-arriver"), Some(&1), "{name}: {}", r.render());
    }
}

#[test]
fn a_missing_release_is_no_completion_and_a_missing_acquire_is_not_visible() {
    let f = probe("probe_last_arriver");
    let r = verify_ir(&without(&f, &|s| matches!(s, Stmt::FenceRelease { .. })));
    assert!(has(&r, "GV-PROTO-NO-COMPLETION"), "{}", r.render());
    let r = verify_ir(&without(&f, &|s| matches!(s, Stmt::FenceAcquire { .. })));
    assert!(has(&r, "GV-PROTO-NOT-VISIBLE"), "{}", r.render());
    // the shipped negative controls (no fence at all) are rejected too
    for name in ["probe_last_arriver_nofence", "probe_last_arriver_plain"] {
        let r = verify_ir(&probe(name));
        assert!(has(&r, "GV-PROTO-NO-COMPLETION") && has(&r, "GV-PROTO-NOT-VISIBLE"), "{name}: {}", r.render());
    }
}

#[test]
fn reading_before_the_acquire_inside_the_last_arriver_branch_is_not_visible() {
    // move the acquire to AFTER the loop: the loads now precede it
    let f = probe("probe_last_arriver");
    let mut g = f.clone();
    let ai = g.stmts.iter().position(|s| matches!(s, Stmt::FenceAcquire { .. })).unwrap();
    let acq = g.stmts.remove(ai);
    let end = g.stmts.iter().rposition(|s| matches!(s, Stmt::EndLoop)).unwrap();
    g.stmts.insert(end + 1, acq);
    let r = verify_ir(&g);
    assert!(has(&r, "GV-PROTO-NOT-VISIBLE"), "{}", r.render());
}

#[test]
fn the_lite_acquire_needs_coherent_loads_and_the_lite_release_needs_write_through_stores() {
    let f = probe("probe_last_arriver_lite");
    let mut g = f.clone();
    for s in g.stmts.iter_mut() {
        if let Stmt::Let { op, .. } = s {
            if let Op::LoadF32Coherent { base, byte_off } = op.clone() {
                *op = Op::LoadF32 { base, byte_off };
            }
        }
    }
    let r = verify_ir(&g);
    assert!(has(&r, "GV-PROTO-NOT-VISIBLE"), "{}", r.render());
    // plain stores under a lite release
    let mut h = f.clone();
    for s in h.stmts.iter_mut() {
        if let Stmt::StoreF32Wt { base, byte_off, value } = s.clone() {
            // only the produced `part` store (the first write-through store), not the final result store
            if matches!(base, Opnd::Val(_)) {
                *s = Stmt::StoreF32 { base, byte_off, value };
                break;
            }
        }
    }
    let r = verify_ir(&h);
    assert!(has(&r, "GV-PROTO-NOT-VISIBLE"), "{}", r.render());
}

#[test]
fn a_missing_ticket_and_an_unguarded_read_are_arrival_mismatches() {
    let mut f = probe("probe_last_arriver");
    // the declared counter is not the argument the atomic ticks
    f.contract.last_arrivers[0].counter_arg = "result".to_string();
    let r = verify_ir(&f);
    assert!(has(&r, "GV-PROTO-ARRIVAL-MISMATCH"), "{}", r.render());
    // the last-arriver branch is removed: everybody reads
    let mut g = probe("probe_last_arriver");
    let bi = g.stmts.iter().position(|s| matches!(s, Stmt::BeginIf { .. })).unwrap();
    let ei = g.stmts.iter().rposition(|s| matches!(s, Stmt::EndIf)).unwrap();
    g.stmts.remove(ei);
    g.stmts.remove(bi);
    let r = verify_ir(&g);
    assert!(has(&r, "GV-PROTO-ARRIVAL-MISMATCH"), "{}", r.render());
}

#[test]
fn the_generation_must_be_the_blocks_split_id() {
    let mut f = probe("probe_last_arriver");
    for s in f.stmts.iter_mut() {
        if let Stmt::TileProduce { version, .. } = s {
            *version = Opnd::I32(0); // every block claims generation 0: no agreement on which slice it owns
        }
    }
    let r = verify_ir(&f);
    assert!(has(&r, "GV-PROTO-STALE-GEN"), "{}", r.render());
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// the trust ledger counts every escape and aggregates

fn trusting_kernel() -> Function {
    let mut b = FnBuilder::new("trust", 64);
    b.lds(4096);
    b.arg("p", DType::Ptr, 0, 8);
    b.arg_tensor("p", Ext::c(4096));
    b.buf_region("g", "p", Ext::c(0), Ext::c(4096));
    b.buf_trust("g", "cross-kernel visibility is argued by the host launch order");
    let z = b.c_i32(0);
    let vars = b.begin_loop(&[(DType::I32, Opnd::Val(z))]);
    b.assume_trusted(vars[0], Ext::c(0), Ext::c(3), "counter bounded by the exit test");
    let nx = b.add_i32(vars[0], Opnd::I32(1));
    b.set_var(vars[0], nx);
    let stop = b.let_(Op::CmpGeI32(Opnd::Val(nx), Opnd::I32(4)));
    b.break_if(Opnd::Val(stop));
    b.end_loop();
    let mut f = b.finish();
    f.contract.tile_trust = Some("periodic by construction".to_string());
    f
}

#[test]
fn the_trust_ledger_counts_every_escape_by_kind_and_aggregates() {
    let r = verify_ir(&trusting_kernel());
    assert!(r.errors().is_empty(), "{}", errors(&r));
    let l = r.ledger();
    assert_eq!(l, Ledger { assume_trusted: 1, tile_trust: 1, region_trust: 1 }, "{}", r.render());
    assert_eq!(l.total(), 3);
    // aggregation over reports (what the certificate line prints)
    let total: usize = [verify_ir(&trusting_kernel()), verify_ir(&trusting_kernel())].iter().map(|r| r.ledger().total()).sum();
    assert_eq!(total, 6);
    // the trust findings carry the GV-PROTO-TRUST name
    assert!(r.findings.iter().filter(|f| f.alias == "GV-PROTO-TRUST").count() >= 3, "{}", r.render());
}

#[test]
fn the_shipped_probe_certificate_counts_its_one_trusted_counter() {
    let r = verify_ir(&probe("probe_last_arriver"));
    assert_eq!(r.ledger(), Ledger { assume_trusted: 1, tile_trust: 0, region_trust: 0 }, "{}", r.render());
}