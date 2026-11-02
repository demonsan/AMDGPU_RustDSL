//! gpu_verify v1.1: LDS tile lifetime / versioning. A two-role (producer wave / consumer wave) ring protocol, one positive case and one negative case per rule.
use amdgpu_compiler::gpu_verify::{verify_ir, Report, Severity};
use amdgpu_dsl::{DType, FnBuilder, Function, Op, Opnd};

#[derive(Clone, Copy)]
struct Opt {
    slots: u32,
    prologue_barrier: bool,
    producer_wait: bool,
    consume_ver_off: i32,
    write_extra: i32,
    consumer_wait: bool,
}

fn good() -> Opt {
    Opt { slots: 2, prologue_barrier: true, producer_wait: true, consume_ver_off: 0, write_extra: 0, consumer_wait: true }
}

fn has(rep: &Report, code: &str) -> bool {
    rep.findings.iter().any(|f| f.code == code && f.severity == Severity::Error)
}

fn errors(rep: &Report) -> String {
    rep.errors().iter().map(|e| format!("[{}] {}", e.code, e.msg)).collect::<Vec<_>>().join("\n")
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
    let mask = Opnd::I32(o.slots as i32 - 1);
    let slot_n = b.let_(Op::AndI32(Opnd::Val(nxt), mask));
    let slot_c = b.let_(Op::AndI32(Opnd::Var(it), mask));
    let c1024 = b.c_i32(1024);
    b.begin_if(is_p);
    b.tile_produce("buf", slot_n, nxt);
    let base_n = b.mul_i32(slot_n, c1024);
    let wa = b.add_i32(base_n, lane16);
    let wa = b.add_i32(wa, Opnd::I32(o.write_extra));
    b.store_lds_b128(wa, v);
    b.tile_end();
    b.else_();
    let ver = b.add_i32(it, Opnd::I32(o.consume_ver_off));
    b.tile_consume("buf", slot_c, ver);
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
    let stop = b.let_(Op::CmpGeI32(Opnd::Val(nxt), Opnd::Val(n)));
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