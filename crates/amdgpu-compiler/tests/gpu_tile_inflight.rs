//! LDS reads in flight across a barrier (the vendor's V read-ahead, docs/gpu-semantic-protocol.md): a read issued before a barrier and retired
//! (wait_lds(0)) after it is safe iff no producer writes the slot in any phase of the read's lifetime [issue phase, retiring-wait phase].
use amdgpu_compiler::gpu_verify::{verify_ir, Report, Severity};
use amdgpu_dsl::{DType, FnBuilder, Function, Op, Opnd};

fn has(rep: &Report, code: &str, sev: Severity) -> bool {
    rep.findings.iter().any(|f| f.code == code && f.severity == sev)
}
fn errors(rep: &Report) -> String {
    rep.findings.iter().map(|e| format!("[{}|{:?}] {}", e.code, e.severity, e.msg)).collect::<Vec<_>>().join("\n")
}

#[derive(Clone, Copy, PartialEq)]
enum Next {
    /// the next version is written after the barrier that follows the consumer's retiring wait
    AfterRetire,
    /// the next version is written in the phase in which the consumer's read is still in flight (between the barrier and the retiring wait)
    WhileInFlight,
    /// the consumer never retires its read (no wait_lds(0) at all) and a barrier follows
    NeverRetired,
}

/// role 0 (lanes < 64) produces `t`[0], role 1 reads it; the read is issued before a barrier and retired after it.
fn kernel(next: Next) -> Function {
    let mut b = FnBuilder::new("inflight", 128);
    b.lds(1024);
    b.lds_tile("t", 0, 256);
    let tid = b.let_(Op::WorkitemId);
    let c64 = b.c_i32(64);
    let is_p = b.cmp_lt_i32(tid, c64);
    let c4 = b.c_i32(4);
    let l = b.and_i32(Opnd::Val(tid), Opnd::I32(63));
    let addr = b.mul_i32(Opnd::Val(l), Opnd::Val(c4));
    let one = b.c_i32(1);
    // phase 0: produce version 0, publish at barrier 0
    b.begin_if(Opnd::Val(is_p));
    b.tile_produce("t", Opnd::I32(0), Opnd::I32(0));
    b.store_lds_b32(Opnd::Val(addr), Opnd::Val(one));
    b.tile_end();
    b.wait_lds(0);
    b.else_();
    b.end_if();
    b.barrier();
    // phase 1: role 1 issues the read and does NOT wait before barrier 1
    b.begin_if(Opnd::Val(is_p));
    b.else_();
    b.tile_consume("t", Opnd::I32(0), Opnd::I32(0));
    let _x = b.load_lds_b32(Opnd::Val(addr));
    b.tile_end();
    b.end_if();
    b.barrier();
    // phase 2: role 1 retires the read here (after the barrier); role 0 may or may not write the next version now
    b.begin_if(Opnd::Val(is_p));
    if next == Next::WhileInFlight {
        b.tile_produce("t", Opnd::I32(0), Opnd::I32(1));
        b.store_lds_b32(Opnd::Val(addr), Opnd::Val(one));
        b.tile_end();
        b.wait_lds(0);
    }
    b.else_();
    if next != Next::NeverRetired {
        b.wait_lds(0);
    }
    b.end_if();
    b.barrier();
    // phase 3: the next version after the barrier that follows the retiring wait
    b.begin_if(Opnd::Val(is_p));
    if next != Next::WhileInFlight {
        b.tile_produce("t", Opnd::I32(0), Opnd::I32(1));
        b.store_lds_b32(Opnd::Val(addr), Opnd::Val(one));
        b.tile_end();
        b.wait_lds(0);
    }
    b.else_();
    b.end_if();
    b.barrier();
    let _ = DType::I32;
    b.finish()
}

#[test]
fn a_read_retired_after_the_barrier_is_safe_when_no_writer_enters_its_window() {
    let r = verify_ir(&kernel(Next::AfterRetire));
    println!("{}", errors(&r));
    assert!(r.errors().is_empty(), "{}", errors(&r));
    assert!(!has(&r, "GV-TILE-PENDING", Severity::Warn), "the read is retired (wait_lds(0) after the barrier): no PENDING\n{}", errors(&r));
}

#[test]
fn a_writer_between_the_barrier_and_the_retiring_wait_is_a_race() {
    let r = verify_ir(&kernel(Next::WhileInFlight));
    println!("{}", errors(&r));
    assert!(has(&r, "GV-TILE-RACE", Severity::Error), "{}", errors(&r));
}

#[test]
fn a_read_that_is_never_retired_before_a_barrier_stays_pending() {
    let r = verify_ir(&kernel(Next::NeverRetired));
    println!("{}", errors(&r));
    assert!(has(&r, "GV-TILE-PENDING", Severity::Warn), "{}", errors(&r));
}
