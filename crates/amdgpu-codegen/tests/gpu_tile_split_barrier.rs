//! Split workgroup barriers (gfx1250 `s_barrier_signal -1` ... `s_barrier_wait -1`) across two wave roles in the tile protocol: the signal is the
//! arrival (it publishes the wave's completed LDS writes), the wait is the pass (only after it may a wave read what the others published).
//! Accesses between a wave's signal and its wait may run concurrently with the other waves on both sides of the barrier.

use amdgpu_codegen::gpu_verify::{verify_ir, Report, Severity};
use amdgpu_dsl::{FnBuilder, Function, MemTy, Opnd};

fn errors(rep: &Report) -> String {
    rep.errors().iter().map(|e| format!("[{}] {}", e.code, e.msg)).collect::<Vec<_>>().join("\n")
}
fn has_error(rep: &Report) -> bool {
    rep.findings.iter().any(|f| f.severity == Severity::Error && f.code.starts_with("GV-TILE"))
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    /// producer: write t, complete, signal, independent work, wait; consumer: signal, wait, read t
    Clean,
    /// producer writes t again (the next version) between its signal and its wait: the consumer may already be reading
    WriteAfterSignal,
    /// consumer reads t between its signal and its wait: the producer may not have arrived yet
    ReadBeforeWait,
    /// producer completes its writes only after its signal (the signal publishes an incomplete write)
    CompleteAfterSignal,
    /// producer writes ANOTHER tile `u` between signal and wait (consumed only after the next full barrier): fine
    OtherTileAfterSignal,
}

fn kernel(mode: Mode) -> Function {
    let mut b = FnBuilder::new("split_barrier_roles", 128);
    b.lds(2048);
    b.lds_tile("t", 0, 1024);
    b.lds_tile("u", 1024, 1024);
    let tid = b.let_(amdgpu_dsl::Op::WorkitemId);
    let c64 = b.c_i32(64);
    let is_p = b.cmp_lt_i32(tid, c64);
    let addr = b.let_(amdgpu_dsl::Op::ShlI32(Opnd::Val(tid), Opnd::I32(2)));
    let a64 = b.let_(amdgpu_dsl::Op::AndI32(Opnd::Val(addr), Opnd::I32(255)));
    let au = b.let_(amdgpu_dsl::Op::AddI32(Opnd::Val(a64), Opnd::I32(1024)));
    let one = b.c_i32(1);
    b.begin_if(Opnd::Val(is_p));
    {
        b.tile_produce("t", Opnd::I32(0), Opnd::I32(0));
        b.store_lds(a64, one, MemTy::B32, false);
        b.tile_end();
        if mode != Mode::CompleteAfterSignal {
            b.wait_lds(0);
        }
        b.barrier_signal(-1);
        if mode == Mode::CompleteAfterSignal {
            b.wait_lds(0);
        }
        if mode == Mode::WriteAfterSignal {
            b.tile_produce("t", Opnd::I32(0), Opnd::I32(1));
            b.store_lds(a64, one, MemTy::B32, false);
            b.tile_end();
            b.wait_lds(0);
        }
        if mode == Mode::OtherTileAfterSignal {
            b.tile_produce("u", Opnd::I32(0), Opnd::I32(0));
            b.store_lds(au, one, MemTy::B32, false);
            b.tile_end();
            b.wait_lds(0);
        }
        b.barrier_wait(-1);
    }
    b.else_();
    {
        b.barrier_signal(-1);
        if mode == Mode::ReadBeforeWait {
            b.tile_consume("t", Opnd::I32(0), Opnd::I32(0));
            let _ = b.load_lds(a64, MemTy::B32);
            b.tile_end();
            b.wait_lds(0);
        }
        b.barrier_wait(-1);
        b.tile_consume("t", Opnd::I32(0), Opnd::I32(0));
        let _ = b.load_lds(a64, MemTy::B32);
        b.tile_end();
        b.wait_lds(0);
    }
    b.end_if();
    b.barrier();
    if mode == Mode::OtherTileAfterSignal {
        b.begin_if(Opnd::Val(is_p));
        b.else_();
        b.tile_consume("u", Opnd::I32(0), Opnd::I32(0));
        let _ = b.load_lds(au, MemTy::B32);
        b.tile_end();
        b.wait_lds(0);
        b.end_if();
    }
    b.finish()
}

#[test]
fn a_split_barrier_publishes_at_the_signal_and_the_consumer_reads_after_its_wait() {
    let r = verify_ir(&kernel(Mode::Clean));
    assert!(!has_error(&r), "{}", errors(&r));
}

#[test]
fn a_write_after_the_signal_to_a_consumed_tile_is_rejected() {
    let r = verify_ir(&kernel(Mode::WriteAfterSignal));
    assert!(has_error(&r), "a producer write between its signal and its wait races with a consumer past its wait");
}

#[test]
fn a_read_before_the_wait_is_rejected() {
    let r = verify_ir(&kernel(Mode::ReadBeforeWait));
    assert!(has_error(&r), "a consumer read between its signal and its wait may run before the producer arrived");
}

#[test]
fn a_write_completed_only_after_the_signal_is_not_published_by_it() {
    let r = verify_ir(&kernel(Mode::CompleteAfterSignal));
    assert!(has_error(&r), "the signal published an incomplete write");
}

#[test]
fn work_on_another_tile_between_signal_and_wait_is_fine() {
    let r = verify_ir(&kernel(Mode::OtherTileAfterSignal));
    assert!(!has_error(&r), "{}", errors(&r));
}
