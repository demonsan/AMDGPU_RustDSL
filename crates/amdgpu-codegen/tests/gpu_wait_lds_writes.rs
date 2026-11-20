//! `wait_lds_writes()`: the compiler computes the LDS count that completes this wave's earlier LDS writes while the reads issued after them stay
//! in flight; the tile protocol takes it as the completion of writes, never of reads.

use amdgpu_codegen::gpu_verify::{verify_ir, Report, Severity};
use amdgpu_dsl::{FnBuilder, Function, MemTy, Op, Opnd};

fn errors(rep: &Report) -> String {
    rep.errors().iter().map(|e| format!("[{}] {}", e.code, e.msg)).collect::<Vec<_>>().join("\n")
}
fn has_error(rep: &Report) -> bool {
    rep.findings.iter().any(|f| f.severity == Severity::Error && f.code.starts_with("GV-TILE"))
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    /// producer: store t, read u (in flight), wait_lds_writes, barrier; consumer reads t after the barrier
    Clean,
    /// a second store of t AFTER the wait_lds_writes, before the barrier: published incomplete
    StoreAfterWait,
    /// the consumer "completes" its reads of t with wait_lds_writes before the barrier after which the producer overwrites t
    ReadsNotCompleted,
}

fn kernel(mode: Mode) -> Function {
    let mut b = FnBuilder::new("wait_lds_writes", 128);
    b.lds(2048);
    b.lds_tile("t", 0, 1024);
    b.lds_tile("u", 1024, 1024);
    let tid = b.let_(Op::WorkitemId);
    let c64 = b.c_i32(64);
    let is_p = b.cmp_lt_i32(tid, c64);
    let a0 = b.let_(Op::ShlI32(Opnd::Val(tid), Opnd::I32(2)));
    let a = b.let_(Op::AndI32(Opnd::Val(a0), Opnd::I32(255)));
    let au = b.let_(Op::AddI32(Opnd::Val(a), Opnd::I32(1024)));
    let one = b.c_i32(1);
    // u is published before (the producer's in-flight reads of it are harmless)
    b.tile_produce("u", Opnd::I32(0), Opnd::I32(0));
    b.store_lds(au, one, MemTy::B32, false);
    b.tile_end();
    b.wait_lds(0);
    b.barrier();
    b.begin_if(Opnd::Val(is_p));
    {
        b.tile_produce("t", Opnd::I32(0), Opnd::I32(0));
        b.store_lds(a, one, MemTy::B32, false);
        b.tile_end();
        b.tile_consume("u", Opnd::I32(0), Opnd::I32(0));
        let _ = b.load_lds(au, MemTy::B32);
        b.tile_end();
        b.wait_lds_writes();
        if mode == Mode::StoreAfterWait {
            b.tile_produce("t", Opnd::I32(0), Opnd::I32(0));
            b.store_lds(a, one, MemTy::B32, false);
            b.tile_end();
        }
        b.barrier();
        b.wait_lds(0);
        if mode == Mode::ReadsNotCompleted {
            b.barrier();
            b.tile_produce("t", Opnd::I32(0), Opnd::I32(1));
            b.store_lds(a, one, MemTy::B32, false);
            b.tile_end();
            b.wait_lds(0);
        }
    }
    b.else_();
    {
        b.barrier();
        b.tile_consume("t", Opnd::I32(0), Opnd::I32(0));
        let _ = b.load_lds(a, MemTy::B32);
        b.tile_end();
        if mode == Mode::ReadsNotCompleted {
            b.wait_lds_writes();
            b.barrier();
        }
        b.wait_lds(0);
    }
    b.end_if();
    b.finish()
}

#[test]
fn writes_completed_by_wait_lds_writes_are_published() {
    let r = verify_ir(&kernel(Mode::Clean));
    assert!(!has_error(&r), "{}", errors(&r));
}

#[test]
fn a_write_after_wait_lds_writes_is_not_completed_by_it() {
    let r = verify_ir(&kernel(Mode::StoreAfterWait));
    assert!(has_error(&r), "the second store is published incomplete");
}

#[test]
fn wait_lds_writes_does_not_complete_reads() {
    let r = verify_ir(&kernel(Mode::ReadsNotCompleted));
    assert!(has_error(&r), "the consumer's read is still in flight when the producer overwrites the tile");
}

/// gfx1250 code: four stores, six loads, `wait_lds_writes`, barrier -> `s_wait_dscnt 0x6` before the barrier (the loads stay in flight).
#[test]
fn the_count_leaves_the_later_reads_in_flight() {
    let mut b = FnBuilder::new("wlw_count", 64);
    b.lds(4096);
    b.arg("out", amdgpu_dsl::DType::Ptr, 0, 8);
    b.arg_tensor("out", amdgpu_dsl::Ext::c(64 * 4 * 6));
    let po = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);
    let a = b.let_(Op::ShlI32(Opnd::Val(tid), Opnd::I32(2)));
    let one = b.c_i32(1);
    for k in 0..4 {
        let ak = b.let_(Op::AddI32(Opnd::Val(a), Opnd::I32(256 * k)));
        b.store_lds(ak, one, MemTy::B32, false);
    }
    let vals: Vec<_> = (0..6)
        .map(|k| {
            let ak = b.let_(Op::AddI32(Opnd::Val(a), Opnd::I32(2048 + 256 * k)));
            b.load_lds(ak, MemTy::B32)
        })
        .collect();
    b.wait_lds_writes();
    b.barrier();
    for (k, v) in vals.iter().enumerate() {
        let o = b.let_(Op::AddI32(Opnd::Val(a), Opnd::I32(256 * k as i32)));
        let q = b.let_(Op::PtrAdd { base: Opnd::Val(po), byte_off: Opnd::Val(o) });
        b.store_global(q, 0, *v, MemTy::B32, false);
    }
    let c = amdgpu_codegen::compile_target(&b.finish(), "gfx1250").unwrap_or_else(|e| panic!("{e}"));
    let lines: Vec<&str> = c.asm.lines().map(|l| l.trim()).collect();
    let bar = lines.iter().position(|l| l.starts_with("s_barrier")).expect("a barrier");
    let w = lines[..bar].iter().rev().find(|l| l.starts_with("s_wait_dscnt"));
    assert_eq!(w.copied(), Some("s_wait_dscnt 0x6"), "{}", c.asm);
}
