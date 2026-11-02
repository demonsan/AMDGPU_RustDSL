//! Independent adversarial cases for the v1.1 tile-lifetime model (kernel-occ audit).
//! These go beyond the shipped positive/negative tests: a version that is not a compile-time
//! function of the loop counters, and a DMA producer published without wait_vm(0).

use amdgpu_compiler::gpu_verify::{verify_ir, Report, Severity};
use amdgpu_dsl::{DType, FnBuilder, Function, Op, Opnd};

fn has(rep: &Report, code: &str) -> bool {
    rep.findings.iter().any(|f| f.code == code && f.severity == Severity::Error)
}
fn errors(rep: &Report) -> String {
    rep.errors().iter().map(|e| format!("[{}] {}", e.code, e.msg)).collect::<Vec<_>>().join("\n")
}

/// A consume version supplied by a kernarg: not derivable from the loop counters.
fn version_from_kernarg() -> Function {
    let mut b = FnBuilder::new("version_from_kernarg", 64);
    b.lds(1024);
    b.arg("seed", DType::I32, 0, 4);
    b.lds_tile("t", 0, 1024);
    let seed = b.let_(Op::KernargI32 { offset: 0 });
    b.tile_consume("t", Opnd::I32(0), seed);
    let _ = b.load_lds_b32(Opnd::I32(0));
    b.tile_end();
    b.finish()
}

/// DMA producer (global -> LDS), optionally completing with wait_vm(0) before the publishing barrier.
fn dma_producer(wait_vm: bool) -> Function {
    let mut b = FnBuilder::new(if wait_vm { "dma_wait" } else { "dma_nowait" }, 128);
    b.lds(4096);
    b.arg_tensor("q", amdgpu_dsl::Ext::c(1024));
    b.arg("q", DType::Ptr, 0, 8);
    b.lds_tile("t", 0, 1024);
    let q = b.let_(Op::KernargPtr { offset: 0 });
    let t = b.let_(Op::WorkitemId);
    let c64 = b.c_i32(64);
    let is_p = b.cmp_lt_i32(t, c64);
    b.begin_if(Opnd::Val(is_p));
    b.tile_produce("t", Opnd::I32(0), Opnd::I32(0));
    b.load_global_lds(q, Opnd::I32(0), 0, 4);
    b.tile_end();
    b.else_();
    b.end_if();
    if wait_vm {
        b.wait_vm(0);
    }
    b.barrier();
    b.begin_if(Opnd::Val(is_p));
    b.else_();
    b.tile_consume("t", Opnd::I32(0), Opnd::I32(0));
    let _ = b.load_lds_b32(Opnd::I32(0));
    b.tile_end();
    b.end_if();
    b.finish()
}

#[test]
fn non_counter_version_is_rejected() {
    let r = verify_ir(&version_from_kernarg());
    println!("version_from_kernarg: {}", r.render());
    assert!(has(&r, "GV-TILE-INDEX"), "{}", errors(&r));
}

#[test]
fn dma_without_wait_vm_is_stale() {
    let r = verify_ir(&dma_producer(false));
    println!("dma_nowait: {}", r.render());
    assert!(has(&r, "GV-TILE-STALE") || has(&r, "GV-TILE-PENDING") || has(&r, "GV-TILE-UNTAGGED"), "{}", errors(&r));
}

#[test]
fn dma_with_wait_vm_is_clean() {
    let r = verify_ir(&dma_producer(true));
    println!("dma_wait: {}", r.render());
    assert!(r.errors().is_empty(), "{}", errors(&r));
}
