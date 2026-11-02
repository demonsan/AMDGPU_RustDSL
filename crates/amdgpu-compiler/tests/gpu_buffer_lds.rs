//! `buffer_load … lds` (MUBUF global→LDS DMA) through `gpu_verify` and the protocol engine.
//!
//! The DMA has two sides that are checked independently:
//!   * global side  - the buffer descriptor must not overstate its allocation (`GV-BUF-OVER`), because the hardware
//!     range check only protects within `num_records`;
//!   * LDS side     - the destination is `M0 + imm + lane*size` for **all 64 lanes**, which must stay inside the
//!     declared LDS segment (`GV-LDS-OOB`) and inside its tile (`GV-TILE-ESCAPE`).
//! Lifetime: a DMA write is a *produce* event whose publication needs `wait_vm(0)` completion before the barrier -
//! a barrier alone does not publish an in-flight DMA (`GV-TILE-STALE` / `GV-PROTO-NO-COMPLETION`).

use amdgpu_compiler::gpu_verify::{verify_ir, Report, Severity};
use amdgpu_dsl::{DType, Ext, FnBuilder, Function, Op, Opnd};

fn has(rep: &Report, code: &str) -> bool {
    rep.findings.iter().any(|f| f.code == code && f.severity == Severity::Error)
}

fn errors(rep: &Report) -> String {
    rep.errors().iter().map(|e| format!("[{}] {}", e.code, e.msg)).collect::<Vec<_>>().join("\n")
}

fn lane(b: &mut FnBuilder) -> amdgpu_dsl::ValueId {
    b.let_(Op::WorkitemId)
}

/// `buffer_load_dwordx4 … lds` with `64 lanes × 16 B` written at `lds_base + lane*16`.
/// `desc_bytes`: descriptor `num_records`; `extent`: declared allocation; `lds_bytes`: declared LDS segment.
fn dma_kernel(desc_bytes: u32, extent: Option<Ext>, lds_bytes: u32) -> Function {
    let mut b = FnBuilder::new("buf_dma", 64);
    b.arg("p", DType::Ptr, 0, 8);
    b.lds(lds_bytes);
    match extent {
        Some(e) => b.arg_tensor("p", e),
        None => {}
    }
    let p = b.let_(Op::KernargPtr { offset: 0 });
    let rsrc = b.buffer_desc(p, desc_bytes);
    // M0 is the LDS base only: the hardware adds `lane*16` itself, so the checked extent is [,1024) for 64 lanes.
    let lds_base = b.c_i32(0);
    let voff = b.c_i32(0);
    let soff = b.c_i32(0);
    b.buffer_load_lds_b128(rsrc, voff, soff, lds_base, 0);
    b.finish()
}

#[test]
fn a_buffer_lds_dma_with_a_consistent_descriptor_and_lds_segment_is_proven() {
    let r = verify_ir(&dma_kernel(64 * 16, Some(Ext::c(64 * 16)), 64 * 16));
    assert!(r.errors().is_empty(), "{}", errors(&r));
    assert!(r.proven.get("lds").copied().unwrap_or(0) >= 1, "the LDS side must count as a proven access: {:?}", r.proven);
    assert!(r.proven.get("global").copied().unwrap_or(0) + r.proven.get("buffer").copied().unwrap_or(0) >= 1);
}

#[test]
fn a_buffer_lds_dma_whose_descriptor_overstates_the_allocation_is_rejected() {
    // 64 B allocated but the descriptor says 4096: the hardware range check would silently read other tensors.
    let r = verify_ir(&dma_kernel(4096, Some(Ext::c(64)), 64 * 16));
    assert!(has(&r, "GV-BUF-OVER"), "{}", r.render());
}

#[test]
fn a_buffer_lds_dma_without_a_declared_extent_is_rejected() {
    let r = verify_ir(&dma_kernel(64 * 16, None, 64 * 16));
    assert!(has(&r, "GV-BUF-OVER") || has(&r, "GV-GLB-NOEXT"), "{}", r.render());
}

#[test]
fn a_buffer_lds_dma_that_overflows_the_lds_segment_is_rejected() {
    // 64 lanes x 16 B = 1024 B, but only 512 B are declared.
    let r = verify_ir(&dma_kernel(64 * 16, Some(Ext::c(64 * 16)), 512));
    assert!(has(&r, "GV-LDS-OOB"), "{}", r.render());
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// Protocol: a DMA produce needs wait_vm(0) completion before the barrier can publish it

fn dma_protocol_kernel(wait_before_barrier: bool) -> Function {
    let mut b = FnBuilder::new("buf_dma_proto", 64);
    b.arg("p", DType::Ptr, 0, 8);
    b.lds(1024);
    b.lds_tile("t", 0, 1024);
    b.arg_tensor("p", Ext::c(1024));
    let p = b.let_(Op::KernargPtr { offset: 0 });
    let rsrc = b.buffer_desc(p, 1024);
    let lds_base = b.c_i32(0); // M0 base; lane*16 is added by the hardware
    let voff = b.c_i32(0);
    let soff = b.c_i32(0);
    b.tile_produce("t", Opnd::I32(0), Opnd::I32(0));
    b.buffer_load_lds_b128(rsrc, voff, soff, lds_base, 0);
    b.tile_end();
    if wait_before_barrier {
        b.wait_vm(0);
    }
    b.barrier();
    b.tile_consume("t", Opnd::I32(0), Opnd::I32(0));
    let v = b.load_lds_b128(lds_base);
    b.tile_end();
    b.wait_lds(0);
    let _ = v;
    b.finish()
}

#[test]
fn a_dma_produce_published_after_wait_vm_is_proven() {
    let r = verify_ir(&dma_protocol_kernel(true));
    assert!(r.errors().is_empty(), "{}", errors(&r));
}

#[test]
fn a_dma_produce_that_is_not_waited_never_publishes() {
    // The barrier only orders execution; the DMA is still in flight, so the consumer would read stale LDS.
    let r = verify_ir(&dma_protocol_kernel(false));
    assert!(
        has(&r, "GV-TILE-STALE") || has(&r, "GV-TILE-NO-COMPLETION") || has(&r, "GV-TILE-UNPUBLISHED"),
        "{}",
        r.render()
    );
}
