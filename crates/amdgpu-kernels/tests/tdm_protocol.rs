//! G3 (docs/gfx1250-plan.md): the protocol verifier's tensor axis. A tensor load is an LDS write completed by `s_wait_tensorcnt` (in order
//! among the wave's tensor operations only), published by a barrier; the slot it overwrites must have been released by every reader.

use amdgpu_kernels::probes::g1250::tdm_ring_with;

fn errors(f: &amdgpu_dsl::Function) -> Vec<String> {
    amdgpu_codegen::gpu_verify::verify_ir_protocol(f).errors().iter().map(|e| format!("[{}] {}", e.code, e.msg)).collect()
}

#[test]
fn the_tdm_ring_verifies() {
    let e = errors(&tdm_ring_with(true, true));
    assert!(e.is_empty(), "{e:#?}");
    amdgpu_codegen::compile_target(&tdm_ring_with(true, true), "gfx1250").unwrap();
}

#[test]
fn a_missing_tensor_wait_is_found() {
    let e = errors(&tdm_ring_with(false, true));
    assert!(!e.is_empty(), "the reads consume tiles whose tensor loads were never completed");
    assert!(amdgpu_codegen::compile_target(&tdm_ring_with(false, true), "gfx1250").is_err());
}

#[test]
fn a_slot_overwritten_before_every_reader_released_it_is_found() {
    let e = errors(&tdm_ring_with(true, false));
    assert!(!e.is_empty(), "the tensor load overwrites a slot other waves may still read");
}
