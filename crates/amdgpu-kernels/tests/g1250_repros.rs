//! Reproductions of open back-end limitations (run with `cargo test --release -p amdgpu-kernels --test g1250_repros -- --ignored`). Each test documents the failure by asserting it, so a fix shows up as a
//! failing test to delete.

#[test]
#[ignore = "documents a back-end limitation: the 256 x 128 MXFP4 tile does not compile yet"]
fn the_256x128_mxfp4_tile_needs_the_scale_operand_in_the_low_vgpr_window() {
    let f = amdgpu_kernels::probes::g1250_gemm::gemm_tdm_stage_f4_256x128_repro();
    let e = amdgpu_codegen::compile_target(&f, "gfx1250").err().expect("the repro compiles now: delete this test and register the kernel");
    let e = format!("{e}");
    assert!(e.contains("must be in v0..v255"), "{e}");
}
