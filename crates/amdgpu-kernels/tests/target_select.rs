//! G0 of docs/gfx1250-plan.md: the gfx1250 target is selectable and complete; gfx950 compiles exactly as before.

#[test]
fn the_targets_are_known_by_name() {
    let t = amdgpu_target::by_name("gfx1250").expect("gfx1250");
    assert_eq!(t.name, "gfx1250");
    assert_eq!(t.wave.lanes, 32);
    assert_eq!(t.regs.vgpr.count, 1024);
    assert!(t.regs.agpr.is_none());
    assert_eq!(t.lds.bytes_per_workgroup, 327_680);
    assert_eq!(t.counters.counters.len(), 7);
    assert!(t.llvm.llc_args.contains(&"-mcpu=gfx1250"));
    assert!(amdgpu_target::by_name("gfx9999").is_none());
}

#[test]
fn gfx950_through_compile_target_is_compile() {
    let f = amdgpu_kernels::kernel_by_name("probe_store").expect("probe");
    let a = amdgpu_codegen::compile(&f).unwrap().asm;
    let b = amdgpu_codegen::compile_target(&f, "gfx950").unwrap().asm;
    assert_eq!(a, b);
    assert_eq!(amdgpu_codegen::target::cur().name, "gfx950");
}

#[test]
fn gfx1250_is_selected_and_restored() {
    let f = amdgpu_kernels::kernel_by_name("probe_store").expect("probe");
    // G0: the pipeline is staged (an error naming the stage), and the thread's target is gfx950 again afterwards
    let e = amdgpu_codegen::compile_target(&f, "gfx1250").err();
    assert!(e.map_or(true, |m| m.contains("gfx1250")));
    assert_eq!(amdgpu_codegen::target::cur().name, "gfx950");
    assert!(amdgpu_codegen::compile_target(&f, "nope").is_err());
}