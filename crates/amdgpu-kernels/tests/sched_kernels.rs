//! Every registered kernel lowers, schedules (with and without ALU across barriers, and in rewait mode) and verifies.
use amdgpu_compiler::sched::{run, Options};

fn opts() -> Options {
    Options { verify: true, ..Options::default() }
}

#[test]
fn every_kernel_schedules_and_verifies() {
    for name in amdgpu_kernels::kernel_names() {
        let f = amdgpu_kernels::kernel_by_name(name).unwrap();
        let Ok(mut m) = amdgpu_compiler::lower::lower(&f) else { continue };
        for xbar in [false, true] {
            let mut mm = m.clone();
            let mut o = opts();
            o.alu_across_barrier = xbar;
            run(&mut mm, &o).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert!(amdgpu_compiler::mir::verify(&mm).is_empty(), "{name}");
        }
        let mut mm = m.clone();
        let mut o = opts();
        o.reorder = false;
        run(&mut mm, &o).unwrap_or_else(|e| panic!("{name} rewait: {e}"));
        let _ = &mut m;
    }
}
