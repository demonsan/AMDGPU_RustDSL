//! The ladder hints (`amdgpu_kernels::ladder_hints`) are exactly the first rung each hinted kernel compiles at without a hint: a stale hint would skip a rung that now succeeds and change the output.
//! Regenerate with `bash tools/regen_ladder_hints.sh` (amdc ladder-hints).

use amdgpu_kernels::{ladder_hints, registry};

#[test]
fn every_hint_is_the_unhinted_first_success_rung() {
    for (name, o, i) in ladder_hints::HINTS {
        let mut f = registry::kernel_by_name_raw(name).unwrap_or_else(|| panic!("hinted kernel `{name}` is not registered"));
        f.ladder_hint = None;
        amdgpu_codegen::compile(&f).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(amdgpu_codegen::last_rung(), (*o, *i), "{name}: the stale hint would change the output: regenerate with tools/regen_ladder_hints.sh");
    }
}

#[test]
fn a_hinted_compile_has_the_asm_of_the_unhinted_one() {
    // the first hinted kernel (if any): same assembly with and without the hint
    if let Some((name, _, _)) = ladder_hints::HINTS.first() {
        let mut a = registry::kernel_by_name_raw(name).unwrap();
        a.ladder_hint = None;
        let b = registry::kernel_by_name(name).unwrap();
        assert!(b.ladder_hint.is_some());
        assert_eq!(amdgpu_codegen::compile(&a).unwrap().asm, amdgpu_codegen::compile(&b).unwrap().asm, "{name}");
    }
}
