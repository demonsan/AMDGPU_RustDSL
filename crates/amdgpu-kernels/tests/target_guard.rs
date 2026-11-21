//! Audit guard: no shared pass consults the compile-time default target (gfx950, `amdgpu_codegen::target::dflt`) while compiling for
//! gfx1250 - every gfx1250 kernel is compiled with the guard on, which panics at such a read (the bug class of the LDS wait range, the
//! WAW limits and the launch register budget).

use std::sync::Mutex;

#[test]
fn no_gfx1250_compile_consults_the_gfx950_default() {
    let hits: &'static Mutex<Vec<String>> = Box::leak(Box::new(Mutex::new(Vec::new())));
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let loc = info.location().map(|l| format!("{}:{}", l.file(), l.line())).unwrap_or_default();
        hits.lock().unwrap().push(loc);
    }));
    let mut bad: Vec<String> = Vec::new();
    for name in amdgpu_kernels::gfx1250_kernel_names() {
        let Some(f) = amdgpu_kernels::kernel_by_name(name) else { continue };
        hits.lock().unwrap().clear();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| amdgpu_codegen::target::with_default_target_guard(|| amdgpu_codegen::compile_target(&f, "gfx1250"))));
        if r.is_err() {
            let at = hits.lock().unwrap().first().cloned().unwrap_or_default();
            bad.push(format!("{name}: {at}"));
        }
    }
    std::panic::set_hook(prev);
    assert!(bad.is_empty(), "the gfx950 default target was read during gfx1250 compiles:\n{}", bad.join("\n"));
}