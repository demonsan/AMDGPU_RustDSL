//! The soundness net of the register allocator work (docs/ra-subreg.md): every reachable kernel's allocation passes the per-dword /
//! per-16-bit-half clobber check (`exact::check_registers`: no instruction writes a physical register dword whose value is still read later
//! on some path).

#[test]
fn no_reachable_kernel_clobbers_a_live_register() {
    let names: Vec<String> = amdgpu_kernels::reachable::manifest_names().into_iter().collect();
    let jobs = std::thread::available_parallelism().map_or(4, |n| n.get());
    let next = std::sync::atomic::AtomicUsize::new(0);
    let bad = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..jobs {
            s.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(k) = names.get(i) else { break };
                let Some(f) = amdgpu_kernels::kernel_by_name(k) else { continue };
                if f.exact {
                    continue; // exact kernels are checked by exact::finalize
                }
                if let Ok((m, a)) = amdgpu_codegen::compile_mir_alloc(&f) {
                    let v = amdgpu_codegen::exact::check_registers(&m, &a);
                    if !v.is_empty() {
                        bad.lock().unwrap().push(format!("{k}: {} clobber(s), e.g. {}", v.len(), v[0]));
                    }
                }
            });
        }
    });
    let mut bad = bad.into_inner().unwrap();
    bad.sort();
    assert!(bad.is_empty(), "{} kernel(s) with clobbered registers:\n{}", bad.len(), bad.join("\n"));
}
