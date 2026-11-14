//! The in-process assembler (`amdgpu_target::gfx950_co`): the code object of every manifest kernel against (1) the golden md5 manifest (clang-free,
//! `tests/golden/co_md5_manifest.tsv`; regenerate with `amdc md5-manifest --co --out ...` for an intended change) and (2) the external clang + lld, byte for byte, when the toolchain is found
//! (`AMD_LLVM_BIN` or `clang` on PATH; skipped otherwise): the oracle the encoder was written against.

use amdgpu_codegen::backend::OwnTail;

#[test]
fn the_code_object_of_every_reachable_kernel_matches_the_golden_co_manifest() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/co_md5_manifest.tsv");
    let golden = amdc::manifest::parse(&std::fs::read_to_string(&path).expect("tests/golden/co_md5_manifest.tsv"));
    let jobs = std::thread::available_parallelism().map_or(4, |n| n.get());
    let now = amdc::manifest::compute_co(jobs).expect("scrubbed environment");
    let d = amdc::manifest::diff(&golden, &now);
    assert!(d.is_empty(), "{} of {} code objects differ from the golden co manifest:\n{}", d.len(), now.len(), d.join("\n"));
}

fn toolchain() -> bool {
    std::env::var("AMD_LLVM_BIN").ok().filter(|v| !v.is_empty()).map(|d| std::path::Path::new(&d).join("clang").exists()).unwrap_or_else(|| std::process::Command::new("clang").arg("--version").output().is_ok())
}

#[test]
fn the_in_process_code_object_is_byte_identical_to_clangs_for_every_reachable_kernel() {
    if !toolchain() {
        eprintln!("skipped: no clang toolchain (set AMD_LLVM_BIN)");
        return;
    }
    let names: Vec<String> = amdgpu_kernels::reachable::manifest_names().into_iter().collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let bad = std::sync::Mutex::new(Vec::new());
    let jobs = std::thread::available_parallelism().map_or(4, |n| n.get()).min(16);
    std::thread::scope(|s| {
        for _ in 0..jobs {
            s.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(k) = names.get(i) else { break };
                let Some(f) = amdgpu_kernels::kernel_by_name(k) else { continue };
                let Ok(c) = amdgpu_codegen::compile_with(&f, &OwnTail) else { continue };
                let ours = amdgpu_target::gfx950_co::assemble(&c.asm);
                let theirs = amdc::batch::clang_co(&c.asm, &format!("t{i}"));
                match (ours, theirs) {
                    (Ok(a), Ok(b)) if a == b => {}
                    (a, b) => bad.lock().unwrap().push(format!("{k}: ours {:?} clang {:?}", a.map(|x| x.len()), b.map(|x| x.len()))),
                }
            });
        }
    });
    let bad = bad.into_inner().unwrap();
    assert!(bad.is_empty(), "{} kernels differ from clang:\n{}", bad.len(), bad.join("\n"));
}
