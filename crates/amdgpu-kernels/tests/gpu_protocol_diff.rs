//! `gpu_protocol::check_pair` has an indexed fast path (the reuse scan only when a reuse is declared, the write-write pairs per (tile, slot) group); the straightforward quadratic scan is kept as the reference
//! (`with_reference_pair_checks`). Both must report the same findings in the same order for every reachable kernel (identical verdicts).

use amdgpu_codegen::gpu_protocol::with_reference_pair_checks;
use amdgpu_codegen::gpu_verify::verify_ir_protocol;

#[test]
fn the_indexed_pair_checks_report_exactly_what_the_reference_scan_reports() {
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
                if f.contract.tiles.is_empty() {
                    continue;
                }
                let fast = verify_ir_protocol(&f).render();
                let slow = with_reference_pair_checks(|| verify_ir_protocol(&f).render());
                if fast != slow {
                    bad.lock().unwrap().push(k.clone());
                }
            });
        }
    });
    let bad = bad.into_inner().unwrap();
    assert!(bad.is_empty(), "the indexed pair checks differ from the reference scan on: {bad:?}");
}