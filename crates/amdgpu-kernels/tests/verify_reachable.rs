//! Every kernel a launcher can reach (`reachable::manifest_names`: the dispatch rules, the Python ops, the registered names) passes the strict GPU-semantics
//! verification (`amdc gpu-verify --strict`, `gpu_verify::analyse`: IR + MIR) with 0 errors. The only exceptions are the intentional probes and verifier
//! attack kernels (explicit allowlist below). Release only (a few minutes); `python/tests/run_all.sh` runs it. A verifier change that makes a shipped
//! kernel unprovable (e.g. a path cap exceeded) fails here instead of slipping through.

/// Kernels expected to carry verifier errors: the hardware probes and the verifier's own attack cases (their errors are the point).
const ALLOW_PREFIXES: [&str; 2] = ["probe_", "gv_attack_"];
/// Further intentional exceptions by exact name, each with its reason.
const ALLOW_NAMES: [(&str, &str); 1] = [(
    "handoff_probe_cross",
    "hardware probe of convert_layout's cross-lane LDS round trip (tests/gpu/handoff_host.cpp checks it); the interval analysis cannot prove its private-slice bound (GV-TILE-PRIVATE-ESCAPE / GV-LDS-OOB)",
)];

#[test]
#[cfg_attr(debug_assertions, ignore = "release only: cargo test --release -p amdgpu-kernels --test verify_reachable (run by python/tests/run_all.sh)")]
fn every_reachable_kernel_verifies_strict() {
    let mut bad: Vec<String> = Vec::new();
    let mut n = 0usize;
    for name in amdgpu_kernels::reachable::manifest_names() {
        if ALLOW_PREFIXES.iter().any(|p| name.starts_with(p)) || ALLOW_NAMES.iter().any(|(n, _)| *n == name) {
            continue;
        }
        let Some(f) = amdgpu_kernels::kernel_by_name(&name) else {
            bad.push(format!("{name}: not registered"));
            continue;
        };
        n += 1;
        match amdgpu_codegen::gpu_verify::analyse(&f) {
            Ok(r) => {
                let errs = r.errors();
                if !errs.is_empty() {
                    let first: Vec<String> = errs.iter().take(3).map(|e| format!("[{}] {}", e.code, e.msg.chars().take(200).collect::<String>())).collect();
                    bad.push(format!("{name}: {} error(s): {}", errs.len(), first.join(" | ")));
                }
            }
            Err(e) => bad.push(format!("{name}: does not compile for the MIR checks: {e}")),
        }
    }
    eprintln!("{n} reachable kernels verified (strict), {} failing", bad.len());
    assert!(bad.is_empty(), "kernels failing strict gpu-verify:\n{}", bad.join("\n"));
}