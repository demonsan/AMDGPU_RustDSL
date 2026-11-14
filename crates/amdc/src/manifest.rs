//! The golden asm manifest: the own-backend asm md5 of every reachable kernel (`amdgpu_kernels::reachable::manifest_names`), the
//! regression oracle of the four-layer refactor (docs/refactor-4layer-plan.md): a refactoring commit must leave it identical; a commit that
//! changes code on purpose regenerates it (`amdc md5-manifest --out tests/golden/md5_manifest.tsv`) and names the gate it passed.
//!
//! Format: one line per kernel, `name<TAB>md5` (or `name<TAB>ERROR` when the kernel does not compile), sorted by name; `#` lines are comments.

use amdgpu_codegen::backend::OwnTail;
use amdgpu_codegen::compile_with;

/// The `AMDC_*` environment variables that are set (non-empty): they change the compiler's output (scheduler, lowering options, backend
/// switches), so the manifest is computed and checked only in a scrubbed environment.
/// Variables that do not affect the generated code (build provenance).
const NEUTRAL: &[&str] = &["AMDC_GIT_DESCRIBE"];

pub fn env_overrides() -> Vec<String> {
    let mut v: Vec<String> = std::env::vars().filter(|(k, val)| k.starts_with("AMDC_") && !val.is_empty() && !NEUTRAL.contains(&k.as_str())).map(|(k, val)| format!("{k}={val}")).collect();
    v.sort();
    v
}

/// (name, md5 or "ERROR" / "MISSING") for every manifest kernel, compiled on `jobs` threads; refused when an `AMDC_*` variable is set.
pub fn compute(jobs: usize) -> Result<Vec<(String, String)>, String> {
    let ov = env_overrides();
    if !ov.is_empty() {
        return Err(format!("the manifest needs a scrubbed environment; unset: {}", ov.join(" ")));
    }
    Ok(compute_unchecked(jobs))
}

fn compute_unchecked(jobs: usize) -> Vec<(String, String)> {
    let names: Vec<String> = amdgpu_kernels::reachable::manifest_names().into_iter().collect();
    let jobs = jobs.max(1);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let out = std::sync::Mutex::new(Vec::with_capacity(names.len()));
    std::thread::scope(|s| {
        for _ in 0..jobs {
            s.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(k) = names.get(i) else { break };
                let v = match amdgpu_kernels::kernel_by_name(k) {
                    None => "MISSING".to_string(),
                    Some(f) => match compile_with(&f, &OwnTail) {
                        Ok(c) => crate::md5::hex(c.asm.as_bytes()),
                        Err(_) => "ERROR".to_string(),
                    },
                };
                out.lock().unwrap().push((k.clone(), v));
            });
        }
    });
    let mut v = out.into_inner().unwrap();
    v.sort();
    v
}

/// (name, md5 of the in-process code object or "ERROR" / "MISSING") for every manifest kernel: the clang-free regression oracle of the assembler (`tests/golden/co_md5_manifest.tsv`).
pub fn compute_co(jobs: usize) -> Result<Vec<(String, String)>, String> {
    let ov = env_overrides();
    if !ov.is_empty() {
        return Err(format!("the manifest needs a scrubbed environment; unset: {}", ov.join(" ")));
    }
    let names: Vec<String> = amdgpu_kernels::reachable::manifest_names().into_iter().collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let out = std::sync::Mutex::new(Vec::with_capacity(names.len()));
    std::thread::scope(|s| {
        for _ in 0..jobs.max(1) {
            s.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(k) = names.get(i) else { break };
                let v = match amdgpu_kernels::kernel_by_name(k) {
                    None => "MISSING".to_string(),
                    Some(f) => match compile_with(&f, &OwnTail).and_then(|c| amdgpu_target::gfx950_co::assemble(&c.asm)) {
                        Ok(co) => crate::md5::hex(&co),
                        Err(_) => "ERROR".to_string(),
                    },
                };
                out.lock().unwrap().push((k.clone(), v));
            });
        }
    });
    let mut v = out.into_inner().unwrap();
    v.sort();
    Ok(v)
}

/// The specialized shapes whose certificates the certificate manifest covers besides the generic IR of every kernel.
pub const CERT_SHAPES: &[(&str, &str)] = &[
    ("silu_mul_bf16_d4096", "m=7"),
    ("silu_mul_bf16_d4096", "m=64"),
    ("rmsnorm_quant_fp8_n4096", "m=64"),
    ("rmsnorm_quant_fp8_n4096", "m=1024"),
    ("gemm_a8w8_rowcol_fp8_64", "m=64"),
    ("gemm_a8w8_rowcol_fp8_64", "m=1152"),
    ("pa_decode16_bf16_hd128_p16_f", "H=8"),
    ("pa_decode16_bf16_hd128_p16_f", "H=64"),
    ("pa_decode16_bf16_hd128_p16_f", "H=128"),
];

/// (name, md5 of the IR-level certificate: every finding rendered plus the trust ledger) for the generic IR of every manifest kernel and for `CERT_SHAPES`: the oracle of the `verify_ir` optimizations
/// (`tests/golden/cert_md5_manifest.tsv`): same findings, same certificates.
pub fn compute_cert(jobs: usize) -> Result<Vec<(String, String)>, String> {
    let ov = env_overrides();
    if !ov.is_empty() {
        return Err(format!("the manifest needs a scrubbed environment; unset: {}", ov.join(" ")));
    }
    let mut items: Vec<(String, String, String)> = amdgpu_kernels::reachable::manifest_names().into_iter().map(|k| (k.clone(), k, String::new())).collect();
    for (k, s) in CERT_SHAPES {
        items.push((format!("{k}@{s}"), k.to_string(), s.to_string()));
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let out = std::sync::Mutex::new(Vec::with_capacity(items.len()));
    std::thread::scope(|sc| {
        for _ in 0..jobs.max(1) {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some((name, k, s)) = items.get(i) else { break };
                let spec = if s.is_empty() { Ok(Vec::new()) } else { amdgpu_dsl::specialize::parse_bindings(s) };
                let v = match spec.and_then(|sp| crate::jitc::load(k, &sp, None)) {
                    Err(_) => "MISSING".to_string(),
                    Ok((f, _)) => {
                        let r = amdgpu_codegen::gpu_verify::verify_ir(&f);
                        let l = r.ledger();
                        md5::hex(format!("{}\n{} {} {} {} {}\n{:?}", r.render(), l.assume_trusted, l.tile_trust, l.region_trust, l.asm_opaque, l.timing_trust, r.proven).as_bytes())
                    }
                };
                out.lock().unwrap().push((name.clone(), v));
            });
        }
    });
    let mut v = out.into_inner().unwrap();
    v.sort();
    Ok(v)
}

/// The manifest text.
pub fn render(entries: &[(String, String)]) -> String {
    let mut s = String::from("# golden asm md5 manifest (own backend): amdc md5-manifest --out tests/golden/md5_manifest.tsv; see tests/golden/README.md\n");
    for (k, v) in entries {
        s.push_str(&format!("{k}\t{v}\n"));
    }
    s
}

/// Parses a manifest text.
pub fn parse(text: &str) -> Vec<(String, String)> {
    text.lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#')).filter_map(|l| l.split_once('\t').map(|(a, b)| (a.trim().to_string(), b.trim().to_string()))).collect()
}

/// The differences `golden` → `now`: changed, added and removed kernels (empty: identical).
pub fn diff(golden: &[(String, String)], now: &[(String, String)]) -> Vec<String> {
    use std::collections::BTreeMap;
    let g: BTreeMap<&str, &str> = golden.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    let n: BTreeMap<&str, &str> = now.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    let mut d = Vec::new();
    for (k, v) in &n {
        match g.get(k) {
            Some(w) if w == v => {}
            Some(w) => d.push(format!("changed  {k}: {w} -> {v}")),
            None => d.push(format!("added    {k}: {v}")),
        }
    }
    for (k, w) in &g {
        if !n.contains_key(k) {
            d.push(format!("removed  {k}: {w}"));
        }
    }
    d
}
