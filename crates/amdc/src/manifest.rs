//! The golden asm manifest: the own-backend asm md5 of every reachable kernel (`amdgpu_kernels::reachable::manifest_names`), the
//! regression oracle of the four-layer refactor (docs/refactor-4layer-plan.md): a refactoring commit must leave it identical; a commit that
//! changes code on purpose regenerates it (`amdc md5-manifest --out tests/golden/md5_manifest.tsv`) and names the gate it passed.
//!
//! Format: one line per kernel, `name<TAB>md5` (or `name<TAB>ERROR` when the kernel does not compile), sorted by name; `#` lines are comments.

use amdgpu_compiler::backend::OwnTail;
use amdgpu_compiler::compile_with;

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
