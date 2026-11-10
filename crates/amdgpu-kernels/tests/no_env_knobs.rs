//! Repo policy: kernels do not read the environment. A non-default behaviour is a fixed choice in the kernel (and an archive tag for a removed experiment), never an env switch that can reach Main
//! (a TEMP_BWD_POL / TEMP_BWD_TOK knob did once). This test scans `amdgpu-kernels/src` and `amdgpu-tile/src` for `env::var` / `env::var_os` / `env::vars` / `option_env!` and fails on any use that is not in
//! the allowlist below, which lists the documented opt-in toolboxes only:
//!   (`weave_auto.rs`, the opt-in auto-weave tool with its AMDC_WEAVE_AUTO_* variables, moved to `amdgpu-codegen` (L4) with the Weaver and is outside the scanned crates; it never changes a shipped kernel)
//!   * `gemm/body.rs`: `GEMM_PROOF_DROP_KB` (proof negative test) and `gemm/mod.rs`: `GEMM_STAMP` (timestamp instrumentation), both off by default;
//!   * `probes/mod.rs`: the hardware probe kernels (`PD_*`, `SCHED_WG`), not shipped kernels.
//! Adding an entry here needs a reason in the commit message; the registry and every shipped kernel module stay env-free.
use std::fs;
use std::path::{Path, PathBuf};

/// (path suffix, allowed variable-name prefixes; "*" = any, including a dynamic name)
const ALLOW: &[(&str, &[&str])] = &[
    ("amdgpu-kernels/src/gemm/mod.rs", &["GEMM_STAMP"]),
    ("amdgpu-kernels/src/gemm/body.rs", &["GEMM_PROOF_DROP_KB"]),
    ("amdgpu-kernels/src/probes/mod.rs", &["PD_", "SCHED_WG"]),
];

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in fs::read_dir(dir).unwrap().flatten() {
        let p = e.path();
        if p.is_dir() {
            walk(&p, out);
        } else if p.extension().map_or(false, |x| x == "rs") {
            out.push(p);
        }
    }
}

/// the variable names read by every env access of a line (a literal, or "<dynamic>")
fn env_names(line: &str) -> Vec<String> {
    let mut names = Vec::new();
    for pat in ["env::var_os(", "env::vars(", "env::var(", "option_env!("] {
        let mut rest = line;
        while let Some(i) = rest.find(pat) {
            let after = rest[i + pat.len()..].trim_start();
            if let Some(lit) = after.strip_prefix('"') {
                names.push(lit.split('"').next().unwrap_or("").to_string());
            } else {
                names.push("<dynamic>".to_string());
            }
            rest = &rest[i + pat.len()..];
        }
    }
    names
}

#[test]
fn kernels_and_tile_crates_read_no_undocumented_env_vars() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    walk(&root.join("src"), &mut files);
    walk(&root.join("../amdgpu-tile/src"), &mut files);
    let mut bad = Vec::new();
    for f in &files {
        let rel = f.canonicalize().unwrap().to_string_lossy().replace('\\', "/");
        let src = fs::read_to_string(f).unwrap();
        for (n, line) in src.lines().enumerate() {
            let code = line.split("//").next().unwrap_or("");
            for name in env_names(code) {
                let ok = ALLOW.iter().any(|(suffix, prefixes)| rel.ends_with(suffix) && prefixes.iter().any(|p| *p == "*" || name.starts_with(p)));
                if !ok {
                    bad.push(format!("{}:{}: reads env var {name}", rel.rsplit("crates/").next().unwrap_or(&rel), n + 1));
                }
            }
        }
    }
    assert!(
        bad.is_empty(),
        "kernels must not read the environment (repo policy: fixed choices, archive tags for removed experiments); allowlist: crates/amdgpu-kernels/tests/no_env_knobs.rs\n{}",
        bad.join("\n")
    );
}

#[test]
fn the_scanner_sees_the_known_toolboxes() {
    // guards the test itself: the allowlisted files do contain env reads (if a refactor removes them, drop the allowlist entry)
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    for (suffix, _) in ALLOW {
        let p = if suffix.starts_with("amdgpu-tile/") { root.join("..").join(suffix) } else { root.join(suffix.trim_start_matches("amdgpu-kernels/")) };
        let src = fs::read_to_string(&p).unwrap_or_else(|_| panic!("allowlisted file {} missing", p.display()));
        assert!(src.lines().any(|l| !env_names(l.split("//").next().unwrap_or("")).is_empty()), "{suffix}: no env read any more; remove the allowlist entry");
    }
    assert_eq!(env_names(r#"let x = std::env::var("FOO").ok();"#), ["FOO"]);
    assert_eq!(env_names("std::env::var(name)"), ["<dynamic>"]);
}