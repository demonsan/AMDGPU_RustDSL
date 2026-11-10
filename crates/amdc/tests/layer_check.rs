//! Architecture check (docs/refactor-4layer-plan.md §1): the crate dependency DAG follows the four layers and nothing points upward.
//!
//! ```text
//!   amdc (9)  ->  kernels (8)  ->  graph (7, L1)  ->  stage (6, L2)  ->  tile (5, L3) / isa (4) / layout (3)  ->  codegen (2, L4) / dsl (1) / target, sym, safeasm (0)
//! ```
//!
//! A crate may depend (normal or dev) only on crates with a LOWER rank, and its sources and tests may not name a crate of equal or higher rank. The lower layers know no kernel by name: no
//! `name.contains("literal")` / `starts_with` / `ends_with` on a function name and no kernel-family string literal in `dsl`, `codegen`, `target`, `tile`, `stage`, `isa`, `layout`, `sym`,
//! `safeasm` (a kernel declares its compile policy on the `Function`, docs/pass-catalog.md §8). The `AMDC_SCHED_*` experiment switches pass the pattern from the environment, which is allowed.
//!
//! A new crate must be added to `RANK` (the test fails on an unclassified crate), which is the review point for its layer.
//! No third-party dependency: the manifests are read line by line.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// (crate name, rank, layer label). Lower rank = deeper layer.
const RANK: &[(&str, u32, &str)] = &[
    ("amdgpu-sym", 0, "foundation"),
    ("amdgpu-safeasm", 0, "foundation"),
    ("amdgpu-target", 0, "L4 target data"),
    ("amdgpu-dsl", 1, "L4 IR"),
    ("amdgpu-codegen", 2, "L4 backend + Weaver"),
    ("amdgpu-layout", 3, "L3"),
    ("amdgpu-isa", 4, "L3 atoms"),
    ("amdgpu-tile", 5, "L3"),
    ("amdgpu-stage", 6, "L2"),
    ("amdgpu-graph", 7, "L1"),
    ("amdgpu-kernels", 8, "kernels"),
    ("amdc", 9, "driver"),
];

/// Crates that must not name a kernel family.
const NAME_FREE: &[&str] = &["amdgpu-sym", "amdgpu-safeasm", "amdgpu-target", "amdgpu-dsl", "amdgpu-codegen", "amdgpu-layout", "amdgpu-isa", "amdgpu-tile", "amdgpu-stage"];

/// String-literal prefixes of kernel names (a kernel family named in a lower layer is a policy by name).
const KERNEL_PREFIXES: &[&str] = &[
    "fmha_", "gemm_", "pa_decode", "mla_", "moe_", "silu_", "gelu_", "rmsnorm", "layernorm", "topk_", "sample_", "argmax_", "quant_", "attn_combine", "rope_", "mlp_fp8", "conv1d_", "conv2d_", "swiglu_",
];

fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn rank_of(name: &str) -> Option<u32> {
    RANK.iter().find(|(n, _, _)| *n == name).map(|(_, r, _)| *r)
}

/// crate dir name -> (normal deps, dev deps) on workspace crates, from the manifest.
fn manifest(dir: &Path) -> (Vec<String>, Vec<String>) {
    let text = fs::read_to_string(dir.join("Cargo.toml")).unwrap();
    let (mut sec, mut deps, mut dev) = (String::new(), Vec::new(), Vec::new());
    for l in text.lines() {
        let l = l.trim();
        if l.starts_with('[') {
            sec = l.to_string();
        } else if let Some((k, _)) = l.split_once('=') {
            let k = k.trim();
            if k.starts_with("amdgpu-") {
                match sec.as_str() {
                    "[dependencies]" => deps.push(k.to_string()),
                    "[dev-dependencies]" => dev.push(k.to_string()),
                    _ => {}
                }
            }
        }
    }
    (deps, dev)
}

fn crates() -> BTreeMap<String, PathBuf> {
    let mut m = BTreeMap::new();
    for e in fs::read_dir(workspace().join("crates")).unwrap().flatten() {
        let p = e.path();
        if p.join("Cargo.toml").exists() {
            let text = fs::read_to_string(p.join("Cargo.toml")).unwrap();
            let name = text.lines().find_map(|l| l.trim().strip_prefix("name = ").map(|n| n.trim().trim_matches('"').to_string())).unwrap();
            m.insert(name, p);
        }
    }
    m
}

fn rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    if let Ok(rd) = fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                rs_files(&p, out);
            } else if p.extension().map_or(false, |x| x == "rs") {
                out.push(p);
            }
        }
    }
}

/// The code part of a source line (comments cut off; a `//` inside a string literal ends the scan early, which only under-reports).
fn code(line: &str) -> &str {
    line.split("//").next().unwrap_or("")
}

#[test]
fn every_crate_is_classified() {
    for name in crates().keys() {
        assert!(rank_of(name).is_some(), "crate `{name}` is not in RANK (crates/amdc/tests/layer_check.rs): classify it in the layer table");
    }
}

#[test]
fn manifests_point_downward_only() {
    let mut bad = Vec::new();
    for (name, dir) in crates() {
        let r = rank_of(&name).unwrap();
        let (deps, dev) = manifest(&dir);
        for (kind, list) in [("dependency", deps), ("dev-dependency", dev)] {
            for d in list {
                match rank_of(&d) {
                    None => bad.push(format!("{name}: {kind} `{d}` is not in RANK")),
                    Some(rd) if rd >= r => bad.push(format!("{name} (rank {r}) has an upward / sideways {kind} on `{d}` (rank {rd}): the layers point downward only")),
                    _ => {}
                }
            }
        }
    }
    assert!(bad.is_empty(), "layer rule violated (docs/refactor-4layer-plan.md §1):\n{}", bad.join("\n"));
}

#[test]
fn sources_do_not_name_higher_crates() {
    let mut bad = Vec::new();
    for (name, dir) in crates() {
        let r = rank_of(&name).unwrap();
        let mut files = Vec::new();
        for sub in ["src", "tests", "benches", "examples"] {
            rs_files(&dir.join(sub), &mut files);
        }
        for f in files {
            let text = fs::read_to_string(&f).unwrap();
            for (n, line) in text.lines().enumerate() {
                let c = code(line);
                for (other, ro, _) in RANK {
                    let ident = other.replace('-', "_");
                    if *ro >= r && *other != name && c.contains(&format!("{ident}::")) {
                        bad.push(format!("{}:{}: `{name}` (rank {r}) names `{other}` (rank {ro})", f.strip_prefix(workspace()).unwrap().display(), n + 1));
                    }
                }
            }
        }
    }
    assert!(bad.is_empty(), "a lower layer names a higher crate:\n{}", bad.join("\n"));
}

#[test]
fn lower_layers_have_no_kernel_name_policy() {
    let crates = crates();
    let mut bad = Vec::new();
    for name in NAME_FREE {
        let dir = &crates[*name];
        let mut files = Vec::new();
        rs_files(&dir.join("src"), &mut files);
        for f in files {
            let text = fs::read_to_string(&f).unwrap();
            for (n, line) in text.lines().enumerate() {
                let c = code(line);
                let rel = f.strip_prefix(workspace()).unwrap().display().to_string();
                for pat in [".contains(\"", ".starts_with(\"", ".ends_with(\""] {
                    let mut rest = c;
                    while let Some(i) = rest.find(pat) {
                        let before = rest[..i].trim_end();
                        let ident: String = before.chars().rev().take_while(|ch| ch.is_alphanumeric() || *ch == '_' || *ch == '.').collect::<String>().chars().rev().collect();
                        if ident.ends_with(".name") || ident == "pname" || ident == "fname" || ident == "kernel_name" || ident == "policy_name" {
                            bad.push(format!("{rel}:{}: policy by function name (`{ident}{pat}…`)", n + 1));
                        }
                        rest = &rest[i + pat.len()..];
                    }
                }
                for q in c.split('"').skip(1).step_by(2) {
                    if KERNEL_PREFIXES.iter().any(|p| q.starts_with(p)) {
                        bad.push(format!("{rel}:{}: kernel-family literal \"{q}\" in a lower layer", n + 1));
                    }
                }
            }
        }
    }
    assert!(bad.is_empty(), "the lower layers must not know kernels by name (kernels declare their policy on the Function):\n{}", bad.join("\n"));
}
