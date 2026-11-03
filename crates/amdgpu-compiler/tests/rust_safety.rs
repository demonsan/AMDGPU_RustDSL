//! Rust-safety invariants of the whole workspace.
//!
//! These tests are the machine-checked part of `docs/rust-safety.md`: they assert that every
//! crate forbids `unsafe`, that no forbidden construct appears in any non-comment source line,
//! and that the workspace has no third-party dependencies (no external code can be trusted to
//! uphold our invariants). They also report the panic surface so the document's numbers stay
//! honest.

use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    // CARGO_MANIFEST_DIR = <root>/crates/amdgpu-compiler
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn crates() -> Vec<(&'static str, PathBuf)> {
    let root = workspace_root();
    ["amdgpu-dsl", "amdgpu-compiler", "amdgpu-safeasm", "amdgpu-layout", "amdc"]
        .iter()
        .map(|c| (*c, root.join("crates").join(c)))
        .collect()
}

fn rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            if p.file_name().unwrap() != "target" {
                rs_files(&p, out);
            }
        } else if p.extension().map(|x| x == "rs").unwrap_or(false) {
            out.push(p);
        }
    }
}

/// Every crate must opt in to `#![forbid(unsafe_code)]`; the compiler will then reject any
/// `unsafe` block, `unsafe fn`, `unsafe impl` or `unsafe trait` added later.
#[test]
fn every_crate_forbids_unsafe_code() {
    for (name, dir) in crates() {
        let root = dir.join("src").join("lib.rs");
        let root = if root.exists() { root } else { dir.join("src").join("main.rs") };
        let text = std::fs::read_to_string(&root)
            .unwrap_or_else(|e| panic!("{}: cannot read {}: {e}", name, root.display()));
        assert!(
            text.lines().any(|l| l.trim() == "#![forbid(unsafe_code)]"),
            "{}: crate root {} does not declare #![forbid(unsafe_code)]",
            name,
            root.display()
        );
    }
}

/// No forbidden construct may appear on a non-comment line of any source file.
///
/// The list is deliberately conservative: these are the constructs that would either introduce
/// undefined behaviour directly (`unsafe`, raw pointers, `transmute`, `union`, uninitialised
/// memory) or bypass a checked API (`get_unchecked`). Comments and doc comments are skipped, so
/// prose may mention them (this test file itself does).
#[test]
fn no_forbidden_constructs_in_any_source_line() {
    const FORBIDDEN: &[&str] = &[
        "unsafe",
        "transmute",
        "static mut",
        "as *const",
        "as *mut",
        "from_raw",
        "extern \"C\"",
        "extern \"system\"",
        "mem::uninitialized",
        "mem::zeroed",
        "get_unchecked",
        "unreachable_unchecked",
        "asm!(",
        "global_asm!(",
        "std::ptr",
        "core::ptr",
    ];
    let mut files = Vec::new();
    for (_, dir) in crates() {
        rs_files(&dir.join("src"), &mut files);
    }
    assert!(files.len() > 10, "expected to scan the workspace sources, found {}", files.len());

    for f in files {
        let text = std::fs::read_to_string(&f).unwrap();
        for (n, line) in text.lines().enumerate() {
            let code = line.split("//").next().unwrap_or("").trim();
            if code.is_empty() {
                continue;
            }
            for bad in FORBIDDEN {
                // `unsafe` is allowed only inside the attribute itself, which is stripped above as
                // a comment? No: the attribute is source code. It is the one legitimate mention.
                if *bad == "unsafe" && code.contains("#![forbid(unsafe_code)]") {
                    continue;
                }
                assert!(
                    !code.contains(bad),
                    "{}:{}: forbidden construct `{bad}`: {line}",
                    f.display(),
                    n + 1
                );
            }
        }
    }
}

/// The workspace must stay dependency-free: every `[dependencies]` entry has to be a path
/// dependency on a sibling crate. A crates.io dependency could introduce `unsafe` code that we
/// cannot audit, which would invalidate the guarantee in `docs/rust-safety.md`.
#[test]
fn no_third_party_dependencies() {
    for (name, dir) in crates() {
        let toml = std::fs::read_to_string(dir.join("Cargo.toml")).unwrap();
        let mut in_deps = false;
        for line in toml.lines() {
            let l = line.trim();
            if l.starts_with('[') {
                in_deps = l == "[dependencies]" || l == "[dev-dependencies]" || l == "[build-dependencies]";
                continue;
            }
            if !in_deps || l.is_empty() || l.starts_with('#') {
                continue;
            }
            // A dependency entry looks like `name = ...` or `name = { ... }`.
            if l.contains('=') {
                assert!(
                    l.contains("path ="),
                    "{}: non-path dependency `{l}` (third-party crates are not allowed)",
                    name
                );
            }
        }
    }
}

/// Report (do not assert) the panic surface: `.unwrap()`, `.expect(...)`, `panic!`,
/// `unreachable!`, `todo!` per crate. `docs/rust-safety.md` §5 explains the policy: user-facing
/// failures are `Result`/`ExitCode`; panics mark internal invariant violations only.
#[test]
fn panic_surface_report() {
    let mut total = 0usize;
    for (name, dir) in crates() {
        let mut files = Vec::new();
        rs_files(&dir.join("src"), &mut files);
        let (mut unwrap, mut expect, mut panic) = (0usize, 0usize, 0usize);
        for f in files {
            for line in std::fs::read_to_string(&f).unwrap().lines() {
                let code = line.split("//").next().unwrap_or("");
                unwrap += code.matches(".unwrap()").count();
                expect += code.matches(".expect(").count();
                panic += code.matches("panic!").count();
                panic += code.matches("unreachable!").count();
                panic += code.matches("todo!").count();
                panic += code.matches("unimplemented!").count();
            }
        }
        eprintln!("{name:16} unwrap={unwrap:3} expect={expect:3} panic/unreachable/todo={panic:3}");
        total += unwrap + expect + panic;
    }
    eprintln!("total panic-capable sites in library sources: {total}");
    // No hard bound: the numbers are documentation, and a new panic site in the compiler is not a
    // safety bug (it is a crash, not UB). The test exists so the report cannot silently vanish.
    assert!(total > 0, "the report should have counted something");
}
