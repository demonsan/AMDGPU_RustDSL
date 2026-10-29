//! Compile-fail suite: every file in `tests/ui/fail_*.rs` must FAIL to compile with a diagnostic containing the text of its first line
//! (`// error-contains: ...`); every `tests/ui/pass_*.rs` must compile. Uses the same rustc and the already built crates (no extra
//! dependency: trybuild cannot be vendored with this toolchain's cargo).

use std::path::{Path, PathBuf};
use std::process::Command;

fn deps_dir() -> PathBuf {
    std::env::current_exe().unwrap().parent().unwrap().to_path_buf()
}

fn newest(deps: &Path, prefix: &str) -> PathBuf {
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    for e in std::fs::read_dir(deps).unwrap() {
        let e = e.unwrap();
        let n = e.file_name().to_string_lossy().to_string();
        if n.starts_with(prefix) && n.ends_with(".rlib") {
            let t = e.metadata().unwrap().modified().unwrap();
            if best.as_ref().map_or(true, |b| t > b.0) {
                best = Some((t, e.path()));
            }
        }
    }
    best.unwrap_or_else(|| panic!("{prefix}*.rlib not found in {}", deps.display())).1
}

fn compile(file: &Path) -> (bool, String) {
    let deps = deps_dir();
    let rlib = newest(&deps, "libamdgpu_safeasm-");
    let out = std::env::temp_dir().join(format!("safeasm-ui-{}-{}", std::process::id(), file.file_stem().unwrap().to_string_lossy()));
    let o = Command::new("rustc")
        .args(["--edition", "2021", "--crate-type", "bin", "--crate-name", "uitest", "-L"])
        .arg(format!("dependency={}", deps.display()))
        .arg("--extern")
        .arg(format!("amdgpu_safeasm={}", rlib.display()))
        .arg("-o")
        .arg(&out)
        .arg(file)
        .output()
        .expect("rustc not runnable");
    let _ = std::fs::remove_file(&out);
    (o.status.success(), String::from_utf8_lossy(&o.stderr).to_string())
}

#[test]
fn ui_cases() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/ui");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().path()).filter(|p| p.extension().map_or(false, |x| x == "rs")).collect();
    files.sort();
    assert!(files.iter().any(|f| f.file_name().unwrap().to_string_lossy().starts_with("fail_")), "no fail_ cases found");
    let mut bad = Vec::new();
    for f in &files {
        let name = f.file_name().unwrap().to_string_lossy().to_string();
        let src = std::fs::read_to_string(f).unwrap();
        let (ok, err) = compile(f);
        if name.starts_with("pass_") {
            if !ok {
                bad.push(format!("{name}: expected to compile:\n{err}"));
            }
        } else if name.starts_with("fail_") {
            let want = src.lines().next().and_then(|l| l.strip_prefix("// error-contains:")).map(|s| s.trim().to_string()).expect("first line must be `// error-contains: ...`");
            if ok {
                bad.push(format!("{name}: compiled, but must be rejected"));
            } else if !err.contains(&want) {
                bad.push(format!("{name}: rejected, but the diagnostic does not contain `{want}`:\n{err}"));
            }
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n\n"));
}