//! Embeds the source revision as AMDC_GIT_DESCRIBE for `amdc jit-info`'s compiler identity: `git describe --always --dirty` when git is
//! available, else the commit read from `.git/HEAD` (refs or packed-refs), else "unknown".
use std::path::Path;

fn head_commit(git: &Path) -> Option<String> {
    let head = std::fs::read_to_string(git.join("HEAD")).ok()?;
    let head = head.trim();
    let Some(r) = head.strip_prefix("ref: ") else { return Some(head.chars().take(12).collect()) };
    if let Ok(h) = std::fs::read_to_string(git.join(r)) {
        return Some(h.trim().chars().take(12).collect());
    }
    let packed = std::fs::read_to_string(git.join("packed-refs")).ok()?;
    packed.lines().find(|l| l.ends_with(r)).map(|l| l.chars().take(12).collect())
}

fn main() {
    let d = std::process::Command::new("git")
        .args(["describe", "--always", "--dirty"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| head_commit(Path::new("../../.git")))
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=AMDC_GIT_DESCRIBE={d}");
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../.git/index");
}