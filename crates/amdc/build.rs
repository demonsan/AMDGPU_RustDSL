//! Embeds `git describe --always --dirty` (or "unknown") as AMDC_GIT_DESCRIBE for `amdc jit-info`'s compiler identity.
fn main() {
    let d = std::process::Command::new("git")
        .args(["describe", "--always", "--dirty"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=AMDC_GIT_DESCRIBE={d}");
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../.git/index");
}