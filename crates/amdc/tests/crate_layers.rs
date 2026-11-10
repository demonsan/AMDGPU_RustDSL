//! The four-layer rule (docs/architecture.md): every path dependency of a workspace crate points DOWN (to a lower layer or the same layer).
//! Layers (top -> bottom): 0 entry (amdc, kernels), 1 graph, 2 stage, 3 function (tile, layout, isa), 4 backend (dsl, codegen, target), 5 foundation (sym, safeasm).

fn layer(c: &str) -> u32 {
    match c {
        "amdc" | "amdgpu-kernels" => 0,
        "amdgpu-graph" => 1,
        "amdgpu-stage" => 2,
        "amdgpu-tile" | "amdgpu-layout" | "amdgpu-isa" => 3,
        "amdgpu-dsl" | "amdgpu-codegen" | "amdgpu-target" => 4,
        "amdgpu-sym" | "amdgpu-safeasm" => 5,
        other => panic!("crate {other} has no layer: add it to docs/architecture.md and this table"),
    }
}

#[test]
fn every_crate_dependency_points_down() {
    let crates = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut bad = Vec::new();
    let mut n = 0;
    for e in std::fs::read_dir(&crates).unwrap() {
        let dir = e.unwrap().path();
        let toml = dir.join("Cargo.toml");
        let Ok(text) = std::fs::read_to_string(&toml) else { continue };
        let name = dir.file_name().unwrap().to_string_lossy().to_string();
        let me = layer(&name);
        // [dependencies] only (dev-dependencies may reach anywhere for tests)
        let mut in_deps = false;
        for line in text.lines() {
            let l = line.trim();
            if l.starts_with('[') {
                in_deps = l == "[dependencies]";
                continue;
            }
            if !in_deps || !l.contains("path") {
                continue;
            }
            let dep = l.split(|c: char| c == ' ' || c == '=').next().unwrap().trim();
            n += 1;
            if layer(dep) < me {
                bad.push(format!("{name} (layer {me}) depends on {dep} (layer {})", layer(dep)));
            }
        }
    }
    assert!(n > 10, "found only {n} path dependencies");
    assert!(bad.is_empty(), "upward dependencies:\n{}", bad.join("\n"));
}
