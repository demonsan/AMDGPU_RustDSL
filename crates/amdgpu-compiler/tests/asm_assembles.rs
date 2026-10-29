//! Every emitted kernel must be accepted by the real assembler (catches operand-legality bugs such as the constant-bus
//! violation of an SGPR source in `v_cndmask_b32_e32` that no unit-level parser noticed).

use amdgpu_compiler::backend::LlvmTools;
use amdgpu_compiler::lower::with_opts;
use amdgpu_compiler::{compile, kernel_by_name, kernel_names};

fn assemble(clang: &std::path::Path, asm: &str, tag: &str) -> Result<(), String> {
    let dir = std::env::temp_dir().join(format!("amdc-asm-{}-{}", std::process::id(), tag));
    std::fs::create_dir_all(&dir).unwrap();
    let s = dir.join("k.s");
    std::fs::write(&s, asm).unwrap();
    let out = std::process::Command::new(clang)
        .args(["-x", "assembler", "-target", "amdgcn-amd-amdhsa", "-mcpu=gfx950", "-mcode-object-version=6", "-c", "-o"])
        .arg(dir.join("k.o"))
        .arg(&s)
        .output()
        .map_err(|e| e.to_string())?;
    let _ = std::fs::remove_dir_all(&dir);
    if out.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).lines().filter(|l| l.contains("error")).take(3).collect::<Vec<_>>().join(" | "))
    }
}

#[test]
fn emitted_kernels_assemble_with_the_aggressive_option_sets() {
    let t = LlvmTools::locate();
    let clang = t.tool("clang");
    if !std::process::Command::new(&clang).arg("--version").output().map(|o| o.status.success()).unwrap_or(false) {
        eprintln!("SKIPPED: clang not runnable");
        return;
    }
    let mut bad = Vec::new();
    let mut names: Vec<&str> = kernel_names().to_vec();
    names.extend(["fmha_fwd_hd128_mxfp8_vs", "fmha_fwd_hd128_mxfp8_vs128"]);
    for k in names {
        if !(k.starts_with("fmha_fwd") || k == "attention_fwd") {
            continue;
        }
        let Some(f) = kernel_by_name(k) else { continue };
        for (i, opts) in [&["dce", "cmpfold", "scalarize", "retarget2", "imm", "sgpr"][..], &["dce", "cmpfold", "scalarize", "sgpr", "imm", "addrfold", "shl"][..]].iter().enumerate() {
            if let Ok(c) = with_opts(opts, || compile(&f)) {
                if let Err(e) = assemble(&clang, &c.asm, &format!("{k}-{i}")) {
                    bad.push(format!("{k} {opts:?}: {e}"));
                }
            }
        }
    }
    assert!(bad.is_empty(), "assembler rejected:\n{}", bad.join("\n"));
}