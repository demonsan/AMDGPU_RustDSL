//! G1 of docs/gfx1250-plan.md: the gfx1250 probes compile through the gfx1250 pipeline and the result assembles with the pinned gfx1250
//! assembler (backend.gfx1250.lock: the MI450 nodes' /opt/rocm/llvm/bin/clang; `GFX1250_CLANG` overrides). Skips (loudly) without it.

use std::process::Command;

const PROBES: &[&str] = &["probe_g1250_add", "probe_g1250_math"];

fn clang() -> Option<String> {
    let c = std::env::var("GFX1250_CLANG").unwrap_or_else(|_| "/opt/rocm/llvm/bin/clang".into());
    let ok = Command::new(&c).arg("--version").output().map(|o| o.status.success()).unwrap_or(false);
    if !ok {
        eprintln!("SKIPPED: no gfx1250 assembler at {c} (set GFX1250_CLANG)");
    }
    ok.then_some(c)
}

#[test]
fn the_probes_compile_for_gfx1250() {
    for k in PROBES {
        let f = amdgpu_kernels::kernel_by_name(k).unwrap();
        let c = amdgpu_codegen::compile_target(&f, "gfx1250").unwrap_or_else(|e| panic!("{k}: {e}"));
        assert!(c.asm.contains(".amdhsa_wavefront_size32 1"), "{k}");
        assert!(!c.asm.contains("s_waitcnt"), "{k}: a gfx950 wait");
    }
}

#[test]
fn the_probes_assemble_for_gfx1250() {
    let Some(cl) = clang() else { return };
    let dir = std::env::temp_dir().join(format!("g1250_emit_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for k in PROBES {
        let f = amdgpu_kernels::kernel_by_name(k).unwrap();
        let c = amdgpu_codegen::compile_target(&f, "gfx1250").unwrap();
        let s = dir.join(format!("{k}.s"));
        let o = dir.join(format!("{k}.co"));
        std::fs::write(&s, &c.asm).unwrap();
        let out = Command::new(&cl)
            .args(["-x", "assembler", "-target", "amdgcn-amd-amdhsa", "-mcpu=gfx1250", "-mcode-object-version=6"])
            .arg(&s)
            .arg("-o")
            .arg(&o)
            .output()
            .unwrap();
        assert!(out.status.success(), "{k}:\n{}\n{}", String::from_utf8_lossy(&out.stderr), c.asm);
    }
    let _ = std::fs::remove_dir_all(&dir);
}
