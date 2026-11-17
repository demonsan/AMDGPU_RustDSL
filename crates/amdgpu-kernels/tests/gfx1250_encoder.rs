//! G1 of docs/gfx1250-plan.md: the in-process gfx1250 encoder against the pinned llvm-mc (backend.gfx1250.lock; `GFX1250_MC` overrides),
//! byte for byte: every instruction line the gfx1250 probes emit, plus the encoder-table samples of tools/gen_enc_tables_gfx1250.py.
//! Skips (loudly) without the assembler.

use std::process::Command;

fn mc() -> Option<String> {
    let c = std::env::var("GFX1250_MC").unwrap_or_else(|_| "/opt/rocm/llvm/bin/llvm-mc".into());
    let ok = Command::new(&c).arg("--version").output().map(|o| o.status.success()).unwrap_or(false);
    if !ok {
        eprintln!("SKIPPED: no gfx1250 llvm-mc at {c} (set GFX1250_MC)");
    }
    ok.then_some(c)
}

/// llvm-mc's bytes for each line (one encoding per input line).
fn oracle(mc: &str, lines: &[String]) -> Vec<Vec<u8>> {
    let mut child = Command::new(mc)
        .args(["-triple=amdgcn-amd-amdhsa", "-mcpu=gfx1250", "-show-encoding"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    use std::io::Write;
    child.stdin.take().unwrap().write_all((lines.join("\n") + "\n").as_bytes()).unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.split("encoding: [").nth(1))
        .map(|e| e.trim_end_matches(']').split(',').map(|x| u8::from_str_radix(x.trim().trim_start_matches("0x"), 16).unwrap()).collect())
        .collect()
}

#[test]
fn the_emitted_lines_encode_like_llvm_mc() {
    let Some(mc) = mc() else { return };
    let mut lines: Vec<String> = Vec::new();
    for k in amdgpu_kernels::gfx1250_kernel_names() {
        let f = amdgpu_kernels::kernel_by_name(k).unwrap();
        let c = amdgpu_codegen::compile_target(&f, "gfx1250").unwrap();
        for l in c.asm.lines().map(|l| l.trim()) {
            let inst = !l.is_empty() && !l.starts_with('.') && !l.starts_with("//") && !l.ends_with(':') && !l.starts_with("s_branch") && !l.starts_with("s_cbranch");
            if inst && !lines.contains(&l.to_string()) {
                lines.push(l.to_string());
            }
        }
    }
    // the table samples, a few operand shapes per format
    for l in [
        "s_mov_b32 s5, 0x12345", "s_add_co_u32 s4, s5, -3", "s_cmp_lt_u32 s4, 7", "s_cselect_b32 s3, s4, 0", "s_load_b128 s[4:7], s[0:1], 0x40",
        "v_mov_b32_e32 v7, 1.0", "v_mov_b32_e32 v7, 0x3f800000", "v_mul_f32_e32 v1, 0x40490fdb, v2", "v_cmp_gt_f32_e32 vcc_lo, v1, v2",
        "v_cndmask_b32_e32 v1, v2, v3, vcc_lo", "v_fma_f32 v1, v2, v3, v4", "v_fma_f32 v1, -v2, v3, 0.5", "v_mul_lo_u32 v3, v1, 12",
        "v_add_co_ci_u32_e32 v5, vcc_lo, v3, v4, vcc_lo", "v_sub_co_u32 v4, vcc_lo, v2, v1", "global_load_b128 v[4:7], v[8:9], off offset:-16",
        "global_load_b32 v1, v2, s[4:5] offset:64", "global_store_b64 v[8:9], v[2:3], off offset:8", "ds_load_b128 v[4:7], v9 offset:1024",
        "ds_store_b64 v9, v[2:3] offset:8", "tensor_load_to_lds s[0:3], s[4:11]", "tensor_store_from_lds s[8:11], s[16:23], s[24:27], s[28:31]",
        "tensor_load_to_lds s[0:3], s[4:11] th:TH_LOAD_BYPASS scope:SCOPE_SYS", "s_wait_tensorcnt 0x3", "s_barrier_signal -3", "s_barrier_wait -3",
        "s_wait_loadcnt_dscnt 0x0", "s_set_vgpr_msb 0x41", "v_readfirstlane_b32 s7, v3", "s_mov_b32 s1, ttmp7",
    ] {
        lines.push(l.to_string());
    }
    let want = oracle(&mc, &lines);
    assert_eq!(want.len(), lines.len(), "llvm-mc gave {} encodings for {} lines", want.len(), lines.len());
    let mut bad = Vec::new();
    for (l, w) in lines.iter().zip(&want) {
        match amdgpu_target::gfx1250_co::encode_line(l, 0, &|_| Some(0)) {
            Ok(b) if &b == w => {}
            Ok(b) => bad.push(format!("{l}: ours {b:02x?} llvm-mc {w:02x?}")),
            Err(e) => bad.push(format!("{l}: {e}")),
        }
    }
    assert!(bad.is_empty(), "{} of {} lines differ:\n{}", bad.len(), lines.len(), bad.join("\n"));
}
