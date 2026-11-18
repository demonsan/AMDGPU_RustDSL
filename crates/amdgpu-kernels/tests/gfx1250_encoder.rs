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
            let inst = ["s_", "v_", "global_", "ds_", "tensor_"].iter().any(|p| l.starts_with(p)) && !l.starts_with("s_branch") && !l.starts_with("s_cbranch");
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
        "s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1", "v_wmma_f32_16x16x32_bf16 v[0:7], v[8:15], v[16:23], v[0:7]", "v_cmp_gt_f32_e64 s10, v1, v2", "s_barrier_signal_isfirst -1", "v_cvt_pk_fp8_f32 v95, v1, v2 op_sel:[0,0,1]", "v_cvt_pk_fp8_f32 v95, v1, v2", "global_atomic_add_u32 v5, v[2:3], v4, off th:TH_ATOMIC_RETURN", "global_atomic_add_u32 v5, v[2:3], v4, off offset:16 th:TH_ATOMIC_RETURN", "global_atomic_add_u32 v[2:3], v4, off", "global_wb scope:SCOPE_DEV", "global_inv scope:SCOPE_DEV", "s_get_pc_i64 s[36:37]", "s_prefetch_inst s[36:37], 0x0, s38, 0", "s_prefetch_inst s[36:37], 0x100, s38, 3", "s_prefetch_inst s[2:3], 0x40, null, 1", "v_wmma_scale_f32_32x16x128_f4 v[16:31], v[0:15], v[8:15], v[16:31], v12, v17 matrix_a_reuse", "v_wmma_scale_f32_32x16x128_f4 v[16:31], v[0:15], v[8:15], v[16:31], v12, v17 matrix_a_reuse matrix_b_reuse", "v_wmma_f32_16x16x64_fp8_fp8 v[0:7], v[8:15], v[16:23], v[0:7] matrix_b_reuse", "v_wmma_f32_16x16x64_fp8_fp8 v[0:7], v[8:15], v[16:23], v[0:7] matrix_a_reuse", "v_pk_add_f32 v[20:21], v[16:17], v[16:17]", "v_pk_mul_f32 v[20:21], v[16:17], v[18:19]", "v_pk_fma_f32 v[20:21], v[16:17], v[18:19], v[22:23]", "s_wait_alu 0xf9f", "s_wait_alu 0xff83", "s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 2", "s_mov_b32 m0, 0x20001", "s_barrier_init m0", "s_barrier_join 1", "s_barrier_leave", "s_barrier_signal 1", "s_barrier_wait 1", "buffer_load_b32 v1, v2, s[4:7], null offen offset:16", "buffer_load_b128 v[4:7], off, s[8:11], s3 offset:64", "buffer_store_b16 v1, v2, s[4:7], s3 offen", "buffer_atomic_add_u32 v1, off, s[4:7], null offset:1024 th:TH_ATOMIC_RETURN", "v_cmp_lt_i32_e64 s12, v1, 5", "v_cndmask_b32_e64 v1, v2, v3, s10", "s_mov_b32 exec_lo, s10", "s_mov_b32 s11, exec_lo",
        "v_wmma_f32_16x16x64_fp8_fp8 v[32:39], v[8:15], v[16:23], 0",
        "v_wmma_scale_f32_32x16x128_f4 v[64:79], v[16:31], v[32:39], 0, v3, v4 matrix_b_scale:MATRIX_SCALE_ROW1", "global_prefetch_b8 v0, s[6:7] scope:SCOPE_SE", "global_load_b32 v1, v2, s[4:5] scope:SCOPE_SE",
    ] {
        lines.push(l.to_string());
    }
    let want = oracle(&mc, &lines);
    assert_eq!(want.len(), lines.len(), "llvm-mc gave {} encodings for {} lines", want.len(), lines.len());
    let mut bad = Vec::new();
    for (l, w) in lines.iter().zip(&want) {
        match amdgpu_codegen::amdgpu_target::gfx1250_co::encode_line(l, 0, &|_| Some(0)) {
            Ok(b) if &b == w => {}
            Ok(b) => bad.push(format!("{l}: ours {b:02x?} llvm-mc {w:02x?}")),
            Err(e) => bad.push(format!("{l}: {e}")),
        }
    }
    assert!(bad.is_empty(), "{} of {} lines differ:\n{}", bad.len(), lines.len(), bad.join("\n"));
}

/// Whole functions: the encoder's bytes of each probe equal the `.text` section clang assembles from the same text (branches and labels
/// included; the code object itself is clang's for now).
#[test]
fn whole_probes_encode_like_clang() {
    let Some(mc) = mc() else { return };
    let bin = std::path::Path::new(&mc).parent().unwrap().to_path_buf();
    let dir = std::env::temp_dir().join(format!("g1250_enc_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for k in amdgpu_kernels::gfx1250_kernel_names() {
        let f = amdgpu_kernels::kernel_by_name(k).unwrap();
        let c = amdgpu_codegen::compile_target(&f, "gfx1250").unwrap();
        let (s, o, t) = (dir.join(format!("{k}.s")), dir.join(format!("{k}.o")), dir.join(format!("{k}.text")));
        std::fs::write(&s, &c.asm).unwrap();
        let st = Command::new(bin.join("clang"))
            .args(["-x", "assembler", "-target", "amdgcn-amd-amdhsa", "-mcpu=gfx1250", "-c"])
            .arg(&s).arg("-o").arg(&o).status().unwrap();
        assert!(st.success(), "{k}: clang");
        let st = Command::new(bin.join("llvm-objcopy")).args(["-O", "binary", "--only-section=.text"]).arg(&o).arg(&t).status().unwrap();
        assert!(st.success(), "{k}: objcopy");
        let want = std::fs::read(&t).unwrap();
        let body: String = c.asm.lines().take_while(|l| !l.starts_with(".size")).collect::<Vec<_>>().join("\n");
        let got = amdgpu_codegen::amdgpu_target::gfx1250_co::encode_text(&body).unwrap_or_else(|e| panic!("{k}: {e}"));
        assert_eq!(got, want[..got.len().min(want.len())].to_vec(), "{k}: .text differs");
        // after the function: the assembler's alignment padding (s_nop 0 words), then the next section's bytes
        let tail = &want[got.len()..];
        assert!(tail.len() < 4 || got.len() % 64 == 0 || tail[..4] == [0x00, 0x00, 0x80, 0xbf] || tail[..4] == [0, 0, 0, 0], "{k}: clang has more code: {:02x?}", &tail[..tail.len().min(32)]);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// A split-counter wait beyond the counter's width never blocks: the encoder refuses it (the `s_wait_asynccnt 72` of a replay; llvm-mc
/// accepts any 16-bit value), the in-range forms encode.
#[test]
fn split_counter_waits_are_range_checked() {
    use amdgpu_codegen::amdgpu_target::gfx1250_co::encode_line;
    for bad in ["s_wait_asynccnt 72", "s_wait_kmcnt 32", "s_wait_dscnt 64", "s_wait_loadcnt 0x40", "s_wait_tensorcnt 64"] {
        let e = encode_line(bad, 0, &|_| Some(0)).err().unwrap_or_else(|| panic!("{bad} must be refused"));
        assert!(e.contains("exceeds"), "{bad}: {e}");
    }
    for ok in ["s_wait_asynccnt 63", "s_wait_kmcnt 31", "s_wait_dscnt 0", "s_wait_tensorcnt 4"] {
        encode_line(ok, 0, &|_| Some(0)).unwrap_or_else(|e| panic!("{ok}: {e}"));
    }
}
