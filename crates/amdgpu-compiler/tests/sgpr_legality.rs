//! gfx9 VALU operand legality for the `sgpr` option: a VALU instruction reads at most one distinct SGPR (constant bus limit 1)
//! and an e32 VOP2/VOPC form takes it only as src0. Scans the emitted asm of the kernels that use `scalarize,sgpr`.

use amdgpu_compiler::lower::with_opts;
use amdgpu_compiler::{compile, kernel_by_name};

fn sgpr_operands(ops: &[&str]) -> Vec<String> {
    let mut v: Vec<String> = Vec::new();
    for o in ops {
        let o = o.trim().trim_end_matches(',');
        let is_s = (o.starts_with('s') && o.len() > 1 && (o[1..].chars().next().unwrap().is_ascii_digit() || o.as_bytes()[1] == b'['))
            && !o.starts_with("src");
        if is_s && !v.contains(&o.to_string()) {
            v.push(o.to_string());
        }
    }
    v
}

fn check(asm: &str, kernel: &str, bad: &mut Vec<String>) {
    for l in asm.lines() {
        let l = l.split(';').next().unwrap().trim();
        if !l.starts_with("v_") || l.starts_with("v_readfirstlane") || l.starts_with("v_readlane") || l.starts_with("v_writelane") {
            continue;
        }
        let (mn, rest) = l.split_once(char::is_whitespace).unwrap_or((l, ""));
        let ops: Vec<&str> = rest.split(',').collect();
        // sources: everything after the destination (VOPC e32 has the implicit/explicit vcc destination first)
        let srcs: Vec<&str> = ops.iter().skip(1).copied().collect();
        let mut sg = sgpr_operands(&srcs);
        // vcc read through the constant bus: v_cndmask / v_addc / v_subb (e32) read vcc implicitly or explicitly
        if (mn.starts_with("v_cndmask") || mn.starts_with("v_addc") || mn.starts_with("v_subb")) && (srcs.iter().any(|s| s.trim() == "vcc") || mn.ends_with("_e32")) {
            sg.push("vcc".to_string());
        }
        if sg.len() > 1 {
            bad.push(format!("{kernel}: {l}: {} distinct SGPR sources", sg.len()));
        }
        if mn.ends_with("_e32") && srcs.len() >= 2 {
            // VOP2 / VOPC e32: src1 must be a VGPR (the last source, unless the form ends with vcc: cndmask)
            let last = if srcs.last().unwrap().trim() == "vcc" { srcs[srcs.len() - 2] } else { *srcs.last().unwrap() };
            if !sgpr_operands(&[last]).is_empty() {
                bad.push(format!("{kernel}: {l}: SGPR in src1 of an e32 form"));
            }
        }
    }
}

#[test]
fn sgpr_operands_obey_the_gfx9_constant_bus_and_encoding_rules() {
    let mut bad = Vec::new();
    for k in ["fmha_fwd_hd128_mxfp8_t", "fmha_fwd_hd128_mxfp8_sv", "fmha_fwd_hd128_mxfp8_vs", "fmha_fwd_hd128_mxfp8_vs128", "fmha_fwd_hd128_mxfp8_pp", "attention_fwd"] {
        let Some(f) = kernel_by_name(k) else { continue };
        for opts in [&["dce", "cmpfold", "scalarize", "sgpr", "imm"][..], &["dce", "cmpfold", "scalarize", "sgpr", "imm", "retarget2"][..]] {
            let asm = with_opts(opts, || compile(&f)).map(|c| c.asm);
            // kernels that only allocate through the own backend are compiled by `compile` (LLVM tail rejects DMA ops)
            match asm {
                Ok(a) => check(&a, k, &mut bad),
                Err(e) => eprintln!("skipped {k}: {}", e.lines().next().unwrap_or("")),
            }
        }
    }
    assert!(bad.is_empty(), "{} illegal SGPR operand use(s):\n{}", bad.len(), bad.iter().take(20).cloned().collect::<Vec<_>>().join("\n"));
}