//! The user's requirement: register allocation and instruction scheduling stay OURS; the LLVM tail only
//! inserts hardware waits / hazard padding and writes the kernel descriptor.
//!
//! For every kernel we compile with the complete own tail and with the LLVM tail and require
//!  1. the instruction streams are identical (same opcodes, same physical registers, same order) once
//!     `s_nop`, `s_waitcnt`, `s_delay_alu`, `s_branch` (LLVM drops fallthrough branches) and cosmetic operand
//!     spelling are normalised away  => LLVM neither re-allocated registers nor reordered anything;
//!  2. the exported MIR is physical-register only (`noVRegs: true`, no virtual register operand);
//!  3. the descriptor LLVM wrote agrees with ours (LDS size, kernarg size, enough VGPRs).
//! Run with AMDC_SCHED=1 / auto / AMDC_SCHED_OPTS=swp to cover the scheduler paths too.
//! Needs the pinned llc (skips loudly without).

use amdgpu_compiler::backend::{LlvmTail, LlvmTools, OwnTail};
use amdgpu_compiler::{compile_with, export_mir, kernel_by_name, kernel_names};

fn llc_available() -> bool {
    let t = LlvmTools::locate();
    let ok = std::process::Command::new(t.tool("llc")).arg("--version").output().map(|o| o.status.success()).unwrap_or(false);
    if !ok {
        eprintln!("SKIPPED: llc not runnable at {}", t.bin.display());
    }
    ok
}

const COMMUTATIVE: &[&str] = &[
    "v_add_co_u32", "v_add_u32", "v_add_f32", "v_mul_f32", "v_max_f32", "v_and_b32", "v_or_b32", "v_xor_b32", "v_mul_lo_u32",
];

fn inline_float(tok: &str) -> Option<u32> {
    Some(match tok {
        "0.5" => 0x3f00_0000,
        "-0.5" => 0xbf00_0000,
        "1.0" => 0x3f80_0000,
        "-1.0" => 0xbf80_0000,
        "2.0" => 0x4000_0000,
        "-2.0" => 0xc000_0000,
        "4.0" => 0x4080_0000,
        "-4.0" => 0xc080_0000,
        "0.15915494" => 0x3e22_f983,
        _ => return None,
    })
}

fn norm_operand(t: &str) -> String {
    let t = t.trim();
    if let Some(v) = inline_float(t) {
        return format!("#{v}");
    }
    let n: Option<i64> = if let Some(h) = t.strip_prefix("0x") {
        i64::from_str_radix(h, 16).ok()
    } else {
        t.parse::<i64>().ok()
    };
    match n {
        Some(v) => format!("#{}", (v as i128).rem_euclid(1i128 << 32)),
        None => t.to_string(),
    }
}

/// Instruction stream without the parts the tail is allowed to change.
fn instruction_stream(asm: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in asm.lines() {
        let l = line.split(';').next().unwrap().split("//").next().unwrap().trim();
        if l.is_empty() || l.starts_with('.') || l.ends_with(':') || l.starts_with('-') || l.contains(": ") {
            continue;
        }
        let (mn, rest) = match l.split_once(char::is_whitespace) {
            Some((m, r)) => (m, r.trim()),
            None => (l, ""),
        };
        if matches!(mn, "s_nop" | "s_waitcnt" | "s_delay_alu" | "s_branch" | "s_set_vgpr_msb") {
            continue;
        }
        // LLVM's vccz-corruption workaround (`s_mov_b64 vcc, vcc` before `s_cbranch_vccz`): a hazard instruction the tail may add.
        if mn == "s_mov_b64" && rest == "vcc, vcc" {
            continue;
        }
        let mn = mn.trim_end_matches("_e32").trim_end_matches("_e64");
        // operands split at top-level commas; modifiers (op_sel_hi:[..], offset:N) stay attached to the last operand
        let rest = rest.replace(" op_sel_hi:[0,0,0]", "");
        let mut ops: Vec<String> = Vec::new();
        let (mut depth, mut cur) = (0, String::new());
        for ch in rest.chars() {
            match ch {
                '[' | '(' => {
                    depth += 1;
                    cur.push(ch)
                }
                ']' | ')' => {
                    depth -= 1;
                    cur.push(ch)
                }
                ',' if depth == 0 => {
                    ops.push(std::mem::take(&mut cur));
                }
                _ => cur.push(ch),
            }
        }
        if !cur.trim().is_empty() {
            ops.push(cur);
        }
        let mut ops: Vec<String> = if mn.starts_with("s_cbranch") { vec!["<label>".to_string()] } else { ops.iter().map(|o| norm_operand(o)).collect() };
        if COMMUTATIVE.contains(&mn) && ops.len() >= 3 {
            let n = ops.len();
            if ops[n - 2] > ops[n - 1] {
                ops.swap(n - 2, n - 1);
            }
        }
        out.push(format!("{mn} {}", ops.join(", ")));
    }
    out
}

fn directive(asm: &str, key: &str) -> Option<u32> {
    asm.lines().find_map(|l| {
        let l = l.trim();
        l.strip_prefix(key).and_then(|v| v.trim().parse().ok())
    })
}

#[test]
fn llvm_tail_changes_only_waits_nops_and_descriptor() {
    if !llc_available() {
        return;
    }
    let only = std::env::var("AMDC_TEST_KERNELS").ok().filter(|s| !s.is_empty());
    let tail = LlvmTail::new();
    let mut bad = Vec::new();
    let mut checked = 0;
    for k in kernel_names() {
        if let Some(o) = &only {
            if !o.split(',').any(|x| x == *k) {
                continue;
            }
        }
        let f = kernel_by_name(k).unwrap();
        let own = compile_with(&f, &OwnTail).unwrap_or_else(|e| panic!("{k}: own tail: {e}"));
        let llv = compile_with(&f, &tail).unwrap_or_else(|e| panic!("{k}: llvm tail: {e}"));
        let (a, b) = (instruction_stream(&own.asm), instruction_stream(&llv.asm));
        checked += 1;
        if a != b {
            let i = a.iter().zip(b.iter()).position(|(x, y)| x != y).unwrap_or(a.len().min(b.len()));
            bad.push(format!(
                "{k}: instruction streams differ (own {} / llvm {} instructions) first at #{i}: own `{}` llvm `{}`",
                a.len(),
                b.len(),
                a.get(i).cloned().unwrap_or_default(),
                b.get(i).cloned().unwrap_or_default()
            ));
        }
        // exported MIR is physical only
        let mir = export_mir(&f, &tail).unwrap();
        if !mir.contains("noVRegs:         true") || !mir.contains("isSSA:           false") {
            bad.push(format!("{k}: exported MIR is not marked noVRegs/!isSSA"));
        }
        let body = mir.split("body:").nth(1).unwrap_or("");
        let has_vreg = body.lines().any(|l| {
            let l = l.split(" :: ").next().unwrap();
            let b = l.as_bytes();
            (0..b.len()).any(|i| b[i] == b'%' && i + 1 < b.len() && (b[i + 1].is_ascii_digit() || (b[i + 1].is_ascii_alphabetic() && !l[i + 1..].starts_with("bb."))))
        });
        if has_vreg {
            bad.push(format!("{k}: exported MIR contains a virtual register"));
        }
        // descriptor sanity
        let vg = directive(&llv.asm, ".amdhsa_next_free_vgpr").unwrap_or(0);
        if vg < own.vgpr.min(512) || vg == 0 {
            bad.push(format!("{k}: LLVM descriptor next_free_vgpr {vg} < our allocation {}", own.vgpr));
        }
        let (la, lb) = (directive(&own.asm, ".amdhsa_group_segment_fixed_size"), directive(&llv.asm, ".amdhsa_group_segment_fixed_size"));
        if la != lb {
            bad.push(format!("{k}: group segment size differs {la:?} vs {lb:?}"));
        }
        let (ka, kb) = (directive(&own.asm, ".amdhsa_kernarg_size"), directive(&llv.asm, ".amdhsa_kernarg_size"));
        if ka != kb {
            bad.push(format!("{k}: kernarg size differs {ka:?} vs {kb:?}"));
        }
    }
    assert!(checked > 0);
    assert!(bad.is_empty(), "{} invariance violations:\n{}", bad.len(), bad.join("\n"));
}