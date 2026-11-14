//! The `auto` backend's guard: every reachable kernel that `auto` routes to the LLVM tail must compile through it, and its
//! `.amdgpu_metadata` `.args` must declare every DSL kernarg (offset and size) - the runtime copies a buffer-pointer launch argument by
//! argument from that list, so an undeclared argument arrives as 0 (the causal_conv1d_fwd_bf16_w4_silu_sched illegal access: its two
//! pointers at offsets 76 / 84 were dropped with the IR shell's padding parameters). The instruction streams of the two tails are compared
//! by tests/tail_invariance.rs.

use amdgpu_codegen::backend::{LlvmTail, LlvmTools};

fn llc_available() -> bool {
    let t = LlvmTools::locate();
    let ok = std::process::Command::new(t.tool("llc")).arg("--version").output().map(|o| o.status.success()).unwrap_or(false);
    if !ok {
        eprintln!("SKIPPED: llc not runnable at {} (set ROCM_LLVM_BIN)", t.bin.display());
    }
    ok
}

/// (offset, size) of every `.args` entry of the metadata.
fn metadata_args(asm: &str) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    let mut in_args = false;
    let mut cur: (Option<u32>, Option<u32>) = (None, None);
    for l in asm.lines() {
        let t = l.trim();
        if t == ".args:" {
            in_args = true;
            continue;
        }
        if !in_args {
            continue;
        }
        if t.starts_with("- ") {
            if let (Some(o), Some(s)) = cur {
                out.push((o, s));
            }
            cur = (None, None);
        } else if !t.starts_with('.') && !t.is_empty() {
            break;
        }
        let kv = t.trim_start_matches("- ");
        if let Some(v) = kv.strip_prefix(".offset:") {
            cur.0 = v.trim().parse().ok();
        } else if let Some(v) = kv.strip_prefix(".size:") {
            cur.1 = v.trim().parse().ok();
        } else if kv.starts_with(".name:") && cur.0.is_some() && cur.1.is_some() && !t.starts_with("- ") {
            // the kernel's own .name after the list ends it
        }
    }
    if let (Some(o), Some(s)) = cur {
        out.push((o, s));
    }
    out
}

#[test]
fn every_auto_routed_llvm_kernel_compiles_and_declares_all_its_args() {
    if !llc_available() {
        return;
    }
    let tail = LlvmTail::new();
    let mut bad = Vec::new();
    let mut n = 0;
    for k in amdgpu_kernels::reachable::manifest_names() {
        let Some(f) = amdgpu_kernels::kernel_by_name(&k) else { continue };
        if amdgpu_codegen::resolve_backend("auto", &f).0 != "llvm-tail" {
            continue;
        }
        n += 1;
        let asm = match amdgpu_codegen::compile_with(&f, &tail) {
            Ok(c) => c.asm,
            Err(e) => {
                bad.push(format!("{k}: auto routes it to the LLVM tail, which fails: {e}"));
                continue;
            }
        };
        let mut want: Vec<(u32, u32)> = f.args.iter().map(|a| (a.kernarg_offset, a.size)).collect();
        want.sort();
        let got = metadata_args(&asm);
        if got != want {
            bad.push(format!("{k}: metadata .args {got:?} != DSL kernargs {want:?}"));
        }
    }
    eprintln!("{n} kernels routed to the LLVM tail by auto");
    assert!(bad.is_empty(), "{} auto-routed kernel(s) broken:\n{}", bad.len(), bad.join("\n"));
}
