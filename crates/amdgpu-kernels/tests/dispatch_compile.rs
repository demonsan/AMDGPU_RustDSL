//! Every kernel a launcher can reach must compile and assemble: the names the dispatch rules produce over a grid of shapes (decode16, bf16 prefill, MLA,
//! attention backward, fused MoE, the GEMM family) and the instantiations the Python ops pick at common shapes, own backend. A kernel on a
//! dispatch path that no test compiled (fmha_fwd_hd128_bf16_v3_causal ran out of SGPRs for a week) must fail here.
//!
//! Default run: the dispatch-derived set. `cargo test --release -p amdgpu-kernels --test dispatch_compile -- --ignored`: the full sweep (the
//! Python-op instantiations and every `amdc list` kernel too), run by python/tests/run_all.sh.

use amdgpu_compiler::backend::OwnTail;
use amdgpu_compiler::compile_with;
use amdgpu_kernels::kernel_by_name;
use amdgpu_kernels::reachable::{dispatched_names, manifest_names};
use std::collections::BTreeSet;

/// Reachable kernels known not to compile (reported; each entry is re-checked below so the list cannot go stale).
/// pa_decode16_bf16_hd128_p1_f: page size 1 runs out of SGPRs since the decode16 round-2 merge (7d483f4e).
const KNOWN_BROKEN: &[&str] = &[];

/// Assembles `asm` with the real assembler (None when clang is not runnable here: the check is skipped).
fn assemble(asm: &str, tag: &str) -> Option<Result<(), String>> {
    let clang = amdgpu_compiler::backend::LlvmTools::locate().tool("clang");
    if !std::process::Command::new(&clang).arg("--version").output().map(|o| o.status.success()).unwrap_or(false) {
        return None;
    }
    // unique per call: the tests of this binary run in parallel threads and both sweeps assemble the same kernel names
    static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("amdc-dc-{}-{:?}-{n}-{tag}", std::process::id(), std::thread::current().id()).replace(['(', ')'], ""));
    std::fs::create_dir_all(&dir).ok()?;
    let s = dir.join("k.s");
    std::fs::write(&s, asm).ok()?;
    let out = std::process::Command::new(&clang)
        .args(amdgpu_compiler::target::T.llvm.asm_args)
        .arg("-o")
        .arg(dir.join("k.co"))
        .arg(&s)
        .output()
        .ok()?;
    let _ = std::fs::remove_dir_all(&dir);
    Some(if out.status.success() { Ok(()) } else { Err(String::from_utf8_lossy(&out.stderr).lines().filter(|l| l.contains("error")).take(2).collect::<Vec<_>>().join(" | ")) })
}

/// The kernel-descriptor resource fields the assembler / loader range-check, checked without an assembler: next_free_sgpr <= 102 (vcc and
/// the trap registers are separate on gfx950), next_free_vgpr <= 512 (unified arch + AGPR file), accum_offset in [4, 256] and a multiple of
/// 4, group segment <= 160 KiB.
fn descriptor_limits(asm: &str) -> Result<(), String> {
    let field = |name: &str| -> Option<i64> {
        asm.lines().find_map(|l| l.trim().strip_prefix(name).and_then(|r| r.trim().parse::<i64>().ok()))
    };
    let mut bad = Vec::new();
    match field(".amdhsa_next_free_sgpr") {
        Some(s) if s <= 102 => {}
        s => bad.push(format!("next_free_sgpr {s:?} > 102")),
    }
    match field(".amdhsa_next_free_vgpr") {
        Some(v) if v <= 512 => {}
        v => bad.push(format!("next_free_vgpr {v:?} > 512")),
    }
    if let Some(a) = field(".amdhsa_accum_offset") {
        if !(4..=256).contains(&a) || a % 4 != 0 {
            bad.push(format!("accum_offset {a} not in [4, 256] / not a multiple of 4"));
        }
    }
    match field(".amdhsa_group_segment_fixed_size") {
        Some(l) if l <= 160 * 1024 => {}
        l => bad.push(format!("group segment {l:?} > 160 KiB")),
    }
    if bad.is_empty() { Ok(()) } else { Err(bad.join(", ")) }
}

#[test]
fn the_descriptor_check_rejects_out_of_range_fields() {
    let ok = ".amdhsa_group_segment_fixed_size 65536\n.amdhsa_next_free_vgpr 512\n.amdhsa_next_free_sgpr 102\n.amdhsa_accum_offset 256\n";
    assert!(descriptor_limits(ok).is_ok());
    assert!(descriptor_limits(&ok.replace("sgpr 102", "sgpr 104")).is_err());
    assert!(descriptor_limits(&ok.replace("vgpr 512", "vgpr 520")).is_err());
    assert!(descriptor_limits(&ok.replace("65536", "163841")).is_err());
}

fn compile_all(names: &BTreeSet<String>) {
    let mut bad = Vec::new();
    for k in names.iter().filter(|k| !KNOWN_BROKEN.contains(&k.as_str())) {
        match kernel_by_name(k) {
            None => bad.push(format!("{k}: not in the registry")),
            Some(f) => {
                match compile_with(&f, &OwnTail) {
                    Err(e) => bad.push(format!("{k}: {}", e.lines().take(3).collect::<Vec<_>>().join(" | "))),
                    // the assembler too (descriptor fields such as next_free_sgpr are only range-checked there)
                    Ok(c) => {
                        if let Err(e) = descriptor_limits(&c.asm) {
                            bad.push(format!("{k}: descriptor: {e}"));
                        }
                        if let Some(Err(e)) = assemble(&c.asm, k) {
                            bad.push(format!("{k}: assembler: {e}"));
                        }
                    }
                }
            }
        }
    }
    assert!(bad.is_empty(), "{} of {} reachable kernels do not compile:\n{}", bad.len(), names.len(), bad.join("\n"));
}

#[test]
fn the_known_broken_kernels_still_fail() {
    for k in KNOWN_BROKEN {
        let f = kernel_by_name(k).expect("registered");
        assert!(compile_with(&f, &OwnTail).is_err(), "{k} compiles now: remove it from KNOWN_BROKEN");
    }
}

#[test]
fn every_dispatched_kernel_compiles() {
    compile_all(&dispatched_names());
}

#[test]
#[ignore = "full sweep (slow in debug): cargo test --release -p amdgpu-kernels --test dispatch_compile -- --ignored"]
fn every_reachable_kernel_compiles() {
    compile_all(&manifest_names());
}
