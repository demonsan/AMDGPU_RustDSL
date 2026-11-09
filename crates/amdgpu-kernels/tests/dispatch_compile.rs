//! Every kernel a launcher can reach must compile and assemble: the names the dispatch rules produce over a grid of shapes (decode16, bf16 prefill, MLA,
//! attention backward, fused MoE, the GEMM family) and the instantiations the Python ops pick at common shapes, own backend. A kernel on a
//! dispatch path that no test compiled (fmha_fwd_hd128_bf16_v3_causal ran out of SGPRs for a week) must fail here.
//!
//! Default run: the dispatch-derived set. `cargo test --release -p amdgpu-kernels --test dispatch_compile -- --ignored`: the full sweep (the
//! Python-op instantiations and every `amdc list` kernel too), run by python/tests/run_all.sh.

use amdgpu_compiler::backend::OwnTail;
use amdgpu_compiler::compile_with;
use amdgpu_kernels::dispatch::*;
use amdgpu_kernels::{kernel_by_name, kernel_names};
use std::collections::BTreeSet;

/// The kernel names the dispatch rules produce over a grid of launch shapes.
pub fn dispatched_names() -> BTreeSet<String> {
    let mut n = BTreeSet::new();
    // paged decode
    for b in [1u64, 4, 16, 64, 256] {
        for hk in [1u64, 8] {
            for ctx in [256u64, 2048, 8192] {
                for page in [1u32, 16] {
                    n.insert(decode16_kernel(page, decode16_plan(b, hk, ctx)).0);
                }
            }
        }
    }
    // bf16 prefill (+ the key-split combine)
    for b in [1u64, 4] {
        for h in [8u64, 32] {
            for (sq, sk) in [(128u64, 128u64), (1000, 1300), (4096, 4096)] {
                for causal in [false, true] {
                    let (k, s) = bf16_prefill_kernel(b, h, sq, sk, causal, 256);
                    n.insert(k.to_string());
                    if s > 1 {
                        n.insert("attn_combine_bf16_hd128_q".into());
                    }
                }
            }
        }
    }
    // per-tensor FP8 prefill (+ the KV-split combine)
    for (b, h, sq) in [(1u64, 8u64, 2048u64), (1, 8, 4096), (2, 32, 8192)] {
        for causal in [false, true] {
            let (k, s) = fp8pt_prefill_kernel(b, h, sq, sq, causal, 256);
            n.insert(k.to_string());
            if s > 1 {
                n.insert("fmha_fwd_hd128_mxfp8_v3_kvsplit_combine8h".into());
            }
        }
    }
    // MLA prefill hd192/hd128 (+ the key-split combine)
    for (b, h, sq) in [(1u64, 8u64, 4096u64), (1, 16, 2048), (4, 16, 2048), (1, 128, 1024)] {
        for causal in [false, true] {
            let (k, s) = hd192_prefill_kernel(b, h, sq, sq, causal, 256);
            n.insert(k.to_string());
            if s > 1 {
                n.insert("attn_combine_bf16_hd128_q".into());
            }
        }
    }
    // MLA decode
    for b in [1u64, 64] {
        for h in [16u64, 128] {
            let (hp, _) = mla_config(b, h);
            for p in [1, 16] {
                n.insert(format!("mla_decode_bf16_{}p{p}", if hp { "hp_" } else { "" }));
            }
        }
    }
    n.insert("attn_combine_bf16_hd512".into());
    // attention backward
    let (dq, dkdv) = bwd_causal_kernels();
    for k in ["fmha_bwd_dot_hd128_bf16", "fmha_bwd_dq_hd128_bf16", "fmha_bwd_dkdv_hd128_bf16", dq, dkdv] {
        n.insert(k.into());
    }
    // fused MoE: routing sort, the two stages, the top-k sum
    for (m, topk, e) in [(16u64, 2u64, 8u64), (4096, 8, 128), (256, 8, 64)] {
        let bm = moe_block_m(m, topk, e);
        let bms = if bm == 64 { "_bm64" } else { "" };
        n.insert(format!("moe_sorting_e{e}_k{topk}_b{bm}"));
        n.insert(format!("moe_sum_bf16_k{topk}"));
        for dt in ["bf16", "fp8"] {
            n.insert(format!("gemm_moe1_{dt}_silu{bms}"));
            n.insert(format!("gemm_moe2_{dt}{bms}"));
        }
    }
    // the GEMM family (blockscale split-K by its rule; the Python tile rule picks the 256 or the _128 kernels)
    for (m, nn, k) in [(256u64, 1024u64, 512u64), (4096, 4096, 4096)] {
        let (rows, s) = blockscale_config(m, nn, k, 256);
        n.insert(match (rows, s > 1) { (128, _) => "gemm_a8w8_blockscale".into(), (_, false) => "gemm_a8w8_blockscale_64".into(), _ => "gemm_a8w8_blockscale_64_splitk".into() });
        if s > 1 {
            n.insert(format!("gemm_splitk_reduce_s{s}"));
        }
    }
    for k in ["gemm_a16w16_bf16", "gemm_a16w16_bf16_128", "gemm_a8w8_rowcol_fp8", "gemm_a8w8_rowcol_fp8_128", "gemm_a4w4_mxfp4", "gemm_a16w4_mxfp4", "gemm_a8w4_mxfp_128"] {
        n.insert(k.into());
    }
    n
}

/// The instantiations the Python memory-bound ops pick at common model shapes.
fn python_op_names() -> BTreeSet<String> {
    let mut n = BTreeSet::new();
    for d in [4096, 7168] {
        for k in [format!("rmsnorm_bf16_n{d}"), format!("add_rmsnorm_bf16_n{d}"), format!("add_rmsnorm_fp8_n{d}"), format!("rmsnorm_quant_fp8_n{d}"),
            format!("rmsnorm_quant_mxfp4_bf16_k{d}"), format!("quant_token_bf16_fp8_n{d}"), format!("quant_token_bf16_int8_n{d}"), format!("quant_group128_bf16_fp8_n{d}"),
            format!("silu_mul_bf16_d{d}"), format!("gelu_tanh_mul_bf16_d{d}"), format!("silu_mul_quant_fp8_d{d}")]
        {
            n.insert(k);
        }
    }
    for k in ["quant_bf16_fp8_static", "rope_neox_bf16_h32_d128", "rope_gptj_bf16_h32_d128", "qknorm_rope_neox_bf16_h32_d128", "qknorm_rope_kvcache_neox_bf16_h8_d128",
        "qknorm_rope_kvcache_neox_fp8_h8_d128_i64", "reshape_cache_flash_bf16_h8_d128", "reshape_cache_blocked_bf16_h8_d128_b16", "reshape_cache_blocked_fp8_h8_d128_b16_i64",
        "topk_softmax_bf16_e128_k8", "topk_softmax_renorm_bf16_e256_k8", "moe_reduce_bf16_k8_d4096", "moe_combine_bf16_k8_d4096", "argmax_bf16_v128256", "sample_bf16_v128256",
        "sample_topkp_bf16_v128256", "topk_per_row_decode_f32_k2048", "topk_per_row_prefill_f32_k2048", "causal_conv1d_fwd_bf16_w4_silu", "causal_conv1d_update_bf16_w4_s3_idx_silu"]
    {
        n.insert(k.into());
    }
    n
}

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
        .args(["-x", "assembler", "-target", "amdgcn-amd-amdhsa", "-mcpu=gfx950", "-mcode-object-version=6", "-o"])
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
    let mut n = dispatched_names();
    n.extend(python_op_names());
    n.extend(kernel_names().iter().map(|s| s.to_string()));
    compile_all(&n);
}
