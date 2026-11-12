//! The kernel names a launcher can reach: the names the dispatch rules produce over a grid of launch shapes, the instantiations the Python
//! ops pick at common model shapes, and the registry list. Shared by `tests/dispatch_compile.rs` (every reachable kernel compiles and
//! assembles) and `amdc md5-manifest` (the golden asm md5 of every reachable kernel, tests/golden/md5_manifest.tsv).

use amdgpu_graph::dispatch::*;
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
                        n.insert("attn_combine_bf16_hd128_q16".into());
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
                n.insert("attn_combine_bf16_hd128_q16".into());
            }
        }
    }
    // MLA decode
    for b in [1u64, 64] {
        for h in [16u64, 128] {
            for ctx in [2048u64, u64::MAX] {
                let (hp, _) = mla_config(b, h, ctx);
                for p in [1, 16] {
                    n.insert(format!("mla_decode_bf16_{}p{p}", if hp { "hp_" } else { "" }));
                }
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
        for inter in [256u64, 2048] {
            let bm = moe_block_m(m, topk, e, inter, 256);
            let bms = if bm == 64 { "_bm64" } else { "" };
            n.insert(format!("moe_sorting_e{e}_k{topk}_b{bm}"));
            n.insert(format!("moe_sum_bf16_k{topk}"));
            for dt in ["bf16", "fp8"] {
                n.insert(format!("gemm_moe1_{dt}_silu{bms}"));
                n.insert(format!("gemm_moe2_{dt}{bms}"));
            }
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

/// The instantiations the Python memory-bound ops pick at common model shapes: DERIVED from the L1 fusion rules over the canonical op graphs
/// (`amdgpu_graph::fusion::canonical`, docs/l1-graph.md step 3).
pub fn python_op_names() -> BTreeSet<String> {
    amdgpu_graph::fusion::canonical::python_op_names()
}

/// Every fixed name `kernel_by_name` matches literally (the `"name" =>` / `"name" |` arms of `registry.rs` and the modules' `by_name`) that builds a kernel: the shipped
/// kernels and the non-parameterized families (`amdc list` holds only the probes).
pub fn registered_literal_names() -> BTreeSet<String> {
    // the registry and every module with a `by_name` (scanned for literal match arms)
    const SOURCES: &[&str] = &[
        include_str!("cache/mla.rs"),
        include_str!("cache/mod.rs"),
        include_str!("conv/mod.rs"),
        include_str!("dequant/mod.rs"),
        include_str!("eltwise/mod.rs"),
        include_str!("gemm/mod.rs"),
        include_str!("mlp/mod.rs"),
        include_str!("moe/mod.rs"),
        include_str!("moe/multi.rs"),
        include_str!("moe/reduce.rs"),
        include_str!("norm/mod.rs"),
        include_str!("probes/handoff.rs"),
        include_str!("registry.rs"),
        include_str!("rope/mod.rs"),
        include_str!("rows/mod.rs"),
        include_str!("rows/mxfp6.rs"),
        include_str!("rows/mxnorm.rs"),
        include_str!("rows/mxquant.rs"),
        include_str!("rows/quant.rs"),
        include_str!("sampling/mod.rs"),
        include_str!("topk/gate.rs"),
        include_str!("topk/mod.rs"),
        include_str!("topk/perrow.rs"),
    ];
    let mut n = BTreeSet::new();
    for src in SOURCES {
        scan(src, &mut n);
    }
    n
}

fn scan(src: &str, n: &mut BTreeSet<String>) {
    let b = src.as_bytes();
    let mut i = 0;
    while let Some(q) = src[i..].find('"').map(|q| q + i) {
        let Some(e) = src[q + 1..].find('"').map(|e| e + q + 1) else { break };
        let name = &src[q + 1..e];
        let rest = src[e + 1..].trim_start();
        let ident = !name.is_empty() && name.bytes().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'_') && b[q + 1].is_ascii_lowercase();
        if ident && (rest.starts_with("=>") || rest.starts_with('|')) && crate::kernel_by_name(name).is_some() {
            n.insert(name.to_string());
        }
        i = e + 1;
    }
}

/// Every name of the golden manifest: the registry's literal names, `amdc list`, the dispatch-reachable instantiations and the Python-op
/// instantiations.
pub fn manifest_names() -> BTreeSet<String> {
    let mut n = dispatched_names();
    n.extend(python_op_names());
    n.extend(crate::kernel_names().iter().map(|s| s.to_string()));
    n.extend(registered_literal_names());
    n
}
