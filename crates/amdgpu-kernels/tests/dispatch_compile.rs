//! Every kernel a launcher can reach must compile: the names the dispatch rules produce over a grid of shapes (decode16, bf16 prefill, MLA,
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
        let s = blockscale_split_k(m, nn, k, 256);
        n.insert(if s > 1 { "gemm_a8w8_blockscale_splitk".into() } else { "gemm_a8w8_blockscale".into() });
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

fn compile_all(names: &BTreeSet<String>) {
    let mut bad = Vec::new();
    for k in names {
        match kernel_by_name(k) {
            None => bad.push(format!("{k}: not in the registry")),
            Some(f) => {
                if let Err(e) = compile_with(&f, &OwnTail) {
                    bad.push(format!("{k}: {}", e.lines().take(3).collect::<Vec<_>>().join(" | ")));
                }
            }
        }
    }
    assert!(bad.is_empty(), "{} of {} reachable kernels do not compile:\n{}", bad.len(), names.len(), bad.join("\n"));
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
