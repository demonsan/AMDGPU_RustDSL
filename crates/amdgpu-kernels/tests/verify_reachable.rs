//! Every kernel a launcher can reach (`reachable::manifest_names`: the dispatch rules, the Python ops, the registered names) passes the strict GPU-semantics
//! verification (`amdc gpu-verify --strict`, `gpu_verify::analyse`: IR + MIR) with 0 errors. The only exceptions are the intentional probes and verifier
//! attack kernels (explicit allowlist below). Release only (a few minutes); `python/tests/run_all.sh` runs it. A verifier change that makes a shipped
//! kernel unprovable (e.g. a path cap exceeded) fails here instead of slipping through.

/// Kernels expected to carry verifier errors: the hardware probes and the verifier's own attack cases (their errors are the point).
const ALLOW_PREFIXES: [&str; 2] = ["probe_", "gv_attack_"];
/// Further intentional exceptions by exact name, each with its reason.
const ALLOW_NAMES: [(&str, &str); 1] = [(
    "handoff_probe_cross",
    "hardware probe of convert_layout's cross-lane LDS round trip (tests/gpu/handoff_host.cpp checks it); the interval analysis cannot prove its private-slice bound (GV-TILE-PRIVATE-ESCAPE / GV-LDS-OOB)",
)];

/// Reachable kernels that keep accesses which are bounds-checked but NOT race-checked (`Report::unchecked`: global-write-untracked,
/// global-rw-untracked, lds-untracked), by (kernel, kind), each with its reason. Any other reachable kernel with an untracked access fails:
/// a cross-workgroup race can't hide behind a clean certificate again (the causal_conv1d_fwd state race was such an untracked access).
const ALLOW_UNTRACKED: &[(&str, &str, &str)] = &[
    ("causal_conv1d_fwd_bf16_w4_silu", "global-write-untracked", "the output rows are indexed through cu_seqlens (sequence start + chunk + token): disjoint across workgroups by the host contract (cu_seqlens increasing, chunks of 8 tokens within a sequence) - a data-dependent index the linear analysis cannot see"),
    ("moe_sorting_", "global-write-untracked", "scatter of token ids to their expert-sorted positions (in-kernel prefix sums of the routing data): a data-dependent permutation"),
    ("topk_per_row_", "global-write-untracked", "per-row top-k outputs written at their in-row rank (data-dependent); rows are workgroup digits, the rank within a row is not linear"),
    ("qknorm_rope_kvcache_neox_", "global-write-untracked", "key / value cache rows written at slot_mapping[token] (host contract: distinct slots per token)"),
    ("reshape_cache_blocked_fp8_", "global-write-untracked", "cache rows written at slot_mapping[token] (host contract: distinct slots per token)"),
    ("fmha_fwd_hd128_bf16_v3_ks", "global-write-untracked", "kv-split: z = batch * S + split with S = 1 << lg_s (host contract: the launcher passes lg_s = log2 S; not expressible in the contract language yet), so (batch, split, row, head) is a mixed radix: disjoint (proof form sym_pow2 planned)"),
    ("fmha_fwd_hd128_bf16_v3_causal_ks", "global-write-untracked", "kv-split: z = batch * S + split with S = 1 << lg_s (host contract: the launcher passes lg_s = log2 S; not expressible in the contract language yet), so (batch, split, row, head) is a mixed radix: disjoint (proof form sym_pow2 planned)"),
    ("fmha_fwd_hd192_hd128_bf16_v3_ks", "global-write-untracked", "kv-split: z = batch * S + split with S = 1 << lg_s (host contract: the launcher passes lg_s = log2 S; not expressible in the contract language yet), so (batch, split, row, head) is a mixed radix: disjoint (proof form sym_pow2 planned)"),
    ("fmha_fwd_hd192_hd128_bf16_v3_causal_ks", "global-write-untracked", "kv-split: z = batch * S + split with S = 1 << lg_s (host contract: the launcher passes lg_s = log2 S; not expressible in the contract language yet), so (batch, split, row, head) is a mixed radix: disjoint (proof form sym_pow2 planned)"),
    ("fmha_fwd_hd128_fp8pt_v3_causal", "global-write-untracked", "causal pairing (q tiles x and n - 1 - x per CTA, the wg_union bijection); each half's rows are clipped below S by its own live-row cap, which the radix check cannot merge across the pair"),
    ("fmha_bwd_dq_hd128_bf16_causal", "global-write-untracked", "q tiles processed in reverse (tile = ceil(S/128) - 1 - z, a bijection of axis z); the radix check has no reversed-digit form yet"),
    ("gemm_a16w16_bf16_64_splitk_lc", "global-write-untracked", "counter reset: one counter per (output tile, wave), reset only by the last arriver of the split axis (wg_winner over z); the tile index by * NT + bx is the remap bijection, but by * nt (runtime NT) is a product the radix check does not linearise"),
    ("gemm_a8w8_blockscale_64_splitk_lc", "global-write-untracked", "counter reset: one counter per (output tile, wave), reset only by the last arriver of the split axis (wg_winner over z); by * nt (runtime NT) is a product the radix check does not linearise"),
    ("gemm_a8w8_i8_64_splitk_lc", "global-write-untracked", "counter reset: one counter per (output tile, wave), reset only by the last arriver of the split axis (wg_winner over z); by * nt (runtime NT) is a product the radix check does not linearise"),
    ("gemm_a8w8_rowcol_fp8_64_splitk_lc", "global-write-untracked", "counter reset: one counter per (output tile, wave), reset only by the last arriver of the split axis (wg_winner over z); by * nt (runtime NT) is a product the radix check does not linearise"),
];

#[test]
#[cfg_attr(debug_assertions, ignore = "release only: cargo test --release -p amdgpu-kernels --test verify_reachable (run by python/tests/run_all.sh)")]
fn every_reachable_kernel_verifies_strict() {
    let mut bad: Vec<String> = Vec::new();
    let mut n = 0usize;
    for name in amdgpu_kernels::reachable::manifest_names() {
        if ALLOW_PREFIXES.iter().any(|p| name.starts_with(p)) || ALLOW_NAMES.iter().any(|(n, _)| *n == name) {
            continue;
        }
        let Some(f) = amdgpu_kernels::kernel_by_name(&name) else {
            bad.push(format!("{name}: not registered"));
            continue;
        };
        n += 1;
        match amdgpu_codegen::gpu_verify::analyse(&f) {
            Ok(r) => {
                let errs = r.errors();
                if !errs.is_empty() {
                    let first: Vec<String> = errs.iter().take(3).map(|e| format!("[{}] {}", e.code, e.msg.chars().take(200).collect::<String>())).collect();
                    bad.push(format!("{name}: {} error(s): {}", errs.len(), first.join(" | ")));
                }
                for (kind, n) in &r.unchecked {
                    let allowed = ALLOW_UNTRACKED.iter().any(|(k, kd, _)| (name == *k || (k.ends_with('_') && name.starts_with(k))) && kd == kind);
                    if *n > 0 && !allowed {
                        bad.push(format!("{name}: {n} {kind} access(es) (bounds-checked, NOT race-checked) without an ALLOW_UNTRACKED entry"));
                    }
                }
            }
            Err(e) => bad.push(format!("{name}: does not compile for the MIR checks: {e}")),
        }
    }
    eprintln!("{n} reachable kernels verified (strict), {} failing", bad.len());
    assert!(bad.is_empty(), "kernels failing strict gpu-verify:\n{}", bad.join("\n"));
}