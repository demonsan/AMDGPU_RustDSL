# Performance vs aiter, v12: memory-bound ops, paged decode, MLA decode, GEMV / small-M, conv (sched's sections)

Main `4ec08dac` (2026-11-15), MI355X / gfx950, `mi355-gpu-53`, **GPU 1**. Same conventions as `perf-vs-aiter-v12.md` (kernel-opt's compute-bound families): ratio = **aiter (vendor) time / ours**, above 1 means ours is faster;
the v9.2 column is the previous vendor table (2026-11-09, [`current-status.md`](current-status.md) section 1.1, [`op-coverage-design.md`](op-coverage-design.md) section 7.5, [`op-coverage-membound.md`](op-coverage-membound.md)).
Both sides run in the same session on the same GPU, interleaved per repetition (`tools/perf_ab.sh`: old = new = the same commit, the vendor side right after ours), the same data and the same CPU correctness check where the host has one.

**aiter is not installed as a Python / torch package on the node** (as in v9.2). The vendor numbers come from the standalone vendor harnesses:
- memory-bound: aiter's HIP sources (`csrc/kernels/{activation,quant,rmsnorm_quant,topk_softmax}_kernels.cu`) compiled without torch and linked with [`tests/gpu/vendor_membound_host.cpp`](../tests/gpu/vendor_membound_host.cpp) (`tools/vendor_membound.sh`), which calls aiter's own C++ entry points; the same ~1 GB working sets and the same bandwidth formula (bytes read + written / time) as our hosts;
- paged decode: aiter's `pa_bf16_noquant_gqa{8,16}_1tg_4w.co` driven by `tests/gpu/pa_host.cpp` (`vendor` mode), context lengths random in [ctx / 2, ctx];
- MLA decode: aiter's `mla_dec_stage1_bf16_a16w16_subQ16_mqa16.co` (H = 16 only, page size 1) with **our** split-major combine, driven by `tests/gpu/mla_host.cpp` (`VENDOR=`); it reads a page-size-1 cache, ours a page-64 one (the layouts differ, the logical KV is the same);
- GEMV / small-M: aiter's prebuilt ASM GEMMs through `tests/gpu/vendor_gemm_host.cpp` (`tools/vs_aiter_gemv.py`), the fastest tile of the family per shape.

Tools added for this table: the vendor side of `perf_ab` for `decode`, `mla` and the `membound` cases that have an aiter entry point (`tools/perf_ab_cases.py`, `docs/perf-ab.md`), `tools/vs_aiter_gemv.py`, `tools/vs_aiter_v12_md.py` (this file's generated tables).
Noise: with old = new the same commit, the scatter between the two sides is the noise of the table (typically within +-1 %, a few cases +-2 - 3 %); a ratio within about 3 % of 1.0 is a tie.

## 7. Memory-bound ops

Stream reference measured in the same session (`tests/gpu/membw_survey.cpp`, 1 GiB working set): the best streaming copy (read + write counted) is **6266 GB/s** (read-only 6408 GB/s; `hipMemcpy` D2D 5470 GB/s read + write). "% of stream" below is ours / 6266. Kernels that mix reads and writes in other proportions (read-heavy quantizers, 2:1 silu_mul) can exceed 100 %.

### 7.1 Same-session comparison with aiter (the ops whose aiter entry point is callable without torch)

| op, shape | ours GB/s | aiter GB/s | ratio | v9.2 |
|---|--:|--:|--:|--:|
| silu_mul, M = 43690, D = 4096 | 6579 | 5880 | **1.12** | 1.11 - 1.13 |
| silu_mul, M = 65535, D = 1024 | 6711 | 6120 | **1.10** | 1.07 |
| silu_mul, M = 12483, D = 14336 | 6525 | 5771 | **1.13** | 1.18 |
| gelu_tanh_mul, M = 43690, D = 4096 | 6574 | 5106 | **1.29** | 1.23 - 1.27 |
| rmsnorm, M = 65536, N = 4096 | 6216 | 5857 | **1.06** | 1.01 - 1.02 |
| rmsnorm, M = 262144, N = 1000 | 5806 | 4885 | **1.19** | 1.17 |
| add_rmsnorm, M = 32768, N = 4096 | 5527 | 6966 | 0.79 (**not a valid comparison**) | 0.86 (same caveat) |
| quant_token fp8, M = 131072, N = 4096 | 6428 | 5914 | **1.09** | 1.08 |
| quant_token fp8, M = 524288, N = 1000 | 6108 | 4115 | **1.48** | 1.45 |
| quant_token int8, M = 131072, N = 4096 | 6343 | 6025 | **1.05** | 1.04 - 1.26 |
| quant_token int8, M = 524288, N = 1000 | 5692 | 4126 | **1.38** | (in 1.04 - 1.26) |
| quant_token int8, M = 36000, N = 14336 (long row) | 5831 | 4618 | **1.26** | (in 1.04 - 1.26) |
| quant_group128 fp8, M = 131072, N = 4096 | 6396 | 5898 | **1.08** | 1.10 |
| quant_static fp8, 5.4e8 elements | 6204 | 6012 | **1.03** | 0.97 |
| silu_mul_quant fp8, M = 65536, D = 4096 | 6030 | 5318 | **1.13** | 1.11 |
| topk_softmax, E = 128, K = 8, T = 65536 (us) | **16.7 us** | 20.4 - 21.4 us | **1.25** | 0.99 (21.2 vs 20.9 us) |

Notes:
- **11 of the 12 bandwidth-bound comparisons are ahead by 3 - 48 %**; the 12th, `add_rmsnorm`, is not a comparison: aiter's 6966 GB/s is above the measured stream limit (6266), so its kernel moves fewer bytes than the formula counts (a different `residual_out` / output contract), and it is kept as data only (as in v9.2).
- The topk_softmax row compares **time** (the gate is not bandwidth-bound: 21 MB of traffic; our host and the vendor harness count different bytes). Ours went from 21.2 us (v9.2, LDS round trips) to 16.7 us with the DPP row reductions.
- Not re-run: `rmsnorm_quant` (ours has no row in the matrix) and `topk_softmax_renorm` (no renorm case in the vendor harness); v9.2: 0.94 and 0.87 - 0.92.

### 7.2 Every memory-bound kernel: bandwidth and % of the stream limit (`tools/b2_matrix.sh` cases, one repetition)

| case | ours | % of the stream limit | aiter | ratio (aiter time / ours) |
|---|--:|--:|--:|--:|
| silu_mul d4096 corr+perf | 6592 GB/s | 105 % | 5957 | 1.107 |
| silu_mul d1024 | 6752 GB/s | 108 % | 6306 | 1.071 |
| silu_mul d14336 | 6527 GB/s | 104 % | 5545 | 1.177 |
| gelu_tanh d4096 | 6607 GB/s | 105 % | 5071 | 1.303 |
| silu_mul_limit d4096 (limit 7) | 6588 GB/s | 105 % | - | - |
| swiglu_oai_mul d4096 | 6596 GB/s | 105 % | - | - |
| silu (unary) | 6329 GB/s | 101 % | - | - |
| gelu_tanh (unary) | 6012 GB/s | 96 % | - | - |
| relu2 (unary) | 6596 GB/s | 105 % | - | - |
| sigmoid (unary) | 6577 GB/s | 105 % | - | - |
| add_bf16 | 6279 GB/s | 100 % | - | - |
| cvt_f32_bf16 | 6551 GB/s | 105 % | - | - |
| cvt_bf16_f32 (re-run alone: the full-pass sample, 451, was a one-off glitch) | 6298 GB/s | 101 % | - | - |
| quant_static | 5288 GB/s | 84 % | 5889 | 0.898 |
| rmsnorm n4096 | 6148 GB/s | 98 % | 5806 | 1.059 |
| rmsnorm n1000 | 5742 GB/s | 92 % | 4911 | 1.169 |
| rmsnorm n200 | 5323 GB/s | 85 % | 3176 | 1.676 |
| add_rmsnorm n4096 | 5590 GB/s | 89 % | 6934 | 0.806 |
| layernorm n4096 | 5686 GB/s | 91 % | - | - |
| layernorm n776 | 4629 GB/s | 74 % | - | - |
| softmax n4096 | 6023 GB/s | 96 % | - | - |
| softmax n1000 | 5708 GB/s | 91 % | - | - |
| softmax n8192 | 5478 GB/s | 87 % | - | - |
| quant_token n4096 | 6378 GB/s | 102 % | 5903 | 1.080 |
| quant_token n14336 (long row) | 5986 GB/s | 96 % | 4679 | 1.279 |
| quant_token n32768 (long row) | 4371 GB/s | 70 % | 4396 | 0.994 |
| quant_token_int8 n14336 (long row) | 2857 GB/s | 46 % | 4645 | 0.615 |
| silu_mul_quant d14336 (long row) | 3890 GB/s | 62 % | - | - |
| silu_mul_quant d32768 (long row) | 3370 GB/s | 54 % | - | - |
| situv2_mul_quant d14336 (long row) | 2861 GB/s | 46 % | - | - |
| quant_token n1000 | 6115 GB/s | 98 % | 4171 | 1.466 |
| situv2_mul_quant d4096 | 4527 GB/s | 72 % | - | - |
| quant_group128 n4096 | 6406 GB/s | 102 % | 5847 | 1.096 |
| quant_token_int8 n4096 | 6338 GB/s | 101 % | 5887 | 1.076 |
| quant_token_int8 n1000 | 5637 GB/s | 90 % | 4069 | 1.385 |
| dequant static | 6390 GB/s | 102 % | - | - |
| dequant token n4096 | 6138 GB/s | 98 % | 5878 | 1.044 |
| dequant group128 n4096 | 6164 GB/s | 98 % | 5678 | 1.086 |
| dequant token n1000 (odd tail) | 5244 GB/s | 84 % | 5086 | 1.031 |
| silu_mul_quant d4096 | 5920 GB/s | 94 % | 5346 | 1.107 |
| silu_mul_quant d1000 | 5780 GB/s | 92 % | - | - |
| rope_neox h32 d128 | 6432 GB/s | 103 % | - | - |
| rope_gptj h32 d128 | 6467 GB/s | 103 % | - | - |
| rope_neox h8 d128 | 6840 GB/s | 109 % | - | - |
| cache h8 d128 | 4955 GB/s | 79 % | - | - |
| cache_blocked h8 d128 b16 random slots | 877 GB/s | 14 % | - | - |
| cache_blocked h8 d128 b16 contiguous slots | 1408 GB/s | 22 % | - | - |
| moe_sorting e128 k8 b32 M=37 | 11.85 us | - | - | - |
| moe_sorting e128 k8 b32 M=1001 skew+invalid | 14.53 us | - | - | - |
| moe_sorting e64 k2 b16 M=513 | 10.40 us | - | - | - |
| moe_sorting e256 k8 b128 M=4096 | 51.28 us | - | - | - |
| moe_sorting e385 k6 b32 M=777 | 17.21 us | - | - | - |
| argmax v32000 B=16384 | 4458 GB/s | 71 % | - | - |
| argmax v128256 B=4096 | 4925 GB/s | 79 % | - | - |
| sample v32000 B=16384 | 1308 GB/s | 21 % | - | - |
| sample v128256 B=4096 | 719 GB/s | 11 % | - | - |
| moe_reduce k8 d7168 T=16384 | 6252 GB/s | 100 % | - | - |
| moe_reduce k8 d4096 T=32768 (re-run alone) | 5914 GB/s | 94 % | - | - |
| moe_combine k8 d4096 T=32768 | 5903 GB/s | 94 % | - | - |
| gated_rmsnorm bf16 n128 (heads) | 6152 GB/s | 98 % | - | - |
| gated_rmsnorm fp8 n128 (heads) | 5492 GB/s | 88 % | - | - |
| gated_rmsnorm bf16 n4096 | 6118 GB/s | 98 % | - | - |
| gated_rmsnorm fp8 n4096 | 5351 GB/s | 85 % | - | - |
| cache_fp8 h8 d128 | 5967 GB/s | 95 % | - | - |
| add_rmsnorm_fp8 n4096 | 4273 GB/s | 68 % | - | - |
| add_rmsnorm_fp8 n1000 | 4219 GB/s | 67 % | - | - |
| qknorm_rope neox h40 d128 | 4475 GB/s | 71 % | - | - |
| conv1d_update w4 s3 idx silu | 3148 GB/s | 50 % | - | - |
| moe_fused_gate e256 g8 tg4 k8 M=131072 | 738 GB/s | 12 % | - | - |
| moe_fused_gate e256 g8 tg4 k8 M=1001 (odd) | 67 GB/s | 1 % | - | - |
| topk_per_row decode k2048 B=256 L<=64K | 353 GB/s | 6 % | - | - |
| topk_per_row decode k2048 odd (ragged, ties) | 35 GB/s | 1 % | - | - |
| topk_per_row prefill k128 (ties) | 26 GB/s | 0 % | - | - |
| conv1d_fwd w4 silu B=128 L=512 uniform | 3412 GB/s | 54 % | - | - |
| conv1d_fwd split w4 silu B=128 L=512 uniform | 3204 GB/s | 51 % | - | - |
| conv1d_fwd w4 silu ragged B=256 (grid) | 1283 GB/s | 20 % | - | - |
| conv1d_fwd w4 silu ragged B=256 (_sched) | 3276 GB/s | 52 % | - | - |
| conv1d_fwd split w4 silu ragged + states | 400 GB/s | 6 % | - | - |
| cache_blocked fp8 per-token h8 d128 b16 (i64 slots) | 955 GB/s | 15 % | - | - |
| qknorm_rope_kvcache neox bf16 h8 d128 | 6442 GB/s | 103 % | - | - |
| qknorm_rope_kvcache neox fp8 h8 d128 (i64) | 5470 GB/s | 87 % | - | - |
| rmsnorm_quant_mxfp4 k4096 M=32768 | 2811 GB/s | 45 % | - | - |
| mla cache flat bf16 r512 p64 | 4301 GB/s | 69 % | - | - |
| mla cache seg fp8 r512 p64 b64 (i64) | 3487 GB/s | 56 % | - | - |
| silu_mul_bias d4096 E=64 | 9718 GB/s | 155 % | - | - |
| gelu_mul_bias d2048 E=64 | 5370 GB/s | 86 % | - | - |
| gelu (erf) unary | 3297 GB/s | 53 % | - | - |
| gelu_mul (erf) d4096 | 4319 GB/s | 69 % | - | - |
| mxfp6 e2m3 k4096 M=32768 | 2933 GB/s | 47 % | - | - |
| mxfp6 e3m2 k4096 M=32768 | 2934 GB/s | 47 % | - | - |
| topk e128 k8 T=65536 | 1252 GB/s | 20 % | 1000 | 1.252 |
| topk renorm e256 k8 T=65536 | 1049 GB/s | 17 % | - | - |

Where aiter has no standalone-callable kernel (the table above is the whole of what can be compared; see the have / missing / why table in [`op-coverage-membound.md`](op-coverage-membound.md)): layernorm (a torch extension), row softmax, RoPE / KV-cache writes (`pos_encoding_kernels.cu` and
`cache_kernels.cu` need torch conventions, and `cache_kernels.cu` does not compile standalone on gfx950), dequant, cast, MoE sorting / reduce / combine / gate, sampling / top-k / top-p, fused qk-norm RoPE KV, mxfp4 / mxfp6 quant: those are measured against the stream limit only.

## 8. Paged decode (`pa_decode16_bf16_hd128`, page 16, bf16 KV, D = 128)

`dispatch decode16`: the fused form `_f` when B x HK >= 256, else the last-arriver form `_c` (S = the smallest power of two with B x HK x S >= 256, at least 512 keys per split). 5 repetitions per case.

| shape (B, H, HK, ctx) | form (S) | ours us | aiter us | ratio | v9.2 |
|---|---|--:|--:|--:|--:|
| B1 H32 HK4 ctx32768 | `_c` (64) | 111.84 | 159.23 | **1.42** | |
| B1 H64 HK8 ctx8192 | `_c` (16) | 29.21 | 35.39 | **1.21** | |
| B8 H64 HK8 ctx4096 | `_c` (4) | 26.93 | 25.32 | 0.94 | |
| B16 H64 HK8 ctx2048 | `_c` (2) | 24.36 | 18.41 | 0.76 | 0.77 |
| B32 H64 HK8 ctx2048 | `_f` | 35.86 | 30.80 | 0.86 | |
| B64 H64 HK8 ctx2048 | `_f` | 83.84 | 77.43 | 0.92 | 0.92 |
| B64 H32 HK4 ctx1024 | `_f` | 19.90 | 17.22 | 0.87 | |
| B4 H16 HK1 ctx16384 | `_c` (32) | 55.67 | 86.10 | **1.55** | |

Range now **0.76 - 1.55** (v9.2: 0.67 - 1.24). The two rows that exist in both tables are unchanged (0.76 / 0.77, 0.92 / 0.92). Small batch and long context (the `_c` grids) beat aiter by 20 - 55 %; medium batch at short context (B16 ctx 2048, B32 / B64 at ctx <= 2048) stays 8 - 24 % behind: the per-wave instruction floor of the fused form ([`pa-decode16-profile.md`](pa-decode16-profile.md)), not memory.

## 9. MLA decode (H = 16 against aiter; H >= 32 has no aiter kernel on this path)

`dispatch mla`: the per-wave form (`mla_decode_bf16_p64`) or the head-parallel form (`mla_decode_bf16_hp_p64`, the 4 waves of a CTA are 4 groups of 16 heads sharing one KV ring); S as listed. aiter = `mla_dec_stage1` gqa16 with our combine, run with the same S, page size 1.
5 repetitions per case.

| shape (B, H, ctx) | ours form (S) | ours us | aiter us | ratio | v9.2 |
|---|---|--:|--:|--:|--:|
| B1 H16 ctx2048 | head-parallel (16) | 23.68 | 21.79 | 0.92 | |
| B2 H16 ctx3072 | head-parallel (16) | 29.52 | 20.71 | 0.70 | |
| B4 H16 ctx4096 | head-parallel (16) | 35.26 | 25.92 | 0.74 | |
| B4 H16 ctx8192 | per-wave (16) | 46.27 | 25.89 | 0.56 | |
| B4 H16 ctx16384 | per-wave (16) | 66.01 | 38.56 | 0.58 | |
| B16 H16 ctx1024 | head-parallel (16) | 22.07 | 23.55 | **1.07** | |
| B32 H16 ctx2048 | per-wave (8) | 37.67 | 25.29 | 0.67 | |
| B64 H16 ctx4096 | per-wave (4) | 67.85 | 47.95 | 0.71 | 0.63 |
| B128 H16 ctx4096 | per-wave (2) | 124.02 | 102.71 | 0.83 | 0.24 (older stage-1-only comparison, a 2026-11-08 kernel) |

Ours alone, no aiter kernel in the harness (H > 16): B1 H32 ctx2048 25.25 us (head-parallel, 16), B8 H32 ctx4096 37.99 us (hp, 16), B12 H48 ctx2048 35.70 us (hp, 16), B1 H128 ctx8192 50.38 us (hp, 16), B8 H128 ctx4096 49.77 us (hp, 16).

MLA decode H16 is **0.56 - 1.07, geomean about 0.74** (v9.2: ~0.63 at B64 ctx4096, now 0.71). It is still the weakest decode family: long contexts at small batch (ctx 8192 - 16384, B4) are 0.56 - 0.58: the per-wave form does not fill the machine there and aiter's stage 1 reaches higher KV bandwidth.

## 10. GEMV / small-M GEMM

Ours = the kernel `amdc dispatch gemm` picks (the `_64` / `_64_splitk` kernels with S = 4 or 8, plus the reduce); aiter = the fastest code object of the family directory that the host drives (int8: `I8gemm_..._{16..160}x128`; fp8 blockscale: `fp8gemm_..._{32..128}x128`;
bf16: only the non-split `bf16gemm_..._256x256_bpreshuffle` the host supports); fp8 per-token x per-channel is compared with the int8 per-token kernels (same layout; there is no fp8 per-token aiter ASM kernel in `hsa`). 3 runs of 200 iterations per side, all `bad = 0`.
These shapes are weight-streaming bound (K x N bytes at about 5 - 6 TB/s: 4096^2 fp8 = 16.8 MB = 3 us, plus the fixed cost per call of about 5 - 8 us).


**int8 per-token** (ours us / aiter us = ratio; N = K for the three columns; aiter = the fastest tile of its family):

| M | 4096 | 7168 | 8192 | 128 x 4096 x 8192 (v9.2 row) |
|--:|--:|--:|--:|--:|
| 1 | 6.08 / 6.42 = **1.06** | 11.58 / 15.13 = **1.31** | 16.56 / 17.13 = **1.03** | |
| 4 | 6.36 / 6.54 = **1.03** | 11.68 / 15.20 = **1.30** | 16.84 / 17.83 = **1.06** | |
| 16 | 6.78 / 6.72 = **0.99** | 11.88 / 15.46 = **1.30** | 16.52 / 18.33 = **1.11** | |
| 32 | 7.86 / 13.20 = **1.68** | 12.74 / 15.44 = **1.21** | 17.26 / 18.47 = **1.07** | |
| 64 | 9.39 / 7.05 = **0.75** | 14.57 / 15.83 = **1.09** | 18.60 / 19.35 = **1.04** | |
| 128 (M = 128, 4096 x 8192) | | | | 13.96 / 17.83 = **1.28** (v9.2 1.27) |
| (v9.2 row 64 x 8192 x 8192: now 1.04, v9.2 1.62) | | | | |

**fp8 per-token x per-channel (vs the int8 kernels)** (ours us / aiter us = ratio; N = K for the three columns; aiter = the fastest tile of its family):

| M | 4096 | 7168 | 8192 | 128 x 4096 x 8192 (v9.2 row) |
|--:|--:|--:|--:|--:|
| 1 | 6.01 / 6.42 = **1.07** | 11.44 / 15.13 = **1.32** | 16.37 / 17.18 = **1.05** | |
| 4 | 6.15 / 6.54 = **1.06** | 11.44 / 15.23 = **1.33** | 16.50 / 17.86 = **1.08** | |
| 16 | 6.61 / 6.72 = **1.02** | 11.71 / 15.45 = **1.32** | 16.51 / 18.33 = **1.11** | |
| 32 | 7.67 / 6.83 = **0.89** | 12.54 / 15.46 = **1.23** | 17.11 / 18.44 = **1.08** | |
| 64 | 9.21 / 7.04 = **0.76** | 14.47 / 15.69 = **1.08** | 18.51 / 19.35 = **1.05** | |
| 128 (M = 128, 4096 x 8192) | | | | 13.60 / 17.85 = **1.31** (v9.2 1.87) |
| (v9.2 row 64 x 8192 x 8192: now 1.05, v9.2 1.62) | | | | |

**fp8 blockscale** (ours us / aiter us = ratio; N = K for the three columns; aiter = the fastest tile of its family):

| M | 4096 | 7168 | 8192 | 128 x 4096 x 8192 (v9.2 row) |
|--:|--:|--:|--:|--:|
| 1 | 19.27 / 8.85 = **0.46** | 11.80 / 17.81 = **1.51** | 17.21 / 20.55 = **1.19** | |
| 4 | 8.05 / 8.98 = **1.12** | 12.01 / 19.09 = **1.59** | 16.97 / 21.93 = **1.29** | |
| 16 | 8.25 / 9.18 = **1.11** | 12.19 / 19.57 = **1.61** | 17.04 / 22.30 = **1.31** | |
| 32 | 8.93 / 9.43 = **1.06** | 13.11 / 19.89 = **1.52** | 17.50 / 22.64 = **1.29** | |
| 64 | 10.49 / 9.69 = **0.92** | 14.61 / 20.15 = **1.38** | 19.06 / 22.81 = **1.20** | |
| 128 (M = 128, 4096 x 8192) | | | | 15.59 / 22.14 = **1.42** (v9.2 1.45) |
| (v9.2 row 64 x 8192 x 8192: now 1.20, v9.2 1.57) | | | | |

**bf16 (aiter: non-split 256x256 only)** (ours us / aiter us = ratio; N = K for the three columns; aiter = the fastest tile of its family):

| M | 4096 | 7168 | 8192 | 128 x 4096 x 8192 (v9.2 row) |
|--:|--:|--:|--:|--:|
| 1 | 7.36 / 58.96 = **8.01** | 18.12 / 100.52 = **5.55** | 24.85 / 114.48 = **4.61** | |
| 4 | 7.44 / 58.93 = **7.92** | 18.43 / 100.50 = **5.45** | 24.01 / 114.85 = **4.78** | |
| 16 | 7.99 / 59.00 = **7.38** | 18.65 / 100.36 = **5.38** | 24.42 / 115.05 = **4.71** | |
| 32 | 10.20 / 59.35 = **5.82** | 19.59 / 100.61 = **5.14** | 25.07 / 115.16 = **4.59** | |
| 64 | 11.76 / 59.67 = **5.07** | 21.52 / 101.42 = **4.71** | 26.55 / 115.21 = **4.34** | |
| 128 (M = 128, 4096 x 8192) | | | | 19.68 / 116.35 = **5.91** (v9.2 3.90) |
| (v9.2 row 64 x 8192 x 8192: now 4.34, v9.2 3.60) | | | | |

Notes:
- **int8 / fp8 per-token at N = K = 4096, M = 32 and 64**: aiter's 16x128 tile reaches 6.8 - 7.0 us where ours (S = 4 split-K plus reduce) needs 7.7 / 9.2 us, i.e. **0.87 - 0.89 at M = 32 and 0.75 at M = 64**. The int8 M = 32 cell shows 1.68 only because the host rejected aiter's 16x128 result for that data (`bad != 0`, so the fallback tile 80x128 at 13.2 us was reported); the fp8 kernels, which use the same aiter code object, show the true 0.89.
- **fp8 blockscale 1 x 4096 x 4096 shows 0.46** (ours 19.3 us) in the best-of-tiles tool, which is a first-call effect of that run (the same case in the `perf_ab` gemv family, 3 interleaved repetitions, gives **7.85 us, ratio 1.13**); the other M at 4096 are 1.06 - 1.12 (0.92 at M = 64).
- **v9.2 rows**: 128 x 4096 x 8192: int8 1.28 (v9.2 1.27 in kernel-opt's table; 1.62 / 1.87 in the 2026-11-08 split-K table, which compared with the non-tuned 128x128 tile), fp8 blockscale 1.42 (1.45); 64 x 8192 x 8192: 1.04 / 1.05 / 1.20 (the 2026-11-08 values 1.62 / 1.62 / 1.57 used aiter's 256-class tiles; the best small tile is closer).
- **bf16**: the 4.3 - 8.0x is **not like-for-like**: aiter's split-K bf16 kernels (`bf16gemm_fp32bf16_*_splitk`, fp32 partials plus their own reduce) are not driven by the host, so the bf16 column compares ours with a 256x256 kernel built for M >= 256.
- On balance, over the 48 int8 / fp8 cells with an aiter value: **40 ahead by more than 3 %**, 3 ties (within 3 %), 5 behind: int8 per-token 64x4096x4096 (0.75), fp8 per-token x per-channel 32x4096x4096 (0.89), fp8 per-token x per-channel 64x4096x4096 (0.76), fp8 blockscale 1x4096x4096 (0.46), fp8 blockscale 64x4096x4096 (0.92). Range 0.46 - 1.68.

## 11. Convolution

aiter has **no convolution GEMM** (no 2-D / 3-D conv kernel in its `hsa` / `csrc`; the v9.2 design note says the same: "FlyDSL not runnable"), so there is no vendor side. Ours (implicit GEMM on the v4 bf16 GEMM structure, `perf_ab conv`, 3 repetitions):

| kernel | shape | ours TF/s |
|---|---|--:|
| `conv2d_bf16_r1s1_st1_p0_d1` | N8 56x56, C256 -> K512, 1x1 | 354.7 |
| `conv2d_bf16_r3s3_st1_p1_d1` | N8 56x56, C256 -> K256, 3x3 | 402.4 |
| `conv3d_bf16_t3r3s3_st1_p1_d1` | N2 16x28x28, C128 -> K256, 3x3x3 | 420.0 |

The small shapes in `docs/op-coverage-design.md` section on conv use the `_c128` tile (`dispatch conv_cta`). aiter does have `causal_conv1d_update` / `causal_conv1d_fwd_split_qkv` (the 1-D causal convolution of the SSM layers, in the memory-bound family above): those are **not compared** here (the standalone harness would need their tensor layouts; ours: [`op-coverage-membound.md`](op-coverage-membound.md), "causal_conv1d").

## 12. Summary

| family | ours vs aiter now (ratio) | v9.2 |
|---|---|---|
| elementwise / norm / quant (bandwidth-bound, 15 comparable cases) | **1.03 - 1.48**, 100 % of the comparable cases ahead except the invalid `add_rmsnorm`; 90 - 107 % of the stream limit | 1.07 - 1.45, 90 - 103 % |
| topk_softmax (E = 128) | 1.25 | 0.99 |
| paged decode16 (incl. `_c`) | 0.76 - 1.55 | 0.67 - 1.24 |
| MLA decode H16 | 0.56 - 1.07 (geomean about 0.74) | ~0.63 |
| GEMV / small-M int8, fp8 per-token, fp8 blockscale | 0.75 - 1.68 (mostly 1.0 - 1.6) | 1.45 - 1.87 (split-K rows) |
| GEMV bf16 | not like-for-like (aiter split-K not driven) | 3.6 - 3.9 (same caveat) |
| conv | no aiter kernel | - |

