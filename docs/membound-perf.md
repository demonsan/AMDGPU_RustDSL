# Memory-bound perf matrix

Median of 3 run(s) of `tools/b2_matrix.sh` (one GPU, same node, same session); the baseline is `docs/membound_baseline.tsv`; "% stream" is against 6500 GB/s (the measured `nt` stream copy). Cases that count bytes read from cache several times (the MoE bias rows) or are latency-bound small problems can sit far from the stream line by construction; the point of the table is the delta column: a drop of more than 10 % against the baseline is flagged (node noise is +-3-10 %).

| case | GB/s | % stream | baseline | delta | status |
|---|---:|---:|---:|---:|---|
| silu_mul d4096 corr+perf | 6574 | 101 % | 6566 | +0.1 % | ok |
| silu_mul d1024 | 6722 | 103 % | 6734 | -0.2 % | ok |
| silu_mul d14336 | 6530 | 100 % | 6527 | +0.1 % | ok |
| silu_mul d24 corr | 25 | 0 % | 25 | -0.8 % | ok |
| gelu_tanh d4096 | 6475 | 100 % | 6396 | +1.2 % | ok |
| silu_mul_limit d4096 (limit 7) | 6570 | 101 % | 6570 | +0.0 % | ok |
| swiglu_oai_mul d4096 | 6510 | 100 % | 6556 | -0.7 % | ok |
| silu (unary) | 6146 | 95 % | 6120 | +0.4 % | ok |
| gelu_tanh (unary) | 5554 | 85 % | 5650 | -1.7 % | ok |
| relu2 (unary) | 6505 | 100 % | 6428 | +1.2 % | ok |
| sigmoid (unary) | 6340 | 98 % | 6422 | -1.3 % | ok |
| add_bf16 | 6467 | 99 % | 6542 | -1.1 % | ok |
| cvt_f32_bf16 | 6602 | 102 % | 6624 | -0.3 % | ok |
| cvt_bf16_f32 | 6323 | 97 % | 6332 | -0.1 % | ok |
| quant_static | 6129 | 94 % | 6121 | +0.1 % | ok |
| rmsnorm n4096 | 6158 | 95 % | 6124 | +0.6 % | ok |
| rmsnorm n1000 | 5709 | 88 % | 5722 | -0.2 % | ok |
| rmsnorm n200 | 5307 | 82 % | 5159 | +2.9 % | ok |
| add_rmsnorm n4096 | 5758 | 89 % | 5703 | +1.0 % | ok |
| layernorm n4096 | 5539 | 85 % | 5506 | +0.6 % | ok |
| layernorm n776 | 4494 | 69 % | 4495 | -0.0 % | ok |
| softmax n4096 | 5983 | 92 % | 5982 | +0.0 % | ok |
| softmax n1000 | 5562 | 86 % | 5548 | +0.3 % | ok |
| softmax n8192 | 5518 | 85 % | 5498 | +0.4 % | ok |
| quant_token n4096 | 6387 | 98 % | 6360 | +0.4 % | ok |
| quant_token n1000 | 5907 | 91 % | 5902 | +0.1 % | ok |
| quant_group128 n4096 | 6456 | 99 % | 6428 | +0.4 % | ok |
| quant_group128 n640 corr | 13 | 0 % | 13 | +0.0 % | ok |
| quant_token_int8 n4096 | 6216 | 96 % | 6202 | +0.2 % | ok |
| quant_token_int8 n1000 | 5275 | 81 % | 5292 | -0.3 % | ok |
| dequant static | 6397 | 98 % | 6389 | +0.1 % | ok |
| dequant token n4096 | 6006 | 92 % | 6007 | -0.0 % | ok |
| dequant group128 n4096 | 5974 | 92 % | 5952 | +0.4 % | ok |
| dequant token n1000 (odd tail) | 5276 | 81 % | 5309 | -0.6 % | ok |
| silu_mul_quant d4096 | 6005 | 92 % | 5884 | +2.1 % | ok |
| silu_mul_quant d1000 | 5861 | 90 % | 5807 | +0.9 % | ok |
| rope_neox h32 d128 | 6413 | 99 % | 6426 | -0.2 % | ok |
| rope_gptj h32 d128 | 6352 | 98 % | 6324 | +0.4 % | ok |
| rope_neox h8 d128 | 6798 | 105 % | 6780 | +0.3 % | ok |
| rope_neox h40 d64 corr | 61 | 1 % | 61 | -0.8 % | ok |
| cache h8 d128 | 5010 | 77 % | 5047 | -0.7 % | ok |
| cache h5 d64 corr | 62 | 1 % | 62 | -0.8 % | ok |
| cache_blocked h8 d128 b16 random slots | 867 | 13 % | 868 | -0.1 % | ok |
| cache_blocked h8 d128 b16 contiguous slots | 1416 | 22 % | 1420 | -0.2 % | ok |
| cache_blocked h8 d64 b64 corr | 105 | 2 % | 107 | -1.3 % | ok |
| moe_sorting e128 k8 b32 M=37 | - | - | - | - | ok |
| moe_sorting e128 k8 b32 M=1001 skew+invalid | - | - | - | - | ok |
| moe_sorting e64 k2 b16 M=513 | - | - | - | - | ok |
| moe_sorting e256 k8 b128 M=4096 | - | - | - | - | ok |
| moe_sorting e385 k6 b32 M=777 | - | - | - | - | ok |
| argmax v32000 B=16384 | 3831 | 59 % | 3848 | -0.5 % | ok |
| argmax v128256 B=4096 | 4391 | 68 % | 4367 | +0.5 % | ok |
| sample v32000 B=16384 | 1251 | 19 % | 1251 | -0.0 % | ok |
| sample v128256 B=4096 | 721 | 11 % | 722 | -0.1 % | ok |
| sample v1000 corr | 6 | 0 % | 6 | +0.0 % | ok |
| moe_reduce k8 d7168 T=16384 | 5967 | 92 % | 6028 | -1.0 % | ok |
| moe_reduce k8 d4096 T=32768 | 5904 | 91 % | 5900 | +0.1 % | ok |
| moe_combine k8 d4096 T=32768 | 5908 | 91 % | 5963 | -0.9 % | ok |
| moe_reduce k6 d2048 corr | 1592 | 24 % | 1588 | +0.3 % | ok |
| gated_rmsnorm bf16 n128 (heads) | 6178 | 95 % | 6143 | +0.6 % | ok |
| gated_rmsnorm fp8 n128 (heads) | 5618 | 86 % | 5627 | -0.2 % | ok |
| gated_rmsnorm bf16 n4096 | 6024 | 93 % | 5873 | +2.6 % | ok |
| gated_rmsnorm fp8 n4096 | 5293 | 81 % | 5219 | +1.4 % | ok |
| cache_fp8 h8 d128 | 5969 | 92 % | 5963 | +0.1 % | ok |
| add_rmsnorm_fp8 n4096 | 4241 | 65 % | 4266 | -0.6 % | ok |
| add_rmsnorm_fp8 n1000 | 3854 | 59 % | 3854 | +0.0 % | ok |
| qknorm_rope neox h40 d128 | 4238 | 65 % | 4230 | +0.2 % | ok |
| conv1d_update w4 s3 idx silu | 3129 | 48 % | 3068 | +2.0 % | ok |
| moe_fused_gate e256 g8 tg4 k8 M=131072 | 638 | 10 % | 638 | +0.0 % | ok |
| moe_fused_gate e256 g8 tg4 k8 M=1001 (odd) | 58 | 1 % | 58 | +0.0 % | ok |
| topk_per_row decode k2048 B=256 L<=64K | 324 | 5 % | 324 | +0.2 % | ok |
| topk_per_row decode k2048 odd (ragged, ties) | 33 | 1 % | 33 | +0.3 % | ok |
| topk_per_row prefill k128 (ties) | 25 | 0 % | 25 | -0.4 % | ok |
| conv1d_fwd w4 silu B=128 L=512 uniform | 2798 | 43 % | 2787 | +0.4 % | ok |
| conv1d_fwd split w4 silu B=128 L=512 uniform | 2594 | 40 % | 2615 | -0.8 % | ok |
| conv1d_fwd w4 silu ragged B=256 (grid) | 997 | 15 % | 996 | +0.1 % | ok |
| conv1d_fwd w4 silu ragged B=256 (_sched) | 2663 | 41 % | 2657 | +0.2 % | ok |
| conv1d_fwd split w4 silu ragged + states | 399 | 6 % | 398 | +0.3 % | ok |
| cache_blocked fp8 per-token h8 d128 b16 (i64 slots) | 930 | 14 % | 920 | +1.1 % | ok |
| qknorm_rope_kvcache neox bf16 h8 d128 | 5875 | 90 % | 5830 | +0.8 % | ok |
| qknorm_rope_kvcache neox fp8 h8 d128 (i64) | 4767 | 73 % | 4727 | +0.8 % | ok |
| rmsnorm_quant_mxfp4 k4096 M=32768 | 2797 | 43 % | 2791 | +0.2 % | ok |
| mla cache flat bf16 r512 p64 | 4293 | 66 % | 4271 | +0.5 % | ok |
| mla cache seg fp8 r512 p64 b64 (i64) | 3046 | 47 % | 3058 | -0.4 % | ok |
| silu_mul_bias d4096 E=64 | 9120 | 140 % | 9315 | -2.1 % | ok |
| gelu_mul_bias d2048 E=64 | 5090 | 78 % | 5107 | -0.3 % | ok |
| gelu (erf) unary | 3056 | 47 % | 3058 | -0.1 % | ok |
| gelu_mul (erf) d4096 | 4113 | 63 % | 4118 | -0.1 % | ok |
| mxfp6 e2m3 k4096 M=32768 | 2905 | 45 % | 2921 | -0.5 % | ok |
| mxfp6 e3m2 k4096 M=32768 | 2905 | 45 % | 2916 | -0.4 % | ok |
| topk e128 k8 T=65536 | 1176 | 18 % | 1176 | -0.0 % | ok |
| topk renorm e256 k8 T=65536 | 1004 | 15 % | 1002 | +0.2 % | ok |

92 cases, 0 correctness failure(s), 0 regression(s).
