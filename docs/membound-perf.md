# Memory-bound perf matrix

Median of 1 run(s) of `tools/b2_matrix.sh` (one GPU, same node, same session); the baseline is `docs/membound_baseline.tsv`; "% stream" is against 6500 GB/s (the measured `nt` stream copy). Cases that count bytes read from cache several times (the MoE bias rows) or are latency-bound small problems can sit far from the stream line by construction; the point of the table is the delta column: a drop of more than 10 % against the baseline is flagged (node noise is +-3-10 %).

| case | GB/s | % stream | baseline | delta | status |
|---|---:|---:|---:|---:|---|
| silu_mul d4096 corr+perf | 6566 | 101 % | 6566 | +0.0 % | ok |
| silu_mul d1024 | 6734 | 104 % | 6734 | +0.0 % | ok |
| silu_mul d14336 | 6527 | 100 % | 6527 | +0.0 % | ok |
| silu_mul d24 corr | 25 | 0 % | 25 | +0.0 % | ok |
| gelu_tanh d4096 | 6396 | 98 % | 6396 | +0.0 % | ok |
| silu_mul_limit d4096 (limit 7) | 6570 | 101 % | 6570 | +0.0 % | ok |
| swiglu_oai_mul d4096 | 6556 | 101 % | 6556 | +0.0 % | ok |
| silu (unary) | 6120 | 94 % | 6120 | +0.0 % | ok |
| gelu_tanh (unary) | 5650 | 87 % | 5650 | +0.0 % | ok |
| relu2 (unary) | 6428 | 99 % | 6428 | +0.0 % | ok |
| sigmoid (unary) | 6422 | 99 % | 6422 | +0.0 % | ok |
| add_bf16 | 6542 | 101 % | 6542 | +0.0 % | ok |
| cvt_f32_bf16 | 6624 | 102 % | 6624 | +0.0 % | ok |
| cvt_bf16_f32 | 6332 | 97 % | 6332 | +0.0 % | ok |
| quant_static | 6121 | 94 % | 6121 | +0.0 % | ok |
| rmsnorm n4096 | 6124 | 94 % | 6124 | +0.0 % | ok |
| rmsnorm n1000 | 5722 | 88 % | 5722 | +0.0 % | ok |
| rmsnorm n200 | 5159 | 79 % | 5159 | +0.0 % | ok |
| add_rmsnorm n4096 | 5703 | 88 % | 5703 | +0.0 % | ok |
| layernorm n4096 | 5506 | 85 % | 5506 | +0.0 % | ok |
| layernorm n776 | 4495 | 69 % | 4495 | +0.0 % | ok |
| softmax n4096 | 5982 | 92 % | 5982 | +0.0 % | ok |
| softmax n1000 | 5548 | 85 % | 5548 | +0.0 % | ok |
| softmax n8192 | 5498 | 85 % | 5498 | +0.0 % | ok |
| quant_token n4096 | 6360 | 98 % | 6360 | +0.0 % | ok |
| quant_token n1000 | 5902 | 91 % | 5902 | +0.0 % | ok |
| quant_group128 n4096 | 6428 | 99 % | 6428 | +0.0 % | ok |
| quant_group128 n640 corr | 13 | 0 % | 13 | +0.0 % | ok |
| quant_token_int8 n4096 | 6202 | 95 % | 6202 | +0.0 % | ok |
| quant_token_int8 n1000 | 5292 | 81 % | 5292 | +0.0 % | ok |
| dequant static | 6389 | 98 % | 6389 | +0.0 % | ok |
| dequant token n4096 | 6007 | 92 % | 6007 | +0.0 % | ok |
| dequant group128 n4096 | 5952 | 92 % | 5952 | +0.0 % | ok |
| dequant token n1000 (odd tail) | 5309 | 82 % | 5309 | +0.0 % | ok |
| silu_mul_quant d4096 | 5884 | 91 % | 5884 | +0.0 % | ok |
| silu_mul_quant d1000 | 5807 | 89 % | 5807 | +0.0 % | ok |
| rope_neox h32 d128 | 6426 | 99 % | 6426 | +0.0 % | ok |
| rope_gptj h32 d128 | 6324 | 97 % | 6324 | +0.0 % | ok |
| rope_neox h8 d128 | 6780 | 104 % | 6780 | +0.0 % | ok |
| rope_neox h40 d64 corr | 61 | 1 % | 61 | +0.0 % | ok |
| cache h8 d128 | 5047 | 78 % | 5047 | +0.0 % | ok |
| cache h5 d64 corr | 62 | 1 % | 62 | +0.0 % | ok |
| cache_blocked h8 d128 b16 random slots | 868 | 13 % | 868 | +0.0 % | ok |
| cache_blocked h8 d128 b16 contiguous slots | 1420 | 22 % | 1420 | +0.0 % | ok |
| cache_blocked h8 d64 b64 corr | 107 | 2 % | 107 | +0.0 % | ok |
| moe_sorting e128 k8 b32 M=37 | - | - | - | - | ok |
| moe_sorting e128 k8 b32 M=1001 skew+invalid | - | - | - | - | ok |
| moe_sorting e64 k2 b16 M=513 | - | - | - | - | ok |
| moe_sorting e256 k8 b128 M=4096 | - | - | - | - | ok |
| moe_sorting e385 k6 b32 M=777 | - | - | - | - | ok |
| argmax v32000 B=16384 | 3848 | 59 % | 3848 | +0.0 % | ok |
| argmax v128256 B=4096 | 4367 | 67 % | 4367 | +0.0 % | ok |
| sample v32000 B=16384 | 1251 | 19 % | 1251 | +0.0 % | ok |
| sample v128256 B=4096 | 722 | 11 % | 722 | +0.0 % | ok |
| sample v1000 corr | 6 | 0 % | 6 | +0.0 % | ok |
| moe_reduce k8 d7168 T=16384 | 6028 | 93 % | 6028 | +0.0 % | ok |
| moe_reduce k8 d4096 T=32768 | 5900 | 91 % | 5900 | +0.0 % | ok |
| moe_combine k8 d4096 T=32768 | 5963 | 92 % | 5963 | +0.0 % | ok |
| moe_reduce k6 d2048 corr | 1588 | 24 % | 1588 | +0.0 % | ok |
| gated_rmsnorm bf16 n128 (heads) | 6143 | 95 % | 6143 | +0.0 % | ok |
| gated_rmsnorm fp8 n128 (heads) | 5627 | 87 % | 5627 | +0.0 % | ok |
| gated_rmsnorm bf16 n4096 | 5873 | 90 % | 5873 | +0.0 % | ok |
| gated_rmsnorm fp8 n4096 | 5219 | 80 % | 5219 | +0.0 % | ok |
| cache_fp8 h8 d128 | 5963 | 92 % | 5963 | +0.0 % | ok |
| add_rmsnorm_fp8 n4096 | 4266 | 66 % | 4266 | +0.0 % | ok |
| add_rmsnorm_fp8 n1000 | 3854 | 59 % | 3854 | +0.0 % | ok |
| qknorm_rope neox h40 d128 | 4230 | 65 % | 4230 | +0.0 % | ok |
| conv1d_update w4 s3 idx silu | 3068 | 47 % | 3068 | +0.0 % | ok |
| moe_fused_gate e256 g8 tg4 k8 M=131072 | 638 | 10 % | 638 | +0.0 % | ok |
| moe_fused_gate e256 g8 tg4 k8 M=1001 (odd) | 58 | 1 % | 58 | +0.0 % | ok |
| topk_per_row decode k2048 B=256 L<=64K | 324 | 5 % | 324 | +0.0 % | ok |
| topk_per_row decode k2048 odd (ragged, ties) | 33 | 1 % | 33 | +0.0 % | ok |
| topk_per_row prefill k128 (ties) | 25 | 0 % | 25 | +0.0 % | ok |
| conv1d_fwd w4 silu B=128 L=512 uniform | 2787 | 43 % | 2787 | +0.0 % | ok |
| conv1d_fwd split w4 silu B=128 L=512 uniform | 2615 | 40 % | 2615 | +0.0 % | ok |
| conv1d_fwd w4 silu ragged B=256 (grid) | 996 | 15 % | 996 | +0.0 % | ok |
| conv1d_fwd w4 silu ragged B=256 (_sched) | 2657 | 41 % | 2657 | +0.0 % | ok |
| conv1d_fwd split w4 silu ragged + states | 398 | 6 % | 398 | +0.0 % | ok |
| cache_blocked fp8 per-token h8 d128 b16 (i64 slots) | 920 | 14 % | 920 | +0.0 % | ok |
| qknorm_rope_kvcache neox bf16 h8 d128 | 5830 | 90 % | 5830 | +0.0 % | ok |
| qknorm_rope_kvcache neox fp8 h8 d128 (i64) | 4727 | 73 % | 4727 | +0.0 % | ok |
| rmsnorm_quant_mxfp4 k4096 M=32768 | 2791 | 43 % | 2791 | +0.0 % | ok |
| mla cache flat bf16 r512 p64 | 4271 | 66 % | 4271 | +0.0 % | ok |
| mla cache seg fp8 r512 p64 b64 (i64) | 3058 | 47 % | 3058 | +0.0 % | ok |
| silu_mul_bias d4096 E=64 | 9315 | 143 % | 9315 | +0.0 % | ok |
| gelu_mul_bias d2048 E=64 | 5107 | 79 % | 5107 | +0.0 % | ok |
| gelu (erf) unary | 3058 | 47 % | 3058 | +0.0 % | ok |
| gelu_mul (erf) d4096 | 4118 | 63 % | 4118 | +0.0 % | ok |
| mxfp6 e2m3 k4096 M=32768 | 2921 | 45 % | 2921 | +0.0 % | ok |
| mxfp6 e3m2 k4096 M=32768 | 2916 | 45 % | 2916 | +0.0 % | ok |
| topk e128 k8 T=65536 | 1176 | 18 % | 1176 | +0.0 % | ok |
| topk renorm e256 k8 T=65536 | 1002 | 15 % | 1002 | +0.0 % | ok |

92 cases, 0 correctness failure(s), 0 regression(s).
