# v11 vs v10: the same-run performance table (final Main)

Evidence for "no performance drop after the four-layer restructure and the two-Weaver work": `tools/perf_ab.py` (the harness of [`perf-ab.md`](perf-ab.md)) builds the code objects of **v10.0-layers** (`7cd290af`) and of the **final Main of this round (`a65de21a`**; the current Main `6a19a973` has identical kernels, it only adds a `forbid(unsafe_code)` attribute)
and runs EVERY integrated kernel family interleaved in one session on node 53, GPU 1 (no other GPU tenant): all 326 case rows of the 12 families (the memory-bound hosts, conv, decode16, MLA, MoE stages, the FMHA families, backward, GEMM, and the small-M GEMV family), **3 interleaved repetitions each**, floor 2 %, k 2.0, in two runs
(`--families membound,conv,decode,mla,moe` and `--families fmha_bf16,fmha_hd192,fmha_fp8pt,fmha_mxfp8,fmha_bwd,gemm,gemv`, `--old 7cd290af --new a65de21a --keep-going`); every case flagged at 3 repetitions was re-run with 9. Each launch takes its own commit's dispatch rule, so the rule changes
(the GEMM / MoE / MLA rules of the composition search) are part of what is measured. Vendor columns (aiter) are the same-run vendor kernel where a host runs one. A case FAILS when new / old < 1 - max(2 %, 2 x the pair's noise); a family when its geometric mean falls below 1 - max(1 %, ...).
This replaces the earlier snapshot (v10 vs `d4349932`).

## Verdict

**Every family is OK: no family geometric mean is below 1.0 beyond noise, and 112 of the 326 case rows are IMPROVED. 4 case rows were below 1.0 beyond the 3-repetition noise; 3 of them are noise (the 9-repetition re-run passes), 1 is a real regression of 3.3 % on one memory-bound kernel (below), reported to and fixed by the back-end owner (`sitfix`).**

| family | cases | geomean new/old | allowed drop | verdict |
|---|--:|--:|--:|---|
| conv | 3 | 1.000 | 1.0% | OK |
| decode | 6 | 1.017 | 1.0% | OK |
| membound | 91 | 1.005 | 1.0% | OK |
| mla | 13 | 1.515 | 1.0% | OK |
| moe | 26 | 1.029 | 1.0% | OK |
| fmha_bf16 | 10 | 0.995 | 1.0% | OK |
| fmha_bwd | 6 | 0.998 | 1.0% | OK |
| fmha_fp8pt | 8 | 0.999 | 1.0% | OK |
| fmha_hd192 | 8 | 1.000 | 1.2% | OK |
| fmha_mxfp8 | 4 | 1.022 | 1.2% | OK |
| gemm | 79 | 1.049 | 1.0% | OK |
| gemv | 60 | 1.061 | 1.0% | OK |

IMPROVED rows by family: gemv 48, gemm 29, mla 13 (up to 2.6x: B16 H16 ctx 1024), moe 9, memory-bound 7, decode16 3, mxfp8 2, bf16 1. The GEMM / GEMV / MoE / MLA gains are the dispatch-rule updates of the composition search (the 256-tile fill threshold, the K-aware split-K, the bf16 128 split form, the small-M GEMV forms,
the MoE block_m band, the head-parallel MLA at short context); the others are kernel schedule changes. **129 of the 241 kernels of the golden md5 manifest have different assembly in v10 and in the final Main** (the scheduler / wait work), so unlike the earlier snapshot this is no longer a mostly-identical-code comparison; it is a measurement of the changed code.

## The four cases flagged at 3 repetitions

| case | 3 reps new/old | analysis | 9 reps new/old |
|---|--:|---|--:|
| bf16/causal/1x8x8x4096x4096 | 0.957 | noise: `fmha_fwd_hd128_bf16_v3_causal_ks` asm identical in both trees (md5 bb90459b); vendor ratios of the same repetitions 1.18 / 1.24 | 0.991 (OK, allowed drop 5.2 %) |
| gemm/i8_rowcol/576x7168x6144 | 0.962 | noise: `gemm_a8w8_i8_128` asm identical (md5 c08dd640), the same tile / split under both rules | 1.014 (OK, 5.1 %) |
| moe/fp4/M2048D7168I256E64K8#stage1 | 0.975 | noise: the fp4 stage kernels are the same at E64 r 256 (the block_m rule is unchanged there) | 1.004 (OK, 3.9 %) |
| **mb/situv2_mul_quant d4096** | 0.969 | **real**: `situv2_mul_quant_fp8_d4096` has different assembly (md5 22e093ac in v10, 5651eda5 in the final Main) | **0.967 (FAIL, allowed drop 2.4 %)**: 4609 -> 4456 GB/s |

**The one regression:** a real 3.3 % loss from code churn, not from lost wait staging (my first reading). The kernel has an explicit `wait_vm(0)` right after its 16 loads in both trees; v10's later `vmcnt(15) .. (0)` waits were dead after that drain. The cause (back-end owner's analysis): `uwaits` recomputed the waits
BEFORE register allocation, which shifted the linear-scan positions, so the registers and the hazard-fill order changed (2429 non-wait lines differ). Fixed on `origin/sitfix` (the unscheduled path allocates with lowering's waits in place and recomputes the waits after RA): `situv2_mul_quant_fp8_d4096` is v10's code again, minus the dead waits. It is one case of the 91 memory-bound cases (family geometric mean 1.005).

Re-run (`--only` the four ids, 9 repetitions):

| case | old | new | new/old | allowed drop | verdict | vendor | new/vendor | old/vendor |
|---|--:|--:|--:|--:|---|--:|--:|--:|
| bf16/causal/1x8x8x4096x4096 | 459.20 | 455.20 | 0.991 | 5.2% | OK | 381.30 | 1.194 | 1.204 |
| gemm/i8_rowcol/576x7168x6144 | 1166 | 1182 | 1.014 | 5.1% | OK | - | - | - |
| mb/situv2_mul_quant d4096 | 4609 | 4456 | 0.967 | 2.4% | FAIL | - | - | - |
| moe/fp4/M2048D7168I256E64K8#stage1 | 106.50 | 106.09 | 1.004 | 3.9% | OK | - | - | - |
| moe/fp4/M2048D7168I256E64K8#stage2 | 273.71 | 274.22 | 0.998 | 5.0% | OK | - | - | - |

## Cases of the run: memory-bound, conv, decode16, MLA, MoE

| case | old | new | new/old | allowed drop | verdict | vendor | new/vendor | old/vendor |
|---|--:|--:|--:|--:|---|--:|--:|--:|
| conv/conv2d_bf16_r1s1_st1_p0_d1 | 338.80 | 339.20 | 1.001 | 2.0% | OK | - | - | - |
| conv/conv2d_bf16_r3s3_st1_p1_d1 | 372.90 | 372.30 | 0.998 | 2.0% | OK | - | - | - |
| conv/conv3d_bf16_t3r3s3_st1_p1_d1 | 387.10 | 387.80 | 1.002 | 2.0% | OK | - | - | - |
| decode16/B1H32HK4C32768 | 114.92 | 111.41 | 1.032 | 2.0% | IMPROVED | - | - | - |
| decode16/B1H64HK8C8192 | 29.33 | 28.72 | 1.021 | 2.0% | IMPROVED | - | - | - |
| decode16/B32H64HK8C2048 | 35.51 | 35.41 | 1.003 | 2.0% | OK | - | - | - |
| decode16/B4H16HK1C16384 | 57.23 | 54.73 | 1.046 | 2.0% | IMPROVED | - | - | - |
| decode16/B64H32HK4C1024 | 20.00 | 19.93 | 1.004 | 2.0% | OK | - | - | - |
| decode16/B8H64HK8C4096 | 26.46 | 26.57 | 0.996 | 2.0% | OK | - | - | - |
| mb/add_bf16 | 6351 | 6552 | 1.032 | 2.0% | IMPROVED | - | - | - |
| mb/add_rmsnorm n4096 | 5771 | 5716 | 0.991 | 6.5% | OK | - | - | - |
| mb/add_rmsnorm_fp8 n1000 | 3912 | 4002 | 1.023 | 7.8% | OK | - | - | - |
| mb/add_rmsnorm_fp8 n4096 | 4289 | 4285 | 0.999 | 2.0% | OK | - | - | - |
| mb/argmax v128256 B=4096 | 4818 | 5054 | 1.049 | 2.0% | IMPROVED | - | - | - |
| mb/argmax v32000 B=16384 | 4382 | 4505 | 1.028 | 2.3% | IMPROVED | - | - | - |
| mb/cache h8 d128 | 5030 | 4955 | 0.985 | 2.4% | OK | - | - | - |
| mb/cache_blocked fp8 per-token h8 d128 b16 (i64 slots) | 924.70 | 918.60 | 0.993 | 2.0% | OK | - | - | - |
| mb/cache_blocked h8 d128 b16 contiguous slots | 1448 | 1448 | 1.000 | 2.0% | OK | - | - | - |
| mb/cache_blocked h8 d128 b16 random slots | 920.50 | 918.10 | 0.997 | 3.3% | OK | - | - | - |
| mb/cache_fp8 h8 d128 | 5903 | 5966 | 1.011 | 6.5% | OK | - | - | - |
| mb/conv1d_fwd split w4 silu B=128 L=512 uniform | 2656 | 2663 | 1.003 | 2.0% | OK | - | - | - |
| mb/conv1d_fwd split w4 silu ragged + states | 397.60 | 397.60 | 1.000 | 2.0% | OK | - | - | - |
| mb/conv1d_fwd w4 silu B=128 L=512 uniform | 2804 | 2817 | 1.005 | 2.0% | OK | - | - | - |
| mb/conv1d_fwd w4 silu ragged B=256 (_sched) | 2654 | 2688 | 1.012 | 2.0% | OK | - | - | - |
| mb/conv1d_fwd w4 silu ragged B=256 (grid) | 995.40 | 1001 | 1.006 | 2.0% | OK | - | - | - |
| mb/conv1d_update w4 s3 idx silu | 3064 | 3090 | 1.008 | 2.7% | OK | - | - | - |
| mb/cvt_bf16_f32 | 6299 | 6280 | 0.997 | 2.0% | OK | - | - | - |
| mb/cvt_f32_bf16 | 6583 | 6521 | 0.991 | 2.0% | OK | - | - | - |
| mb/dequant group128 n4096 | 5929 | 6024 | 1.016 | 2.0% | OK | - | - | - |
| mb/dequant static | 6406 | 6398 | 0.999 | 2.0% | OK | - | - | - |
| mb/dequant token n1000 (odd tail) | 5235 | 5241 | 1.001 | 2.5% | OK | - | - | - |
| mb/dequant token n4096 | 6040 | 6010 | 0.995 | 2.0% | OK | - | - | - |
| mb/gated_rmsnorm bf16 n128 (heads) | 6200 | 6186 | 0.998 | 2.3% | OK | - | - | - |
| mb/gated_rmsnorm bf16 n4096 | 6046 | 6121 | 1.012 | 2.2% | OK | - | - | - |
| mb/gated_rmsnorm fp8 n128 (heads) | 5729 | 5726 | 1.000 | 2.0% | OK | - | - | - |
| mb/gated_rmsnorm fp8 n4096 | 5365 | 5333 | 0.994 | 3.3% | OK | - | - | - |
| mb/gelu (erf) unary | 3032 | 3025 | 0.998 | 2.0% | OK | - | - | - |
| mb/gelu_mul (erf) d4096 | 4080 | 4108 | 1.007 | 2.0% | OK | - | - | - |
| mb/gelu_mul_bias d2048 E=64 | 5163 | 5145 | 0.997 | 2.0% | OK | - | - | - |
| mb/gelu_tanh (unary) | 5514 | 5620 | 1.019 | 3.9% | OK | - | - | - |
| mb/gelu_tanh d4096 | 6500 | 6522 | 1.003 | 2.0% | OK | - | - | - |
| mb/layernorm n4096 | 5560 | 5539 | 0.996 | 2.0% | OK | - | - | - |
| mb/layernorm n776 | 4478 | 4482 | 1.001 | 3.9% | OK | - | - | - |
| mb/mla cache flat bf16 r512 p64 | 4287 | 4270 | 0.996 | 2.0% | OK | - | - | - |
| mb/mla cache seg fp8 r512 p64 b64 (i64) | 3207 | 3172 | 0.989 | 2.0% | OK | - | - | - |
| mb/moe_combine k8 d4096 T=32768 | 6354 | 6336 | 0.997 | 2.0% | OK | - | - | - |
| mb/moe_fused_gate e256 g8 tg4 k8 M=1001 (odd) | 70.60 | 70.70 | 1.001 | 2.0% | OK | - | - | - |
| mb/moe_fused_gate e256 g8 tg4 k8 M=131072 | 805.50 | 801.50 | 0.995 | 2.0% | OK | - | - | - |
| mb/moe_reduce k8 d4096 T=32768 | 6381 | 6357 | 0.996 | 2.0% | OK | - | - | - |
| mb/moe_reduce k8 d7168 T=16384 | 6315 | 6270 | 0.993 | 2.0% | OK | - | - | - |
| mb/moe_sorting e128 k8 b32 M=1001 skew+invalid | 14.80 | 14.77 | 1.002 | 2.0% | OK | - | - | - |
| mb/moe_sorting e128 k8 b32 M=37 | 12.13 | 11.99 | 1.012 | 2.0% | OK | - | - | - |
| mb/moe_sorting e256 k8 b128 M=4096 | 52.64 | 52.56 | 1.002 | 2.0% | OK | - | - | - |
| mb/moe_sorting e385 k6 b32 M=777 | 17.96 | 17.79 | 1.010 | 2.0% | OK | - | - | - |
| mb/moe_sorting e64 k2 b16 M=513 | 10.54 | 10.44 | 1.010 | 2.0% | OK | - | - | - |
| mb/mxfp6 e2m3 k4096 M=32768 | 2910 | 2910 | 1.000 | 2.0% | OK | - | - | - |
| mb/mxfp6 e3m2 k4096 M=32768 | 2905 | 2914 | 1.003 | 2.0% | OK | - | - | - |
| mb/qknorm_rope neox h40 d128 | 4238 | 4462 | 1.053 | 2.0% | IMPROVED | - | - | - |
| mb/qknorm_rope_kvcache neox bf16 h8 d128 | 5835 | 6010 | 1.030 | 2.0% | IMPROVED | - | - | - |
| mb/qknorm_rope_kvcache neox fp8 h8 d128 (i64) | 4749 | 5365 | 1.130 | 2.0% | IMPROVED | - | - | - |
| mb/quant_group128 n4096 | 6499 | 6495 | 0.999 | 2.0% | OK | - | - | - |
| mb/quant_static | 6132 | 6136 | 1.001 | 2.0% | OK | - | - | - |
| mb/quant_token n1000 | 6062 | 6025 | 0.994 | 2.5% | OK | - | - | - |
| mb/quant_token n14336 (long row) | 6060 | 5972 | 0.985 | 3.6% | OK | - | - | - |
| mb/quant_token n32768 (long row) | 4368 | 4380 | 1.003 | 2.0% | OK | - | - | - |
| mb/quant_token n4096 | 6394 | 6381 | 0.998 | 2.0% | OK | - | - | - |
| mb/quant_token_int8 n1000 | 5437 | 5470 | 1.006 | 2.0% | OK | - | - | - |
| mb/quant_token_int8 n14336 (long row) | 5796 | 5788 | 0.999 | 2.0% | OK | - | - | - |
| mb/quant_token_int8 n4096 | 6318 | 6286 | 0.995 | 2.0% | OK | - | - | - |
| mb/relu2 (unary) | 6459 | 6474 | 1.002 | 2.0% | OK | - | - | - |
| mb/rmsnorm n1000 | 5710 | 5706 | 0.999 | 2.0% | OK | - | - | - |
| mb/rmsnorm n200 | 5314 | 5229 | 0.984 | 2.0% | OK | - | - | - |
| mb/rmsnorm n4096 | 6179 | 6173 | 0.999 | 3.4% | OK | - | - | - |
| mb/rmsnorm_quant_mxfp4 k4096 M=32768 | 2772 | 2787 | 1.005 | 2.0% | OK | - | - | - |
| mb/rope_gptj h32 d128 | 6339 | 6452 | 1.018 | 2.0% | OK | - | - | - |
| mb/rope_neox h32 d128 | 6408 | 6415 | 1.001 | 2.0% | OK | - | - | - |
| mb/rope_neox h8 d128 | 6747 | 6786 | 1.006 | 2.0% | OK | - | - | - |
| mb/sample v128256 B=4096 | 723.30 | 725.00 | 1.002 | 2.0% | OK | - | - | - |
| mb/sample v32000 B=16384 | 1288 | 1296 | 1.006 | 2.0% | OK | - | - | - |
| mb/sigmoid (unary) | 6319 | 6383 | 1.010 | 2.0% | OK | - | - | - |
| mb/silu (unary) | 6090 | 6131 | 1.007 | 2.0% | OK | - | - | - |
| mb/silu_mul d1024 | 6712 | 6706 | 0.999 | 2.0% | OK | - | - | - |
| mb/silu_mul d14336 | 6535 | 6513 | 0.997 | 2.0% | OK | - | - | - |
| mb/silu_mul_bias d4096 E=64 | 9127 | 9100 | 0.997 | 6.0% | OK | - | - | - |
| mb/silu_mul_limit d4096 (limit 7) | 6537 | 6540 | 1.000 | 2.0% | OK | - | - | - |
| mb/silu_mul_quant d1000 | 5929 | 5973 | 1.007 | 2.0% | OK | - | - | - |
| mb/silu_mul_quant d14336 (long row) | 4047 | 4083 | 1.009 | 2.0% | OK | - | - | - |
| mb/silu_mul_quant d32768 (long row) | 3301 | 3323 | 1.007 | 2.0% | OK | - | - | - |
| mb/silu_mul_quant d4096 | 6041 | 6067 | 1.004 | 2.0% | OK | - | - | - |
| mb/situv2_mul_quant d14336 (long row) | 2816 | 2808 | 0.997 | 2.0% | OK | - | - | - |
| mb/situv2_mul_quant d4096 | 4623 | 4479 | 0.969 | 2.0% | FAIL | - | - | - |
| mb/softmax n1000 | 5658 | 5725 | 1.012 | 3.1% | OK | - | - | - |
| mb/softmax n4096 | 6008 | 6067 | 1.010 | 2.9% | OK | - | - | - |
| mb/softmax n8192 | 5512 | 5470 | 0.992 | 3.4% | OK | - | - | - |
| mb/swiglu_oai_mul d4096 | 6546 | 6548 | 1.000 | 2.0% | OK | - | - | - |
| mb/topk e128 k8 T=65536 | 1220 | 1224 | 1.004 | 2.0% | OK | - | - | - |
| mb/topk renorm e256 k8 T=65536 | 1032 | 1034 | 1.002 | 2.0% | OK | - | - | - |
| mb/topk_per_row decode k2048 B=256 L<=64K | 329.10 | 339.60 | 1.032 | 2.0% | IMPROVED | - | - | - |
| mb/topk_per_row decode k2048 odd (ragged, ties) | 33.20 | 33.60 | 1.012 | 2.0% | OK | - | - | - |
| mb/topk_per_row prefill k128 (ties) | 25.20 | 25.60 | 1.016 | 2.0% | OK | - | - | - |
| mla/B12H48C2048 | 59.67 | 36.20 | 1.648 | 2.0% | IMPROVED | - | - | - |
| mla/B16H16C1024 | 57.75 | 22.19 | 2.603 | 2.0% | IMPROVED | - | - | - |
| mla/B1H128C8192 | 54.14 | 50.52 | 1.072 | 2.0% | IMPROVED | - | - | - |
| mla/B1H16C2048 | 50.45 | 23.65 | 2.133 | 2.0% | IMPROVED | - | - | - |
| mla/B1H32C2048 | 52.08 | 25.07 | 2.077 | 2.0% | IMPROVED | - | - | - |
| mla/B2H16C3072 | 55.56 | 29.69 | 1.871 | 2.0% | IMPROVED | - | - | - |
| mla/B32H16C2048 | 45.24 | 37.94 | 1.192 | 2.0% | IMPROVED | - | - | - |
| mla/B4H16C16384 | 80.83 | 65.92 | 1.226 | 2.0% | IMPROVED | - | - | - |
| mla/B4H16C4096 | 57.92 | 35.56 | 1.629 | 2.0% | IMPROVED | - | - | - |
| mla/B4H16C8192 | 61.13 | 46.21 | 1.323 | 2.0% | IMPROVED | - | - | - |
| mla/B64H16C4096 | 71.96 | 68.21 | 1.055 | 2.9% | IMPROVED | - | - | - |
| mla/B8H128C4096 | 53.57 | 49.99 | 1.072 | 2.0% | IMPROVED | - | - | - |
| mla/B8H32C4096 | 62.09 | 38.10 | 1.630 | 2.0% | IMPROVED | - | - | - |
| moe/bf16/M1024D2048I768E32K8#stage1 | 96.45 | 96.54 | 0.999 | 2.7% | OK | - | - | - |
| moe/bf16/M1024D2048I768E32K8#stage2 | 69.58 | 70.12 | 0.992 | 3.6% | OK | - | - | - |
| moe/bf16/M256D2048I768E64K8#stage1 | 85.27 | 85.40 | 0.998 | 2.0% | OK | - | - | - |
| moe/bf16/M256D2048I768E64K8#stage2 | 44.48 | 44.87 | 0.991 | 2.0% | OK | - | - | - |
| moe/bf16/M320D2048I768E16K8#stage1 | 43.20 | 41.24 | 1.048 | 2.0% | IMPROVED | - | - | - |
| moe/bf16/M320D2048I768E16K8#stage2 | 27.49 | 27.82 | 0.988 | 3.6% | OK | - | - | - |
| moe/bf16/M384D2048I768E16K8#stage1 | 41.65 | 41.74 | 0.998 | 2.0% | OK | - | - | - |
| moe/bf16/M384D2048I768E16K8#stage2 | 28.83 | 28.93 | 0.997 | 2.0% | OK | - | - | - |
| moe/bf16/M4096D2048I768E64K8#stage1 | 364.79 | 361.59 | 1.009 | 2.0% | OK | - | - | - |
| moe/bf16/M4096D2048I768E64K8#stage2 | 263.68 | 263.06 | 1.002 | 2.0% | OK | - | - | - |
| moe/fp4/M2048D7168I256E64K8#stage1 | 105.84 | 108.52 | 0.975 | 2.0% | FAIL | - | - | - |
| moe/fp4/M2048D7168I256E64K8#stage2 | 287.93 | 287.37 | 1.002 | 2.0% | OK | - | - | - |
| moe/fp4/M256D7168I256E64K8#stage1 | 40.83 | 41.15 | 0.992 | 3.1% | OK | - | - | - |
| moe/fp4/M256D7168I256E64K8#stage2 | 41.67 | 42.09 | 0.990 | 2.3% | OK | - | - | - |
| moe/fp4/M384D7168I256E16K8#stage1 | 47.79 | 41.14 | 1.162 | 2.0% | IMPROVED | - | - | - |
| moe/fp4/M384D7168I256E16K8#stage2 | 46.09 | 43.54 | 1.059 | 2.0% | IMPROVED | - | - | - |
| moe/fp4/M512D7168I256E32K8#stage1 | 49.17 | 43.29 | 1.136 | 2.0% | IMPROVED | - | - | - |
| moe/fp4/M512D7168I256E32K8#stage2 | 62.52 | 58.12 | 1.076 | 2.0% | IMPROVED | - | - | - |
| moe/fp8/M1280D4096I256E64K8#stage1 | 63.34 | 63.67 | 0.995 | 2.0% | OK | - | - | - |
| moe/fp8/M1280D4096I256E64K8#stage2 | 118.54 | 119.25 | 0.994 | 2.0% | OK | - | - | - |
| moe/fp8/M2048D4096I256E32K8#stage1 | 84.10 | 84.70 | 0.993 | 2.5% | OK | - | - | - |
| moe/fp8/M2048D4096I256E32K8#stage2 | 164.35 | 164.04 | 1.002 | 2.0% | OK | - | - | - |
| moe/fp8/M512D4096I256E32K8#stage1 | 36.92 | 33.56 | 1.100 | 2.0% | IMPROVED | - | - | - |
| moe/fp8/M512D4096I256E32K8#stage2 | 50.34 | 47.69 | 1.056 | 2.0% | IMPROVED | - | - | - |
| moe/fp8/M640D4096I256E32K8#stage1 | 44.00 | 38.28 | 1.149 | 2.0% | IMPROVED | - | - | - |
| moe/fp8/M640D4096I256E32K8#stage2 | 61.70 | 56.06 | 1.101 | 2.0% | IMPROVED | - | - | - |

## Cases of the run: FMHA families, backward, GEMM, GEMV

| case | old | new | new/old | allowed drop | verdict | vendor | new/vendor | old/vendor |
|---|--:|--:|--:|--:|---|--:|--:|--:|
| bf16/causal/1x32x8x2048x2048 | 535.30 | 547.90 | 1.024 | 2.2% | IMPROVED | 545.30 | 1.005 | 0.982 |
| bf16/causal/1x8x8x4096x4096 | 468.30 | 448.20 | 0.957 | 2.0% | FAIL | 379.30 | 1.182 | 1.235 |
| bf16/causal/2x32x32x8192x8192 | 981.40 | 984.40 | 1.003 | 2.0% | OK | 1122 | 0.878 | 0.875 |
| bf16/causal/4x16x16x1024x1024 | 416.20 | 419.00 | 1.007 | 2.0% | OK | 419.00 | 1.000 | 0.993 |
| bf16/causal/4x16x16x2048x2048 | 624.10 | 638.30 | 1.023 | 4.8% | OK | 790.10 | 0.808 | 0.790 |
| bf16/full/1x32x8x2048x2048 | 838.80 | 809.60 | 0.965 | 5.0% | OK | 974.60 | 0.831 | 0.861 |
| bf16/full/1x8x8x4096x4096 | 757.90 | 756.70 | 0.998 | 5.3% | OK | 702.90 | 1.077 | 1.078 |
| bf16/full/2x32x32x8192x8192 | 1107 | 1111 | 1.003 | 2.0% | OK | 1180 | 0.941 | 0.938 |
| bf16/full/4x16x16x1024x1024 | 745.50 | 737.10 | 0.989 | 3.2% | OK | 858.40 | 0.859 | 0.868 |
| bf16/full/4x16x16x2048x2048 | 806.70 | 791.50 | 0.981 | 2.0% | OK | 900.30 | 0.879 | 0.896 |
| bwd/causal/1x16x16x1024x1024 | 133.80 | 133.20 | 0.996 | 2.0% | OK | 100.10 | 1.331 | 1.337 |
| bwd/causal/1x16x4x1024x1024 | 133.60 | 133.20 | 0.997 | 2.0% | OK | 99.80 | 1.335 | 1.339 |
| bwd/causal/2x8x8x1024x1024 | 135.90 | 135.30 | 0.996 | 2.0% | OK | 100.20 | 1.350 | 1.356 |
| bwd/full/1x16x16x1024x1024 | 253.70 | 253.10 | 0.998 | 2.0% | OK | 255.80 | 0.989 | 0.992 |
| bwd/full/1x16x4x1024x1024 | 255.80 | 256.10 | 1.001 | 2.0% | OK | 256.00 | 1.000 | 0.999 |
| bwd/full/2x8x8x1024x1024 | 256.50 | 256.10 | 0.998 | 2.0% | OK | 256.20 | 1.000 | 1.001 |
| fp8pt/causal/1x8x8x4096x4096 | 874.60 | 871.90 | 0.997 | 2.0% | OK | 503.50 | 1.732 | 1.737 |
| fp8pt/causal/2x32x32x8192x8192 | 2048 | 2048 | 1.000 | 2.0% | OK | 1819 | 1.126 | 1.126 |
| fp8pt/causal/2x32x8x4096x4096 | 1853 | 1845 | 0.996 | 2.0% | OK | 1678 | 1.100 | 1.104 |
| fp8pt/causal/4x16x16x2048x2048 | 1308 | 1307 | 0.999 | 2.0% | OK | 1198 | 1.090 | 1.092 |
| fp8pt/full/1x8x8x4096x4096 | 1618 | 1614 | 0.997 | 2.0% | OK | 1086 | 1.486 | 1.490 |
| fp8pt/full/2x32x32x8192x8192 | 2444 | 2446 | 1.001 | 2.0% | OK | 2060 | 1.187 | 1.186 |
| fp8pt/full/2x32x8x4096x4096 | 2328 | 2330 | 1.001 | 2.0% | OK | 1984 | 1.174 | 1.173 |
| fp8pt/full/4x16x16x2048x2048 | 1941 | 1949 | 1.005 | 2.5% | OK | 1596 | 1.221 | 1.216 |
| hd192/causal/1x128x128x1024x1024 | 642.60 | 636.20 | 0.990 | 2.8% | OK | 653.00 | 0.974 | 0.984 |
| hd192/causal/1x16x16x4096x4096 | 727.20 | 723.20 | 0.994 | 2.1% | OK | 928.00 | 0.779 | 0.784 |
| hd192/causal/1x8x8x4096x4096 | 587.50 | 578.90 | 0.985 | 4.2% | OK | 615.90 | 0.940 | 0.954 |
| hd192/causal/4x16x16x2048x2048 | 813.20 | 808.80 | 0.995 | 5.4% | OK | 826.20 | 0.979 | 0.984 |
| hd192/full/1x128x128x1024x1024 | 884.10 | 885.60 | 1.002 | 2.1% | OK | 854.60 | 1.036 | 1.035 |
| hd192/full/1x16x16x4096x4096 | 1135 | 1144 | 1.008 | 2.0% | OK | 1181 | 0.969 | 0.962 |
| hd192/full/1x8x8x4096x4096 | 948.30 | 967.60 | 1.020 | 2.1% | OK | 1203 | 0.804 | 0.788 |
| hd192/full/4x16x16x2048x2048 | 1036 | 1042 | 1.006 | 4.9% | OK | 1077 | 0.968 | 0.962 |
| mxfp8/1x8x2048 | 672.20 | 688.40 | 1.024 | 2.0% | IMPROVED | 711.62 | 0.967 | 0.945 |
| mxfp8/1x8x8192 | 2327 | 2342 | 1.007 | 2.0% | OK | 2403 | 0.975 | 0.968 |
| mxfp8/2x32x8192 | 2503 | 2506 | 1.001 | 2.0% | OK | 2526 | 0.992 | 0.991 |
| mxfp8/4x8x2048 | 1716 | 1812 | 1.056 | 4.1% | IMPROVED | 1899 | 0.954 | 0.904 |
| gemm/bf16/1024x7168x2048 | 812.40 | 854.10 | 1.051 | 4.3% | IMPROVED | - | - | - |
| gemm/bf16/1152x4096x6144 | 720.00 | 718.00 | 0.997 | 2.0% | OK | - | - | - |
| gemm/bf16/1280x7168x4096 | 775.70 | 917.40 | 1.183 | 2.2% | IMPROVED | - | - | - |
| gemm/bf16/128x7168x8192 | 31.64 | 28.82 | 1.098 | 2.0% | IMPROVED | - | - | - |
| gemm/bf16/160x4096x8192 | 402.80 | 413.00 | 1.025 | 2.0% | IMPROVED | - | - | - |
| gemm/bf16/16x8192x8192 | 25.57 | 24.17 | 1.058 | 2.0% | IMPROVED | - | - | - |
| gemm/bf16/1792x8192x4096 | 955.00 | 1207 | 1.264 | 3.5% | IMPROVED | - | - | - |
| gemm/bf16/192x4096x4096 | 347.70 | 392.40 | 1.129 | 2.0% | IMPROVED | - | - | - |
| gemm/bf16/2048x7168x7168 | 1028 | 1295 | 1.259 | 2.0% | IMPROVED | - | - | - |
| gemm/bf16/256x4096x8192 | 569.90 | 581.90 | 1.021 | 6.6% | OK | - | - | - |
| gemm/bf16/256x7168x2048 | 579.50 | 580.70 | 1.002 | 2.8% | OK | - | - | - |
| gemm/bf16/320x8192x6144 | 543.20 | 585.70 | 1.078 | 3.5% | IMPROVED | - | - | - |
| gemm/bf16/32x4096x14336 | 22.47 | 22.02 | 1.020 | 2.0% | IMPROVED | - | - | - |
| gemm/bf16/4096x4096x8192 | 1442 | 1441 | 0.999 | 2.0% | OK | - | - | - |
| gemm/bf16/576x7168x6144 | 656.20 | 656.20 | 1.000 | 2.0% | OK | - | - | - |
| gemm/bf16/64x8192x8192 | 26.27 | 26.49 | 0.992 | 2.0% | OK | - | - | - |
| gemm/bf16/768x4096x8192 | 676.00 | 717.80 | 1.062 | 2.0% | IMPROVED | - | - | - |
| gemm/bf16/8192x8192x8192 | 1484 | 1482 | 0.998 | 2.0% | OK | - | - | - |
| gemm/bf16/96x8192x8192 | 29.60 | 29.67 | 0.998 | 2.0% | OK | - | - | - |
| gemm/fp8_blockscale/1024x7168x2048 | 967.00 | 946.40 | 0.979 | 7.4% | OK | - | - | - |
| gemm/fp8_blockscale/1152x4096x6144 | 923.70 | 935.10 | 1.012 | 2.0% | OK | - | - | - |
| gemm/fp8_blockscale/1280x7168x4096 | 849.40 | 847.10 | 0.997 | 2.0% | OK | - | - | - |
| gemm/fp8_blockscale/128x7168x8192 | 23.75 | 24.22 | 0.981 | 2.0% | OK | - | - | - |
| gemm/fp8_blockscale/160x4096x8192 | 498.30 | 503.10 | 1.010 | 2.0% | OK | - | - | - |
| gemm/fp8_blockscale/16x8192x8192 | 18.21 | 16.96 | 1.074 | 2.0% | IMPROVED | - | - | - |
| gemm/fp8_blockscale/1792x8192x4096 | 1119 | 1104 | 0.986 | 2.1% | OK | - | - | - |
| gemm/fp8_blockscale/192x4096x4096 | 348.70 | 348.20 | 0.999 | 2.2% | OK | - | - | - |
| gemm/fp8_blockscale/2048x7168x7168 | 1181 | 1186 | 1.004 | 2.0% | OK | - | - | - |
| gemm/fp8_blockscale/256x4096x8192 | 721.10 | 715.00 | 0.992 | 2.0% | OK | - | - | - |
| gemm/fp8_blockscale/256x7168x2048 | 713.90 | 712.80 | 0.998 | 2.0% | OK | - | - | - |
| gemm/fp8_blockscale/320x8192x6144 | 809.50 | 821.20 | 1.014 | 2.0% | OK | - | - | - |
| gemm/fp8_blockscale/32x4096x14336 | 18.17 | 17.54 | 1.036 | 2.0% | IMPROVED | - | - | - |
| gemm/fp8_blockscale/4096x4096x8192 | 1336 | 1330 | 0.995 | 2.0% | OK | - | - | - |
| gemm/fp8_blockscale/576x7168x6144 | 1059 | 1041 | 0.983 | 3.6% | OK | - | - | - |
| gemm/fp8_blockscale/64x8192x8192 | 19.34 | 19.21 | 1.007 | 2.4% | OK | - | - | - |
| gemm/fp8_blockscale/768x4096x8192 | 938.30 | 933.70 | 0.995 | 2.8% | OK | - | - | - |
| gemm/fp8_blockscale/8192x8192x8192 | 1503 | 1507 | 1.003 | 2.0% | OK | - | - | - |
| gemm/fp8_blockscale/96x8192x8192 | 23.22 | 22.97 | 1.011 | 2.0% | OK | - | - | - |
| gemm/fp8_rowcol/1024x7168x2048 | 1428 | 1452 | 1.017 | 4.3% | OK | - | - | - |
| gemm/fp8_rowcol/1152x4096x6144 | 1274 | 1278 | 1.003 | 2.5% | OK | - | - | - |
| gemm/fp8_rowcol/1280x7168x4096 | 1341 | 1590 | 1.186 | 2.0% | IMPROVED | - | - | - |
| gemm/fp8_rowcol/128x7168x8192 | 21.36 | 18.25 | 1.170 | 2.8% | IMPROVED | - | - | - |
| gemm/fp8_rowcol/160x4096x8192 | 609.10 | 669.80 | 1.100 | 2.4% | IMPROVED | - | - | - |
| gemm/fp8_rowcol/16x8192x8192 | 17.49 | 16.43 | 1.065 | 3.2% | IMPROVED | - | - | - |
| gemm/fp8_rowcol/1792x8192x4096 | 1640 | 2030 | 1.238 | 2.0% | IMPROVED | - | - | - |
| gemm/fp8_rowcol/192x4096x4096 | 403.90 | 549.10 | 1.359 | 3.5% | IMPROVED | - | - | - |
| gemm/fp8_rowcol/2048x7168x7168 | 1769 | 2247 | 1.270 | 3.0% | IMPROVED | - | - | - |
| gemm/fp8_rowcol/256x4096x8192 | 938.00 | 949.90 | 1.013 | 2.0% | OK | - | - | - |
| gemm/fp8_rowcol/256x7168x2048 | 911.30 | 910.80 | 0.999 | 2.0% | OK | - | - | - |
| gemm/fp8_rowcol/320x8192x6144 | 1032 | 1041 | 1.009 | 5.1% | OK | - | - | - |
| gemm/fp8_rowcol/32x4096x14336 | 14.02 | 13.65 | 1.027 | 2.0% | IMPROVED | - | - | - |
| gemm/fp8_rowcol/4096x4096x8192 | 2467 | 2457 | 0.996 | 2.0% | OK | - | - | - |
| gemm/fp8_rowcol/576x7168x6144 | 1166 | 1171 | 1.005 | 5.3% | OK | - | - | - |
| gemm/fp8_rowcol/64x8192x8192 | 18.49 | 18.33 | 1.009 | 2.0% | OK | - | - | - |
| gemm/fp8_rowcol/768x4096x8192 | 1213 | 1208 | 0.996 | 2.2% | OK | - | - | - |
| gemm/fp8_rowcol/8192x8192x8192 | 2882 | 2884 | 1.001 | 2.0% | OK | - | - | - |
| gemm/fp8_rowcol/96x8192x8192 | 19.31 | 19.15 | 1.008 | 2.0% | OK | - | - | - |
| gemm/i8_rowcol/1024x7168x2048 | 1410 | 1378 | 0.978 | 2.8% | OK | - | - | - |
| gemm/i8_rowcol/1152x4096x6144 | 1278 | 1264 | 0.989 | 2.0% | OK | - | - | - |
| gemm/i8_rowcol/1280x7168x4096 | 1324 | 1543 | 1.165 | 4.1% | IMPROVED | - | - | - |
| gemm/i8_rowcol/128x7168x8192 | 21.69 | 18.15 | 1.195 | 2.0% | IMPROVED | - | - | - |
| gemm/i8_rowcol/160x4096x8192 | 589.20 | 669.40 | 1.136 | 2.2% | IMPROVED | - | - | - |
| gemm/i8_rowcol/16x8192x8192 | 17.82 | 16.63 | 1.072 | 3.2% | IMPROVED | - | - | - |
| gemm/i8_rowcol/1792x8192x4096 | 1628 | 1998 | 1.227 | 2.5% | IMPROVED | - | - | - |
| gemm/i8_rowcol/192x4096x4096 | 401.90 | 535.40 | 1.332 | 2.0% | IMPROVED | - | - | - |
| gemm/i8_rowcol/2048x7168x7168 | 1773 | 2216 | 1.250 | 2.0% | IMPROVED | - | - | - |
| gemm/i8_rowcol/256x4096x8192 | 939.30 | 939.30 | 1.000 | 2.0% | OK | - | - | - |
| gemm/i8_rowcol/256x7168x2048 | 889.80 | 890.90 | 1.001 | 2.0% | OK | - | - | - |
| gemm/i8_rowcol/320x8192x6144 | 1032 | 1034 | 1.002 | 2.7% | OK | - | - | - |
| gemm/i8_rowcol/32x4096x14336 | 14.50 | 13.83 | 1.048 | 3.7% | IMPROVED | - | - | - |
| gemm/i8_rowcol/4096x4096x8192 | 2408 | 2397 | 0.995 | 2.0% | OK | - | - | - |
| gemm/i8_rowcol/576x7168x6144 | 1182 | 1137 | 0.962 | 3.3% | FAIL | - | - | - |
| gemm/i8_rowcol/64x8192x8192 | 18.38 | 18.42 | 0.998 | 2.9% | OK | - | - | - |
| gemm/i8_rowcol/768x4096x8192 | 1230 | 1228 | 0.998 | 4.8% | OK | - | - | - |
| gemm/i8_rowcol/8192x8192x8192 | 2833 | 2826 | 0.998 | 2.0% | OK | - | - | - |
| gemm/i8_rowcol/96x8192x8192 | 19.81 | 19.27 | 1.028 | 3.7% | OK | - | - | - |
| gemm/mxfp8_v4/1024x4096x8192 | 988.42 | 988.29 | 1.000 | 2.0% | OK | - | - | - |
| gemm/mxfp8_v4/4096x4096x8192 | 2997 | 3037 | 1.013 | 2.0% | OK | - | - | - |
| gemm/mxfp8_v4/8192x8192x8192 | 3357 | 3366 | 1.003 | 2.0% | OK | - | - | - |
| gemv/bf16/16x4096x4096 | 10.18 | 9.29 | 1.096 | 2.0% | IMPROVED | - | - | - |
| gemv/bf16/16x7168x7168 | 19.80 | 18.59 | 1.065 | 2.0% | IMPROVED | - | - | - |
| gemv/bf16/16x8192x8192 | 25.43 | 24.45 | 1.040 | 2.0% | IMPROVED | - | - | - |
| gemv/bf16/1x4096x4096 | 9.94 | 9.09 | 1.094 | 2.0% | IMPROVED | - | - | - |
| gemv/bf16/1x7168x7168 | 19.32 | 18.19 | 1.062 | 2.0% | IMPROVED | - | - | - |
| gemv/bf16/1x8192x8192 | 26.12 | 24.99 | 1.045 | 2.0% | IMPROVED | - | - | - |
| gemv/bf16/32x4096x4096 | 10.78 | 10.15 | 1.062 | 2.0% | IMPROVED | - | - | - |
| gemv/bf16/32x7168x7168 | 20.27 | 19.62 | 1.033 | 2.0% | IMPROVED | - | - | - |
| gemv/bf16/32x8192x8192 | 25.92 | 25.01 | 1.036 | 2.0% | IMPROVED | - | - | - |
| gemv/bf16/4x4096x4096 | 9.92 | 9.12 | 1.088 | 2.0% | IMPROVED | - | - | - |
| gemv/bf16/4x7168x7168 | 19.59 | 18.49 | 1.059 | 2.0% | IMPROVED | - | - | - |
| gemv/bf16/4x8192x8192 | 25.12 | 23.91 | 1.051 | 2.0% | IMPROVED | - | - | - |
| gemv/bf16/64x4096x4096 | 11.93 | 11.94 | 0.999 | 2.0% | OK | - | - | - |
| gemv/bf16/64x7168x7168 | 21.48 | 21.42 | 1.003 | 2.0% | OK | - | - | - |
| gemv/bf16/64x8192x8192 | 26.62 | 26.33 | 1.011 | 2.0% | OK | - | - | - |
| gemv/fp8_blockscale/16x4096x4096 | 9.06 | 8.28 | 1.094 | 2.0% | IMPROVED | - | - | - |
| gemv/fp8_blockscale/16x7168x7168 | 13.45 | 12.20 | 1.102 | 2.0% | IMPROVED | - | - | - |
| gemv/fp8_blockscale/16x8192x8192 | 18.02 | 17.11 | 1.053 | 2.1% | IMPROVED | - | - | - |
| gemv/fp8_blockscale/1x4096x4096 | 8.81 | 8.03 | 1.097 | 2.0% | IMPROVED | - | - | - |
| gemv/fp8_blockscale/1x7168x7168 | 13.04 | 11.74 | 1.111 | 4.0% | IMPROVED | - | - | - |
| gemv/fp8_blockscale/1x8192x8192 | 18.21 | 16.94 | 1.075 | 2.0% | IMPROVED | - | - | - |
| gemv/fp8_blockscale/32x4096x4096 | 9.43 | 9.00 | 1.048 | 2.0% | IMPROVED | - | - | - |
| gemv/fp8_blockscale/32x7168x7168 | 13.90 | 13.12 | 1.059 | 2.0% | IMPROVED | - | - | - |
| gemv/fp8_blockscale/32x8192x8192 | 18.17 | 17.53 | 1.037 | 2.1% | IMPROVED | - | - | - |
| gemv/fp8_blockscale/4x4096x4096 | 8.93 | 8.15 | 1.096 | 2.0% | IMPROVED | - | - | - |
| gemv/fp8_blockscale/4x7168x7168 | 13.27 | 12.04 | 1.102 | 2.0% | IMPROVED | - | - | - |
| gemv/fp8_blockscale/4x8192x8192 | 18.19 | 17.06 | 1.066 | 2.0% | IMPROVED | - | - | - |
| gemv/fp8_blockscale/64x4096x4096 | 10.54 | 10.56 | 0.998 | 2.0% | OK | - | - | - |
| gemv/fp8_blockscale/64x7168x7168 | 14.79 | 14.71 | 1.005 | 2.0% | OK | - | - | - |
| gemv/fp8_blockscale/64x8192x8192 | 19.07 | 19.33 | 0.987 | 2.6% | OK | - | - | - |
| gemv/fp8_rowcol/16x4096x4096 | 7.76 | 6.96 | 1.115 | 2.0% | IMPROVED | - | - | - |
| gemv/fp8_rowcol/16x7168x7168 | 12.97 | 11.74 | 1.105 | 2.0% | IMPROVED | - | - | - |
| gemv/fp8_rowcol/16x8192x8192 | 17.72 | 16.27 | 1.089 | 2.0% | IMPROVED | - | - | - |
| gemv/fp8_rowcol/1x4096x4096 | 7.36 | 6.66 | 1.105 | 5.4% | IMPROVED | - | - | - |
| gemv/fp8_rowcol/1x7168x7168 | 12.65 | 11.24 | 1.125 | 2.0% | IMPROVED | - | - | - |
| gemv/fp8_rowcol/1x8192x8192 | 17.51 | 16.77 | 1.044 | 2.0% | IMPROVED | - | - | - |
| gemv/fp8_rowcol/32x4096x4096 | 8.11 | 7.67 | 1.057 | 2.0% | IMPROVED | - | - | - |
| gemv/fp8_rowcol/32x7168x7168 | 13.41 | 12.61 | 1.063 | 2.0% | IMPROVED | - | - | - |
| gemv/fp8_rowcol/32x8192x8192 | 17.99 | 17.06 | 1.055 | 2.0% | IMPROVED | - | - | - |
| gemv/fp8_rowcol/4x4096x4096 | 7.62 | 6.80 | 1.121 | 2.0% | IMPROVED | - | - | - |
| gemv/fp8_rowcol/4x7168x7168 | 12.74 | 11.47 | 1.111 | 2.0% | IMPROVED | - | - | - |
| gemv/fp8_rowcol/4x8192x8192 | 17.60 | 16.35 | 1.076 | 3.1% | IMPROVED | - | - | - |
| gemv/fp8_rowcol/64x4096x4096 | 9.21 | 9.23 | 0.998 | 2.0% | OK | - | - | - |
| gemv/fp8_rowcol/64x7168x7168 | 14.41 | 14.49 | 0.994 | 3.5% | OK | - | - | - |
| gemv/fp8_rowcol/64x8192x8192 | 18.30 | 18.20 | 1.005 | 2.1% | OK | - | - | - |
| gemv/i8_rowcol/16x4096x4096 | 7.93 | 7.14 | 1.111 | 2.0% | IMPROVED | - | - | - |
| gemv/i8_rowcol/16x7168x7168 | 13.13 | 11.94 | 1.100 | 2.0% | IMPROVED | - | - | - |
| gemv/i8_rowcol/16x8192x8192 | 17.88 | 16.67 | 1.073 | 3.2% | IMPROVED | - | - | - |
| gemv/i8_rowcol/1x4096x4096 | 7.63 | 6.90 | 1.106 | 2.0% | IMPROVED | - | - | - |
| gemv/i8_rowcol/1x7168x7168 | 12.70 | 11.55 | 1.100 | 2.0% | IMPROVED | - | - | - |
| gemv/i8_rowcol/1x8192x8192 | 17.80 | 16.86 | 1.056 | 2.0% | IMPROVED | - | - | - |
| gemv/i8_rowcol/32x4096x4096 | 8.27 | 7.84 | 1.055 | 2.0% | IMPROVED | - | - | - |
| gemv/i8_rowcol/32x7168x7168 | 13.55 | 12.73 | 1.064 | 2.0% | IMPROVED | - | - | - |
| gemv/i8_rowcol/32x8192x8192 | 18.06 | 17.42 | 1.037 | 3.1% | IMPROVED | - | - | - |
| gemv/i8_rowcol/4x4096x4096 | 7.77 | 6.97 | 1.115 | 2.0% | IMPROVED | - | - | - |
| gemv/i8_rowcol/4x7168x7168 | 12.85 | 11.64 | 1.104 | 2.8% | IMPROVED | - | - | - |
| gemv/i8_rowcol/4x8192x8192 | 17.68 | 16.50 | 1.072 | 2.5% | IMPROVED | - | - | - |
| gemv/i8_rowcol/64x4096x4096 | 9.36 | 9.39 | 0.997 | 2.0% | OK | - | - | - |
| gemv/i8_rowcol/64x7168x7168 | 14.57 | 14.62 | 0.997 | 2.0% | OK | - | - | - |
| gemv/i8_rowcol/64x8192x8192 | 18.38 | 18.59 | 0.989 | 2.0% | OK | - | - | - |