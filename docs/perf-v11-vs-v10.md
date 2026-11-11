# v11 vs v10: the same-run performance table (no drop after the restructure)

Evidence for "no performance drop after the four-layer restructure and the two-Weaver work": `tools/perf_ab.py` (the harness of [`perf-ab.md`](perf-ab.md)) builds the code objects of **v10.0-layers** (`7cd290af`) and of the current tree (**`d4349932`**: Main ac0d8e16 + the GEMM rule updates of `sched-rules`)
and runs EVERY integrated kernel family interleaved in one session on node 53, GPU 1 (no other GPU tenant): `perf_ab.py --old 7cd290af --new v11head --reps 3 --keep-going` (all cases, 3 interleaved repetitions each, floor 2 %, k 2.0). Each launch takes its own commit's dispatch
rule, so the rule changes are part of what is measured. Vendor columns (aiter) are the same-run vendor kernel where a host runs one. A case FAILS when new / old < 1 - max(2 %, 2 x the pair's noise); a family when its geometric mean falls below 1 - max(1 %, ...).

## Verdict

**Every family is OK (no family geometric mean below 1.0 beyond noise); 3 of the 231 case rows were below 1.0 beyond the 3-repetition noise, and all 3 are noise**: their kernels are byte-identical between the two trees (asm md5 equal in both `tests/golden/md5_manifest.tsv`, same dispatch), and a 9-repetition re-run of just those cases passes
(table below). 28 cases are IMPROVED (mostly GEMM, MLA and decode16).

| family | cases | geomean new/old | what moved |
|---|--:|--:|---|
| gemm | 79 | 1.040 | the composition-search rule updates (256-tile fill threshold, K-aware split-K, bf16 128 split form): 19 cases 1.04 - 1.33x |
| mla | 4 | 1.139 | the MLA Weaver / scheduler work: 1.07 - 1.23x |
| decode | 6 | 1.011 | decode16 (+2 - 4 % on three shapes) |
| fmha_mxfp8 | 4 | 1.015 | the `_vs128` replica configuration (+2.4 % on 1x8x2048) |
| fmha_hd192 | 8 | 1.004 | within noise |
| conv, fmha_fp8pt, fmha_bwd, fmha_bf16, moe, membound | 3 / 8 / 6 / 10 / 12 / 91 | 1.002 / 1.001 / 0.998 / 0.994 / 0.997 / 1.001 | within noise (every family inside its allowed drop) |

## The three cases flagged at 3 repetitions (investigated)

| case | 3 reps new/old | cause | 9 reps new/old |
|---|--:|---|--:|
| bf16/causal/1x32x8x2048x2048 | 0.975 | noise: `fmha_fwd_hd128_bf16_v3_causal` asm identical in both trees (md5 5d554e8f), same dispatch; vendor ratio of the same repetitions 0.979 / 1.005 | 1.005 (OK, allowed drop 4.3 %) |
| gemm/fp8_blockscale/16x8192x8192 | 0.979 | noise: `gemm_a8w8_blockscale_64_splitk` + `gemm_splitk_reduce_s4` asm identical, rule unchanged (a 16-row, 118 TFLOP/s kernel: 3-rep scatter exceeds 2 %) | 0.997 (OK, allowed drop 3.5 %) |
| gemm/i8_rowcol/576x7168x6144 | 0.967 | noise: `gemm_a8w8_i8_128` asm identical, the same tile / split under both rules (128 x 128, S1) | 1.008 (OK, allowed drop 2.2 %) |

Re-run (`--only` the three ids, 9 repetitions, verdict PASS):

| case | old | new | new/old | allowed drop | verdict | vendor | new/vendor | old/vendor |
|---|--:|--:|--:|--:|---|--:|--:|--:|
| bf16/causal/1x32x8x2048x2048 | 535.30 | 537.90 | 1.005 | 4.3% | OK | 540.00 | 0.996 | 0.991 |
| gemm/fp8_blockscale/16x8192x8192 | 117.50 | 117.20 | 0.997 | 3.5% | OK | - | - | - |
| gemm/i8_rowcol/576x7168x6144 | 1172 | 1181 | 1.008 | 2.2% | OK | - | - | - |


## What changed between the trees

235 kernels are in the golden md5 manifest; **205 have byte-identical assembly in v10 and v11**, 30 differ (1 new kernel): attn_combine_bf16_hd512, fmha_bwd_dkdv_hd128_bf16[_causal], fmha_bwd_dq_hd128_bf16[_causal], fmha_fwd_hd128_bf16_v3[_ks], fmha_fwd_hd128_fp8pt_v3[_causal][_kvsplit], fmha_fwd_hd128_mxfp8_v3_vs128, fmha_fwd_hd192_hd128_bf16_v3_causal[_ks], fmha_fwd_hd192_hd128_bf16_v3_ks, gelu_tanh_mul_bf16_d4096 / d7168, gemm_moe1_fp8_silu[_bm64], mla_decode_bf16_p1 / p16, moe_reduce_bf16_k8_d4096, pa_decode16_bf16_hd128_p16_c, probe_atomic_p1, probe_big_addr, quant_group128_bf16_fp8_n7168, rmsnorm_quant_fp8_n7168, silu_mul_bf16_d4096 / d7168, silu_mul_quant_fp8_d4096.
Everything else (every GEMM kernel, the MoE bf16 / fp4 stages, the paged decode fused form, the memory-bound families, conv) is the same machine code, so any difference there is dispatch (GEMM rules) or noise, and the table shows no drop.

## Families

| family | cases | geomean new/old | allowed drop | verdict |
|---|--:|--:|--:|---|
| conv | 3 | 1.002 | 1.0% | OK |
| decode | 6 | 1.011 | 1.0% | OK |
| fmha_bf16 | 10 | 0.994 | 1.5% | OK |
| fmha_bwd | 6 | 0.998 | 1.0% | OK |
| fmha_fp8pt | 8 | 1.001 | 1.0% | OK |
| fmha_hd192 | 8 | 1.004 | 1.3% | OK |
| fmha_mxfp8 | 4 | 1.015 | 2.3% | OK |
| gemm | 79 | 1.040 | 1.0% | OK |
| membound | 91 | 1.001 | 1.0% | OK |
| mla | 4 | 1.139 | 1.0% | OK |
| moe | 12 | 0.997 | 1.0% | OK |


## Cases

| case | old | new | new/old | allowed drop | verdict | vendor | new/vendor | old/vendor |
|---|--:|--:|--:|--:|---|--:|--:|--:|
| conv/conv2d_bf16_r1s1_st1_p0_d1 | 337.70 | 338.00 | 1.001 | 2.0% | OK | - | - | - |
| conv/conv2d_bf16_r3s3_st1_p1_d1 | 370.60 | 371.40 | 1.002 | 2.0% | OK | - | - | - |
| conv/conv3d_bf16_t3r3s3_st1_p1_d1 | 386.80 | 388.10 | 1.003 | 2.0% | OK | - | - | - |
| decode16/B1H32HK4C32768 | 114.69 | 111.79 | 1.026 | 2.0% | IMPROVED | - | - | - |
| decode16/B1H64HK8C8192 | 29.41 | 28.80 | 1.021 | 2.0% | IMPROVED | - | - | - |
| decode16/B32H64HK8C2048 | 35.42 | 35.64 | 0.994 | 2.0% | OK | - | - | - |
| decode16/B4H16HK1C16384 | 56.95 | 54.91 | 1.037 | 2.0% | IMPROVED | - | - | - |
| decode16/B64H32HK4C1024 | 20.06 | 20.16 | 0.995 | 2.0% | OK | - | - | - |
| decode16/B8H64HK8C4096 | 26.53 | 26.75 | 0.992 | 2.0% | OK | - | - | - |
| bf16/causal/1x32x8x2048x2048 | 537.90 | 524.20 | 0.975 | 2.0% | FAIL | 535.20 | 0.979 | 1.005 |
| bf16/causal/1x8x8x4096x4096 | 453.70 | 466.60 | 1.028 | 3.6% | OK | 382.50 | 1.220 | 1.186 |
| bf16/causal/2x32x32x8192x8192 | 982.60 | 984.30 | 1.002 | 2.0% | OK | 1122 | 0.877 | 0.876 |
| bf16/causal/4x16x16x1024x1024 | 419.90 | 419.40 | 0.999 | 4.8% | OK | 415.60 | 1.009 | 1.010 |
| bf16/causal/4x16x16x2048x2048 | 635.70 | 629.10 | 0.990 | 4.1% | OK | 776.00 | 0.811 | 0.819 |
| bf16/full/1x32x8x2048x2048 | 832.40 | 826.70 | 0.993 | 10.6% | OK | 972.80 | 0.850 | 0.856 |
| bf16/full/1x8x8x4096x4096 | 747.50 | 743.20 | 0.994 | 3.0% | OK | 701.70 | 1.059 | 1.065 |
| bf16/full/2x32x32x8192x8192 | 1108 | 1111 | 1.003 | 2.0% | OK | 1181 | 0.941 | 0.938 |
| bf16/full/4x16x16x1024x1024 | 745.20 | 725.30 | 0.973 | 3.7% | OK | 868.90 | 0.835 | 0.858 |
| bf16/full/4x16x16x2048x2048 | 817.50 | 804.60 | 0.984 | 4.5% | OK | 896.90 | 0.897 | 0.911 |
| bwd/causal/1x16x16x1024x1024 | 133.80 | 133.20 | 0.996 | 2.0% | OK | 100.20 | 1.329 | 1.335 |
| bwd/causal/1x16x4x1024x1024 | 133.90 | 133.20 | 0.995 | 2.0% | OK | 99.90 | 1.333 | 1.340 |
| bwd/causal/2x8x8x1024x1024 | 135.60 | 135.40 | 0.999 | 2.0% | OK | 100.00 | 1.354 | 1.356 |
| bwd/full/1x16x16x1024x1024 | 252.70 | 253.60 | 1.004 | 2.0% | OK | 255.60 | 0.992 | 0.989 |
| bwd/full/1x16x4x1024x1024 | 256.00 | 255.60 | 0.998 | 2.0% | OK | 256.30 | 0.997 | 0.999 |
| bwd/full/2x8x8x1024x1024 | 256.10 | 255.50 | 0.998 | 2.0% | OK | 256.20 | 0.997 | 1.000 |
| fp8pt/causal/1x8x8x4096x4096 | 871.10 | 867.10 | 0.995 | 2.0% | OK | 504.30 | 1.719 | 1.727 |
| fp8pt/causal/2x32x32x8192x8192 | 2046 | 2049 | 1.001 | 2.0% | OK | 1829 | 1.120 | 1.119 |
| fp8pt/causal/2x32x8x4096x4096 | 1854 | 1846 | 0.996 | 2.0% | OK | 1674 | 1.103 | 1.108 |
| fp8pt/causal/4x16x16x2048x2048 | 1306 | 1322 | 1.012 | 2.0% | OK | 1220 | 1.084 | 1.071 |
| fp8pt/full/1x8x8x4096x4096 | 1620 | 1614 | 0.996 | 2.0% | OK | 1092 | 1.478 | 1.484 |
| fp8pt/full/2x32x32x8192x8192 | 2448 | 2450 | 1.001 | 2.0% | OK | 2059 | 1.190 | 1.189 |
| fp8pt/full/2x32x8x4096x4096 | 2328 | 2330 | 1.001 | 2.0% | OK | 1981 | 1.176 | 1.175 |
| fp8pt/full/4x16x16x2048x2048 | 1971 | 1976 | 1.003 | 5.0% | OK | 1624 | 1.216 | 1.213 |
| hd192/causal/1x128x128x1024x1024 | 643.50 | 640.20 | 0.995 | 7.0% | OK | 654.00 | 0.979 | 0.984 |
| hd192/causal/1x16x16x4096x4096 | 721.90 | 728.50 | 1.009 | 2.0% | OK | 924.70 | 0.788 | 0.781 |
| hd192/causal/1x8x8x4096x4096 | 583.90 | 579.70 | 0.993 | 3.4% | OK | 619.50 | 0.936 | 0.943 |
| hd192/causal/4x16x16x2048x2048 | 818.10 | 817.40 | 0.999 | 2.6% | OK | 820.30 | 0.996 | 0.997 |
| hd192/full/1x128x128x1024x1024 | 882.70 | 896.20 | 1.015 | 2.0% | OK | 859.60 | 1.043 | 1.027 |
| hd192/full/1x16x16x4096x4096 | 1134 | 1134 | 1.000 | 2.0% | OK | 1178 | 0.963 | 0.963 |
| hd192/full/1x8x8x4096x4096 | 941.70 | 950.70 | 1.010 | 6.3% | OK | 1183 | 0.804 | 0.796 |
| hd192/full/4x16x16x2048x2048 | 1018 | 1033 | 1.015 | 2.0% | OK | 1084 | 0.953 | 0.939 |
| mxfp8/1x8x2048 | 674.26 | 690.26 | 1.024 | 2.0% | IMPROVED | 711.91 | 0.970 | 0.947 |
| mxfp8/1x8x8192 | 2344 | 2353 | 1.004 | 2.0% | OK | 2441 | 0.964 | 0.960 |
| mxfp8/2x32x8192 | 2488 | 2511 | 1.009 | 2.0% | OK | 2535 | 0.990 | 0.982 |
| mxfp8/4x8x2048 | 1728 | 1767 | 1.023 | 8.8% | OK | 1855 | 0.952 | 0.931 |
| gemm/bf16/1024x7168x2048 | 839.00 | 836.70 | 0.997 | 3.7% | OK | - | - | - |
| gemm/bf16/1152x4096x6144 | 737.90 | 724.60 | 0.982 | 4.3% | OK | - | - | - |
| gemm/bf16/1280x7168x4096 | 771.10 | 915.10 | 1.187 | 3.2% | IMPROVED | - | - | - |
| gemm/bf16/128x7168x8192 | 471.70 | 519.20 | 1.101 | 3.7% | IMPROVED | - | - | - |
| gemm/bf16/160x4096x8192 | 405.70 | 416.90 | 1.028 | 3.1% | OK | - | - | - |
| gemm/bf16/16x8192x8192 | 83.80 | 84.20 | 1.005 | 2.0% | OK | - | - | - |
| gemm/bf16/1792x8192x4096 | 943.80 | 1205 | 1.277 | 3.4% | IMPROVED | - | - | - |
| gemm/bf16/192x4096x4096 | 347.90 | 386.90 | 1.112 | 3.3% | IMPROVED | - | - | - |
| gemm/bf16/2048x7168x7168 | 1027 | 1292 | 1.259 | 2.0% | IMPROVED | - | - | - |
| gemm/bf16/256x4096x8192 | 574.70 | 571.30 | 0.994 | 3.1% | OK | - | - | - |
| gemm/bf16/256x7168x2048 | 586.50 | 584.10 | 0.996 | 2.0% | OK | - | - | - |
| gemm/bf16/320x8192x6144 | 544.70 | 581.30 | 1.067 | 2.5% | IMPROVED | - | - | - |
| gemm/bf16/32x4096x14336 | 166.70 | 166.40 | 0.998 | 2.5% | OK | - | - | - |
| gemm/bf16/4096x4096x8192 | 1441 | 1438 | 0.998 | 2.2% | OK | - | - | - |
| gemm/bf16/576x7168x6144 | 653.70 | 661.40 | 1.012 | 3.6% | OK | - | - | - |
| gemm/bf16/64x8192x8192 | 322.00 | 323.70 | 1.005 | 2.7% | OK | - | - | - |
| gemm/bf16/768x4096x8192 | 687.60 | 714.70 | 1.039 | 5.4% | OK | - | - | - |
| gemm/bf16/8192x8192x8192 | 1485 | 1484 | 0.999 | 2.0% | OK | - | - | - |
| gemm/bf16/96x8192x8192 | 437.90 | 434.80 | 0.993 | 2.1% | OK | - | - | - |
| gemm/fp8_blockscale/1024x7168x2048 | 929.20 | 949.30 | 1.022 | 5.7% | OK | - | - | - |
| gemm/fp8_blockscale/1152x4096x6144 | 921.90 | 910.70 | 0.988 | 2.0% | OK | - | - | - |
| gemm/fp8_blockscale/1280x7168x4096 | 851.80 | 848.20 | 0.996 | 2.0% | OK | - | - | - |
| gemm/fp8_blockscale/128x7168x8192 | 620.50 | 623.10 | 1.004 | 4.4% | OK | - | - | - |
| gemm/fp8_blockscale/160x4096x8192 | 497.00 | 498.30 | 1.003 | 2.8% | OK | - | - | - |
| gemm/fp8_blockscale/16x8192x8192 | 117.90 | 115.40 | 0.979 | 2.0% | FAIL | - | - | - |
| gemm/fp8_blockscale/1792x8192x4096 | 1105 | 1109 | 1.003 | 2.0% | OK | - | - | - |
| gemm/fp8_blockscale/192x4096x4096 | 348.10 | 355.60 | 1.022 | 3.0% | OK | - | - | - |
| gemm/fp8_blockscale/2048x7168x7168 | 1186 | 1190 | 1.003 | 2.0% | OK | - | - | - |
| gemm/fp8_blockscale/256x4096x8192 | 721.80 | 720.50 | 0.998 | 2.0% | OK | - | - | - |
| gemm/fp8_blockscale/256x7168x2048 | 712.70 | 712.60 | 1.000 | 2.0% | OK | - | - | - |
| gemm/fp8_blockscale/320x8192x6144 | 817.00 | 820.80 | 1.005 | 2.0% | OK | - | - | - |
| gemm/fp8_blockscale/32x4096x14336 | 207.30 | 206.10 | 0.994 | 2.0% | OK | - | - | - |
| gemm/fp8_blockscale/4096x4096x8192 | 1330 | 1328 | 0.998 | 2.1% | OK | - | - | - |
| gemm/fp8_blockscale/576x7168x6144 | 1036 | 1057 | 1.021 | 2.0% | IMPROVED | - | - | - |
| gemm/fp8_blockscale/64x8192x8192 | 450.30 | 446.40 | 0.991 | 2.0% | OK | - | - | - |
| gemm/fp8_blockscale/768x4096x8192 | 944.50 | 937.80 | 0.993 | 2.0% | OK | - | - | - |
| gemm/fp8_blockscale/8192x8192x8192 | 1500 | 1506 | 1.004 | 2.0% | OK | - | - | - |
| gemm/fp8_blockscale/96x8192x8192 | 557.90 | 542.90 | 0.973 | 2.9% | OK | - | - | - |
| gemm/fp8_rowcol/1024x7168x2048 | 1454 | 1437 | 0.988 | 2.0% | OK | - | - | - |
| gemm/fp8_rowcol/1152x4096x6144 | 1276 | 1251 | 0.980 | 8.8% | OK | - | - | - |
| gemm/fp8_rowcol/1280x7168x4096 | 1362 | 1577 | 1.157 | 6.0% | IMPROVED | - | - | - |
| gemm/fp8_rowcol/128x7168x8192 | 703.30 | 819.80 | 1.166 | 5.7% | IMPROVED | - | - | - |
| gemm/fp8_rowcol/160x4096x8192 | 605.00 | 674.20 | 1.114 | 5.7% | IMPROVED | - | - | - |
| gemm/fp8_rowcol/16x8192x8192 | 120.80 | 120.50 | 0.998 | 2.3% | OK | - | - | - |
| gemm/fp8_rowcol/1792x8192x4096 | 1641 | 2095 | 1.277 | 11.3% | IMPROVED | - | - | - |
| gemm/fp8_rowcol/192x4096x4096 | 424.50 | 548.80 | 1.293 | 2.0% | IMPROVED | - | - | - |
| gemm/fp8_rowcol/2048x7168x7168 | 1774 | 2253 | 1.270 | 2.1% | IMPROVED | - | - | - |
| gemm/fp8_rowcol/256x4096x8192 | 935.00 | 927.00 | 0.991 | 4.3% | OK | - | - | - |
| gemm/fp8_rowcol/256x7168x2048 | 913.10 | 910.80 | 0.997 | 2.0% | OK | - | - | - |
| gemm/fp8_rowcol/320x8192x6144 | 1031 | 1029 | 0.999 | 6.4% | OK | - | - | - |
| gemm/fp8_rowcol/32x4096x14336 | 264.00 | 262.60 | 0.995 | 2.0% | OK | - | - | - |
| gemm/fp8_rowcol/4096x4096x8192 | 2503 | 2468 | 0.986 | 3.0% | OK | - | - | - |
| gemm/fp8_rowcol/576x7168x6144 | 1158 | 1162 | 1.004 | 2.0% | OK | - | - | - |
| gemm/fp8_rowcol/64x8192x8192 | 467.60 | 468.30 | 1.001 | 2.5% | OK | - | - | - |
| gemm/fp8_rowcol/768x4096x8192 | 1203 | 1208 | 1.004 | 3.6% | OK | - | - | - |
| gemm/fp8_rowcol/8192x8192x8192 | 2879 | 2875 | 0.999 | 2.0% | OK | - | - | - |
| gemm/fp8_rowcol/96x8192x8192 | 661.10 | 670.40 | 1.014 | 2.0% | OK | - | - | - |
| gemm/i8_rowcol/1024x7168x2048 | 1409 | 1390 | 0.987 | 12.9% | OK | - | - | - |
| gemm/i8_rowcol/1152x4096x6144 | 1255 | 1265 | 1.008 | 6.8% | OK | - | - | - |
| gemm/i8_rowcol/1280x7168x4096 | 1337 | 1524 | 1.140 | 2.0% | IMPROVED | - | - | - |
| gemm/i8_rowcol/128x7168x8192 | 689.80 | 833.20 | 1.208 | 3.5% | IMPROVED | - | - | - |
| gemm/i8_rowcol/160x4096x8192 | 593.20 | 658.20 | 1.110 | 2.0% | IMPROVED | - | - | - |
| gemm/i8_rowcol/16x8192x8192 | 120.20 | 120.90 | 1.006 | 2.3% | OK | - | - | - |
| gemm/i8_rowcol/1792x8192x4096 | 1643 | 2029 | 1.235 | 2.0% | IMPROVED | - | - | - |
| gemm/i8_rowcol/192x4096x4096 | 404.20 | 535.00 | 1.324 | 2.0% | IMPROVED | - | - | - |
| gemm/i8_rowcol/2048x7168x7168 | 1772 | 2200 | 1.242 | 2.0% | IMPROVED | - | - | - |
| gemm/i8_rowcol/256x4096x8192 | 940.00 | 932.90 | 0.992 | 2.2% | OK | - | - | - |
| gemm/i8_rowcol/256x7168x2048 | 891.80 | 892.00 | 1.000 | 2.0% | OK | - | - | - |
| gemm/i8_rowcol/320x8192x6144 | 1036 | 1028 | 0.992 | 5.0% | OK | - | - | - |
| gemm/i8_rowcol/32x4096x14336 | 260.80 | 261.10 | 1.001 | 5.0% | OK | - | - | - |
| gemm/i8_rowcol/4096x4096x8192 | 2444 | 2422 | 0.991 | 4.3% | OK | - | - | - |
| gemm/i8_rowcol/576x7168x6144 | 1189 | 1150 | 0.967 | 2.0% | FAIL | - | - | - |
| gemm/i8_rowcol/64x8192x8192 | 465.70 | 464.40 | 0.997 | 2.0% | OK | - | - | - |
| gemm/i8_rowcol/768x4096x8192 | 1235 | 1212 | 0.981 | 2.0% | OK | - | - | - |
| gemm/i8_rowcol/8192x8192x8192 | 2830 | 2824 | 0.998 | 2.0% | OK | - | - | - |
| gemm/i8_rowcol/96x8192x8192 | 664.60 | 662.70 | 0.997 | 2.0% | OK | - | - | - |
| gemm/mxfp8_v4/1024x4096x8192 | 989.49 | 990.34 | 1.001 | 2.0% | OK | - | - | - |
| gemm/mxfp8_v4/4096x4096x8192 | 3007 | 3007 | 1.000 | 4.7% | OK | - | - | - |
| gemm/mxfp8_v4/8192x8192x8192 | 3358 | 3364 | 1.002 | 2.0% | OK | - | - | - |
| mb/add_bf16 | 6355 | 6368 | 1.002 | 2.0% | OK | - | - | - |
| mb/add_rmsnorm n4096 | 5770 | 5744 | 0.996 | 2.0% | OK | - | - | - |
| mb/add_rmsnorm_fp8 n1000 | 3814 | 3820 | 1.002 | 2.0% | OK | - | - | - |
| mb/add_rmsnorm_fp8 n4096 | 4260 | 4252 | 0.998 | 3.5% | OK | - | - | - |
| mb/argmax v128256 B=4096 | 4975 | 5003 | 1.006 | 2.0% | OK | - | - | - |
| mb/argmax v32000 B=16384 | 4364 | 4409 | 1.010 | 2.5% | OK | - | - | - |
| mb/cache h8 d128 | 5059 | 4970 | 0.982 | 25.4% | OK | - | - | - |
| mb/cache_blocked fp8 per-token h8 d128 b16 (i64 slots) | 922.90 | 919.00 | 0.996 | 2.0% | OK | - | - | - |
| mb/cache_blocked h8 d128 b16 contiguous slots | 1420 | 1421 | 1.001 | 2.0% | OK | - | - | - |
| mb/cache_blocked h8 d128 b16 random slots | 868.10 | 867.40 | 0.999 | 2.0% | OK | - | - | - |
| mb/cache_fp8 h8 d128 | 5989 | 5935 | 0.991 | 2.0% | OK | - | - | - |
| mb/conv1d_fwd split w4 silu B=128 L=512 uniform | 2602 | 2592 | 0.996 | 2.0% | OK | - | - | - |
| mb/conv1d_fwd split w4 silu ragged + states | 395.00 | 393.90 | 0.997 | 2.0% | OK | - | - | - |
| mb/conv1d_fwd w4 silu B=128 L=512 uniform | 2800 | 2808 | 1.003 | 2.0% | OK | - | - | - |
| mb/conv1d_fwd w4 silu ragged B=256 (_sched) | 2645 | 2656 | 1.004 | 2.0% | OK | - | - | - |
| mb/conv1d_fwd w4 silu ragged B=256 (grid) | 996.50 | 996.10 | 1.000 | 2.0% | OK | - | - | - |
| mb/conv1d_update w4 s3 idx silu | 3084 | 3101 | 1.005 | 2.0% | OK | - | - | - |
| mb/cvt_bf16_f32 | 6339 | 6316 | 0.996 | 2.0% | OK | - | - | - |
| mb/cvt_f32_bf16 | 6593 | 6594 | 1.000 | 2.0% | OK | - | - | - |
| mb/dequant group128 n4096 | 5998 | 6008 | 1.002 | 2.0% | OK | - | - | - |
| mb/dequant static | 6417 | 6412 | 0.999 | 2.0% | OK | - | - | - |
| mb/dequant token n1000 (odd tail) | 5232 | 5263 | 1.006 | 3.6% | OK | - | - | - |
| mb/dequant token n4096 | 6014 | 5936 | 0.987 | 2.0% | OK | - | - | - |
| mb/gated_rmsnorm bf16 n128 (heads) | 6264 | 6287 | 1.004 | 3.2% | OK | - | - | - |
| mb/gated_rmsnorm bf16 n4096 | 6041 | 6005 | 0.994 | 2.0% | OK | - | - | - |
| mb/gated_rmsnorm fp8 n128 (heads) | 5706 | 5742 | 1.006 | 2.0% | OK | - | - | - |
| mb/gated_rmsnorm fp8 n4096 | 5324 | 5330 | 1.001 | 2.0% | OK | - | - | - |
| mb/gelu (erf) unary | 3021 | 3042 | 1.007 | 2.4% | OK | - | - | - |
| mb/gelu_mul (erf) d4096 | 4072 | 4072 | 1.000 | 2.0% | OK | - | - | - |
| mb/gelu_mul_bias d2048 E=64 | 5034 | 5088 | 1.011 | 2.0% | OK | - | - | - |
| mb/gelu_tanh (unary) | 5560 | 5570 | 1.002 | 2.0% | OK | - | - | - |
| mb/gelu_tanh d4096 | 6505 | 6485 | 0.997 | 2.0% | OK | - | - | - |
| mb/layernorm n4096 | 5546 | 5614 | 1.012 | 2.0% | OK | - | - | - |
| mb/layernorm n776 | 4449 | 4482 | 1.007 | 2.0% | OK | - | - | - |
| mb/mla cache flat bf16 r512 p64 | 4267 | 4276 | 1.002 | 2.0% | OK | - | - | - |
| mb/mla cache seg fp8 r512 p64 b64 (i64) | 3212 | 3202 | 0.997 | 2.0% | OK | - | - | - |
| mb/moe_combine k8 d4096 T=32768 | 5899 | 5983 | 1.014 | 2.0% | OK | - | - | - |
| mb/moe_fused_gate e256 g8 tg4 k8 M=1001 (odd) | 70.50 | 70.70 | 1.003 | 2.0% | OK | - | - | - |
| mb/moe_fused_gate e256 g8 tg4 k8 M=131072 | 801.50 | 805.40 | 1.005 | 2.0% | OK | - | - | - |
| mb/moe_reduce k8 d4096 T=32768 | 5958 | 5912 | 0.992 | 4.3% | OK | - | - | - |
| mb/moe_reduce k8 d7168 T=16384 | 6094 | 6083 | 0.998 | 5.5% | OK | - | - | - |
| mb/moe_sorting e128 k8 b32 M=1001 skew+invalid | 14.78 | 14.81 | 0.998 | 2.0% | OK | - | - | - |
| mb/moe_sorting e128 k8 b32 M=37 | 12.07 | 12.06 | 1.001 | 2.0% | OK | - | - | - |
| mb/moe_sorting e256 k8 b128 M=4096 | 52.45 | 52.61 | 0.997 | 2.0% | OK | - | - | - |
| mb/moe_sorting e385 k6 b32 M=777 | 17.88 | 18.01 | 0.993 | 2.0% | OK | - | - | - |
| mb/moe_sorting e64 k2 b16 M=513 | 10.48 | 10.47 | 1.001 | 2.0% | OK | - | - | - |
| mb/mxfp6 e2m3 k4096 M=32768 | 2903 | 2935 | 1.011 | 2.0% | OK | - | - | - |
| mb/mxfp6 e3m2 k4096 M=32768 | 2921 | 2919 | 0.999 | 2.0% | OK | - | - | - |
| mb/qknorm_rope neox h40 d128 | 4217 | 4222 | 1.001 | 2.0% | OK | - | - | - |
| mb/qknorm_rope_kvcache neox bf16 h8 d128 | 5838 | 5829 | 0.998 | 2.5% | OK | - | - | - |
| mb/qknorm_rope_kvcache neox fp8 h8 d128 (i64) | 4769 | 4767 | 1.000 | 2.4% | OK | - | - | - |
| mb/quant_group128 n4096 | 6469 | 6488 | 1.003 | 2.0% | OK | - | - | - |
| mb/quant_static | 6143 | 6140 | 1.000 | 2.0% | OK | - | - | - |
| mb/quant_token n1000 | 6020 | 6040 | 1.003 | 2.0% | OK | - | - | - |
| mb/quant_token n14336 (long row) | 5969 | 5975 | 1.001 | 2.0% | OK | - | - | - |
| mb/quant_token n32768 (long row) | 4372 | 4386 | 1.003 | 2.0% | OK | - | - | - |
| mb/quant_token n4096 | 6375 | 6371 | 0.999 | 2.0% | OK | - | - | - |
| mb/quant_token_int8 n1000 | 5478 | 5453 | 0.995 | 2.0% | OK | - | - | - |
| mb/quant_token_int8 n14336 (long row) | 5822 | 5821 | 1.000 | 2.0% | OK | - | - | - |
| mb/quant_token_int8 n4096 | 6286 | 6307 | 1.003 | 2.0% | OK | - | - | - |
| mb/relu2 (unary) | 6410 | 6490 | 1.012 | 2.2% | OK | - | - | - |
| mb/rmsnorm n1000 | 5699 | 5704 | 1.001 | 2.0% | OK | - | - | - |
| mb/rmsnorm n200 | 5306 | 5278 | 0.995 | 2.0% | OK | - | - | - |
| mb/rmsnorm n4096 | 6197 | 6141 | 0.991 | 4.4% | OK | - | - | - |
| mb/rmsnorm_quant_mxfp4 k4096 M=32768 | 2774 | 2796 | 1.008 | 2.0% | OK | - | - | - |
| mb/rope_gptj h32 d128 | 6325 | 6345 | 1.003 | 2.0% | OK | - | - | - |
| mb/rope_neox h32 d128 | 6431 | 6418 | 0.998 | 2.0% | OK | - | - | - |
| mb/rope_neox h8 d128 | 6818 | 6814 | 0.999 | 2.0% | OK | - | - | - |
| mb/sample v128256 B=4096 | 724.60 | 724.10 | 0.999 | 2.0% | OK | - | - | - |
| mb/sample v32000 B=16384 | 1292 | 1290 | 0.998 | 2.0% | OK | - | - | - |
| mb/sigmoid (unary) | 6336 | 6381 | 1.007 | 2.0% | OK | - | - | - |
| mb/silu (unary) | 6119 | 6154 | 1.006 | 2.0% | OK | - | - | - |
| mb/silu_mul d1024 | 6690 | 6698 | 1.001 | 2.0% | OK | - | - | - |
| mb/silu_mul d14336 | 6526 | 6527 | 1.000 | 2.0% | OK | - | - | - |
| mb/silu_mul_bias d4096 E=64 | 8846 | 9044 | 1.022 | 2.0% | IMPROVED | - | - | - |
| mb/silu_mul_limit d4096 (limit 7) | 6544 | 6565 | 1.003 | 2.0% | OK | - | - | - |
| mb/silu_mul_quant d1000 | 5967 | 5925 | 0.993 | 5.1% | OK | - | - | - |
| mb/silu_mul_quant d14336 (long row) | 3881 | 3870 | 0.997 | 2.0% | OK | - | - | - |
| mb/silu_mul_quant d32768 (long row) | 3228 | 3241 | 1.004 | 2.0% | OK | - | - | - |
| mb/silu_mul_quant d4096 | 5941 | 5875 | 0.989 | 4.1% | OK | - | - | - |
| mb/situv2_mul_quant d14336 (long row) | 2796 | 2821 | 1.009 | 2.0% | OK | - | - | - |
| mb/situv2_mul_quant d4096 | 4593 | 4557 | 0.992 | 2.0% | OK | - | - | - |
| mb/softmax n1000 | 5635 | 5613 | 0.996 | 2.0% | OK | - | - | - |
| mb/softmax n4096 | 5972 | 5988 | 1.003 | 2.0% | OK | - | - | - |
| mb/softmax n8192 | 5536 | 5492 | 0.992 | 2.0% | OK | - | - | - |
| mb/swiglu_oai_mul d4096 | 6512 | 6544 | 1.005 | 2.0% | OK | - | - | - |
| mb/topk e128 k8 T=65536 | 1221 | 1219 | 0.998 | 2.0% | OK | - | - | - |
| mb/topk renorm e256 k8 T=65536 | 1034 | 1032 | 0.999 | 2.0% | OK | - | - | - |
| mb/topk_per_row decode k2048 B=256 L<=64K | 328.80 | 329.00 | 1.001 | 2.0% | OK | - | - | - |
| mb/topk_per_row decode k2048 odd (ragged, ties) | 33.10 | 33.40 | 1.009 | 2.7% | OK | - | - | - |
| mb/topk_per_row prefill k128 (ties) | 25.00 | 25.10 | 1.004 | 2.4% | OK | - | - | - |
| mla/B1H128C8192 | 54.22 | 50.63 | 1.071 | 2.0% | IMPROVED | - | - | - |
| mla/B32H16C2048 | 45.08 | 37.88 | 1.190 | 2.0% | IMPROVED | - | - | - |
| mla/B4H16C16384 | 80.87 | 65.88 | 1.228 | 2.0% | IMPROVED | - | - | - |
| mla/B8H128C4096 | 53.69 | 49.99 | 1.074 | 2.0% | IMPROVED | - | - | - |
| moe/bf16/M256D2048I768E64K8#stage1 | 84.56 | 85.56 | 0.988 | 2.0% | OK | - | - | - |
| moe/bf16/M256D2048I768E64K8#stage2 | 44.66 | 44.87 | 0.995 | 2.0% | OK | - | - | - |
| moe/bf16/M4096D2048I768E64K8#stage1 | 364.41 | 364.77 | 0.999 | 2.0% | OK | - | - | - |
| moe/bf16/M4096D2048I768E64K8#stage2 | 263.31 | 263.42 | 1.000 | 2.0% | OK | - | - | - |
| moe/fp4/M2048D7168I256E64K8#stage1 | 106.10 | 106.56 | 0.996 | 2.7% | OK | - | - | - |
| moe/fp4/M2048D7168I256E64K8#stage2 | 287.27 | 286.82 | 1.002 | 2.0% | OK | - | - | - |
| moe/fp4/M256D7168I256E64K8#stage1 | 40.83 | 41.08 | 0.994 | 2.0% | OK | - | - | - |
| moe/fp4/M256D7168I256E64K8#stage2 | 41.65 | 42.29 | 0.985 | 2.0% | OK | - | - | - |
| moe/fp8/M2048D4096I256E32K8#stage1 | 83.91 | 83.99 | 0.999 | 2.0% | OK | - | - | - |
| moe/fp8/M2048D4096I256E32K8#stage2 | 164.30 | 164.57 | 0.998 | 2.0% | OK | - | - | - |
| moe/fp8/M512D4096I256E32K8#stage1 | 37.16 | 36.96 | 1.005 | 2.0% | OK | - | - | - |
| moe/fp8/M512D4096I256E32K8#stage2 | 50.40 | 50.10 | 1.006 | 2.0% | OK | - | - | - |
