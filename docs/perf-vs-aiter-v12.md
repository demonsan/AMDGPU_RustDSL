# Performance vs aiter, v12 (Main `4ec08dac`, 2026-11-15)

Every number here was measured in the same process on the same data, on MI355X / gfx950 (`mi355-gpu-53`, GPU 0). Ratio = aiter (vendor) time / ours, so above 1 means ours is faster.
The v9.2 column is the previous full vendor table (2026-11-09; [`op-coverage-design.md`](op-coverage-design.md) sections 7.4 - 7.7, [`current-status.md`](current-status.md) section 1.1).
The shapes are those of that table.

Sections by owner:
- kernel-opt: the shipped MXFP8 FMHA, the GEMM family, attention prefill (fp8 per-tensor, bf16 v3 hd128, hd192), attention backward and fused MoE;
- sched: the other families (appended below).

## 1. GEMM family (`tools/vs_aiter_gemm.py`)

- Ours: the kernel `amdc dispatch gemm` picks (own backend, with its split-K / reduce).
- aiter: every code object of the family's `hsa/gfx950` directory that `tests/gpu/vendor_gemm_host.cpp` drives; the fastest is reported, with its tile.
- Both sides: the median of 3 runs of 100 iterations, all `bad=0`.

| family | shape (M x N x K) | ours us (kernel) | aiter us (best tile) | ratio | v9.2 |
|---|---|--:|--:|--:|--:|
| MXFP4 (a4w4) | 8192 x 8192 x 8192 | 255.4 - 256.0 (`gemm_a4w4_mxfp4`) | 228.1 - 229.2 (f4gemm 256x256) | 0.89 - 0.90 | 0.90 |
| MXFP4 (a4w4) | 4096 x 4096 x 4096 | 41.3 - 42.6 (`gemm_a4w4_mxfp4`) | 36.6 - 37.5 (f4gemm 128x512 / 256x256) | 0.88 - 0.91 | 0.86 |
| int8 per-token | 8192 x 8192 x 8192 | 387.2 (`gemm_a8w8_i8`) | 619.5 (i8gemm 160x128) | 1.60 | 1.53 |
| int8 per-token | 4096 x 4096 x 4096 | 63.4 (`gemm_a8w8_i8`) | 105.3 (i8gemm 128x128) | 1.66 | 1.63 |
| int8 per-token | 1024 x 4096 x 8192 | 50.6 (`gemm_a8w8_i8_128`) | 50.6 (i8gemm 128x128) | 1.00 | 1.08 |
| int8 per-token | 256 x 7168 x 4096 | 13.7 (`gemm_a8w8_i8_64`) | 14.2 (i8gemm 64x128) | 1.03 | 1.01 |
| int8 per-token | 128 x 4096 x 8192 | 13.7 (`_64_splitk`, S=4) | 17.6 (i8gemm 16x128) | 1.28 | 1.27 |
| fp8 blockscale | 4096 x 4096 x 4096 | 115.6 (`gemm_a8w8_blockscale`) | 120.7 (128x128) | 1.04 | 1.05 |
| fp8 blockscale | 1024 x 4096 x 8192 | 61.9 (`_64`) | 60.5 (128x128) | 0.98 | 0.98 |
| fp8 blockscale | 512 x 7168 x 7168 | 51.3 (`_64`) | 50.2 (128x128) | 0.98 | 1.01 |
| fp8 blockscale | 256 x 7168 x 4096 | 18.3 (`_64`) | 17.1 (64x128) | 0.93 | 0.91 |
| fp8 blockscale | 128 x 4096 x 8192 | 15.3 (`_64_splitk`, S=4) | 22.2 (32x128) | 1.45 | 1.45 |
| bf16 | 2048 x 4096 x 4096 | 74.0 (`gemm_a16w16_bf16_128`) | 76.1 (bf16gemm 256x256 bpreshuffle) | 1.03 | 0.96 |
| bf16 | 128 x 4096 x 8192 | 19.0 (`_64_splitk`, S=4) | 116.0 (bf16gemm 256x256 bpreshuffle) | 6.1 | - |

Notes:
- **bf16 small M**: aiter's split-K bf16 code objects (`bf16gemm_fp32bf16_*_splitk`) are not driven by the host, so the 6.1x compares our split-K against aiter's non-split kernel. It is not a like-for-like number. v9.2 left it out for the same reason.
- MXFP4: the ranges are three separate re-runs (5 runs each). A first sweep had measured 311 us for 8192^3; see section 6.

## 2. Shipped MXFP8 FMHA (`tools/port_gate.sh fmha_fwd_hd128_mxfp8_v3`, OVERALL PASS)

- aiter: the vendor `fmha_fwd_hd128_mxfp8` code object in the same process (`ref_host`).
- Each value is the median of 5 runs; v9.2 is the last `port_gate` of that round.

| path | shape | ours TF/s | ratio | v9.2 |
|---|---|--:|--:|--:|
| default (KV-split main + combine) | B1 H8 S2048, real data | 768.0 | **1.063** | 1.028 |
| `_vs128` per CU (64 blocks) | real data | 733.4 | **1.017** | 0.952 |
| `_vs128` | B4 H8 S2048 | 1945.7 | **1.033** | 0.955 |
| `_vs128` | B1 H8 S8192 | 2567.5 | **1.011** | 0.981 |
| `_vs128` | B2 H8 S2048 | 1334.3 | **1.022** | - |
| `_vs128` per CU, median over 8 layouts | | 732.0 | 1.011 | 0.943 |
| `_t128` | B2 H8 S2048, median over 8 layouts | 1170.7 | 0.896 | 0.900 |
| default (KV-split) | B1 H8 S2048, palette data (`lcg`) | 745.0 | 1.392 | - |
| `_vs128` | the shapes above, palette data | | 0.994 - 1.002 | - |

On real data the shipped kernel is now ahead of the vendor kernel on every gate shape except `_t128`. The `_vs128` gain since v9.2 (+5 - 7 %) comes from this round's uniform-scalar lowering defaults ([`lowering-defaults.md`](lowering-defaults.md)).

## 3. Attention prefill (perf_ab with the vendor side: `tools/perf_ab.sh --old HEAD --new . --reps 3`, families fmha_bf16 / fmha_hd192 / fmha_fp8pt / fmha_mxfp8 / fmha_bwd)

- Ours: the form `amdc dispatch` picks (key split, q128, pairing, KV split).
- aiter: `hsa/gfx950/fmha_v3_fwd` (`fwd_hd128_bf16[_causal]`, `fwd_hd192_hd128_bf16[_causal]`, `fwd_hd128_fp8[_causal]`) and the MXFP8 vendor object, in the same process on the same data.
- Each value is the median of 3 interleaved repetitions, TF/s; all correctness markers OK.
- Shapes are B x H x HK x S_q x S_k.

**fp8 per-tensor** (`fmha_fwd_hd128_fp8pt_v3[_causal][_kvsplit]`)

| shape | causal | ours TF/s | aiter TF/s | ratio | v9.2 |
|---|---|--:|--:|--:|--:|
| 2x32x32x8192x8192 | no | 2454 | 2038 | 1.20 | 1.20 |
| 2x32x32x8192x8192 | yes | 2156 | 1827 | 1.18 | 1.13 |
| 4x16x16x2048x2048 | no | 2002 | 1608 | 1.25 | 1.22 |
| 4x16x16x2048x2048 | yes | 1367 | 1209 | 1.13 | 1.07 |
| 2x32x8x4096x4096 (GQA4) | no | 2350 | 1979 | 1.19 | 1.17 |
| 2x32x8x4096x4096 (GQA4) | yes | 1961 | 1697 | 1.16 | 1.10 |
| 1x8x8x4096x4096 (KV split S2) | no | 1650 | 1057 | 1.56 | 1.48 |
| 1x8x8x4096x4096 (KV split S2) | yes | 888 | 503 | 1.77 | 1.70 |

**bf16 v3 hd128**

| shape | causal | ours TF/s | aiter TF/s | ratio | v9.2 |
|---|---|--:|--:|--:|--:|
| 2x32x32x8192x8192 | no | 1123 | 1172 | 0.96 | 0.96 |
| 4x16x16x2048x2048 | no | 812 | 899 | 0.90 | 0.89 |
| 1x8x8x4096x4096 (`_ks`) | no | 748 | 713 | 1.05 | 1.07 |
| 4x16x16x1024x1024 | no | 750 | 878 | 0.85 | - |
| 1x32x8x2048x2048 | no | 851 | 949 | 0.90 | - |
| 2x32x32x8192x8192 | yes | 1001 | 1120 | 0.89 | 0.90 |
| 4x16x16x2048x2048 | yes | 638 | 758 | 0.84 | 0.80 |
| 1x8x8x4096x4096 (`_causal_ks`) | yes | 472 | 379 | 1.24 | 1.23 |
| 4x16x16x1024x1024 (`_causal_q128`) | yes | 431 | 419 | 1.03 | 1.01 |
| 1x32x8x2048x2048 (`_causal_q128`) | yes | 560 | 541 | 1.04 | 1.01 |

**bf16 hd192 / hd128 (MLA prefill)**

| shape | causal | ours TF/s | aiter TF/s | ratio | v9.2 |
|---|---|--:|--:|--:|--:|
| 1x16x16x4096x4096 | no | 1150 | 1181 | 0.97 | 0.95 |
| 4x16x16x2048x2048 | no | 1043 | 1096 | 0.95 | 0.91 |
| 1x8x8x4096x4096 (`_ks`) | no | 932 | 1184 | 0.79 | 0.81 |
| 1x128x128x1024x1024 | no | 896 | 849 | 1.06 | - |
| 4x16x16x2048x2048 (paired) | yes | 825 | 826 | 1.00 | 0.98 |
| 1x128x128x1024x1024 (paired) | yes | 629 | 677 | 0.93 | 0.96 |
| 1x16x16x4096x4096 (`_np`) | yes | 719 | 920 | 0.78 | 0.77 |
| 1x8x8x4096x4096 (`_causal_ks`) | yes | 577 | 614 | 0.94 | 0.95 |

**MXFP8 `_vs128` forced at 256 rows per block** (real data; the dispatched paths are in section 2)

| B x H x S | ours TF/s | aiter TF/s | ratio | v9.2 |
|---|--:|--:|--:|--:|
| 2 x 32 x 8192 | 2584 | 2504 | 1.03 | 0.99 |
| 1 x 8 x 8192 | 2417 | 2411 | 1.00 | 0.96 |
| 4 x 8 x 2048 | 1906 | 1900 | 1.00 | 1.02 |
| 1 x 8 x 2048 | 720 | 712 | 1.01 | - |

**Backward at S = 1024** (the perf_ab shapes, complementing section 4)

| shape | causal | ours TF/s | aiter TF/s | ratio |
|---|---|--:|--:|--:|
| 1x16x16, 2x8x8, 1x16x4 | no | 272 - 275 | 256 | 1.06 - 1.08 |
| the same three | yes | 141 - 144 | 100 | 1.41 - 1.44 |

Summary:
- fp8 per-tensor is ahead of aiter everywhere (1.13 - 1.77) and improved since v9.2.
- bf16 hd128 is at or slightly above v9.2 (0.84 - 1.24); full-grid shapes are at 0.85 - 0.96, the split / q128 forms at 1.03 - 1.24.
- hd192 is unchanged within noise (0.78 - 1.06); its two weakest forms are the split `_ks` (0.79) and the unpaired causal `_np` (0.78).
## 4. Attention backward (`tools/attn_family_table.sh`, `ONLY=bwd`)

- Ours: `fmha_bwd_dot` + `dq` + `dkdv` (causal: the dispatched forms).
- aiter: the `fmha_v3_bwd` pipeline (odo + dqdkdv a32 + dq_convert).
- Shapes: B H HK S, bf16, hd 128; all `bad=0`.

| shape | causal | ours us | aiter us | ratio | v9.2 |
|---|---|--:|--:|--:|--:|
| B4 H16 2048^2 | no | 601.8 | 496.5 | 0.83 | 0.79 |
| B1 H16 4096^2 | no | 554.0 | 462.9 | 0.84 | 0.79 |
| B4 H16 HK4 2048^2 | no | 587.2 | 490.5 | 0.84 | 0.79 |
| B4 H16 2048^2 | yes | 361.9 | 313.6 | 0.87 | 0.82 |
| B1 H16 4096^2 | yes | 332.7 | 342.7 | 1.03 | 0.97 |
| B4 H16 HK4 2048^2 | yes | 356.2 | 306.6 | 0.86 | 0.81 |

## 5. Fused MoE, stage 1 (`tools/vs_aiter_moe.sh`)

- Ours: `gemm_moe1_<dt>_silu`, run by `tests/gpu/moe_host.cpp`.
- aiter: the `fmoe_2stages` stage 1 `fmoe_stage1_bf16_pertokenFp8[_blockscale]_g1u1_128x128_pf2`, on the same sorted routes.
- Both at BLOCK_M 128; aiter's stage-1 output matches ours (MOE_OK).

| dtype | shape (M D I E topk) | ours us | aiter us | ratio | v9.2 |
|---|---|--:|--:|--:|--:|
| fp8 per-token | 1024 4096 1024 8 2 | 37.0 | 40.3 | 1.09 | 1.07 |
| fp8 per-token | 4096 7168 512 32 8 | 323.0 | 370.4 | 1.15 | 1.14 |
| fp8 per-token | 256 7168 2048 64 8 | 392.1 | 382.7 | 0.98 | 0.93 |
| fp8 blockscale | 1024 4096 1024 8 2 | 59.2 | 51.0 | 0.86 | 0.87 |
| fp8 blockscale | 4096 7168 512 32 8 | 483.7 | 426.5 | 0.88 | 0.87 |
| fp8 blockscale | 256 7168 2048 64 8 | 471.6 | 401.6 | 0.85 | 0.79 |

## 6. The MXFP4 8192^3 outlier: not a regression

The first GEMM sweep measured `gemm_a4w4_mxfp4` at 311 us for 8192^3 (ratio 0.73, against 270 us in v9.2). Three re-runs of the MXFP4 rows (5 runs each, same build, same GPU) measured 255.4 / 255.9 / 256.0 us, ratio 0.89 - 0.90, which is v9.2's 0.90.

The scalar defaults don't explain it either: the same kernel with `AMDC_NO_SCALAR_DEFAULT=1` runs at 258.3 us, against 257.8 us with them.

So the 311 us was one transient measurement (the first kernel of the sweep, measured cold), not a code change, and no bisection was needed. Since v9.2 the kernel is 5 % faster (270 -> 256 us), but so is the vendor (242 -> 228 us), so the ratio is unchanged.

## 7. Summary (kernel-opt families)

| family | v12 ratio (aiter time / ours) | v9.2 |
|---|---|---|
| shipped MXFP8 FMHA, `port_gate`, real data | KV-split 1.063; `_vs128` 1.011 - 1.033, per CU 1.017; `_t128` 0.896 | 1.028; 0.952 - 0.981; 0.900 |
| GEMM: int8 per-token | 1.00 - 1.66 | 1.01 - 1.63 |
| GEMM: fp8 blockscale | 0.93 - 1.45 (split-K small M 1.45) | 0.91 - 1.45 |
| GEMM: bf16 | 1.03 (2048 x 4096^2); small M split-K vs aiter's non-split kernel 6.1 | 0.96 |
| GEMM: MXFP4 | 0.88 - 0.91 | 0.86 - 0.90 |
| attention fp8 per-tensor | 1.13 - 1.77 | 1.07 - 1.74 |
| attention bf16 v3 hd128 | full 0.85 - 1.05, causal 0.84 - 1.24 | 0.89 - 1.07, 0.80 - 1.23 |
| attention hd192 (MLA prefill) | 0.78 - 1.06 | 0.73 - 0.98 |
| attention backward | 0.83 - 1.03 (S 2048 / 4096); 1.06 - 1.44 (S 1024) | 0.79 - 0.97 |
| fused MoE stage 1 | fp8 per-token 0.98 - 1.15; fp8 blockscale 0.85 - 0.88 | 0.93 - 1.14; 0.79 - 0.87 |

Where aiter has no comparable kernel:
- bf16 GEMM split-K: aiter has split-K bf16 code objects, but our host does not drive them;
- MXFP8 blockscale GEMM v4: compared against aiter PR #6696 by `tools/gemm_gate.sh`, not in this table;
- MoE stage 2: aiter's stage 2 is not driven by the host; stage 1 only.
