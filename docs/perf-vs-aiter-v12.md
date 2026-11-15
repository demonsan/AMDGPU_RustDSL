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
| MXFP4 (a4w4) | 8192 x 8192 x 8192 | 311.2 (`gemm_a4w4_mxfp4`) | 227.6 (f4gemm 256x256) | **0.73** | 0.90 |
| MXFP4 (a4w4) | 4096 x 4096 x 4096 | 41.3 (`gemm_a4w4_mxfp4`) | 38.4 (f4gemm 256x256) | 0.93 | 0.86 |
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
- **MXFP4 8192^3 regressed** from 0.90 to 0.73: ours went from 270 to 311 us. Being investigated (see section 6).
