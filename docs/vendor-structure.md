# Vendor FMHA (MXFP8, hd128) - structure, and our final `_vs128` re-implementation

Source: disassembly of `fwd_hd128_mxfp8.co` (9229 lines), rocprof counters, kernel-wiki; cross-checked with `docs/vendor-spec.md` (kernel-occ).

## Phase structure (per 128-key step, per wave)
Two groups of 4 waves (all 8 waves own 32 query rows) run the SAME code one barrier apart, so on a SIMD one wave is in its MFMA phase while the other is in its VALU phase.

| phase | prio | work |
|---|---|---|
| VALU (softmax) | `s_setprio 0` | 32 `v_max3_f32`; one `v_cmp_nge` + branch to a rare out-of-line rollback block (~191 VALU); fast path = 32 `v_cvt_pknorm_u16_f32` + 32 `v_cvt_scalef32_pk_fp8_bf16` (Schraudolph exp, no `v_exp`); first 8 `ds_read_b64_tr_b8` issued before the barrier |
| barrier | `s_setprio 1` + `s_barrier` | |
| MFMA | 1 | 2 x `v_smfmac_f32_16x16x128_fp8_fp8` (row sums), 8 UNSCALED `v_mfma_f32_32x32x64_f8f6f4` (PV), 1 `v_mfma_f32_32x32x8_bf16` (bias C, reused by the 4 QK tiles), 8 `v_mfma_scale_f32_32x32x64_f8f6f4` (QK of the NEXT 128 keys); `buffer_load_dwordx4 ... lds` DMA of later tiles interleaved |
| barrier | `s_setprio 0` + `s_barrier` | |

2 barriers per 128 keys, loop body is 2x unrolled (two ring-slot parities). ~113 VALU per 128 keys on the fast path (the early `_vs` scaffold was ~400; the final `_vs128` is described below). Staging is global->LDS DMA only (no VGPR round trip), 1040-byte runs. The row-sum smfmac uses per-lane A / index operands built with `v_cndmask` from a lane mask (not a constant).

## Final `_vs128` implementation (Main v2.2-round75-final)

This supersedes the early `_vs` scaffold; the shipped kernel is `fmha_fwd_hd128_mxfp8_vs128`
(own backend), selected by `fmha_fwd_hd128_mxfp8_auto` for the 64-block benchmark and for filled grids;
the 128-row fallback is the separate `_t128` kernel.

- **256-row blocks** (8 waves x 32 query rows, 512 threads); 1 block/CU, 2 waves/SIMD, 8 waves/CU.
- **Q prologue:** Q staged with **16-byte (`b128`) loads** and held resident in registers for the
  loop (`q0reg`/`q1reg`); one Q load per tile instead of per S tile.
- **8-byte epilogue stores** (`StoreF32x2`): fp16 partial/output stores are 8 B, halving the epilogue
  write traffic; the exact-reference recipe and the 4-`smfmac` row sums are unchanged.
- **KV-split inherits the Q prologue vectorisation:** `_kvsplit` + `combine8h` uses the same 16 B Q
  loads and the same 8 B epilogue stores; the split path is bit-identical to the single kernel
  (`c3301d1c` on B1H2S200 SK1337) and 1200/1200 repeated launches are a single md5.
- LDS 102400 B, VGPR 249 (`next_free_vgpr` 252), SGPR 48; dense XOR-swizzled K/V staging with
  `kdlds,dbuf`; own backend only (own-only DMA/ops).
- **Certified numbers (Main `8fec359`, `v2.2-round75-final`):** per-CU 64-block
  **453-455 TFLOP/s = 7.07-7.10 TF/CU** vs vendor 8.41 (1.19x gap); filled B4H8S2048
  **0.846x** (1443.6/1454.8 vs 1706.8), filled B1H8S8192 **0.841x** (1998.6/2002.7 vs 2376.5);
  accuracy mean **8.3e-4** vs vendor 1.5-2.9e-3. Full certified table and closure statement:
  `docs/current-status.md` and `docs/vendor-spec.md` section 7.12.

## Measured (historical `_vs` scaffold, B1 H8 S2048, own backend, GPU 0, 30 iters)
default `_t` 292 | `_vs` (svcorr) 255 | `_vs`+svscale 312 | vendor 515. See `docs/status.md` round 47
for the historical ablations and negatives; those numbers are not the shipped `_vs128`.