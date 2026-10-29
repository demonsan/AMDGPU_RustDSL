# Vendor FMHA (MXFP8, hd128) - structure, and our `_vs` re-implementation

Source: disassembly of `fwd_hd128_mxfp8.co` (9229 lines), rocprof counters, kernel-wiki; cross-checked with `docs/vendor-spec.md` (kernel-occ).

## Phase structure (per 128-key step, per wave)
Two groups of 4 waves (all 8 waves own 32 query rows) run the SAME code one barrier apart, so on a SIMD one wave is in its MFMA phase while the other is in its VALU phase.

| phase | prio | work |
|---|---|---|
| VALU (softmax) | `s_setprio 0` | 32 `v_max3_f32`; one `v_cmp_nge` + branch to a rare out-of-line rollback block (~191 VALU); fast path = 32 `v_cvt_pknorm_u16_f32` + 32 `v_cvt_scalef32_pk_fp8_bf16` (Schraudolph exp, no `v_exp`); first 8 `ds_read_b64_tr_b8` issued before the barrier |
| barrier | `s_setprio 1` + `s_barrier` | |
| MFMA | 1 | 2 x `v_smfmac_f32_16x16x128_fp8_fp8` (row sums), 8 UNSCALED `v_mfma_f32_32x32x64_f8f6f4` (PV), 1 `v_mfma_f32_32x32x8_bf16` (bias C, reused by the 4 QK tiles), 8 `v_mfma_scale_f32_32x32x64_f8f6f4` (QK of the NEXT 128 keys); `buffer_load_dwordx4 ... lds` DMA of later tiles interleaved |
| barrier | `s_setprio 0` + `s_barrier` | |

2 barriers per 128 keys, loop body is 2x unrolled (two ring-slot parities). ~113 VALU per 128 keys on the fast path (ours ~400). Staging is global->LDS DMA only (no VGPR round trip), 1040-byte runs. The row-sum smfmac uses per-lane A / index operands built with `v_cndmask` from a lane mask (not a constant).

## What `fmha_fwd_hd128_mxfp8_vs` implements (own backend only)
- same two-group ping-pong, `s_setprio 1/0` pair per phase, but 64-key units (4 barriers per 128 keys) - 128-key units do not fit the register file (see below);
- QK of the next unit inside the MFMA phase; DMA of tile t+2 issued in the VALU phase of tile t into a 3-slot ring, one deferred `vmcnt` wait;
- dense XOR-swizzled LDS layout (`swz`), 0 bank conflicts (524288 -> 0);
- sv recipe (bias in C, pknorm + scalef32, `svcorr` LUT, rollback); masked last tile peeled; group B = group A's sequence one phase later (a carried P fragment gave non-determinism);
- `scalarize` lowering opt by default (needed to fit).

## Measured (B1 H8 S2048, own backend, GPU 0, 30 iters)
default `_t` 292 | `_vs` (svcorr) 255 | `_vs`+svscale 312 | vendor 515. See `docs/status.md` round 47 for ablations and negatives.