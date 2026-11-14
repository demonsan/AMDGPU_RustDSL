# Lowering defaults: what is on for every kernel, and the measurements behind it

Lowering tokens (`crates/amdgpu-codegen/src/lower.rs`, `opt_on`) start as opt-in switches. A token becomes a default only after a
measured win:
- `tools/perf_ab.sh` over every kernel family, same commit on both sides, the new side emitting with the token;
- the md5 manifest updated, the full test gate, and port_gate.

## Experiment switches

| Switch | Effect |
|---|---|
| `AMDC_LOWER_ADD=<tokens>` | comma-separated tokens appended to every non-exact kernel on top of its compile policy (`name-substring:tokens` entries separated by `;` restrict them to matching kernels; `m0reuse` included) |
| `AMDC_NO_SCALAR_DEFAULT=1` | switches the uniform-scalar default set (below) off |
| `perf_ab.sh --new-emit-env AMDC_LOWER_ADD=<tokens>` | the A/B recipe: `--old HEAD --new . --quick --release --no-vendor` |
| `bash tools/lower_screen.sh "<tokens>" OUTDIR` | static screen: every manifest kernel with and without the tokens (new compile failures, instruction / VALU counts per family) |

## The uniform-scalar default set

`DEFAULT_SCALAR` (`crates/amdgpu-codegen/src/lib.rs`) is applied to every non-exact kernel. It contains:
- `scalarize`, `scalarize2`: wave-uniform integer chains, clamps, selects and loop-exit compares in SALU, branching on SCC;
- `waveuni`: `workitem_id >> 6` is a wave constant;
- `sgpr`, `sgpr2`: SGPR values as VALU operands without a `v_mov` copy.

If a kernel runs out of SGPRs with the set, it is recompiled without it, which gives the lowering it had before. Only
`fmha_fwd_hd192_hd128_bf16_v3_causal` does this; `waveuni` pushes it past the SGPR file.

perf_ab `--quick`, 12 families, 3 interleaved repetitions (node 53 GPU 0), family geomean new / old:

| conv | decode | fmha_bf16 | fmha_bwd | fmha_fp8pt | fmha_hd192 | fmha_mxfp8 | gemm | gemv | membound | mla | moe |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.022 | 0.998 | 1.011 | 1.042 | 1.027 | 1.000 | 1.025 | 1.002 | 1.007 | 1.013 | 1.005 | 1.001 |

Case wins above noise:
- FMHA: fmha_bwd +3.5 … +4.9 % (all four cases); fp8pt causal +3.8 / +4.4 %; mxfp8 1x8x2048 +4.6 % and 2x32x8192 +2.1 %;
- conv2d / conv3d +2.7 / +2.3 %;
- membound: silu_mul_bias +9.8 %, conv1d_update +4.3 %, topk_per_row decode +4.0 %, mxfp6 quant +2.8 %.

No case regressed beyond its noise threshold.

Full perf_ab (all 397 cases, 3 repetitions, old = Main f28244c4):
- Every family geomean passes: conv 1.021, decode 0.995, fmha_bf16 0.997, fmha_bwd 1.040, fp8pt 1.023, hd192 1.003, mxfp8 1.035, gemm 0.998, gemv 1.004, membound 1.029, mla 1.003, moe 0.997.
- 8 cases fell below their thresholds (fp8 GEMM, moe, moe_reduce, one bf16 fwd case). A 7-repetition re-run of exactly those cases passed all of them (moe_reduce +6.8 %). After the rebase onto Main 9f60ca6d (edgepad on by default), a `--quick` re-check passed every family; its one failing case, hd192 causal 1x16x16x4096, measured 1.014 in a 7-repetition re-run.

port_gate (`fmha_fwd_hd128_mxfp8_v3`): OVERALL PASS. vs128 per-CU is 1.017x the vendor (recorded 0.971x); B4 S2048 1.023x, B1 S8192 1.010x, KV-split 1.062x.

### Measured and left out

| Token | Result |
|---|---|
| `salu32` (unsigned / equality uniform compares on SCC) | decode16 −2 … −3 % on every case (family 0.977, alone and in the set); not a default |
| `imm` (constants as VALU operands) | families neutral (0.986 … 1.005); the topk_per_row −30 % was a miscompile (below), equal after the fix; opt-in |

## A latent unsound fold found by the `imm` screen (fixed)

**Symptom.** With `imm`, `topk_per_row_decode` returned wrong results on every row (`TKR_FAIL`, bad_rows 256 / 256) and ran 30 %
slower. perf_ab did not flag it: the membound ok pattern `bad[a-z_]*=0` was matched by `past_end_bad=0`.

**Cause.** `cmpfold` and `dce_cmp_pairs` recognised a 0/1 compare materialisation (`v_cmp ; v_cndmask t, 0, one`) by its SHAPE alone.
- With `imm`, a select whose false arm is the constant 0 lowers to `v_cndmask d, 0, x`, with x an arbitrary per-lane value, right after the select's own compare.
- cmpfold then rewrote a later `d != 0` into the select's compare `c`. That is wrong for the lanes where c holds and x is 0.
- No shipped kernel was affected. The kernels that carry `imm` in their policy (vs128 / kvsplit / t128, fp8pt, mla_decode, pa_decode16, attn_combine, …) compile byte-identically with the fixed fold (md5 manifest unchanged): the shape never met a later `!= 0` test there. Only `topk_per_row` under `imm` (not its policy) hit it.

**Fix.**
- `cmp_i1` registers the materialisations it emits (`MAT01`); cmpfold and dce_cmp_pairs act only on those.
- `crates/amdgpu-codegen/tests/cmpfold_select_zero.rs` reproduces the pattern; it fails on the pre-fix lowering.
- perf_ab fails a case whose output carries a host failure marker (`XYZ_FAIL` / `FAIL`) even when the ok pattern matches, and the membound ok pattern no longer accepts `past_end_bad=0`.
- Goldens unchanged (an interim workaround that kept the inline-0 select arm in a register changed 24 kernels; it was replaced by the MAT01 fix).
