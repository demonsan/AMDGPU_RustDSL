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

### Measured and left out

| Token | Result |
|---|---|
| `salu32` (unsigned / equality uniform compares on SCC) | decode16 −2 … −3 % on every case (family 0.977, alone and in the set); not a default |
| `imm` (constants as VALU operands) | families neutral (0.986 … 1.005); topk_per_row decode −30 %, hd192 causal −3.9 %, mxfp8 4x8x2048 −2.8 %, fp8 blockscale GEMM −2.4 %; opt-in |
