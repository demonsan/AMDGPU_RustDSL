# Performance A/B harness (`tools/perf_ab.sh`)

The performance gate of [`two-weavers-plan.md`](two-weavers-plan.md) W4: **one harness for every integrated kernel family**, comparing the code objects of an OLD and a NEW commit in
ONE GPU session, with a noise model and a PASS / FAIL verdict ("no family slower beyond noise"). Every behaviour-changing merge (the back-end Weaver, scheduler, register allocation, any
front-end change that is not md5-identical) is gated by it. The md5 manifest (`amdc md5-manifest`) stays the gate for moves that must not change any output.

## 1. Use

```
# on the GPU node, in a clone that has both commits (git fetch both refs); HIP_VISIBLE_DEVICES: GPUs 0-3 only, default 0; 4-7 are refused by the guard (`tools/gpu_guard.sh` / `gpu_guard()` in `perf_ab.py`)
bash tools/perf_ab.sh --old origin/Main --new my-branch --quick          # the per-merge gate: first / last case of every group
bash tools/perf_ab.sh --old v10.0-layers --new .                          # everything, the working tree as NEW
bash tools/perf_ab.sh --old A --new B --families fmha_bf16,gemm --reps 7  # some families
bash tools/perf_ab.sh --old A --new B --only 'mb/rope|decode16' --show    # run each selected case once on NEW and print the host output (writing / debugging a case)
bash tools/perf_ab.sh --old A --new B --list                              # the case table
```

`--old` / `--new` are git refs (`.` = the working tree as NEW). Options: `--reps N` (default 5), `--floor` (2 %), `--k` (2), `--family-floor` (1 %), `--out DIR`, `--release` (release
`amdc`), `--no-vendor`, `--old-emit-env K=V` / `--new-emit-env K=V` (environment of that side's `amdc emit`: A/B an experiment switch such as `AMDC_SCHED_ADD` on one commit).
Exit code 0 = PASS, 1 = FAIL, 2 = harness error. Node environment as for the other GPU tools: `AMD_ROCM_MOUNT`, `AMD_LLVM_BIN` (container cargo), `ROCM_PATH`, `CLANG`, `AITER_HSA`, `REF_CO`.

What a run does (`build/ab/run-<old>-<new>/`):
1. a git worktree per commit (`build/ab/trees/<sha>`) and `amdc` of each (`build/ab/target/<sha>`, cached by commit);
2. every kernel the selected cases need is emitted and assembled once per commit (`<side>/co/<kernel>.co`); a case asks that commit's OWN launch rules (`amdc dispatch ...`), so a commit
   that changes a rule is measured with its rule; a subcommand only the new commit has is answered by the new commit for both sides;
3. the GPU hosts are built once, from the NEW tree (`tests/gpu/*_host.cpp`; the same host binary runs both sides);
4. the session: per repetition, per case: OLD and NEW back to back (order alternating per repetition), then the vendor side where the case has one;
5. `report.md`, `report.json` (per-case medians, ratio, allowed drop, verdict, vendor ratios), `raw.jsonl` (every sample), the stdout summary.

## 2. Noise model and verdict

Per case (a case with several metrics, e.g. MoE stage 1 / stage 2, is judged per metric):
- samples of each side are converted to a performance number (higher is better; a time is inverted);
- `ratio = median(new) / median(old)`; each side's scatter is the robust relative sigma `1.4826 * MAD / median`;
- allowed drop `thr = max(--floor, --k * hypot(sigma_old, sigma_new))`: **FAIL** when `ratio < 1 - thr`, **IMPROVED** when `ratio > 1 + thr`, else OK;
- a missing correctness marker of the host (`ATTN_OK`, `GEMM_OK`, `MOE_OK`, ...), a non-zero exit, or a kernel that no longer builds on NEW is a FAIL; a kernel that exists only on NEW is `NEW`
  (informational);
- with fewer than 3 samples per side the noise estimate is replaced by the plain floor.

Per family: the geometric mean of the case ratios must not fall below `1 - max(--family-floor, --k * rms of the case sigmas / n)`. **Overall verdict PASS iff no case and no family fails.**
The vendor columns are `new/vendor` and `old/vendor` (performance, > 1 = ours is faster) from the vendor run of the same repetition.

## 3. Cases (`tools/perf_ab_cases.py`, 170 cases, `--list`)

| family | cases | host | vendor side |
|---|---|---|---|
| `membound` | every timed case of `tools/b2_matrix.sh` (parsed from it, so the matrix stays the single list): eltwise / rows / quant / cast / rope / cache / dequant / topk / sampling / moe sorting, reduce, gate / conv1d / mxnorm / mla cache / mxfp6 ... | the `*_host` of the matrix | none in the table (`tools/vendor_membound.sh` for aiter) |
| `fmha_mxfp8` | 4 shapes, forced `_vs128` or the commit's `dispatch mxfp8` kernels | `ref_host` | aiter code object in the same process |
| `fmha_fp8pt` | full / causal x 4 shapes, the commit's `dispatch fp8pt` kernel + key split | `fp8pt_host` | aiter in the same process |
| `fmha_bf16` | full / causal x 5 shapes, `dispatch prefill_bf16` | `attn_host` | aiter `fwd_hd128_bf16[_causal]` |
| `fmha_hd192` | full / causal x 4 shapes, `dispatch hd192` | `attn_host` (192 / 128) | aiter `fwd_hd192_hd128_bf16[_causal]` |
| `fmha_bwd` | full / causal x 3 shapes (dot + dq + dkdv) | `attn_bwd_host` | aiter fmha_v3_bwd in the same process |
| `decode` | paged decode16, 6 (B, H, HK, ctx) shapes, `dispatch decode16` (`_f` fused / `_c` last-arriver + split count) | `pa_host` | - |
| `mla` | 4 (B, H, ctx) shapes, `dispatch mla` (per-wave / head-parallel + splits) | `mla_host` | - |
| `moe` | bf16 / fp8 / fp4 stages (stage 1, stage 2 + sum judged separately), the commit's `dispatch moe` block_m | `moe_host`, `moe_fp4_host` | - |
| `conv` | 2-D 3x3, 1x1, 3-D 3x3x3 implicit GEMM | `conv_host` | - |
| `gemm` | bf16 / int8 / fp8 per-token x per-channel / fp8 block-scale x 7 shapes incl. small M and split-K (+ the reduce kernel), the commit's `dispatch gemm`; MXFP8 v4 x 3 shapes | `vendor_gemm_host` (`ours_v4_*`), `gemm_ref_host` | - |
| `gemv` | small-M GEMM (W6): M ∈ {1, 4, 16, 32, 64} x N = K ∈ {4096, 7168, 8192} for every dense kind (bf16 / int8 / fp8 per-token x per-channel / fp8 block-scale), the commit's dispatched kernel + split-K reduce; weight-bandwidth-bound (convert TF/s to bytes per second for the roofline) | 60 |

Sampling / top-k are in `membound` (`sampling_host`, `topk_host`, `topk_per_row_host`). The `--quick` set takes the first and the last case of every group (the group is the first two id
segments, a host for the memory-bound cases: the first only), the per-merge gate.

Add a case: append a `Case(...)` to a family function of `perf_ab_cases.py`: `build(side) -> Run(emit, cmd, env)` (tokens `{co:KERNEL}`, `{codir}`, `{host:NAME}`), the `metric` (`tflops`,
`gbs`, `ms`, `us` or a regex with one group), the `line` filter, the correctness marker `ok`, optionally `vendor` / `vendor_same` + `vendor_line`, `parts`. Check it with `--only ID --show`.

## 4. Calibration and detection evidence (node 58, GPU 1)

- **Same compiler on both sides (A / A):** `--old <commit> --new <commit + tools only> --quick --reps 5`, 73 cases: every case within its allowed drop (the largest spread: ratio 0.985 for `gemm/bf16/32x4096x14336`, 1.010 for
  `bf16/full/1x32x8x2048x2048`; per-case allowed drops 2.0 - 8.9 %, the families' geometric means 0.994 - 1.001). The one FAIL of that run was a harness defect (the MXFP8 case took the KV-split kernel
  from the dispatch rule, which `ref_host` cannot run without its split environment; the case now forces `_vs128` like `tools/attn_family_table.sh`). The re-run after the fix (`--quick`, 3 repetitions, 67 cases): **PASS**, family
  geomeans 0.999 - 1.008.
- **Detection (fault injection):** `--new-emit-env AMDC_SCHED=0` (the NEW side's kernels emitted without the scheduler) on one case per family: **FAIL** on all four cases and four families, ratios 0.863 (`gemm/bf16/4096x4096x8192`),
  0.869 (`hd192/full/1x16x16x4096x4096`), 0.900 (`bf16/full/4x16x16x1024x1024`), 0.922 (`fp8pt/full/4x16x16x2048x2048`); the vendor ratio of the same repetition moves with them (e.g. hd192 `new/vendor` 0.828 vs `old/vendor` 0.953).
- **Real commits:** `c043ebeb~1` (before the hd192 region scheduling) against the tree: hd192 +0.3 % geometric mean, two cases `IMPROVED` / within noise; a regression of that size (a few %) is at the edge of the floor, which is by design: the gate catches
  what the noise model can distinguish, and a smaller expected effect needs more repetitions (`--reps 9`) on the families concerned.
- **Cost:** the `--quick` set (67 cases, 3 repetitions) runs in about 35 minutes (about 10 minutes per repetition plus the builds); the full table (177 cases) takes hours.

## 5. Limits

- The hosts come from the NEW tree: a host change affects both sides alike. The harness compares code objects, not the compilers' run time.
- Runs are short, the node is shared: the floor (2 %) and `k` are set from the same-compiler (A / A) scatter below; real run-to-run drift of the GPU (clocks, other jobs) is not modelled, rerun a
  failing case with `--only ID --reps 9` before concluding.
- The vendor side exists only where the host runs the vendor kernel itself (attention families, backward); GEMM / decode / MLA vendor ratios are in their own tools
  (`tools/gemm_gate.sh`, `tools/pa_auto.sh`).
- A full run is long (hours: the memory-bound hosts initialise and check on the CPU); the `--quick` set is the merge gate, the full table is for milestones.

## Metric precision

The metric of a case is read from the host's output line. `vendor_gemm_host` prints TF/s with one decimal and microseconds with two, so the **gemv family and the small-M GEMM cases (M <= 128) are judged on the microseconds** (a lower-is-better metric; the ratio is still new / old in performance terms): at M = 1 the weight-streaming kernels run at about 4 TF/s, where one decimal step (4.3 vs 4.2) is 2.4 % and would exceed the 2 % floor by itself. Large GEMMs keep TF/s.
