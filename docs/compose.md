# Composition planning (`amdgpu-graph::compose`)

Part of the front-end Weaver ([`two-weavers.md`](two-weavers.md), W2): for a launch shape, which COMPOSITION of stage variants runs. A [`Composition`](../crates/amdgpu-graph/src/compose.rs) names the kernels of the
launch in order (main kernel, then its reduce / combine kernel), the key / K splits, the CTA tile where the family has one, the rule that decided, and the **plan facts of the main kernel**: the CTA shape
(threads, waves) and the LDS rings (slots, bytes per slot, wave-private or CTA-shared) its `KernelPlan` declares.

```
amdc compose decode16 <B> <HK> <CTX> [--page P]
amdc compose prefill_bf16|fp8pt|hd192 <B> <H> <SQ> <SK> <causal> [--cus N]
amdc compose mxfp8 <B> <H> <S> [SK]          amdc compose mla <B> <H> [--page P]
amdc compose gemm <kind> <M> <N> <K> [--cus N]   (kind: bf16 | i8_rowcol | fp8_rowcol | fp8_blockscale | mxfp4)
amdc compose moe <bf16|fp8|fp4> <M> <TOPK> <E>   amdc compose bwd_causal
```
Each prints one JSON object (`family`, `kernels`, `splits`, `tile`, `rule`, `plan`).

## Rules and evidence

The decisions are the per-shape-bucket rules of [`dispatch.rs`](../crates/amdgpu-graph/src/dispatch.rs); each rule's doc comment records the measured evidence it was fitted on (same-run times against the vendor
kernel per bucket, the rule it replaced). `compose` adds no second copy of any rule: it calls them and returns their answer as one record. There is no runtime search: a rule is a pure function of the shape,
and a rule change needs its evidence (the [`perf_ab.md`](perf-ab.md) harness is the gate).

## Facts

`compose_facts.rs` is GENERATED from the kernels' own plans (`amdc compose-facts --out crates/amdgpu-graph/src/compose_facts.rs`, source in `crates/amdgpu-kernels/src/compose_facts_gen.rs`): one row per registered
kernel that declares a plan (CTA threads / waves, phases, units, rings). `crates/amdgpu-kernels/tests/compose_facts.rs` fails when the checked-in file is stale and checks that every kernel the rules
dispatch to has a row, so the table cannot drift from the kernels. The facts are what the back end's constraints are made of ([`two-weavers.md`](two-weavers.md) section 2): the front end decides the composition
and the CTA shape, the plan carries the rings and phases.

## Search (offline, GEMM first)

The rules above are fitted by hand today. `compose_search` makes the fitting a repeatable pipeline per family (GEMM first: tile x split-K x ring depth per operand kind):

1. **Enumerate** (`amdgpu_graph::compose_search::gemm_candidates`, `amdc compose-search gemm <kind> M N K`): every registered tile variant (256 x 256, 128 x 128, 64 x 128 with the 3-slot ring) x split-K (2 / 4 / 8 + the
   reduce kernel), filtered by the kernels' shape constraints (the N tile divides N, whole 256 tiles for the 256-row form, a split needs N % 1024 == 0 and whole K steps per part).
2. **Prune** (`prune`, conservative): a coarse resource / time model (CTAs, rounds over the CUs, a compute-or-operand-stream K loop, the reduce's fp32 partials) drops only candidates another is at least 1.5x
   faster than in the model without using more LDS; the model is relative and never decides. The resource vector (LDS bytes) comes from the kernels' plan facts (`compose_facts.rs`).
3. **Compile-only feasibility** (`tools/compose_search.py`): the survivors are emitted and checked with `amdc summary --json` (VGPR + AGPR <= 512, LDS <= 160 KiB, the kernel compiles).
4. **Measured top-k**: the `--top` best survivors by model plus the recorded rule's own choice are measured INTERLEAVED (per repetition every candidate once, order rotating) on `vendor_gemm_host`; the median decides.
5. **Record**: the measured winner per shape bucket is what `dispatch::{i8,bf16,blockscale}_config` records with its evidence; `tools/compose_search.py check-gemm` re-runs the search over the evidence shapes and lists
   where the recorded rule is not within `--tol` (3 %) of the measured best.

[`compose-gemm-evidence.md`](compose-gemm-evidence.md) is the full table (node 58, GPU 1): **29 of the 36 (kind, shape) pairs confirm the recorded rule** within 3 %. The seven that do not (rule vs the measured best):
- `2048x7168x7168`, int8 / fp8 per-token x per-channel / bf16: the rule takes the 128 x 128 tile (its test is `(M/256)(N/256) >= CUs`, here 224 < 256 CTAs), the 256 x 256 kernel is 23 - 28 % faster (one round of 224 CTAs
  against 3.5 rounds of 128-row CTAs); the same shape for block-scale: the 64-row form is 3.3 % faster than the 128 tile;
- bf16 `512x7168x7168` (+8.9 %, `_128_splitk` S2) and `1024x4096x8192` (+5.9 %, `_128_splitk` S2): the split 128-tile forms the bf16 rule never picks;
- int8 `64x8192x8192`: S2 is 3.8 % faster than the rule's S4 (fp8 per-token: +2.5 %, within tolerance).
These are rule CANDIDATES, not changes: a rule change goes through the `perf_ab` gate with the new evidence recorded in `dispatch.rs`.

## Rule updates derived from the search (GEMM, 2026-11-12)

The search found rule defects, and the rules in `dispatch.rs` now follow the measurements ([`compose-gemm-sweeps.md`](compose-gemm-sweeps.md): the sweeps and the gate; node 53, GPU 1, interleaved, 7 repetitions):
1. **The 256 x 256 tile from 53 % fill** (`F256_PERCENT`; was 100 %): bf16, int8 and fp8 per-token x per-channel agree: the 128 tile wins up to 128 CTAs of 256 (fill 50 %), the 256 tile from 140 (55 %), by 17 - 28 % at fill 88 - 98 %.
   The threshold is the midpoint of the last loss and the first win.
2. **The split-K factor of the 64 x 128 form is a model, not a ladder**: `a + b K / (4096 S) + [S > 1] (r0 + r1 S M N / 8e6)` (a fixed part, the K loop split S ways, the reduce's fixed part and its fp32 partials) fitted per
   operand width (72 points 8-bit, 27 bf16; mean error 5.6 % / 7.6 %), and the rule takes the S in {1, 2, 4} it predicts fastest among those the kernels allow and one round of CTAs. The rule depends on K now
   (`i8_gemm_config` / `bf16_gemm_config` take it): the old ladder split short-K small grids that are fastest unsplit (up to 45 % slower). Against the measured best split on the sweep: mean regret 0.0 - 0.1 % (old rule 9.5 % on int8 / fp8).
3. **bf16 128 x 128 split form** (`gemm_a16w16_bf16_128_splitk` S2) for K >= 6144 on a 128-row grid of 0.625 - 1 x CUs: 5 - 12 % faster than `_64` / `_128` unsplit; shorter K and larger grids lose.

Gate (`perf_ab.py --families gemm`, old = the tree before the rule change): **PASS**, 79 cases, family geometric mean **1.042**, 19 cases IMPROVED (1.04 - 1.33x), none failing. The hold-out shapes (K = 6144, N = 8192, other M; not
used by the fits) are among the improved ones (1792x8192x4096 +23 - 26 %, 160x4096x8192 +9 - 13 %, 320x8192x6144 bf16 +6 %) and the rest are unchanged.
