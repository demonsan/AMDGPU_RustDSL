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