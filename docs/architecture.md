# Architecture: four layers, as built

The decision record is [refactor-4layer-plan.md](refactor-4layer-plan.md). This page describes what Main has now, and links to the detailed
documents. Dependencies point **down only**; `crates/amdc/tests/layer_check.rs` checks the crate ranks, manifests, source references and that lower layers hold no kernel-name policy.

## 1. Layers and crates

| Layer | Crate | What lives there | Details |
|---|---|---|---|
| entry | `amdc`, `amdgpu-kernels` | CLI; per-family kernel composition and per-shape orchestration (skeleton slots, ABI, registry, probes) | [current-status.md](current-status.md) §3 |
| L1 graph | `amdgpu-graph` | model op graph, fusion rules, launch plans (`amdc graph plan`); dispatch rules (one rule per Python entry point); per-kernel op graphs and skeleton generation | [l1-graph.md](l1-graph.md), [opgraph.md](opgraph.md) |
| L2 stage | `amdgpu-stage` | stage instances and registry, contracts, pipelines (`Cadence`, `PipeBody`), components, pipeline templates (rings, ping-pong, split + merge, grid maps, scan, select), the GEMM and attention family bodies | [l2-templates.md](l2-templates.md), [l2-family-bodies.md](l2-family-bodies.md), [fusion-stage-library-design.md](fusion-stage-library-design.md) |
| L3 function | `amdgpu-tile`, `amdgpu-layout`, `amdgpu-isa` | tile ops, fragments, copies and DMA plans, softmax, reductions, epilogues; the layout algebra; per-target instruction atoms (`amdgpu_isa::gfx950`) | [dsl-tile-abstractions.md](dsl-tile-abstractions.md), [layout-tour.md](layout-tour.md), [op-coverage-tileops.md](op-coverage-tileops.md) |
| L4 backend | `amdgpu-dsl`, `amdgpu-codegen`, `amdgpu-target` | the IR (region tree, irv1 text, contracts, compile policy); analyses, passes, the pass manager, the Weaver (`amdgpu_stage::weave`), lowering / scheduling / allocation / hazards / waits / emission, verification; target descriptions | [pass-catalog.md](pass-catalog.md), [target-design.md](target-design.md), [gpu-semantics-verifier.md](gpu-semantics-verifier.md) |
| foundation | `amdgpu-sym`, `amdgpu-safeasm` | symbolic arithmetic; safe-asm windows | [sym.md](sym.md), [safe-asm-design.md](safe-asm-design.md) |

```
amdc ─┐
amdgpu-kernels ─┼─► amdgpu-graph (L1) ─► amdgpu-stage (L2) ─► amdgpu-tile / -layout / -isa (L3) ─► amdgpu-dsl / -codegen / -target (L4) ─► amdgpu-sym / -safeasm
```

## 2. Two examples through the layers

**`rmsnorm → quant_token(fp8)`** (a model graph with two ops):
1. L1: `amdc graph plan` matches the fusion rule `rmsnorm+quant_token.fp8` (`amdgpu-graph/src/fusion/rules.rs`), checks the row width (multiple of
   512, 512..8192) and plans one launch of `rmsnorm_quant_fp8_n<N>`; Python `rmsnorm_quant_fp8` uses the same rule.
2. Kernel: `amdgpu-kernels/src/norm/mod.rs` builds it from four L2 stage instances: `move.row_load` → `reduce.row_sumsq` → `quant.fp8_row` →
   `move.row_store` (`amdgpu_stage::stage::StageInst`, each with a stage contract checked by GV-STAGE).
3. L3: the stages use tile primitives (`bf16x8_to_f32`, `wave_reduce`, `rsqrt_f32`, `fp8_scale`, `fp8_pack`) and the Weaver's units.
4. L4: compile policy `Gemm` (lowering tokens, M0 reuse, the GEMM scheduler preset), then the gfx950 pipeline of section 3.

**MXFP8 prefill** (`fmha_fwd_hd128_mxfp8_auto`):
1. L1: `dispatch::mxfp8_prefill_path(B, H, S, SK)` picks `_vs128`, `_t128` or `_kvsplit` + `_kvsplit_combine8h` (with the split count).
2. Kernel: the generated skeleton (`skeleton_vs128` from the op graph) with the FMHA slots in `amdgpu-kernels/src/fmha/`.
3. L2: the attention family body in `amdgpu_stage::attention` (`attn3`, `block3`: operands, ABI, epilogues, per-block state) and the pipeline
   templates (2-group ping-pong, the 3-slot K/V ring with counted waits).
4. L3: `amdgpu-layout` / `amdgpu-tile` partitions, DMA, softmax, MFMA atoms (`amdgpu_isa::gfx950`).
5. L4: compile policy `Vs128` (lowering tokens `dce,cmpfold,scalarize,retarget2,imm,sgpr,zeroinit`) + the kernel's scheduler tokens
   `ldsink=1,wsink,ldahead32=1`, then the pipeline below.

## 3. The gfx950 pipeline

`amdc passes --target gfx950 [--evidence]` prints it (`amdgpu-codegen/src/pass/mod.rs`; catalogue and as-built notes in
[pass-catalog.md](pass-catalog.md)):

- front: `verify-ir-structure`, `strip-any-lane`, `analysis-uniform-values`, `lower-mir`, the 8 lowering clean-ups (`lower::CLEANUP`), `assign-agpr`;
- scheduled body (one inner-ladder rung): 11 scheduler passes (`sched-cfg-cleanup` … `waitcnt-recompute`, `verify-waits`), `m0-reuse`, `m0-explicit`,
  `verify-pinned-blocks`, `hazard-value` (strict), `verify-mir-structure`, `regalloc`, `check-workgroup-budget`, `hazard-inflight-waw`,
  `hazard-phys-war`, `hz-fill`, `m0-pad`, `hazard-valu-sgpr`;
- unscheduled body (fallback): the same objects without the scheduler, `hazard-value` non-strict, `verify-waits` before `regalloc`;
- back: `check-workgroup-budget`, `verify-gpu`, `emit-own | emit-llvm-tail`;
- the two Weavers: the front-end Weaver (`amdgpu_stage::weave`) emits the units and records the kernel plan (`Function::plan`, completed at
  `FnBuilder::finish` by `amdgpu-dsl/src/plan_derive.rs`); the back-end Weaver (`amdgpu-codegen/src/bweave.rs`) heads both bodies with
  `bweave-order` (today's order, plus the interleave of Free phases) and feeds the wait pass the plan's rings. Status, opt-in tokens and gates:
  [two-weavers.md](two-weavers.md) section 7;
- retry ladders as data: the outer one adds lowering tokens (`remat1..4`, `ldsspill8..64`), the inner one cuts pressure / hoisting; both retry on
  `out of VGPRs`.

Target dependence (each pass's row in `amdc passes`):

| Class | Passes |
|---|---|
| generic | `verify-ir-structure`, `strip-any-lane`, lowering dce / remat / coalesce, `sched-cfg-cleanup`, `waits-strip-auto`, `sched-dce-dead-selects`, `sched-phase-strip` |
| gfx950-parameterized (numbers from `amdgpu-target`) | `analysis-uniform-values`, `lower-lds-spill`, `sched-ldshoist`, `sched-phase`, `sched-list`, `regalloc`, `verify-mir-structure`, `verify-gpu` |
| gfx950-only (rules / instructions of the target) | `lower-mir`, `lower-select-fold`, `lower-kabatch`, `assign-agpr`, `sched-slotfill`, `sched-wsink`, `sched-ldahead`, `waitcnt-recompute`, `verify-waits`, `verify-pinned-blocks`, M0 passes, all hazard passes, `hz-fill`, `check-workgroup-budget`, emission |

## 4. gfx1250 status

Specification only: [target-gfx1250.md](target-gfx1250.md) (public sources only) and `amdgpu_target::gfx1250` (the extension fields of
[target-design.md](target-design.md) §3.2 as data). No pass reads them; `amdc passes --target gfx1250` reports "specification only".
From the description's analysis (target-gfx1250.md §2):

- **re-parameterized** (same algorithm, numbers from `Target`): register allocation (1024 VGPRs in blocks of 16, 106 fixed SGPRs, no AGPR),
  occupancy (`BlockPool`), the wait passes (split counters, `PerCounter`), the hazard rows (new unit `IndependentValu`), the scheduler model,
  LDS-read placement, emission and the LLVM tail settings, verifier limits;
- **new**: the high-VGPR window (`s_set_vgpr_msb`), per-class order on a shared counter and the XCNT rules, `s_wait_alu` rows, async / cluster /
  tensor DMA lowering and its completion protocol, split / named / cluster barriers, VOPD pairing;
- **n/a**: AGPR assignment, M0 reuse for LDS DMA.

## 5. Invariants (each enforced by a test or a gate)

| Invariant | Where |
|---|---|
| Asm of every reachable kernel is unchanged by a refactor (234 kernels) | `tests/golden/md5_manifest.tsv`, `crates/amdc/tests/md5_manifest.rs` ([tests/golden/README.md](../tests/golden/README.md)) |
| Dependencies point down; no kernel-name policy below L2 | `crates/amdc/tests/layer_check.rs` |
| Kernels, tiles and stages read no environment | `crates/amdgpu-kernels/tests/no_env_knobs.rs` |
| Every pass lists evidence, and the evidence paths exist | unit tests in `amdgpu-codegen/src/pass/mod.rs` |
| The compile policy is declared data (no kernel-name inspection) | `amdgpu_dsl::CompileProfile::policy` |
| Strict certificates (shipped 21 / 100 / 0) and the trust ledger | `amdc gpu-verify --shipped --strict`, [trust-inventory.md](trust-inventory.md) |
| Every reachable kernel compiles, passes the descriptor limits and assembles | `crates/amdgpu-kernels/tests/dispatch_compile.rs`, `python/tests/run_all.sh` |
| Every emitted instruction expansion declares its implicit SCC / VCC effects in the instruction table | `crates/amdgpu-kernels/tests/emit_effects.rs` |
| A behaviour change keeps every kernel family within noise | `tools/perf_ab.sh` ([perf-ab.md](perf-ab.md)), `tools/port_gate.sh` |
