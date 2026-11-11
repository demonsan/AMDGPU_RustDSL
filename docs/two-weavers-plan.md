# Four layers + two Weavers (v11 plan)

Decision record of 2026-11-11. It extends [`refactor-4layer-plan.md`](refactor-4layer-plan.md); the detailed designs live in the
documents each work item produces.

## 1. Structure

| Part | Responsibility | Inputs | Outputs |
|---|---|---|---|
| L1 graph | op graph, fusion rules, dispatch rules | model op graph | segments for the front-end Weaver |
| L2 stage | stage variant families (resource vector, layouts, measured standalone performance), pipeline templates | — | units + contracts |
| L3 tile | functional building blocks, instruction atoms per target | — | units with facts |
| **Front-end Weaver** | fusion / composition analysis and extension: segments, stage-variant choice (composition optimum, not standalone optimum), CTA shape, wave roles, phase structure (ring depth, ping-pong, task chains), unit-level order or order constraints, per-phase resource budgets, hand-off edges (register / LDS / L2, layout), cross-CTA sync protocol. Decisions are pruned by a cost model, filtered by compile-only feasibility and decided by measurement; the result is recorded as per-shape-bucket rules | L1 segments, L2 variants, target resource model, back-end feedback | kernel plan + constraints / hints for the back end |
| **Back-end Weaver** (ISA level) | on the machine-instruction unit graph: final order across blocks and iterations (software pipelining), register allocation integrated with the schedule, register / LDS / M0 / counter hand-off between phases, hazards, waits, emission | unit graph (virtual-register MIR fragments, dependences, resource tags, phase boundaries), constraints / hints | assembly + feedback (resources used, feasibility, critical path) |
| L4 IR / target | IR, analyses, target descriptions (`gfx950`; `gfx1250` spec only) | — | — |

Interface between the Weavers: **constraints** (phase boundaries, unit dependences, authored order, resource budgets, hand-off
layouts, sync protocol — binding), **hints** (interleave candidates, priorities, prefetch depth — optional), **feedback**
(actual resources, feasibility, critical path). Both sides are independently verifiable: front end by stage contracts and
the tile / hand-off protocols; back end by the schedule verifier, the enforcing hazard oracle and `gpu_verify`.

## 2. Work items

| Item | Content | Acceptance |
|---|---|---|
| W1 design | `docs/two-weavers.md`: interface data structures, code ownership, migration steps | reviewed |
| W2 front-end Weaver | move unit capture / ordering (`weave`, `weave_auto`) and the composition planning out of codegen into the front-end crate; formal constraint / hint structures replacing ad-hoc region tokens and register hints | md5 manifest identical |
| W3 back-end Weaver | machine-instruction unit graph; per-unit lowering; ordering at ISA level reproducing today's output; absorb the scheduler; then cross-block / cross-iteration scheduling and schedule-aware register allocation | first md5-identical, then gated performance changes |
| W4 performance baseline | one harness covering every integrated kernel family: same-run A/B of the old and the new compiler output, vendor ratios where available | no family slower beyond noise; correctness and certificates unchanged |
| W5 experiment | reproduce aiter's FMHA assembly 1:1 through the safe DSL and the back-end Weaver (authored order, pinned registers), diff and measure | instruction-level diff report; performance parity |
| W6 laggards | apply back-end Weaver features to the kernels behind aiter (small-batch decode, MLA H16, backward, hd192 small grids, MXFP4, small-M GEMV) | only measured wins adopted |

## 3. Rules

- Rust safety: zero `unsafe` in the core crates (unchanged rule; the safe-asm path stays the only escape and is counted).
- Asm may change after the restructuring; performance may not drop: W4 is the gate for every behaviour-changing merge.
- Per-shape decisions stay recorded as rules with their evidence; no runtime search.
