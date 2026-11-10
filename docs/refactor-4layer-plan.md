# Four-layer refactor plan (v10)

Decision record for the restructuring requested on 2026-11-10. It fixes the target
architecture, the phases, the ownership and the acceptance rules; the detailed design of each
phase lives in the documents each phase produces.

## 1. Layers

| Layer | Purpose | Crate(s) after P1 | Comes from |
|---|---|---|---|
| L1 graph | kernel fusion, op graph, dispatch rules, host plans | `amdgpu-graph` | `amdgpu-tile/opgraph`, `amdgpu-kernels/dispatch.rs`, the Python plan entry points |
| L2 stage | pipeline composition: stages, contracts, pipeline templates | `amdgpu-stage` | `amdgpu-tile` stage / contract / pipeline / handoff / components, plus templates extracted from the kernel families |
| L3 function | functional building blocks: tile ops, fragments, copies, DMA, softmax, reductions, epilogues; instruction atoms per target | `amdgpu-tile`, `amdgpu-layout` | the rest of `amdgpu-tile`, `amdgpu-isa` |
| L4 backend + Weaver | IR, analyses, target-independent passes, Weaver (region ordering and register organisation), per-target lowering / scheduling / allocation / hazards / waits / emission, verification | `amdgpu-dsl` (IR), `amdgpu-codegen` (passes, Weaver, verification), `amdgpu-target` (target descriptions: `gfx950`, `gfx1250` spec only) | `amdgpu-dsl`, `amdgpu-compiler`, `amdgpu-tile/weave*.rs` |
| kernels | per-family composition and per-shape orchestration only | `amdgpu-kernels` | unchanged role, thinner |

Dependencies point downward only: kernels → L1 → L2 → L3 → L4.

## 2. Phases

| Phase | Content | Acceptance |
|---|---|---|
| P0 | `Target` description: wave size, register files, LDS, dependency counters, hazard rule table, latency / scheduling model, instruction atoms, descriptor and emission format, LLVM tail settings. Move every gfx950 hard-coding into `target/gfx950`. | Golden md5 manifest of every registered kernel and every dispatch-reachable instantiation unchanged; certificates unchanged; cargo test green |
| P1 | Split crates per §1; move Weaver to L4; extract the L2 pipeline templates (N-slot ring with counted waits, two-group ping-pong, split-K / KV split + merge variants, grid maps, scale-through-DMA, row reductions, streaming element-wise, radix select); named L4 passes and a pass manager with one pipeline per target, each pass carrying its evidence (test / probe / LLVM oracle) | Same md5 manifest; `amdc passes --target gfx950` lists the pipeline; docs updated |
| G (parallel, spec only) | `target/gfx1250` as a documented description for MI450 **B0**: wave32, VGPR file and allocation granularity, fixed SGPRs, the split counters (LOAD / STORE / DS / KM / ASYNC / TENSOR / X) and their `s_wait_*` instructions, WMMA / SWMMAC software hazards, VALU dependency distances, VOPD, TDM descriptor and completion protocol, clusters, B0 rules; assembler smoke checks with the rocm-10.1 toolchain only | No passes, no hardware runs; every fact cites its source page; smoke checks assemble; no effect on gfx950 output |

gfx1250 implementation is future work; this round only checks that the `Target`
abstraction can express it.

## 3. Rules

- The md5 manifest is the regression oracle for P0 / P1: any change is a bug unless gated.
- One owner per moved file at a time; moves land in small merges so branches do not rot.
- No behaviour change is mixed into a move commit.
- Per-shape orchestration stays in kernel code (not in L1 / L2 generic rules).
