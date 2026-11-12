# Two Weavers: interface, unit graph, ownership, migration (W1 design)

Design for W1 of [two-weavers-plan.md](two-weavers-plan.md). The front-end Weaver (W2, owner sched) plans a kernel at
unit level. The back-end Weaver (W3, owner backend-llvm-tail) orders, allocates and emits it at machine-instruction level.
This page fixes the interface between them, the machine-instruction unit graph, what today's mechanisms become, who owns
which code, and the migration steps of W3 with their gates.

Status: design for review. Nothing here is implemented yet.

## 1. Today

| Information | Carried today by | Read by |
|---|---|---|
| unit order | statement order of the IR (`Weaver::emit` splices the captured units in the chosen order) | lowering, unchanged by the back end except `sched-list` / `slotfill` / `ldahead` moves |
| phase boundaries | `Stmt::RegionMark { name, begin }` (weave regions, no code) → `MInst::region` → `MFunction::reg_hints.regions` | scheduler per-region strategies, register caps, `amdc regplan` |
| order policy per phase | `Function::regions: Vec<RegionDecl { name, authored, tokens }>` (+ `AMDC_SCHED_REGION`) | `sched::region::RegionSpec` |
| register plan | `Function::reg_plan: Vec<RegMark>`: `PinV`, `Share`, `Separate`, `Cap { region, vgprs, agprs }`, `Ring` → `MFunction::reg_hints` | `regalloc`, `regplan::check` |
| unit dependences, class, latency, register profile, iteration offset | `weave::Unit` (front end only; lost at emission) | `Weaver::plan` / `check_order`, `weave_auto` |
| software pipelining | `weave::rotate` (prologue / steady / epilogue copies by iteration offset), authored in the kernel | — (the back end sees three straight-line copies) |
| compile policy | `CompilePolicy` (lowering tokens, M0 reuse, scheduler preset) + `Function::sched_opts` / `lower_opts` | lowering, `sched` |
| register-pressure recovery | the outer ladder (`remat1..4`, `ldsspill*`) and the inner ladder (pressure cut / hoist cap / launch budget), `pass::OUTER_LADDER` / `inner_ladder` | `lower_and_allocate` |
| sync | barriers / waits as statements; stage contracts (`StageBegin`) and the tile / hand-off protocols (`gpu_protocol`) | lowering (auto waits), `waitcnt-recompute`, `gpu_verify` |

Two problems follow. First, everything the front end knows about a unit (dependences, iteration offset, class) is lost at
emission, so the back end recovers order freedom from statement facts with a token-configured list scheduler. Second,
lowering is whole-function and order-dependent (section 3.2), so the back end cannot reorder units without re-lowering.

## 2. Interface (crate `amdgpu-dsl`, module `plan`)

The plan travels with the `Function`: it is printed and parsed by irv1 (generated codec, `--from-ir` keeps working) and
checked by `amdgpu_dsl::verify`. Types are plain data with no behaviour. Both Weavers depend on them; neither depends on the
other.

```rust
pub struct KernelPlan {
    pub phases: Vec<Phase>,          // tree (parent), in program order
    pub units: Vec<UnitDecl>,        // every captured unit, with the IR range it owns
    pub deps: Vec<Dep>,              // binding (constraints)
    pub handoffs: Vec<HandOff>,      // binding
    pub sync: Vec<SyncRule>,         // binding
    pub hints: Hints,                // optional
}
pub struct Phase {
    pub id: PhaseId, pub name: String, pub parent: Option<PhaseId>,
    pub kind: PhaseKind,             // Straight | LoopBody { var: VarId, trip: Option<u32> } | Branch
    pub order: OrderSpec,            // binding
    pub budget: Budget,              // binding
}
pub enum OrderSpec {
    Authored,                        // the units' IR order is the order (today: `RegionDecl.authored`, `Weaver::order`)
    Free,                            // any order respecting `deps` / `sync` / `budget`
    Partial(Vec<(UnitId, UnitId)>),  // Free plus these extra edges
}
pub struct Budget {                  // per-phase caps; None = the kernel's limit
    pub vgpr: Option<u32>, pub agpr: Option<u32>, pub sgpr: Option<u32>,
    pub lds_bytes: Option<u32>,
    pub inflight: Vec<(CounterName, u32)>,   // e.g. at most N outstanding vmem in the phase
    pub m0: M0Use,                   // Free | Owned (this phase owns M0) | Pinned(value source)
}
pub struct UnitDecl {
    pub id: UnitId, pub name: String, pub phase: PhaseId,
    pub class: UnitClass,            // Mfma | Valu | LdsRead | Dma | Salu | Branch | Setup (today's weave::UnitClass)
    pub anchor: bool,                // keeps its position relative to the phase boundaries
    pub iter: i32,                   // iteration offset (today's `rotate` offsets)
    pub latency: u32,                // nominal, for the back end's priority only
}
pub struct Dep { pub from: UnitId, pub to: UnitId, pub kind: DepKind, pub distance: u32 }
                                     // DepKind: Data | Memory { space } | Order; distance = iterations (0 = same)
pub struct HandOff {
    pub value: ValueRef,             // a DSL value / variable (today's RegRef)
    pub from: PhaseId, pub to: PhaseId,
    pub via: Via,                    // Reg { file: Vgpr | Agpr | Sgpr, pin: Option<u32> } | Lds { offset, bytes } | Global
    pub layout: Option<LayoutRef>,   // the tile layout the consumer expects (contracts check it)
}
pub enum SyncRule {
    Barrier { after: UnitId, before: UnitId },                  // workgroup barrier between them
    Counted { producer: UnitId, consumer: UnitId, counter: CounterName, slack: u32 }, // e.g. ring slots
    CrossCta { flag: ValueRef, scope: Scope },                  // inter-CTA protocol (kept by gpu_protocol)
}
pub struct Hints {
    pub interleave: Vec<(UnitId, Vec<UnitId>)>,  // skeleton unit -> fillers to place in its shadow (today's `spread`)
    pub priority: Vec<(UnitId, i32)>,
    pub prefetch_depth: Vec<(PhaseId, u32)>,
    pub tokens: Vec<(PhaseId, String)>,          // transitional: today's scheduler tokens, removed at B7
}
pub struct Feedback {                // back end -> front end, per kernel (written by `amdc weave-report`, JSON)
    pub phases: Vec<PhaseFeedback>,  // peak vgpr / agpr / sgpr, lds, cycles on the critical path, wait states added
    pub feasible: Result<(), Vec<Infeasible>>,   // which constraint failed (budget, authored order vs hazard, hand-off)
    pub order: Vec<(PhaseId, Vec<UnitId>)>,      // the order chosen for Free phases
    pub rungs: Vec<String>,          // the recovery steps used (today's ladder rungs)
}
```

Additions after sched's review (in `plan.rs`): `KernelPlan::cta` (threads, waves, occupancy target), `roles` (`RoleDecl`: named wave
sets) and `UnitDecl::roles` (the waves that run a unit), `rings` (`RingDecl`: slots, bytes per slot, cadence, producer / consumer phase,
owner CTA / wave; the slot-version checks key on its id), `LoopBody { trip: Ext, peel }` (symbolic trip, the front end's peeled
iterations until B5), `Via::Reg { pin, dwords, align }`, `SyncRule::CrossCta { protocol: LastArriver | SeparateKernel | InCtaLds,
order }`, and `reg_constraints` (`Share` / `Separate` stay value-level allocator constraints, not hand-offs; `Pin { value, base }` = today's `PinV`, `RegRing { name, bufs }` = today's register `Ring`); `CrossCta::flag` is a `FlagRef` (a value, or a kernarg by name = the last-arriver counter).

Rules: constraints are binding, so violating one is a compile error naming the constraint, never a silent fallback. Hints
may be ignored. A phase without a plan entry gets `OrderSpec::Authored` and no budget, which is exactly today's behaviour
for code outside weave regions.

Mapping of today's mechanisms:

| Today | Plan |
|---|---|
| weave region (`RegionMark`) | `Phase { kind: Straight }` (loops / branches inside it become child phases) |
| `RegionDecl { authored: true }` / `Weaver::order` / `seq` | `OrderSpec::Authored` |
| `RegionDecl { authored: false, tokens }` | `OrderSpec::Free` + `Hints::tokens` (until the tokens are replaced) |
| `Weaver::spread(skeleton, fillers, per)` | `Hints::interleave` (front end may still emit the spread order as Authored) |
| `RegMark::Cap { region, .. }` | `Phase::budget` |
| `RegMark::PinV` / `Ring` / `Share` / `Separate` | `HandOff { via: Reg { pin } }` for values crossing phases; `Share` / `Separate` stay as allocator constraints on values (a `RegConstraint` list in the plan) |
| `weave::Unit::deps`, `iter` | `Dep`, `UnitDecl::iter` |
| `rotate` (prologue / steady / epilogue) | `Phase { kind: LoopBody }` + `UnitDecl::iter`: the back end may do the rotation (B5); until then the front end keeps emitting the three copies |
| stage contracts, ring waits | `SyncRule::Counted` / `Barrier` + `HandOff::layout` (the contracts stay the checker) |
| `CompilePolicy.sched_preset` | defaults for phases without a plan (Gemm / T128 presets become hint sets) |
| retry ladders | back-end internal; reported in `Feedback::rungs` |
| `AMDC_WEAVE_AUTO*`, `AMDC_SCHED_REGION` | experiment switches that edit the plan (front end), never the back end |

## 3. Machine-instruction unit graph (back end)

```rust
pub struct MUnitGraph {
    pub units: Vec<MUnit>,
    pub deps: Vec<MDep>,             // plan deps + dependences found on the MIR (registers, memory, M0, counters)
    pub phases: Vec<MPhase>,         // plan phases with their budgets and order spec
}
pub struct MUnit {
    pub decl: UnitId,                // the plan unit (or a synthetic one: prologue setup, constants, kernarg batch)
    pub body: Vec<MBlock>,           // virtual-register MIR; one block unless the unit has internal control flow
    pub live_in: Vec<VReg>, pub live_out: Vec<VReg>,
    pub res: ResVector,              // issue cycles per pipe, LDS / VMEM bytes, counter increments, M0 def / use
    pub iter: i32,
}
```

Today's `MInst::region` is joined by `MInst::unit` (a plan unit id, set by lowering from `Stmt::UnitMark` markers that the
front end emits around each spliced unit, in the same way `RegionMark` works today). The graph is built from the lowered MIR
by grouping instructions by unit. Instructions outside any unit form synthetic units: the entry setup, the epilogue, and
glue between phases.

### 3.1 Back-end Weaver pipeline

1. Build the graph (plan + per-unit MIR).
2. Order per phase:
   - Authored: identity.
   - Free: unit-level list scheduling over `deps`, then instruction-level interleaving inside the chosen unit order. This absorbs `sched-list`, `slotfill`, `ldshoist`, `wsink`, `ldahead` and `phase`.
   - LoopBody with `iter` offsets: modulo scheduling (B5).
3. Register allocation with phase-local budgets and hand-off registers fixed at phase boundaries (B6). Until then, today's `regalloc` runs on the linearised function.
4. The existing post-order passes, unchanged: waits, hazards (including the enforcing oracle), `m0-pad`, verify, emit.
5. Feedback.

### 3.2 What lowering does whole-function today (and must become per-unit)

| Lowerer state / pass | Why it is order-dependent | Per-unit form |
|---|---|---|
| `pending_vmem / smem / lds` + auto waits (`MInst::auto`) | waits are inserted from the program order at lowering | lowering emits no waits; `waitcnt-recompute` owns them on every path (today only the scheduled path) |
| `consts` cache + `scope_snaps` | a constant materialised by an earlier unit is reused by later ones | constants per unit, or a synthetic "constant pool" unit at phase entry that the units depend on |
| `rfl_cache` (readfirstlane per block) | reuse across units of a block | per unit; cross-unit reuse becomes a CSE after ordering |
| `alias_cache`, `view_copy` (view snapshots, `facts::stale_view_roots`) | a view of a loop variable is valid only until the next write in program order | the snapshot is taken at unit entry; the copy is decided per unit from the unit's own reads / writes |
| `lazy` (`addrfold`: value = reg + k) | the k is folded into later users in order | per unit; a value crossing units is materialised (or recorded as a hand-off with its constant) |
| `kabatch` (kernarg loads batched in the entry block) | entry-block whole-function rewrite | a transform on the synthetic entry unit |
| M0 (`m0-explicit`, `m0-reuse`, `m0-pad`) | assumes the final order | M0 is a resource on units (def / use); reuse is decided after ordering; `Budget::m0` declares phase ownership |
| `coalesce` (+ reg-plan renames) | copies across the whole function | per unit before ordering; cross-unit copies kept explicit and coalesced after ordering |
| `dce`, `cmpfold`, min / max fusion | dead values across units | per unit, with `live_out` from the graph |
| `remat`, `lds-spill` (outer ladder) | loop-level rewrites on the whole function | back-end recovery actions on the graph (B6) |
| `uniform_analysis` (`scalarize`) | dataflow over the whole function | stays a whole-function analysis before splitting; its result is a fact per value, independent of order |
| `assign-agpr` | function-level | phase-budget aware (B6); until then unchanged on the linearised function |

## 4. Code ownership

| Code | Today | After W2 / W3 | Owner |
|---|---|---|---|
| `Units`, `Unit`, `capture`, `Weaver`, `rotate`, `check_order` (`codegen/src/weave.rs`) | L4 | front-end Weaver crate (name chosen by sched) | sched (W2) |
| `weave_auto` (cost-based unit order, the machine model it reads) | L4 | front end; the model stays `amdgpu-target` `latency.weave` | sched (W2) |
| composition planning (stage-variant choice, CTA shape, wave roles, ring depth) | in the kernels | front end, recorded as per-shape-bucket rules | sched (W2) |
| `plan` types, codec, verify | — | `amdgpu-dsl::plan` | backend-llvm-tail drafts, sched reviews |
| unit graph, per-unit lowering, back-end ordering, schedule-aware RA | `lower.rs`, `sched/*`, `regalloc` | `amdgpu-codegen` (`bweave` module); `sched/*` folds into it step by step | backend-llvm-tail (W3) |
| waits, hazards, emit, verify | L4 | unchanged | backend-llvm-tail |

## 5. W3 migration steps and gates

Every step is a separate merge. Steps B0–B3 must reproduce today's asm exactly. Steps B4–B7 change behaviour and are gated
by W4.

| Step | Content | Gate |
|---|---|---|
| B0 | `plan` types + irv1 codec + verify; the front end (W2) emits a plan derived from today's markers / tokens / reg plan; nothing reads it | md5 manifest identical |
| B1 | `Stmt::UnitMark` → `MInst::unit`; `MUnitGraph` built from the lowered MIR (analysis only); `amdc weave-report <kernel>` prints graph + feedback | md5 identical; graph check: every plan dep holds in today's order |
| B2 | per-unit lowering, item by item from 3.2 (waits on every path first, then constants, views, `lazy`, M0, `kabatch`, `coalesce`, `dce`), units concatenated in today's order | md5 identical after each item; where an item cannot be identical (waits on the unscheduled path), it is split out as a gated step |
| B3 | back-end ordering on the graph: Authored = identity, Free = today's scheduler run per phase (`sched-*` passes become graph passes) | md5 identical |
| B4 | Free phases ordered by the back-end Weaver at unit level, using hints instead of tokens | W4: no family slower beyond noise; certificates, oracle, `gpu_verify` |
| B5 | cross-iteration (modulo) scheduling for LoopBody phases from `iter` / `Dep::distance`; the front end may stop emitting rotated copies per kernel | W4 per kernel; the rule is recorded per shape bucket |
| B6 | schedule-aware register allocation: phase-local budgets, hand-off registers at boundaries, remat / spill as graph actions (replaces the ladders and `RegMark::Cap`) | W4; `Feedback` reports used vs budget |
| B7 | remove `Hints::tokens`, region tokens, `AMDC_SCHED_*` switches and the ladders | md5 change allowed only where B4–B6 already changed it |

Every step: zero `unsafe` in the core crates; `cargo test` + `run_all.sh` (incl. the hazard oracle) green; certificates unchanged.

### 5.1 Features from the vendor-FMHA replica (W5, [fmha-replica.md](fmha-replica.md))

kernel-opt measured these against the vendor MXFP8 FMHA. They are scheduled into the steps above as follows:

| Feature | Where | Gate |
|---|---|---|
| missing MOps (`v_mov_b64` imm, `s_sub_u32`, `s_cmp_{ge,eq,le}_u32`, `s_addk_i32`, `s_or_b32`, `v_mul_i32_i24`, `v_cmp_nge_f32`, `v_mfma_f32_32x32x8_bf16`, `s_mov_b64 exec`) | first, next to B1: table rows + emit + LLVM rows + oracle coverage | md5 identical (unused until a kernel asks) |
| VALU VCC write -> VCC read distance (2 -> 1) | done separately (`valu-vcc-valu` target row; `tools/vcc_hazard_probe.sh`) | W4 (md5 changes) |
| authored `s_nop` fillers in pinned windows | already counted by the hazard passes (an explicit `s_nop N` covers N+1 states); checked in the replica | md5 identical |
| cross-block hazard distance instead of block-end MFMA padding (`edgepad` generalised) | hazard pass over the CFG (state carried along fall-through / back edges), before B4 | W4 |
| M0 planning across LDS-DMA issues (one M0 per group, offsets in the instruction) | B2 M0 item: M0 as a unit resource, grouping after ordering | md5 identical while order-preserving, then W4 |
| SGPR operands for wave-uniform values in VALU, constant-bus aware | per-unit lowering (B2) + operand-form selection after RA | W4 |
| block layout: cold blocks after the loop, branch polarity | a CFG layout pass on the graph (after B3) | W4 |
| multi-exit loops at machine level (one exit per unrolled half) | DSL construct (kernel-opt) + CFG support in the graph (after B3) | W4 |

## 6. Front-end coordination (with sched)

To be agreed before B0:
- the crate the front-end Weaver moves to, and when `Stmt::UnitMark` is emitted (around each spliced unit in `Weaver::emit`);
- the plan's place in the IR (a `Function::plan` field) and the irv1 printing order;
- how stage contracts become `HandOff::layout` and `SyncRule::Counted` (the contracts remain the checker);
- which experiments edit the plan (front end) and which stay back-end recovery (`Feedback::rungs`).

### 6.1 Status of the front-end derivation (W2b, sched)

`amdgpu_stage::weave::Weaver::emit` records the plan while it emits (no behaviour change: the asm md5 manifest is identical):
- one `Phase` per weave: the labelled weaves (`Weaver::phase`, i.e. the weave regions) keep their region name; the unlabelled weaves of `rotate` get plan-only names `rotate.prologue` / `rotate.steady` / `rotate.epilogue` (no region is emitted for them);
  `OrderSpec::Authored` (today every weave emits its declaration or explicit order);
- one `UnitDecl` per unit in emission order (id = the position in `plan.units`, class, `anchor` from the `Anchor` hint, `iter` from `rotate`, nominal latency) and one `Dep` per derived dependence (`Data` when a value
  flows from the producer to the consumer, otherwise `Order`; `distance` 0);
- `Hints::interleave` from `Weaver::spread` (skeleton unit -> the fillers placed in its shadow);
- each unit's statements are bracketed by `Stmt::UnitMark` with the unit's plan id, so `MInst::unit` ties the lowered instructions to it (`amdc weave-report` shows `N plan units`).
W2c completes the plan at `FnBuilder::finish` (`amdgpu-dsl/src/plan_derive.rs`, derived from the function's own declarations; nothing changes in the emitted code):
- `RegionDecl { authored, tokens }` -> `Phase::order` (`Authored` / `Free`) and `Hints::tokens`; `RegMark::Cap` -> `Phase::budget`; `RegMark::Share` / `Separate` -> `reg_constraints`;
- phases whose units sit inside a loop -> `PhaseKind::LoopBody` (`rotate.steady`: peel = (prologue present, epilogue present); `trip` stays None: the IR has no induction variable);
- stage contracts -> `HandOff` (a `live_out` value of an earlier stage that a later stage takes as `live_in`: registers, an LDS tile or global; the recorded fragment layout);
- the contract's multi-slot LDS tiles -> `RingDecl` (producer / consumer phases from the units that contain the tile's produce / consume regions; private tiles `owner: Wave`; the cadence is inferred from the slot count: 3 slots one window, else the two-phase offset);
- `CtaShape` from the workgroup size.
W2d derives `RegMark::PinV` -> `RegConstraint::Pin` and `RegMark::Ring` -> `RegConstraint::RegRing`. Not derived yet: the last-arriver `CrossCta`
(the plan form `LastArriver { counter: FlagRef, partials, splits }` exists), `Counted` / `Barrier` rules (the Order dependences carry them), wave roles,
the loop trip counts.

## 7. As built (status)

What is live on Main, what is opt-in, and how each part is gated. "Opt-in" means a lowering token or a plan entry that no shipped kernel sets
yet; such a part is md5-identical by construction and is enabled per kernel with a W4 gate.

| Part | Where | State | Gate |
|---|---|---|---|
| plan types, irv1 codec, `plan::verify` | `amdgpu-dsl/src/plan.rs`, `ir_codec.rs` | live; every weaved kernel carries a plan (W2b-W2d) | md5; `tests/plan_ir.rs`; `compose_facts` |
| unit marks, `MInst::unit`, unit graph, `amdc weave-report` | `lower.rs`, `bweave.rs` | live (analysis) | md5; `bweave` unit tests |
| B2 per-unit lowering caches | lowering token `unitlocal` | opt-in | `tests/unit_local.rs` |
| B2 uniform SGPR operands | tokens `sgpr` (binary ops, compares), `sgpr2` (commutative VOP2 ops, `v_sub` minuend) | opt-in | `tests/b2_operands.rs`, `tools/sgpr2_probe.sh` (hardware) |
| B2 M0 grouping of LDS-DMA issues | policy `m0_reuse` (GEMM) or token `m0reuse` (any kernel; the replica: 54 -> 18 M0 writes) | live for GEMM, opt-in elsewhere | md5; port_gate when a kernel adopts it |
| B2 waits on the unscheduled path | unscheduled body: allocation with lowering's waits in place, then `waits-strip-auto` + `waitcnt-recompute` + `hazard-value` + `verify-waits` on the allocated order | live | perf_ab all families; the situv2 regression of the first version (recompute before RA shifted the allocation) fixed by recomputing after RA |
| B3 ordering pass `bweave-order` (authored order) | scheduled body after `waits-strip-auto` (lowering's program-order waits do not pin a merge); unscheduled body at its head | live | md5 |
| B4 instruction-level interleave in Free phases | `bweave::interleave_with` (plan `Hints::interleave`; each skeleton instruction preceded by the fillers it depends on; validated against `sched::depgraph` built with ALU across LDS-DMA fences, minus ring DMA / read edges of different slots or tiles) | live, acts only on Free phases (none shipped yet; kernel-opt's `ring2` experiment uses it) | md5; unit tests; perf_ab + port_gate per kernel |
| B5 ring-slot dependences in the wait pass | `sched/waits.rs` `RingModel::of_function` (ring membership per unit from its tile markers; RAW DMA -> read, WAR read -> DMA, ageing at back edges) | active only for a plan with a cross-iteration DMA in a Free phase | `tests/ring_waits.rs` (mutation tests); `gpu_verify` GV-PLAN-RING checks the plan's slot claims against the protocol simulation |
| B6a phase budgets as scheduler limits | `scheduler_options` (Free phases) | live, Free phases only | md5; `tests/budget_sched.rs` |
| B6b register buffers / rotation | `RegMark::Ring` (pairwise disjoint buffers, live); `reg_rotate` / `RegMark::Rotate` (fragment i in buffer i mod n: one register per residue class from `lower-rotate`, liveness-checked) | disjoint buffers live; rotation archived as tag `archive/b6b-reg-rotate` (not merged: no kernel compiles with it) (the replica's `ring2r` needs 263 > 256 live dwords) | `regplan` verification; the rotation test in tag `archive/b6b-reg-rotate` |
| B6c per-phase fallback | the rung before the unscheduled fallback | live, Free phases only | `tests/budget_sched.rs` |
| CFG: jump threading, cold-block layout | tokens `jthread`, `coldlayout` (`cfg_thread.rs`, `begin_if_unlikely`) | opt-in (the replica uses them) | `tests/jthread.rs`, `tools/jthread_probe.sh` |
| W5 instruction forms | tokens `cmpneg`, `movb64`, `salu32`; `BinOp::MulI32I24`; bf16 32x32x8 MFMA | opt-in | `tests/mops_w5.rs`, `tools/salu32_probe.sh` |
| implicit SCC / VCC effects of every expansion | `tests/emit_effects.rs` | live check | cargo test |
| VALU VCC write -> VALU VCC read distance 1 (was 2) | target row `valu-vcc-valu` | live (user decision; vs128 B4 S2048 -0.5..0.8 % accepted) | perf_ab --quick, port_gate, hazard oracle allowance |
| verifier: select-refined loop variables | `gpu_verify` (`recompute` applies the override to `Opnd::Var`; a literal constant decides the refinement side) | live | `tests/gv_cycled_slot.rs` |
| B7 removal of tokens / switches / ladders | - | not started | - |

The pass list as built: `amdc passes --target gfx950` (catalogue: [pass-catalog.md](pass-catalog.md) section 9).

Gates for every step: md5 manifest (or, for a behaviour change, `tools/perf_ab.sh --quick` PASS against Main plus `tools/port_gate.sh` for the
shipped FMHA set), certificates unchanged, `cargo test`, `python/tests/run_all.sh` (incl. the hazard oracle). Design of B5 / B6:
[w3-b5-b6.md](w3-b5-b6.md).
