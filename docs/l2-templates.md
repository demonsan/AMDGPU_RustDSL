# L2 pipeline templates: catalogue and extraction design (P1)

Design document for phase P1 of [`refactor-4layer-plan.md`](refactor-4layer-plan.md) (item 1 of the P1 design). It is based on Main `a5fc452d`. **No code moves until
the P0 `Target` abstraction has landed**; this document only fixes what the L2 layer (`amdgpu-stage`) will contain, which parts of today's kernel code
become generic template parameters, and which parts must stay per-shape orchestration in `amdgpu-kernels`.

Companion documents: [`pass-catalog.md`](pass-catalog.md) (L4 passes), [`crate-split.md`](crate-split.md) (module → crate mapping and merge sequence),
[`op-coverage-memtemplates.md`](op-coverage-memtemplates.md) (the memory-bound templates T1–T7, which this document reuses as the row / streaming / select taxonomy).

## 0. Definitions and the generic / orchestration test

An **L2 template** is a parameterised composition of L3 building blocks (tile ops, copies, DMA, softmax, reductions, epilogues) that fixes a **protocol**:
the cadence (who stages which tile when), the slot / generation discipline, the waits and barriers, the roles, the merge discipline. The caller supplies the
**work** of each phase and a handful of numbers. A template is the unit on which the verifier (tile protocol markers, wait-path analysis, buffer protocols)
can prove the protocol once and the kernels inherit the proof.

A piece of kernel code is **generic** (moves to a template) when all three hold:

1. the same structure appears in at least two kernels or kernel families,
2. the copies differ only by constants, closures or tile geometry,
3. its correctness is a property of the protocol (ordering, counted waits, versions), which the verifier already checks or can check.

It **stays in kernel code** (per-shape orchestration, plan §3) when it is any of: address maps and masks of one op, ABI / kernarg / contract declarations,
op-specific numerics, which template and which parameters a shape uses (dispatch thresholds are L1), measured per-shape tuning constants.

Template parameters come in four kinds:

| Kind | Example today | Lives in |
|---|---|---|
| value parameters | ring slots, tile rows, `vm_inflight`, `late_tile1_vm`, `rs_depth`, group count | the template's `Params` struct |
| body (work) | `PipeBody` hooks (`stage`, `v_phase`, `m_phase`, `window`, …) | trait implemented by the kernel |
| policy | `CompileProfile`, `sched_opts` tokens, `declare_region(..)` strategies, weave region names (`kv_stage`, `v_phase`, `m_phase`) | `Policy` attached to the template instance |
| target query | wave size 64, LDS 160 KB, `vmcnt` / `lgkmcnt` counters, `buffer_load … lds` DMA, M0, 8 XCDs | `Target` (P0), never a literal in a template |

## 1. What already exists as L2 material in `amdgpu-tile`

| Piece | File:line | Role today | Fate in P1 |
|---|---|---|---|
| `Pipeline`, `Cadence`, `TileIt`, `PipeBody` | `crates/amdgpu-tile/src/pipeline.rs:51 / 120 / 68 / 93` | tile pipeline driver: `TwoPhaseOffset`, `OneWindow`, `RegStaged` (2 or 3 slots) | becomes the core of `amdgpu-stage::ring` (templates T1, T2, T3); `TileIt` itself moves **down** into `amdgpu-tile::lane` first, because `reduce.rs:5` (L3) already uses it (`crate-split.md` P2) |
| `Roles` (`split`, `single`, `each`), `Role` | `crates/amdgpu-tile/src/lane.rs:54` (+ `Roles` in the same crate) | two-group dispatch with compile-time role tags | `amdgpu-stage::roles` |
| `StageInst`, `compose`, `merge`, `Prim` | `crates/amdgpu-tile/src/stage.rs:15 / 38 / 112 / 125` | stage composition with weave handover (fusion stage library) | stays, moves to `amdgpu-stage` |
| `check_boundary`, `StageBoundary`, `LiveFrag` | `crates/amdgpu-tile/src/contract_stage.rs:50 / 71` | producer / consumer fragment-layout contract check | moves to `amdgpu-stage` |
| `ConvPlan` handoff | `crates/amdgpu-tile/src/handoff.rs:15 / 115` | layout conversion between fused GEMM stages (`mlp`) | stays in `amdgpu-tile` (L3, decision D2 of `crate-split.md`: it is function-level code over `PrivRef` / `ConvPlan`); the L2 template only calls it |
| `kv_split_decode`, `kv_split_range(_var)`, `gqa_kv_head` | `crates/amdgpu-tile/src/components.rs:20 / 35 / 58 / 8` | KV-split index maths shared by decode / prefill | template T5 helper |
| `kvsplit_partial_f16`, `combine_splits` | `crates/amdgpu-tile/src/epilogue.rs:34 / 89` | partial-store and combine epilogues | template T5 |
| op graph + skeleton generator | `crates/amdgpu-tile/src/opgraph/{mod,skeleton,json}.rs`; generated `fmha/skeleton_{vs128,t128,kvsplit}.rs` | graph → `build_<kernel><S: Slots>` skeleton with slots filled by the kernel | L1 (graph) → L2 (skeleton) boundary; see §3.14 |
| `RowsCopy`, `issue_rows_shared`, `DwordRowsCopy` | `crates/amdgpu-tile/src/rows.rs:45 / 110 / 164` | DMA / dword row staging helpers | L3 (copy), used by T1 / T3 / T7 |
| `wave_reduce`, `lane_reduce`, `group_reduce_dpp`, `wave_allreduce`, `group_allreduce` | `crates/amdgpu-tile/src/prims.rs:163 / 152 / 240`, `tileop.rs:114 / 162` | reductions | L3; T8 composes them |
| `OnlineSoftmax` | `crates/amdgpu-tile/src/softmax.rs:31` | FMHA online softmax recipes | L3 |

## 2. Catalogue

| ID | Template | Status today | Main users |
|---|---|---|---|
| T1 | CTA-shared N-slot ring with counted waits (`RegStaged` 2 / 3 slots, `OneWindow` 3 slots, hand-rolled 2-slot) | **extracted** (`Pipeline`) for GEMM and FMHA t128; **hand-rolled** in bwd, MLA-hp, bf16 v1 | `gemm`, `fmha::{mod,bf16,bwd,mla}` |
| T2 | Two-group ping-pong (`TwoPhaseOffset`) + `s_setprio` | **extracted** | `fmha::{mod (vs128, kvsplit), bf16v3, fp8pt}` |
| T3 | Wave-private double buffering with counted waits (no barrier) | hand-rolled in each decode kernel | `fmha::{decode, decode16, mla}` |
| T4 | Split-K / KV-split + merge (separate kernel, in-CTA LDS, last-arriver) | partials + index maths extracted; merges per family | `gemm` (`splitk`), `fmha::{bf16v3 _ks, fp8pt, mod kvsplit, decode*, mla}`, `moe` multi-block |
| T5 | Grid maps (XCD raster, causal pairing, LPT, head packing, split index) | per kernel; `block3_x_mode` shared by the FMHA forward family | `fmha::{mod, bf16v3, fp8pt, bwd}`, dispatch (L1) |
| T6 | Scale-through-DMA (K-scale ring, scale panel, shuffled scales) | per family | `fmha::mod`, `gemm` |
| T7 | Row reductions (wave-per-row, group, DPP-only, block via LDS, sub-wave) | helpers in L3, orchestration per kernel | `rows`, `norm`, `topk`, `sampling`, `rope`-fused, `quant` |
| T8 | Streaming element-wise (+ `nt`), gather / scatter (T1 / T7 of the memory templates) | per kernel with shared helper pieces | `eltwise`, `dequant`, `cast`, `rope`, `cache`, `conv` |
| T9 | Radix select / histogram / prefix scan / counting sort | duplicated in `sampling` and `topk::perrow`, scan duplicated in `moe` | `sampling`, `topk::perrow`, `moe` |
| T10 | Epilogue families (LDS-staged transposed stores, fused act / quant, partial stores) | L3 helpers + per-kernel | `gemm`, `fmha` |
| T11 | Fused stage chain with layout handoff | extracted (`handoff`, `contract_stage`) | `mlp` |

Sections 3.1 – 3.11 describe each template with the same fields: **What**, **Code today**, **Users**, **Proposed L2 API**, **Generic vs orchestration**,
**Target-dependent**, **Notes** (measurements, risks). Section 4 is the cross-cutting design (verification contract, policies, naming, extraction order).

## 3. Templates

### 3.1 T1 — CTA-shared N-slot ring with counted waits

**What.** K / V / A / B tiles stream through a ring of N LDS slots with direct DMA (`buffer_load … lds`); each tile is *staged* into a slot, completed by a
`vmcnt` wait, published by one `s_barrier` and consumed; the slot is released for tile t + N. The verified protocol is: slot index and generation come from the
pipeline, every write sits under `tile_produce`, every read under `tile_consume`, the completion wait is a *counted* `s_waitcnt vmcnt(n)` when later DMAs may stay
in flight.

**Code today.**

- Extracted driver: `crates/amdgpu-tile/src/pipeline.rs` — `Cadence` (`:51`), `TileIt` (`:68`), `PipeBody` (`:93`), `Pipeline::run` (`:162`). Parameters that are already
  template parameters: `tile_rows`, `nkv_bound`, `prebind` (slot-offset values bound at the top of each iteration), `vm_inflight` (`OneWindow`: counted wait at the end
  of a window), `late_tile1_vm` (`RegStaged`: DMAs of tile 1 allowed in flight at the prologue barrier), `rs_depth` (`RegStaged`: **2 or 3 slots**, three tiles per loop
  pass with static slots 0 / 1 / 2, tile t + 3 staged while t + 2 is still in flight under `vmcnt(late_tile1_vm)`), `scalar_exit`, `key_unclamped`.
- `RegStaged` users: `crates/amdgpu-kernels/src/gemm/mod.rs:1943` (`Pipeline::new(Cadence::RegStaged, …)`), depth 3 selected at `:1948`
  (`cfg.depth3`, `GemmConfig` field `:67`), `late_tile1_vm` computed from the DMA issue counts at `:1945`; the 3-slot rings are declared at `:1715-1722`
  (`lds.ring("A" | "B" | "SC", …, slots)`).
- `OneWindow` user: `crates/amdgpu-kernels/src/fmha/mod.rs:823` (`_t128`, 3-slot K / V ring, `pipe.prebind = vec![32768, 512]` at `:824`).
- Hand-rolled rings (no `Pipeline`): `fmha/bwd.rs:170-171` (dq: `ring("K")`, `ring("V")`, `stage` closure `:182`, loop `:232`, `wait_vm(0)` `:240`),
  `fmha/bwd.rs:387-390` (dkdv: `ring("Q")`, `ring("dO")`, `ring("LD")`, `stage` closure `:424`, loop `:485`, `wait_vm(0)` `:497`),
  `fmha/mla.rs:391` (`mla_decode_bf16_hp`: 2-slot 36 KB KV ring, each wave issues 9 of 36 1-KB DMAs, loop `:451`, `wait_vm(0)` `:458`, barrier `:460`),
  `fmha/bf16.rs:187-188` (v1 attention forward, 2-slot K / V ring, kept as the non-v3 reference), `fmha/mod.rs:102-103` (MXFP8 K / V ring construction `KvRing3`).

**Users.** GEMM family (all `gemm_*` names of `gemm::by_name`, `crates/amdgpu-kernels/src/gemm/mod.rs:1437`: MXFP8 block-scale, a8w8 row/col fp8 and i8, a16w16, a4w4 /
a8w4 / a16w4, MoE grouped stages, conv2d / conv3d as implicit GEMM, `_bmm`, `_64` forms on the 3-slot ring, `_splitk` forms); FMHA `t128`; bwd dq / dkdv;
MLA-hp.

**Proposed L2 API** (`amdgpu-stage::ring`).

```rust
pub struct RingParams {
    pub slots: u8,                 // 2 | 3 (| 4: kernel-opt's measured lever for the small i8 GEMM)
    pub tile_rows: i32,
    pub cadence: Cadence,          // RegStaged | OneWindow | TwoPhaseOffset (T2) | Barrierless (T3)
    pub inflight_vm: u8,           // generalises vm_inflight / late_tile1_vm: VMEM ops of ONE stage that may stay in flight at the publishing wait
    pub prebind: Vec<i64>,         // slot strides whose offsets are bound at the top of every iteration
    pub scalar_exit: bool,
    pub key_unclamped: bool,
    pub nkv_bound: Ext,
}
pub trait RingBody {                // = today's PipeBody
    fn stage(&mut self, b: &mut FnBuilder, role: Role, key0: &I32, slot: &SlotIdx);
    /* prologue_stores, prologue_compute, loop_state, bind_state, phases as today */
}
pub fn run_ring(b: &mut FnBuilder, p: &RingParams, role: Role, body: &mut dyn RingBody);
```

The hand-rolled rings (bwd dq / dkdv, MLA-hp, bf16 v1) are ported onto `RingParams` with `Cadence::OneWindow` or `Cadence::RegStaged`-like phase sets only if the
resulting asm is md5-identical or the difference is measured and gated; the first step in P1 is *only* to move the existing `Pipeline` and to add the missing
parameter documentation, not to port the hand-rolled kernels (no behaviour change in a move commit).

**Generic vs orchestration.**

| Generic (template) | Orchestration (kernel) |
|---|---|
| slot / generation arithmetic, `prebind`, tile counter, peeled last tile, the order *stage → wait → barrier → consume*, counted-wait derivation from `inflight_vm`, protocol markers, loop exit through `read_first_lane` | what a stage issues (DMA descriptors, voffsets, swizzles, scale words), the work of each phase (MFMA blocks, softmax), key masks and causal limits, epilogue, LDS layout of one slot, which waves issue which blocks |

**Target-dependent.** DMA instruction form and its M0 setup (`buffer_load_dwordx4 … lds`, gfx950-only; gfx1250 uses the tensor data mover and the `ASYNC` / `TENSOR`
counters), counter semantics (`vmcnt` counts DMA completions here), the barrier primitive (`s_barrier`; clusters on gfx1250), LDS capacity (160 KB limits the slot
count × tile bytes: hd192 uses 80 KB per CTA, `bf16v3.rs` `small` form), `s_setprio` availability. In the template these are `Target` queries
(`target.lds_bytes()`, `target.dma_issue_form()`, `target.counters()`), not literals.

**Notes.** Depth 3 on the 64 × 128 GEMM forms measured 256×7168×4096 19.8 → 15.1 µs and 128×4096×8192 S4 16.0 → 13.9 µs (`gemm/mod.rs:1443-1451` comments);
a 3 – 4 slot ring with counted `vmcnt` for the small i8 GEMM is the recommendation of `docs/gemm-small-tile-profile.md` (not implemented). The `OneWindow` `vm_inflight`
constraint (the body must issue no other VMEM in the window and must not use `stage_complete`) is checked by the verifier's counted-`vmcnt` rule; the template must keep
that check in force by construction.

### 3.2 T2 — Two-group ping-pong with `s_setprio`

**What.** 512 threads = two groups of 4 waves running the same loop one phase apart: group A issues the matrix phase while group B runs the VALU (softmax) phase, then
they swap at the barrier; `s_setprio` raises the priority of the matrix phase. Cadence `TwoPhaseOffset` (`pipeline.rs:5-10`): `stage(t+1) | v_phase(t) | wait_vm(0) |
barrier | m_phase(t) | wait_lds(0) barrier`; the last barrier exists only in group A in the tail so both groups execute equal barrier counts.

**Code today.** `Pipeline::new(Cadence::TwoPhaseOffset, …)` at `fmha/mod.rs:713` (`_vs128`) and `:766` (`_kvsplit`), `fmha/bf16v3.rs:560`
(bf16 v3 hd128 / hd192), `fmha/fp8pt.rs:100` (per-tensor FP8). Groups: `Roles::split` (`bf16v3.rs:455`, `fp8pt.rs:188 / 214 / 266`, `fmha/mod.rs:709 / 764`,
`Roles::each` at `bf16v3.rs:565`); priorities: `set_prio` in `fmha/bf16v3.rs` (`v_phase` / `m_phase` hooks) and `fmha/mod.rs`; regions: `kv_stage`, `v_phase`,
`m_phase` weave regions (`bf16v3.rs` markers added in P0-adjacent work; `fp8pt.rs:178 / 262` `declare_region("v_phase", true, "")`).

**Users.** `fmha_fwd_hd128_mxfp8_v3_{vs128,kvsplit}`, `fmha_fwd_hd128_bf16_v3[_causal|_ks]`, `fmha_fwd_hd192_hd128_bf16_v3[…]`, `fmha_fwd_hd128_fp8pt_v3[_causal|_kvsplit]`.

**Proposed L2 API.** Same `RingParams` with `Cadence::TwoPhaseOffset` plus `PingPong { groups: u8, waves_per_group: u8, prio: bool }`:

```rust
pub struct PingPong { pub waves_per_group: u8, pub prio: bool }   // groups = 2
pub trait TwoPhaseBody: RingBody {
    fn v_phase(&mut self, b: &mut FnBuilder, role: Role, t: &TileIt);
    fn m_phase(&mut self, b: &mut FnBuilder, role: Role, t: &TileIt);
}
pub struct PhasePolicy {            // what ATT and the A/B runs showed is a per-kernel decision
    pub regions: &'static [(&'static str, RegionStrategy)],   // e.g. ("m_phase", Reorder), ("kv_stage", Reorder), ("v_phase", Authored)
    pub sched_tokens: &'static str,                              // e.g. "ldsink=1,wsink,ldahead32=1,rewait"
}
```

**Generic vs orchestration.** Generic: the phase offset, the equal-barrier-count rule, `set_prio` placement, the group-B prologue barrier, peeled tail, `key_unclamped`.
Orchestration: where the per-tensor m-scale multiply goes (it moved from the softmax phase to the end of the matrix phase and cut the traced wave from 302 K to 202 K cycles),
the softmax recipe, key masks, causal q-tile pairing, the choice of `PhasePolicy` per shape (e.g. hd192 uses `rewait` + regional `reorder` on `m_phase` and `kv_stage`;
hd128 keeps the whole-kernel reorder).

**Target-dependent.** Waves per group and the 2 × 4-wave arrangement assume 4 SIMDs per CU and wave size 64; `s_setprio` and the barrier semantics; the scheduling-policy tokens are
gfx950 list-scheduler knobs. gfx1250 (wave32, 2 SIMDs per WGP half?) is **not** assumed to support this shape; the template must take `groups` and `waves_per_group` from the target.

**Notes.** Phase balance, not register pressure, decides the speed: removing 21 VGPRs of offsets in hd192 did not help (`archive/bf16v3-derived-offsets`), moving the 32 `v_pk_mul`
did (1083 → 933 µs at B2 H32 8192², fp8pt); authored order for `v_phase` alone gave 1198 → 1083. These are policy parameters, and the template doc must list them as such.

### 3.3 T3 — Wave-private double buffering with counted waits

**What.** No cross-wave sharing: each wave owns private LDS tiles for its own key rows (DMA'd directly), issues the next tile's DMA before it computes on the current one and
waits with a counted `vmcnt(n)`; no barrier in the loop. Used where the wave's key range is private (paged-KV decode: each wave walks its own page list).

**Code today.**

- `fmha/decode.rs:125-126` (`lds.private("KV", 2 * KT * ROW, 4, …)`, loop `:148`, DMA `:177-178`, `wait_vm(0)` `:181`).
- `fmha/decode16.rs:259-262` (`lds.private("K" | "V" | "VS", …, 4, …)`), loop `:337`, K DMA `:418`, V DMA `:469 / :479 / :488`, **counted wait** `wait_vm(nv)` `:492`, final `wait_vm(0)` `:595`.
- `fmha/mla.rs:113-116` (`lds.private("KV0" | "KV1", …)` two private buffers per wave), loop `:152`, DMA `:186`, **counted wait** `wait_vm(if blk == 0 { 18 } else { 0 })` `:193`.
- Page lookup helper shared by these: `fmha/paged.rs` (`PageGroups`, derived from `decode16.rs`, see `docs/pa-decode16-profile.md`); `decode16.rs:315` `group_page` closure.

**Users.** `pa_decode_bf16_hd128_p<P>`, `pa_decode16_*` (bf16, fp8-token, MTP, fused, last-arriver), `mla_decode_bf16_p<P>`.

**Proposed L2 API** (`amdgpu-stage::ring` as `Cadence::Barrierless`):

```rust
pub struct WaveRingParams {
    pub bufs: u8,                       // 2 (K/V tiles per wave) or more
    pub inflight_vm: u8,                // counted wait: DMAs of tile t+1 stay in flight while t is computed
    pub private_bytes: Vec<(&'static str, i64)>,   // K, V, VS ... per wave
}
pub trait WaveRingBody {
    fn issue(&mut self, b: &mut FnBuilder, tile: &TileIt, buf: &SlotIdx);   // DMAs of one tile for this wave
    fn compute(&mut self, b: &mut FnBuilder, tile: &TileIt, buf: &SlotIdx); // QK / softmax / PV
}
```

**Generic vs orchestration.** Generic: buffer index / generation bookkeeping with `PrivRef` markers, the counted wait derived from the number of DMA issues per tile, the empty-part rule
(an empty key part runs one fully masked tile so there is no branch around the loop). Orchestration: page-table walk (`PageGroups`, `group_page`), `page` size, key masks and context length,
G-head packing of the 16 MFMA columns, fp8 / MTP variants.

**Target-dependent.** Wave size (64 lanes × b128 per DMA = 1 KB per DMA instruction), LDS-private bytes per wave against the 160 KB budget, counter semantics, the `lds_private` marker model
(per-wave windows); `permlane16/32_swap` used in the softmax combination (gfx950 instructions).

**Notes.** The counted waits are exactly what the vmcnt protocol checker proves (`wait-paths`); `mla.rs:193` shows a count that depends on the block index (18 vs 0), which the template
must express as a per-tile expression, not a constant.

### 3.4 T4 — Split-K / KV-split with merge variants

**What.** One reduction axis (K of a GEMM, keys of attention, experts' positions of a count) is cut into S parts computed by different CTAs; a merge combines the partials. Three merge variants:

| Variant | Mechanism | Where |
|---|---|---|
| **(a) separate merge kernel** | partials in a workspace (fp32 or fp16 + LSE), a second kernel combines them (log-sum-exp weights for attention, a sum for GEMM) | `gemm/reduce.rs:10` (`gemm_splitk_reduce`); `fmha/decode.rs:245-262` (`attn_combine_bf16_hd128`, `_x`, `_xs`); `fmha/bf16v3.rs:692` (`attn_combine_bf16_hd128_q16`); `fmha/mod.rs:795` (`…kvsplit_combine8h`); `fmha/mla.rs:5` (`attn_combine_bf16_hd512`); `gemm/reduce.rs:92 / 97` (`moe_sum_bf16`, `moe_sum_bf16_x`) |
| **(b) in-CTA merge through LDS** | the CTA's waves each hold a partial of their own key region and merge it in LDS over dead buffers; no partial ever leaves the CTA | `fmha/decode16.rs:47-49` (`pa_decode16_fused`: "the CTA's 4 wave partials are merged in LDS (over the dead K / V regions)", `EPI` tile `:263-267`, `lds.tile_over("EPI", …, &[&kreg, &vreg])`) |
| **(c) last-arriver** | every split CTA stores its partials, releases (per-wave drain + L2 write-back), takes a ticket with one atomic on `counter[b·HK + hk]`; the CTA that draws ticket S − 1 acquires and merges, and resets the counter | `fmha/decode16.rs:54-56` (`pa_decode16_lastc`), body `:750-862` (`buffer_atomic_add_ret` `:767`, ticket broadcast through the `TK` tile `:269 / :776-777`, `wg_winner` contract `:782`); probes `probe_last_arriver*` in `crates/amdgpu-kernels/src/probes` |

Related: the **multi-block MoE sorting** (`moe/multi.rs`: `moe_count` → `moe_scan` → `moe_scatter` / `moe_scatter_ordered`, workspace `ws[e][g]`, three kernels) is the same
"partition → workspace → merge by a later kernel" shape applied to a counting sort (variant (a)), see T9.

**Code today for the split side (partials and index maths).**

- GEMM: `GemmConfig::split_k` (`gemm/mod.rs:50`, builder `:222` `.splitk()`), `KS` / `KT` symbol ranges `:474-480`, loop bound `ks_v` `:1892`, contract bound `Ext::ka(88)` `:1943`,
  partial store `:2142`.
- Attention: `components::kv_split_decode` / `kv_split_range(_var)` (`components.rs:20 / 35 / 58`), `epilogue::kvsplit_partial_f16` (`epilogue.rs:34`) for the FP8 / MXFP8 forms,
  fp32 partial O / l + LSE for bf16 v3 `_ks` (`bf16v3.rs:411 / :425` split grid, partial store `:579`; kernarg `S` at 660, partials at 672 / 688 documented at `:317`),
  `fp8pt.rs:19-29 / :68` (`declare_io_pt(…, kvsplit)`, shares the MXFP8 partial layout and combine kernel), `fmha/mla.rs:79 / :276 / :313` (NS parts per (b, h)).
- Dispatch of the variants (L1): `dispatch::decode16_plan` chooses fused (`_f`, S = 1) / last-arriver (`_c`) / split + combine; `dispatch::hd192_prefill_kernel` etc.

**Users.** GEMM `_splitk` forms, FMHA `_ks` / `kvsplit` forms (bf16 hd128 / hd192, MXFP8, fp8pt), all paged-KV decode kernels, MLA decode, `moe_sorting` multi-block.

**Proposed L2 API** (`amdgpu-stage::split`).

```rust
pub enum MergeKind { Kernel(CombineSpec), LdsInCta(LdsMergeSpec), LastArriver(TicketSpec) }
pub struct SplitParams {
    pub axis: SplitAxis,               // K (GEMM), Keys (attention), Positions (counting)
    pub parts: Ext,                    // S, a power of two with a contract bound
    pub partial: PartialLayout,        // dtype (f32 | f16), LSE slot, [..][S][..] major order (split-major / split-minor)
    pub merge: MergeKind,
}
pub fn split_range(b: &mut FnBuilder, p: &SplitParams, z: ValueId, len: ValueId) -> (ValueId /*lo*/, ValueId /*hi*/, ValueId /*part*/); // = components::kv_split_*
pub fn store_partial(b: &mut FnBuilder, p: &SplitParams, acc: &AccTile<..>, part: ValueId, lane: &Lane);                               // = epilogue::kvsplit_partial_f16 and the fp32 form
pub fn combine_kernel(spec: &CombineSpec) -> Function;                       // attn_combine_* / gemm_splitk_reduce / moe_sum_*
pub fn lds_merge(b: &mut FnBuilder, p: &LdsMergeSpec, ..);                   // decode16 fused
pub fn last_arriver(b: &mut FnBuilder, p: &TicketSpec, release: impl FnOnce(..), merge: impl FnOnce(..));   // decode16 _c
```

**Generic vs orchestration.**

| Generic (template) | Orchestration (kernel / L1 dispatch) |
|---|---|
| partial layout and index maths, the empty-part rule (an empty part writes O = 0, LSE = −∞ so the merge weights it 0), the combine kernels (log-sum-exp merge is one family: `_x` / `_xs` variants for split-major vs split-minor and attention sinks), the last-arriver release / ticket / acquire protocol with its `wg_winner` contract, the LDS-merge over dead buffers | the number of parts S and which merge a shape uses (`dispatch.rs`: measured same-run against the vendor), per-op partial widths (hd512 for MLA, 128 for hd128), the masks that make a part empty, MTP / qlen rows, the causal range of a part |

**Target-dependent.** Release / acquire primitives for the last-arriver (L2 write-back + `buffer_atomic_add_ret` with the gfx950 memory model; the cache-coherence rules are target facts), atomic widths,
cross-XCD visibility (8 XCDs, separate L2s: the release must reach the coherence point), LDS size for the in-CTA merge.

**Notes.** The last-arriver correctness argument (one lane takes the ticket: "an atomic by every lane … serializes 256 same-cache-line atomics per CTA: ~100 µs at B16",
`decode16.rs:760-761`) is part of the template contract, not of the kernel. The decode16 fused and last-arriver forms are the production path of `dispatch::decode16_plan`; the older split + combine form
remains for experiments and the MTP / fp8 variants (`decode16.rs:25-26`).

### 3.5 T5 — Grid maps (XCD raster, causal pairing, LPT, head packing, split index)

**What.** The mapping from the hardware workgroup id (x, y, z) to the logical work item (q tile, head, batch, part) is a first-class design choice: it decides L2 sharing across the 8 XCDs
and load balance in causal kernels.

**Code today.**

| Map | Where | Notes |
|---|---|---|
| **XCD raster** | `fmha/mod.rs:926-980` (`block3_x`, `block3_x_mode`, `xcd` flag; the remapping that gives XCD x the contiguous logical range `[x·N/8, (x+1)·N/8)` of the (q tile, head) plane; for X > 16 chunks of 16 consecutive blocks cycle across XCDs, measured +0.2 … +3.3 % at S 8192 / 16384; its verification contract is `wg_bijection` + `assume_trusted` ranges `:974-976`) | users: MXFP8 vs128 / kvsplit / t128, fp8pt non-causal (`fp8pt.rs:181-183`), via `FmhaBlock`; **not used** by bf16 v3 ("XCD order of the (q tile, head) grid measured no gain", `bf16v3.rs:452`) |
| **Causal pairing** | `fmha/bf16v3.rs:408-409` (`grid_is(0, ceil(sq/qr)/2)` when paired), pass loop `:494`, `fresh2` lane partitions for the second pass `:481`, hd192 pairing rules `:325-330`; `fp8pt.rs:67` (`pair`) | CTA x takes q tiles x and nt − 1 − x (equal work per CTA, grid halved); second pass needs fresh lane partitions or the pair does not fit the VGPRs (hd192) |
| **LPT (longest first)** | `fmha/bwd.rs:346-380` (causal dq / dkdv launch order built in: key-tile-fastest grid, long tiles first, because the hardware deals workgroups round-robin to 8 XCDs in id order); host side `dispatch.rs:84-86` | B1 H16 4096² 598 / 450 → 407 µs, B4 H8 2048² 323 / 249 → 228 (dispatch comment) |
| **Split index** | `bf16v3.rs:424-425` (z = batch · S + part), `components::kv_split_decode` (`components.rs:20`) | part of T4 |
| **Head packing / GQA** | `components::gqa_kv_head` (`components.rs:8`), decode16 G ≤ 16 heads in the 16 MFMA columns | per kernel |
| **Persistent / grid-stride** | none in the shipped kernels; `op-coverage-memtemplates.md` §5.10 item 6 records per-tile ≥ grid-stride for streaming kernels | recorded as *not used* |
| **Snake** | retired (archived tag; not in Main) | do not reintroduce without a measurement |

**Users.** FMHA forward family (XCD, pairing), FMHA backward (LPT), decode (split index), `dispatch.rs` (L1) picks forms and part counts.

**Proposed L2 API** (`amdgpu-stage::gridmap`).

```rust
pub enum GridMap {
    Plain,
    XcdRaster { xcds: u8, chunk: u32 },         // xcds from Target; chunk 16 for X > 16
    CausalPair,                                 // CTA x -> tiles (x, nt - 1 - x)
    Lpt { fastest: Axis },                      // longest-first launch order
    SplitMajor { parts: Ext },                  // z = batch * S + part
}
pub struct GridPlan { pub dims: [Ext; 3], pub map: GridMap }
pub fn logical_ids(b: &mut FnBuilder, g: &GridPlan, rows_lg: i32) -> LogicalIds;        // emits the maths + the wg_bijection contract
pub fn grid_dims(g: &GridPlan, shape: &ShapeExt) -> [Ext; 3];                           // also used by L1 host plans
```

**Generic vs orchestration.** Generic: the id maths and its bijection / range contracts (so verification is by construction), the formulas of the grid dims shared by the kernel and the host plan.
Orchestration (stays): which map a shape uses (measured per shape: XCD helps MXFP8 and hurts bf16 v3), the causal limit arithmetic of a tile, the q-tile / head sizes. The choice itself is L1 / kernel.

**Target-dependent.** **All of it**: the number of XCDs (8 on MI355X), the round-robin deal of workgroups to XCDs, the L2 per XCD, CU count per XCD (32), 256 CUs. These must come from `Target`;
on a different part (one L2) the XCD raster degenerates to `Plain`.

**Notes.** The `wg_bijection` and `assume_trusted` ranges at `fmha/mod.rs:974-976` are *trust entries* in the certificate (see `docs/trust-inventory.md`); extraction should replace them
by a verified construction (the map is a bijection by formula) where the verifier can prove it.

### 3.6 T6 — Scale-through-DMA

**What.** Block / group scales (E8M0 or fp32) of quantised operands are staged by the same DMA path as the data and read from LDS at fixed lane positions, instead of per-lane global loads;
the layout of the scale tensor is chosen so one dword per lane is a coalesced load (shuffled layout) or so one DMA fills a whole panel.

**Code today.**

- FMHA MXFP8: K-scale staging as a layout-derived DMA, "one dword per row, waves 2 and 3 repeat waves 0 and 1" (`fmha/mod.rs:59`, `KvRing3` ring with `ks / vs / kd` slots `:72-143`, reader TV `:67`, `kdpart` `:141`, buffer-form DMA `:168`); the `_t128` form stages K scales through VGPRs (`:176`, `DwordRowsVgpr` in `rows.rs:132`);
  per-tensor FP8 removes the scale DMA entirely (`fp8pt.rs:7`, `kscale = false` at `mod.rs:99-110`).
- GEMM: `Scales::{Block128, Mx128, Mx32, RowCol, None}` (`gemm/mod.rs:174`); Mx128 scale **panel** (`PANEL_KB = 64`, `:252`; `SCP` tile `:1724`; "K ≤ 128 PANEL_KB: the panel holds every K group of the CTA" `:474`);
  Mx32 with the **shuffled layout** `[rows/16][KT/2][64 lanes][4 bytes]` (`:55`, `scale_shuf`, `a4w4` config `:226`); depth-3 + Block128: per-tile fp32 scales through the DMA stage into the `SC` ring (`:869`, `:914`).

**Users.** `fmha_fwd_hd128_mxfp8_v3_*`, `gemm_mxfp8_bs_v4`, `gemm_a4w4_*`, `gemm_a8w4_*`, `gemm_a8w8_blockscale*`.

**Proposed L2 API** (`amdgpu-stage::scales`).

```rust
pub enum ScaleLayout { PerRowDword, Panel { kb: u32 }, Shuffled { rows_per_group: u32, lanes: u32 }, BlockF32 }
pub struct ScaleStage { pub layout: ScaleLayout, pub through: ScalePath }       // Dma(ring slot) | Vgpr | Unit
pub fn scale_stage(b: &mut FnBuilder, s: &ScaleStage, ring: &Shared, ..) -> ScaleHandle;  // issue + (later) read
pub trait ScaleConsumer { fn read(&mut self, b: &mut FnBuilder, h: &ScaleHandle, tile: &TileIt) -> Vec<ValueId>; }
```

**Generic vs orchestration.** Generic: "one DMA slot per tile in the same ring generation as the data", the reader TV derivation, the `ScalePath::Unit` degenerate case (the LDS plan keeps its place so the shipped layout and md5 are unchanged, `fmha/mod.rs:99`), the panel size bound as a contract.
Orchestration: which scale format the op uses (E8M0 vs fp32), the fold of scales into the MFMA (`v_mfma_scale_*` operand slots) or into VALU epilogue, the scale shuffle of the host tensors.

**Target-dependent.** `v_mfma_scale_f32_*_f8f6f4` scale operand format (gfx950), E8M0 semantics, dword DMA size, 64-lane coalescing of the shuffled layout.

### 3.7 T7 — Row reductions (wave-per-row, group, DPP, block, sub-wave)

**What.** Reduction along a row with the result fed back to every element (norm, softmax, per-token / per-group quant, top-k gating). The scope ladder is: sub-wave group (16 / 32 lanes), wave (one row per wave, N ≤ 8192), block (LDS + barrier, one row per block), and the long-row two-pass form.

**Code today.**

- Primitives (L3): `prims::lane_reduce` (`prims.rs:152`), `wave_reduce` (LDS private scratch, `:163`), `wave_reduce_dpp` (`:240`), `tileop::{row_allreduce16, wave_allreduce, group_allreduce}` (`tileop.rs:103 / 114 / 162`), `group_reduce_dpp` (used by `topk/mod.rs:17 / 43-45`; "the DPP all-reduce, +9…19 % measured against the former LDS exchange for this latency-bound kernel").
- Wave-per-row kernels: `rows/mod.rs` (`row_kernel` `:96`, `rows_per_wave` `:70`, `group_lanes` `:82`, wrapper `wr` `:56`); quantisation `rows/quant.rs` (`group16_max` `:75`, `quant_kernel` `:100`, `lane_reduce` + `wave_reduce` `:218-232`); the MXFP4 norm+quant (`rows/mxnorm.rs`, cross-wave `RS` tile); `norm/` (RMSNorm + per-token fp8, stage-library composed).
- Block-level: `sampling/mod.rs:80` (`block_reduce`, LDS wave exchange + reduction ring, `ws` private + `blk` ring tiles), `topk/perrow.rs` (DPP wave scan + 4 wave totals).
- Gating: `topk/mod.rs` (softmax over experts + top-k), `topk/gate.rs` (`moe_fused_gate`).
- Online (row statistics in registers inside a pipelined loop): `softmax::OnlineSoftmax` (`softmax.rs:31`), `reduce::row_max` (`reduce.rs:52`), `RowSum2` smfmac row sums (`rowsum.rs:20`).

**Users.** `rmsnorm_*`, `add_rmsnorm_*`, `layernorm`, quant kernels, `rmsnorm_quant_mxfp4_*`, `topk_softmax`, `moe_fused_gate`, `argmax`, `sample`, FMHA softmax.

**Proposed L2 API** (`amdgpu-stage::rowred`).

```rust
pub enum RowScope { Group { lanes: u8 }, Wave, Block { waves: u8 }, TwoPass }
pub struct RowRed { pub scope: RowScope, pub op: RedOp, pub exchange: Exchange }   // Exchange: Dpp | LdsPrivate | LdsBlock(ring)
pub fn row_reduce(b: &mut FnBuilder, r: &RowRed, x: ValueId, ctx: &mut RowCtx) -> ValueId;
pub struct RowKernel<'a> { pub rows_per_wave: i64, pub vec: u8, pub load: LoadStage, pub stats: &'a [RowRed], pub epilogue: &'a dyn Fn(..) }
```

**Generic vs orchestration.** Generic: scope selection from (row length, rows per wave, group lanes), the exchange tile / ring declarations with verified versions (the rounds counter of `block_reduce` is exactly the template's slot discipline: one slot per round, `SlotIdx(round, …)`), the DPP-vs-LDS choice (a policy with the measured crossover from `op-coverage-memtemplates.md` §5.10 item 3), the masked tail handled by buffer descriptors (hw-guarded, no tail branch).
Orchestration: what is reduced (sum of squares, amax, max with index), the element-wise stages before / after, the quantisation format, `rows_per_wave` and vector width per N.

**Target-dependent.** Wave size 64 (quad / row-16 / row-32 DPP shapes, `permlane16/32_swap`), DPP control encodings (`row_shr`, `row_bcast`, `wave_shr` availability and semantics), LDS bank structure for the private scratch, max VGPR budget (occupancy target ≥ 16 waves / CU).

**Notes.** The memory-bound performance of these kernels is decided by occupancy, load-first ordering and bytes in flight, not by the scheduler (`op-coverage-memtemplates.md` §5.10 items 1, 8, 9) — they are template parameters (`vec`, `unroll`, `rows_per_wave`, `vgpr_budget`), not discovered.

### 3.8 T8 — Streaming element-wise and gather / scatter (with `nt`)

**What.** One b128 access per lane and input, one tile per block (no grid-stride loop), all loads issued before the compute, outputs stored with the non-temporal hint; the out-of-bounds tail by buffer descriptor `num_records`. Gather / scatter forms add an index load (page table, slot map, route list) in front of the same body.

**Code today.** `eltwise/mod.rs` (`by_name` `:12`, cache policy default `NT` `:15-16`, `act_mul_bf16` `:81`, bias / limit / OAI variants `:35`, unary kernels `:164`, quant static `:22`), `dequant/mod.rs`, cast kernels (`cvt_*`), `rope/`, `cache/mod.rs` and `cache/mla.rs` (KV-cache write: scatter by slot map, paged, flash / blocked / fp8 layouts), `conv/mod.rs` (`conv1d_update` `:35`, `conv1d_fwd` `:176`: chunked with a state handoff), `topk` / `moe` gathers, `moe_reduce` (`gemm/reduce.rs:92`).

**Users.** `silu_mul_*`, `gelu_tanh_mul_*`, `swiglu_oai_*`, unary / binary, `cvt_*`, `quant_*`, `dequant_*`, `rope_*`, `reshape_and_cache_*`, `concat_and_cache_mla*`, `conv1d_*`.

**Proposed L2 API** (`amdgpu-stage::stream`).

```rust
pub struct StreamParams { pub vec: u8 /*8 bf16 per b128*/, pub tile_elems: u32, pub tiling: Tiling /*FlatRows | ColumnTiles*/, pub cache: CachePolicy /*NT default, NONE for shared tables*/, pub inputs: u8 }
pub trait StreamBody { fn compute(&mut self, b: &mut FnBuilder, x: &[RegTile]) -> Vec<RegTile>; }
pub fn stream_kernel(name: &str, p: &StreamParams, body: &mut dyn StreamBody) -> Function;
pub struct GatherStream { pub index: IndexSource /*SlotMap | PageTable | Routes*/, pub p: StreamParams }
```

**Generic vs orchestration.** Generic: loads-first issue order, tile geometry from `vec`, the `nt` policy (streamed x / residual and outputs `nt`; small tables reused by many blocks not), descriptor-guarded tails, the grid formula. Orchestration: the activation, the dequant / quant maths, RoPE's pairing, cache layout formulas (page, block, flash), conv state carry.

**Target-dependent.** The b128 access width and the cache-policy bits (`sc0 / sc1 / nt` on gfx950), buffer descriptor format and `num_records` OOB semantics, 64-lane coalescing, bf16 / fp8 conversion instructions (`v_cvt_pk_bf16_f32`, fp8 conversion with / without saturation, gfx950-specific).

**Notes.** `nt` measured +3 … +14 % on streaming kernels (silu_mul D = 4096: 5.98 → 6.56 TB/s) (`op-coverage-memtemplates.md` §5.10 item 5); the template defaults must keep that.

### 3.9 T9 — Radix select, histogram, prefix scan, counting sort

**What.** Selection by 8-bit digit passes: clear the histogram, build it with LDS atomics, scan it, find the bin that contains the target rank, narrow the candidate set; plus the emit pass with a counter. The prefix scan (Hillis–Steele ping-pong in LDS, or DPP wave scan + wave totals) and the counting-sort (histogram → exclusive scan → scatter with returning atomics) are the building blocks of `moe_sorting`.

**Code today (duplicated).**

| Block | `sampling/mod.rs` | `topk/perrow.rs` | `moe/mod.rs`, `moe/multi.rs` |
|---|---|---|---|
| histogram pass (LDS atomics) | `hist_pass` `:266` | `hist_pass` `:194` | `atomic_lds` `moe/mod.rs:124` (counts), `:294` (returning scatter) |
| clear (version-bumping) | `clear` `:315` | `clear` `:227` | — |
| scan + find the target bin | `scan_find` `:325`, `select` `:414` | `scan_find` `:235` (DPP wave scan + wave totals) | Hillis–Steele ping-pong `while s < 512` `moe/mod.rs:145`, `sampling/mod.rs:545-555` (`while s < T`) |
| emit with counter | (sampling emits through the CDF) | emit pass `:307` (counters + emit sink in the `CS` tile) | `moe_scatter_ordered` (`multi.rs`: per-wave counts `WC`, `CURSOR`, `BASE`) |
| block reduce / exchange | `block_reduce` `:80` | — | — |

All three now declare their LDS through tiles (`lds-untracked` = 0): `lds_decl.rs` helpers (`tile_at`, `ring_at`, `private_at`, `owned_at`) and compile-time version counters (`hv` per clear, `rv` per `scan_find`).
The scan ping-pong versions (`step` / `step + 1`) and the max-clamped neighbour address (`pi = max(prev_i, 0)`) are the same code in all three places.

**Users.** `argmax_*`, `sample_*`, `sample_topkp_*`, `topk_per_row_decode / prefill`, `topk_plain`, `moe_sorting_*`, `moe_count / scan / scatter / scatter_ord`.

**Proposed L2 API** (`amdgpu-stage::select`).

```rust
pub struct Hist { pub bins: u32, pub digit_bits: u8, pub sink: SinkMode /*per-lane sink slot for masked lanes*/ , pub tile: HistTile }
pub fn hist_clear(b: &mut FnBuilder, h: &mut Hist);                       // bumps the version
pub fn hist_add(b: &mut FnBuilder, h: &Hist, bin: ValueId, weight: Option<ValueId>);
pub fn scan_find(b: &mut FnBuilder, h: &Hist, target: ValueId, kind: ScanKind /*DppWave | LdsHillisSteele*/) -> (ValueId /*bin*/, ValueId /*rank_below*/);
pub fn radix_select(b: &mut FnBuilder, p: &SelectParams, key: ValueId, k: ValueId) -> (ValueId /*threshold*/, ValueId /*rem*/);
pub fn counting_sort(b: &mut FnBuilder, p: &SortParams, ..);              // moe_sorting offsets / scatter
```

**Generic vs orchestration.** Generic: histogram layout (256 bins + per-lane sink slot), version discipline across clears and scans, the scan ping-pong with the clamped neighbour address, the owner-column prefix pattern (`lds_owned`), the threshold / rem contract. Orchestration: the key transform (order-preserving f32 key, bf16 logits, weighting), what is emitted (indices, CDF, top-p filter), the MoE padding / expert-id tile rules, tie-breaking policy.

**Target-dependent.** LDS atomics semantics (`ds_add_u32`, returning forms, rate), bank structure, wave size for the wave-scan totals (4 wave totals assume 1024 threads / wave 64), DPP scan encodings.

**Notes.** The tile-protocol work found two real races in this family (missing barrier in `moe_sorting`, missing `wait_lds(0)` before the publishing barrier in `moe_scan`); the template must make "complete before publish" structural.

### 3.10 T10 — Epilogue families

**What.** The accumulator → memory path: direct bf16 stores, LDS-staged transposed stores (wave-private region reusing dead ring storage), fused activation / quantisation, partial stores for splits, scaled stores (per-tensor / per-row / per-column).

**Code today.** `amdgpu-tile/src/epilogue.rs` (`store_bf16` `:12`, `kvsplit_partial_f16` `:34`, `combine_splits` `:89`), `gemm/mod.rs` (`lds_epilogue`, `epi_phases`, `EPI_PITCH` stage at `:1725`), `fmha/mod.rs` (R2S staging `:1219`), `fmha/bf16v3.rs:555-660` (O^T / l, 8-byte stores, LSE), `fp8pt.rs` (`out_scale`), `decode16.rs` (EPI tile over dead K / V regions).

**Users.** GEMM, FMHA forward, decode.

**Proposed L2 API.** Keep as L3 helpers plus an L2 `EpiloguePlan` that picks direct / LDS-staged by acc layout and target; do not extract a pipeline template (the phases differ in too many per-kernel ways). **Not a template in the first P1 round**; included for completeness because the plan lists epilogues in L3.

### 3.11 T11 — Fused stage chain with layout handoff

**What.** Two (or more) pipelined stages in one kernel, the producer's accumulator fragments handed to the consumer's operand layout through registers / LDS with a verified layout conversion.

**Code today.** `handoff.rs` (`ConvPlan` conversion planning `:15-115`), `contract_stage.rs` (`check_boundary` `:71`), `stage.rs` (`compose` / `merge` `:112 / :125`), `mlp/mod.rs` (`Y = fp8(relu(αXW1)) W2`, GEMM → GEMM with handoff), `docs/fusion-layout-handoff.md`.

**Users.** `mlp_fp8_*` only (plus probes `handoff_probe_*`).

**Proposed L2 API.** `contract_stage.rs` (boundary check) and `stage.rs` (`compose` / `merge`) move to `amdgpu-stage::{boundary, compose}`; `handoff.rs` and `contract.rs` stay in `amdgpu-tile` (`crate-split.md` D2, cheap to reverse). No template change in P1.

## 4. Cross-cutting design

### 4.1 Verification contract per template

Every template ships the protocol facts the verifier needs so kernels do not restate them:

| Template | Tile / buffer protocol | Counted waits | Contracts / trust entries today |
|---|---|---|---|
| T1 ring | `tile_produce` / `tile_consume` with `SlotIdx(slot, generation)` from `TileIt` | `vmcnt(n)` derived from `inflight_vm` (checked by the counted-`vmcnt` rule) | none in the template; per-kernel ranges |
| T2 ping-pong | same + equal barrier counts across groups (checked by the barrier-count rule) | `wait_vm(0)` before the first barrier | group role tags |
| T3 wave ring | `lds_private` + `PrivRef` markers (per-wave windows) | counted per tile | page-table bounds contracts |
| T4 merge | global `buf_produce` / `buf_consume` for the partial workspace; last-arriver release → ticket → acquire | fence + `wait_vm(0)` | `wg_winner` (a **trust entry**); candidates for a verified construction |
| T5 grid | `wg_bijection` + range facts | — | `assume_trusted` ×2 + `wg_bijection` in `block3_x_mode` |
| T9 select | `lds_tile` / `lds_ring` / `lds_owned` markers, versions per clear / scan step | `wait_lds(0)` before every publishing barrier | `tile_trust_periodic` removed (`origin/per-iter`) |

### 4.2 Policy objects

The compile policy is part of the template instance, not of the kernel body: `CompileProfile` (`Gemm`, `Vs128`, `Default`), `sched_opts` tokens, regional strategies (`declare_region`), and the Weaver region names.
Observed facts that the policy layer must express (all measured): GEMM uses `rewait,slotfill`-style tokens (`gemm/mod.rs:193`); FMHA v3 uses `Vs128` with `ldsink=1,wsink,ldahead32=1`; hd192 cannot use the whole-kernel reorder (peak live 277 > 256 VGPRs) and uses
`rewait` + regional `reorder` on `kv_stage` and `m_phase` (−3 … −13 %). Policy tokens remain target-specific (the list-scheduler knobs are gfx950 / `own` backend facts) and are validated by the L4 pass manager (`pass-catalog.md`).
No environment knobs in templates (repository rule): fixed choices plus archive tags for removed experiments.

### 4.3 Naming and crate placement

- `amdgpu-stage::ring` (T1, T2, T3), `::split` (T4), `::gridmap` (T5), `::scales` (T6), `::rowred` (T7), `::stream` (T8), `::select` (T9), `::handoff` + `::compose` (T11).
- L3 stays in `amdgpu-tile` (copies, DMA, softmax, tile ops, atoms) and `amdgpu-layout`.
- Kernels: `amdgpu-kernels` keeps per-family composition, per-shape orchestration and `registry.rs` (names → builder + params).
- The op-graph skeletons (`opgraph`, `fmha/skeleton_*.rs`) move to `amdgpu-graph` (generator) and stay generated; their `Slots` traits are the bodies of T1 / T2 / T4.

### 4.4 Extraction order (each step an md5-neutral move)

1. Move `Pipeline` / `Cadence` / `Roles` / `TileIt` to `amdgpu-stage::ring` unchanged; add `Params` structs as thin wrappers (no codegen change).
2. Move `components.rs` KV-split maths and `epilogue::{kvsplit_partial_f16, combine_splits}` to `amdgpu-stage::split`; leave the combine kernels in `amdgpu-kernels` first, then move them behind `CombineSpec`.
3. Extract `block3_x_mode` into `gridmap` with the same formulas and contracts (the `Target` supplies `xcds`).
4. Extract the histogram / scan / select blocks of `sampling` and `topk::perrow` into `select` **after** golden md5s are in the manifest (they are generated through closures with compile-time version counters; a mechanical lift keeps the order of emission).
5. Port the hand-rolled rings (bwd, MLA-hp, bf16 v1) only where the asm is byte-identical; otherwise leave them and record the difference.
6. Behaviour-changing refinements (ring depth 4 for the small i8 GEMM, scale-through-DMA for the new formats, owner-column verified construction) land **after** the moves, each with its own A/B and md5 re-baseline.

### 4.5 What the plan's rules mean for this catalogue

- *Per-shape orchestration stays in kernel code.* Examples that must **not** be pulled into L2 rules: the choice of XCD raster per kernel, the hd192 policy tokens, S (number of parts) per shape, the page-group lookup of one decode kernel, MoE padding rules, quantisation formulas.
- *No behaviour change in a move commit.* Hence §4.4 step ordering and the "unchanged first" rule for the hand-rolled kernels.
- *The md5 manifest is the oracle.* Templates are lifted with their emission order untouched (the gfx950 asm is order-sensitive: e.g. the position of the m-scale multiply moved timing by 40 %).

## 5. Open questions

1. **Ring cadence naming.** `Cadence::{RegStaged, OneWindow, TwoPhaseOffset}` are named after the kernel that introduced them. Rename in the move (breaking) or keep and add aliases?
2. **T3 as a cadence or a separate template?** The wave-private ring uses the same slot / generation machinery but has no barrier and per-wave windows; proposal: `Cadence::Barrierless` inside T1's driver.
3. **Where do the combine kernels live?** They are kernels (registry entries), but their layout is a property of the T4 partials. Proposal: in `amdgpu-stage::split` as constructors, registered by `amdgpu-kernels`.
4. **Verified construction for the XCD map and the last-arriver.** Both are trust entries today (`assume_trusted`, `wg_bijection`, `wg_winner`). Worth a verifier extension before or after extraction? (Backend decision.)
5. **Depth-4 ring and counted-wait derivation.** `inflight_vm` assumes the stage issues a fixed VMEM count; the MLA kernel's `18 vs 0` shows a per-tile expression. Express as a closure or a small table?
6. **gfx1250.** The templates parameterise counters and DMA form through `Target`, but T2 (two 4-wave groups) and T5 (XCD) are gfx950 shapes; the spec round (phase G) should state which cadences a wave32 / TDM target can instantiate.

## 6. Extraction status (kernel-opt, 2026-11-10)

All steps below are md5-neutral: `amdc md5-manifest --check tests/golden/md5_manifest.tsv` reports 234 kernels, 0 differences. The template code
is in `crates/amdgpu-tile/src/templates/` so that M7 (`amdgpu-stage`) can move the directory as a unit.

| Template | Done | Users ported | Notes |
|---|---|---|---|
| T1, hand-rolled 2-slot ring | `templates::pair_ring::PairRing`: the barrier-per-tile ring over tile pairs, with static slots 0 / 1, the clamped next-tile stage and the pair-counter exit | `fmha_bwd_dq*`, `fmha_bwd_dkdv*`, `mla_decode_bf16_hp_p*`, `fmha_fwd_hd*_bf16*` (v1) | The four loops were the same protocol with small variations, now parameters: `first` (the causal tile offset), `stage_region` (the dkdv `dkdv_stage` weave region) and `next_first`. `next_first` only reproduces the bf16 v1 emission order of the next-tile index; without it the two `fmha_fwd_hd128_bf16*` md5s change. The kernels keep their stage closures and loop variables; the template owns the slot / generation / wait / barrier / exit protocol. This is not the `Pipeline` driver: the hand-rolled cadence (stage after the barrier, one barrier per tile) is none of `RegStaged` / `OneWindow` / `TwoPhaseOffset`. Folding it into `Pipeline` as a fourth `Cadence` is a later, md5-neutral step |
| T3, wave-private double buffering | **not extracted** | — | The three "T3" loops do not share a protocol beyond the L3 private-region API (`LdsPlan::private`, `PrivRef::write` / `read_back`), which they already use:<br>- `decode.rs` uses one private buffer per tile and waits for its DMAs (`wait_vm(0)`) before computing, so it is not double-buffered;<br>- `decode16.rs` issues K, then V with a counted `wait_vm(nv)` inside the tile;<br>- `mla.rs` alternates two private buffers by block parity, with a count that depends on the block (`18` / `0`).<br>A shared `WaveRing` would be a closure wrapper with no protocol of its own (§0 test 1 fails). Revisit when a second kernel uses the `decode16` K / V counted form |
| T4 (a), separate merge kernel | `templates::split::{lse_combine, LseCombine}` (moved from `fmha/decode.rs`) and `templates::split::lse_combine_rows16_hd128` (moved from `fmha/bf16v3.rs`) | `attn_combine_bf16_hd128`, `attn_combine_bf16_sink_hd128` / `_sink_hd64`, `attn_combine_bf16_hd512` / `_hd512_sm` (via the `fmha/decode.rs` wrappers), `attn_combine_bf16_hd128_q16` | The kernel-crate names stay as thin wrappers (registry unchanged). The MXFP8 / fp8pt `combine8h` was already a template: `epilogue::combine_splits` via `io::combine_kernel`. The GEMM split-K reduce and `moe_sum` are sums, not LSE merges, and stay in `gemm/reduce.rs` |
| T4 (b) in-CTA LDS merge, (c) last-arriver | **not extracted** | — | Each has one user (`pa_decode16_fused`, `pa_decode16_lastc`), so §0 test 1 fails. The last-arriver protocol would also need closures at four emission points (counter index, counter bytes, merge, counter reset) to stay md5-identical. Extract when a second user appears, e.g. a last-arriver prefill split |
| T5 grid maps | `templates::grid`: `xcd_tile_head` (the FMHA (q tile, head) XCD order), `xcd_raster_16x16` (the GEMM 16 × 16 tile raster), `causal_pair` (+ `PairForm`), `split_pow2_z`, `tile_grid` + `tile_ids` (the LPT longest-first order), `wave_split_index` / `split_lg` / `key_chunk` (the decode per-wave key parts), and a re-export of `components::kv_split_decode` | FMHA forward (`block3_x_mode`: MXFP8 vs128 / kvsplit / t128, fp8pt non-causal); GEMM (`gemm_*` except MoE / conv); bf16 v3 causal pairing + `_ks` split index (hd128 and hd192); fp8pt causal pairing; bwd dq / dkdv (LPT); the KV-split decode of `io.rs` / `mod.rs`; paged decode (`pa_decode_*`, `pa_decode16_*`) and MLA decode key parts | Each map now emits its verifier facts with the maths (`wg_bijection`, `wg_union`, trusted id ranges), so a kernel never states them. The facts stayed identical except for one addition: **fp8pt causal pairing gained the `wg_union` fact** that bf16 v3 already had. That adds one trusted contract to `fmha_fwd_hd128_fp8pt_v3_causal` (31 contracts, strict 0 errors); codegen is unchanged. `PairForm` keeps the two kernels' forms of the second tile index: bf16 v3 `nt - 1 - x`; fp8pt `max(nt - 1 - x, 0)` re-typed as i32 |
| PairRing → `Pipeline` | **not folded** | — | Folding is md5-feasible but costs more than it gives. `Pipeline::run` owns the loop (`loop_state` / `bind_state` / peeled tail) through a `PipeBody`. The four pair-ring kernels own their loop variables, pin AGPRs between `begin_loop` and the first pass (dkdv, bf16 v1 hd256), and have no peeled tail (the last pass re-stages a clamped tile). Folding would either rewrite the four kernels as `PipeBody` impls, moving each stage closure's captured state into structs, or add a `Cadence` that `Pipeline::run` cannot run. `PairRing` stays the T1 template for the barrier-per-tile form. Revisit if `PipeBody` gains a kernel-owned-loop mode |
