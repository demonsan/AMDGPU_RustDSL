# W6: back-end Weaver features on the laggard kernels

Work item W6 of [`two-weavers-plan.md`](two-weavers-plan.md). The candidates are the families where we trail the vendor or have no reference: MLA, small-batch paged decode, attention backward, hd192 small grids, and the MXFP4 GEMMs. Every change is measured with [`tools/perf_ab.sh`](../tools/perf_ab.sh) ([`perf-ab.md`](perf-ab.md)), old = Main, same session. Only wins are adopted.

## 1. Lowering tokens (`movb64`, `jthread`, `cmpneg`) on every laggard family — neutral

Tried: the W5 tokens added to every kernel of the families (21 kernels change their md5). `perf_ab --families mla,decode,fmha_bwd,fmha_hd192`, release `amdc`, 5 reps, old = Main `972408ea`:

| family | cases | geomean new / old | verdict |
|---|--:|--:|---|
| decode | 6 | 1.000 | OK |
| fmha_bwd | 6 | 1.001 | OK |
| fmha_hd192 | 8 | 0.997 | OK |
| mla | 4 | 1.001 | OK |

No case moves beyond noise. **Not adopted.** These kernels have no rare-branch / multi-exit / zeroing hot spots for the tokens to change.

Vendor ratios from the same run show where the real gaps are:
- **hd192**, small grids: full 1×8×8×4096² 0.81×, causal 1×16×16×4096² 0.77×;
- **bwd causal**: we are 1.33× the vendor; bwd full ≈ 1.00×;
- decode / MLA: no vendor side in the harness.

## 2. hd192 dispatch rule variants — negative

Tried two rule changes:
- non-causal: key split S = 4 already at U ≤ CUs / 2 with ≥ 32 key tiles;
- causal: the key split S = 2 also at U = CUs with ≥ 32 key tiles.

`perf_ab --families fmha_hd192`, 7 reps:

| case | new / old |
|---|--:|
| full 1×8×8×4096² | **0.872 FAIL** |
| causal 1×16×16×4096² | **0.826 FAIL** |
| family geomean | 0.961 FAIL |

**Not adopted.** The current rule is right at these points, and the gap is in the kernels, not the choice.

## 3. Plan

The back-end Weaver orders **units**. decode16, MLA and the backward dq have none (no `Units` / Weaver capture), so back-end ordering cannot act on them yet. The default scheduler measured +3–4 % on decode16 and +7 % on MLA at some shapes, but breaks MLA correctness at B32H16C2048 / B4H16C16384; backend-llvm-tail is fixing that first. Next:
1. Express the decode16 and MLA bodies as Weaver units in today's emission order. This is md5-neutral: `Stmt::UnitMark` brackets do not change codegen.
2. Then a per-family scheduler / ordering policy, measured with perf_ab.

## 4. decode16 and MLA bodies as Weaver units (md5-neutral)

The loop bodies are now captured as `amdgpu_dsl::units::Units` and emitted by `amdgpu_stage::weave::Weaver::seq` with a plan label (no weave region), in today's order. The plan records the units, and the `UnitMark` brackets carry no code, so the md5 manifest is identical (235 / 0) and `cargo test` passes 552 / 0.

| kernel family | plan label | units |
|---|---|---|
| `pa_decode16_*` (all forms) | `decode16.loop` | `decode16.issue`, `wait_k` (anchor), `qk`, `softmax`, `wait_v` (anchor), `vconv`, `pv` |
| `mla_decode_bf16_p*` | `mla.loop` | `mla.issue`, `wait0` / `qk0`, `wait1` / `qk1` (anchors + reads / MFMAs per half), `softmax`, `pv` |
| `mla_decode_bf16_hp_p*` | `mla_hp.tile` (per tile of the `PairRing`) | `mla_hp.qk`, `softmax`, `pv` |

`amdc weave-report` now shows the plan units:
- `pa_decode16_bf16_hd128_p16_f` / `_c`: 6;
- `mla_decode_bf16_p16`: 7;
- `mla_decode_bf16_hp_p16`: 6.

Next: a per-family ordering / scheduler policy on these units, measured with perf_ab, once backend-llvm-tail's fix of the default scheduler's MLA miscompile is in.

## 5. hd192 small grids

The same-run vendor ratios put the largest gaps here:
- causal 1×16×16×4096²: 0.77×, run by the unpaired `_causal_np`, U = 256 = CUs;
- full 1×8×8×4096²: 0.79–0.81×, run by the key split `_ks` S = 2, U = 128.

**Dispatch alternatives, all measured negative.** `perf_ab --families fmha_hd192`, 7 reps:

| rule variant | the case it targets | new / old |
|---|---|--:|
| non-causal S = 4 at U ≤ CUs / 2 | full 1×8×8×4096² | 0.872 FAIL |
| causal ks2 at U = CUs | causal 1×16×16×4096² | 0.826 FAIL |
| causal `np_q128` up to U = CUs | causal 1×16×16×4096² | 0.884 FAIL |

So the choice among today's kernels is right, and the gap is inside the kernels. The untouched 4×16×16×2048² case moved by −2.1 % in the same run, which gives the harness's run-to-run noise on this family.

**What the back-end Weaver could act on.** `fmha_fwd_bf16_v3_d` (hd192 / hd128) marks three weave regions:
- `kv_stage`;
- `v_phase`;
- `m_phase`: `reorder` strategy on `m_phase` / `kv_stage`, and the `rewait` scheduler token for hd192.

It has no `Units`, so no plan units. The next md5-neutral step, as for decode16 / MLA (§4), is to capture the three regions' contents as units. That gives the back-end Weaver phases it can mark *Free*, letting it reorder across the region boundaries where the dependences allow. A structural difference against the vendor kernel needs an ISA comparison like W5's (`tools/isa_diff.py`, `isa_cfg.py`), on the vendor's `fwd_hd192_hd128_bf16[_causal]` code objects, before any kernel change is chosen.

**Step 1 done (md5-neutral).** The three phases of `fmha_fwd_bf16_v3_d` are Weaver units inside their existing weave regions:

| plan label | units |
|---|---|
| `bf16v3.kv_stage` | `kv.stage`, `kv.rope` |
| `bf16v3.v_phase` | `v.mask`, `v.max`, `v.rescale`, `v.exp` |
| `bf16v3.m_phase` | `m.pv`, `m.qk` |

- `weave-report fmha_fwd_hd192_hd128_bf16_v3` shows 66 plan units.
- The md5 manifest is identical (235 / 0).
- `compose_facts.rs` is regenerated (`tools/regen_compose_facts.sh`); `cargo test` 557 / 0.

**Step 2: ISA comparison with the vendor's `fwd_hd192_hd128_bf16`.** The aiter code object is disassembled at run time with ROCm 10.1 `llvm-objdump`; nothing vendor-derived is committed. Tools: `isa_cfg.py`, `isa_diff.py`. Group A's main loop:

| | ours `fmha_fwd_hd192_hd128_bf16_v3` | vendor |
|---|---:|---:|
| tiles per loop pass | 1 | **2** (exits after each tile, to one shared drain) |
| instructions per tile (without fillers) | 837 | 675 |
| `v_mfma_f32_32x32x16_bf16` / `ds_read_b64_tr_b16` / `ds_read_b128` per tile | 80 / 64 / 48 | 80 / 64 / 48 |
| VALU per tile | 575 | 423 |
| `s_waitcnt` / `s_nop` per tile | **83 / 59** | **5 / 3.5** |
| LDS-DMA issues per tile | 12 | 20 |
| barriers per tile | 2 | 4 |
| `s_setprio` | 2 | 0 |
| VGPRs / LDS | 252 / 160 KiB | — |

What the comparison shows:
1. **Same matrix work.** The MFMA and LDS-read counts per tile are identical, so the gap is in the order and the VALU.
2. **The vendor interleaves its softmax VALU with the MFMA stream in each wave.** The loop opens with `v_exp_f32` × 63 woven between 45 MFMAs and 32 V reads. Our kernel separates a VALU phase and an M phase per wave group, a ping-pong of two groups with `s_setprio`. The vendor waits 5 times per tile; we wait 83 times and pad 59 nops, because every phase drains its loads before the next.
3. **The vendor runs two tiles per pass** (the same shape as the W5 replica's step 2) and spends no instructions on our per-tile masking address arithmetic: 111 `v_add_u32` and 64 `v_cmp_ge_i32` against its 64 `v_cmp_lt_i32` per tile.

Classes:
- (2) is a **back-end Weaver** capability: interleaving two phases' units in one wave (Free phases with interleave hints). It also depends on the **front end** choosing a single-group, in-wave pipelined plan for this family, which is a different composition from today's ping-pong.
- (3) is **expressible now**: `Pipeline::unroll2` plus a cheaper mask (hoisting the key index arithmetic).

These are the next measurable steps:
- `unroll2` on the bf16 v3 pipeline;
- the mask arithmetic;
- then, with W3's interleaving, the in-wave softmax / MFMA weave.

**Step 3: two tiles per pass on bf16 v3 (`Pipeline::unroll2` + `jthread`) — neutral, not adopted.** `perf_ab --families fmha_hd192,fmha_bf16`, 7 reps:
- hd192 geomean 0.998;
- bf16 full 0.992 (OK).

The **bf16 causal cases did not run**: the paired-causal kernel with `unroll2` produced no result. The cause was not investigated, since the form gives nothing on the shapes that ran. In the W5 replica the two-tile pass only paid together with the other changes. Here the stall is the per-phase drain (83 waits per tile), which unrolling does not remove.

### 5.1 Request to the back-end Weaver (W3 B4): in-wave softmax / MFMA interleave for bf16 v3

**Target.** `fmha_fwd_bf16_v3_d` (hd192 and hd128), group A's loop. The plan units already exist:
- `bf16v3.v_phase`: `v.mask`, `v.max`, `v.rescale`, `v.exp`;
- `bf16v3.m_phase`: `m.pv`, `m.qk`.

**What the vendor does** (§5, step 2): the softmax of tile t (`v.exp`: 64 `v_exp_f32` + 64 `v_fma_f32` + the row sum) runs **between** the MFMAs that do not depend on it. The loop opens with 63 `v_exp` woven into 45 MFMAs and 32 V reads. Result: 5 `s_waitcnt` and 3.5 `s_nop` per tile, against our 83 and 59.

**Dependences that allow it.**
- `m.qk` of tile t + 1 (S(t+1) = Q K(t+1)ᵀ) does not depend on `v.exp(t)`, so the two can interleave.
- `m.pv(t)` needs P(t), i.e. the end of `v.exp(t)` + the bf16 packing.
- `v.max(t)` / `v.rescale(t)` need S(t), complete after `m.qk(t)`.

**Asks.**
1. A **Free phase** spanning `v.exp(t)` and `m.qk(t+1)` within one wave, so the back end may order across today's phase boundary. That boundary is a front-end choice (`v_phase` / `m_phase` regions + the ping-pong barrier).
2. An **interleave hint**: spread `v.exp(t)` over `m.qk(t+1)`'s MFMAs, about 4 VALU per MFMA. That keeps the VALU latency (trans ops: `v_exp` 4 passes) under the MFMA issue (16 passes each).
3. Waits placed per unit dependence (counted `lgkmcnt` for the K reads of `m.qk`), not a drain at the region end.

**Front-end part (ours, after B4 lands).** The plan must allow it. Either:
- **(a)** keep the two-group ping-pong but move `m.qk(t+1)` into the softmax phase of the same group; or
- **(b)** a single-group in-wave pipeline, as the vendor runs (no `s_setprio`, 4 barriers per tile).

We would measure both with perf_ab on `fmha_hd192` / `fmha_bf16`.

**Expected.**
- Waits per tile 83 → about 10, nops 59 → under 10. The vendor's numbers are 5 / 3.5.
- The VALU phase (423–575 instructions per tile) mostly hidden behind the 80 MFMAs per tile.
- This is the gap to the vendor on the small grids: 0.77–0.82×.

## 6. Scheduler policy on MLA and decode16 (adopted)

The list scheduler with reordering is now applied per kernel through `sched_opts = "reorder"`:
- the MLA kernels (`mla_decode_bf16_p*`, `_hp_p*`) and their combine (`attn_combine_bf16_hd512`);
- the last-arriver form of decode16 (`pa_decode16_*_c`).

It relies on backend-llvm-tail's SCC fix (`origin/sccfix`): the descriptor pseudos wrote SCC without declaring it, so the scheduler could separate an `s_cmp` from its `s_cselect` (the earlier MLA miscompile). perf_ab, release `amdc`, 7 reps, same session:

| family | change | geomean new / old | cases |
|---|---|--:|---|
| mla | scheduler on (kernels + combine) | **1.138** | B1H128C8192 1.069, B32H16C2048 **1.193**, B4H16C16384 **1.227**, B8H128C4096 1.071 (all IMPROVED) |
| decode | scheduler on the `_c` form only | **1.015** | `_c`: B1H32HK4C32768 1.029 / B4H16HK1C16384 1.043 IMPROVED, B1H64HK8C8192 1.017, B8H64HK8C4096 0.992; `_f`: 1.004 / 1.003 |
| decode | scheduler on every form (rejected) | 1.009 | the fused `_f` form lost 0.4–1.6 % |

The correctness markers of the harness hosts (`MLA_OK`, `PA_OK`) held in every run.

### 5.2 B4 on bf16 v3: PV(t) woven into the softmax of tile t — measured negative

**Correction to §5.1.** The vendor's loop head weaves its exps between **PV** MFMAs. The 32 `ds_read_b64_tr_b16` among them are V reads, so it overlaps exp(t) with PV(t), not with QK(t + 1). In our pipeline that needs no ring change: V(t) is published one iteration earlier.

**Change** (`V3Opts::pv_early`, option kept, default off):
- PV(t) moves from the matrix phase to the end of the softmax phase;
- `{v.exp, m.pv}` becomes one Free plan phase (`Weaver::free`) with `interleave(m.pv, [v.exp])`;
- the matrix phase keeps QK(t + 1); the barriers stay outside as anchors.

The B4 merge did weave the exps between the PV MFMAs in the emitted loop, and strict verify passes.

**perf_ab** (old = Main `f42c2ef5`, 7 reps):

| variant | `fmha_bf16` geomean | `fmha_hd192` geomean |
|---|--:|--:|
| `pv_early` | 0.918 FAIL (causal 0.83–0.92, full 0.95–0.98) | 0.933 FAIL |
| `pv_early` without `s_setprio` | 0.916 FAIL | 0.932 FAIL |

**Not adopted.**

**Cause.** In our two-group ping-pong, moving PV into the softmax phase takes the 32 PV MFMAs out of the matrix phase. The group in its softmax phase then competes for the matrix pipe with the other group's matrix phase, which now holds only QK. The phases also stop alternating cleanly: one is long and mixed, the other short. The vendor's in-wave overlap belongs to a different composition (no priority ping-pong, 4 barriers per tile, two tiles per pass). Reproducing it is a front-end composition change: a single-group or re-balanced phase plan, a W2 / front-end Weaver decision, not a B4 placement. Until such a plan exists, the B4 merge has no profitable Free phase in this kernel.

### 5.3 Composition experiment: a single-group in-wave pipeline (time-boxed prototype) — negative

**Plan** (`V3Opts::single`, option kept, default off).
- **Composition:** all 8 waves form one group. There is no phase offset and no priority ping-pong, and every wave stages its share of K and V. The cadence is `Cadence::OneWindow`: tiles are staged two windows ahead in 3-slot rings, and the previous window's `wait_vm(0)` drains them, so tile t+1 is published when window t starts.
- **Tiles:** 64 keys, so the three slots fit (96 KiB of LDS for hd128).
- **Per window t:** the softmax head (mask, max, rescale) runs authored. Then one Free plan phase `{exp(t), QK(t + 1), PV(t)}` with the exps interleaved into the QK MFMAs (`Body::window`).

**perf_ab** (old = `bf16-b4` `f80c2ef2`, 5 reps; correctness markers held):
- **fmha_bf16:** geomean 0.911. hd128 non-causal 0.85–0.90; causal 0.89–1.02.
- **fmha_hd192:** geomean 0.950, but only the unpaired causal form uses the prototype: causal 1×16×16×4096² 0.80, 1×8×8 0.90. The other hd192 forms are unchanged.

**Not adopted.**

**Strict verify:** 2–4 `GV-LDS-OOB` errors. The interval analysis cannot bound the LDS-DMA base of a select-cycled 3-slot ring slot. This is a proof gap, not a measured fault: the hosts' result checks pass.

**Cause** (as far as one prototype shows).
- The ping-pong hides each group's softmax behind the other group's matrix phase.
- In one group, only the exps can overlap the MFMAs. The softmax head (masking, row max with its cross-lane swap, the lazy rescale branch) stays serial.
- The 64-key tile halves the matrix work per window, while the barrier and the drain per window stay.

The vendor's in-wave kernel overlaps more of the head (two tiles per pass, 128-key tiles in a 2-slot K / 3-slot V ring). Getting there needs:
- the B5 ring (K2 / V3, distance 2) with 128-key tiles, which LDS allows at hd128 (2 × 32 + 3 × 32 = 160 KiB, at the limit), not the 3 × 3 slots used here;
- a head that overlaps: the row max of tile t+1 woven into PV(t).

This is the replica's step 2 territory. The composition choice is recorded as a measured negative at 64-key tiles.
