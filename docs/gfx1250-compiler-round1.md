# gfx1250 compiler optimisations, round 1

The first round of back-end work on the gfx1250 (MI450) target, driven by instruction-level comparisons of our generated loops with a reference
implementation (the comparison method is [`gfx1250-replica.md`](gfx1250-replica.md); this note records only what we measured on our own code).
Every row says what changed, the measured effect, whether it is **kept** (default), **opt-in** or **rejected**, and where the evidence is.
Numbers are from the MI450 nodes a08-1 / e08-01 (compared only within one node); "not measured separately" means the change went in with
others and the repo has no isolated A/B for it.

## 1. What the first comparison found

`tools/isa_loop_profile.py` counts instruction classes per hot loop (counts only). The same MXFP4 GEMM loop body, before the round and now:

| per WMMA in the K loop | first stage-template kernel (128 x 64 tile, round start) | `gemm_g1250_tdm_stage_f4_256x256_ks4_s2_epit_roles` (now) |
|---|---|---|
| instructions (all classes) | 57 | **3.7** (474 per 128 WMMA) |
| VALU | 20 (16 of them `v_mov_b32` accumulator copies) | 0.03 |
| SALU (descriptor / ring-slot / address recomputation) | 23.8 | 0.47 |
| LDS reads | 4 | 1.38 |
| hazard `v_nop` | 4 | 0 |
| memory waits | 1.5 (load all, wait, compute) | 0.28 (counted waits) |
| `s_set_vgpr_msb` | 0 (at most 256 VGPRs) | 0.48 (1,024 VGPRs through MSB windows) |

The rows below are the compiler changes behind that table, plus the correctness fixes that the same work shook out.

## 2. Code generation

| item | change | measured effect | state | evidence |
|---|---|---|---|---|
| In-place WMMA accumulation | the loop-carried accumulator is the WMMA's D = C register (the phi copy is coalesced), also for per-lane accumulator scaling | removes 8 `v_mov_b32` per WMMA and the 4 `v_nop` that waited for the copy's read: the loop has no VALU; MLA `in-place O rescale and accumulation` -3 % (about 155 `v_mov` per page) | kept | [`gfx1250-status.md`](gfx1250-status.md) §1 and §5; probes `probe_g1250_varscale{4,32}` (`83ed01b0`); loop profile above |
| Value numbering of address arithmetic (`cse.rs`) | repeated slot bases, lane sums, `x + 0` copies merged; descriptor words written by their producers | GEMM loop **147 -> 101 instructions** (`8a1bed0a`); later versions fixed two miscompiles it caused (two materialised VCC booleans merged: `96cf0c5f`; a copy renamed past a loop variable's in-place write: `6e0028c3`, rename invariant asserted on every compile) | kept | `crates/amdgpu-codegen/src/target/gfx1250/cse.rs`, regression tests `12c1368c`, `96cf0c5f` |
| Kernarg preload | up to 16 kernarg dwords arrive in user SGPRs (no `s_load` + wait in the prologue); the start sequence provably never clobbers the ABI inputs; recompile without the preload when out of SGPRs | removes 7 `s_load` and their wait from the kernel start; measured by backend: flat on the MXFP4 GEMM, **+7 % on the add bandwidth probe** (2.82 -> 3.02 TB/s) | kept | `3978c7cc` (the start sequence zeroed preloaded dwords 4 - 5: the hang of 2026-11-18), `90094339` (compile-time ABI-input survival check, mutation test), `1617e9d9` (fallback) |
| Accumulator zeroing in persistent loops | `FragZero` + `zeroinit`: no zero tuple kept live across the tile loop | makes the 256 x 256 persistent tile loop fit its registers; not measured separately | kept | `a2ff7b83` |
| Register allocation for 1,024 VGPRs | two-phase allocation (low-window values first, then tuples, then single dwords best-fit by start) as the last resort after the scan policies; a low-window class for WMMA scale operands; `s_set_vgpr_msb` windows | the 256 x 256 kernels compile at all (accumulators alone are 512 VGPRs); MSB toggling costs 0.48 instructions per WMMA | kept | `12f6650c`, `fb8066be`, `290d3a72`; `tools/mi450/ra_vgprs.sh` |
| Accumulators in the high window (`AMDC_ACC_HIGH`) | a second allocation with the matrix accumulators above v255 | no gain on the MLA and the GEMMs | **opt-in** (archived) | `69a110f2`, `0aac8b5e` |
| WMMA operand reuse hints, `matorder`, `AMDC_ACCWIN` | scheduling / ordering experiments | neutral or worse | **rejected** (opt-in switches only) | [`gfx1250-status.md`](gfx1250-status.md) §1 "Code generation" |

## 3. Waits and hazards

| item | change | measured effect | state | evidence |
|---|---|---|---|---|
| Wait immediate range per counter | the shared wait pass had used the previous architecture's LDS counter range, so LDS loads were treated as complete too early; every counter now has its width (63; KMcnt 31), checked in lowering, emitter, encoder, by G5 `GV-1250-WAIT-RANGE` and by an independent re-check of the final waits | a correctness fix: it removed the wrong results of a GEMM sweep (and a dropped wait is now a compile error); no performance change | kept | [`gfx1250-status.md`](gfx1250-status.md) §6 incident 1, [`gfx1250-hang-rules.md`](gfx1250-hang-rules.md), `9b7cc392`, `f2b57553` |
| Split LOADcnt / STOREcnt waits | a vector-memory op is never assumed complete by the shared saturation; operations beyond the known queue are not counted as the waited class; the emitted wait is clamped | correctness (a wait could be too weak); no performance change | kept | `d083cd94`, `26e70eb6` |
| Counted tensor / DS waits in loops | waits with N > 0 right before the consumer (TENSORcnt in order; the relaxed wait leaves `slots - 2` stages in flight) instead of load-all-wait-0 | 1.5 -> 0.28 waits per WMMA; the relaxed tensor wait is worth +10 % at 8192^3 on 3-slot rings (8.4 -> 9.3 PF/s) | kept (`relax` kernels) | [`mxfp4-gemm-gfx1250-plan.md`](mxfp4-gemm-gfx1250-plan.md) |
| TRANS co-execution `v_nop` | LLVM pads every VALU that reads a TRANS result (or overwrites a TRANS source) with one `v_nop`; on MI450 the pad is removed for the reader classes proven bit-identical by probes (`v_add / v_sub / v_fma / v_max / v_mul_f32`, `v_pk_add_f32`, `v_cvt_pk_fp8_f32`: 0 mismatches over 5 runs x 1,024 workgroups x 512 work-items, every TRANS op) | removes the pad at those sites (MLA softmax: 66 -> 2 `v_nop` per pass together with the batched-exponent restructure `f68de235`) | **kept for the proven classes only**; every other reader keeps the `v_nop` (relaxing unproven classes: rejected) | [`gfx1250-hazards.md`](gfx1250-hazards.md) "TRANS co-execution: hardware evidence", `tools/mi450/trans_all.sh`, `tools/mi450/trans_ab.sh`, `AMDC_TRANS_PROVEN_OLD` for A/B |
| WMMA co-execution windows | the `v_nop` count per WMMA kind (the windows of LLVM's co-execution model); no `v_nop` remains in the GEMM loop because consecutive WMMAs are independent | 4 -> 0 `v_nop` per WMMA together with the in-place accumulation | kept | [`gfx1250-hazards.md`](gfx1250-hazards.md), `target/gfx1250/hazards.rs` |

## 4. Scheduling

| item | change | measured effect | state | evidence |
|---|---|---|---|---|
| gfx1250 machine model | list-scheduler latencies from the public LLVM speed model for gfx1250 (`AMDC_SCHED_MODEL950` selects the old model for A/B) | no gain on its own: the K loop's order is fixed by the template's software pipeline (fragment reads of sub-step s + 1 issued before the WMMAs of s); reordering with it is the -28 .. -33 % case of the next row | kept | `5e3607ca` |
| Scheduler preset `SchedPreset::Gemm` (scoreboard waits + DS-read hoisting, no instruction reordering) | adopted for the wave-split MLA (`_wsg`) | **+2 - 5 %**, paired on e08 (backend g25, kernel-opt `_wsg`); list-scheduler reordering (either machine model) **-28 .. -33 %** | preset opt-in per kernel (kept for the MLA); reordering rejected | `docs/lgkm-lookahead.md` (the preset), [`gfx1250-status.md`](gfx1250-status.md) §5 |
| Scheduling mode | our code runs in mode 0; mode 2 without the extra `s_wait_alu` waits is wrong on hardware (measured) | none (a restriction) | kept | `tools/mi450/mode2.sh`, [`gfx1250-status.md`](gfx1250-status.md) §1 |

## 5. Data movement (kernel templates, found by the same comparisons)

| item | change | measured effect (MXFP4 GEMM, 8192^3 unless noted) | state | evidence |
|---|---|---|---|---|
| TDM pad planner | the LDS row padding of a tensor load is planned from the declared read patterns of the window (`amdgpu_dsl::tdm_pad`): a 256-byte row stride put 8 lanes of a `ds_load_b128` in one bank | **3.3 -> 6.7 PF/s** (GEMM), **2.5 x** (MLA, 16-byte pad after every 512 B) | kept | `crates/amdgpu-dsl/src/tdm_pad.rs`; [`gfx1250-status.md`](gfx1250-status.md) §5 |
| LDS-staged C epilogue | the accumulators go through LDS (f32, unpadded rows) and leave in one `tensor_store_from_lds` per wave instead of 64 `global_store_b32` per wave; a padded store descriptor is a compile error (G5 `GV-1250-TDM-STORE-PAD`) | **6.7 -> 8.1 PF/s** | kept (default) | `0943801d`, `77f70ef7`; `RingOpts::epilogue_lds` |
| Wave roles | waves 0 / 1 issue A and its scales, waves 2 / 3 B and its scales (two loop copies by wave): 2 tensor loads per wave per stage instead of 4 | roles alone +3 %; the relaxed wait on top 8.4 -> 9.3 PF/s | kept (`_roles` kernels); needs a trusted trip-match fact for the cluster barrier inside a cluster | `db4306f1`, `f8d4d7d0` |
| Ring depth | `Pipeline::slots` (2 .. 4): deeper ring, or shallower with larger K slots | 4 slots: 0 (GEMM and MLA: not load bound); **2 slots with 512-K slots: +16 .. +19 % over the 256-K roles kernel** (8.4 -> 9.99 PF/s at 8192^3, 6.3 -> 7.3 at 4096^3; +8 % over its relaxed-wait variant: one barrier and one stage per 512 K) | 2-slot 512-K kernel is the default; 4-slot opt-in | `fb8ccfdb`, `15f0ff10`; [`mxfp4-gemm-gfx1250-plan.md`](mxfp4-gemm-gfx1250-plan.md) |
| Cluster multicast | a tensor load multicast to a 2 x 2 / 4 x 4 cluster with a lockstep cluster barrier | 256-K kernels: -7 .. -22 %; 512-K slots: 8192^3 equal, 4096^3 +8 % | opt-in (`_c4x4_lock`) | [`mxfp4-gemm-gfx1250-plan.md`](mxfp4-gemm-gfx1250-plan.md), `75143425` |
| Preshuffled scale layout | 32x fewer scale TDM rows | 0 | opt-in | `55fd0ccb` |
| Persistent grid, tile-order swizzle | tile loop inside the kernel; remapped tile order for L2 sharing | +0.5 % at best (the LDS epilogue leaves no room to overlap tiles) | opt-in | `30a6998b`, `462f0bd5` |

Kernel-level progression of the MXFP4 GEMM (8192^3, PF/s): 0.98 (first stage-template kernel) -> 1.37 (cooperative loads) -> 2.44 (two K sub-steps per slot) -> 3.3
(256 x 256 tile) -> 6.7 (TDM padding) -> 8.1 (staged epilogue) -> 9.3 (roles + relaxed wait) -> **10.0** (512-K slots, 2-slot ring).

## 6. Measuring and verifying (tools that came out of the round)

- **Per-K-step stamps** (`RingOpts::stamps`, host `STAMPS=1`, `tools/mi450/f4stamps.sh`): the shader-cycle counter at four points of every steady iteration (stage issue, window,
  wait, barrier), summed in loop variables, plus prologue / loop / epilogue per wave; `s_get_shader_cycles_u64` is in the encoder tables. It showed that the K loop is at the
  reference cycle count per WMMA and that the remaining gap is outside it.
- **Deletion profiling** (`tools/mi450/f4prof.sh`): removes the tensor loads, the WMMAs, the DS reads or the barriers from the assembly (hang-scanned first) to attribute time.
- **Loop instruction profile** (`tools/isa_loop_profile.py`), **paired alternating runs** with the reference on one GPU (`tools/mi450/f4vs256.sh`).
- **Verifier extensions that the new kernels needed**: `GV-LDS-ASYNC-OVERLAP`, prefetch bounds, `GV-1250-GETREG-IDLE`, a persistent tile loop whose body is simulated once and must end
  quiescent (see [`gfx1250-hang-rules.md`](gfx1250-hang-rules.md)).

## 7. Open after round 1

- The per-tile prologue (7 %) and staged epilogue (14 - 15 %) of the GEMM: the next tile's stage loads must be issued before the epilogue, which needs a staging buffer outside the
  ring and a protocol rule comparing the loop back-edge state with the pre-loop prologue state ([`mxfp4-gemm-gfx1250-plan.md`](mxfp4-gemm-gfx1250-plan.md), last section).
- VGPR window churn (`s_set_vgpr_msb` 0.48 per WMMA in the GEMM, about 3 per WMMA in the wave-split MLA): window-aware placement of accumulators versus streamed operands.
- Single-wave latency hiding in the MLA softmax (instruction-level software pipelining under the previous page's WMMAs).
