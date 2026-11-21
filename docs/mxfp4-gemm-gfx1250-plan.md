# MXFP4 GEMM on gfx1250 (MI450): status and the plan toward the vendor structure

Kernel: `gemm_g1250_tdm_stage_f4*` in `crates/amdgpu-kernels/src/probes/g1250_gemm.rs`, body `amdgpu_stage::gemm::tdm_f4::TdmF4Body` driven by `Pipeline` with `Cadence::OneWindow` (3-slot ring, producer at iteration offset 2). Host `tests/gpu/g1250_f4own_host.cpp` (CPU reference on sampled rows, polled waits, `ONLY_K` bisect aid), payloads `tools/mi450/f4hw.sh`, `f4vs.sh`, `f4bisect.sh`, `f4variants.sh`.

## Where we are (one GPU, f32 output, natural row-major A / B / scales; all bit-exact against the CPU reference)

| kernel (tile, waves) | declared LDS | VGPRs | 4096^3 | 8192^3 (TFLOP/s) | note |
|---|---:|---:|---:|---:|---|
| `gemm_g1250_tdm_stage_f4` (128 x 64, every wave loads the whole tile) | 39 KiB | 160 | - | 0.98 | first version |
| the same, cooperative loads (a slice per wave, one barrier publishes) | 39 KiB | 160 | 0.99 | 1.37 | step 1 |
| `..._128x128` (wave tile 2 x 4 WMMAs) | 51 KiB | 240 | 1.55 | 1.70 | step 2 |
| `..._256x64`, `..._8w_256x64` (8 waves) | 64 / 85 KiB | 256 | 1.16 / 1.06 | 1.66 / 1.60 | |
| `..._256x128` (wave tile 4 x 4, 757 `s_set_vgpr_msb`) | 76 KiB | 640 | 1.58 | 1.94 | needed the scale-operand register class |
| `..._128x128_ks2` (two K sub-steps per slot) | 102 KiB | 304 | 1.71 | **2.44** | needed the DScnt wait-range fix |
| `..._128x128_ks2_relax` (`wait_tensor(4)`: two windows of load latency) | 102 KiB | 304 | 1.73 | 2.44 | no gain: not latency bound |
| `..._256x128_ks2` | 153 KiB | 736 | 1.86 | 2.22 | |
| `..._128x128_ks4` | 204 KiB | 544 | 1.04 | 1.16 | register pressure / window flips |
| vendor `f4gemm` 256 x 256, 4 x 4 cluster, persistent, bf16 output | 320 KiB | 1024 | 9.0 - 9.2 | 10 - 11.4 | the target |

The gap is about 4x. The replica comparison (kernel-opt, `docs/gfx1250-replica.md` on the replica branch) puts the instruction overhead at 57 instructions per WMMA against the vendor's 4.3 (accumulator copies and their `v_nop`s, about 95 SALU per K step for descriptors and ring slots, 4 LDS reads per WMMA against 1.5), and the operand traffic at about 9.5 TB/s of L2 -> LDS for the 128 x 128 tile: both limit us, which is why a relaxed tensor wait changed nothing.

## What the vendor kernel does (read from the public code object and kernel-opt's exact lift)

* 128 threads = 4 waves; workgroup tile 256 x 256; each wave owns 128 x 128 of C = 4 x 8 scaled WMMAs (32 x 16 x 128) per K step of 128, accumulators 512 VGPRs of a 1024-VGPR wave.
* The window is software-pipelined by groups of 8 WMMAs: the fragment loads of the next group (16 `ds_load_b128` + the scale words) are issued between the WMMAs of the current one; scales go through `v_wmma_ld_scale_paired_b32`.
* One **split barrier** per K tile (`s_barrier_signal -1` early, `s_barrier_wait` about 25 instructions later) and one `s_wait_tensorcnt 2` (two tiles in flight); no full drain per tile.
* The TDM issue is split across waves by wave id (two wave groups run two code paths), the loads multicast inside a 4 x 4 cluster (16 workgroups share A / B tiles; one wave per workgroup signals the cluster barrier per K tile).
* Persistent grid of 256 workgroups, tile scheduler in scalar code; up to 5 stages of ring in 288 KiB of LDS.
* B and the scales are pre-shuffled so a lane's fragment is one contiguous load; output converted to bf16 in the epilogue.

## Steps toward it (each measured with `f4vs.sh` on the same GPU, bit-exactness checked every time)

1. **Cooperative loads** (done): `ReadFirstLane`-uniform wave slice, one barrier. Needed no new compiler feature.
2. **Bigger CTA tile / wave tile** (done up to 128 x 128): more WMMAs per barrier and per fragment load. The 256 x 128 tile (`..._256x128`) needed the back end's low-window register class for the scale operands of `v_wmma_scale_*` (accumulators above v255); it now compiles (640 VGPRs) but flips the VGPR MSBs before almost every WMMA (757 `s_set_vgpr_msb`): *compiler feature 1 (done) and 1b: window-aware placement of operands and accumulators.*
3. **Software-pipelined fragments** (next): window(t) issues the fragment loads of tile t + 1 (the body reads `t.next`, already complete because the pipeline completes tile t + 1 one window earlier) and loop-carries the fragment registers; K sub-steps per slot (`ks`) give more WMMAs per barrier without more registers. Blocked today by the `ks = 2` wrong-result case (bisect in progress, see `tools/mi450/f4bisect.sh`). *Compiler feature 2: schedule ds loads of the next group ahead of the WMMAs of the current group (B5 Free phase).*
4. **Relax the waits**: `stage_complete` is a `wait_tensor(0)` at the end of each window; with the producer at iteration offset 2 the ring allows `wait_tensor(2)`. *Compiler feature 3: the auto `s_wait_tensorcnt` from the ring slot tokens (B5) replacing the conservative IR wait.*
5. **Split barrier**: signal after the last read of a slot, wait right before the slot's reuse (`barrier_signal` / `barrier_wait`), instead of the full barrier at the end of the window. Verified by G5 (alternation, named barriers).
6. **bf16 output** (`v_cvt_pk_bf16_f32`, packed stores) and a **scale preload instruction** (`v_wmma_ld_scale_paired_b32`) in the atoms.
7. **Pre-shuffled B / scales** (the host layout of aiter's `shuffle.py`): contiguous per-lane fragment loads, fewer `ds_load` instructions per WMMA.
8. **Cluster multicast + persistent grid**: cluster launch (4 x 4), `tensor_load_to_lds` with the workgroup mask, one wave per workgroup signalling the cluster barrier (the G5 BAR-COUNT proof covers exactly this shape), a scalar tile scheduler. Needs the host to launch with the cluster attribute.
9. **Deeper ring** (4 - 5 stages) once LDS above 64 KiB is allowed (cleared by the P1 / P2 micro-probes).

## Config list for the next hardware runs (declared LDS)

Only after the ks = 2 wrong-result case is understood; one idle GPU, hard timeouts, smallest shapes first, 1 work-group before the full grid:

| name | tile / waves | declared LDS |
|---|---|---:|
| `gemm_g1250_tdm_stage_f4` | 128 x 64, 4 | 39,168 |
| `gemm_g1250_tdm_stage_f4_128x128` | 128 x 128, 4 | 52,224 |
| `gemm_g1250_tdm_stage_f4_256x64` | 256 x 64, 4 | 65,280 |
| `gemm_g1250_tdm_stage_f4_ks2`, `_ks2_grouped` (bisect only) | 128 x 64, 4, ks = 2 | 78,336 / 103,936 |

Not to be run: the 8-wave variant that hung (not registered). `..._256x128` (78,336 B) only after the ks = 2 / dscnt verification.

## Best kernels and the relaxed-wait dispatch (a08-1, bit-exact, f32 output)

`gemm_g1250_tdm_stage_f4_256x256_ks2_epit_roles_relax` (256 x 256 tile, ks 2, TDM row pad, LDS-staged TDM-store epilogue, wave roles A / B, relaxed tensor wait) reaches 9.3 PF/s at 8192^3 and 10.0 PF/s at 8192 x 8192 x 16384; `..._epit_roles` (the same without the relaxed wait) is the better choice for K in about 3840 - 4608. Measured, PF/s (M = N = 4096 unless noted):

| K | `epit_roles` | `epit_roles_relax` |
|---:|---:|---:|
| 2048 | 4.50 | 5.02 |
| 3328 | 5.76 | 6.24 |
| 3584 | 6.02 | 6.54 |
| 3840 | 5.52 | 4.65 |
| 4096 | 6.25 | 5.30 |
| 4352 | 5.00 | 4.61 |
| 4608 | 5.28 | 5.06 |
| 5120 | 6.01 | 6.22 |
| 6144 | 5.84 | 6.25 |
| 8192 | 7.79 | 8.54 |
| 8192 (M = N = 8192) | 8.47 | 9.28 |
| 16384 (M = N = 8192) | 8.27 | 10.06 |

The dependence on K is not monotonic and is not explained by a power-of-two row stride (K = 3840 and 4352 lose as well) nor by tile order: a tile-order swizzle (`RingOpts::swizzle`, host `SWIZZLE=G`, G = 4, 8) changes nothing measurable. Cluster multicast on the same kernel (`_epit_c2x2_lock`, `_epit_c4x4_lock`) is slower (6.5 / 7.4 PF/s at 8192^3 against 8.3). Dispatch rule until the K dependence is understood: relaxed for K <= 3584 or K >= 5120, not relaxed for 3840 <= K <= 4608.

## 512-K slots in a 2-slot ring (e08 round, supersedes the relaxed-wait dispatch above)

`gemm_g1250_tdm_stage_f4_256x256_ks4_s2_epit_roles`: ks 4 (one stage = 512 K), ring of 2 slots, one barrier and one stage per 512 K; 9.99 - 10.04 PF/s at 8192^3 and 7.1 - 7.35 at 4096^3 against the vendor's 12.4 - 12.5 / 11.0 (paired, e08). No K-dependent resonance (the relaxed-wait dispatch table above applies only to the 256-K kernels). Host contract: K a multiple of 512. Multicast composes with it (`_roles_c4x4_lock`: 7.9 at 4096^3, equal at 8192^3). Stamps and the profile that led here: `docs/gfx1250-status.md` section 4.

## What did not close the last 20 % (e08 round, 2026-11-19)

Measured on e08-01 against the 512-K kernel above, all bit-exact; the details of the stamps are in `docs/gfx1250-status.md`.

| attempt | result (8192^3 unless noted) | why |
|---|---|---|
| cluster multicast with 512-K slots (`_roles_c4x4_lock`) | 4096^3 7.9 (+8 %), 8192^3 9.99 (equal) | the cluster barrier is amortised over 512 K; traffic is no longer the limiter at 8192^3 |
| 256 x 128 tile (512 workgroups at 4096^3) | slower at every shape except 2048^3 | smaller tiles raise the per-K-step overhead and the epilogue share |
| lazy prologue, instruction prefetch | 0 to +2 % / 0 | the prologue is one DRAM round trip; the loop's code is already in the instruction cache |
| tile staggering (odd tiles sleep at the start) | 4096^3 +5 %, 8192^3 -2 % | desynchronises the write bursts, costs the sleep |
| wave order (Gray / transposed) | 0 | the 4 % window skew follows the wave id, not the block position |
| operand swap + bf16 epilogue (direct 16-byte stores / conflict-free staged chunks) | 8.3 / 8.6 - 8.8 | the epilogue is a fixed latency (about 7,000 cycles) whatever the width; the row permutation makes the window 12 % slower |
| epilogue stores overlapped into the next tile's K loop (packed bf16 `prev`, 8 - 16 blocks) | 7.8 - 8.4 | the global stores contend with the TDM loads (loop +17 %); the full 32-block carry does not fit in 1,024 VGPRs |
| 192 x 256 tile, K slot 256, separate bf16 staging, persistent, stores draining behind the next prologue | 6144 x 8192 x 8192: 8.23 (f32 aliased epilogue 8.56, 512-K slots 8.86) | epilogue 2,250 cycles but the 256-K slot costs 21.3 cycles per WMMA in the loop (17.9 for the 256 x 256 512-K kernel); the stores sit ahead of the next prologue's loads in the in-order tensor queue |

Where the time goes (per 256 x 256 tile, 8192^3): prologue 3,200 cycles (7 %), K loop 36,100 (78 %: window 81 %, barrier 12 %, stage issue 3 %, wait 3 %), epilogue 6,300 -
8,400 (14 - 15 %). Left: issue the next tile's stage loads before the epilogue (a tail prologue across the loop back edge: needs the protocol verifier to compare the back-edge
state with the pre-loop prologue state) with a staging buffer outside the ring, which no K slot of 512 leaves room for.

### The per-tile fixed parts of the 512-K kernel (e08, 2026-11-19 evening)

- **Prologue (3.2 - 3.5 k cycles).** The first `tensor_load_to_lds` is instruction 166 of 6,121 (150 SALU / address instructions, about 300 cycles); everything after it is the wait for
  the first stage (147 KiB per workgroup, 256 workgroups at once), so issuing the first stage before the rest of the setup, or precomputing descriptor words, can save at most the
  300 cycles. Not changed.
- **Epilogue (about 7 k cycles).** LDS staging is 2,300 cycles = the LDS write bandwidth (4 waves x 64 KiB at 128 B per clock = 2,048); the tensor-store phase is 3,000 - 5,000 cycles
  for 256 KiB per workgroup (about 52 B per cycle per CU, the L2 write rate), the two waves served first finish at 3,000, the other two at 5,000, so it is throughput-bound, not
  latency-bound.
- **Banded epilogue** (`RingOpts::epi_bands` = 2 or 4: each 32 / 64-row band of the wave's block is stored by its own tensor store as soon as it is staged, so the staging of the next band
  overlaps the drain of the previous): e08, alternating with the default kernel, 2 reps. 4096 x 4096 x K: K = 2048 +5 %, 4096 **-12 %**, 8192 +3 %, 16384 +1 - 2 %; 8192 x 8192 x K:
  K = 2048 +3 - 4 %, 4096 -3 %, 8192 +0 - 2 %, 12288 -3 %, 16384 +2 - 3 %, 32768 about 0. Mixed and shape-dependent (the K = 3840 - 4608 region that the relaxed-wait dispatch also
  sees), so it stays opt-in (archive tag `archive/g1250-gemm-variants-2026-11-19d`); the default kernel is unchanged.
- **Where it could still come from:** fewer bytes through the store phase (bf16 output with a conflict-free staging whose tensor-store rows are full width: the padded layout that
  makes `ds_store_b128` conflict-free cannot be stored, since the tensor store reads its LDS rows contiguously) or overlapping the store phase with the next tile's loads (the ring is the
  staging buffer: see the 192 x 256 prototype above).
- **bf16 output on the 512-K kernel without the operand swap** (e08, paired with the f32 default, bit-exact; archive tag `archive/g1250-gemm-variants-2026-11-19e`). The tensor-store phase does halve (3,000 - 5,000 -> about 1,500 cycles), but the f32 accumulators of a lane are different rows of ONE column, so packing needs either per-element 16-bit stores or a neighbour-lane exchange: (a) `v_cvt_pk_bf16_f32` + 512 `ds_store_b16` per wave: staging 13,000 cycles (f32: 2,300), epilogue 14,700, 8192^3 8.3 PF/s; (b) a `quad_perm` DPP swap + two cvt + a select per pair + one `ds_store_b32` per pair (the lane-pair exchange): staging 15,000 cycles, 7.7 PF/s (VALU work with accumulator registers in the upper VGPR windows costs about 12 cycles per instruction in this phase, and `v_mov_b32_dpp` needed an MSB-table entry that the encoder tables do not have yet). Versus the f32 default 10.0 PF/s: **both lose**; bf16 with the operand swap (16-byte pieces, no shuffles) is the layout that fits, and it has the 12 % window cost (row permutation) or the 64-byte-chunk store cost measured above.
- **bf16 staging with a cheaper lane-pair exchange (2026-11-20, e08, archive tag `archive/g1250-gemm-variants-2026-11-20`, kernel `_bf16q`, `epilogue_lds: 7`).** Rounding first (all `v_cvt_pk_bf16_f32` of a block in one MSB window), then a `quad_perm` DPP swap of the packed pairs and ONE `v_perm_b32` per stored dword (per-lane byte selector: the even lane builds {E[r], O[r]}, the odd lane {E[r + 1], O[r + 1]}): 3 VALU per dword, 3 `s_set_vgpr_msb` per block instead of about 40. Bit-exact. Staging **7,400 cycles** (the DPP / select version 15,000, per-element `ds_store_b16` 13,000, the f32 staging 2,300); the store phase is still 2,300 - 3,300 cycles (not half of the f32's 2,800 - 4,500: all four waves store at once). 8192^3 **9.14 - 9.22 PF/s vs 9.76 - 9.86** for the f32 default in the same alternating run (-6 %), 4096^3 5.9 vs 6.2, 8192 x 8192 x 16384 9.1 - 9.2 vs 9.4 - 9.5. The remaining cost is the VALU issue rate: the 1,120 instructions of the staging take 7,400 cycles, about 6.6 cycles per instruction (a wave32 VALU is not single-cycle on this part), so ANY cross-lane pairing, `v_permlane*` included (one VALU per dword like the DPP swap), costs about 768 VALU = 3,000+ cycles per wave, more than the half-sized store phase saves. A smaller ring (ks2 2-slot, 156 KiB) with a separate 128 KiB bf16 staging plus the tail-prologue overlap would hide the prologue behind this staging (per-tile non-loop 10.2 k -> about 7.4 k, +6 %) but the 256-K slot costs about 4 % in the loop: no net gain, so it was not built.

### 4096^3 (one tile per CU): stamps and the K = 4096 anomaly (e08, 2026-11-20)

- **Stamps, default kernel, 4096^3** (per workgroup, about 31.8 k cycles = 19.5 us of the 22 us kernel): prologue **4,960** (15.6 %; 2,900 - 3,300 at 8192^3: all 256 workgroups cold-load their first
  147 KiB stage at once), K loop 19,100 (60 %: 2,455 cycles per 512-K iteration vs 2,285 at 8192^3: barrier 325 - 500 vs 220 - 300, tensor wait 60 - 180), epilogue 5,600 - 7,700 (24 %, fully exposed: nothing follows
  it). `..._roles_c4x4_lock_stamps`: prologue 3,260 (multicast cuts the cold-start traffic), loop 20,500, epilogue the same.
- **The K dependence** (4096 x 4096 x K, default kernel, us): K = 2048 12.5, 3072 15.4, **4096 22.3**, 5120 22.8, 6144 27.4, 7168 30.0, 8192 30.7, 9216 35.4, 10240 39.5, 12288 47: K = 4096 sits about 3.5 us above
  the line through its neighbours. It is NOT the row pitch: a kernarg pitch (`RingOpts::pitch`, host `LDA_PAD`, K / 2 + pad bytes) gives 21.8 - 22.1 us at K = 4096 with pads of 256 B, 1 KiB and 4 KiB (a 64-byte pad is
  worse, 28 us: a misaligned row start), and the same curve elsewhere.
- **4 x 4 lockstep multicast removes most of it** (paired alternating, 2 reps, 4096 x 4096 x K, default / `_roles_c2x2_lock` / `_roles_c4x4_lock`, us): K = 2048 12.4 / 11.8 / **11.4**; 3072 15.0 / 14.6 / **14.4**;
  **4096 21.9 / 20.6 / 17.3** (7.9 PF/s vs 6.3: -21 %); 5120 **22.9** / 24.7 / 24.6 (+7 %); 6144 **27.7** / 28.6 / 28.9 (+4 %); 8192 30.3 / 30.7 / **28.8** (-5 %). At 8192 x 8192 it is neutral or worse (K = 6144 +14 %,
  12288 +4 %, 4096 -1 %, 16384 -3 %). Dispatch rule for the output shapes with one tile per CU (256 tiles, grid divisible by the 4 x 4 cluster): `_roles_c4x4_lock` for K <= 4096 and K = 8192, the default
  kernel for K = 5120 - 6144 and for larger outputs.
- **Occupancy 2 does not help** (two 256 x 128 workgroups per CU, so that one fills the other's prologue / barrier / epilogue bubbles): the LDS fits (131,072 B each, ks2 2-slot ring) and, with backend's `racompact` allocation
  (`RingOpts::racompact`: 496 VGPRs for a kernel whose peak live set is 477, the default scan emits 720), two waves per SIMD are possible. Measured on e08 at 4096^3 / 8192^3 (PF/s): default allocation (720 VGPRs, one workgroup per CU) 4.94 / 5.57; **compact allocation, two workgroups per CU 3.99 / 5.08 (-19 % / -9 %)**;
  compact allocation with the LDS declared at 176 KiB (one workgroup per CU, `RingOpts::lds_pad`) 4.90 / 5.74. So the allocation is fine and it is the co-residency that loses: the K loop of one wave per SIMD is already
  WMMA-bound (the window is 81 % of the iteration), a second wave only adds LDS / TDM / L2 contention. (The 256 x 128 tile itself is far below the 256 x 256 kernel, 4.9 vs 6.2 PF/s at 4096^3.)

### Tail prologue + slot-staged epilogue in a persistent loop (e08, 2026-11-21): the cycles drop 5.5 %, the time does not

Idea (lead): in a persistent loop with the 2-slot 512-K ring, the peeled last window of tile t issues tile t + 1's first stage into the free slot, and tile t's epilogue is staged through the CONSUMED slot
only (the A and B regions of the slot, two 32-row bands deep per wave, each band's tensor store drained by `wait_tensor(1)` before band i + 2 is staged), so the stores are finished before the next tile's
stage 1 reuses the slot, and the next tile starts with its stage 0 already landed. Built as `RingOpts::carry` (`exp_gemm_g1250_tdm_stage_f4_256x256_ks4_s2_epit_roles_persist_carry`, the `exp_` family is not enumerated by the manifests; `Pipeline::carry`,
`PipeBody::stage_tile0` / `stage_next_tile`, a second set of operand origins for the next tile with its row clamped, no stage after the last tile; host PERSIST = G, K a multiple of 1024). It is a measuring build:
the tile-protocol data-race findings are warnings (`AMDC_TILE_PROTO_WARN`), every hang rule, bound and wait check stays enforced, and the output is bit-exact on every run (host check). `..._persist_rot` = the epilogue
alone, `..._persist_tst` / `..._persist_carry_tst` = per-tile shader-clock stamps (`RingOpts::tstamps`, host `TSTAMPS=1`).

- **The persistent loop alone costs 1 - 3 %** against the default (one tile per workgroup launched by the hardware): 8192^3 9.88 vs 10.08 PF/s, 8192 x 8192 x 4096 7.75 vs 8.01, 4096^3 6.12 vs 6.29 (the next stage's load queues
  behind the epilogue stores in the in-order tensor queue).
- **Per-workgroup cycles do improve** (8192^3, 256 workgroups x 4 tiles, wave 0, every workgroup): persistent 184.4 k cycles for the tile sequence (median; the K loop plus prologue wait 38 k, epilogue 5.6 - 6.7 k, 380 between tiles)
  vs carry **174.3 k (-5.5 %)** (tiles 1 - 3: loop 35.0 - 35.6 k, epilogue 5.0 - 6.6 k; the first tile is 40 k in both: a cold instruction cache and the first stage), as the model said (the 3.0 k prologue is gone).
- **The time does not follow.** 8192^3 on 256 workgroups, paired alternating: default 10.1, persistent 9.82 - 9.89, **carry 9.0 - 9.23 PF/s (-7 % against persistent)**; on another GPU of the node carry equals persistent. The cycles-per-microsecond
  of the stamped runs show why: persistent 184 k cycles in 113 us = 1.63 GHz, carry 174 k in 119.4 us = **1.46 GHz** - at full occupancy the kernel is power / bandwidth limited, and the schedule that keeps the machine busier gets
  the lower clock. With fewer workgroups (less power, less contention) the cycles show up in the time: PG = 64 **-3.4 %** (316.9 vs 327.7 us), PG = 128 **-3.9 %** (172.5 vs 179.1 us), PG = 256 +7 %.
- **The shapes that were the second target do not win either** (us, default / persistent / carry): 8192 x 8192 x 4096 69.0 / 69.5 / 69.9; 8192 x 8192 x 2048 42.2 / 44.1 / 44.4; 8192 x 4096 x 4096 35.1 / 36.5 / 38.1; 16384 x 8192 x 4096 135.9 / 134.2 / 138.8.
- **Verdict:** projected 4 - 5 % at 8192^3 and 8 % at K <= 4096 from the cycle model, measured -2 % ... -9 % against the default at full occupancy: no dispatch rule, and the protocol extension (the back-edge state, slot-level
  tile aliasing) was not written. Lessons: (1) at 1.5 - 1.6 GHz this GEMM is limited by something other than the cycles of one workgroup (the L2 / HBM traffic at about 28 TB/s of tile loads, or the power cap), so hiding a fixed
  cost of a workgroup is worth less than its cycles; (2) a clock estimate (cycles / us) belongs next to every stamp comparison.

### The core clock the GEMM holds, with and without multicast (e08, 2026-11-21)

The stamp host now prints `"clock":{"est_ghz":...}` next to the phase stamps: the per-tile cycles of the stamped kernel times the tiles per CU, over the kernel's microseconds (`g1250_f4own_host`; for the persistent
tile stamps: the median workgroup's tile sequence over the microseconds). Default `..._ks4_s2_epit_roles` (D) against the 4 x 4 lockstep multicast `..._roles_c4x4_lock` (M), 2 reps each, same GPU, us and the clock:

| shape | D us | M us | D clock | M clock | D loop cycles / 512 K | M loop cycles / 512 K |
|---|---:|---:|---:|---:|---:|---:|
| 4096^3 | 22.0 - 22.4 | **17.3 - 17.4** | 1.36 GHz | 1.64 - 1.66 | 18.8 k per tile loop (8 iterations) | 20.6 k |
| 8192^3 | 109.8 - 111.0 | 112.6 - 113.3 | 1.60 - 1.65 | **1.75 - 1.77** | 35.7 k per tile (2,233 per iteration) | 42.1 - 42.5 k (2,634 per iteration) |
| 8192 x 8192 x 4096 | 69.1 - 69.7 | **67.3 - 67.5** | 1.45 - 1.51 | 1.69 - 1.70 | 17.7 k | 20.4 - 20.5 k |
| 16384 x 8192 x 4096 | 135.0 - 136.5 | **130.8 - 131.2** | 1.51 - 1.53 | 1.71 - 1.73 | 17.6 k | 20.5 k |

So the multicast raises the clock by 7 - 20 % (less L2 -> LDS traffic: the A tile is fetched once per cluster row, the B tile once per cluster column, 4 x less L2 read traffic per WMMA than the 272 bytes per 32 x 16 x 128
WMMA of a lone 256 x 256 tile), but its K loop is 15 - 18 % longer in cycles (the lockstep cluster barrier per 512 K: window 1,900 as in the default, barrier + wait +400 cycles per iteration: with a 2-slot ring there
is no slack for the slowest of 16 workgroups), so the time is a wash at 8192^3 and 3 - 4 % better at K = 4096 and 21 % better at 4096^3. The clock of the lone-workgroup kernel is what the earlier cycle comparisons
missed (persistent loop with a carried first stage: 184 k -> 174 k cycles per workgroup (-5.5 %) at 1.63 -> 1.46 GHz).

**Per-FLOP traffic (ours vs the vendor's `f4gemm_bf16_mxfp4_ABpreShuffle_256x256_4x4_ps`).** Both are 256 x 256 tiles of four waves with 128 x 128 per wave: L2 -> LDS fill is 34 KiB per 128 K per workgroup =
272 bytes per WMMA (A 16 KiB + B 16 KiB + the 1 + 1 KiB scales) without multicast, about 70 with the 4 x 4 cluster. LDS reads: ours 1.0 `ds_load_b128` + 0.375 `ds_load_b32` per WMMA over the whole kernel (768 + 288 for 768
WMMAs), the vendor 1.25 + 0.31 (1,280 + 320 for 1,024 WMMAs of several forms): the LDS read traffic per FLOP is the same, it is NOT the difference. What differs is (1) the cluster (L2 traffic / clock above), (2) the
vendor hides the prologue and epilogue (persistent: our 3.0 k + 5.3 - 6.9 k cycles of a 43.6 k tile), (3) its loop iteration does not pay a lockstep barrier on a 2-slot ring (it runs more stages in flight). Putting these
together: a vendor tile of about 38.5 k cycles at 1.75 GHz = 22 us per tile reproduces 12.4 PF/s; ours with the multicast is 50.5 k cycles (2.6 prologue + 42.1 loop + 5.8 epilogue) at 1.75 GHz.

### Persistent 4 x 4 cluster, with and without the carried first stage (e08, 2026-11-21)

`RingOpts::persistent` now composes with a cluster: the grid is (TX, GYS) work-groups in clusters of 4 x 4, every work-group loops over the tile rows gy, gy + GYS, .. (GYS a multiple of the cluster rows, dividing the tile rows: host
PERSIST = G gives GYS = G / TX); the cluster barrier at every tile start (after the previous epilogue, before any peer multicasts into the LDS) keeps a peer's multicast out of a ring still used for staging. The contract is
declared (grid x = TX, grid y = GYS, tiles_y = GYS x NR, ystep = GYS), the tile row is a checked `assume`: **the kernel compiles strict, G5 and the tile protocol accept it with no trusted fact beyond the multicast kernel's**
(`exp_..._c4x4_lock_persist`); `..._persist_carry` adds the carried first stage and the slot-staged epilogue of the previous section (`PipeBody::stage_next_tile`, multicast: every work-group of the cluster issues the next
tile's stage 0 at the same point, into a slot every peer freed at the last cluster barrier). All runs bit-exact (`bad = 0`). Cycles and clock at 8192^3 (stamps, median work-group's tile sequence): PC 197.3 k cycles at
**1.74 GHz** (113 us; the loop 39.1 k per tile against 42.1 k for the non-persistent multicast); the carry variant was not stamped (the scalar register budget is full: out of SGPRs with two stamps), its time is
108.1 - 108.5 us. The carry composes better with the multicast than with the lone work-group (where it lost 7 % to the clock sag): with 4 x less L2 read traffic the saved cycles are not paid back by the clock.

**The strict acceptance of the carry kernels is NOT a proof.** A negative test (epilogue staging deliberately into the slot that holds the carried stage in flight) also compiles: the epilogue's staging stores are plain `ds_store`s
outside any declared tile, and the next tile's first stage sits under a wave-uniform `if` (first tile / a next tile exists), so the protocol does not see the hazard. Before this kernel ships the epilogue staging must be a
declared tile aliasing one ring SLOT (A and B slot regions, two bands deep, drained by `wait_tensor(1)`), the stage under the `if` must be seen by the protocol, and the persistent tile loop must check that the end-of-body
state (stage 0 of the next tile in flight in slot 0) equals the state after the pre-loop prologue; with backend.

### The persistent cluster with the carried first stage, proven and promoted (e08, 2026-11-21)

With backend's g54 protocol work (GV-TILE-UNTAGGED-OVERLAP, GV-TILE-CARRY, `tile_over_slots`) the carry kernel is proven rather than accepted vacuously: every GEMM epilogue's staging is a declared region (`EPIW`, private slices over the dead
rings; `EPIA` / `EPIB`, `LdsPlan::private_over_slots` over the A / B regions of ring slot 1 only) with produce / consume markers; the pipeline's carry mode runs pairs of windows in the static slots 0 / 1 and peels the steady and the last
window (the carried stage goes to the static slot 0, the epilogue stages in slot 1; K is declared as 2 ks KQ x 128, KQ >= 2, so the host refuses other K); the verifier's role descent checks the prologue / carry `if` of each role arm
(GV-TILE-CARRY) and the negative case (epilogue regions over slot 0) is rejected with GV-TILE-ALIAS-RACE on both the lone and the cluster kernel. Paired alternating, 2 reps, us (default D / multicast M / **PCC**):
4096^3 22.0 / 17.3 / **16.7 - 17.3**; 8192^2 x 2048 41.6 - 43.3 / 43.8 - 44.1 / **39.5 - 39.9**; 8192^2 x 4096 67.3 - 68.9 / 66.7 - 68.3 / **63.4 - 64.1**; 8192^3 108.8 - 112.7 / 111.2 - 111.4 / **106.6 - 107.6** (10.2 - 10.3 PF/s);
16384 x 8192 x 4096 135.0 - 136.8 / 129.4 - 131.7 / **124.6 - 126.2**. Without the cluster the carry loses (9.1 - 9.2 PF/s at 8192^3: the clock sag): the multicast's lower L2 traffic is what lets the saved cycles show up in the time.
Registered as `gemm_g1250_tdm_stage_f4_256x256_ks4_s2_epit_roles_c4x4_lock_persist_carry`; regression and A/B script `tools/mi450/f4pcc.sh`.

### The cluster loop's barrier and the epilogue overlap, after the promotion (e08, 2026-11-21)

**Where the +211 cycles per 512-K iteration of the cluster loop go** (stamps now cover the lock path; 8192^3, wave 0, cycles per iteration: issue / window / tensor wait / barriers = total): default D 76 / 1,915 / 62 / 200 = 2,253;
lockstep multicast M 76 / 1,936 / **12** / **376** = 2,400. The data has landed (the tensor wait is 12, shorter than D's), the extra is the cluster barrier: +176 cycles on top of the workgroup barrier. Splitting it (`F4Shape::split`, kernel
`exp_..._c4x4_lock_persist_carry_split`): every workgroup signals the cluster barrier as soon as all its waves have read their fragments (a workgroup barrier + one `s_barrier_signal -3` before the last sub-step's WMMAs of a staged
window) and waits at the end of the window after its tensor wait and the workgroup barrier: 76 / 2,018 / 12 / 209 = **2,316 (-84 cycles, -3.5 %)**: the barrier phase drops to D's level, but the mid-window workgroup barrier costs 82 cycles inside the
window. The kernel compiles strict (the protocol follows the split cluster barrier across the role copies) and is bit-exact, but the wall time moves by only about -1 % on average (4096^3 17.2 -> 16.8 us, 8192^3 107.1 -> 104.9, 16384 x 8192 x 4096
124.2 -> 123.2, 8192^2 x 2048 39.7 -> 40.2, 8192^2 x 4096 63.4 -> 62.9, 3 paired reps, run-to-run noise +-3 %): not promoted. Lockstep every N iterations is not possible with a 2-slot ring: the rendezvous of the next stage's multicast needs the peers' reads of
the slot to be finished.

**Overlapping the epilogue with the next tile's compute does not fit the LDS.** The epilogue needs the accumulators out of the registers before the next tile's first WMMA (they are the same 512 VGPRs), so the staging (2.3 k cycles) is exposed whatever
the layout; only the tensor-store drain (2.8 - 3.0 k cycles, of the 5 - 6 k) could run under window 0. The f32 tile is 256 KiB; the 2-slot ring is 288 KiB of the 320 KiB, i.e. 32 KiB are free: staging has to go through slot 1 in two rounds, and slot 1
is exactly where the next tile's stage 1 lands at the start of window 0, so the drain must finish before it, i.e. before window 0. Delaying stage 1 until the drain is done pushes its landing (about one window of latency) into window 1 and costs as much
as the drain saves; a bf16 tile (128 KiB) fits one slot but the slot is the same one. The overlap needs either a third slot (no room), bf16 staging into the 32 KiB plus 128 KiB of a slot that the stage does not need until later (no such slot with 512-K stages)
or accumulators held in a smaller footprint; not built.

### Profiles of the K loop: ingest floor, window, ring depth (e08, 2026-11-21; per tile, 8192^3, loop cycles)

| kernel (all stamped, same GPU) | loop | per 512 K: issue / window / tensor wait / barriers |
|---|---:|---|
| default D (2 slots, 512 K) | 35.7 k | 76 / 1,916 / 62 / 205 = 2,257 |
| D, no window at all (ingest only, wrong results) | **35.8 k** | - |
| D, fragments read once per window (wrong results) | 35.7 k | 76 / 1,364 / 545 / 297 = 2,282 |
| multicast M (lock) | 38.1 k | 76 / 1,935 / 12 / 376 = 2,400 |
| M, no window (ingest only) | **29.8 k** | (1,865 per iteration) |
| M, fragments read once per window | **30.7 k** | 76 / 1,365 / 186 / 312 = 1,938 |
| M, fragments read two sub-steps ahead (`ahead2`, bit-exact) | 37.4 k | 76 / 1,891 / 12 / 366 = 2,345 |
| M split cluster barrier | - | 76 / 2,018 / 12 / 209 = 2,316 |
| 3-slot 256-K ring, no cluster (`ks2` relax) | 37.8 k | per 256 K: 76 / 977 / 12 / 120 = 1,185 |
| same, workgroup barrier deleted (wrong results) | **35.7 k** | per 256 K: 83 / 896 / 42 / 106 = 1,127 |
| 3-slot 256-K ring, multicast, existing lock path | 61.3 k | per 256 K: 76 / 966 / 606 / 290 = 1,938 (the lock path waits for ALL stages) |
| same, deferred stage + early cluster signal (prototype, bit-exact, strict) | 44.2 k | per 256 K ~ 1,385 |

* **D is bound by the ingest** (L2 -> LDS, 147 KiB in about 2.24 k cycles = 66 B/clk per CU): its loop is 35.8 k with or without the window, with or without a deeper ring (the 3-slot loop with its barrier deleted is again 35.7 k).
* **The multicast loop is bound by the WINDOW**, not the ingest: its ingest floor is 29.8 k (4 x less L2 traffic), its window is 1.9 k cycles (LDS-read bound: 262 KiB of fragment reads plus 147 KiB of stage writes per iteration is about 85 % of the
  LDS bandwidth the microbenchmarks give), and reading a quarter of the fragments brings the loop to 30.7 k (-19 %). Reading two sub-steps ahead only cuts the window by 2 % (the latency is already hidden), so what would pay is fewer LDS bytes per
  WMMA, not scheduling: the 4 x 8 fragment grid of a 128 x 128 wave tile already reuses every A fragment 8 x and every B fragment 4 x (512 bytes per WMMA); a larger per-wave tile does not fit the registers (512 accumulator VGPRs + two fragment sets).
* **A deeper ring does not help the multicast loop either:** the 256-K slots double the barriers and issues per K (about 420 cycles of overhead per 256 K against 450 per 512 K), the existing 3-slot lock path serialises on a full tensor wait, and
  a deferred-issue prototype (stage t + 2 issued in the middle of window t after the cluster wait of the previous window's signal, workgroup barrier and relaxed tensor wait at the window end) is correct but 16 % slower (44.2 k).
* Where the vendor wins, then, is not in the loop (its 12.4 PF/s needs about 2.4 k cycles per 512 K including everything, our PCC loop is 2.32 k) but in the 5 - 6.5 k cycles per tile of epilogue and the ~1 k of tile transitions that stay exposed.

### Smaller first stage (analysis, not built)

The idea: split the first stage's loads into K halves, wait for the first half only, and run the first two sub-steps while the second half lands (peeling tile 0's window into the prologue). A layout-compatible
split exists (half-width tiles written with a pad of 36 dwords per 32-dword interval land in the same 272-byte rows; the scales as 8-byte tiles padded to 16), but the compute overlap needs (a) a peeled window of two sub-steps
plus a second barrier in the prologue, and (b) the tile protocol to know that a consumer reads only the first half of a slot (half-slot tile regions for A / B / SA / SB, i.e. eight rings instead of four, and every
window read declared per half): the protocol's granule is the slot. The bound: the prologue's data wait is about 4.7 k cycles of the 4,960 at 4096^3 (3,260 with the 4 x 4 multicast), the first two sub-steps are about 960 cycles of
compute, so at most 960 cycles = 3.0 % of the 31.8 k-cycle tile (about 1.5 - 2 % after the extra descriptors, the second barrier and the unchanged exposed epilogue) and about 2 % with the multicast variant. Below the
2 % bar that was set for it, against a restructure of the pipeline driver, the body and the protocol contract, so it was not built.

### Dispatch table (MXFP4 GEMM, e08, 2026-11-20)

| output tiles (256 x 256) | K | kernel | PF/s |
|---|---|---|---|
| 256 (e.g. 4096 x 4096), grid divisible by 4 x 4 | <= 4096 and 8192 | `..._ks4_s2_epit_roles_c4x4_lock` | 4096^3 7.9 (default 6.3 - 7.1); K = 2048 / 3072 6.0 / 7.2 (default 5.5 / 6.9); K = 8192 9.5 (default 9.1) |
| 256 | 5120 - 6144, other K | `..._ks4_s2_epit_roles` | 7.5 (default) |
| >= 1,024 (8192 x 8192 and larger) | any multiple of 512 | `..._ks4_s2_epit_roles` | 9.9 - 10.1 (8192^3, K = 16384) |
| **any output divisible by 4 x 4 tiles (>= 256 tiles; 4096^2 and larger)** | multiple of 1024, >= 2048 | **`..._ks4_s2_epit_roles_c4x4_lock_persist_carry`** (persistent 4 x 4 cluster, carried first stage, slot-staged epilogue; fully proven: GV-TILE-CARRY, negative case rejected; host PERSIST defaults to 256) | 4096^3 7.9 - 8.25 (multicast 7.95, default 6.25); 8192^2 x 2048 6.9 (6.3 / 6.5); 8192^2 x 4096 8.6 - 8.7 (8.05 - 8.25 / 8.0 - 8.2); 8192^3 **10.2 - 10.3** (9.9 / 9.8 - 10.1); 16384 x 8192 x 4096 8.7 - 8.8 (8.35 - 8.5 / 8.0 - 8.15) |
| outputs NOT divisible by 4 x 4 tiles, or K a multiple of 512 but not of 1024 | | the rows below | |

Clock-aware update (2026-11-21, same GPU, paired alternating, 2 reps; us of the default D / multicast M / persistent cluster PC / persistent cluster + carry PCC): 4096^3 21.95 / **17.36** / 17.6 / **16.9 - 17.1**;
8192 x 8192 x 4096 68.4 / 66.4 / 66.5 / **64.6**; 16384 x 8192 x 4096 134.5 / 136.8 / 131.8 / **127.2**; 8192^3 112.1 / 112.9 / 112.3 / **108.3**; 8192 x 8192 x 2048 42.1 / 44.3 / 43.4 / **40.4**.
So M still wins 4096^3 (-21 % against D) and the K = 4096 outputs by 2.8 - 3.4 % where D sags to 1.45 - 1.53 GHz, D wins 8192^3 and 16384 x 8192 x 4096 against M; PCC is the best everywhere:
-22 % at 4096^3, **-5.5 % / -5.3 %** at 8192 x 8192 x 4096 / 16384 x 8192 x 4096, -3.4 % at 8192^3 (10.15 PF/s), -4.1 % at K = 2048.
| any, K = 256 mod 512 | | the 256-K kernels (`..._ks2_epit_roles[_relax]`) | 9.3 (relaxed for K <= 3584 or >= 5120, plain for 3840 - 4608) |

Vendor reference on the same node: 11.0 at 4096^3, 12.4 - 12.5 at 8192^3 (so the best of the table is 72 % / 80 %).
