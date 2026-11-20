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
