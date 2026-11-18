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
