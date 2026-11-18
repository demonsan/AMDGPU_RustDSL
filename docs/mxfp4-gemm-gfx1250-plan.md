# MXFP4 GEMM on gfx1250 (MI450): status and the plan toward the vendor structure

Kernel: `gemm_g1250_tdm_stage_f4*` in `crates/amdgpu-kernels/src/probes/g1250_gemm.rs`, body `amdgpu_stage::gemm::tdm_f4::TdmF4Body` driven by `Pipeline` with `Cadence::OneWindow` (3-slot ring, producer at iteration offset 2). Host `tests/gpu/g1250_f4own_host.cpp` (CPU reference on sampled rows, polled waits, `ONLY_K` bisect aid), payloads `tools/mi450/f4hw.sh`, `f4vs.sh`, `f4bisect.sh`, `f4variants.sh`.

## Where we are (one GPU, f32 output, natural row-major A / B / scales, 8192^3)

| kernel (tile, waves) | declared LDS | TFLOP/s | note |
|---|---:|---:|---|
| `gemm_g1250_tdm_stage_f4` (128 x 64, 4 waves, every wave loads the whole tile) | 39 KiB | 0.98 | first version |
| the same with cooperative loads (each wave loads a slice, one barrier publishes) | 39 KiB | 1.42 | step 1 |
| `..._128x128` (wave tile 2 x 4 WMMAs) | 51 KiB | 1.73 | step 2 |
| `..._256x64` (wave tile 4 x 2) | 64 KiB | 1.66 | step 2 |
| vendor `f4gemm` 256 x 256, 4 x 4 cluster, persistent, bf16 output | 320 KiB | 10 - 11.4 | the target |

All bit-exact against the CPU reference (E8M0 scales 2^-2 .. 2^2, random fp4). The gap is about 6x.

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
