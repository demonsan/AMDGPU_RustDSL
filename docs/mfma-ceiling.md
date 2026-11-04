# MFMA per-instruction ceiling on a loaded MI355X (gfx950)

Probe: `tools/mfma_ceiling.sh` (`tools/mfma_ceiling.py` generates 24 kernels, `tests/gpu/mfma_ceiling_host.cpp` runs them). Each wave times its own loop with `s_memtime` (core-clock ticks) and `s_memrealtime` (100 MHz), so cycles per MFMA and the **effective clock** come from inside the kernel, not from an assumed 2.4 GHz. 32 independent `v_mfma_scale_f32_16x16x128_f8f6f4` per iteration over 16 accumulators (4 A fragments x 4 B fragments, same A for 4 MFMAs, B varying), 4 waves per workgroup, one workgroup per CU. GPU 4 of mi355-gpu-53, ROCm 7.0.2 runtime, 2026-11-04.

## Results

1. **Cycles per MFMA are 32.25 for every variant** (acc VGPR / AGPR / 12 VGPR + 4 AGPR mix; A/B fragments in VGPR or AGPR; no op_sel vs the four GEMM op_sel/op_sel_hi byte-select forms; zero or random operands; 1 or 256 workgroups). min = median = max over all waves. The matrix pipe issue interval (32 cycles plus 0.25 of loop overhead) does not depend on the operand forms or scale byte selects. Scale operands in SGPRs or as inline constants do not assemble on gfx950 ("invalid operand"): VGPR scales are the only form, so that axis has nothing to choose.
2. **The clock is what depends on the data.** Effective clock (in-kernel), full device (256 workgroups):

| operands | clock first launches | sustained (60 back-to-back 1.8 ms launches) | per-MFMA |
|---|---|---|---|
| zero | 2.23-2.39 GHz | 2.39 GHz | 13.5 ns |
| random fp8 (bit 3 of every byte cleared, so no NaN) | 1.63-1.95 GHz | **2.03 GHz** | **15.9 ns** (16.9-17.7 ns in the first launches) |

   A single workgroup (one CU) runs at 2.38 GHz with random data: the drop is a whole-chip power effect (about 19 % in a cold single 1.8 ms launch, about 15 % sustained), not a per-CU effect. A 29 ms launch holds 2.02-2.04 GHz.
3. So the per-MFMA ceiling for real data on a fully loaded chip is **32.25 cycles at about 2.0 GHz = about 15.9 ns**, not 13.9 ns. The pure MFMA time of one wave of the 4096x4096x8192 GEMM (4096 MFMAs) is 4096 x 15.9 ns = **65 µs** (59 µs if the clock were 2.2 GHz).
4. The GEMM numbers: the profiled vendor and candidate runs of 2026-11-04 (`GRBM_GUI_ACTIVE` / duration: vendor 181 k cycles per XCD in 86.5 µs = 2.09 GHz, ours 205 k in 96.6 µs = 2.13 GHz) show the GEMMs themselves run at 2.1 GHz, the same as the random-data probe, so **clock is not the difference between the kernels**. 20.1 ns per MFMA (MMA-only ablation, whole kernel / 4096) = 82 µs against a 65 µs MFMA floor: about 17 µs is not MFMA time (prologue about 6 µs, epilogue about 5 µs from the ATT traces, launch ramp and tail), and the vendor's 84 µs is a 77 % MFMA duty cycle. No operand or scale form lifts the ceiling; the only slack is outside the MFMA stream.

## Per-tile synchronisation cost (`tools/mfma_block.py`, MPI=64)

One loop iteration = one 64-MFMA tile with the GEMM v4 accumulator split (40 VGPR-form + 24 AGPR-form), the four `op_sel` forms, random operands, 4 waves per workgroup, in-kernel `s_memtime`. Cycles per MFMA, full chip (256 workgroups) and one workgroup are equal to 0.01:

| sync per 64-MFMA tile | cycles/MFMA | extra cycles per tile vs none |
|---|---|---|
| none | 32.38 | 0 |
| `s_barrier` after the 64th MFMA | 32.72 | +22 |
| `ds_read_b64` at the block start, `s_waitcnt lgkmcnt(0)` + `s_barrier` after the 64th MFMA | 32.86 | +30 |
| vendor order: `ds_read_b64` after the 64th MFMA, `s_waitcnt lgkmcnt(0); s_barrier` after 4 MFMAs of the next tile (pre_sync) | 32.50 | +8 |
| `ds_read_b64` + wait, no barrier | 32.50 | +8 |

The operand file (srcA/srcB AGPR, VGPR, or srcA AGPR + srcB VGPR) changes none of these numbers (rows identical to 0.01 for A, V, M). With balanced waves a tile barrier costs at most 30 of 2072 cycles (1.4 %), the vendor's order is cheapest. These are pure sync latencies: no DMA, no skew between waves, no LDS bank pressure.

Against the GEMM's 187 k cycles per wave (MMA-only ablation, 45.6 cycles per MFMA): 4096 MFMAs x 32.9 = 135 k cycles is the pipe floor with per-tile sync, so about 52 k cycles (28 %) are spent outside the MFMA issue stream: the prologue and epilogue (about 24 k in the earlier ATT traces), launch ramp and tail, and bubbles inside the stream (about 22 k MFMA-stream excess in the ATT trace). Stamping `s_memtime` at kernel start / prologue end / loop end / kernel end inside the GEMM would split the 52 k exactly.

## Clock and power, read from the SMU (amd-smi) while the probe ran

With `HIP_VISIBLE_DEVICES=4` the loaded GPU appeared as **amd-smi index 7** on this node (indices 0-6 idle at 157-177 MHz / 250-260 W); index mapping differs between HIP and amd-smi, so identify the loaded device by its clock. During a sustained random-data run on all 256 CUs: `SOCKET_POWER` 1323-1333 W, GFX clock 2076-2091 MHz, against 2.04 GHz measured in-kernel (`s_memtime` / `s_memrealtime`): the board is at its power cap, the SMU clock and the in-kernel clock agree to 2 %.
Earlier note (superseded by the section above): Not conclusive: amd-smi sampled during the run read `SOCKET_POWER` 258-288 W but a `GFX_0 CLK` of 158 MHz for the device it was pointed at, i.e. it was not reading the loaded GPU (HIP_VISIBLE_DEVICES vs amd-smi indexing), so no power figure is quoted; the in-kernel clock is the measurement.