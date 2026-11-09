# MXFP4 GEMM (`gemm_a4w4_mxfp4`) vs aiter `f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256`: counters and ATT, 8192^3

Node 58 (MI355X, GPU 4), rocprofv3 1.0.0, `tests/gpu/vendor_gemm_host.cpp` (`vgh`), `opcov-moe` d5668189. Unprofiled same-run: ours 278.6-280.5 us, vendor 229.4-230.3 us (0.82x). Trace: ours VGPR 228 (+ AGPR, 456 unified), LDS 139264 B; vendor VGPR 256,
LDS 163840 B; both 256 threads, grid 32 x 32 CTAs, one wave per SIMD. Counters are per-dispatch medians over 121 dispatches (10 passes, one replay each: cross-pass ratios are diagnostic). ATT: one SIMD of CU 1 (ours) / CU 1 (vendor), the 5th dispatch;
the stats cover four consecutive CTA waves on that SIMD (per-wave numbers below = class totals / 4), the gap analysis is one wave.

## Counters (ours / vendor)

| counter | ours | vendor | ratio |
|---|---:|---:|---|
| SQ_INSTS | 4.17e7 | 3.58e7 | 1.16x |
| SQ_INSTS_MFMA | 1.68e7 | 1.68e7 | 1.00 (same work) |
| SQ_INSTS_VALU | 2.21e7 | 2.12e7 | 1.04x |
| SQ_INSTS_SALU | 5.42e6 | 4.76e6 | 1.14x |
| SQ_INSTS_VMEM | 4.39e6 | 2.88e6 | **1.53x** |
| SQ_INSTS_LDS | 4.59e6 | 5.33e6 | 0.86x |
| TCC_EA0_RDREQ_sum (HBM reads) | 3.34e6 | 3.35e6 | 1.00 (same bytes) |
| SQ_LDS_BANK_CONFLICT | 9.96e6 | 0 | **inf** |
| SQ_LDS_IDX_ACTIVE | 2.83e7 | 1.92e7 | 1.5x |
| SQ_VMEM_TA_ADDR_FIFO_FULL | 7.59e6 | 4.79e5 | **16x** |
| SQ_VMEM_TA_CMD_FIFO_FULL | 1.30e7 | 5.17e5 | **25x** |
| SQ_WAIT_INST_ANY | 5.10e7 | 1.76e7 | 2.9x |
| SQ_WAIT_ANY | 2.66e7 | 1.48e7 | 1.8x |
| TCP_PENDING_STALL_CYCLES | 6.21e7 | 3.49e7 | 1.8x |
| TA_BUFFER_READ_LDS_WAVEFRONTS | 2.16e6 | 1.39e6 | 1.55x |

## ATT, one wave (32 K tiles of 128 MFMA 16x16x128 each)

| | ours | vendor |
|---|---:|---:|
| wave life | 137.2 K cycles (4.29 K / tile, 33.4 cycles per MFMA) | 81.4 K cycles (2.55 K / tile, 19.9 cycles per MFMA; peak 16) |
| MFMA stream span / ideal (4096 x 16) | 121.8 K / 65.5 K: excess 64.7 K (53 %) | 68.2 K / 65.5 K: excess 9.1 K (13 %) |
| blamed excess | DMA / load issue 51 %, MFMA gaps 17 %, s_nop 16.5 %, wait vmcnt 8.5 %, barrier 6 % | MFMA gaps 75 %, barrier 14 %, wait 7 %, DMA 4 % |

Per-wave class totals (cycles), ours vs vendor: MFMA 57.8 K vs 52.4 K; **`buffer_load ... lds` DMA issue 36.9 K vs 3.3 K** (1040 vs 670 issues per wave, 35.5 vs 4.9 cycles per issue); s_nop 9.4 K vs 1.2 K (578 vs 193 per wave);
ds_read 5.6 K vs 5.4 K; SALU 5.4 K vs 4.6 K; waitcnt(vm) 5.0 K vs ~0; VALU 4.8 K vs 2.8 K; barrier 2.4 K vs 1.2 K. MFMA operand stall cycles (the MFMA waiting for its inputs): ours 28.7 K vs vendor 19.6 K per wave.

## Attribution

1. **The gap is DMA issue back-pressure** (about 60 % of the 56 K-cycle gap per wave): the `buffer_load_dwordx4 ... lds` instructions stall at ISSUE (60 - 64 hits of 280 - 300 cycles on the A / B tile DMAs), consistent with the 16x / 25x higher TA address / command FIFO full counters. Ours issues
   1.53x the VMEM instructions per dispatch (4.39e6 vs 2.88e6): per tile and wave 16 `buffer_load_dwordx4` (DMA) plus 16 `buffer_load_dword` (the per-tile scale loads; `buffer_load_dword v49 ..` sites also stall 5 K cycles) = 32 VMEM issues, against about 21 for the vendor. HBM bytes are identical.
2. **The scale loads are a third of the VMEM instructions**: 16 dword loads per tile; the shuffled scale layout would allow one `dwordx4` per 4 tiles (4x fewer issues, or a prologue-resident scale block) - the cheapest lever.
3. **A 256-byte K tile does not reduce the DMA issue count** (16 B per lane per instruction either way): it would halve barrier / wait / nop overheads per MFMA (barrier 2.4 K + waitcnt 5.0 K + nop 9.4 K = 16.8 K per wave of the 56 K gap), not the DMA stall. Expect at most the 28 % that those three classes make of the gap.
4. **LDS bank conflicts** (9.96e6 vs 0): the A / B `ds_read_b128` pattern conflicts; `SQ_LDS_IDX_ACTIVE` is 1.5x the vendor's although the LDS instruction count is lower. The ds_read class itself is small (5.6 K), but it lengthens MFMA operand waits (28.7 K vs 19.6 K per wave).
5. **s_nop**: 578 per wave against 193: the hazard padding between dependent MFMA accumulator chains and the M0 writes before each DMA (one `s_mov m0` + nop per DMA); grouping four DMAs per M0 value with the instruction offset removes three of four.

## Rerun on `opcov-moe` 003fc5ea (scale loads once per tile pair)

Same-run unprofiled: ours 262.6 - 263.4 us, vendor 229.9 - 230.9 us (0.875x; was 0.82x). Counters (ours / vendor): SQ_INSTS 3.81e7 / 3.58e7 (+6 %, was +16 %); SALU 4.77e6 / 4.76e6 (parity); VALU parity; VMEM 3.34e6 / 2.88e6 (+16 %, was +53 %);
TA_ADDR_FIFO_FULL 5.4e6 / 5.4e5 (10x, was 16x); TA_CMD_FIFO_FULL 8.2e6 / 5.8e5 (14x, was 25x); SQ_LDS_BANK_CONFLICT 9.96e6 / 0 (unchanged); TA_BUFFER_READ_LDS_WAVEFRONTS 2.16e6 / 1.39e6 (1.55x, unchanged); SQ_WAIT_INST_ANY 4.0e7 / 1.8e7 (2.3x, was 2.9x).

ATT, one wave: life 118.9 K cycles (3.72 K per tile, **29.0 cycles per MFMA**; was 137.2 K / 33.4; vendor 81.4 K / 19.9). MFMA stream excess over ideal 46.8 K (45 %; vendor 9.1 K), blamed: DMA / load issue 38.7 %, s_nop 22.9 %, MFMA gaps 20.4 %, wait vmcnt 15.3 %, barrier 1.7 %.
Per-wave class cycles (ours vs vendor): MFMA 58.0 K vs 52.4 K; **DMA issue 22.8 K vs 3.3 K** (was 36.9 K); s_nop 9.9 K vs 1.2 K; ds_read 4.9 K vs 5.4 K; waitcnt(vm) 4.8 K vs ~0; SALU 4.7 K vs 4.6 K; VALU 3.8 K vs 2.8 K; barrier 1.3 K vs 1.2 K.
Remaining gap 37.5 K per wave: DMA issue +19.5 K (52 %), s_nop +8.7 K (23 %), waitcnt(vm) +4.8 K (13 %), MFMA operand waits +5.6 K, VALU +1.0 K, barrier ~0. The B-operand DMA sites (`buffer_load_dwordx4 v16 / v14 ... s27 lds`) carry most of the issue stall.
Reading: the DMA issue stall is still the largest item and tracks the LDS-DMA wavefront count (1.55x the vendor's), so removing the B tile's DMA-to-LDS (B direct to registers) targets it; a 256-byte K tile can only halve the
s_nop + waitcnt(vm) + barrier share (16 K of the 37.5 K).