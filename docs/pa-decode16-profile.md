# Paged decode (pa_decode16 vs the aiter vendor kernel): counter attribution

Measured on node 58 (MI355X, gfx950, GPU 4), rocprofv3 1.0.0 (ROCm 7.0.2), `tests/gpu/pa_host.cpp`; ours = `pa_decode16_bf16_hd128_p16` (split-K, S = 2, plus `attn_combine_bf16_hd128`) and `_p16_f` (fused, S = 1) from `opcov-moe` @4ea7a5dd, the vendor =
`aiter::pa_bf16_noquant_gqa8_1tg_4w` (the shipped .co). B16 / B64, H = 64, HK = 8, ctx 2048, page 16. Counters are per-dispatch medians over 61 dispatches (`tools/rocprof_counter_table.py`); 10 counter passes (the kernel-wiki
triage + stall + LDS + TA/TCP passes), one replay per pass, so cross-pass numbers are diagnostic ratios, not exact identities. Unprofiled timings (same run): vendor B16 18.3 µs, ours S2 28.7 µs (decode kernel alone 21.6 µs, combine 7.2 µs), vendor B64 76.9 µs, ours fused 85.6 µs.

## Resources and geometry (kernel trace)

| | vendor | ours (`_p16`, `_p16_f`) |
|---|---|---|
| VGPR_Count (column units) / SGPR | 256 / 112 | 56 / 64 |
| LDS per workgroup | 163840 B (the whole 160 KB LDS) | 65536 B (4 waves x 16 KB) |
| workgroup / grid (B16) | 256 / (8, 16) = 128 CTAs, one per CU | 256 / (8, 16, 2) = 256 CTAs |
| resident per CU | 1 CTA = 4 waves | 2 CTAs = 8 waves (S = 1) |

The vendor is not a low-occupancy accident: it takes all of the LDS and the full register file on purpose (1 CTA per CU, 4 waves), and still reaches 5.4 TB/s on 128 of the 256 CUs at B16. `SPI_RA_VGPR_SIMD_FULL_CSN` / `SPI_RA_LDS_CU_FULL_CSN` (dispatcher stalls
for registers / LDS) are non-zero only for the vendor at B64 (2.0e7 / 5.0e6); ours is 0.

## Counters (per dispatch, decode kernel only)

| counter | vendor B16 | ours B16 S2 | vendor B64 | ours B64 fused | ours / vendor |
|---|---:|---:|---:|---:|---|
| GRBM_GUI_ACTIVE | 4.95e5 | 5.53e5 | 1.59e6 | 1.73e6 | 1.12x / 1.09x |
| SQ_WAVES | 512 | 1024 | 2048 | 2048 | |
| SQ_INSTS (all) | 1.71e6 | 3.64e6 | 7.10e6 | 1.44e7 | 2.1x / 2.0x |
| SQ_INSTS_VALU | 1.37e6 | 2.18e6 | 5.69e6 | 8.72e6 | 1.6x / 1.5x |
| SQ_INSTS_SALU | 6.5e4 | 6.97e5 | 2.64e5 | 2.57e6 | **10.7x / 9.7x** |
| SQ_INSTS_MFMA | 2.04e5 | 9.9e4 | 8.49e5 | 4.12e5 | 0.5x |
| SQ_INSTS_VMEM | 1.10e5 | 1.63e5 | 4.60e5 | 6.30e5 | 1.5x / 1.4x |
| TA_BUFFER_READ_LDS_WAVEFRONTS (DMA to LDS) | 1.0e3 | 9.9e4 | 4.1e3 | 4.1e5 | **~100x** |
| SQ_LDS_BANK_CONFLICT | 6.0e4 | 1.39e6 | 2.5e5 | 6.75e6 | **23x / 27x** |
| SQ_LDS_IDX_ACTIVE | 2.8e5 | 1.78e6 | 1.17e6 | 8.61e6 | 6x / 7x |
| SQ_WAIT_INST_LDS | 7.3e4 | 2.6e5 | 3.0e5 | 1.3e6 | 3.6x / 4.4x |
| SQ_VMEM_TA_ADDR_FIFO_FULL | 1.1e4 | 3.7e5 | 4.6e5 | 1.12e7 | **35x / 24x** |
| SQ_WAIT_ANY / SQ_WAVE_CYCLES | 26 % | 54 % | 13 % | 57 % | 2.1x / 4.4x |
| TCC_EA0_RDREQ_sum (HBM read requests) | 7.94e5 | 7.88e5 | 3.29e6 | 3.28e6 | 1.0x (same bytes) |
| TCC_EA0_RDREQ_LEVEL_sum | 7.2e8 | 8.8e8 | 7.52e9 | 7.50e9 | 1.2x / 1.0x |

## Attribution

1. **Same HBM traffic, different time**: read requests are identical (the vendor does not read less). At B64 the EA read-queue level is identical too (7.5e9): both saturate the memory system, and ours loses 8.7 % of the active cycles (tail + per-wave latency). At B16 the vendor
   reaches 5.4 TB/s with only 128 CTAs (one per CU), ours needs more CTAs (S = 2) and still takes 17 % longer in the decode kernel, plus the 7.2 µs combine launch: **the gap at small B is not parallelism, it is per-wave efficiency** (the fused S = 1 kernel has the same 128 CTAs as the vendor and takes 32.7 vs 18.4 µs).
2. **Waiting fraction**: ours spends 54-57 % of wave cycles in `SQ_WAIT_ANY` against 13-26 % for the vendor, with half as many waves per CU. That is the signature of a serialised lookup -> DMA -> wait -> compute chain per wave (one K/V tile in flight, nothing issued ahead while computing). The vendor's resource profile (160 KB LDS, the full
   register file, 1 CTA per CU) is what a deeper-buffered design needs: **the data say the vendor overlaps the next tile's loads with the current tile's compute** (circumstantial: allocation + wait fraction; an ATT timeline would prove it, not collected). So the per-wave prefetch variant (2 K/V slots per wave, 32 KB per wave, 128 KB per CTA, 1 CTA per CU) matches what the vendor does.
3. **LDS bank conflicts**: 23-27x the vendor's, and 78 % of ours LDS-active cycles at B64 (6.75e6 of 8.61e6) against 21 % (2.5e5 of 1.17e6). It is not saturating the LDS (about 17 % busy per CU) but the conflicts lengthen every dependent LDS read in the chain. Cause to check: the ds_read pattern / row stride of the K and V tiles in LDS (swizzle or padding), and
   the DMA-to-LDS write footprint (`TA_BUFFER_READ_LDS_WAVEFRONTS` is 100x the vendor's: the vendor barely uses buffer_load...lds: its loads land in registers).
4. **TA address FIFO**: `SQ_VMEM_TA_ADDR_FIFO_FULL` 24-35x the vendor's: the LDS-DMA (`buffer_load ... lds`) path backs up the texture-address unit; with 16 DMAs per tile per wave issued back to back, issue stalls on the FIFO.
5. **Instruction count**: ours executes 2.0-2.1x the instructions, 10x the scalar ones (about 680 SALU per wave against 127: block-table lookups and descriptor / offset arithmetic redone per tile) and 1.5-1.6x the VALU (softmax and index math). At B64 (memory saturated) this is hidden behind the HBM; at B16 it sits on the critical path of a latency-bound chain.
6. **MFMA is not the issue**: ours issues half the MFMAs (the vendor 16x16x16 shape, twice the instructions for the same work) and has half the MFMA busy cycles.
7. **Combine launch**: 7.2 µs (25 % of the 28.7 µs at B16) is the second kernel; the fused variant removes it but cannot split K (S = 1), so at B16 the fused kernel needs the efficiency fix of 2 to win.

Levers by evidence: (a) per-wave software pipelining of tile t + 1 (2 K/V slots, 1 CTA per CU); (b) fix the K/V LDS read/write swizzle (bank conflicts) and/or move the K loads to registers like the vendor; (c) hoist / amortise the block-table lookup and descriptor setup out of the per-tile path (SALU 10x), e.g. resolve a whole page-table row into SGPRs once per 4-8 tiles; (d) fused combine for S = 1 at B >= 64 (already there).