# Small-tile i8 GEMM (`gemm_a8w8_i8_64`) vs aiter `I8gemm_bf16_perTokenI8_BpreShuffle_64x128`: counters and ATT, 256 x 7168 x 4096

Node 58 (MI355X, GPU 4), `opcov-moe` ccf0ca14, `vgh ours_v4_i8 ... 256 7168 4096 64 128` / `vgh asm_i8 ...`. Same-run: ours 19.7 - 20.0 us, vendor 13.9 - 14.6 us (1.42x). 224 CTAs of 256 threads (one round on 256 CUs), 32 K tiles of 128 B per CTA.
Trace: ours VGPR 60 + AGPR, LDS 53760 B; vendor VGPR 256, LDS 65536 B. ATT: one SIMD of CU 1; per-wave numbers (the traces hold one wave: 512 MFMA 16x16x64 for ours, 1024 MFMA 16x16x32 for the vendor, the same work).

## Counters (ours / vendor)

SQ_INSTS 2.46e6 / 2.70e6; SQ_INSTS_VMEM 1.86e5 / 3.87e5 (ours issues HALF); SQ_INSTS_LDS 3.5e5 / 2.4e5; SALU 7.5e5 / 8.3e5; VALU 7.7e5 / 1.13e6; TCC_EA0_RDREQ_sum 2.96e5 / 2.96e5 (same bytes);
**SQ_WAIT_ANY 5.39e6 / 1.21e6 (4.5x)**; SQ_WAVE_CYCLES 9.4e6 / 6.07e6; SQ_VMEM_TA_ADDR_FIFO_FULL 2.0e4 / 2.4e5 and TA_CMD_FIFO_FULL 4.9e4 / 2.7e5 (ours LOWER: no DMA issue pressure here); SQ_LDS_BANK_CONFLICT 5.2e5 / 0 (small: 0.27 of LDS_IDX_ACTIVE 1.9e6);
TCP_PENDING_STALL_CYCLES 4.35e6 / 2.23e6.

## ATT, one wave

| | ours | vendor |
|---|---:|---:|
| wave life | 40.7 K cycles (1.27 K / tile) | 24.5 K cycles (0.77 K / tile) |
| MFMA stream span / ideal | 34.7 K / 10.2 K: excess 26.7 K (77 %) | 18.5 K / 12.3 K: excess 6.5 K (35 %) |
| blamed excess | **wait vmcnt 81 %**, s_nop 10 %, ds op issue 7.7 % | MFMA 66 %, barrier 18.5 %, wait 15 % |
| class latency | **waitcnt(vm) 36.7 % (33 hits x 438 cycles)**, DMA issue 12.5 %, MFMA 11.5 %, s_nop 9.2 %, SALU 8.5 %, ds_read 7.1 %, barrier 6.1 % | MFMA 50.7 %, SALU 15.6 %, DMA issue 9.8 %, waitcnt(both) 6.7 % (33 x 48 cycles), VALU 5.5 %, ds_read 4.7 %, barrier 2.1 % |

The two `s_waitcnt vmcnt(0)` sites of the K loop (one per slot) cost about 10.5 K cycles of MFMA-stream excess each per wave (15 hits, about 700 cycles each); the class table has the same 14.4 K cycles (35 % of the wave) in `waitcnt(vm)`.
The vendor never waits for a whole tile: its waits are `s_waitcnt vmcnt(12) lgkmcnt(0)` (twelve loads still in flight) at 48 cycles per hit.

## Attribution

The gap is **exposed global-load latency from the 2-slot ring**: the loop waits for ALL outstanding DMA (`vmcnt(0)`) before every barrier, so the load of tile t+1 (issued at tile t) is the only load in flight and the whole HBM / L2 latency (about 440 cycles per wait) is exposed 33 times per wave.
It is not DMA-issue pressure (TA FIFO-full counters are lower than the vendor's, ours issues half the VMEM instructions), not the HBM volume (identical), and the LDS bank conflicts are small. `s_nop` (9 %) and the barrier (6 %) are the next items.
Lever: a 3- or 4-slot ring (the LDS budget allows it: 53760 B now) that issues tile t+2 (t+3) before consuming tile t and waits with a COUNTED `vmcnt(N)` like the vendor's `vmcnt(12)`; the estimate is the 14.4 K exposed cycles per wave, which would put the wave life (40.7 K) at the vendor's 24.5 K.