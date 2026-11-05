# GEMM issue-level comparison: vendor (aiter PR #6696 4-wave pin-AGPR) vs gemm_mxfp8_bs_v3/v4 (v3 removed in the 2026-11 cleanup; v4 is current)

Setup: GPU 4 (MI355X), M=N=4096, K=8192, CTA 256x256, same process / same inputs (`tools/gemm_ref.sh` host). Tools: `tools/prof_gemm.sh` (rocprofv3 SQ counters, 3 passes, per-wave
normalised, GEMM dispatches only) and `rocprofv3 --att` (thread trace of one wave: CU 1, SIMD 0, decoder `rocprof-trace-decoder 0.1.6`) aggregated with `tools/att_classes.py`.
Caveats: SQ counters are sampled on a subset of waves (ratios are meaningful, absolutes are not); ATT shows ONE wave and its per-instruction Latency / Stall / Idle (cycles) - the class table
is a distribution, not wall time; profiled runs perturb timing (v4 shows 1.65x vendor wave cycles under counters, 1.43x in the benchmark).

## 1. Instruction mix per wave (SQ counters; vendor | v3 | v4)
| counter / wave | vendor | v3 | v4 |
|---|---|---|---|
| MFMA | 4096 | 4096 | 4096 |
| VALU incl. MFMA (SQ_INSTS_VALU) | 5051 | 10388 (2.06x) | 11086 (2.19x) |
| => VALU excluding MFMA | 955 | 6292 (6.6x) | 6990 (7.3x) |
| SALU | 2154 | 1170 (0.54x) | 1188 (0.55x) |
| LDS instr | 2264 | 2624 (1.16x) | 2624 |
| VMEM instr | 1060 | 1216 (1.15x) | 1234 |
| wave cycles | 40.7k | 54.6k (1.34x) | 67.0k (1.65x) |
| SQ_ACTIVE_INST_VALU | 9149 | 14484 (+5.3k) | 15182 (+6.0k) |
| SQ_WAIT_INST_ANY (cycles waiting on dependencies / waitcnt) | 17.8k | 25.2k (+7.4k) | 36.8k (+18.9k) |
| SQ_WAIT_INST_LDS | 236 | 544 | 1264 |
| LDS bank conflicts | 4577 | 4096 | 4096 |
Wave-cycle excess over the vendor: v3 +13.9k = about 38% extra VALU issue time (+5.3k) and 53% extra dependency waiting (+7.4k); v4 +26.3k = 23% VALU, 72% dependency waiting (LDS waits x5).

## 2. ATT class distribution of one wave (share of summed instruction latency)
| class | vendor hits / lat% | v3 hits / lat% |
|---|---|---|
| MFMA | 4096 / 47.6% (24.0 cyc/hit) | 4096 / 52.4% (26.3) |
| `s_waitcnt vmcnt(0) lgkmcnt(0)` (full drain before the barrier, 2 per iteration) | 68 / 17.6% | none |
| `s_barrier` | 69 / 4.4% | 64 / 0.9% |
| DMA (`buffer_load ... lds`) | 1028 / 14.2% | 1152 / 6.9% |
| VALU (non-MFMA) | 778 / 2.0% | 6036 / 15.8% |
| `s_nop` | 864 / 1.7% | 1660 / 5.3% |
| ds_read | 2208 / 4.4% | 2624 / 6.2% |
| global/buffer stores (epilogue) | 32 / 1.0% (16-byte `buffer_store_dwordx4`, 56 `ds_write` to transpose) | 64 / 6.4% (4-byte `global_store_dword`, 207 cyc/hit) |
| SALU | 2154 / 4.2% | 1170 / 2.3% |
| `v_accvgpr_*` | 176 | 256 |
The vendor spends its non-MFMA time in the explicit tile synchronisation (full drain + barrier + the DMA issue stalls = 36%), ours in VALU, nops and the epilogue.

## 3. What the vendor loop does that ours does not (beyond the DMA address VALU chains)
1. **Epilogue**: transposes the accumulators through LDS (`ds_write` x56) and stores 16 bytes per lane (`buffer_store_dwordx4` x32); ours issues 64 four-byte `global_store_dword` (6.4% vs 1.0% of latency, ~5% of this shape's time; fixed cost, shrinks with K).
2. **Fewer `s_nop`s**: 864 vs 1660 hits (1.7% vs 5.3%). Ours come from the hazard pass (MFMA-result/VALU-write rules); worth a dedicated look at which pairs force them in the v3/v4 loop.
3. **VALU is off the critical path**: 778 non-MFMA VALU hits (+176 accvgpr) against 6036 in ours; in our trace the VALU instructions interleaved with MFMAs show 28-32 stall cycles per hit (they wait for the MFMA pipe), so each is far more expensive than its 4-cycle issue.
4. **Explicit full sync once per tile** (`vmcnt(0) lgkmcnt(0)` + `s_barrier`): the vendor pays 22% of its wave latency there but lets the DMA run uncontended behind 4 MFMAs (`pre_sync`); ours replaces it with partial waits and spreads the DMA issue, and loses on VALU/nop/ds_read stalls instead (ds_read stall+idle 5.2k vs 1.5k).
5. **Setprio / stagger**: vendor 8 `s_setprio` static, ours 4; no wave stagger found in either trace (one-wave view; the 4-wave interplay needs a multi-wave trace).
6. Instruction counts that are LOWER in the vendor: LDS instructions -14% and VMEM instructions -13% (wider/fewer scale reads and DMA issues) - ours issues more small ones.

Conclusion for the next steps (ranked by estimated gain on this shape): (a) remove the ~5.3k extra VALU per wave (SALU address chains, backend-llvm-tail), (b) the LDS-transposed 16-byte epilogue (~5%), (c) the hazard-pad `s_nop`s (~3.5% of latency, inspect pairs), (d) fewer/wider LDS + DMA instructions, (e) re-test `pre_sync` MMAs once (a) lands.