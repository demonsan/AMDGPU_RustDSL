# Kernel generation 2: overlapping the VALU and MFMA work inside one wave

Status: design for review (kernel-opt). Baseline = `fmha_fwd_hd128_mxfp8_vs128` (own backend, B1H8S2048): **366.7 TFLOP/s**, vgpr 252, vendor ~514 (gap 1.40x).
Everything below is derived from measurements already in `docs/status.md` (rounds 49-56) plus one new probe (section 2). Predictions are ranges, with the assumption that decides the range stated explicitly.

## 1. What the measurements say (the model's inputs)

Per 128-key tile, per SIMD (2 resident waves, each doing one tile): wall W = **5.37k cycles** (0.0468 ms at B1H8S2048).
Timing-only skeletons (round 56): VALU-only (`noqk,nopv,nodma,nobar`) 2.3k, MFMA/LDS-only (`novalu,nodma,nobar`) 3.0k, DMA 0.44k, barriers ~0.3k. Sum of parts 5.3k = wall: **no overlap of the VALU and MFMA work, neither between the two waves of a SIMD (complementary phasing is already in place and worth +15%) nor inside a wave.**
Counters per wave-iteration: VALU 352, MFMA busy 1216, LDS instructions 52.8 (vendor parity), WAIT_ANY 560, wave cycles 1599.
Per wave: V phase (max, rollback, P conversion) = 1.15k, M phase (QK 12 MFMA, PV 8 MFMA, row sums 4 smfmac, ~53 LDS instructions) = 1.5k. Ideal complementary phasing would give a window of max(V, M) = 1.5k and W = 3.0k + overhead; we see 5.3k = 2 (V + M).

### 1.1 Hardware overlap probes (new: `sched_ovl_*`, `sched_coissue_*`; `tests/gpu/sched_host.cpp`)
Cross-wave (coissue, round 51): MFMA alone 368, VALU alone 715, both 763 cycles/iteration (max + 48).
Intra-wave and 2 waves/SIMD, same code in every wave, 8 independent MFMAs + 16 softmax-conversion chains (pknorm, shift/mask/perm LUT, pk_sub, fp8 pack) per iteration, cycles per iteration (list scheduler on / off):
| waves per SIMD | MFMA only | VALU only | both, MFMA first | both, VALU first | sum | max |
|---|---|---|---|---|---|---|
| 1 (block 256) | 604 / 663 | 791 / 913 | **927** / 1306 | 943 / 1439 | 1395 | 791 |
| 2 (block 512) | 1090 / 1138 | 1499 / 1635 | **1796** / 2030 | 1792 / 2125 | 2589 | 1499 |
So when MFMA and VALU work are INDEPENDENT and interleaved by the scheduler, a wave overlaps ~66% (1 wave/SIMD) to ~73% (2 waves/SIMD: 793 of 1090 MFMA cycles hidden) of the smaller pipe: wall = max + 0.27 min. In-order issue is not a barrier by itself; the kernel loses the overlap because its VALU and MFMA work are DEPENDENT inside a tile (QK -> max/convert -> PV) and are separated by barriers into phases that its own two waves execute back to back.

## 2. The register / LDS accounting of the baseline (vgpr 252)
O 64 (4 x V16) + S 64 (4 x V16) + Q resident 16 + K fragment transient 16 + V fragment 8 + P words 16 + row-sum accumulator 4 + a_rs 4 + bias word 2 + K scales 4 + conversion temporaries ~12 + addresses/tables/constants ~40 (K/V LDS address tables 8, DMA bases 8, pointers, ring/loop state, constants). LDS 102400 B (ring 2 x 32768 + K scales + Q staging 34816, which is now unused with resident Q: **34 KB free**). The LDS bandwidth budget: 128 B/cycle/CU; the baseline moves per tile per wave 16 KB of K fragments + 16 KB of V (transposed) reads = 256 KB per CU-tile = 2.0k cycles (of 5.37k).

## 3. The options, with budgets and costs

The overlap needs QK(i+1) MFMAs in flight while the VALU of tile i runs. QK(i+1) writes S, which the VALU still reads: either S is double-buffered, or each S tile is converted before QK overwrites it, or S lives elsewhere.

### A) 64-key units, double-buffered S
VGPR: O 64, S 2 buffers x 2 V16 = 64 (same as now), Q 16, P 8 (half the words), K fragments 16, V 8, la 4, a_rs 4, rest as now -> ~244 (8 free). LDS unchanged. Overlap: QK(i+1) (into the other buffer) in flight during the VALU of tile i, PV(i) after. Cost: the exact-reference rollback (threshold re-aligns EVERY unit) now fires every 64 keys: VALU per 128 keys 352 -> ~474 (rollback ~90 VALU twice; plus the in-flight S(i+1) must be shifted by the same delta: +16 pk_add) unless the decision is made before QK(i+1) is issued, which removes the in-flight shift but serialises max+rollback before the MFMA issue. Model (wall = max(XDL 3.0k, VALU 3.1k) + (1-f) min + 0.5k fixed): f = 0.73 -> 4.2k (+28%, ~470 TFLOP/s); f = 0.5 -> 5.1k (+5%). Risk: the doubled rollback VALU makes the kernel VALU-bound; the whole gain depends on f.

### B) 128-key units, S in LDS
S as packed bf16 = 8 KB/wave/tile written + read back: +16 ds_write_b128 + ~16 ds_read_b128 per wave-iteration (LDS 52.8 -> ~85) and +256 KB/CU-tile of LDS traffic: LDS busy 2.0k -> 4.0k cycles per tile, above the 3.5-4k wall target: **LDS-bandwidth bound. Rejected.** (LDS capacity would also need the 34 KB free + ring recycling.)

### C) 128-key units, half of O in LDS
Frees 32 VGPR, costs +128 KB/CU-tile of LDS traffic (+1.0k cycles) and still cannot hold a second S buffer (64 VGPR). **Rejected** (does not enable the overlap).

### D1) "QK inside the V phase" (single S, 3-slot ring) - chosen
Keep the 128-key unit, the exact reference, the complementary two-group phasing and the two barriers per tile, but move the QK of the NEXT tile into the V phase of the current one, tile by tile: after S tile s of tile t has been converted (its 16 registers are dead), the 2 scaled MFMAs (+ bias) of QK(t+1, s) write into that very S tile. The V phase becomes: max/rollback(t) -> for s in 0..4 { read K(t+1, s) fragments; convert S(t, s) (VALU, ~100 instructions); QK(t+1, s) MFMAs (in flight during the next conversion) } ; the M phase keeps only PV(t) + row sums (12 MFMA instead of 24) and its LDS reads. No double buffer: the same registers as now; the rollback decision stays per 128 keys; QK(t+1) uses the bias AFTER the rollback of t (no in-flight shift).
VGPR: O 64, S 64, Q 16, K fragments 16 (live only across one conversion step), V 8 (M phase only), P words 16, la 4, a_rs 4, K scales 4, temporaries 12, rest ~40 -> **~248-252 (same as now)**; the V phase peak adds the K fragments (16) to the baseline's V-phase set but drops the V fragment: expected within +-8 of the baseline. If it does not fit, `kfrag` loads are issued per half (8 regs).
LDS: 3-slot ring (3 x 32768 = 98304) + K scales 3 x 512 = 99840 B (the Q/ones staging 34 KB is dropped: Q is resident) - fits. DMA: tile t+2 is staged in V(t), waited for at the end of V(t) (K(t+1), staged in V(t-1), is already there when V(t) starts).
Added cost: the K fragment LDS reads move from the M phase into the V phase (no change in count: 52.8), 3 slots instead of 2 (no cost). Hazards: QK(t+1,s) must follow the last VALU read of S tile s (the conversion of tile s) - an ordinary RAW/WAR through the registers; `decide` must precede the first QK (needs the final bias).
Expected overlap: the 2 x (8 scaled + 4 bias) QK MFMAs (~0.77k XDL cycles) execute under the ~1.15k of VALU of the same wave; PV(t) + smfmac (~0.58k XDL) and the V transposed reads form the M window; the partner group runs the opposite window. Window times: V' ~ 1.15k + 0.1k (MFMA issue) = 1.3k, M' ~ 0.8k -> W = 2 max(V', M') + barriers 0.3k + DMA 0.2k ~ 3.1k (+73%, 600+ TFLOP/s) is the OPTIMISTIC bound; the XDL floor is 2.43k; with the measured overlap efficiency f = 0.73 of the probe the model gives ~3.9k (+38%, ~500); with the pessimistic f = 0.5 ~4.5k (+19%, ~436). **Predicted 4.0-4.5k, i.e. +20..35% (440-495 TFLOP/s).**
Risks: (1) the list scheduler may re-order the MFMA issue (mitigation: a `Stmt::Asm` block for the V-phase inner loop, see section 5); (2) the unexplained observation that the complementary phasing gives 2 (V + M) instead of 2 max(V, M) - if its cause is contention between the VALU of one wave and the MFMA of its partner (register-file ports), D1 moves MFMAs INTO the VALU phase and would pay the same contention; the first prototype therefore measures exactly that (V-phase wall with QK inside); (3) peeled tail and masked tile need the same restructuring (QK(t+1) is not issued in the last iteration).

### D2) 96-key units
Not costed in detail: S = 3 V16 (48) halves the double-buffer cost (96) but needs 3-tile tiles (key count 96 does not divide the 128-row LDS layout and the 16-row staging); rejected as too invasive for the uncertain benefit.

## 4. Decision
D1 (QK-in-V, 3-slot ring) is the only option that keeps the register budget, the LDS traffic, the exact-reference recipe (no rollback doubling) and the complementary pairing; A is the fallback if D1's MFMA-in-VALU-window turns out to serialise (A has the same contention risk and a larger rollback bill). B and C are rejected by LDS traffic.

## 5. Prototype plan (new kernel `fmha_fwd_hd128_mxfp8_g2`, `_vs128` stays the fallback)
1. Step 0 (1 day): copy `emit_fmha_vs128_role` -> `emit_fmha_g2_role`; 3-slot ring (reuse the carried-scale `kdp` scheme of the old `_vs` role or the "stage t+2 / wait vm(0)" scheme above); move QK(t+1, s) into the V phase after the conversion of tile s; M phase = PV + smfmac only. Gate: correct vs `_vs128` (bit-identical output expected: same arithmetic), then timing.
2. Step 1: measure V' and M' windows by ablation (`novalu`, `nopv`, `noqk`) and the counters (VALU, MFMA busy, WAIT_ANY); compare with the model above.
3. Step 2 (only if the scheduler reorders the MFMAs): author the V-phase inner loop as a `Stmt::Asm` block (typed MFMA/SMFMA tokens, explicit waits) via backend's `Op::Asm`/`Stmt::VarAsm` bridge so the interleave is explicit.
4. Gates: 15 shapes, 0 bad at 16 and 64 rows, determinism 3x5, parity (REF_ROWS=64); candidate default only if >= 3% over 366.7.

## 6. Step 1 result (prototype `fmha_fwd_hd128_mxfp8_g2`, own backend; options `g2split`)
Builds first time (vgpr 252), **bit-identical output** to `_vs128` on 1 8 2048 2048 (md5 identical; the other shapes differ only because `_vs128` runs the LUT copy for < 9 tiles). Timing B1H8S2048: `_vs128` 367.2; **g2 333.4 (-9%)**; **g2 + `g2split` 344.2 (-6%, vgpr 256)** (conversion + QK of S tiles 2,3 moved into the M phase after PV half 0 to balance the windows). Scheduler variants of g2: `AMDC_SCHED=0` 293.7, `ld` 304.5, default (full) 333.1.
Asm check (A-role loop): the QK MFMAs are inside the V phase interleaved with the conversion VALU as designed (M x3 / W / M x8 ... between `v` runs), the M phase holds only the PV/smfmac chain: 24 MFMA + 53 ds per iteration, 2 barriers, 14 waitcnt.
Counters per wave-iteration, g2 vs `_vs128`: VALU 385.6 vs 352.1 (+33: per-tile call overhead; hoisting the conversion constants did not change the time); LDS 52.8 = 52.8; MFMA busy 1216 = 1216; WAIT_INST_ANY 428.7 vs 543.4 (**-115: fewer dependency stalls**); SQ_WAIT_ANY 812.8 vs 560.4 (**+252**: waits that are not instruction dependencies, i.e. barrier/waitcnt skew between the windows); wave cycles 1760 vs 1599.
Ablations of g2 (TFLOP/s, 0.0516 ms = 6.09k cycles per tile vs 5.37k): `novalu` 361.4 (-0.7k), `noqk` 346.1 (QK removal saves only 0.3k although it is ~0.77k of XDL work: **the QK MFMAs inside the V phase ARE overlapped with the VALU**, as the model predicted), `noqk,nopv` 348.3, `nobar` 318.5, `nodma` 337.4; skeletons `novalu,nodma,nobar` 534.3 (3.3k), `noqk,nopv,nodma,nobar` 516.7 (3.4k).
Interpretation: the intra-wave overlap works (QK hidden, dependency stalls down by 115 cycles/wave-iteration) but the windows are unbalanced: V' carries VALU 1.15k + MFMA issue + K reads, M' only PV; the complementary partner arrives early at the barrier and waits (+252 non-dependency wait cycles), costing more than the overlap gains. `g2split` recovers 3.3% by moving half of the conversion into M. Predicted headroom from step 1 is therefore NEGATIVE at the current balance (-6..-9% vs 367.2); the model's V'/M' window assumption (M' ~ 0.8k) did not hold because the measured barrier skew costs 0.25k per wave-iteration.

## 7. What the experiment established, and the prerequisites for the next attempt
1. **Intra-wave overlap works when the work is independent** (QK inside the V window: removing QK saves 0.3k of its ~0.77k XDL cycles; dependency stalls -115 per wave-iteration; first direct evidence in a kernel of ours). The earlier "in-order issue prevents it" is refined: the DEPENDENCE structure of the tile, not issue order.
2. **It did not pay because of window balance and phase rendezvous**: V' (VALU + MFMA issue + K reads) vs M' (PV only) are unbalanced, the complementary partner waits at the barrier (WAIT_ANY 560 -> 813, wave cycles 1599 -> 1760); `g2split` (+3.3%) shows balance matters but is not sufficient.
3. **Minimum viable gen2 design**: (a) balanced V'/M' windows; (b) finer-than-block synchronisation - per-group named barriers (if gfx950 has them for this use), double-buffered tile state so one group can run a phase ahead without rendezvous, or a single merged window per tile with only the ring-recycle barrier (the earlier `_vsm` merged form, 287 TFLOP/s, predates the exact-reference recipe, smfmac row sums and resident Q and is untried with today's ingredients); (c) an explicitly authored inner loop (`Stmt::Asm`) if the list scheduler reorders the window.
4. `fmha_fwd_hd128_mxfp8_g2` stays registered by name only (own backend, not in `kernel_names()`) as the starting point; `_vs128` (367 TFLOP/s, 1.40x from the vendor) is the fast path and `_t` (292, llvm-tail) the accurate default.

### 7.1 Merged single window (g3) result
See docs/status.md round 57 addendum: g3 343.3 TFLOP/s (-6.5%), bit-identical, WAIT_ANY 558 (rendezvous tax gone) but WAIT_INST_ANY 627 (dependency stalls back), wave cycles 1702 vs 1599. The single window trades the barrier rendezvous for dependency stalls; the complementary offset is what hides them. Final negative of the line.

