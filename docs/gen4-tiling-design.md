# Generation 4: 128-row blocks (tiling change) - design

Status: design (kernel-opt). Baseline `fmha_fwd_hd128_mxfp8_vs128` 367 TFLOP/s on B1H8S2048 (own backend).
Origin: kernel-occ §7.9: the grid is only 25% full. `ref_host` launches `(ceil(S/256), H, B)` blocks of 512 threads (8 waves, 32 query rows per wave, 256 rows per block); B1H8S2048 = 8 x 8 x 1 = **64 blocks on 256 CUs** (none of the 15 shapes fills the GPU). 367 vs 515 TFLOP/s is a per-CU ratio (5.7 vs 8.05 TFLOP/s/CU).

## 1. The key arithmetic (and a correction to the 2-blocks-per-CU premise)
**What limits the absolute number on this shape is the number of busy SIMDs, not only the per-SIMD efficiency.** B1H8S2048 has 2048 x 8 = 16384 query rows. With 32 rows per wave (the MFMA 32x32 geometry; a wave cannot own fewer than 32 rows) at most 16384 / 32 = **512 waves** are useful = 50% of the 1024 SIMDs of the GPU even in the best tiling. Today: 64 blocks x 8 waves = 512 waves on 64 CUs x 4 SIMDs = 256 SIMDs, TWO waves per SIMD.
**Sum-of-parts means a second wave per SIMD buys nothing in this kernel.** The measured wall per tile (5.37k cycles per SIMD for the two resident waves) equals 2 x (V + M) = 2 x (1.15k + 1.5k): the partner wave only time-slices. A single wave on a SIMD therefore processes its tile in ~2.65k (V + M, upper bound; less if V and M overlap inside the wave) and delivers the same rows per cycle per SIMD as the pair does now. So moving from 2 waves/SIMD on 64 CUs to **1 wave/SIMD on 128 CUs (128-row blocks of 4 waves, one block per CU)** keeps the per-SIMD rate and doubles the busy SIMDs: the same 16384 rows per tile-step finish in T = 2.65k instead of 5.37k cycles.
**The 2-blocks-per-CU packing (lead's premise) is the opposite of what we want.** If the dispatcher placed two 4-wave blocks on one CU (2 waves per SIMD) the 128 blocks would occupy only 64 CUs and, by the sum-of-parts argument, give no gain (~367). Hardware dispatch normally spreads blocks over CUs before doubling up, but the safe way to force it is the LDS budget: a block that needs > 81920 B of LDS (e.g. the 3-slot ring, ~100 KB) cannot share a CU. So the design uses the 3-slot ring (LDS 99840 B) deliberately, not the 2-slot ring (65 KB); the 2-slot form is kept as an experiment (it may co-reside; if the blocks free-run in complementary phases it could even beat 1 wave/SIMD, see §4).
The speedup is independent of the block count: S = 5.37k / T where T is the single-wave tile time.

## 2. Predicted absolute TFLOP/s on B1H8S2048
| single-wave tile time T | assumption | speed-up vs 5.37k | TFLOP/s |
|---|---|---|---|
| 3.5k | latency exposed (no partner to hide LDS/DMA/MFMA dependencies; conservative) | 1.53x | ~560 |
| 2.65k | V + M sequential (measured per-wave skeleton times) | 2.03x | ~745 |
| 2.2k | partial in-wave overlap of the merged window (g3-like) | 2.4x | ~890 |
| 1.9k | the probe's 66% intra-wave overlap (upper bound) | 2.8x | ~1030 (not credible) |
**Central estimate ~700 TFLOP/s (1.9x), range 560-890**, i.e. above the vendor's 515 ONLY because the vendor kernel keeps its 256-row ABI tile on the same under-filled grid (the vendor would gain the same from the same tiling). The gain is a utilisation gain on a small shape; for shapes with >= 256 blocks (large S x H x B) the new tiling gains nothing and the kernel must not lose: per-SIMD rate is unchanged (sum-of-parts), K/V traffic per FLOP doubles (section 5).
Bound from the other side: 50% SIMD fill is the maximum for this shape (512 waves of 32 rows); further fill needs sequence-parallel splitting of K/V across blocks with a cross-block softmax merge (an algorithm change, not part of this design).

## 3. Kernel design (`fmha_fwd_hd128_mxfp8_t128`)
- 256 threads = 4 waves = ONE wave group; rows per block 128 (wave w owns rows 128 bx + 32 w ... +31); GQA/head mapping, q tail masking and the exact-reference/vsx recipes unchanged (row-independent: the output must be **bit-identical to `_vs128`** row by row).
- Kernarg ABI unchanged (`ts_qo` stays 256 in the ABI); the kernel derives its 128-row tile from `blockIdx.x`; the HOST launches `ceil(S/128)` blocks per (H, B) - `ref_host` gets `REF_GRID128=1` (default for `t128`); the vendor `.co` keeps its own grid.
- Structure: the merged single-window form of `g3` (max/rollback -> convert + QK(t+1) into the freed S registers -> PV + smfmac) with ONE block barrier per tile (4 waves). There are no complementary groups inside a block (and none are needed: 1 wave/SIMD). Each wave stages BOTH its K rows (32 w ... +31, 4 DMA per tile) and its V rows (4 DMA per tile) because there is no second group; K scales via the wave's own lane loads.
- Ring: 3 slots x 32768 B + kd 1536 = 99840 B of live data; the emitted kernel declares **135168 B** (see §10). VGPR as `g3` (254) - one wave per SIMD has 512 available, unused for now.
- Peeled masked tail, short-sequence LUT copy (nkv < 9), exact reference: copied from `_vs128`.

## 4. Phase options if two blocks co-reside (2-slot variant, LDS <= 81920 B)
(a) free-run: both waves on a SIMD run identical code; the g3 free-running ablation (302.7) shows drift can be destructive, but those were two groups synchronised by the same block; here the blocks are fully independent (no barrier coupling) - measure. (b) shift by `blockIdx & 1`: odd blocks start with an extra half-tile prologue (V first, M later) - no deadlock risk because blocks only rendezvous within themselves. Expected value: if the sum-of-parts result holds, co-residence gives nothing and 1 wave/SIMD (3-slot, one block per CU) is the target; (a)/(b) are experiments only.

## 5. K/V traffic and LDS
Per block per 128 keys: K 16 KB + V 16 KB = 32 KB, read once from L2 into LDS by the DMA (each wave 8 KB). Today a 256-row block loads this once per 8 waves; the new tiling loads it once per 4 waves: traffic per FLOP **x2**. Absolute: 128 blocks x 32 KB per T (2.65k cycles = 1.1 us) = 3.7 TB/s total vs ~0.9 TB/s today (64 blocks / 2.24 us), i.e. 3.7 TB/s over 8 XCDs = 0.46 TB/s per XCD L2 (L2 peak ~4 TB/s per XCD: ~12%) and 29 GB/s per CU (CU fabric ~150 GB/s: ~20%); the working set is the K/V of 8 heads = 4 MB unique (fits the L2/MALL). Not a limiter on this shape; for large shapes the doubled traffic per FLOP needs a check (large S: K/V per head 512 KB x ... still L2-resident per head).
LDS per block 99840 B (1 block/CU); the Q staging region is gone (Q resident). LDS instructions per wave-iteration unchanged (52.8): the LDS read traffic per CU halves (4 waves) while the time per tile halves: same bandwidth per cycle.

## 6. Risks
(1) The dispatcher co-residing two blocks (guarded by the LDS size); (2) single wave per SIMD exposes dependency stalls: the single-wave tile time may exceed 2.65k (conservative row of the table; this is the number to measure first with `ablate` runs of the prototype); (3) L2 traffic doubling for large shapes; (4) correctness at small shapes (rows beyond S, SK < tile), all 15 shapes need the new grid; (5) the benchmark shape gains from underfill; the other shapes (S = 256..2048, H <= 16) have the same under-filled grids (max 25% today) and gain similarly, but a fully-filled large shape must be checked for no regression (it should be equal: per-SIMD rate unchanged).

## 7. Plan
1. Host: `ref_host` grid option (`REF_GRID128`) and `ts_qo`-independent row mapping in the builder (rows = 128 bx + 32 w).
2. Kernel: `emit_fmha_t128_role` = `g3` role with a single group doing both staging roles; ring 3 slots; one barrier per tile.
3. Correctness (bit-identical md5 on 1 8 2048 2048 vs `_vs128`), 15 shapes, bad = 0 at 64 rows, determinism, parity.
4. Measure absolute TFLOP/s on all 15 shapes (B1H8S2048 first), per-CU rate, the single-wave tile time T by ablation, and a large filled shape for the no-regression check.

## 9. Prototype results (`fmha_fwd_hd128_mxfp8_t128`, own backend, by name only)
Kernel: 4 waves (256 threads) per block, 128 rows per block, `g3` merged window (one barrier per tile), 3-slot ring (LDS 135168 B: one block per CU), cooperative staging (wave w stages K rows [32w, 32w+32) and V rows [32w, 32w+32): the 4 waves cover the tile exactly once; 8 DMA per wave per tile, 32 per block = 32 KB, nothing redundant), short/long recipe copies like `_vs128`; host launches `ceil(S/128)` blocks of 256 threads (`ref_host` does it automatically for `*t128`).
**Scheduler fit (the decisive item).** The default lowering list made the full list scheduler fall back to the unscheduled code ("out of VGPRs allocating r2040 (V8)": eight V16 loop-carried S/O tiles + five V8 P/Q fragments + 54 single registers = 252 of 256): 416.5 TFLOP/s. Adding the `addrfold` lowering option (DMA address folding; `_t128` only) makes it fit (vgpr 252) with the full scheduler: **494.1**. The `ld` preset (no reordering) gave 456.8-464.1. Defaults for `_t128` in `lib.rs`: lowering + `addrfold`, full list scheduler.
**B1H8S2048 (64 blocks of 256 rows today, 128 blocks of 128 rows)**: `_vs128` 367.8 -> **t128 484.8-494.1 TFLOP/s (1.32-1.34x)**; output bit-identical (md5 faadf00c); T = (0.0348 ms - 0.011 ms) x 2.4e9 / 16 = **3.57k cycles per tile** (4.5k with the fallback code, 3.9k with `ld`). Per-CU: 494 / 128 = 3.8 TFLOP/s/CU vs 5.75 for `_vs128` (0.66x) vs the vendor 8.05; absolute 494 vs the vendor's 515 (0.96x) on a grid where the vendor uses 25% of the CUs.
**Gates (default recipe, vsx two-copy)**: 15/15 vs the CPU reference at 16 rows, 0 bad on all 15 shapes at 16 and 64 rows (the 3 REF_KERNEL_FAIL verdicts at 64 rows are the vendor's own 1/4/5 bad elements), parity 15/15 at REF_ROWS=64, determinism 5/5 x3, output bit-identical to `_vs128` on 1 8 2048 2048 and 4 8 2048 2048.
**Filling shapes.** Sweep, absolute TFLOP/s (`_vs128` | `t128` | ratio), blocks of 256 rows / 128 rows:
| shape (B H S) | blocks 256 / 128 | `_vs128` | `t128` | t128 / vs128 |
|---|---|---|---|---|
| 1 8 1024 | 32 / 64 | 127.9 | 191.9 | 1.50 |
| 1 8 2048 | 64 / 128 | 367.8 | 484.8 | 1.32 |
| 2 8 2048 | 128 / 256 | 700.0 | 870.0 | 1.24 |
| 1 16 2048 | 128 / 256 | 696.0 | 865.1 | 1.24 |
| 1 8 4096 | 128 / 256 | 838.2 | 1034.4 | 1.23 |
| 4 8 2048 | 256 / 512 | 1215.4 | 905.3 | **0.74** |
| 1 32 2048 | 256 / 512 | 1202.1 | 899.6 | 0.75 |
| 1 8 8192 | 256 / 512 | 1546.9 | 1158.8 | 0.75 |
| 8 8 2048 | 512 / 1024 | 1219.9 | 932.2 | 0.76 |
| 2 16 4096 | 512 / 1024 | 1440.6 | 1079.0 | 0.75 |
| 4 16 4096 | 1024 / 2048 | 1462.6 | 1093.0 | 0.75 |
Reading: t128 wins 1.23-1.5x while the 256-row grid has at most 128 blocks (<= 50% of the CUs) and loses 25% as soon as the 256-row grid fills the GPU (>= 256 blocks), where the per-SIMD rate of the 2-waves-per-SIMD kernel is higher (per CU: `_vs128` ~5.7 vs `t128` ~4.3 TFLOP/s). A launcher rule follows (CORRECTED by kernel-occ's independent sweep, §10): use t128 iff **2 * ceil(S/256) * H * B <= 256** (N <= 128), `_vs128` otherwise (the kernels share the ABI; only the grid and block size differ).
**K/V L2 traffic** (rocprofv3, B1H8S2048): TCC_REQ_sum 666k (`_vs128`) -> 831k (t128), +25%; TCP_TCC_READ_REQ 318k -> 482k (+52%); TCC_HIT 461k -> 626k; HBM-bound reads (TCC_EA0_RDREQ) 173.5k -> 173.0k (identical: the unique K/V data); so the per-FLOP L2 request traffic is 1.25-1.5x (not 2x: Q, O and the K-scale traffic are per row) and the HBM traffic is unchanged. On B4H8S2048: TCC_REQ 2.64M -> 3.29M (+24%), EA reads 684k = 684k.
**Why the per-CU rate is lower**: with one wave per SIMD the dependency latency the partner used to hide is exposed (single-wave VALU-only skeleton 1.8k vs 1.15k per tile in the two-wave kernel; MFMA/LDS-only 1.8k vs 1.5k). Wall per tile 3.57k = ~1.8k V + 1.8k M with only partial overlap (the full scheduler fits and gives the QK(t+1) MFMAs room inside the convert VALU).
Next (not done): overlapping more of V and M inside the wave (the full scheduler fit is the first step; an explicit `Stmt::Asm` window is the second), and the K/V-split across blocks (sequence-parallel with a cross-block merge) that is the only way to have more than 512 waves on this shape (total waves = 16384 rows / 32 = 512 of 1024 SIMDs).

## 10. Corrections after kernel-occ's independent validation
1. **Dispatch threshold.** The measured crossover is where the 128-row grid (2N blocks, N = ceil(S/256)*H*B) stops fitting the 256 CUs: t128 / `_vs128` = 1.345 (N = 64), 1.33 (N = 128, 256 t128 blocks), then it collapses: N = 136 -> 0.41, 144 -> 0.51, 152 -> 0.67, 160 -> 0.49, 192 -> 0.73, 224 -> 0.72, 256 -> 0.64. The rule is **t128 iff 2N <= 256 (N <= 128)**, else `_vs128`; `ref_host` implements it (`[grid]` line: chosen tiling + reason; overrides `REF_ROWS_PER_BLOCK`, `REF_OURS_BLOCK_THREADS`, `REF_GRID128`). The earlier "N < 256" statement in §9 was wrong for N = 129..255 (the 15 gate shapes all have N <= 64 and take t128).
2. **LDS of the emitted kernel is 135168 B, not 99840 B.** The geometry (`fmha_t_build`, `vs_active()` kernels `_vsm/_g2/_g3/_t128`): 3 ring slots x 32768 B (K tile 16384 + V tile 16384 each) = 98304, + K-scale ring 3 x 512 = 1536 (declared as 2048), + 2048 B ones block + 16384 B Q b1 + 16384 B Q b0 staging regions inherited from the `g3` code = 3*32768 + 2048 + 2048 + 16384 + 16384 = **135168**. The ones/Q regions are only written (the Q fragments are register-resident) and could be dropped (99840 B); either size exceeds 81920 B, so two blocks cannot co-reside on a CU (the occupancy argument of §1 holds).
3. **Loop-variable lint**: `t128` now has the same SSA bias-word fix as `_vs128` (0 warnings, bit-identical output, 491.8 TFLOP/s); the new bias word is written to the loop variable at the end of the window (writing it early kept two copies of the bias live and pushed the kernel over the 256-VGPR limit: the full scheduler fell back, 410 TFLOP/s).

## 11. Dispatch guard for the KV-split candidate (design, harness implemented)

The KV-split prototype (`docs/kvsplit-design.md`) is a second way to spend the same waves: split the key
dimension `s` ways, keep the `_vs128` 8-wave block shape (one block/CU = 2 waves/SIMD), and merge the `s`
normalized partials (`Ohat` fp32 + `lse` fp32) with a small combine kernel. It only helps when it can turn an
under-filled grid into a full one, and its partial-write + combine cost is a fixed overhead, so it must not be
selected blindly. The harness (`tests/gpu/ref_host.cpp`) applies this guard (implemented, waiting for the Step A
kernel):

```
N256     = ceil(S/256) * H * B                      # blocks of the 256-row tiling
rows     = B * S * H                                # output rows (D = 128)
F        = 4 * B * H * S * SK * D                   # attention FLOPs
T_cur    = F / (4.03e12 * min(256, 2*N256))         # _t128 model (N256 <= 128), else
           F / (5.90e12 * min(256, N256))           # _vs128 model
part_b   = s*rows*D*4 + s*rows*4                    # Ohat (fp32) + lse write bytes
comb_b   = part_b + rows*D*2                        # combine read + bf16 write bytes
T_kv(s)  = max(F / (5.90e12 * min(256, N256*s)),    # phase-1 compute at the measured 2-wave/SIMD rate
               part_b / 4.8e12)                     # measured partial-write BW (partially hidden)
         + comb_b / 7.4e12 + 1.5e-6                 # measured combine BW + launch gap
ntiles   = ceil(SK/128); tps = ceil(ntiles/s)
use_split = (N256 <= 128) && (N256*s <= 256) && ((s-1)*tps < ntiles) && (T_kv(s) < 0.95 * T_cur)
```

* `N256 <= 128` keeps the rule inside the regime where the 256-row grid is under-filled (the certified auto
  boundary: `t128` wins iff `2*N256 <= 256`).
* `N256*s <= 256` requires the split grid to fit **one** block wave; two block waves give no extra CU fill and
  only add traffic (measured `_t128` collapse at 2N > 256, `docs/vendor-spec.md` §7.9.5c).
* `T_kv < 0.95*T_cur` keeps a 5% safety margin and prevents the fixed ~5 us combine from eating tiny shapes.
* The constants are the measured rates from `docs/kvsplit-design.md` (5.9/4.03 TFLOP/s/CU, 4.8 TB/s write,
  7.4 TB/s combine); they are model inputs, to be re-calibrated after the Step A measurement.
* Overrides: `REF_SPLIT=s`, `REF_CO2_SPLIT`/`REF_KNAME2_SPLIT`, `REF_CO2_COMBINE`/`REF_KNAME2_COMBINE`,
  `REF_SPLIT_FORCE=1` (bypasses the guard for the gate), `REF_SPLIT_ULP`, `REF_COMBINE_THREADS/GRID`.
  `REF_SPLIT` unset -> the certified `_auto` path is byte-unchanged.
* Interaction with the certified dispatch: if the guard rejects the split candidate, `ref_host` falls back to
  the normal rule (`_t128` for `N256 <= 128`, `_vs128` for `N256 > 128`), so filling shapes keep the
  single-launch path and cannot regress.

