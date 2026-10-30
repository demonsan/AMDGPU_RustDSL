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
- Ring: 3 slots x 32768 B + kd 1536 = 99840 B (> 81920: one block per CU). VGPR as `g3` (254) - one wave per SIMD has 512 available, unused for now.
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
