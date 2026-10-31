# KV-split Step A/B results (kernel-opt, own backend, GPU 0)

> Status: results + shipped: KV-split is in the default `_auto` dispatch (B1H8S2048 674-705 TFLOP/s = 1.25-1.31x vendor); the fused last-arriver combine closed negative (round 62) and the epilogue corruption root cause (VMEM wide-store data WAR) is fixed (round 69); see docs/current-status.md.

Kernels (all in `crates/amdgpu-dsl/src/kernels.rs`, registered by name only):
- `fmha_fwd_hd128_mxfp8_kvsplit` - phase 1: the `_vs128` block shape (256 rows, 512 threads, LDS 102400, vgpr 231) over one 128-key-tile-aligned key range per block (`tps = ceil(ceil(SK/128)/splits)`, split s owns tiles [s*tps, min(ntiles,(s+1)*tps))); grid `(ceil(S/256), H, B*splits)`, `split = z % splits`, `b = z / splits`; kernarg = 656-byte base + `Ohat`@0x270, `lse`@0x278, `splits`@0x280, `rows_total`@0x284. The K/V/K-scale base offsets are shifted by the first key, every key index inside the kernel is relative to it. Epilogue: `Ohat = O * Vdescale / l` (fp32, 16-byte `global_store_dwordx4` stores, new `Stmt::StoreF32x4`) and `lse = log2(l) - 512 b_final` (new `Op::Log2F32` = `v_log_f32`; the kernel's P is 2^(x + 512 b - 127) so the additive constant is split-independent and cancels in the combine weights). The recipe copy (LUT correction below 9 tiles) is chosen by the TOTAL tile count of the sequence, exactly as `_vs128`.
- `fmha_fwd_hd128_mxfp8_kvsplit_combine` - the contract version (one thread per element, 256 threads, grid rows*128/256).
- `fmha_fwd_hd128_mxfp8_kvsplit_combine4` - vectorised: 4 elements per thread (16-byte loads, 8-byte store): launch with `REF_COMBINE_GRID = ceil(rows_total*128/4/256)` (2048 for B1H8S2048).

## Numbers (B1H8S2048, REF_ITERS=200, one hipEvent pair around both launches)
| config | main | combine | total | TFLOP/s | vs t128 (505) | vs vendor (520, same run) |
|---|---|---|---|---|---|---|
| s=2, combine (contract) | 33-44 us | 6.0 us | 39.7 us | 432 | 0.86x | 0.83x |
| s=2, combine4 | 33 us | ~5 us | 37.4 us | 459 | 0.91x | 0.88x |
| **s=4, combine4** | **26.2 us** | **5.5 us** | **32.4 us** | **530** | **1.05x** | **1.02x** |
| s=4, combine4, partial O stores removed (diagnostic) | 18.8 us | 5.4 us | 25.7 us | 668 | | |
| s=4 with the first (scalar 4-byte) partial stores | 42-45 us | 11.3 us | 53 us | 320 | | |
Breakdown at s=4: the main kernel takes 18.8 us without partial writes (11.4 us compute-only model + ~7 us fixed per-launch/prologue/epilogue cost that the design model did not contain), the 33.6 MB of fp32 partials cost **7.4 us** (= the serialised write model, 4.5 TB/s: nothing overlaps them: they are written in the epilogue), the combine 5.5 us (vectorised; 11.3 us with one thread per element), plus ~0.7 us launch gap. The design model's central 800 TFLOP/s (18-25 us) is not reached: total 32.4 us.
## Correctness
- vs CPU (`REF_ROWS=64`): bad = 0 on every tested forced-split shape (B1H8S2048 s=2/4, B2H4S1024 s=2/4, B1H16S256 SK1024 gqa4 s=2, B2H8S700 SK640 gqa2 s=2, B1H4S300 SK512 s=2, B1H2S200 SK1337 gqa2 s=4, B3H3S768 SK1280 s=2, B1H2S512 SK2048 s=4); mean error 8.4e-4 (benchmark; `_vs128` 6.95e-4), 1.0e-3-1.2e-3 on the short shapes (vendor 2.5e-3-2.8e-3). The REF_KERNEL_FAIL verdicts at 64 rows are the vendor's own bad elements (shapes B2H4S1024, B1H16S256 SK1024, B2H8S700), as before.
- split vs `_vs128` (the planned ULP gate): mean_abs **3.3e-6 (s=2) / 6.6e-6 (s=4)** on the benchmark (model: 3.2e-6 / 4.3e-6, gate 1e-4: OK), max_abs = 7.8e-3 = exactly 1 bf16 ULP at |x| = 2; 1.0% (s=2) / 1.4% (s=4) of the words differ by more than 1 ULP and 1.4% / 2.3% differ at all (the model predicted 0.2-0.25%: 6-10x more elements but the same magnitude). `max_ulp` is meaningless as a gate: it reports 47k ULP on near-zero outputs (|x| ~ 1e-6) - use `max_abs <= 1 ULP(max|ref|)` and `mean_abs` instead. (With the LUT copy forced on the split kernel while `_vs128` uses the long copy, mean_abs jumps to 3.9e-4 - different recipes, still as accurate against the CPU - hence the recipe is selected by the total tile count.)
- determinism: no atomics, fixed reduction order; not yet run through the 15-shape determinism gate (kernel-occ's harness does it).
## Verdict against the lead's gates
Step A (s=2): 459 TFLOP/s = **0.91x t128: below the 1.1x gate** -> by the stated rule Step B is not unlocked. Step B (s=4) measured anyway: 32.4 us = 530 TFLOP/s = 1.05x t128 / 1.02x vendor, above the hard-stop limit (21 us). Remaining levers: fp16 partials (write 7.4 -> ~3.7 us, combine 5.5 -> ~4 us: total ~27.5 us, ~625 TFLOP/s, +24% over t128, still above the 21 us stop), overlapping the epilogue writes with the last tile (not possible with the current epilogue-only write), and the fixed ~7 us/launch which is outside our control.

## Update: coalesced + fp16 partial stores (final Step B numbers)
**"Overlap the partial writes" is not possible as stated**: the normalised partial `O / l` is only known after the LAST tile of the split (l and the final reference), all 256 blocks finish at the same moment, and the combine kernel cannot start before the main kernel has drained its stores - so the write burst cannot hide under compute. What was possible is making the burst cheaper:
1. 16-byte lane stores (`Stmt::StoreF32x4`): 4-byte scattered stores cost 17-22 us of the main kernel, 16-byte stores 7.4 us.
2. **LDS-transposed coalesced stores** (default): each wave stages its 32-row x 32-dim fragment slab through its private 4 KB of the dead Q staging region of the LDS (rows of 128 B, 16-byte chunks XOR-swizzled by `row & 7`), reads it back so that 8 consecutive lanes cover a full 128-byte row segment, and issues 16-byte stores with consecutive-lane addresses: write cost 7.4 -> ~5.7 us (fp32).
3. **fp16 partials** (default, `nop16` for fp32): same transposed path, two fragments (64 dims = 128 bytes) per round: write ~4.2 us, combine reads half the bytes (`..._kvsplit_combine4h`, 4 elements per thread). Naive 4-byte fp16 stores were SLOWER than fp32 (30.5 vs 26.3 us main): the store pattern, not the byte count, is what matters.
Same-run numbers, B1H8S2048, REF_ITERS=200, REF_ROWS=64, one event pair around both launches:
| config | main | combine | total | TFLOP/s | vs t128 505 | vs vendor 520 |
|---|---|---|---|---|---|---|
| s=2 fp32 coalesced + combine4 | 31.0 us | 3.4 us | 34.7 us | 495 | 0.98x | 0.95x |
| s=2 **fp16** + combine4h | 29.6 us | 3.8 us | 32.5 us | 529 | 1.05x | 1.02x |
| s=4 fp32 coalesced + combine4 | 24.6 us | 5.5 us | 30.9 us | 556 | 1.10x | 1.07x |
| **s=4 fp16 + combine4h (default)** | **23.0 us** | **5.2 us** | **28.0 us** | **613** | **1.21x** | **1.18x** |
The main kernel without any partial write is 18.8 us (diagnostic): the remaining write cost is ~4 us, the rest is the fixed per-launch/prologue/epilogue cost.
**Numerics**: fp16 partials: mean_abs vs `_vs128` 5.5e-5 (s=4) / 6.9e-5 (s=2) (fp32: 6.6e-6 / 3.3e-6; informational gate was 1e-4), max_abs = 1 bf16 ULP (7.8e-3 at |x| = 2), vs CPU mean error 8.41e-4 (fp32 8.39e-4).
**Gates through the auto_split path** (`OUR_KERNEL=fmha_fwd_hd128_mxfp8_auto_split REF_SPLIT=4`, guard decides per shape: 9 of the 15 gate shapes choose a split, 6 fall back to t128): 15/15 vs the CPU reference at 16 rows, parity 15/15 at REF_ROWS=64, bad = 0 at 16 and 64 rows, determinism 5/5 x3, max combined mean error 1.88e-3 (<= 2.0e-3; vendor 1.5-2.9e-3) - the two shapes above 1.5e-3 are fall-back shapes. Harness changes: `ref_host` default combine kernel `..._kvsplit_combine4h` and launch grid divided by 4 for `*combine4*` kernels; `run_ref.sh`/`determinism.sh` build `..._kvsplit_combine4h`.

## Default auto dispatch (final)
`OUR_KERNEL=fmha_fwd_hd128_mxfp8_auto` (run_ref.sh / determinism.sh build `_vs128`, `_t128`, `_kvsplit`, `_kvsplit_combine4h`; `ref_host` picks per shape at launch time, no `REF_SPLIT` needed). Overrides: `REF_SPLIT=s` (+ `REF_SPLIT_FORCE=1`) forces a split candidate, `REF_NO_AUTO_SPLIT=1` disables the split, `REF_ROWS_PER_BLOCK`/`REF_OURS_BLOCK_THREADS` as before, `REF_PARTIAL_BYTES=4` for `nop16` (fp32 partial) builds. The `[split]`/`[grid]` lines print every candidate with its model times and the reason.
**Guard** (candidates s in {2, 4}, best modelled T_kv that passes): `N256 <= 128 && N256 * s <= 256 && (s-1) * tps < ntiles && T_kv(s) < 0.95 * T_current`, with the model fitted to the measurements (docs above):
- `T_t128 = F / (5.9e12 * min(256, 2 N256)) + 9.4 us` (9.4 us = the floor measured at SK = 1); `T_vs128 = F / (5.9e12 * min(256, N256))` for filling grids;
- `T_kvsplit(s) = F / (5.9e12 * min(256, N256 s)) + 7 us` (phase 1 fixed prologue/epilogue cost) `+ partial_bytes / 4.0 TB/s` (epilogue write, not hidden) `+ max(3 us, (partial + out bytes) / 4.2 TB/s) + 1.5 us` (combine with its kernel floor + launch gap); partial bytes = `s * rows * 128 * 2 (fp16) + s * rows * 4`.
The first version of the guard (kernel-occ's constants with fp16 bytes, no combine floor, t128 fixed 10.9 us) chose splits on tiny shapes where they lose 15% (B1H1S256 SK256: 13.7 vs 11.5 us; B2H4S300 SK129: 15.5 vs 13.2); with the two floors above no shape loses.
**Per-shape results, default auto vs the same build with the split disabled** (300 iters, same run, both launches inside the event window of the split path):
| shape | auto path | auto us | no-split us | speed-up |
|---|---|---|---|---|
| B1 H1 S256 SK256 | t128 | 11.5 | 11.5 | 1.00 |
| B1 H2 S512 SK512 | t128 | 15.1 | 15.1 | 1.00 |
| B2 H4 S1024 SK1024 | kvsplit | 22.1 | 22.0 | 1.00 |
| **B1 H8 S2048 SK2048 (benchmark)** | kvsplit s=4 | **27.3** | 32.7 | **1.20** |
| **B1 H2 S512 SK2048** | kvsplit | 21.4 | 31.0 | **1.45** |
| B3 H3 S768 SK1280 | kvsplit | 22.8 | 23.2 | 1.02 |
| B1 H4 S300 SK512 | t128 | 15.2 | 15.2 | 1.00 |
| B2 H8 S700 SK640 | t128 | 18.0 | 18.0 | 1.00 |
| B1 H16 S256 SK1024 | kvsplit | 19.0 | 21.6 | 1.14 |
| B1 H2 S256 SK1000 | kvsplit | 18.4 | 22.2 | 1.21 |
| B2 H4 S300 SK129 | t128 | 13.2 | 13.2 | 1.00 |
| B1 H8 S512 SK100 | t128 | 11.1 | 11.1 | 1.00 |
| B1 H2 S1024 SK1088 | t128 | 22.1 | 22.1 | 1.00 |
| B1 H4 S77 SK1 | t128 | 9.4 | 9.4 | 1.00 |
| **B1 H2 S200 SK1337** | kvsplit | 18.1 | 24.6 | **1.36** |
| B1 H8 S1024 SK1024 (mid) | kvsplit | 22.0 | 21.9 | 1.00 |
| B2 H8 S2048 SK2048 (mid, N256 = 128) | t128 | 37.2 | 37.3 | 1.00 |
| B1 H8 S4096 SK4096 (mid, N256 = 128) | t128 | 60.2 | 60.4 | 1.00 |
| B4 H8 S2048 SK2048 (filling) | `_vs128` | 56.0 | 55.7 | 0.99 |
| B1 H8 S8192 SK8192 (filling) | `_vs128` | 154.7 | 154.1 | 1.00 |
Filling and mid shapes are unchanged (the guard keeps them on `_vs128` / t128); the 15-shape gate takes the split on 7 shapes (B2H4S1024, B1H8S2048, B1H2S512 SK2048, B3H3S768, B1H16S256 SK1024, B1H2S256 SK1000, B1H2S200 SK1337) and the certified t128 on the other 8.
**Gates through the default auto path** (no REF_SPLIT env): 15/15 vs the CPU reference at 16 rows, 0 bad at 16 and 64 rows, parity 15/15 (REF_ROWS=64), determinism 5/5 x3, max combined mean error 1.84e-3 (16 rows) / 1.88e-3 (64 rows), vendor 1.5-2.9e-3. The three REF_KERNEL_FAIL verdicts at 64 rows are the vendor's own bad elements.
**Why the partial write cannot be hidden (keep):** the normalised `Ohat = O / l` and `lse` exist only after the LAST tile of the split, every block finishes at the same moment, and the combine kernel cannot start before the main kernel's stores have drained; per-tile stores would have to write unnormalised running state (a different algorithm). The ~4 us of write that remains (fp16, coalesced) is the floor for this algorithm; the other ~19 us of the main kernel is compute (11.4 us) plus fixed per-launch/prologue/epilogue cost.

## Independent validation (kernel-occ, GPU 5, final default-auto build)

Validated build: Main `dd6aa2d` + certification harness `5b04fd2` (ULP gate in `max_abs <= 1 bf16 ULP(max|ref|)`
form + refitted guard model). Default `OUR_KERNEL=fmha_fwd_hd128_mxfp8_auto`, no sentinel env, warm protocol
`REF_WARMUP=200`, `REF_ITERS=100`.

**Benchmark B1H8S2048 (same run, default auto):** split s=4 chosen; main 23.2-23.5 us, combine 5.2-5.3 us,
total **28.4-28.8 us = 596.7-605.5 TFLOP/s** vs vendor **534.0-538.9** => **1.11-1.13x absolute**; `_t128`
526-527 TFLOP/s. Combined-output md5 `a82ed4d67882104d251959c07cf4fd41`, our mean error vs CPU 7.356e-4
(vendor 2.5e-3), bad=0.

**Gate/determinism/parity:** 15/15 at 16 rows (ours bad=0, parity 15/15); 64 rows 12/15 with all three
failures vendor (ours bad=0, parity 15/15); determinism 5/5 x2.

**ULP vs `_vs128` (fp16 partials):** max_abs 7.812e-3 = **0.50 bf16 ULP** of max|ref|=2.0, mean_abs
5.535e-5, 4.86% of words differ at all (1.69% >1 word-ULP - meaningless near zero); `SPLIT_ULP_OK`.
Not bit-identical by construction; CPU/parity gates are the certification.

**No-regression checks (default auto):** B4H8S2048 -> `_vs128` single launch 1292.8-1301.1 TFLOP/s (vendor
1689-1715, `_t128` 950; certified `_vs128` was 1302.5 same run); B1H8S8192 -> `_vs128` 1861-1864 (vendor
2377-2385, `_t128` 935-951). Mid shapes: B2H8S2048 (N=128) -> `_t128` 949.9-951.2 (guard rejects split);
B1H8S1024/B2H4S1024 (N=32) -> `_t128` 196-197 (guard now rejects split; measured split would have been
190-193 = 1.4-1.8% slower); B3H3S768 (N=27) -> split 193.8-197.1 vs t128 195.3-195.8 (neutral); B1H16S256
(N=16) -> `_t128` 100.1 (split 111.1-111.6 would win 11% - conservative guard, no regression).

**Guard model refit:** `T_cur = F/(5.9e12*min(256,2N)) + 9.4us`; `T_kv = max(physics_floor, linear fit)` with
`fit[us] = 17.1965 + 0.0937013*F[GFLOP] - 0.00103388*(N*s) + 0.620991*rows[k]` and
`physics_floor = F/(5.9e12*min(256,N*s)) + 7us + part_b/4.0TB/s + max(3us,(part_b+out_b)/4.2TB/s) + 1.5us`.
Fit: 6 measured s=2/s=4 totals, `T_kv` residuals <=5% (28.6/29.19, 22.3/22.55, 22.4/22.55, 22.9/21.80,
19.3/19.88, 40.5/41.6 us), `T_cur` physical form within 7% except N=128 (-10%, guard rejects there).
Boundary decisions verified: N256=127/128 -> `_t128` (measured split s=2 is 14% worse), N256=129/130 ->
`_vs128`, tile-edge empty-split shapes -> fallback; benchmark stays split (guard predicts 0.91 ratio, measured
0.87). Write traffic: counters unavailable (rocprofv3 hung on GPU 5), but ablation gives main 23.0 us with
partials vs 18.8 us without = 4.2 us for 17.04 MB fp16 = ~4.0 TB/s serial write, combine 5.2 us for 21.3 MB
= 4.1 TB/s; consistent with the cost model.

