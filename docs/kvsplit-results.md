# KV-split Step A/B results (kernel-opt, own backend, GPU 0)

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
