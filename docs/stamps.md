# Timing stamps (`FnBuilder::memtime_lo`) and the GEMM cycle split

## DSL

- `b.memtime_lo() -> ValueId`: the low 32 bits of the wave's `s_memtime` (core-clock ticks), a wave-uniform i32 value. Lowered to `s_memtime s[a:a+1]`, `s_waitcnt lgkmcnt(0)`, `s_mov_b32 d, s_a` (SMEM returns out of order, so the wait is a full lgkm drain). Opt-in: a kernel without stamps compiles to the same instructions as before (FMHA and GEMM md5s unchanged). Put stamps where the kernel already drains (right after a tile's `vmcnt(0) lgkmcnt(0)` + barrier). Cost measured on `probe_stamp`: 44 cycles back to back; inside the GEMM loop a stamp (memtime + drain + address VALU + `global_store_dword`) costs about 118 cycles per tile.
- `b.stamp_rec_off(wg, slots, waves_per_wg) -> ValueId`: byte offset of this wave's record in a debug buffer laid out `u32 [workgroup][wave][slot]`; `wg` is the linear workgroup index.
- `b.stamp_store(buf_ptr, rec_off, slot, value)`: stores the stamp (or any i32) at `rec_off + 4 * slot`.
- The debug buffer is an ordinary pointer argument; declare it with `b.arg` and a tensor extent (`b.arg_tensor`) so the verifier checks the stores, or use `assume_trusted` on the record offset for a debug-only build (the GEMM build does).
- Scheduler: `s_memtime` is an SMEM op with `Fence::Barrier` (pinned against memory ops by the list scheduler); the wait scoreboard tracks its result (`sched/waits.rs`). The `ldshoist` option may still move an LDS read across a stamp in the Gemm profile.
- The `s_memtime` counters of different CUs / XCDs are not synchronised: compare stamps of the same wave (or the same CU), never across workgroups. `hwreg(29)` is not a cycle counter on gfx950 (reads 0).

## Host

`tests/gpu/stamp_host.h` (`StampBuffer`: allocate, read, `summary()`, `dump()`), `tests/gpu/stamp_probe_host.cpp` (runs `probe_stamp`), `tools/stamps.py` (analysis of a dump: per-slot intervals, `--phases`, `--tiles FIRST:LAST` steady state / ramp / bubble tiles / wave spread; `python3 -m unittest tools/test_stamps.py`).

## GEMM instrumentation (`GEMM_STAMP=ends|tiles`)

`GEMM_STAMP=ends` (build time env of `amdc emit`, any `gemm_mxfp8_bs_*` name): one extra pointer argument `dbg_stamps` at kernarg offset 96 (kernarg size 104) and 5 stamps per wave: slot 0 kernel start, 1 setup done (pipeline entry), 2 prologue done (tile 0 fragments requested, loop entry), 3 tile loop + peeled last tile done, 4 epilogue stores issued. `GEMM_STAMP=tiles` adds slot 16 + t at the matrix phase of every tile t (after the tile's top wait + barrier and DMA issue). `gemm_ref_host` with `STAMPS=1` passes the buffer through the hand ABI and writes the last launch's records to `STAMP_DUMP`. `tools/stamps_gemm.sh` builds and runs a list of kernels in both modes and prints the split.

## Result (GPU 4, MI355X, 4096x4096x8192, 256x256 tile, ~2.0-2.1 GHz sustained, cycles per wave, median of 1024 waves, two runs agree within 0.5 %)

| kernel | setup | prologue | tile loop | epilogue | in-kernel total | loop per tile |
|---|---|---|---|---|---|---|
| v4 (ends) | 1.9 k | 6.3 k | 153.5 k | 6.3 k | 168.0 k | 2395 |
| v4 (tiles) | 1.9 k | 6.2 k | 161.0 k | 6.2 k | 175.3 k | 2516 |
| v3 | 1.9 k | 4.9 k | 170.5 k | 6.3 k | 183.7 k | 2665 |
| v3 nodma + nords (MMA, scales, sync only) | 1.4 k | 3.1 k | 163.8 k | 6.2 k | 174.6 k | 2560 |
| v3 nodma | 1.6 k | 3.0 k | 200.7 k | 6.5 k | 211.8 k | 3136 |
| v3 nords | 1.8 k | 5.0 k | 173.8 k | 6.2 k | 186.8 k | 2715 |
| v3 nomma (no MFMAs) | 1.9 k | 4.9 k | 147.2 k | 6.2 k | 160.3 k | 2300 |

Reference points: MFMA pipe floor 64 MFMAs x 32.38 cycles = 2072 cycles per tile (+8..30 for the per-tile sync) = 133-135 k per wave; kernel wall time of v4 (0.0883 ms at about 2.09 GHz) is about 184 k cycles, so about 16 k cycles lie outside the stamped window (wave launch, program exit, workgroup tail).

What it says:
1. v4's 168 k in-kernel cycles = 134 k MFMA floor + **19.5 k loop bubbles (about 305 cycles per tile, 12.7 % of the loop)** + 6.3 k prologue + 1.9 k setup + 6.3 k epilogue. The first and the last tile equal the steady tile (1.00x / 1.01x); no tile is slower than 1.15x the steady state; the wave-to-wave spread inside a tile is 52 cycles (median): the bubbles are steady-state and per tile, not ramp or skew.
2. **The non-MFMA stream alone is longer than the MFMA floor**: with every MFMA removed (`nomma`) the loop still needs 2300 cycles per tile (DMA issue, LDS reads, scale reads and selects, waits, barriers), above the 2072-cycle MFMA floor. v4's 2395 per tile is only about 100 above that stream floor. Getting below about 2300 cycles per tile needs a cheaper data-movement stream, not better overlap.
3. MMA + scales + sync with DMA and operand reads removed (`nodma_nords`) still costs 2560 per tile: 490 cycles over the MFMA floor remain in the scale path, VALU and waits of the v3 structure.
4. Removing only the DMA (`nodma`) makes v3 slower (3136 per tile), so ablations that remove a stream are not additive.
5. Stamps in the loop cost 118 cycles per tile (`tiles` vs `ends`): use `ends` for totals and `tiles` only for the tile-to-tile shape.
6. Prologue + setup + epilogue = 14.5 k cycles (v4), 8.6 % of the in-kernel time.