# gfx950 (MI355X) scheduler microbenchmarks - measured results

Sources: `gen.py` (generates asm kernels `mb*.s`, `tests.json`, `groups.json`), `host.cpp` (hipModuleLaunchKernel harness),
`analyze.py` (table printer), `src.tgz` (the three files). Raw analyses: `run1_analysis.txt` (all kernels, waves 1/4/8, 3 reps),
`run3_analysis.txt` (wave sweeps 1/2/4/6/8), `an1.txt` (early run). Run on node 53, GPU 1, ROCm 7.0.2.

## Method / units
* Counter: `s_memtime` (+ `s_waitcnt lgkmcnt(0)` after each read). Calibrated against `s_memrealtime` (100 MHz) with a 3.2M-cycle s_nop loop:
  3168032 memtime ticks / 132680 realtime ticks => **2.39 GHz => memtime = shader-clock cycles**. (`HW_REG_SHADER_CYCLES` is not supported on gfx950.)
* Each kernel: init regs, `s_barrier`, t0, unrolled body (N=256 VALU / 32 MFMA / 64-128 LDS), final waitcnt, t1. Per-instr = (dt-48)/N; 48 cycles = measured fixed
  overhead of an empty timed region (`aa_dummy_first`). The final timestamp is taken after the last *issue* (MFMA/VALU completion is not waited for), so for
  MFMA the figure is issue-to-issue spacing.
* Chain execution checked by storing a result: add chain = 256.0 (0x43800000), MFMA chain = 2048.0 (32 x 64), 0x45000000; independent MFMA = 64.0 etc.
* waves: 1 wave = 64 threads; 2 waves = different SIMDs; 4 = 1/SIMD; 8 = 2 waves/SIMD (wave w -> SIMD w%4, confirmed via s_mov/MFMA serialization of waves w and w+4).
  Kernels use 256 VGPRs so >2 waves/SIMD is impossible (launch fails).
* "V" = 3 reps agree within 5% (rep 0 is often a cold-cache outlier, noted); "E" = experimental/noisy. Wait state = one `s_nop 0` slot = 4.0 cycles
  (s_nop 15 = 64.5 cycles); SALU/VALU/s_nop single-wave issue cadence is 4 cycles.

## 1. VALU (cycles per wave64 instruction)
| op | dep chain, 1 wave | indep, 1 wave | indep, 2 waves/SIMD (per-wave; SIMD aggregate) | Status |
|---|---|---|---|---|
| v_add_f32 | 4.0 | 4.0 | 4.4 each both waves (=> aggregate ~2.2/instr, **2x**) | V |
| v_mul_f32 | 4.0-4.4 | 4.0 | 4.4 both waves (2x) | V (dep E) |
| v_fma_f32 | 5.5-5.9 (noisy) | 4.0-5.8 (noisy) | 4.4 both waves (2x) | E at 1 wave, V at 2 |
| v_mov_b32 | 4.0 | 4.0-4.4 | 4.4 both waves (2x) | V |
| v_max3_f32 | 4.0 | 4.0 | 4.4 / 8.4 (serialized, 4.0 aggregate) | V |
| v_pk_add_f32 / v_pk_mul_f32 | 4.0 | 4.0-4.4 | 4.4 / 8.4 (serialized) | V |
| v_cvt_pk_bf16_f32 | 4.4-4.5 | 4.0 | 4.4 / 8.4 (serialized) | V |
| v_cndmask_b32 (VOP2, vcc) | 4.7-5.0 | 4.8-5.5 | **w=4: ~19-20 cycles!** (slow VCC path when several waves run) | V for w>=4 slowdown |
| v_cndmask_b32_e64 (SGPR pair mask) | 4.4 | 4.0 | 4.4 / 8.4 (serialized) | V -> use this form |
| v_cvt_pk_fp8_f32 | 8.3-9.1 | 8.0 | 8.5 / 16.5 (serialized, 8.0 aggregate) | V (dep E) |
| v_permlane32_swap_b32 | 8.0 | 8.0 | 8.5 / 16.5 (serialized) | V |
| v_exp_f32 | 8.2-9.8 | 8.0 | 8.5 / 16.5 (serialized, 8.0 aggregate) | V ind, E dep |

Findings: (a) a dependent chain costs no more than independent for add/mul/mov/pk/max3/bf16 => RAW latency is hidden by the 4-cycle issue
cadence (true latency <= 4 cycles; cannot be resolved lower with a single wave). cvt_pk_fp8/exp/bf16 show at most +0.4..+1 cycle for dependency.
(b) One wave issues at most one VALU per ~4 cycles. A 2nd wave on the same SIMD doubles throughput ONLY for plain fp32 add/mul/fma/mov (2 cycles/instr/SIMD);
packed fp32, max3, cvt_bf16, cndmask, etc. serialize (4 cycles/SIMD); fp8 cvt, permlane32_swap, exp are 8 cycles/SIMD (half rate), no gain from a 2nd wave.
(c) 4 waves (1/SIMD) per-wave cost is ~4.4 vs 4.0 at 1 wave (small shared-resource penalty).

## 2. Transcendental (v_exp_f32) mixed with v_add (1 exp + k independent adds per group, single wave)
Group time = 8 + 4k cycles exactly (k=0: 7.9, 1: 11.9, 2: 15.9, 4: 23.9, 8: 39.9) => **no overlap** of exp with adds in one wave: exp is an 8-cycle
issue slot, adds 4 each. With 2 waves/SIMD the totals are still ~serialized (k=4: 50.6 for the slower wave vs 48 predicted serial; k=8: 82.5 vs 80).
=> model v_exp as a half-rate (8 cycle) VALU op on the same pipe; the trans unit is not separately overlappable by VALU in these tests. Status V.

## 3. MFMA (cycles, issue-to-issue spacing, 1 wave)
| instruction | dep chain (C=D in place) | independent (different D, rot 2/4/8) | 2 waves/SIMD | Status |
|---|---|---|---|---|
| v_mfma_scale_f32_32x32x64_f8f6f4 | 64.0 | 64.0 | MFMA unit shared: one wave 64, other 128 (aggregate 64/MFMA) | V |
| v_mfma_f32_32x32x16_fp8_fp8 | 32.0 | 32.0 | serialized (aggregate 32) | V |
| v_mfma_f32_16x16x32_fp8_fp8 | 16.0 | 16.0 (one noisy first run 18-26) | serialized | V |
Back-to-back dependent accumulation (SrcC = previous D) has **no extra stall** (hardware forwarding). Throughput = 2048 flop/cycle/SIMD for the scale op.

### VALU under MFMA (1 MFMA : k independent v_add, group time in cycles; N=32 groups, rep>=1)
* scale32 (64-cycle MFMA): k=0 63.6; k=4 64.4; k=8 64.9; k=11 65.3; k=12 68.1; k=13 69.4; k=14 76.4; k=15 80.4; k=16 81.4. **VALU issues freely under a running MFMA; ~11-12 VALU are free**
  (knee 12-13, then ~+4.5/VALU; theoretical 15 = 64/4-1, ~3 slots lost). Same for dependent and independent MFMA chain. E (knee position +-1).
* 32x32x16 fp8 (32-cycle): k=0 32.5, k=2 32.75, k=4 33.0, k=8 37.4 (issue-bound 4.15/instr) => ~6-7 free VALU.
* 16x16x32 fp8 (16-cycle): k=2 17.25, k=4 22.5, k=8 37.4 => ~3 free VALU.
* Rule of thumb: free VALU per MFMA ~ MFMA_cycles/4 - 1 (scale32: ~12 measured). With two waves/SIMD the VALU of the other wave also overlaps
  (w=8,k=16: 164 cycles per pair-group vs 128 MFMA-only => only +36 for 32 VALU).

### MFMA -> VALU read of D (NO hardware interlock; stale data is read!)
Sentinel written to D[0]/D[last], MFMA with inline-0 C, `s_nop` (m wait states) then `v_mov` reads D. Value correct ("OK") from:
| MFMA | first D reg correct from | **last D reg correct from** (safe) |
|---|---|---|
| scale32 (64-cyc, 16 pass) | m=7 | **m=20** (19 still stale) |
| 32x32x16 fp8 (32-cyc, 8 pass) | m=4 | **m=11** (10 stale) |
| 16x16x32 fp8 (16-cyc, 4 pass) | m=5 | **m=7** (6 stale) |
m counts `s_nop` wait states (4 cycles each) between the MFMA and the reading VALU (assembler never inserts them). Matches/just exceeds ISA 18/10/6. Status V (deterministic over 3 waves-configs x 3 reps).
A dependent `v_mov` read directly after each MFMA ("depvalu_pair") shows no stall (pair=64.0) and returns the stale value, confirming no interlock.
VALU write of an MFMA A operand followed by MFMA reading it with 0..4 wait states: correct result in all cases (no wait needed in this test).

## 4. LDS (per wave, 1 wave unless noted; slope from N=64 vs N=128)
| op | issue throughput 1 wave | 2 waves (diff SIMDs) | 4 waves | 8 waves (CU aggregate) | latency issue->lgkmcnt(0), single op | Status |
|---|---|---|---|---|---|---|
| ds_read_b128 | 8.0 c/instr (128 B/clk) | 8.7 each | 16.6 each | aggregate 1 per ~4.0 c = ~250 B/clk/CU | ~52 (w8: 57) | V |
| ds_read_b32 | 4.0 | 4.7 each | 8.6 each | ~2.1 c/instr CU-wide | ~48-52 | V |
| ds_read_b64_tr_b8 | ~5.5-5.75 | 5.7 | 9.3 | ~2.1 c/instr CU-wide | ~49-53 | E (noisy 1 wave) |
| ds_write_b128 | ~26 (!) | 26.7 | 52.6 | ~13 c/instr CU-wide (~80 B/clk) | ~72-77 | V |
Reads: CU bandwidth saturates at ~2 waves' worth: aggregate ~4 cycles per 1 KB b128 read. Writes of b128 are ~3x slower than reads here (lane*16 B contiguous addresses, no bank conflict expected).

## 5. Global loads (1 KB/wave footprint, 4-64 KB buffer)
| measurement | cycles | Status |
|---|---|---|
| global_load_dwordx4 latency issue->vmcnt(0), L1-resident | ~108 (w=8: ~130) | V |
| global_load_dword latency, L1-resident | ~100 (first run 110) | V |
| latency with L1 thrashed (L2 hit) dword / dwordx4 | ~205-215 / ~230-250 | E (includes SALU address update subtracted ~24) |
| issue spacing of 4 back-to-back loads | ~4 c each (last issued at 16 c) | V |
| steady-state dwordx4 issue (64 loads, 1 wave) | 14 c/load (~73 B/clk) | V |
| 8 waves, dwordx4 | CU aggregate ~16.7 c per wave-load (~61 B/clk/CU) | V |
(`gl_dword_lat_L1x8`/`sc0 sc1` variants fault on this host - an unresolved bug in my kernel, so cache-bypass numbers are Unknown.)

## 6. Misc
| item | cycles | Status |
|---|---|---|
| s_barrier, 1 / 2-4 / 5-8 waves | 3.9 / 8.2 / 12.2 per barrier (waves in lockstep) | V |
| s_setprio | one 4-cycle issue slot (4.3 incl. loop overhead); no other cost seen | V |
| s_nop N | (N+1) x 4 cycles; s_nop 0 = 4.0 | V |
| SALU (s_mov): 1 wave 4.0; SALU shared per SIMD: 2nd wave sees 8 | V |

## Caveats
* All kernels use a lane-contiguous, conflict-free pattern and fixed data; power/clock may vary (rep 0 usually slower: cold I-cache). Timings are device-clock cycles of shader clock ~2.39 GHz.
* Single-wave issue cadence 4 cycles means dependent latency below 4 cycles is unobservable; use 2 waves/SIMD for packing evidence.
* Harness quirk: the first descriptor in a code object (`aa_pad0`) faults when launched, so a never-launched pad kernel is placed first; code objects are split (`mb.co`, `mb_b.co`, `mb_c*.co`); a SALU write of an SGPR used as a VMEM base needs ~5 wait states (hit this once).
* Sweeps use v_mov reads only; other VALU/ DS/ VMEM readers of MFMA results were not individually tested.
