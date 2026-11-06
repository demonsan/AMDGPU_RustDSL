> 状态（2026-11-06）：4-wave 原型已搁置。最好成绩 0.57× / 0.70× / 0.80× 厂商（每 CU / B4H8S2048 / B1H8S8192，真实分布数据）；代码只在实验分支 `fmha-4w2` / `w4sched` 上，未合入 Main。详见 §8 与 [lgkm-lookahead.md](lgkm-lookahead.md) 末节。

# FMHA 4-wave design (one wave per SIMD, AGPR accumulators) — design + first probe

Status: design note + overlap probe (kernel-opt, branch `fmha-4w`, GPU 5). Nothing shipped.

## 1. Why

Warm ATT of `_vs128` on realistic data (`docs/fmha-vendor-stream-design.md` §1): our MFMA stream span is 44.6 k cycles vs the vendor's 36.5 k for the
same work (ideal 17.5 k). The excess is the end-of-VALU-phase copy wait (36 %) and the two-group barrier hand-off (37 %). The two-group
ping-pong ties every tile of one group to a barrier with the other group. One wave per SIMD that overlaps its **own** MFMA stream with its softmax
VALU and its copies (FA3-style intra-wave software pipelining) removes the hand-off and gives every copy a whole tile to land.

## 2. Per-tile work (32 query rows × 128 keys, head dim 128)

| | count | cycles (probe-calibrated) |
|---|---:|---:|
| scaled MFMA 32×32×64 (QK 8 + PV 8) | 16 | 16 × ~70 = **1 130** (MFMA-only loop: 1 131 / 16) |
| bias MFMA + 2 row-sum smfmacs | 3 | ~100 |
| VALU (softmax fast path, conversion, address) | ~280 executed | ~4.7 / op → **~1 300** |
| LDS reads (K b128 16, V tr_b8 32, KD b32 4) | 52 | hidden if issued ahead |

The tile is VALU-bound by ~15 % even with perfect overlap. Measured today: our `_vs128` spends ~1 980 SIMD cycles per 32-row tile (per-CU
648 TF/s), the vendor ~1 770 (724 TF/s).

## 3. Probe: does ONE wave overlap its own MFMA and VALU? (`probe_ovl_w<waves>_m<mfma>_v<valu>`)

`crates/amdgpu-kernels/src/probes/mod.rs::probe_ovl` + ovl_host.cpp（实验分支 `fmha-4w2`，未合入 Main）: per iteration `m` in-place scaled 32×32×64 MFMAs on 4 accumulators and `v`
independent VALU ops (mul / max3 / add on 16 loop-carried chains), interleaved one MFMA per `v/m` VALU ops by the builder, 200 iterations, `s_memtime`
per wave. One block per CU; 4 waves = 1 per SIMD, 8 waves = 2 per SIMD.

| 1 wave / SIMD | cycles / iteration | vs MFMA-only | vs VALU-only | sum |
|---|---:|---:|---:|---:|
| 16 MFMA only | 1 131 | — | — | — |
| 256 VALU only | 1 196 | — | — | — |
| 16 MFMA + 128 VALU | 1 188 | **1.05×** | | |
| 16 MFMA + 256 VALU | 1 455 | | **1.22×** (82 % overlap) | 2 327 |
| 16 MFMA + 384 VALU | 1 969 | | 1.10× of VALU-only (≈ 1 794) | |

With 2 waves per SIMD (each wave the same mix) the per-iteration SIMD time of the pair is close to max(2·MFMA, 2·VALU) (the current ping-pong
regime).

**Reading.** A single wave issues independent VALU while its MFMAs run: up to ~8 VALU per MFMA the MFMA pipe stays the bound (1.05×); at the
FMHA balance (~17 VALU per MFMA) one wave reaches ~82 % of perfect overlap. At our VALU count that is ~1 550–1 650 cycles per 32-row tile if
the copies and LDS reads are fully hidden — **~10 % fewer than the vendor's 1 770** and ~20 % fewer than our current 1 980. So the hypothesis
is plausible; the margin depends on hiding the memory side, which the probe does not model.

## 4. Candidate geometries

| | A: 32 rows / wave, 128-row block | B: 64 rows / wave, 256-row block |
|---|---|---|
| waves / block, blocks / CU | 4, 1 (2 if LDS allows) | 4, 1 |
| MFMA / VALU per wave-tile | 16 / ~280 | 32 / ~560 |
| registers (per lane) | O 64 (AGPR, pinned) · S_cur 64 · S_next 64 · Q 16 · P 16 · K/V frags ~48 · softmax/address ~40 → **VGPR ~250 + AGPR 64** | O 128 (AGPR) · S_cur 128 · S_next 128 · Q 32 · P 32 · frags ~48 · misc ~40 → **~410 VGPR**: S_next must live in AGPR (MFMA D=AGPR) and be read back (+128 `v_accvgpr_read` per tile, +~10 % VALU) |
| K/V copy bytes per flop | **2×** today (a 32 KiB K+V tile per 128 rows) | 1× (as today) |
| LDS | K/V ring ≥ 3 slots (copy a full tile ahead): 3 × 32 KiB + KD → ~100 KiB, 1 block / CU | same |
| barriers | one per tile (K/V slot publication across the 4 waves) | same |

Geometry A fits the register file without AGPR round trips but doubles the copy traffic per flop. The in-situ measurement says one extra tile of
copies costs 6–14 %, so A probably gives most of the overlap gain back. Geometry B keeps the copy traffic but needs S_next in AGPRs (or a
half-tile QK pipeline: QK of the next tile in two halves, so only 64 S_next registers are live at once). **B with a half-tile S_next is the
preferred design.**

## 5. Per-tile issue schedule (geometry B, one wave, steady state)

```
tile t (rows 0..63 of the wave):
  softmax VALU of S_t (rows 0..31, then 32..63)       — ~560 VALU
  interleaved:  PV(t-1) 16 MFMAs (O in AGPR, P_{t-1} from VGPR)
                QK(t+1) 16 MFMAs (half 1 into AGPR S_next_hi, half 2 into VGPR after the softmax of S_t frees it)
                K/V copies of tile t+2 (issued early; a full tile to land)
                K reads for QK(t+1), V tr reads for PV(t-1), each 1–2 MFMAs ahead
  one s_barrier (slot t+2 published, slot t-1 free)
```
MFMA 32 × ~70 = 2 260 cycles vs VALU ~560 × 4.7 = 2 630 cycles: still VALU-bound; with 82 % overlap ≈ 3 100 cycles per 64-row tile
(1 550 per 32 rows).

## 6. Compiler support needed (checked with backend-llvm-tail)

* AGPR C/D: **yes** for `v_mfma_scale_f32_32x32x64_f8f6f4` and `v_mfma_f32_32x32x8_f16` (`pin_agpr` / `pin_agpr_at` on the accumulator vars,
  `b.agpr(value)` for MFMA results); **no** for `v_smfmac_f32_16x16x128_fp8_fp8` (row-sum accumulators stay in VGPRs; an AGPR smfmac form
  would need an assembler + hardware probe first).
* The O rescale on AGPR-resident O becomes `v_accvgpr_read ×2 → v_pk_mul → v_accvgpr_write ×2` per pair (automatic in the agpr pass) — fine
  because it only runs in the lazy rollback branch.
* Unified budget 256 VGPR + 256 AGPR at one wave per SIMD is supported and verified (GV-REG-LIMIT); the scheduler's pressure model does not
  count AGPRs.
* Scheduling: the interleave of one wave's MFMAs with its own softmax VALU across tile boundaries is the scheduler's job. Today's list scheduler
  works inside a basic block; the rollback branch splits the softmax into blocks. The pipeline must be expressed in the IR order (as `probe_ovl`
  does) or needs sched's cross-block / software-pipelining support — to be agreed with sched.
* Pipeline: a new `Cadence` (one group, 3-slot ring, stage t+2, one barrier per tile, body interface with the QK/PV/softmax split by halves).

## 7. Next steps

1. Prototype the steady loop of geometry B without masking / tails (realistic data, timing only): measure cycles per 64-row tile vs the probe
   bound and vs the vendor's per-CU throughput.
2. If it beats the vendor per CU, build the full kernel (masking, tails, KV split variant, epilogue), strict verify, port_gate.

## 8. Prototype results (geometry B, `fmha_fwd_hd128_mxfp8_v4w_proto`, fmha/w4.rs（实验分支 `fmha-4w2`，未合入 Main）)

One wave per SIMD, 64 query rows per wave as two 32-row groups offset by half a tile inside the wave (the `_vs128` ping-pong moved into one wave),
O in AGPRs, 3-slot K/V ring (K two tiles ahead, V one), one barrier per tile, long-sequence recipe, no masking. Correct on GPU (normal data:
bad = 0, mean_err equal to the vendor's). TF/s, same run, normal data (vendor 712 / 1894 / 2405):

| variant | per-CU B1H8S2048 | B4H8S2048 | B1H8S8192 | note |
|---|--:|--:|--:|---|
| Q / K frags / g1-P in AGPR, scheduler falls back (authored order) | 366 | 1179 | 1607 | list scheduler out of VGPRs at every limit |
| + sched `rewait,hoist=24,ovl` (authored order, counted waits, VALU spread behind MFMAs) | 405 | 1321 | 1927 | 0.57 / 0.70 / 0.80× vendor |
| S of both groups in AGPR + one VGPR working copy per softmax (list scheduler runs) | 368 | 1140 | 1595 | +64 `v_accvgpr_read` per 32 rows; VALU-class now ≈ MFMA time |

Warm ATT (B4H8S2048, one wave = 64 rows): ~39 MFMA (≈ 2 800 cycles at 72) and ~380 VALU (+ copies) per 64-row tile; the MFMA stream takes
≈ 4 800 cycles per tile with the best schedule (vendor ≈ 3 600 per equivalent SIMD-tile). The excess is VALU stretches not covered by MFMAs
(41–69 %), LDS waits (11–28 %) and hazard padding (5–20 %). The hardware can overlap a wave's own MFMA and VALU (probe, §3), but our
schedulers do not produce that interleave for this loop: the list scheduler cannot run under one-wave register pressure (its model under-counts
by ~26 dwords) and the authored-order path recovers only +10 %. **Status: not competitive yet (0.57–0.80×); reaching the projected ≈ 3 000
cycles per tile needs a modulo / software-pipelining scheduler (MFMA skeleton with the softmax VALU, LDS reads and copies filled into its
slots).**
