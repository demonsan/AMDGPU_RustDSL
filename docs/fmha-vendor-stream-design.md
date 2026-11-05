# FMHA: the vendor's copy stream vs ours (design note, realistic data)

Status: analysis + first prototypes (kernel-opt, GPU 5, `REF_DATA=normal`). Experiment code: branch `fmha-ring3` (not for merge).

## 1. What the remaining gap is made of

Warm ATT (`tools/att_fmha.sh`, B4H8S2048, CU 1 SIMD 0, wave 0 = group A), MFMA-stream gap attribution (`tools/att_gaps.py`),
after the bias-copy fix (Main `97ebaf31`):

| | ours | vendor |
|---|---:|---:|
| MFMA stream span / ideal (cycles) | 44 648 / 17 536 | 36 496 / 17 280 |
| excess: `s_barrier` (the partner group's phase) | 11 040 (37 %) | 19 828 (98 %) |
| excess: `s_waitcnt vmcnt(0)` at the end of the VALU phase | **10 712 (36 %)** | — |
| excess: `lgkmcnt(0)` in the M phase | 3 324 (11 %) | 52 |
| excess: VALU (a rollback tile) | 2 492 (8 %) | — |

The vendor's only stall is the ping-pong barrier. Ours additionally waits for the K copy of the next tile: group A issues it at the start
of its VALU (softmax) phase and needs it at the start of its M phase, so the copy has one VALU phase (~1.5–2 k cycles at our current
softmax cost) to land, which is not enough.

## 2. The vendor's slot ownership and copy issue points (from the disassembly)

LDS map (84 480 B): `K0 K1 | V0 V1 V2 | KD0 KD1` — K **two** slots of 0x4100 B (16 384 + 256 pad) at 0 / 0x4100, V **three** slots
at 0x8200 + 0x4100·i, K scales two 512 B slots at 0x14500 / 0x14700. Q lives in VGPRs (`v[4:19]`), not in LDS.

The main loop is unrolled by two (two tiles per pass, `label_block_A_loop` / `label_block_B_loop`, separate code per group):

| | group A (waves 0–3) | group B (waves 4–7) |
|---|---|---|
| copies | K (`s[12:15]`, 4 × `buffer_load_dwordx4 … lds`) + K scales (1 × `buffer_load_dword … lds`) | V (`s[16:19]`, 4 × `buffer_load_dwordx4 … lds`) |
| issue point | **inside the QK window of its M phase**, one copy between consecutive scaled QK MFMAs | same, in its own QK window |
| distance | K(t+2) into the slot K(t) used (2 slots: the immediate alternates 0 / 0x4100) | V(t+2) into a 3-slot ring (the base `s72` steps +0x4100, +0x4100, −0x8200) |
| completion | `s_waitcnt vmcnt(0)` at the end of the VALU phase of t+1, just before the mid barrier | same |

Legality: K(t) is read by QK(t) in the M phase of t−1 of both groups; group B's M phase of t−1 runs concurrently with A's VALU phase
of t, i.e. it is over (barrier) when A reaches its M phase of t and overwrites the slot. V(t+2) can only go one phase early with a
third slot, because V(t) is still read by PV(t) in the M phases of t of both groups.

So the vendor gives every copy one M phase **plus** one VALU phase to land, and lets it land while the M phases read LDS.

## 3. Mapping onto our Pipeline / protocol model

* `TileIt` carries ONE slot index for K and V; the vendor needs per-buffer rings (K period 2, V period 3) — `KvRing3` would get one
  `SlotIdx` per buffer, the pipeline two ring counters (`^1` for K, the `select`-cycled mod-3 counter of `OneWindow` for V). The
  verifier already handles both counter forms and counted `vmcnt(n)` completion; no protocol extension is needed (the prototypes
  below verify strictly with 0 errors).
* Issue inside the QK window = the body issues the stage between QK blocks (`issue_buffer_part`, as the earlier `vlate` experiment did).
* Budgets: LDS 2·16 KiB (K) + 3·16 KiB (V) + K scales + our Q / ones / epilogue (32 KiB + 2 KiB) ≈ 118 KiB (< 160 KiB, one block
  per CU unchanged). VGPRs unchanged (the copies are LDS-direct).

## 4. Prototypes (same-run 7×, normal data; base = Main `_vs128` with ldsink)

| prototype | verifier | B4H8S2048 | B1H8S8192 | per-CU B1H8S2048 |
|---|---|---:|---:|---:|
| 3-slot ring for K and V, t+2 staged at the start of the VALU phase of t (`Pipeline::ahead2`) | strict PASS | −6.2 % | −7.3 % | −9.9 % |
| both groups stage t+2 at the end of their M phase, 2 slots (`stage_early`) | **race on V** (needs the 3rd V slot) | −4.8 % | −5.0 % | −7.2 % |
| group A only: K(t+2) at the end of its M phase into K(t)'s slot (the vendor's K schedule; V unchanged) | strict PASS | −1.5 % | −2.8 % | −1.4 % |
| group B's V spread between its QK MFMAs (`vlate`, round 2) | strict PASS | −6 % | −6 % | −15 % |

ATT of the 3-slot prototype: the `vmcnt` stall disappears, but the M-phase LDS waits grow ~4× (`lgkmcnt` 5.6 k → 20.8 k cycles) and the
barrier waits grow (9.7 k → 16 k): a copy landing in LDS while the M phase streams `ds_read_b64_tr_b8` / `ds_read_b128` slows those
reads more than the earlier issue gains. **The key hypothesis of the vendor model ("copies issued in the QK window land without
hurting the M phase") does not hold for our M phase as it is today.**

## 5. Why it may hold for the vendor but not for us, and the next tests

* LDS traffic per M phase: both issue 32 `ds_read_b64_tr_b8` + 16 `ds_read_b128` + 4 `ds_read_b32` per tile, but our read bases and
  the slot pitch differ (our slots are 16 384 B apart with no pad; the vendor pads every slot by 256 B, 0x4100) — the DMA writes and the
  reads may hit the same banks for us. Test: pad our K / V slots to 0x4100 (one `LdsPlan` constant) and re-run prototype 3 (K only).
* Our M phase keeps more reads in flight (`ldsink` now 1 group, like the vendor) and issues the K reads of the next tile in a different
  order (QK block order); with the copy landing concurrently, the order matters. Test: prototype 3 + the vendor's QK read order.
* Q in VGPRs (vendor) vs LDS (ours) does not matter in the loop (Q is read once).

**Result of the padded-slot test** (vendor pitch 0x4100 for our K / V slots, same run 7×, normal): padding alone is neutral
(B4 1722 → 1715, S8192 2390 → 2391, per-CU 636.5 → 636.1) and does not rescue the vendor K schedule (padded + K at the end of the M
phase: 1682 / 2327 / 625, −2.4 / −2.6 / −1.7 %). Bank aliasing between the copy and the M-phase reads is therefore not the cause.

**Conclusion.** On our M phase, every schedule that lets a copy land during an M phase loses more than it gains, including the
vendor's exact K schedule. Building the full vendor stream (per-buffer rings, V 3 slots, copies inside the QK window) is not justified
by these prototypes. The `vmcnt` part of the gap remains; the levers left are (1) a shorter copy latency per tile (issue order /
grouping of the 5 copies of group A), (2) more VALU-phase time for the copy to land (move work from the M phase into the VALU phase),
and (3) finding out why our M phase is more sensitive to concurrent LDS writes than the vendor's (per-phase LDS counters, ours vs
vendor).

## 6. Copy-path investigation (closed; all measured negative, same-run 7×, normal data, base = Main at the time)

| experiment | what changed | B4H8S2048 | B1H8S8192 | per-CU | note |
|---|---|---:|---:|---:|---|
| LDS counters (rocprofv3) | — | | | | loop: no bank / address conflicts for us or the vendor; our 393 216 conflicts are all in the LDS-staged epilogue (direct epilogue: 0, but −3..−9 % slower) |
| epilogue scratch `Swizzle<3,4,4>` | chunk XOR from query bits 1..3 | −0.1 % | −0.4 % | +0.3 % | conflict count unchanged (b32 column writes cannot be conflict-free without dword padding) |
| extra 16 KiB copy per group/tile (timing only) | issued at the VALU phase start / M phase start | −6.2 / −6.6 % | −8.7 / −10.3 % | −13.3 / −14.2 % | copy traffic itself is expensive; landing phase matters little |
| copies spread over the VALU phase | one copy per softmax quarter | −11.4 % | −11.6 % | −22.4 % | later issue → longer exposed copy latency |
| vendor K formation (`khalf`) | half-row runs, 2 voffsets × imm 0/64, swizzle by row&3 | −5.4 % | −7.5 % | −12.1 % | K issue stall drops to the vendor's level, but the vmcnt wait doubles (10.7 k → 22.0 k cycles) |

| row sums at the end of the VALU phase (rebalancing) | 2 smfmacs moved before the mid barrier | −10.4 % | −11.8 % | −20.3 % | matrix work in the VALU phase collides with the other group's M phase and lengthens the critical path |

Conclusion: in the two-group structure the per-tile copy latency is the floor of the VALU phase + barrier; issue backpressure
(our 7–23 cycles per copy vs the vendor's 4) is not the cost, and none of the vendor's copy-side choices transfer. Experiment code:
branches `fmha-ring3`, `fmha-episwz`, `fmha-dmaform`, `fmha-khalf` (not for merge).