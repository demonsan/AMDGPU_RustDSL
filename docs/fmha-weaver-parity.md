# `_vs128` parity attempt with the weaver (closed, all negative)

Status: closed (kernel-opt, 2026-11-07). Nothing adopted; `_vs128` remains the shipped kernel (asm md5 `8938adfa`).

What was tried: the weaver stack (Main `28e99c30` / `1250bf8f`):
- emission units and explicit weaves ([`fusion-stage-library-design.md`](fusion-stage-library-design.md) §7);
- per-region scheduler strategies ([`weaver-sched.md`](weaver-sched.md));
- register planning ([`weaver-regs.md`](weaver-regs.md)).

These were applied to the filled shapes. The question was whether anything a weave can express closes the remaining gap to the vendor (0.89–0.96×).

**Protocol**: all A/B are same-run interleaved, `REF_DATA=normal`, GPU 5 (mi355-gpu-53). Tables give the median of 5 runs unless marked; units are TF/s. Shapes are B1H8S2048 (per-CU; `ROWS_PER_BLOCK=256` → `_vs128`), B4H8S2048, B1H8S8192 and B8H8S8192. Accuracy of every variant: `bad = 0` at 16 / 64 rows.

## 1. m_phase unit orders fed to the list scheduler (branch `weave-exp`)

Unit ids for this round:
- PV units: `setup`, `rowsum`, then `(hh, e)` read + MMA;
- QK units: `setup`, blocks, `end`.

The shipped order is all PV units, then all QK units.

| order | B4H8S2048 | B1H8S8192 | B8H8S4096 |
|---|--:|--:|--:|
| shipped (PV then QK) | 1581 | 2311 | 2309 |
| QK first | 1505 (−4.8 %) | 2177 (−5.8 %) | 2173 (−5.9 %) |
| PV(hh,e) / QK block alternating | 1495 (−5.4 %) | 2211 (−4.3 %) | 2207 (−4.4 %) |
| rowsum at the end | 1555 (−1.6 %) | 2284 (−1.2 %) | 2297 (−0.5 %) |
| PV e-paired (h0e0, h1e0, …) | 1559 (−1.4 %) | 2292 (−0.8 %) | 2300 (−0.4 %) |
| setups first, PV / QK interleaved | 1494 (−5.5 %) | 2159 (−6.6 %) | 2159 (−6.5 %) |

## 2. Region strategies on the shipped weave (`AMDC_SCHED_REGION`)

| regions | per-CU | B4H8S2048 | B1H8S8192 | B8H8S8192 |
|---|--:|--:|--:|--:|
| none (kernel default) | 642 | 1621 | 2310 | 2433 |
| `v_phase:authored` | 661 (+3.1 %) | 1607 (−0.8 %) | 2284 (−1.1 %) | 2464 (+1.3 %) |
| `kv.stage:authored` | 620 (−3.3 %) | 1582 (−2.4 %) | 2300 (−0.4 %) | 2395 (−1.6 %) |
| `v_phase` + `kv.stage` authored | 648 (+0.9 %) | 1588 (−2.0 %) | 2283 (−1.2 %) | 2449 (+0.7 %) |
| `m_phase` + `v_phase` authored | 656 (+2.1 %) | 1582 (−2.4 %) | 2275 (−1.5 %) | 2450 (+0.7 %) |
| `v_phase` authored + `m_phase` `ldsink=2` | 648 (+0.9 %) | 1589 (−1.9 %) | 2258 (−2.3 %) | 2434 (0) |

Region tokens only trade per-CU against B4 / S8192 (±1–3 %). No setting wins everywhere.

## 3. Hand m_phase orders in an authored region (branch `parity-w`)

Finer units here: each QK block and each PV `(hh, e)` is split into a read unit and an MMA unit.

| order (`m_phase:authored`) | per-CU | B4H8S2048 | B1H8S8192 | B8H8S8192 |
|---|--:|--:|--:|--:|
| base (list scheduler) | 643 | 1574 | 2309 | 2434 |
| h1, vendor-like: PV reads 2 ahead, rowsum after the first reads, QK reads 1 ahead | 566 (−11.9 %) | 1510 (−4.1 %) | 2167 (−6.1 %) | 2263 (−7.0 %) |
| h2: K reads interleaved into the PV MFMA stream | 566 (−12.0 %) | 1509 (−4.1 %) | 2167 (−6.2 %) | 2250 (−7.6 %) |
| h3: QK reads 2 ahead + MFMA alternation | 574 (−10.7 %) | 1504 (−4.4 %) | 2181 (−5.6 %) | 2261 (−7.1 %) |
| h2 + `v_phase:authored` | 647 (+0.7 %) | 1545 (−1.8 %) | 2244 (−2.8 %) | 2431 (−0.1 %) |
| h2 order, NOT authored (list scheduler), 9 runs | 648 (+0.9 %) | 1563 (−2.3 %) | 2282 (−0.9 %) | 2448 (+0.1 %) |

In the 9-run confirmation the base measured 642 / 1600 / 2303 / 2446.

## 4. Register side (branch `parity-regs`; `amdc regplan` as the truth)

With 2 waves per SIMD each wave has 256 unified registers. `_vs128` peaks at 239 arch VGPRs in `m_phase`.

| variant | arch VGPR / AGPR | m_phase footprint | per-CU | B4H8S2048 | B1H8S8192 | B8H8S8192 |
|---|---|--:|--:|--:|--:|--:|
| shipped | 239 / 0 | 239 | 641 | 1551 | 2306 | 2437 |
| Q fragments in AGPR (QK MFMA B operands) | 231 / 16 | 223 | 600 (−6.4 %) | 1524 (−1.7 %) | 2251 (−2.4 %) | 2365 (−2.9 %) |
| Q in AGPR + one more read block ahead (`m_phase` `ldsink=2`) | 231 / 16 | — | 567 (−11.4 %) | 1489 (−4.0 %) | 2181 (−5.4 %) | 2262 (−7.2 %) |
| shipped + `m_phase` `ldsink=2` | 247 / 0 | — | 572 (−10.8 %) | 1511 (−2.6 %) | 2177 (−5.6 %) | 2290 (−6.0 %) |

Moving Q to AGPRs frees 16 VGPRs, which is enough room for a K / V `reg_ring` of one more fragment set. But the move alone costs 2–6 %, and deeper read-ahead costs another 4–10 %. The register-side lever does not pay.

## 5. Conclusion

- **Nothing beats the shipped schedule.** Every in-phase weave, authored region, region token or register-placement variant measured on the filled shapes is either below it, or only trades per-CU against the large shapes.
- **The list scheduler beats every hand order written here** (−4 … −12 % when authored).
- **The remaining excess sits outside what a weave can change.** Per the ATT attribution in [`fmha-vendor-stream-design.md`](fmha-vendor-stream-design.md), it is the barrier hand-off between the two groups (37 %) and the end-of-VALU copy wait (36 %). Both are set by the two-group structure and the copy latency, not by the instruction order inside a phase. The structural alternatives (3-slot ring, early / late copy issue, vendor copy formation, 4-wave) were measured negative earlier.
- **The weaver remains the vehicle for future structural experiments.** Units, regions and register planning stay in Main: md5-identical, with certificates unchanged.

Experiment branches (not for merge): `weave-exp`, `parity-w`, `parity-regs`.
