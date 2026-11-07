# FMHA on filled shapes: wave organisation vs memory locality (Part 2 design note)

Status (kernel-opt, 2026-11-08):
- **Adopted**: the XCD-aware block order (branch `xcd-order`; `port_gate` PASS).
- **Measured and set aside**: the wave-organisation redesigns.

Protocol: same-run A/B, `REF_DATA=normal`, GPU 5.

## 1. Can the wave organisation be changed?

| candidate | check | result |
|---|---|---|
| (A) dedicated producer waves (8 compute + 4 copy waves per block) | gfx950 allocates VGPR + AGPR **per kernel**: every wave of the block gets the same 240-register allocation, so 3 waves per SIMD need 720 of 512 registers | not possible unless compute waves fit in ≤ 168 registers (`_vs128` uses 239) |
| (B) per-group K / V rings | gfx950 has only the workgroup-wide `s_barrier` (no named barriers), so the cross-group barrier stays; each group also has to copy every tile itself (2× copy traffic) | no structural gain |
| (C) asymmetric / 3-phase rotation | see the barrier ablations below: the barriers carry the ping-pong | not pursued |

## 2. Where the time goes: timing upper bounds (ablations of `_vs128`, results deliberately wrong)

TF/s, same run, median of 5.

| variant | per-CU B1H8S2048 | B4H8S2048 | B1H8S8192 |
|---|--:|--:|--:|
| shipped | 642 | 1603 | 2303 |
| no mid barrier | 484 | 1401 | 1939 |
| no end barrier | 534 | 1437 | 2052 |
| no barriers | 459 | 1360 | 1874 |
| no `vmcnt(0)` before the mid barrier | 664 | 1571 | 2295 |
| **no K / V / K-scale copies at all** | **715** | **1926** | **2657** |
| no group-A copies (K, K-scale) | 705 | 1727 | 2452 |
| no group-B copies (V) | 697 | 1755 | 2495 |
| vendor | 709 | 1732 | 2406 |

What the ablations show:
- **The barriers are not overhead.** Removing them is 16–29 % slower: they keep the two groups' MFMA / VALU phases interleaved.
- **The copy wait is not the cost either.** Dropping the `vmcnt(0)` gains about 0 %.
- **The cost is the copies themselves.** Without them `_vs128` meets or beats the vendor.

## 3. Root cause: L2 locality across XCDs

rocprofv3, B4H8S2048:

| counter | ours | vendor | ours + XCD order |
|---|--:|--:|--:|
| `TCC_HIT_sum` | 0.31 M | 1.48 M | 1.21 M |
| `TCC_MISS_sum` | 1.27 M | 0.36 M | **0.36 M** |
| `TCC_EA0_RDREQ_sum` (memory reads) | 1.14 M | 0.23 M | — |

Why the plain order misses:
- Workgroups are dispatched round-robin over the 8 XCDs (linear id mod 8), and each XCD has its own L2.
- With the plain order (x = query tile fastest), the query tiles of one head land on 8 different XCDs, so every XCD streams the same K / V from memory.

The vendor's fix, `label_xcd_skip` in its disassembly:
- Remap the linear id within one batch's (query tile, head) plane: `L = y·X + x` → `M = (L mod 8)·(N/8) + L/8`, with N = X·H.
- Apply it when N ≡ 0 (mod 8) and H ≤ 32.
- With the remap, XCD k owns the contiguous logical range [k·N/8, (k+1)·N/8): the blocks that share K / V share an L2.

## 4. Our implementation (`fmha/mod.rs`, `block3_x`)

- **Computation**: SALU, once per block. The head comes from a 5-step binary division, then `readfirstlane`.
- **Verifier**: the remapped ids carry two trusted range facts: a bijection over [0, X·H) cannot be followed by the interval engine. The certificate goes from 24 to 26 trusted assumes.
- **Kernels**:
  - used by `_vs128` and `_t128`;
  - NOT by `_kvsplit`: −10.6 % at B1H8S2048, −6.2 % at B1H4S4096.
- **Extra condition X ≤ 16** (S ≤ 4096 for `_vs128`, S ≤ 2048 for `_t128`). With X = 32 the order was +0.8…+2.3 % on B ≤ 4 but −6.9 % at B8H8S8192, even though the L2 misses dropped to the vendor's there too. Cause not identified; a candidate is all 32 CUs of an XCD streaming one head in lockstep.

Same-run A/B, normal data, auto path (5 runs, TF/s, base → XCD order, then the ratio to the vendor):

| shape | base | XCD order | change | ×vendor (base → new) |
|---|--:|--:|--:|--:|
| B4H8S2048 | 1782 | 1845 | +3.5 % | 0.966 → 1.000 |
| B2H8S2048 | 1159 | 1219 | +5.1 % | 0.916 → 0.963 |
| B1H16S2048 | 1150 | 1228 | +6.8 % | 0.915 → 0.976 |
| B8H8S2048 | 1629 | 1764 | +8.3 % | 0.893 → 0.967 |
| B16H8S2048 | 1783 | 1897 | +6.4 % | 0.899 → 0.957 |
| B4H32S2048 | 1778 | 1898 | +6.7 % | 0.914 → 0.976 |
| B2H8S4096 | 1870 | 2014 | +7.7 % | 0.896 → 0.966 |
| B8H8S4096 | 2230 | 2296 | +2.9 % | 0.953 → 0.980 |
| B1H8S8192 / B4H8S8192 / B8H8S8192 | — | — | 0 (X = 32: plain order) | unchanged |

Gate (`tools/port_gate.sh fmha_fwd_hd128_mxfp8_v3 xcd`, GPU 5): OVERALL PASS. VERIFY 0 errors; ACC16 / ACC64 vendor parity 15/15 on every path; DET 5/5; STRESS pass.

| PERF item (normal data) | ×vendor |
|---|--:|
| KV-split B1H8S2048 | 1.025× |
| per-CU | 0.959× (layout median 0.948×) |
| B4H8S2048 | 0.976× |
| B1H8S8192 | 0.988× |
| B2H8S2048 | 0.978× |

Experiment code (ablations, the per-path variants): branch `wave-org` (not for merge).
