# Gen6 rebalance input: vendor V/M split, our solo windows, equal-window VALU split

> Status: input document for the gen6 round (K-ahead ring + merged window + rebalance), which closed NEGATIVE; see docs/archive/gen6-results.md and round 64 of docs/archive/status.md.

Inputs: gen5 asm extraction (`docs/archive/gen5-window-spec.md` §1/§5), our kddma counters (per wave-iteration,
`docs/vendor-spec.md` §7.10.1/§7.10.3), kernel-opt's partner-free solo measurements (V 1.15k + M 1.5k +
other 0.3k = 2.96k compute; DMA/barrier 1.17k; solo tile 4.1k; paired tile 5.34k cycles). The quick
solo-ablation re-check on `82ff0bf` was attempted but the node was heavily loaded (aidle/bidle timings and
counters contaminated); the numbers below are the quiet control values and will be re-checked in the gen6
gate.

## 1. Vendor V/M split per window (per tile = 128 keys, per wave)

| | vendor |
|---|---|
| M window (matrix) | **20 MFMA + 4 SMFMAC = 1086 XDL-busy cycles** (static XDL 1088; 93.7-94% of the wave) |
| V window (VALU) | **251 (group B) / 267 (group A) VALU instructions** from the loop asm; at the measured active rate 293.8 VALU-cycles / 250.9 instructions = 1.17 cycles/instruction => **~294 / ~312 active-VALU cycles**. If the gen6 brief's "~100 instructions" is the post-merge V phase, that is **~117 cycles** |
| wave time | 1146-1157 cycles |
| V/M ratio | **0.27** (canonical 251-instruction V phase) or **0.11** (~100-instruction phase) |

So the vendor's V work is *hidden*: M is 94% of the wave and VALU fits in its XDL shadow; there is no
V-shaped 1.15k window in their timeline. Their 2 barriers + setprio are what keeps the half-tile pairing
aligned while V hides.

## 2. Our V and M windows on the solo path (per tile, one group)

| component | cycles | note |
|---|---|---|
| V window (VALU phase) | **1.15k** | our VALU phase is 0.77x the M window |
| M window (matrix) | **1.50k** | XDL busy 1216 + ~0.28k of M-window stall/waits |
| other compute | 0.31k | loop/rollback glue |
| **compute total** | **2.96k** | |
| DMA + barriers | **1.17k** | 2 barriers/tile + K/V staging + K-scale DMA |
| **solo tile** | **4.10k** | one group active |
| paired tile | **5.34k** | both groups (half-tile complementary pairing); pairing penalty 1.24k |

Our V/M ratio is **0.77**, i.e. 2.9-7x the vendor's 0.11-0.27: our VALU window is far too long to hide, and
the M window is also ~0.4k longer than the vendor's 1086 (XDL +130, tail ~0.28k). The merged-window target
is therefore: move matrix work into the V shadow (QK(t+1) x cycles) and rebalance the VALU phase so the two
groups' windows are equal.

## 3. Equal-window VALU split

Let `V = 1.15k`, `M = 1.50k`, `x` = matrix cycles moved from the M window into the V window (QK(t+1) issue),
`y` = VALU cycles moved from the V window into the M window (explicit phase rebalance). Then
`V' = V + x - y`, `M' = M - x + y`, and `V' = M'` iff

    y = x - (M - V)/2 = x - 0.175k.

| QK moved x | VALU split (V window : M window) | V' = M' |
|---|---|---|
| 0.175k | **1.15k : 0** (100% : 0%) | 1.325k |
| 0.30k | **1.025k : 0.125k** (89% : 11%) | 1.325k |
| **0.50k** (planned QK move) | **0.825k : 0.325k (71.7% : 28.3%)** | **1.325k** |
| 0.675k | 0.65k : 0.50k (57% : 43%) | 1.325k |

**Recommendation for the planned x ~= 0.5k QK move: pin the V window at 0.825k of VALU + 0.5k of QK
matrix (1.325k) and the M window at 0.325k of VALU + 1.0k of PV/SMFMAC (1.325k), i.e. move ~28% of the
VALU phase (the address/pk_add/pk_mul part that can run after the QK issue, not the pknorm/skalef32 chain)
into the M window.** For a more conservative x = 0.30k move, the split is 89/11. The compute total stays
2.65k either way; the win comes from removing the window imbalance/barrier skew, expected solo 4.1k ->
~3.85-3.95k (`+4..6%` per tile) and paired 5.34k -> ~5.0k (`+7..10%` per-CU) -> 410-430 TFLOP/s at 64
blocks; the floor (QK fights the partner's VALU for issue slots) is ~0, as gen2/gen3 showed.

## 4. Barrier / priority proposal (beyond today's 2-per-tile complementary pairing)

Today: 2 barriers/tile (B1 publishes the staged tile; B2 recycles the ring) + one setprio pair; the
half-tile offset is enforced by the barrier pair. With ring depth 2 this is the vendor's mechanism too.

**Proposal: role-swapped single-barrier tile with ring depth 3.** Per tile use **one barrier** at the tile
end; group A runs V(t) then M(t), group B runs M(t) then V(t) (explicit swapped order, not the identical
code half a tile later). The half-tile offset is then encoded in the ring depth/slot parity: with depth 3,
the slot written for t+2 (=(t+2) mod 3) is never the slot read for t (t mod 3), so no M->V barrier is
needed to protect the ring; the single barrier at the tile end keeps the two groups aligned and lets
setprio toggle there so the group entering M gets priority. Expected effect: remove one barrier plus the
associated skew (~0.1-0.25k cycles/tile = **+2..5%**), and it composes with the balanced windows above
(single barrier -> no half-tile drift to correct). Cost/risk: the third K/V stage slot must fit the LDS
budget (current LDS 102400 B/block, 1 block/CU; need the stage-slot size check against 160 KB); the
P/S dependencies stay per-group (independent), and the drain-closed boundary contract is unchanged.
Fallback if the ring cannot grow: keep the current 2-barrier complementary pairing - the balanced VALU
split above still gives most of the win, just with B2/skew left in place.

## 5. What to measure in the gen6 gate

(a) `_vs5` winasm `.s` differs from `_vs128`; (b) md5 `faadf00c...` bit-identical; (c) counters:
WAIT_INST_ANY down, XDL busy up per wave-iter, wave cycles down; (d) per-CU best >=410 TFLOP/s;
(e) filled B4H8S2048/B1H8S8192 no regression; (f) auto gate 15/15 + det 5/5. Also report the *V-window*
and *M-window* occupancy separately (VALU active vs XDL busy per window) so the rebalance itself is
visible, not just the total.
