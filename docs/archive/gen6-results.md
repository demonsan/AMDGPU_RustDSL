# Gen6 results (kernel-opt): K-ahead ring + merged window + rebalance - FINAL NEGATIVE

> Status: FINAL NEGATIVE (K-ahead ring -10.5%, staging VALU in the M window -12..-17%); see docs/current-status.md §5 and round 64 of docs/archive/status.md.

Kernel `fmha_fwd_hd128_mxfp8_vs6` (by name only; `_vs128` and the auto dispatch are untouched). Base: Main 82ff0bf (kernel-occ control: `_vs128` 390.6 best / 389.5 median TFLOP/s at 64 blocks, md5 `faadf00c95f4eaffff94ece7d1429b3d`).

## Step 1: the K-ahead ring (DSL, no asm) - correct, -10.5%
Protocol: QK(t+1) runs in the V window of tile t (right after the conversion of S(t); each S tile is overwritten when its scores were consumed), K(t+1) and its scales are already in LDS because group A staged K(t+2) one tile earlier
(K ring one tile ahead of V), V(t+1) is staged by group B as before; PV(t) stays in the M window; group A waits for K(t+2) (`vmcnt(0)`) at the end of M(t), group B for V(t+1) at the end of V(t); the prologue also stages K(1).
**A race that the first version had (kept here because it is the non-obvious constraint of this protocol):** the two wave groups are one barrier apart (group B computes the prologue QK(0) between the first and the second barrier, i.e. during group A's V(0)),
so group A may not issue the K(2) DMA into the slot-0 K region at the start of V(0): B is still reading K(0) there. Result of that first version: non-deterministic output (4 different md5s in 4 runs). Fix: iteration 0 issues K(2) after barrier B1, all other iterations at the V-window start. With the fix the output is bit-identical and deterministic.
Gates for the DSL control: bf16 dump md5 identical to `_vs128` on 1 8 2048, 4 8 2048, 3 3 768 and 1 2 200 (`faadf00c…`, `c8b7a893…`, `d17778b0…`, `20180cb5…`), 15/15 shapes at 16 rows, 12 + 3 vendor-bad at 64 rows with our bad = 0, parity 15/15, determinism 5/5 x3 (explicit vmcnt waits pinned: `_vs6` sets `PIN_VMWAIT_KERNEL`).
**Timing (B1H8S2048, same runs, 300 iters): `_vs128` 387.4-390.0 vs `_vs6` 347.4-347.9 (V-start K issue) = -10.5%** (341.2 with K issued after B1, 321-337 with the V DMA also moved to the M window). Same as g2/g3 (-6..-9%), now with a verified-correct protocol and a deterministic output.

## Step 2: rebalance iteration - the model's direction is not what the hardware does
kernel-occ's model (`docs/archive/gen6-rebalance.md`: equal windows iff y = x - 0.175k) says: with the QK work (x ~ 0.5k) moved into V, move ~0.3k of VALU into M. Measured, all md5-identical, TFLOP/s on B1H8S2048 (`_vs128` 387.4-388.1 in the same runs):
| variant | TFLOP/s | vs `_vs128` |
|---|---|---|
| all QK in V, K(t+2) DMA issued at the V start (A), V(t+1) at the V start (B) (`qkv4`) | 347.4-347.9 | -10.5% |
| + A's K DMA issue (address VALU + 5 `global_load_lds`) moved to the M window (`kmv`) | 341.2-341.8 | -12.0% |
| + B's V DMA issue moved to the M window (`vmv`) | 321.1-322.1 | -17.1% |
| + both (`kmv,vmv`) | 336.7-337.5 | -13.1% |
| QK tiles split between V and M (`qkv0..3`) | **invalid**: nondeterministic output - with the V-start K issue the remaining QK tiles in M(t) of group B (one barrier behind) read K(t+1) while group A already overwrites that slot with K(t+3) |
Moving VALU (the staging address math) into the M window makes the kernel SLOWER, not more balanced: the M window is not a stall-dominated window with free VALU slots (it carries 52 DS reads, 20 MFMAs and 16-18 partial waits), so every VALU instruction placed there delays the matrix stream, while the V window with the QK MFMAs interleaved gets no faster.
So the rebalance target of the model cannot be reached by moving the staging VALU, and the other VALU of the V window (max chain, rollback decision, pknorm/scalef32 conversion) cannot move into M without a second copy of the S registers.
## Verdict
Hard stop reached: the best gen6 variant is 347.9 TFLOP/s (-10.5%), well below the certified 391 per-CU; no asm merged window was authored because (a) the DSL control of the exact same schedule is already -10.5% with the scheduler's order and (b) the rebalance iteration moved the wrong way. `_vs128`, the KV-split and the auto dispatch are unchanged (shipped ~622 TFLOP/s on B1H8S2048, per-CU 391 TFLOP/s at 64 blocks). `_vs6` stays in the tree as an opt-in, correct, documented negative.
