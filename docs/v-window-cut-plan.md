# V-window VALU audit and cut plan (filled-grid parity)

Measured from the emitted `_vs128` long/no-LUT hot loop on Main 5f4a3cf (own backend), per loop-body
V phase (one 128-key step, role A / role B). Method: barrier-delimited regions of the emitted asm;
loop bodies at backedges BB7/BB26/BB48/BB67; the long/no-LUT bodies are the last two (251/267); the
LUT/short bodies are 381/397. Vendor composition from docs/gen5-window-spec.md / gen9-vendor-mirror.

| class | ours A / B | vendor B | delta / note |
|---|---:|---:|---|
| conversion `v_cvt_pknorm_u16_f32` + `v_cvt_scalef32_pk_fp8_bf16` | 32+32 / 32+32 = 64 | 64 | equal; vendor uses s89=0.25 |
| bias shift/scale `v_pk_add_f32` + `v_pk_mul_f32` | 32+32 = 64 | 0 | **vendor bakes bias into QK C**, we do not |
| bias coefficient unpack (cvt_f32_f16 x4, mul_f32 x5, add3, sub3, cvt_f16 x2, exp1) | 18 | ~1 | our carried hi/lo f16 reference word |
| max chain (`v_max3_f32` 31, `v_max_f32` 2, `v_permlane32_swap` 1) | 34 | 32-33 | close; vendor has swap+max too |
| decision/masks (cndmask 8/7, cmp_lt_i32 7/6, cmp_lt_f32 3, readfirstlane 2/1) | 20 / 17 | ~2-4 | our clamp/rollback decision glue |
| address/INT32 (add_u32 23/19, mul_lo 6/4, add_co 5/4, addc 5/4, shl 3/2, lshr, and) | 44 / 35 | 0 VALU (~40 SALU) | vendor ring state in SGPR |
| constants / mov | 23 / 20 | ~? | many are loop-invariant materialisations |
| **total V-phase VALU** | **267 / 251** | **251** | |

The 353/wave-iteration counter includes M-phase VALU and LUT/short copies; this table is the hot
long/no-LUT V phase only. The LUT path adds ~130 VALU (perm/sub/and/lshr) to short copies; filled
shapes use the long/no-LUT body.

## Dynamic vs static (measured, rocprofv3, B1H8S2048)

The static V-phase table above is NOT the dynamic cost. Ablations that disable the rollback branch
body leave the hardware counters exactly unchanged:

| build | SQ_INSTS_VALU | SQ_INSTS_VALU_INT32 | SQ_WAVE_CYCLES |
|---|---:|---:|---:|
| base | 2,892,288 | 493,824 | 11,167,268 |
| `rbnos` (no S pk_add rescale) | 2,892,288 | 493,824 | 11,226,252 |
| `rbnoo` (no O pk_mul rescale) | 2,892,288 | 493,824 | 11,147,253 |
| `rbnos,rbnoo` | 2,892,288 | 493,824 | 11,130,331 |
| `biasv` (bias via 32 pk_add) | 3,080,704 | 493,824 | 13,347,630 |

So the 32 `VarAddPk` + 32 `VarScalePk` + `exp2` in the rollback body are **branch-only and never
executed on this shape**. Dynamic V-phase VALU is instead:

* conversion 64
* bias coefficient/unpack **18** (executed every tile)
* max chain 34
* decision/masks 20/17
* address/INT32 44/35
* constants/mov 23/20
* plus rollback-only 64 (not dynamic)

`biasv` confirms VALU-issue sensitivity: +23 dynamic VALU/wave costs ~12% throughput. The real
cut-1 target is only the **18-VALU coefficient chain**, not 82.

## Ranked cuts

1. **Replace the hi/lo f16 reference carry with an exact f32 reference + one f16/bf16 word for the
   QK C matrix** (vendor structure; the 4 bias f16 MFMAs stay). Delete only the 18-VALU coefficient
   unpack; the 64 pk_add/pk_mul are branch-only and give no dynamic win. **-14..-18 dynamic VALU
   (~1.5-2% wave)**. Accuracy: reference delta/O rescale stays f32-exact; only the C matrix word is
   rounded (vendor-equivalent); gate 15/15 and compare output. Risk: low-medium.
   Do NOT plan -82: the static rollback body is not executed.
2. **Move address/INT32 ring math to SALU / hoist per-tile:** **-35..-44 VALU** (vendor does it in
   ~40 SALU). Requires ring state in SGPR (vendor 99 SGPR vs our 45) and scalar pointer bumps; risk
   SGPR pressure and barrier/ring correctness. Second-highest value.
3. **Slim the decision/mask chain:** **-10..-15 VALU** (remove redundant clamp cndmask/cmp, one
   readfirstlane, fold rollback compare/select). Risk: tail/reference correctness; keep exact rollback.
4. **Hoist constants / reduce v_mov:** **-8..-12**; use SGPR/scalar immediates or move outside loop.
5. **Partner-max lowering copies:** **-2..-4** visible (swap+max already minimal); low priority.
6. **LUT/svcorr:** not on the filled long body; removing it only helps short copies/IFETCH. No filled
   VALU gain.

Realistic **dynamic** total: **-55..-75 VALU** (cut1 -14..-18, SALU/address -25..-35, decision -8..
-12, constants -8..-10); the static code shrinks more but static size is not a run-time lever
(`onecopy` measured neutral).

## Honest payoff

VALU issue is ~1.17 cycles/instruction. -65 dynamic VALU ~= -76 cycles; wave time 1349->~1273
(**~5.6%**), and only if the V phase is critical after MUBUF. The target wave is 1157 = **-14.2% /
+16-19% throughput**; the dynamic V cuts alone **cannot reach parity**; with MUBUF and a smaller M
window it may narrow the fill ratio from ~0.84 to ~0.89-0.92. M-window XDL (ours 1216 vs vendor
1088) and phase overlap remain the largest missing terms.
