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

## Ranked cuts

1. **Fold the reference/bias into the QK C operand only** (remove pk_add x32, pk_mul x32 and the
   hi/lo unpack): **-82 VALU**, at the cost of the vendor's 4 bias f16 MFMAs in M (128 XDL). This is
   exactly the vendor scheme (`x` already contains `s91-r`); p4 makes the bias coefficient work
   unnecessary. Risk: exact reference/rollback bookkeeping; accuracy should be bit-equal if the same
   bf16 bias matrix is used. **Highest-value, lowest-risk cut.**
2. **Move address/INT32 ring math to SALU / hoist per-tile:** **-35..-44 VALU** (vendor does it in
   ~40 SALU). Requires ring state in SGPR (vendor 99 SGPR vs our 45) and scalar pointer bumps; risk
   SGPR pressure and barrier/ring correctness. Second-highest value.
3. **Slim the decision/mask chain:** **-10..-15 VALU** (remove redundant clamp cndmask/cmp, one
   readfirstlane, fold rollback compare/select). Risk: tail/reference correctness; keep exact rollback.
4. **Hoist constants / reduce v_mov:** **-8..-12**; use SGPR/scalar immediates or move outside loop.
5. **Partner-max lowering copies:** **-2..-4** visible (swap+max already minimal); low priority.
6. **LUT/svcorr:** not on the filled long body; removing it only helps short copies/IFETCH. No filled
   VALU gain.

Realistic total: **-100..-115 VALU** (V 267->~155-165; below vendor 251 in count, but V is not the
whole wave).

## Honest payoff

VALU issue is ~1.17 cycles/instruction. -110 VALU ~= -129 cycles of V phase; wave time 1349->~1220
(**~9.5%**). MUBUF/filled staging may add a few percent. The target wave is 1157, i.e. **-14.2% /
+16-19% throughput**; V-window cuts alone therefore **narrow the gap but do not reach absolute
parity** unless the M window also loses XDL/waits (ours XDL 1216 vs vendor 1088) or the overlap
improves. Recommended sequence: bias fold (largest, exact) -> SALU address -> MUBUF verification ->
decision/const trims; measure wave/VALU counters after each. Expected outcome: fill ratio ~0.84 ->
~0.91-0.94, not >=1.0, on current evidence.
