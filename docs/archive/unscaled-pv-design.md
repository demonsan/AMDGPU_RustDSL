# Unscaled-PV numerics design (P scale moved into the fp8 conversion)

> Status: prototype implemented as opt-in `FMHA_OPT=p4` (numerics pass the accuracy gate; -1.2..-2.2% alone); a prerequisite for the vendor-order window, which closed negative; see round 70 of docs/archive/status.md and docs/current-status.md §5.

Status: design + prototype on `p4-unscaled` @ 4cc2cff (based kernel-opt9 p4 f5e72fe + backend-mfma
unscaled op 8db717a). Goal: allow the PV MFMAs to use `v_mfma_f32_32x32x64_f8f6f4` with **no scale
operands**, while preserving the current accuracy and enabling the vendor-style M-window staging.

## 1. Scale algebra

Current vs128 path:

```
P0      = fp8(pknorm(x))                    # conversion scale 1.0; P0 carries the 16x amplitude
O       = sum(P0 * 2^-4 * V)                # PV: v_mfma_scale ... scale_a=127, scale_b=123 (2^-4)
l       = sum(P0)                           # four sparse smfmac, D dword2=16*sum; epilogue /16
out     = (O / l) * vd_raw
```

Target (p4, matches the vendor `s89 = 0.25`):

```
P4      = 0.25 * P0                         # scalef32 scale operand = 0.25
O4      = sum(P4 * V)                       # PV MFMA UNSCALED (no scale operands)
l4      = sum(P4)                           # same 4 smfmac, D2=16*sum, epilogue /16
out     = ((O4 / l4) * vd_raw) * 1/16      # fin x 1/16
```

Why the `fin x 1/16`: removing the old PV `2^-4` scale changes the ratio by 16 (`O4 = 4 O`,
`l4 = 0.25 l`), independent of whether P's conversion scale is 0.25 or 2^-4. The 1/16 restores
`out` exactly up to fp8 rounding.

**QK is unchanged.** `sq0m = sq0 - 9`, `sq1m = sq1 - 9` already implement the vendor Q scale
`max(qscale,9) - 9` (Q lowered 2^-9). P's amplitude does not enter S.

**Row sums are unchanged.** The default `smfv` row sum is four sparse `v_smfmac` (no scale
operands), `D dword2 = 16 * sum(P)`, and the epilogue divides by 16. With P4 both `l` and the
epilogue ratio stay consistent; only the final `fin` x 1/16 is new. The `smfrs` fallback, if
enabled, must also use scale 127 for its P operand.

**V descale is unchanged.** PV accumulates the raw V tile; the per-tensor `vd_raw` descale is
applied once in the epilogue (`fin = rcp(l) * vd_raw`). There is no pre-descale of V in the PV
input; the vendor applies its V descale at output too.

**LSE shift.** `lse = log2(l) - 512*b`; P4 makes `l` 4x smaller, so `lse` shifts by -2 for every
split. The shift is split-independent and cancels in the combine weights `2^(lse_s - max lse)`;
the split partials and the combined output remain correct.

## 2. Prototype and measurements (GPU 5, own backend)

`FMHA_OPT=...,p4` (opt-in; kernel-opt9 commit f5e72fe) with `p4-unscaled` @4cc2cff:

* static: scaled MFMA 160 -> 64 (QK only), unscaled fp8 MFMA 96 (PV); vgpr 249 -> 247 for
  `_vs128`, 234 -> 232 for `_kvsplit`; output of the unscaled opcode is **bit-identical** to the
  equivalent scale-127 build.
* accuracy: 15 shapes at 64/16 rows, `gen9_gate.py` 15/15 PASS, parity 15/15, our bad 0; vs CPU
  B1H8S2048 mean 8.2875e-4 (base 8.2760e-4), worst_err identical 2.7913e-2; split B1H2S200 mean
  1.6800e-3 (base 1.6821e-3), bad 0.
* output vs base: max abs 0.0078 (1 bf16 ULP at that magnitude), mean abs 1.8e-5; 3.3% of bf16
  words differ, mostly 1 ULP (69k/2.1M on B1H8S2048). Not bit-identical, as expected from moving
  the scale into fp8 conversion; accuracy is unchanged.
* standalone speed (the reason M2 is still needed): `_vs128` B1H8S2048 443 vs 452 TFLOP/s (-2%);
  split B1H8S2048 677 vs 682 (-0.7%); filled B1H8S8192 1983 vs 1997 (-0.7%); B4H8S2048 ~1436 vs
  ~1443 (noise). **P4/unscaled alone is not the parity lever**; it is the prerequisite for the
  vendor M-window (160 ready XDL cycles, unscaled PV, no bias MFMAs).

## 3. M2 retry requirements

* M-window pinned block must use `v_mfma_f32_32x32x64_f8f6f4` (unscaled) for the 2 preloaded V
  fragments and the remaining PV, and the 4 sparse smfmac row sums; no P-scale operand.
* VGPR budget: p4 saves 2 VGPR, but the 2 preloaded V fragments need 16; base 247 + 16 = 263 > 256.
  Free >=7 VGPR (Q1 via LDS, or drop another resident fragment) before the M-window retry.
* Bias: keep the bias matrix C in the QK half; no bias MFMAs.

## 4. Handoff

Integrate the p4 scale layout (conversion 0.25, fin x 1/16, scale 127 or unscaled op) into
`_vs128`/`_kvsplit`, merge the unscaled fp8 MFMA op (backend-mfma), then retry the pinned M-window
with unscaled PV. Until then, p4 should stay opt-in; the current shipped fast-LDS path (0.25 not
applied) remains the certified default.
