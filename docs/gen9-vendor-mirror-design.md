# gen9: vendor-mirror FMHA design (from `ref_mxfp8.co` disassembly)

Status: design spec, no kernel code yet. Derived from the vendor code object
`/shared/amdgpu/home/blyu_qle/sessions/ref_mxfp8.co` (disassembly: `tools/vendor_disasm.sh`,
`llvm-objdump -d --mcpu=gfx950 --no-show-raw-insn`) and the MI350/gfx950 kernel-wiki
(`mi350-isa-low-precision-sparse`, `mi350-isa-mfma-layouts`, `mi350-arch-compute-mfma`,
`mi350-pmc-compute-mfma`). Addresses below are virtual addresses in the vendor `.co`; the vendor
kernel has named labels (`label_block_A_loop` = 0x2b68, `label_block_B_loop` = 0x6aa0,
`label_rollback_overflow_block_s_0` = 0x9d4c, ...). Our current control is Main `ee7c361`
(`_vs128` 453.4-453.5 TFLOP/s = 7.07-7.09 TF/CU, vendor same run 538-539 = 8.41 TF/CU, gap 1.19x).

Evidence grade: **V** = read from disassembly/descriptor/counters; **E** = measured hardware,
interpretation is a hypothesis; **U** = unknown, must be confirmed in M1/M2.

---

## 1. Vendor reference / softmax math (V)

### 1.1 Constants (exact bit patterns, live in `label_block_A_loop` and the four rollback copies)

| constant | value | role |
|---|---|---|
| `s26`/`s27` | `0x43ffff00` = 511.9921875 = 65535/128 | exponent-field scale: `E_bf16 = floor(65535*x/128) = floor(s26*x)` |
| `s89` | `0x3e800000` = 0.25 | scale operand of every `v_cvt_scalef32_pk_fp8_bf16` |
| `s90` | 1.0 (the 1/512 at 0x1a68 is dead) | `v_mul_f32_e64 v184, -s90, v192` = minus carried max `r` |
| `s91` | `0x3e7df17e` = 0.24799132 | bias added to every score via the bias matrix C |
| `s92` | `0x3e85d086` = 0.26135647 | lazy-rollback threshold: `v_cmp_nge_f32 vcc, s92, v193` |
| `s93` | `0x3b000080` = 1/512 (1+2^-16) | `r += s93 * s26 * delta` (product 0.9999997) |
| Q scale | `v236 = max(qscale,9) - 9`, 0x1d50/0x1d54 | Q operand lowered by 2^-9; QK output S is `K*Q/512` |
| `s94` | 0/1 | seed-phase flag (no numerics) |

### 1.2 Fast-path formula (V)

```
x    = S + (s91 - r)                     // S = K*Q^T with Q lowered 2^-9; r = carried f32 row max
x    = x                                 // already includes the bias matrix through the QK C operand (see 1.3)
u16  = v_cvt_pknorm_u16_f32(x)           // = round(65535 * clamp(x,0,1))
bf16 = reinterpret_u16(u16)              // E = floor(65535*x/128), m = low 7 bits
P    = s89 * bf16 = 0.25 * bf16         // ~= 0.24513 * 2^(s - s_max)
```

i.e. the **Schraudolph piecewise-linear exp2** (`2^f ~ 1+f`, error 0..+6.2%, mean +4.0%). The vendor
does **not** correct it: the whole binary has **zero `v_pk_sub_u16`, zero `v_perm_b32`, no u16 mask/LUT
correction**. The long-sequence path and all four rollback copies use the identical 32× pknorm +
32× scalef32(s89) pair. There is no hidden "better scaling" behind s89/s91/s92; it is the same
approximation. (Our current `svcorr` LUT corrects it and must be removed for gen9; it is dead code for
long sequences anyway.)

### 1.3 How bias reaches S - the QK C operand (V)

The vendor never executes a per-tile bias MFMA into each S tile. Instead:

1. `r` (v192) is an f32 row reference, initialised `-inf` (`v_mov v192, 0xff800000`, 0x1e90) and
   updated only on a rollback.
2. On every 128-key step the vendor builds **two bf16 bias words** from `bias = s91 - r`
   (`v_sub_f32 v184, s91, v192`, 0x27a4; repacked at 0x27cc/0x27e4 and in every rollback block
   at 0xa1a8-0xa1e8): a hi word `v244 = bf16(hi + lo)` and a residual `v245`, i.e. ~16 bits of
   precision, not a single f16 word. `v197 == r` is special-cased to bias 1.0
   (`v_cndmask_b32_e64 v184, v184, 1.0, s[66:67]`, 0xa1b8).
3. A single **bf16 32x32x8 MFMA** builds the bias matrix in VGPRs v[100:115]:
   `v_mfma_f32_32x32x8_bf16 v[100:115], v[242:243], v[244:245], 0` (0x3098). v[242:243] are
   bf16 ones (`v242=0x3f803f80`, v243=0x3f80). This is 32 XDL cycles per 128 keys - not 4×32.
4. The first QK scaled MFMA of each S tile takes **v[100:115] as its C operand**:
   `v_mfma_scale_f32_32x32x64_f8f6f4 v[52:67], v[20:27], v[4:11], v[100:115], v238, v236`
   (0x30bc); the second half accumulates into v[52:67] (0x3108). The QK output S therefore already
   equals `K*Q/512 + (s91-r)`; the softmax max/conversion needs **no add**.

### 1.4 Rollback (V, `label_rollback_overflow_block_s_0` = 0x9d4c)

Triggered when `v_cmp_nge_f32 vcc, s92, v193` fails after a **cross-lane max**
(`v_permlane32_swap_b32 v184, v193`; `v_max_f32 v193,v193,v184` at 0x9d58-0x9d5c). The first
compare in the fast path (0x2e84 region) is the same test before the per-step overflow branch.

```
delta = max - s91                       // v_sub_f32_e64 v188, v193, s91; v_cndmask 0 if below threshold
S    -= delta                           // 64 × v_sub_f32, v52..v115 (0x9d74..0x9e70)
r    += s93 * s26 * delta               // v184 = 0x43ffff00*delta; v184 *= s93; v_add_f32 v192,v192,v184
O    *= v_exp_f32(-s26 * delta)         // v184 = 0xc3ffff00*delta; v_exp_f32 v188; 32 × v_pk_mul_f32
                                        // over O v116..179 and row sums v224..227 (0x9e84-0x9fa0)
bias  = s91 - r; repack v244/v245 hi+lo // 0xa1a8-0xa1e8
```

`v_exp_f32` computes 2^x; `-s26*delta` is `-512*delta`, so O is rescaled by 2^(-512 delta) exactly
(not through a fractional f16 bias). This is why the vendor's reference path is accurate
(mean 2.5e-3) despite the crude P approximation: the **reference is f32 and reconciled exactly**;
only the exponent conversion is approximate, and its error is a multiplicative bias that cancels in
softmax to first order.

### 1.5 What gen9 must compute / what our kernel removes

Vendor-equivalent arithmetic for gen9:

* f32 carried `r` per lane, updated only by the rollback (`r += ~delta`), never quantised to f16.
* one `v_mfma_f32_32x32x8_bf16` per 128 keys to materialise `ones × (s91-r)` bias matrix (hi+lo
  bf16); first QK MFMA of each S tile uses it as C.
* QK scaled MFMA scale operands exactly as vendor: K-scale bytes v238..v241 from LDS aux ring,
  Q-scale bytes v236/v237 built from `max(qscale,9)-9`; Q lowered 2^-9.
* P = 32× `v_cvt_pknorm_u16_f32` + 32× `v_cvt_scalef32_pk_fp8_bf16` with s89=0.25; no LUT/LUT
  correction, no u16 perm/sub.
* rollback: cross-lane max, `delta=max-s91`, 64 f32 subtracts, f32 r update, `v_exp_f32` O rescale,
  bias repack; rare path only.

Removed relative to our current `_vs128`: the 4 per-tile f16 bias MFMAs (128 XDL), the `svcorr`
LUT path and its short/long code copies, the per-unit exact rebase with threshold 120/512 and
`bias2` f16-word quantisation, the per-tile bias coefficient chain, and the Q scale -9 not being
applied to Q (vendor lowers Q once). Keep our K/V DMA staging until M4 replaces it.

---

## 2. M-window staging - the core to replicate (V, addresses from `label_block_A_loop`)

### 2.1 Pre-barrier (V window writes the operands the M window needs)

At the end of the softmax phase, after P conversion, the vendor issues the **first two V-fragment
groups before the barrier** (0x2e88-0x2ec0):

```
0x2e84 s_addk_i32 s68, 0x80
0x2e88 ds_read_b64_tr_b8 v[20:21], v219
0x2e90 ds_read_b64_tr_b8 v[22:23], v219 offset:256
0x2e98 ds_read_b64_tr_b8 v[24:25], v219 offset:512
0x2ea0 ds_read_b64_tr_b8 v[26:27], v219 offset:768
0x2ea8 ds_read_b64_tr_b8 v[28:29], v219 offset:8320
0x2eb0 ds_read_b64_tr_b8 v[30:31], v219 offset:8576
0x2eb8 ds_read_b64_tr_b8 v[32:33], v219 offset:8832
0x2ec0 ds_read_b64_tr_b8 v[34:35], v219 offset:9088
0x2ec8 s_waitcnt vmcnt(0)
0x2ed0 s_setprio 1
0x2ed4 s_barrier
```

The 8 transposing reads load **two complete V fragments** (fragment for O tile 0 = v20:27, for O
tile 1 = v28:35; each `ds_read_b64_tr_b8` pair produces 4 VGPR, two reads per fragment half), and
P halves v52:59 / v60:67 are already in registers from the softmax conversion. The DMA of the next
K/V tile was issued earlier in the compute group and is awaited by `s_waitcnt vmcnt(0)` **before**
the barrier, so the barrier itself has no DMA wait.

### 2.2 Post-barrier - exactly the first matrix work (V)

```
0x2edc s_waitcnt lgkmcnt(4)                       // 4 in-flight LDS reads remain (the last V frags)
0x2ee0 v_smfmac_f32_16x16x128_fp8_fp8 v[224:227], v[228:231], v[52:59], v232   // row sum half0
0x2ee8 v_smfmac_f32_16x16x128_fp8_fp8 v[224:227], v[228:231], v[60:67], v232   // row sum half1
0x2ef0 v_mfma_f32_32x32x64_f8f6f4 v[116:131], v[20:27], v[52:59], v[116:131]   // PV O0, P half0
0x2ef8 ds_read_b64_tr_b8 v[36:37], v219 offset:32
0x2f00 ds_read_b64_tr_b8 v[38:39], v219 offset:288
0x2f08 ds_read_b64_tr_b8 v[40:41], v219 offset:544
0x2f10 ds_read_b64_tr_b8 v[42:43], v219 offset:800
0x2f18 s_waitcnt lgkmcnt(4)
0x2f1c v_mfma_f32_32x32x64_f8f6f4 v[116:131], v[28:35], v[60:67], v[116:131]   // PV O0, P half1
0x2f24 ds_read_b64_tr_b8 v[44:45], v219 offset:8352
...
```

**Rule (the thing that removes the bias MFMAs):** the first two row-sum SMFMACs consume P from
registers (no LDS), and the first two PV MFMAs use **A=V fragments v20:27/v28:35 preloaded in the
V window** and B=P registers; they have **zero LDS dependency**. Thus the M window starts with
2×16 + 2×64 = **160 XDL cycles of useful, already-ready matrix work**, while the next V fragments
(v36:43, v44:51, ...) are read with `s_waitcnt lgkmcnt(4)` and cover 4-read groups. There is no
bias MFMA and no exposed post-barrier LDS latency. Our gen9 must reproduce exactly this:
preload 2 V fragments before the barrier (16 VGPR), issue 2 row-sum SMFMACs + 2 unscaled PV MFMAs
immediately after the barrier, then stream the remaining 6 V fragments and waits.

### 2.3 Remaining M-window (V, 0x2f24-0x3248)

After the first two PV MFMAs the vendor streams the remaining 6 V-fragment groups:
`ds_read_b64_tr_b8` x4 at +0x20/+0x120/+0x220/+0x320 style offsets from v219, then
`ds_read_b64_tr_b8` x4 at 0x2080+0x20 offsets (second half); every group is followed by
`s_waitcnt lgkmcnt(4)` and two unscaled `v_mfma_f32_32x32x64_f8f6f4` (D v132:147, v148:163,
v164:179) with P halves v52:59/v60:67. 8 PV MFMAs total per 128 keys.

Then the QK half of the M window (0x3004 onward):

* `s_cselect_b32 s66, 0xffff7e00, s66` / `s_add_u32 s98,s66,s98` / `v_add_u32 v219,s66,v219`:
  ring-parity address bump **in scalar**, only a single VALU add for the lane part.
* `ds_read_b128 v[20:23], v218` and `v[24:27], v218 offset:32` (K, 16B/lane, first half);
  `ds_read_b32 v238..v241, v184` (K-scale aux, offsets 0x0/0x80/0x100/0x180);
  `ds_read_b128 v[28:31],v218 offset:0x1040` etc (second K half).
* `v_mfma_f32_32x32x8_bf16 v[100:115], v[242:243], v[244:245], 0` (bias matrix, 32 XDL).
* 8 scaled QK MFMAs, first of each half uses C=v[100:115], second accumulates:
  `v_mfma_scale_f32_32x32x64_f8f6f4 v[52:67], v[20:27], v[4:11], v[100:115], v238, v236`
  (0x30bc), `v[52:67], v[28:35], v[12:19], v[52:67], v238, v237` (0x3108),
  `v[68:83] ...`, `v[84:99] ...`, `v[100:115] ...` (0x3150-0x3234).
* `s_setprio 0; s_barrier` at 0x3244/0x3248 closes the M window; the next 128-key step begins.

The DMA is interleaved **inside this QK half**: `s_add_u32 m0,s71,imm` + `buffer_load_* ... lds`
at 0x3130/0x3168/0x319c/0x31d0/0x3204, so the next tile is in flight while the current QK executes.

---

## 3. V window - ~251 VALU composition and order (V)

Per 128-key step, fast path (0x2b68-0x2e84):

1. `s_setprio 0`; `v_mul_f32_e64 v184, -s90, v192` (bias/reference negate; s90=1.0).
2. **32× `v_max3_f32`** over S v52..v115 in pairs (`v193` running): 64 scores/lane → one row max.
3. `v_cmp_nge_f32_e32 vcc, s92, v193; s_nop 1; s_cbranch_vccnz
   label_rollback_overflow_block_s_N` (lazy rollback; rare path goes to 0x9d4c, 191 VALU).
4. **32× `v_cvt_pknorm_u16_f32`**: `v52,v53 -> v52`, `v54,v55 -> v53`, ..., `v114,v115 -> v83`.
   `x` already contains `s91-r` from the QK C bias matrix; no add.
5. **32× `v_cvt_scalef32_pk_fp8_bf16`**: `v52,v52,s89` then `v52,v53,s89 op_sel:[0,0,1]` etc,
   producing 32 packed fp8 dwords (P halves v52:59/v60:67 get reused by PV/SMFMAC).
6. Address/scalar work around it is `s_add_u32`/`s_addk_i32`/`s_cselect_b32`/`s_waitcnt` (≈40 SALU/
   iteration total), plus `v_mov` constants and the V-fragment preloads.

That is ~64 conversion VALU + 32 max3 + compare/branch + address/loop ≈ 100-110 per step, not 251;
the vendor's 250.9 `SQ_INSTS_VALU`/wave-iteration includes both 2x-unrolled steps (two 128-key
steps per loop body) and the rare-path/seed instructions, while the hot path has no other VALU.
Our current `_vs128` has 353 VALU/wave-iteration: the difference is our per-tile bias coefficient
chain (~20), svcorr LUT path (~64, short-copy only but code/IFETCH), rollback S/O shift code
(rare), and INT32 address math that the vendor does in SALU (99 SGPR vs our 45).

Rollback block composition (0x9d4c-0xa1f0, **rare**): 1 permlane+max+cmp, 64 `v_sub_f32`, 2
`v_mul_f32` + 1 `v_add_f32` for r, 1 `v_mul_f32`+1 `v_exp_f32`, 32 `v_pk_mul_f32` for O/rowsum,
32 pknorm+32 scalef32 for P, bias repack (sub/pack/lshl/sub/pack/sub/lshl/sub/pack). It is not
executed on the benchmark shape in steady state; budget it as branch-only.

---

## 4. Ring / DMA / barrier / priority (V/E)

* **Budget**: vendor 246 VGPR, **99 SGPR**, 84224 B LDS per wave; 8 waves/CU, 2 waves/SIMD.
  Ring/address state lives in SGPR (52 spare SGPR vs our current 45-48); per-iteration scalar work
  is `s_add_u32`/`s_cselect_b32`/`s_addk_i32` (40 SALU/wave-iteration), not VALU INT32.
* **2x unrolled outer loop**: one loop pass = two 128-key steps, **4 block-wide barriers per pass**
  (2 per step). Role A at 0x2b68, role B mirror at 0x6aa0; both execute the same phase sequence and
  meet at the barriers. Drain1/drain0 labels peel the last steps to avoid prefetching past K/V end;
  `label_dense_short_schedule` is the short-sequence path.
* **Staging**: `buffer_load_dwordx4 ... lds` with an SGPR pair base and `m0` LDS destination
  immediate (no VGPR base, no ds_write); per 128-key step the wave loads 4×1040-B runs for K/V
  (M0 immediates 0x0/0x410/0x1000/0x1410 for the K tile half and 0x4100/0x4510/0x5100/0x5510 for
  V; a second aux region at 0x14500/0x14700 for K scales). Scales are
  `buffer_load_dword ..., lds` with an EXEC mask writing one dword per lane. Double-buffered ring:
  parity from `s66`/`s67` with `s_cselect_b32 s66,0xffff7e00,s66` and `s_add_u32`/`v_add_u32`
  bumps; no descriptor arithmetic per tile.
* **Barrier placement**: exactly 2 barriers per 128 keys per wave: **barrier 1 after P conversion**
  (`s_setprio 1` immediately before it, 0x2ed0/0x2ed4) and **barrier 2 after PV+QK**
  (`s_setprio 0`, 0x3244/0x3248). The block-wide barrier is shared by A/B role groups; the two
  wave groups are offset by a half tile so one group's V phase overlaps the other's M phase.
* **setprio ping-pong**: role A sets priority 1 before barrier1 / 0 before barrier2; role B mirror.
  Priority must be complementary between the two role groups.
* **Which tile is in flight**: during V(t) the DMA for K/V(t+1) (issued in the previous M/QK half)
  is in flight; `s_waitcnt vmcnt(0)` before barrier1 retires it; the DMA for K/V(t+2) is issued
  inside the QK half of M(t) (`s_add_u32 m0` + `buffer_load_* lds` at 0x3130-0x3204) and overlapped
  with QK. This is the exact staging rule gen9 must copy; our current kernel stages through VGPR
  (`global_load_lds` is fine) but its ring state is VALU/lane-based and it has 7.9 VMEM/wave-iter
  vs vendor 5.7.

---

## 5. Implementation plan (milestones, measurable gates)

New experimental kernel name **`fmha_fwd_hd128_mxfp8_vs9`** (by-name emit, not in default auto
path until M5). Do not touch `_vs128`/`_t128`/guard; keep the harness as acceptance.

### M1 - reference math parity (no performance gate)
Implement vendor reference scheme inside the existing `_vs` skeleton:
- f32 carried `r` (v192) + cross-lane rollback (`s92`, `delta=max-s91`, 64 f32 sub, `r += s93*s26*delta`,
  `v_exp_f32(-s26*delta)` O rescale);
- one `v_mfma_f32_32x32x8_bf16` bias matrix per 128 keys from hi/lo bf16 words, used as C of the first
  QK MFMA of each S tile;
- QK scales exactly (K-scale bytes from LDS aux, Q scale `max(qscale,9)-9`, Q lowered 2^-9);
- P = 32× pknorm + 32× scalef32(0.25); **no svcorr, no LUT**; drop our bias2 f16 chain and threshold 120/512.
Gate: build/gates pass; 15 shapes mean_err ≤ **2.6e-3**, bad count ≤ vendor's own bad count on the
same shape/rows; no NaN/Inf; determinism 5/5×3. Performance may fall (acceptable; no gate).

### M2 - M-window staging with pre-read operands (the core)
- Preload first 2 V fragments (16 VGPR) in the V window after P conversion, before barrier1.
- After barrier1: issue 2 row-sum SMFMACs + 2 unscaled PV MFMAs using preloaded V/P registers
  (exact 0x2ee0-0x2f1c order), then stream remaining V fragments with `s_waitcnt lgkmcnt(4)`.
- Remove the 4 f16 bias MFMAs from the hot path; bias matrix C only in QK half.
- Keep current DMA for this milestone (or existing `saddrlds` opt-in).
Gate: per-CU **≥ 460 TFLOP/s** at 64 blocks (vs 453-455 today); XDL/wave-iter ≈ 1088-1150;
WAIT_ANY ≤ 500; accuracy unchanged; md5/ULP gate vs M1 output (M1/M2 should be bit-identical to each
other on the same math).

### M3 - conversion chain / VALU
- Match vendor softmax fast path exactly: 32 max3 + cmp s92 + branch + 32 pknorm + 32 scalef32(0.25),
  no per-tile bias unpack, no svcorr code copies; f32 r and bias repack only in rollback.
- Apply Q scale once, not per tile.
- Optional: `saddrlds`/`pmdead` (already opt-in, +0.5-0.8%) and SALU address hoist.
Gate: per-CU **≥ 480 TFLOP/s**; `SQ_INSTS_VALU` ≤ 270/wave-iter; IFETCH ≤ 90; accuracy ≤ M1 bar.

### M4 - staging/DMA/ring
- Replace VGPR-round-trip staging with vendor-style `buffer_load_dwordx4 ... lds` (SGPR base + m0
  immediate), 4×1040-B runs per 128 keys per wave, aux K-scale region, double-buffered parity ring
  in SGPR, `s_cselect`/`s_add` address bumps; no ds_write.
- Keep 2 barriers/128 keys; setprio ping-pong; half-tile A/B offset.
Gate: per-CU **≥ 510 TFLOP/s**; VMEM level ≤ 60, VMEM inst ≤ 6/wave-iter; IFETCH ≤ 85; WAIT_ANY ≤ 450;
LDS 52-55; accuracy/determinism unchanged.

### M5 - 2x unroll + final schedule, shipped-path decision
- 2x-unrolled loop (two 128-key steps/pass) with drain labels; exact barrier/priority schedule;
  register budget test (Vendor 246 VGPR/99 SGPR; if spills, stop at M4).
Gate: **per-CU ≥ vendor 8.4 TF/CU** (538 TFLOP/s at 64 blocks) on a quiet node; interim fallback:
best of M2-M4 if M5 regresses. If M5 fails, gen9 does not replace defaults; document measured best.

**Failure modes / risks** (each milestone): M1 accuracy regression (vendor approximation + our
reference differences) - fallback keep vendor constants but reuse our exact rebase; M2 register
pressure (16 VGPR preload; vendor uses 246; our 249 -> must free/spill) - reduce preload to 1
fragment if needed; M3 VALU scheduling changes - per-milestone A/B, no md5 gate across math changes;
M4 DMA descriptor/ring correctness - differential vs current staging on 15 shapes; M5 scheduling
instability - stop at M4, no default swap without per-CU ≥ vendor.

---

## 6. Acceptance criteria (gen9, final)

1. **Per-CU >= vendor**: `_vs128`-equivalent 64-block B1H8S2048 ≥ **8.4 TF/CU** (≈538 TFLOP/s),
   same-run vendor, ≥12 interleaved rounds × 200 iters, `.s` provenance; interim bars 460/480/510.
2. **Accuracy** (vendor-class accepted): mean_err ≤ **2.6e-3** vs CPU (our current 7.4e-4 may be
   traded); **bad count ≤ the vendor's own `REF_KERNEL_FAIL` count on the same shape/rows**; no
   NaN/Inf; 15 gate shapes at 16 and 64 rows; parity gate vs vendor mean error within 10%.
3. **Determinism**: 5/5 ×3 (or stronger), output identical across runs for the new math.
4. **15 shapes** pass; shipped auto path gate 15/15 @16 rows; 64-row vendor-fail count no worse.
5. **Counters** (target): XDL/wave-iter ≤ 1100, VALU ≤ 265, SALU ≥ 35, LDS 52-55, VMEM inst ≤6,
   IFETCH ≤85, WAIT_ANY ≤450, level-VMEM ≤60.
6. New kernel is opt-in until all bars met; shipped `_vs128`/`_t128`/guard untouched until then.
   Output md5 is not an invariant for gen9 (math changes); use mean/bad/ULP + determinism.

---

## 7. Keep / replace map against current code

| current piece | gen9 action |
|---|---|
| K/V DMA `global_load_lds` + LDS ring (`dbuf`,`kdlds`) | keep until M4, then vendor SADDR `buffer_load_* lds` + SGPR ring |
| 4×f16 bias MFMAs | **remove**; replace with one bf16 ones×bias MFMA (C of first QK) |
| bias2 f16 word / `bias2w` / threshold 120/512 | **remove**; f32 `r`, s92 threshold, s91 bias matrix |
| svcorr LUT / short-long code copies / `svpk` | **remove** for gen9 (`svcorr` only existed to fix the approximation; vendor has no correction) |
| P conversion `cvt_pknorm_p8x4` | keep shape (2 pknorm + 2 scalef32) but use s89=0.25, no LUT perm/sub |
| row-sum 4×smfmac | keep, use P registers and place first after barrier1 |
| Q loads `load_b128`/`qdw` | keep (already bit-identical Q) |
| guard/auto dispatch, combine8h, ABI | unchanged; gen9 is a new by-name kernel |
| Q scale | apply `max(qscale,9)-9` once, lower Q 2^-9; QK scale operands v236/v237 |

Tools: `tools/vendor_disasm.sh` (survey/phase/mfma/lds/staging), `tools/vendor_counters.sh`;
reproduce section addresses with `llvm-objdump -d --mcpu=gfx950 --no-show-raw-insn`.

## 8. Implementation status (measured, GPU 5)

First implementation attempt: kernel-opt4 commit `e2e1733` (M1 "bias path" + M2 pinned M-window
`pv_qk_window_block_v2`; `nom2` restores the scaffold). Independent gate, GPU 5, `_vs128` base in the
same run (vendor 538.4 TFLOP/s), 10 interleaved rounds x 200 iters, `REF_WARMUP=300`, md5 checked:

| variant | vgpr | wave | XDL | VALU | WAIT_ANY | best / median TF |
|---|---|---|---|---|---|---|
| `_vs128` base | 249 | 1349.2 | 1216 | 353.1 | 484.8 | **454.7 / 453.1** |
| `_vs9` M1+M2 | 256 | 1501.2 | 1130 | 378.7 | 522.9 | 454.7 / **410.8 (-9.3%)** |
| `_vs9,nom2` | 249 | - | - | - | - | 454.7 / 453.0 |

filled B4H8S2048: base 1417.2/1410.2, `_vs9` 1384.9/1341.8 (-5% median). Accuracy gate: 15/15 shapes
at 64 and 16 rows, parity 15/15, bad 0, det 5/5x3 - **but the output md5 is still `faadf00c` identical
to `_vs128` and the mean is unchanged 8.276e-4, so the literal vendor reference math (f32 `r`,
`s91-r` bias matrix, `s92`/`s93` rollback, Q scale -9, P `scalef32`) is NOT in this binary**; only the
bit-identical `biasacc` path was enabled. kernel-opt's round-68 position is that M1 is "nothing to
implement" because our carried reference is already exact hi/lo f16 (2^-22), the rollback is the same
algorithm, and our accuracy is ~3x better than the vendor; **that is an accuracy-equivalence claim, not
the literal vendor algorithm** - if the M1 gate is vendor-class accuracy it is met by the scaffold; if
it is vendor-equivalent math it is not implemented. Either way the performance-relevant M2 change (one
bias MFMA instead of four + preloaded V) was implemented and lost.

Further M2 commits (`544f290` interleave first reads with register-fed MFMAs, `78154cd` Q-from-LDS
window variant) do not recover: resident Q 410.8 (our independent gate), `noqreg` **421.9 vs base
444.3 = -5.0%**, interleave unchanged; filled -5%. Best 421.9 is still 38 TFLOP/s short of the 460
bar. The pinned block carries 24 NOPs (vs 10) of explicit wait padding, 58 DS reads (vs 52, the Q
fragments), K-scale `lshr`/address work inside the pinned region, an `lgkmcnt(0)` entry wait, and uses
scaled PV MFMAs (c127/c123) instead of the vendor's unscaled PV. Recommendation: stop at
`nom2`/shipped state; do not change defaults. If one bounded attempt is authorized, the untried levers
are unscaled PV (numerics change), vendor per-step bf16 bias instead of the persistent 16-VGPR matrix,
and moving K-scale/address work out of the pinned region.
