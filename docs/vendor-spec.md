# Vendor reference kernel specification (`/shared/amdgpu/home/blyu_qle/sessions/ref_mxfp8.co`)

Purpose: characterise the vendor (aiter) `fmha_fwd_hd128_mxfp8` gfx950 kernel precisely enough to
(a) guide the `fmha_fwd_hd128_mxfp8_t` rewrite and (b) serve as the acceptance checklist for it.
Everything here is derived in this session from the code object and hardware counters on MI355X
(gfx950), GPU 5, node 53. Branch `kernel-occ`, base `origin/Main`.

Evidence grades used below:

* **Verified** - read directly from the disassembly / descriptor / counters (reproducible with the
  commands in section 9).
* **Experimental** - measured on hardware but with an interpretation that is a hypothesis.
* **Unknown** - stated explicitly, not guessed.

The vendor `.co` is never committed; only derived numbers and the commands that produce them are.

## 0. Method

```bash
export REF_CO=/shared/amdgpu/home/blyu_qle/sessions/ref_mxfp8.co
bash tools/vendor_disasm.sh survey                 # summary/labels/loops/barrier+setprio map
bash tools/vendor_disasm.sh phases label_block_A_loop --seq
bash tools/vendor_disasm.sh mfma | lds | staging | branches
OURS_CO=build/ours_t.co bash tools/vendor_counters.sh vendor-spec   # counter table vendor vs ours
```

`tools/vendor_disasm.sh` disassembles with `llvm-objdump -d --mcpu=gfx950 --no-show-raw-insn` and
runs `tools/vendor_analyze.py` on the dump; the vendor kernel is unusually friendly because it is
built from hand-written aiter assembly with named labels (`label_block_A_loop`,
`label_rollback_overflow_block_s_0`, ...). 9189 static instructions, 37 labels.

Both tools were validated end-to-end on GPU 5 / node 53 (branch `kernel-occ` head `fddc0e9`): the
disassembly survey/phases output above and the counter table in section 6 are their output.

## 1. Budget (Verified)

From the code-object metadata (`.vgpr_count` / `.sgpr_count` / `.group_segment_fixed_size`; Verified,
read out of the .co ELF, reproducible with the snippet in section 9):

| property | vendor | ours (`fmha_fwd_hd128_mxfp8_t`, default llvm-tail) |
|---|---|---|
| block / grid | 512 threads, `grid = (S/256, H, B)` = 64 blocks | same (harness-identical) |
| VGPR per wave | **246** (descriptor; 8 waves x 246 = 1968 of 2048 VGPR/CU) | **256** (8 x 256 = 2048, i.e. the full VGPR budget) |
| SGPR per wave | **99** (8 waves x 99 = 792 of 800 SGPR/CU - the vendor uses the whole SGPR budget to keep ring/address state scalar) | **48** (8 x 48 = 384; ~416 SGPR/CU unused) |
| LDS | **84224 B** (82.4 KB) | **69632 B** (68 KB) |
| kernarg | 656 B | 656 B |
| wavefront | 64 | 64 |
| waves per CU / SIMD | 8 / 2 | 8 / 2 |
| register split (live ranges) | O accumulators 4x16 = 64 VGPR (26%); S^T tiles 64; V frags 32; K frags 16; l/rowsum state 24; staging ~5; addresses/ring | O 64; S 64; P 16; V/K frags; staging ~32 (register round trip); addresses/ring in VALU |

Key budget observation for the rewrite: the vendor spends **99 SGPR** (vs our 48) on ring pointers,
descriptors and loop state, which is why its per-iteration scalar work is `s_add_u32`/`s_cselect_b32`
(40 SALU wave-instructions/iteration) while ours appears as VALU INT32 (43/iteration). We have 52 SGPR/wave
of headroom (up to 99) to do the same.

## 2. Phase structure of one outer iteration (Verified)

One outer iteration consumes **128 keys** (`SK/128 = 16` iterations for SK=2048). The vendor loop is
**2x unrolled**: one pass of the loop body = two 128-key steps, with 4 block-wide barriers per pass
(2 per step). The role A path is at `label_block_A_loop` (0x2b68); the role B path is a mirror copy at
`label_block_B_loop` (0x6aa0) reached by an early role branch; both execute the same phase sequence
and meet at the same barriers. Epilogue `label_block_A_drain1/drain0` (and B equivalents) peel the
last steps to avoid prefetching past the end of K/V; `label_dense_short_schedule` handles the short
sequence path.

Per 128-key step, program order (role A path):

| # | phase | address | content (instruction counts are static, per wave) |
|---|---|---|---|
| 1 | **softmax** | `label_block_A_loop` 0x2b68 | `s_setprio 0`; 32x `v_max3_f32` running max over 64 values/lane (the 4 S^T tiles = 128 keys x 32 rows); `v_mul_f32_e64`, lazy-rollback threshold compare `v_cmp_nge_f32 vcc, s92, v193` + `s_cbranch_vccnz` to the rare rescaler; fast path: 32x `v_cvt_pknorm_u16_f32` + 32x `v_cvt_scalef32_pk_fp8_bf16` (Schraudolph-style exp + FP8 pack, **no `v_exp` on the fast path**) |
| 2 | **boundary 1** | 0x2ed0/0x2ed4 | `s_setprio 1` + `s_barrier` - P published, tile handoff, priority switches to the MFMA phase |
| 3 | **compute group** | `label_rollback_done_block_s_N` | `s_waitcnt lgkmcnt(4)`; 2x `v_smfmac_f32_16x16x128_fp8_fp8` (row sums, accumulate into the same D); 8x **unscaled** `v_mfma_f32_32x32x64_f8f6f4` (PV: 4 O tiles x 2 key halves); 1x `v_mfma_f32_32x32x8_bf16` (l helper); 8x `v_mfma_scale_f32_32x32x64_f8f6f4` (**QK for the next 128 keys**: 4 S^T tiles x 2 chained K halves). Interleaved: 32x `ds_read_b64_tr_b8` (V fragments), 16x `ds_read_b128` (K fragments), 4x `ds_read_b32`; the next tile is staged by `buffer_load_dwordx4 ... lds` x4 + `buffer_load_dword ... lds` (K scales); LGKM waits after each 2-read group feeding an MFMA; ring addressing via `s_cselect_b32`/`s_add_u32` |
| 4 | **boundary 2** | 0x3244/0x3248 | `s_setprio 0` + `s_barrier` |
| 5 | next | fallthrough | the next step's softmax starts immediately (no branch), then compute group, `s_setprio 0` + barrier at 0x3924/0x3928, `s_branch label_block_A_loop` |

So per 128 keys: **2 barriers, one `s_setprio 1` before the MFMA/PV group and one `s_setprio 0` at the
following barrier, 1 softmax block, 1 compute group (2 smfmac + 9 MFMA + LVDS/VMEM)**. Static totals:
48 `s_setprio`, 35 `s_barrier`, 333 `s_waitcnt` (mostly `lgkmcnt(4/2/0)`).

**Why the two waves on a SIMD are offset (Verified static, Experimental mechanism):** the barriers are
block-wide (all 8 waves), so both role waves reach each boundary together; the `s_setprio` window is
placed exactly over the MFMA/PV phase (`setprio 1` at the start, `setprio 0` at the next barrier). With
2 waves/SIMD the wave that is inside its MFMA phase issues with priority while the other wave, which is
in its softmax/VALU phase or waiting at the barrier, yields. The A and B role paths are mirror images,
so on a SIMD the two resident waves are in complementary setprio states - the "ping-pong". This is
**not** two independent softmax-phase shifts; it is priority-steered interleaving at the two barriers.

## 3. MFMA inventory and cycle accounting (Verified opcodes, cycle table source-reported)

Per 128-key step per wave:

| role | opcode | count | wiki MI350 cycles each | cycles |
|---|---|---|---|---|
| QK | `v_mfma_scale_f32_32x32x64_f8f6f4` (block-scaled, both operand scales) | 8 (4 S^T tiles x 2 chained K=64 halves) | 64 (F8 involved) | 512 |
| PV | `v_mfma_f32_32x32x64_f8f6f4` (unscaled) | 8 (4 O tiles x 2 key halves) | 64 | 512 |
| row sum | `v_smfmac_f32_16x16x128_fp8_fp8` (sparse FP8) | 2 | 16 | 32 |
| l helper | `v_mfma_f32_32x32x8_bf16` | 1 | 32 | 32 |
| **total XDL cycles / wave / 128 keys** | | **19** | | **1088** |

Per SIMD (2 resident waves) that is **2176 XDL cycles per 128-key iteration**. Measured wall per
iteration is vendor **3.6k cycles** and ours **7.3k cycles** (same run, same shape, B1 H8 S=SK=2048;
~11 us launch overhead removed, 2.4 GHz nominal). Implied XDL pipe occupancy: vendor ~60 %, ours ~30 %.
**Our kernel executes the same 16 QK/PV MFMAs (all as `v_mfma_scale_...`, 64 cycles each = 1024) and a
row-sum path; the 2x gap is therefore not MFMA work.** The rewrite must recover ~3.7k cycles/iteration
of VALU / LDS-latency / barrier / issue overhead, not tensor throughput.

Static MFMA histogram of the whole vendor binary (all paths incl. role B, drains, rare rescalers):
`v_mfma_f32_32x32x64_f8f6f4` x192, `v_mfma_scale_f32_32x32x64_f8f6f4` x128,
`v_smfmac_f32_16x16x128_fp8_fp8` x48, `v_mfma_f32_32x32x8_bf16` x3. The 16 QK/PV per step is what the
counters see; the rest is the other role path, the peeled drains, and the rare rollback blocks.

## 4. Softmax path (Verified)

Exact fast-path sequence for one 128-key step (per wave, 64 f32 scores/lane = 4 S^T tiles):

1. `v_max3_f32` chain: 32 instructions accumulate a running max over the 64 values (`v_max3 d, a, b, d`),
   preceded by `v_mul_f32_e64 v184, -s90, v192` (scale for the row max / lazy threshold state).
2. Lazy-rollback decision: `v_cmp_nge_f32_e32 vcc, s92, v193` then `s_cbranch_vccnz` to
   `label_rollback_overflow_block_s_{0,1,2,3}`. Granularity: **one check per 128 keys** (per step),
   not per 64. `s92` is the lazy threshold (running max + tau); the overflow blocks are 1192 B / 191
   VALU each (4 of them; they contain the only `v_exp_f32` uses in the kernel) and rejoin the PV phase
   at `label_rollback_done_block_s_*`. This is the "lazy rescale, pay only when max grows past tau"
   scheme.
3. Fast quantization to FP8 P: 32x `v_cvt_pknorm_u16_f32` (pack 2 f32 -> 2 u16, the Schraudolph exp2
   approximation scaled by the max) + 32x `v_cvt_scalef32_pk_fp8_bf16` (u16 + E8M0 block scale -> 2
   FP8 E4M3 values, low/high halves merged by `op_sel`). Result = 64 FP8/lane = P for 128 keys x 32
   rows.

**Row-sum path (this decides the smfmac question): YES, smfmac is used, and the SPARSE operand is A.**
Verified operands: `v_smfmac_f32_16x16x128_fp8_fp8 v[224:227], v[228:231], v[52:59], v232` and the same
with `v[60:67]`; `v[224:227]` (4 VGPR = 16x16 f32 tile) is the accumulated D; `v[228:231]` (4 VGPR) is
the compressed **A** operand initialised to fp8 ones (`0x38383838`, masked by the role condition);
`v232` = SRC2 index VGPR initialised to **`0x44444444`** (2-bit fields = select positions 0 and 1 of
each group of 4); B = the two dense P halves. So a legal row-sum implementation for us is: A = constant
2:4-compressed ones matrix, SRC2 = constant index pattern, B = our E4M3 P (dense), D accumulated by
two smfmac instructions; 16 cycles each per the wiki table. **Caution (Experimental):** each sparse
smfmac contracts only the 2:4-selected half of K; the vendor's P layout must make the two instructions
cover the full 128-key row sum (their output matches CPU, so it does). Kernel-opt should either verify
that layout equivalence against aiter's P packing or use two complementary index patterns
(`0x44444444` selects {0,1}, `0xeeeeeeee` selects {2,3}) and rely on D accumulation; cost is 2 smfmac
per 128 keys either way (32 cycles).


## 4.5 SMFMAC sparse layout - hardware probe (Verified / in progress)

The rewrite wants to replace the two row-sum scaled MFMAs with `v_smfmac_f32_16x16x128_fp8_fp8`.
`tests/gpu/smfmac_host.cpp` + `tools/vendor_smfmac_probe.sh` run a 14-experiment probe
(`probe_smfmac_ones` in `crates/amdgpu-dsl/src/kernels.rs`, own backend) that dumps the 4-dword D of
every lane for each experiment. The probe is built on the new additive DSL/MIR support
`b.smfmac_fp8_16x16x128(a_frag4, b_frag8, idx_v1, c_frag4)` (MOp `VSmfmac16x16x128Fp8`), so the
rewrite can use it directly.

Verified on MI355X / gfx950 (node 53, GPU 5), 5 repeated runs:

| question | answer | evidence |
|---|---|---|
| operand register shape | A = **4 VGPR** (2:4-compressed A), B = **8 VGPR** (dense), SRC2 = **1 VGPR**, D = **4 VGPR** (f32) | emitted asm + hardware results |
| SRC2 encoding | 32 bits = 16 x 2-bit fields, LSB-first; each pair of fields selects two of the four K positions of that group and must be strictly increasing. `0x44444444` = pairs (0,1), `0xeeeeeeee` = pairs (2,3) | `0x44444444` and `0xeeeeeeee` both accepted; `0x55555555` (pair (1,1)) is also accepted but is not a valid strictly-increasing pattern |
| K coverage of one smfmac | **64 of the 128 K positions**: A/B all ones with either pattern gives D = 64 for **all four D dwords** | slots 0/1/5/6 of the probe |
| K coverage of two smfmacs | **128 (all K)**: accumulating `0x44444444` then `0xeeeeeeee` into the same D (both B halves) gives D = 128 on all four D dwords; order-independent; a third instruction gives 192 | slots 2/3/4, stable across 5 runs |
| is A per-lane data? | **yes**: zeroing A on lanes 32..63 halves every D element (64 -> 32), i.e. the two lane halves supply complementary halves of the compressed A nonzeros | slot 9 |
| is SRC2 per-lane? | accepted and runs with a lane-masked index (`tid<32 ? 0x44444444 : 0xeeeeeeee`); with all-ones A/B the value is still 64, so this does not yet prove whether the hardware reads SRC2 per lane or per wave. Follow-up experiment uses per-lane dense B to settle it (Experimental) | slot 8 |
| marker A dwords (1,2,4,8) | all-ones B gives D = 240 on all four dwords = 64 contributions x mean 3.75, consistent with "64 selected nonzeros per row, each carrying one of the four dword marker values" | slots 10/11 |

**Open issue (probe/backend hazard, not resolved yet):** a *single* smfmac experiment sometimes
returns nondeterministic garbage in the **4th D dword** (run-to-run varying values such as 64.0002 or
69.5 where 64 is expected; some runs are fully clean). Two-instruction accumulation chains are always
clean on all four dwords. Working hypothesis: a write-after-write hazard between the zero-initialisation
of the next experiment's in-place accumulator D and the previous smfmac's writeback, which our hazard
pass does not currently pad for `VSmfmac16x16x128Fp8`. Mitigation until fixed: use accumulating
chains (which the row-sum recipe does anyway) or distinct D registers per result. Recorded here so
the rewrite does not trust a single-smfmac 4th dword.

### 4.5.1 Row-sum recipe for the rewrite (Verified on hardware)

Replacing the two accumulated scaled MFMAs (msum) with two sparse MFMAs works **exactly**, on all four
D dwords, lane for lane. Copy-pasteable recipe (own backend; P1/P2 are the two Frag8 halves of the P
fragments the PV path already builds, i.e. P^T in the vendor's orientation):

```rust
// A = constant 2:4-compressed ones (E4M3 1.0 = 0x38383838 in all four bytes of each A dword).
let a_ones = b.frag4(ones, ones, ones, ones);
// 2-bit sparse index patterns: 0x44444444 selects K positions {0,1} of every group of 4,
// 0xeeeeeeee is the complementary {2,3} pattern. Together they contract all 128 K positions.
let idx01 = b.c_i32(0x44444444_u32 as i32);
let idx23 = b.c_i32(0xeeeeeeee_u32 as i32);
let zero = b.frag_zero(DType::Frag4);
// D accumulates: first P half with {0,1}, second P half with {2,3}. 2 x 16 cycles = 32 per 128 keys
// (vs 2 x 64 = 128 for the two scaled MFMAs we use today).
let l = b.smfmac_fp8_16x16x128(a_ones, p1, idx01, zero);
let l = b.smfmac_fp8_16x16x128(a_ones, p2, idx23, l);
// Consume: D dword j of `l` == dword j of the msum Frag16 accumulator (j = 0..3), so use exactly the
// dword pair the kernel already reduces for l (the vendor reads v224+v225 = D dwords 0 and 1).
```

Evidence (probe slots 40/44, 3 repeated runs): with a lane-dependent P (even lanes 1.0, odd lanes 2.0
in one half; constant 4.0 in the other) the 2-smfmac chain and the msum reference produced **identical**
values: lane0 = 320, lane1 = 384 on every one of the four D dwords. A flat-marker P gave 480 = 480
(slots 14/15). Dropping one P half gives half the value (128/256 instead of 320/384), and swapping which
half pairs with which index pattern also sums correctly on dwords 0-2 (dword 3 only differs because of
the C-accumulation quirk below; with the zero-initialised C used in the recipe all four dwords are
correct).

**Constraint:** the first smfmac of the chain must start from an all-zero C (zero fragment is fine -
a dropped zero contribution is harmless). Do **not** start the chain from a non-zero,
VALU-initialised accumulator: see 4.5.2.

### 4.5.2 The 4th-D-dword anomaly: root cause and required padding rule (Verified + ISA)

Controlled probe experiments:

| experiment | d0 | d1 | d2 | d3 |
|---|---|---|---|---|
| preset C = 1000.0f, no smfmac, dump | 1000 | 1000 | 1000 | 1000 |
| preset C, one smfmac, read back through VALU | 1064 | 1064 | 1064 | **~64 (C dropped)** |
| preset C + 1/2/4 used filler VALU ops, one smfmac | 1064 | 1064 | 1064 | **still A x B only** |
| chain of 2 smfmacs from a zero C (row-sum recipe) | 128 | 128 | 128 | 128 (correct) |
| single smfmac read directly by a global_store | 64 | 64 | 64 | run-to-run garbage |

So the preset writes and the read path are fine (first row), and the smfmac writes dwords 0..3; the
anomaly is specific to the **implicit accumulator C**: when C is written by a *non-XDL* (VALU) producer
shortly before the SMFMAC, the 4th C dword is not seen by the instruction (the result is A x B only for
that dword), and a *direct* consumer reading D too early sees stale/garbage in that dword because matrix
results are written back in stages. A second SMFMAC reading the same D (same-opcode exact-C path)
accumulates correctly on all four dwords, and a zero C makes the missing contribution harmless - which
is why the row-sum recipe above is safe.

ISA basis (CDNA4 §7.6; kernel-wiki `mi350-isa-waitcnt-hazards` p74/p77, PASS = 4 clocks):
`VALU write VGPR -> MFMA/SMFMA read` is a **Required Wait of 2 waits (8 clocks)**, and p77 lists the
**WAR** direction (`XDL/SMFMA reads SrcC -> VALU overwrite of the source`) as a separate hazard that a
RAW-only check misses. Matrix results are explicitly documented as written back in multiple stages and
possibly partially visible (p49), matching the dword-3-only behaviour.

**Required padding rule for our backend and for backend-llvm-tail:**
1. Model the SMFMAC in-place accumulator `C` as a *real source operand* in MIR. Today the MIR op has
   sources A, B, SRC2 only, so no hazard pass can see that the instruction reads D. (Our own backend
   currently pads chained SMFMACs only by accident, through the WAW branch.)
2. After a non-DL VALU write to any A/B/C register, insert the ISA's **2 waits = 8 clocks** before the
   SMFMAC (e.g. `s_nop 7` = 8 cycles).
3. Add the p77 **WAR** direction as well: a VALU write to A/B/C shortly after an SMFMAC reads them must
   wait for the operand read to complete (same class of spacing).
4. Keep the existing 20-wait rule before reading D, but apply it before the **last** dword read too
   (partial writeback) and prefer the accumulator chain pattern over a single smfmac + immediate read.

Open point (Experimental): our used-filler experiment shows that 8 clocks of VALU distance were *not*
enough to make a non-zero VALU-initialised C dword 3 visible in the single-smfmac case; treat the exact
minimum for that path as unconfirmed and avoid it (start chains from zero C or from a previous SMFMAC).

## 4.7 The vendor's softmax conversion: exact constants, formula, and why it needs no correction

Reverse-engineered from `ref_mxfp8.co` (`tools/vendor_disasm.sh`); all values are the live scalars of
`label_block_A_loop` at 0x2b68 and the four rollback copies at 0x9d4c/0xa1f4/0xa69c/0xab44.

**Constants (Verified, exact bit patterns):**

| constant | value | where / role |
|---|---|---|
| `s26`/`s27` | `0x43ffff00` = **511.9921875** = 65535/128 | exponent-field scale: `E_bf16 = floor(65535*x/128) = floor(s26*x)` |
| `s93` | `0x3b000080` = 1/512 (1+2^-16) | reference update, `r += s93*s26*delta` (product = 0.9999997) |
| `s90` | **1.0** (the 1/512 computed at 0x1a68 is dead; overwritten at 0x1ab4) | `v_mul_f32_e64 v184, -s90, v192` = minus the carried max reference |
| `s91` | `0x3e7df17e` = **0.24799132** | exp bias added to every score before `v_cvt_pknorm_u16_f32` |
| `s92` | `0x3e85d086` = **0.26135647** | lazy-rollback threshold, `v_cmp_nge_f32 vcc, s92, v193` |
| `s89` | `0x3e800000` = **0.25** (f32) | scale operand of every `v_cvt_scalef32_pk_fp8_bf16` |
| Q scale | `v236 = max(qscale,9) - 9` (0x1d50/0x1d54) | Q operand is lowered 2^-9, so the QK output S = decoded K·Q^T / 512 |
| `s94` | 0/1 | seed-phase flag (no numerics) |

**Formula (fast path):**
```
x   = (S - r) + s91                      // S = K·Q^T with the Q scale lowered 2^-9, r = carried f32 row-max ref
u16 = v_cvt_pknorm_u16_f32(x)            // = round(65535 * clamp(x,0,1))
bf16 = reinterpret_u16_as_bf16(u16)      // = 2^(E-127)*(1+m/128), E = floor(65535*x/128), m = low 7 bits
P   = s89 * bf16  = 0.25 * bf16
    ~= 0.24513 * 2^(s - s_max)           // s = decoded score in exp2-exponent units
```
i.e. the **Schraudolph piecewise-linear exp2** (`2^f ~ 1+f`, error 0 .. +6.2 %, mean +4.0 %) — exactly
the approximation our uncorrected path uses (our bias 131/512 and cvt scale 1.0 give P = 16*2^(s-s_max);
the two differ only by the constant factor 2^-6.0284, whose 2^-0.0284 part changes no error statistic).

**Negative result (Verified): the vendor does not correct the approximation.** The whole binary contains
**zero** `v_pk_sub_u16`, zero `v_perm_b32` and no u16 masking/correction; the fast path and all four
rollback copies use the identical 32x pknorm + 32x scalef32 (scale s89) pair. There is no "better
scaling" behind s89/s91/s92 — they are the same approximation, not a correction.

**What the vendor does differently is the reference path, not the conversion (Verified static):**
1. The row-max reference is carried **in f32** (`v192`) and updated exactly: rollback subtracts
   `delta = max - s91` from all 64 scores in f32, updates `r += s93*s26*delta` (~= delta) and rescales
   O by `v_exp_f32(-s26*delta)` = 2^(-512*delta).
2. The bias is expanded into **multiple bf16 words** (`v244`, `v245`: hi word + residual, 0x1a1c0-0x1e8)
   so the carried bias keeps ~16+ bits, not one f16 word.
3. The rollback recomputes the row max **cross-lane** (`v_permlane32_swap_b32` + `v_max_f32`) before the
   `s92` compare; the bias/threshold margin is 0.0134 (=6.84/512), ours is 4/512.
4. P is rescaled per lane with an **integer Delta** (`123 + Delta` MFMA scale), no fractional loss.

**Our sv/vs recipe instead round-trips the bias through a single f16 word:**
`bias2 = CvtF32ToF16((B_old + (c_b131 - mt))/8)` and the rollback delta used for the score add, the O/l
rescale (`exp2(512*delta)`) and the new bias is `b_q - b_old`, i.e. **quantised to the f16 bias grid**
(~1.2e-4 in x units = ~0.06 in exponent units = up to ~4-6 % P error per rollback, at ~70 % rollback
frequency). This is the only place where our implemented recipe loses precision relative to the vendor.

**Recommendation for kernel-opt (to drop svcorr and recover the +9.7 % path):** keep the pknorm+scalef32
Schraudolph conversion uncorrected, and make the reference path exact like the vendor: carry the max
reference in an f32 register, compute delta = max - s91 in f32, add it to the scores in f32, rescale O/l
with 2^(-512*delta) exactly, and keep the bias word as a two-word (hi+residual) expansion instead of a
single f16 round-trip. Then our uncorrected error should fall to the vendor's ~2.4e-3 mean and pass the
parity gate. Cheapest first experiment: keep the current recipe but store the bias word as two f16
words (hi + residual of B/8), so the delta no longer loses its low bits; expected to remove the
52 bad elements. If it still fails, port the f32-reference path above.

## 5. LDS layout, staging and VGPR budget (Verified static)

* **Global -> LDS is DMA, not a register round trip.** 63 static `buffer_load_dwordx4 ... lds`
  (16 B/lane per instruction) plus `buffer_load_dword ... lds` (K block scales) write straight to LDS;
  the LDS destination is `M0 + immediate`, with M0 set by `s_add_u32 m0, s71, imm`. Only ~5 VGPR are
  spent on staging temporaries (`v211`, `v212`, `v184`) plus 2-3 normal `buffer_load_dwordx4` for
  Q/scales. Our kernel instead stages through registers (16x `global_load_dwordx4` +
  `ds_write_b128`), which costs VGPR and issue slots.
* **Run size 1040 B (0x410)** - the M0 immediates are 0, 0x410, 0x1000, 0x1410, 0x4100, 0x4510,
  0x5100, 0x5510, i.e. 1040-B runs in groups of 4 (K and V separately), plus aux buffers at
  `s67 + 0x14500` / `0x14700`. Ring parity/wrap is selected with `s_cselect_b32` (95 static), so the
  buffers are a double-buffered ring of 1040-B runs; the LDS footprint is **84224 B** (descriptor
  `.group_segment_fixed_size`; ours is 69632 B).
* **LDS reads** are `ds_read_b64_tr_b8` (688 static; V-fragment transpose loads) and `ds_read_b128`
  (304; K fragments) plus `ds_read_b32` (112; scales); observed immediate offsets form 32/64/96-B
  groups with 256-B steps and second-buffer groups at +8192 (up to ~29.7 KB dynamic read window).
  No `ds_write*` at all in the kernel - consistent with all stores coming from LDS-DMA.
* **VGPR split** (from live ranges): O accumulators 4 x 16 = 64 VGPR (v116..v179, 26 %); S^T tiles
  64 VGPR (v52..v115, 26 %); V fragments v20..v51; K fragments v4..v19; row-sum/l state v224..v247;
  addresses/misc; staging ~5. Descriptor: `.vgpr_count = 246`; SGPR `.sgpr_count = 99` (the whole
  per-wave budget: 8 waves x 99 = 792 of 800). Our kernel: VGPR 256 (budget exhausted), SGPR 48.

## 6. Side-by-side, ours vs vendor (same counters, same run)

Method (Verified numbers, Experimental interpretation): both kernels are launched from the same
`ref_host` process, same shape B1 H8 S=SK=2048 (64 blocks, 512 threads = 8 waves/block, 16 iterations
of 128 keys), GPU 5, no other load. Wall time is `ref_host`'s hipEvent timing over 20 unprofiled
launches. Counters are one `rocprofv3 --pmc` set (ROCm 7.2.4 / v1.1.0), 4 passes, averaged per dispatch.
Normalisation: *per wave-iteration* = raw / (512 waves x 16 iterations); *per SIMD-iteration* = the two
resident waves' counters (x2), i.e. the work the SIMD must absorb in one 128-key iteration. Cycle columns
are at the assumed 2.4 GHz MI355X boost clock; counters themselves are clock-independent and the ratio
conclusions do not depend on the clock assumption.

**Wall (unprofiled):**

| | vendor | ours | delta |
|---|---|---|---|
| per launch (16 iterations) | **0.0333 ms** | **0.0587 ms** | +76.3 % |
| per 128-key iteration | **2.081 us** (4994 cyc @2.4 GHz) | **3.669 us** (8806 cyc) | **+1.588 us** (+3812 cyc) |
| throughput | **516.28 TFLOP/s** | **292.61 TFLOP/s** | -43.3 % |

**Per wave-iteration (raw counters, one wave = one role):**

| metric | vendor | ours | delta | comment |
|---|---|---|---|---|
| `SQ_INSTS` (all) | 381.6 | 511.6 | **+130.0** | +34 % instructions per iteration |
| `SQ_INSTS_VALU` | 250.9 | 404.5 | **+153.6** | the dominant extra work |
| ... `_MUL_F32` | 32.3 | 87.3 | +54.9 | our per-iteration O rescale multiplies |
| ... `_ADD_F32` | 54.7 | 37.3 | -17.4 | vendor does more (softmax path) |
| ... `_CVT` | 69.4 | 34.0 | **-35.4** | vendor's 32 pknorm + 32 scalef32 (exp path) |
| ... `_INT32` | 10.8 | 42.9 | **+32.1** | our ring/address arithmetic is in VALU, vendor's in SALU |
| ... `_FMA_F32` | 4.1 | 0.0 | -4.1 | |
| ... other (MOV/TRANS/CMP/...) | 79.6 | 203.1 | **+123.5** | ~150 `v_mov` + 66 `v_exp` per iteration in ours, ~30 moves + ~1 exp in vendor |
| `SQ_INSTS_LDS` | 52.5 | 56.8 | +4.3 | similar LDS read count |
| `SQ_INSTS_VMEM` | 5.7 | 7.9 | +2.2 | our staging round trip costs instructions |
| `SQ_INSTS_SALU` | 40.0 | 0.2 | **-39.8** | vendor keeps ring/loop state scalar |
| `SQ_IFETCH` | 81.4 | 97.3 | +15.9 | +19 % code footprint per iteration |
| `SQ_VALU_MFMA_BUSY_CYCLES` | **1086.0** | **1152.0** | +66.0 | vendor = 8x64(QK) + 8x64(PV) + 32 + 32; ours = 18x64 (row sums as 2 extra scaled MFMA) |
| `SQ_ACTIVE_INST_VALU` | 293.8 | 521.8 | **+228.0** | VALU pipe busy per wave-iteration |
| `SQ_BUSY_CYCLES` | 294.4 | 518.3 | +223.9 | consistent with the VALU delta |
| `SQ_WAIT_ANY` | 415.1 | 739.0 | **+323.9** | dependency/latency stalls, the second lever |
| `SQ_INST_LEVEL_VMEM` | 53.5 | 147.3 | **+93.8** | VMEM latency exposure 2.75x ours |
| `SQ_INST_LEVEL_LDS` | 32.1 | 39.0 | +6.9 | |
| `SQ_LDS_BANK_CONFLICT` | 0 | 1.06 | +1.06 | small but non-zero in ours |
| `SQ_WAVE_CYCLES` | 1156 | 1997 | +841 | wave-level active cycles (not wall time) |

**Per SIMD-iteration (both resident waves, cycles at 2.4 GHz) - overlapping categories, they do not sum
to the wall:**

| category | vendor | ours | delta |
|---|---|---|---|
| wall | 4994 | 8806 | **+3812** |
| XDL / MFMA pipe busy | 2172 (43.5 % of wall) | 2304 (26.2 %) | +132 |
| VALU issue/active | 588 | 1044 | **+456** |
| VMEM latency level | 107 | 295 | **+188** |
| LDS latency level | 64 | 78 | +14 |
| instruction fetch | 163 | 195 | +32 |
| wait-any (dependency stalls) | 830 | 1478 | **+648** |
| LDS bank conflicts | 0 | 2 | +2 |

Delta attribution: the *added work* counters account for **+822 cycles/iteration** (XDL 132 + VALU 456 +
VMEM 188 + LDS 14 + ifetch 32); the remaining **+2990 cycles** of wall growth are elapsed-time inflation
- the vendor hides ~all non-XDL work inside its MFMA/dependency shadow with the two-wave
setprio ping-pong, ours serialises it. The actionable levers are therefore (a) delete the extra work
(VALU above all), then (b) restore the latency hiding (VMEM/DMA + prefetch + priority schedule).
Note on the older "ours 7.3k vs vendor 3.6k cycles/iteration" figures from earlier rounds: those used a
different frequency convention; the same-session measurement here is 2.081 us vs 3.669 us per iteration
(ratio 1.76) at 2.4 GHz nominal, and the counters (clock-free) give the ratios above.

## 7. What to change to close each delta (ranked by expected cycles saved)

1. **Cut VALU work - the largest lever (+456 cycles/SIMD-iteration to remove).**
   a. *Other VALU (moves/trans): -247 cycles/SIMD* (`+123.5`/wave). Replace per-element `v_exp_f32`
      (66/iteration) with the vendor's `v_cvt_pknorm_u16_f32` + `v_cvt_scalef32_pk_fp8_bf16`
      Schraudolph path (costs +32 CVT, saves 66 TRANS -> net -34 VALU/wave), and eliminate the ~150
      redundant `v_mov` copies we still emit per iteration (vendor ~30). Both are emitter/codegen changes.
   b. *MUL_F32: -110 cycles/SIMD* (`+54.9`/wave). Stop rescaling O every iteration: carry the scale in
      the FP8-conversion E8M0 exponent operand (`v_cvt_scalef32_*`) as the vendor does, and only
      rescale on the rare lazy-rollback path. Today we emit ~87 `v_mul_f32` per iteration, vendor 32.
   c. *INT32/address: -64 cycles/SIMD* (`+32.1`/wave). Move ring/descriptor/tile address arithmetic from
      VALU to SALU with `s_cselect_b32`/`s_add_u32` (we have 52 SGPR/wave of unused budget; vendor spends
      99/100 SGPR). This is the vendor's single biggest scalar/vector trade.
2. **Kill VMEM latency exposure (-188 level cycles, -2.2 VMEM instructions/iteration).** Replace the
   `global_load_dwordx4 -> VGPR -> ds_write_b128` staging round trip with LDS-DMA
   (`buffer_load_dwordx4 ... lds`, 1040-B runs, `M0 = base + s_cselect` ring), and keep the next tile's
   DMA in flight during the PV/QK MFMAs with one `s_waitcnt vmcnt(0)` per iteration (vendor pattern).
   This also removes the staging VGPR pressure.
3. **Row sums with `v_smfmac_f32_16x16x128_fp8_fp8` (-192 XDL cycles/SIMD, -66/wave).** We currently burn
   2 extra scaled MFMAs (2 x 64 cycles); vendor uses 2 smfmac of 16 cycles = 32 total. Legal with A = a
   constant 2:4-compressed ones matrix + constant SRC2 index, B = dense E4M3 P (section 4). Verify the
   2:4 coverage against the P layout (or use two complementary index patterns) before relying on it.
4. **Instruction count / ifetch (-32 cycles, -16 IFETCH/wave).** Falls out of 1/2: target <= 400
   instructions and <= 82 IFETCH per wave-iteration (vendor 381.6 / 81.4).
5. **Barrier + priority schedule (structural; it is what turns the remaining stalls into shadow).**
   Keep exactly 2 block barriers per 128 keys (after P conversion, after PV/QK), `s_setprio 1` over the
   MFMA/PV phase and `s_setprio 0` at the next barrier, and prefetch the **next** tile's QK inside the
   current compute group (vendor: current PV + next QK in one instruction window). The A/B role paths
   mirror each other so the two waves on a SIMD are always in complementary setprio states. This is the
   direct countermeasure for our `SQ_WAIT_ANY` 739 vs vendor 415.
6. **LDS ring**: adopt the 1040-B-run double-buffered ring and the transposed V / `b128` K reads
   (`ds_read_b64_tr_b8` for V, `ds_read_b128` for K) - the ring is also the enabler for item 2. Bank
   conflicts are small today (0 vs 1.06/wave-iteration) but must reach 0.

Projection: items 1-3 remove the whole +822 cycles of extra work; item 5 attacks the stall remainder.
Acceptance target after the rewrite: `<= 0.040 ms/launch` (>= 420 TFLOP/s, >= 81 % of the vendor) with
`<= 0.036 ms` (>= 470 TFLOP/s) as the stretch goal, from 0.0587 ms / 292.6 TFLOP/s today.

## 7.5 Acceptance checklist for kernel-opt's rewrite

Measured with the same protocol (`ref_host` hipEvent timings, `tools/vendor_counters.sh` for counters,
same shape and GPU). Pass = every line at or better than the "target" column; the vendor column is the
reference. A faster kernel that changes numerics fails the correctness gate regardless.

| check | vendor | our current | target |
|---|---|---|---|
| wall per launch (16 iters of 128 keys) | 0.0333 ms | 0.0587 ms | **<= 0.040 ms** (stretch 0.036) |
| throughput | 516.3 TFLOP/s | 292.6 TFLOP/s | **>= 420** (stretch 470) |
| `SQ_INSTS` /wave-iter | 381.6 | 511.6 | <= 400 |
| `SQ_INSTS_VALU` | 250.9 | 404.5 | <= 265 |
| `_MUL_F32` | 32.3 | 87.3 | <= 40 |
| `_INT32` | 10.8 | 42.9 | <= 15 |
| other VALU (MOV/TRANS/CMP) | 79.6 | 203.1 | <= 95 |
| `_CVT` (pknorm/scalef32 path present) | 69.4 | 34.0 | >= 60 |
| `SQ_INSTS_SALU` (ring in scalar) | 40.0 | 0.2 | >= 35 |
| `SQ_INSTS_LDS` | 52.5 | 56.8 | <= 55 |
| `SQ_INSTS_VMEM` | 5.7 | 7.9 | <= 6 |
| `SQ_IFETCH` | 81.4 | 97.3 | <= 85 |
| `SQ_VALU_MFMA_BUSY_CYCLES` /wave-iter | 1086 | 1152 | <= 1100 |
| `SQ_ACTIVE_INST_VALU` | 293.8 | 521.8 | <= 310 |
| `SQ_WAIT_ANY` | 415.1 | 739.0 | <= 450 |
| `SQ_INST_LEVEL_VMEM` | 53.5 | 147.3 | <= 60 |
| `SQ_INST_LEVEL_LDS` | 32.1 | 39.0 | <= 35 |
| LDS bank conflicts | 0 | 1.06 | <= 0.1 |
| barriers per 128-key step | 2 | 2 | 2 (softmax->P and PV/QK->next) |
| setprio windows | 1 set/clear pair per step over the MFMA phase | not used | required |
| next-tile QK inside current compute group | yes (8 scaled MFMA) | separate | required |
| staging | LDS-DMA, 1040-B runs, ring | register round trip | required |
| row-sum structure | 2x `v_smfmac_f32_16x16x128_fp8_fp8` with complementary index patterns (0x44444444 then 0xeeeeeeee), chained from a zero C (recipe in 4.5.1) | 2 scaled MFMAs (no smfmac) | required |
| row-sum cycle cost per 128 keys | 32 (2 smfmac) | 128 (2 scaled MFMA) | <= 32 |
| smfmac operand hazard | no VALU-written A/B/C within 8 clocks (2 waits) of an smfmac; no non-zero VALU-initialised accumulator feeding an smfmac | n/a (no smfmac yet) | required (see 4.5.2) |
| iteration unit | 128 keys per barrier pair | checked per push | required |
| correctness | 15/15 REF shapes + determinism | same | unchanged (docs/verification.md) |

Failure of the wall/throughput lines with all counter lines green means the scheduler/prefetch structure
(item 5) is still wrong; failure of the VALU lines points at the emitter/codegen (items 1, 4).

## 7.6 Baseline run of the checklist: kernel-opt `_vs` @ be6c1f1 (GPU 5)

Commit under test: kernel-opt `be6c1f1` (merged Main `e0e9a77`; validation branch `vs-validate`,
head `a84fe4a`). Command: `bash tools/vendor_validate.sh fmha_fwd_hd128_mxfp8_vs`; same-run wall,
B1 H8 S=SK=2048, 20 unprofiled launches. The generated asm has two role loops of 687/673 instructions;
one loop iteration = 128 keys.

| §7.5 line | vendor | `_vs` @ be6c1f1 | target | delta closed? |
|---|---|---|---|---|
| wall per launch | 0.0333 ms (516.0 TFLOP/s) | 0.0658 ms (**261.0 TFLOP/s**) | <= 0.040 ms / >= 420 TFLOP/s | **no** (also below current `_t`: 292.6) |
| `SQ_INSTS` /wave-iter | 381.6 | 678.4 | <= 400 | no |
| `SQ_INSTS_VALU` | 250.9 | 543.7 | <= 265 | no |
| `_MUL_F32` | 32.3 | 58.1 | <= 40 | no |
| `_INT32` | 10.8 | 75.6 | <= 15 | no |
| other VALU (MOV/TRANS/CMP) | 79.6 | 311.8 | <= 95 | no |
| `_CVT` | 69.4 | 70.4 | >= 60 | **yes** (pknorm+scalef32 path ported) |
| `SQ_INSTS_SALU` | 40.0 | 4.4 | >= 35 | no (ring/address math still VALU) |
| `SQ_INSTS_LDS` | 52.5 | 52.5 | <= 55 | **yes** (exact match) |
| `SQ_INSTS_VMEM` | 5.7 | 7.9 | <= 6 | no (marginal, same as `_t`) |
| `SQ_IFETCH` | 81.4 | 136.4 | <= 85 | no (worse than `_t` 97.3) |
| `SQ_VALU_MFMA_BUSY_CYCLES` | 1086 | 1280 | <= 1100 | no (20 MFMA/wave-iter) |
| `SQ_ACTIVE_INST_VALU` | 293.8 | 597.2 | <= 310 | no |
| `SQ_WAIT_ANY` | 415.6 | 1113.2 | <= 450 | no (worse than `_t` 739.0) |
| `SQ_INST_LEVEL_VMEM` | 52.7 | 119.5 | <= 60 | no (improved from `_t` 147.3) |
| `SQ_INST_LEVEL_LDS` | 32.1 | 39.5 | <= 35 | no |
| LDS bank conflicts | 0 | **64.0** | <= 0.1 | **no - new regression** (the `_t` kernel has 1.06) |
| barriers per 128 keys | 2 | **4** | 2 | no (asm: 4 `s_barrier` per role loop iteration) |
| setprio windows | 1 set/clear pair per 128 keys | **4 `s_setprio` per iteration** | 1 pair | no |
| next-tile QK in compute group | 8 scaled QK MFMA | 18 `v_mfma_scale` per iteration (QK+PV, next tile prefetched in-loop) | present | **yes (structure present)** |
| staging | LDS-DMA, 1040-B runs | 4 `global_load_lds_dwordx4` per iteration | required | **yes** |
| iteration unit | 128 keys | 128 keys but built from 2x64-key tiles | 128 keys | no (barrier/setprio rate doubled) |
| row-sum structure | 2x smfmac, complementary patterns, chained from zero C | 0 smfmac; 4x `v_mfma_f32_32x32x8_f16` msum (128 cyc/128 keys) | required | **no** |
| smfmac operand hazard | n/a | n/a (no smfmac) | required when smfmac lands | n/a |

Correctness gate at that commit: **15/15 reference shapes, 5/5 determinism** (own backend) - passes.
So be6c1f1 is numerically correct and has the DMA/CVT/next-tile-QK structure in place, but the
64-key-tile barrier/setprio rate (4 per 128 keys instead of 2), the VALU address math (INT32 75.6,
other 311.8), the remaining msum MFMAs and a **new LDS bank-conflict regression (64/wave-iteration)**
keep it at 261 TFLOP/s - below the existing `_t` (292.6) and far from the 420 target.

Note: `fmha_fwd_hd128_mxfp8_vs128` (the 128-key-unit variant) is *not buildable* at be6c1f1:
`amdc emit fmha_fwd_hd128_mxfp8_vs128 --backend own` fails with
`out of VGPRs allocating r827 (V16, live 832..2626); 73 live vregs holding 231 dwords`
(after 8 DSL lint warnings "loop variable read after set_var in the same iteration"). It must be
made to allocate before the next validation round.

## 7.7 Full checklist: kernel-opt `fmha_fwd_hd128_mxfp8_vs128` @ Main 3cc73c0 (GPU 5)

Measured 2026-10-29 on GPU 5 with `bash tools/vendor_validate.sh fmha_fwd_hd128_mxfp8_vs128` plus
`tools/vendor_counters.sh` for the counter rows (same-run wall, B1 H8 S=SK=2048, 20 unprofiled
launches; vendor measured in the same process). Commit: Main `3cc73c0` (kernel-opt `34efdeb`
"vs128: default lowering opts scalarize+imm and the full scheduler" + backend-tail pressure fix).

**Headline: 298.1 TFLOP/s (0.0576 ms) vs vendor 515.8 TFLOP/s (0.0333 ms) - the first kernel of ours
above the tuned default in the same run** (`_t` llvm-tail 290.4, `_t` own 281.4; `_vs` @be6c1f1 261.0).
Remaining gap to the vendor: **1.73x**. Correctness gate passes: 15/15 reference shapes (own backend),
5/5 determinism. Numeric recipe: svcorr rollback; bit-parity with `_sv`/`_vs` is *not* given (vs128 md5
`6891688c…` vs sv/vs `9aced7bc…`; mean_err 1.7738e-3 vs sv 1.7517e-3, `_t` 1.2540e-3, vendor ~2.4e-3;
vendor-parity gate PARITY_OK is the round-49 claim). LDS footprint 102400 B (vendor 84480); vgpr 244,
sgpr 45.

| §7.5 line | vendor | `_vs128` @3cc73c0 | target | delta closed? |
|---|---|---|---|---|
| wall per launch | 0.0333 ms (515.8 TFLOP/s) | 0.0576 ms (**298.1 TFLOP/s**) | <= 0.040 ms / >= 420 TFLOP/s | **no** (best of ours; 1.73x from vendor) |
| `SQ_INSTS` /wave-iter | 381.6 | 642.5 | <= 400 | no |
| `SQ_INSTS_VALU` | 250.9 | 491.3 | <= 265 | no |
| `_MUL_F32` | 32.3 | 37.2 | <= 40 | **yes (new)** |
| `_INT32` | 10.8 | 60.8 | <= 15 | no |
| other VALU (MOV/TRANS/CMP) | 79.6 | 293.3 | <= 95 | no |
| `_CVT` | 69.4 | 68.7 | >= 60 | **yes** |
| `SQ_INSTS_SALU` | 40.0 | 22.8 | >= 35 | no (improved from `_vs` 4.4) |
| `SQ_INSTS_LDS` | 52.5 | 72.8 | <= 55 | no (regressed from `_vs` 52.5) |
| `SQ_INSTS_VMEM` | 5.7 | 7.6 | <= 6 | no |
| `SQ_IFETCH` | 81.4 | 136.0 | <= 85 | no |
| `SQ_VALU_MFMA_BUSY_CYCLES` | 1086 | 1280 | <= 1100 | no (22 MFMA/wave-iter: 18 scaled + 4 f16) |
| `SQ_ACTIVE_INST_VALU` | 293.8 | 543.3 | <= 310 | no |
| `SQ_WAIT_ANY` | 412.9 | 803.6 | <= 450 | no (improved from `_vs` 1113) |
| `SQ_INST_LEVEL_VMEM` | 52.1 | 117.1 | <= 60 | no |
| `SQ_INST_LEVEL_LDS` | 32.1 | 49.4 | <= 35 | no |
| LDS bank conflicts | 0 | **82.0** | <= 0.1 | no (worse than `_vs` 64.0) |
| barriers per 128 keys | 2 | **2** | 2 | **yes** |
| rollbacks per 128 keys | 1 | **1** (`v_cmp_lt_f32` + `s_cbranch_vccnz` per loop iteration) | 1 | **yes** |
| setprio pair | 1 set/clear pair | **1 pair** (`s_setprio 1` … `s_setprio 0` per iteration) | 1 pair | **yes** |
| DMA staging | ~5 LDS-DMA writes /128 keys | **4 `global_load_lds_dwordx4` per iteration per role** | 4 | **yes** |
| next-tile QK in compute group | 8 scaled QK MFMA | 18 `v_mfma_scale` per iteration (QK+PV, next tile prefetched in-loop) | present | **yes** |
| row-sum path | 2x smfmac, complementary patterns, chained from zero C | ones-MFMA: 4x `v_mfma_f32_32x32x8_f16` (128 cyc) | 2 smfmac (32 cyc) | **no** |
| iteration unit | 128 keys | **128 keys** (loop body = one 128-key tile) | 128 keys | **yes** |
| smfmac operand hazard | n/a | n/a (no smfmac yet) | required when smfmac lands | n/a |
| correctness | 15/15 + det | **15/15, 5/5 determinism** | pass | **yes** |

Movement versus the stored baselines (`_vs` @be6c1f1 261.0 TFLOP/s; `_t` 292.6):

* NO -> YES: barriers per 128 keys (4 -> 2), rollback rate (2 -> 1), setprio pair (4 -> 2), iteration
  unit (2x64-key tiles -> one 128-key tile), `_MUL_F32` (58.1 -> 37.2 <= 40). DMA staging, CVT path and
  next-tile QK were already YES for `_vs` and stay YES.
* Still NO: smfmac row sums (unmoved), all VALU classes except MUL, INT32, other-VALU, IFETCH,
  MFMA busy (1280), ACTIVE_INST_VALU, WAIT_ANY, VMEM/LDS level cycles; LDS instructions regressed
  (52.5 -> 72.8) and bank conflicts regressed (64 -> 82 per wave-iteration); SALU improved but below
  target.
* Wall: 261.0 -> 298.1 (+14.2%, now above `_t` llvm-tail 290.4) - the structural fixes (barrier/
  setprio/iteration unit) bought the improvement, while the remaining gap is work + stalls.

**Honest decomposition of the remaining 1.73x gap** (per SIMD-iteration at 2.4 GHz: vendor 4995 cycles,
ours 8625, +3630):

| component | cycles/SIMD-iter | evidence |
|---|---|---|
| XDL (22 vs 19 MFMA/wave-iter: 18 scaled + 4 f16 msum) | +388 | MFMA busy 1280 vs 1086 per wave-iter |
| VALU work | +499 | ACTIVE_INST_VALU 543.3 vs 293.8 per wave-iter (other-VALU 293.3 vs 79.6, INT32 60.8 vs 10.8) |
| VMEM latency exposure | +130 | INST_LEVEL_VMEM 117.1 vs 52.1 |
| LDS latency exposure | +35 | INST_LEVEL_LDS 49.4 vs 32.1 |
| instruction fetch | +109 | IFETCH 136.0 vs 81.4 |
| **accounted work delta** | **+1161** | |
| stalls / lost overlap (incl. 82 bank-conflict cycles/wave = +164 and 1607 vs 826 WAIT_ANY) | **+2470** | WAIT_ANY, bank conflicts, barrier/priority windows, LDS count |

Ranked remaining actions (with expected savings):
1. **Row sums via the 2-smfmac recipe (4.5.1)** and remove the 2 extra scaled MFMAs (18 vs vendor 16):
   up to **-448 XDL cycles/SIMD** plus ~-66 VALU/wave; brings MFMA busy to the vendor's 1086.
2. **VALU work (-~500/SIMD)**: cut "other VALU" (register copies) 293.3 -> <=95 and INT32 60.8 -> <=15
   by moving ring/address arithmetic to SALU (we use 22.8 vs vendor 40) and by removing the copies the
   scalarize/imm lowering still leaves.
3. **LDS (-164 conflict cycles + issue slots)**: fix the 82/wave bank conflicts and the 72.8 vs 52.5
   LDS instruction excess (pitch/swizzle for the 128-key ring, avoid redundant reads).
4. **Latency hiding**: VMEM level 117 -> ~52, LDS level 49 -> ~32 and IFETCH 136 -> ~81 (loop 605 vs
   vendor 473 instructions) - deeper DMA prefetch and less prologue/epilogue code.
5. Remaining barrier/priority/issue stalls (~+1500 cycles not explained by the work counters).

## 7.9 Final state - independent acceptance validation (2026-10-30, GPU 5)

**Commit validated: Main `f4cccb60d53b51e30695b89ee4590039f270de18`** (merge of the kernel-opt gen3 branch
and backend-tail). Kernel `fmha_fwd_hd128_mxfp8_vs128`, own backend, default `FMHA_OPT`, vendor
`ref_mxfp8.co` measured in the same process/run, shape B1 H8 S=SK=2048, 20 unprofiled launches per timing
sample. Numbers below were re-measured independently (not copied from the kernel-opt report).

Protocol caveat (node is shared): wall time is noisy - the same kernel sample ranges 0.0470-0.1492 ms
under contention. The table quotes the **best-case (uncontended) sample and the median of 15 interleaved
runs**; the uncontended same-run pair is what the 1.41x ratio is based on. A first validation attempt
during a busy period gave vendor 0.0522 ms / 328.9 TFLOP/s and ours 0.0664 ms / 258.9 TFLOP/s - both
deflated, not used.

### 0) Fair-comparison protocol for a tiling difference (prepared for the 128-row `t128` kernel)

The vendor binary is fixed: its kernarg/ABI hard-codes a 256-query-row tiling (`ts_qo = 256`, Q tile stride
`256*H*D`) and 512 threads (8 waves = 2 waves/SIMD). Our kernels choose their own tiling; the candidate
`fmha_fwd_hd128_mxfp8_t128` uses 128-row blocks (256 threads = 4 waves = 1 wave/SIMD) and deliberately
enforces one block per CU (3-slot ring = 99840 B LDS; two blocks need 199680 B > 163840 B). A single
TFLOP/s number is therefore ambiguous, and from now on **every tiling comparison is reported as two
explicitly labelled lines**:

* **(a) Per-CU efficiency, same grid.** Both kernels at the vendor's 256-row tiling (grid
  `(ceil(S/256), H, B)`), or equivalently the clock-free counters normalised per wave-iteration (they
  divide `blocks x 8 x (SK/128)` away). This number answers "which kernel is more efficient per CU" and
  is the only line where the two kernels are on identical launch geometry. For `_vs128` vs vendor on
  B1H8S2048 this is **1.41x slower (365.7 vs 515.5 TFLOP/s)** and is unchanged by tiling work.
* **(b) Absolute throughput at each kernel's own best legal tiling on this shape.** Vendor at 256 rows
  (it cannot do 128); ours at its best (`_vs128`: 256 rows; `_t128`: 128 rows). This number answers
  "what does this shape deliver" and **must always be quoted with both grids and block counts**, e.g.
  B1H8S2048: vendor `(8,8,1)=64 blocks = 25 % of 256 CUs`, `_t128` `(16,8,1)=128 blocks = 50 %`.
  It is not a per-CU efficiency claim: if `_t128` beats the vendor's absolute number by filling more
  CUs, the conclusion is "our tiling uses this shape's grid better", not "our kernel is faster per CU".

Grid-fill reference (256 active CUs/package, MI350 `mi350-arch-topology`; 1 block/CU for both the
vendor kernel and `_t128`): benchmark B1H8S2048 = 64 blocks (25 %) at 256 rows / 128 blocks (50 %) at
128 rows; the filling shape B4H8S2048 = 256 blocks (100 %) at 256 rows / 512 blocks (two sequential
block waves, all 256 CUs busy) at 128 rows. Absolute package-level claims are only meaningful for
shapes/grids that reach 256 blocks; the 15-shape gate mostly does not (see the table below).

Harness support (kept backward-compatible; defaults reproduce the historical 256-row/512-thread run):
`tests/gpu/ref_host.cpp` reads `REF_ROWS_PER_BLOCK` (candidate rows/block, default 256),
`REF_VENDOR_ROWS_PER_BLOCK` (vendor, default/only 256), `REF_OURS_BLOCK_THREADS` (default 512),
`REF_CUS` (default 256) and `REF_GRID_ONLY=1` (print the grid line and exit). It patches the
tiling-dependent ABI fields per kernel (output pointer, Q tile stride), launches the vendor on its
256-row grid and our kernel on `ceil(S/REF_ROWS_PER_BLOCK)` blocks, and prints one `[grid] ...` line;
the `timing vendor/ours` lines now carry `rows/block` and `blocks`. The 15-shape and determinism scripts
inherit the environment variable. The two t128 result lines are filled in here once the kernel lands.

### 1) Same-run wall vs vendor (final)

| | vendor | `_vs128` final @f4cccb6 | previous `_vs128` @3cc73c0 (§7.7) |
|---|---|---|---|
| per launch (16 x 128 keys) | **0.0333 ms** (median; min 0.0332) | **0.0471 ms** (median; min 0.0470) | 0.0576 ms |
| throughput | **515.5 TFLOP/s** (median; max 516.9) | **365.7 TFLOP/s** (best; median 364.8) | 298.1 TFLOP/s |
| gap to vendor | 1.00x | **1.41x slower (70.7 % of vendor per-CU throughput)** | 1.73x |
| movement vs §7.7 | - | **+22.7 % throughput, gap 1.73x -> 1.41x** | - |

The kernel-opt report quoted 367.2 TFLOP/s; the independent best-case here is 365.7 (0.4 % lower,
inside the run-to-run clock variance of this shared node).

### 2) Full §7.5 counter table for the FINAL kernel

Values are per wave-iteration (raw / (blocks x 8 waves x 16 iterations), the same normalisation as
section 6; this divides the grid away, so it is **per-CU and unaffected by the 25 %-filled grid**).
`previous` = §7.7 @3cc73c0; `target` = §7.5. `**closed?` = final vs target.

| metric | vendor | final | previous | target | closed? |
|---|---|---|---|---|---|
| `SQ_INSTS` (all) | 381.6 | **474.2** | 642.5 | <= 400 | no (improved 26 %) |
| `SQ_INSTS_VALU` | 250.9 | **352.1** | 491.3 | <= 265 | no (improved 28 %) |
| ... `_MUL_F32` | 32.3 | **41.1** | 37.2 | <= 40 | **no, marginal regression** (+3.9 vs prev; 1.1 over target) |
| ... `_ADD_F32` | 54.7 | 38.0 | (37.3 @sec6) | - | (we do less than vendor) |
| ... `_CVT` | 69.4 | **72.0** | 68.7 | >= 60 | **yes** |
| ... `_INT32` | 10.8 | **61.3** | 60.8 | <= 15 | no (unmoved) |
| ... `_FMA_F32` | 4.1 | 0.0 | - | - | - |
| ... other VALU (MOV/TRANS/CMP) | 79.6 | **139.7** | 293.3 | <= 95 | no (improved 52 %) |
| `SQ_INSTS_SALU` | 40.0 | **22.8** | 22.8 | >= 35 | no (unmoved) |
| `SQ_INSTS_LDS` | 52.5 | **52.8** | 72.8 | <= 55 | **yes (new; from 72.8)** |
| `SQ_INSTS_VMEM` | 5.7 | **7.6** | 7.6 | <= 6 | no |
| `SQ_IFETCH` | 81.4 | **101.4** | 136.0 | <= 85 | no (improved 25 %) |
| `SQ_VALU_MFMA_BUSY_CYCLES` | 1086* | **1216*** | 1280 | <= 1100 | no (improved; *static opcode accounting, see note) |
| `SQ_ACTIVE_INST_VALU` | 293.8 | **402.1** | 543.3 | <= 310 | no (improved 26 %) |
| `SQ_WAIT_ANY` | 420.6 | **570.6** | 803.6 | <= 450 | no (improved 29 %) |
| `SQ_INST_LEVEL_VMEM` | 1610* | **2148*** | 117 (old scale) | <= 60 (old scale) | no; **ratio 1.33x vendor** (was 2.25x) - counter scale is run/replay dependent, only the ratio is used |
| `SQ_INST_LEVEL_LDS` | 32.06 | **31.54** | 49.4 | <= 35 | **yes (new)** |
| `SQ_LDS_BANK_CONFLICT` | 0 | **0.0** | 82.0 | <= 0.1 | **yes (new; from 82)** |
| `SQ_BUSY_CYCLES` | 295.4 | 422.8 | (518.3 @sec6) | - | - |
| `SQ_WAVE_CYCLES` (ratio only) | 1151 | 1608 | 1997 | - | improved |
| `GRBM_GUI_ACTIVE` | 113.2 | 127.3 | - | - | - |

Note on the two starred counters: `SQ_VALU_MFMA_BUSY_CYCLES` and `SQ_INST_LEVEL_VMEM` were not
reproducible across two profiling runs on this (shared) node (vendor MFMA-busy 7625 vs 3422 per
wave-iter in the two samples; its static XDL accounting is 1088). The final kernel's MFMA-busy sample 2
is exactly its static value 1216, so the static accounting is quoted: ours 1216 vs vendor 1088 XDL
cycles/wave-iteration (16 scaled MFMA = 1024, 4 smfmac = 64, 4 f16-bias MFMA = 128; vendor 16 scaled =
1024 + 2 smfmac = 32 + 2 bf16 = 32). All other counters were identical across the two samples and match
section 6's vendor numbers, which is why they are used.

### 3) Structural line checks (final emitted assembly, 9423 static instructions, 4 loops)

| §7.5 structural line | final | evidence (per role loop body, 128 keys) | verdict |
|---|---|---|---|
| 2 barriers per 128-key unit | **2** | `s_barrier` = 2 in each of the 4 loops | **yes** |
| 1 realign per 128-key unit | **1** | `v_permlane` = 1 per loop | **yes** |
| setprio windows (1 set/clear pair) | **0** | `s_setprio` = 0; source comment: measured -3 % (`299.8 -> 308.5 TFLOP/s` when removed); opt-in `FMHA_OPT=prio` | **not present - waived by measurement** (not a defect) |
| LDS-DMA staging | **4** | `global_load_lds_dwordx4` = 4 per loop | **yes** |
| next-tile QK inside the compute group | **yes** | 16 `v_mfma_scale` per loop (QK+PV/next-tile QK fused in one group, gen2 structure) | **yes** |
| 128-key iteration unit | **yes** | one loop iteration = 128 keys, branch + 2 barriers per iteration | **yes** |
| row-sum path = 4x smfmac complementary chain | **yes (4x)** | 4 `v_smfmac_f32_16x16x128_fp8_fp8` per loop, chained into one Frag4; static index constants: 0x44444444 x8 and 0xeeeeeeee x8 | **yes structurally; cost 64 cyc/unit vs vendor 32** (the 4-smfmac variant is required for our P layout, §4.5.3; §7.5 target "2 smfmac / <= 32 cyc" is not met) |
| LUT-correction code copies | **2 of 4** | loops 0/1 (short-sequence copy) have `v_perm_b32`=32, `v_pk_sub`=32, `v_and_b32`=32 per unit; loops 2/3 (long copy) have 0 of each (correction only for < 9 tiles of 128 keys) | **yes** |
| smfmac operand hazard (2 waits / no VALU-written C/A/B < 8 cyc) | **0 violations** | `VAL smfmac_d/a/b_from_valu_closer_than_8cyc = 0/0/0`; 8 chained smfmac detected | **yes** |
| rollback decisions | **2 `v_cmp_lt_f32` per 128-key body** | one per 64-key conversion half (two-word f16 bias recipe) | **deviation from §7.5 "1/128 keys" - flagged to kernel-opt** |
| descriptors | VGPR **252**, SGPR **48**, LDS **102400 B** | emitted `.vgpr_count/.sgpr_count/.group_segment_fixed_size` | (vendor LDS 84480) |
| DSL lint warnings | **16 statements** "loop variable read after set_var in the same iteration" | emit log (`amdc emit`); contract says such reads are undefined (lowering happens to return the just-written value) | **latent risk, see below** |

### 4) Gap decomposition for the remaining 1.41x (per SIMD-iteration @2.4 GHz convention)

Wall per SIMD-iteration: vendor 4995 cycles, final **7065** (+2070, +41 %). The counters (per
wave-iteration, x2 for the two resident waves) attribute:

| component | cycles/SIMD-iter | evidence | upper-bound note |
|---|---|---|---|
| XDL / matrix pipe | **+256** | static 1216 vs 1088 XDL cycles/wave-iter (4 smfmac ans 4 f16-bias vs vendor 2+2) | <= +192 recoverable (2 smfmac would save 64, 2 bias MFMA 128) but 4 smfmac is required by our P layout |
| VALU work | **+217** | `SQ_ACTIVE_INST_VALU` 402.1 vs 293.8 per wave-iter; other-VALU +60/wave, INT32 +50/wave, MUL +9/wave | <= +217 recoverable at vendor levels |
| instruction fetch | **+40** | `SQ_IFETCH` 101.4 vs 81.4 | <= +40 (falls out of VALU/code-size cuts) |
| VMEM latency exposure | ~+80 (inferred) | level ratio 1.20-1.33x vendor (was 2.25x); counter scale unstable | <= +130 (old §7.7 estimate) |
| LDS latency / instructions | **-1 / +1** | `SQ_INST_LEVEL_LDS` 31.5 vs 32.1; LDS inst 52.8 vs 52.5 | closed |
| **accounted work delta** | **~+593 (29 %)** | | |
| **stalls / lost overlap** | **~+1477 (71 %)** | `SQ_WAIT_ANY` +150/wave = +300 cycles/SIMD measured directly; the rest is not visible in counters (barrier windows, latency overlap) | upper bound +1477, but gen2/gen3 measured that structural attempts make it worse (barrier removal WAIT_INST_ANY 543->627; gen3 single-window 334.6 vs 367.2 TFLOP/s) |

Ranked remaining items: (1) **lost overlap/stalls, ~+1477 cycles/SIMD (21 % of wall)** - no known
structural lever left at tile granularity (gen2/gen3 closed negative); (2) VALU work, +217 (3.1 %) -
other-VALU/INT32 down to vendor levels; (3) XDL, <= +192 (2.7 %); (4) VMEM/IFETCH, <= +120 (1.7 %).
Even closing every accounted-work line leaves ~6472 cycles/SIMD vs vendor 4995 (1.30x): the residual
gap is overlap, not instructions. The kernel line is therefore final at 1.41x / 365.7 TFLOP/s.

### 5) Correctness gate (independently re-run) and grid-occupancy caveat

* **15 shapes, 16 rows, 3 independent repeats**: `REF_KERNEL_OK` **15/15 in all three**; `PARITY_OK`
  15/15 in all three; our kernel's `bad=0` on every shape (mean_err 0.8-1.1e-3 vs vendor 2.2-2.8e-3,
  i.e. we are numerically closer to the CPU model than the vendor).
* **15 shapes, 64 rows, 2 repeats**: `REF_KERNEL_OK` 12/15 - **all three failures are the VENDOR
  kernel exceeding the absolute tolerance** (vendor `bad=1`, `bad=4`, `bad=5` elements on the three
  shapes); our kernel's `bad=0` on all 15, `PARITY_OK` 15/15. (The gate is `bad(ours)||bad(vendor)`;
  the same 12/3 was already the case at §7.7.)
* **determinism 3 runs x 5 shapes**: **5/5 DET_OK** in both repeats.
* **Grid occupancy (verified from `tests/gpu/ref_host.cpp`: `grid=(ceil(S/256), H, B)`,
  block=512 threads = 8 waves)**: MI350 has **256 active CUs/package** (kernel-wiki
  `mi350-arch-topology`: 8 XCD x 32 active CUs). One block occupies one CU (252 VGPR => 2 waves/SIMD
  => 8 waves = 1 block/CU; LDS 102400 B also allows only one block/CU).

| shape (B,H,S,SK,gqa) | blocks = ceil(S/256)*H*B | % of 256 CUs | fills the GPU? |
|---|---|---|---|
| B1 H8 S2048 SK2048 (benchmark) | **64** | **25.0 %** | no |
| B2 H8 S700 SK640 | 48 | 18.8 % | no |
| B2 H4 S1024 SK1024 | 32 | 12.5 % | no |
| B3 H3 S768 SK1280 | 27 | 10.5 % | no |
| B2 H4 S300 SK129 / B1 H16 S256 / B1 H8 S512 | 16 | 6.2 % | no |
| B1 H4 S300 / B1 H2 S1024 SK1088 | 8 | 3.1 % | no |
| B1 H2 S512 / B1 H4 S77 | 4 | 1.6 % | no |
| B1 H2 S256 SK1000 / B1 H2 S200 SK1337 | 2 | 0.8 % | no |
| B1 H1 S256 | 1 | 0.4 % | no |

  **None of the 15 shapes reaches 256 blocks**, so every absolute throughput number in this document
  is a **per-CU efficiency number**, not a package-capability number: on the benchmark shape
  365.7 TFLOP/s is achieved on 64 of 256 CUs = 5.71 TFLOP/s/CU (vendor 515.5/64 = 8.05 TFLOP/s/CU).
  A fully filled 256-CU grid would project to ~1463 (ours) / ~2062 (vendor) TFLOP/s under linear
  scaling - a projection only (memory/compute balance may reduce it), not a measurement.
* **Counter normalisation is unaffected by the grid fill**: the per-wave-iteration values divide by
  `waves x iters = blocks x 8 x (SK/128)`, i.e. they are per-resident-wave/per-CU numbers already.

**What is certified:** at Main `f4cccb60d53b51e30695b89ee4590039f270de18`, `fmha_fwd_hd128_mxfp8_vs128`
emits, assembles and passes the 15-shape gate at 16 rows (15/15, bad=0, parity 15/15, x3) and at 64
rows our kernel is bad=0 on all 15 shapes; determinism 5/5; it runs at 365.7 TFLOP/s in the best-case
same-run measurement vs the vendor's 515.5 TFLOP/s on B1H8S2048 = 1.41x slower per CU (70.7 % of vendor
efficiency), with the structural lines above (2 barriers/128 keys, 1 realign, 4 DMA staging ops,
next-tile QK in the compute group, 4-smfmac complementary row sums, 2-of-4 LUT-correction copies,
0 smfmac hazard violations, 0 LDS bank conflicts, LDS instruction count at vendor level).

**What is not certified:** (a) the absolute TFLOP/s figure is a per-CU number at 25 % grid fill - no
full-package claim; (b) the kernel is not instruction- or stall-equal to the vendor: 1.41x from work
+ overlap, with the remaining 71 % of the gap in non-counter-visible overlap; (c) `setprio` is absent
by design (opt-in, measured -3 %) so the §7.5 setprio line is waived, not closed; (d) the row-sum
cost is 64 cycles/128 keys (4 smfmac, needed for our P layout) vs the vendor's 32; (e) the emitted
kernel contains **16 loop-variable read-after-`set_var` statements that the DSL contract declares
undefined** - the built binary is deterministic and passes, but this is a latent semantic hazard that
should be converted to explicit SSA values before the kernel is frozen; (f) two profiling counters
(MFMA-busy, VMEM level) are not reproducible on this shared node and are quoted only as ratios/static
accounting; (g) wall-time is a same-run ratio on a shared node (best-case + median of 15 samples), not
a benchmark-grade absolute timing; the cycle columns use the 2.4 GHz convention.

## 8. Unknowns / limitations

* **Clock assumption.** Counters are clock-free; the cycle columns use the 2.4 GHz MI355X boost clock as
  a convention. A lower bound comes from the vendor's XDL busy cycles (2172/SIMD-iteration must fit in
  the wall), which forces >= ~2.09 GHz if the per-SIMD busy interpretation is right; the wall-time and
  TFLOP/s numbers (section 6) are direct measurements and do not depend on this. `SQ_WAVE_CYCLES` is
  *not* a wave lifetime (it implies ~0.55 GHz for both kernels), so it is only used for ratios.
* The `v_mfma_f32_32x32x8_bf16` "l helper" purpose is not proven (Experimental: it consumes
  `v[242:243] = 0x3f803f80` (a bf16 1.0 pair) and `v[244:245]` converted from f32; it likely folds a
  deferred l scale into the O accumulators instead of a per-element VALU multiply).
* The exact complementary-coverage proof for the two smfmac row-sum instructions vs aiter's P packing
  is Experimental (section 4).
* Counter values are from single `rocprofv3 --pmc` invocations with hardware replay; the kernels are
  deterministic, so per-counter values are comparable, but the counters in one invocation may come
  from different replays - no ratio in this document is formed across replays. `SQ_INSTS_VALU_TRANS`,
  `_MOV`, `_CMP` are not exposed by this rocprofv3 build, so the "other VALU" row is a by-difference
  estimate whose split into MOV/TRANS comes from the static disassembly (66 exp and ~150 moves per
  iteration in ours).
* LDS addressing is fully dynamic (`M0 + s_cselect` ring); the offsets listed are the static ones, the
  dynamic window is inferred from them.
* The kernel trace columns `VGPR_Count`/`SGPR_Count` disagree with the code-object descriptors
  (they report 124 VGPR for a 246-VGPR kernel and 64 SGPR for our 48-SGPR kernel); the descriptors are
  authoritative and are what section 1 quotes.

## 9. Reproduce

```bash
export REF_CO=/shared/amdgpu/home/blyu_qle/sessions/ref_mxfp8.co
bash tools/vendor_disasm.sh survey
bash tools/vendor_disasm.sh phases label_block_A_loop --seq
bash tools/vendor_disasm.sh mfma; bash tools/vendor_disasm.sh lds; bash tools/vendor_disasm.sh staging
OURS_CO=build/ours_t.co bash tools/vendor_counters.sh vendor-spec
# §7.5 acceptance checklist for one of our kernels (structural VAL lines + same-run wall + 15 shapes +
# determinism + counters):
bash tools/vendor_validate.sh fmha_fwd_hd128_mxfp8_vs
bash tools/vendor_validate.sh fmha_fwd_hd128_mxfp8_vs --no-counters    # skip the slow profile passes
```

Wall time (unprofiled, the number to quote for performance) comes from `ref_host` itself:

```bash
REF_B=1 REF_H=8 REF_S=2048 REF_SK=2048 REF_HK=8 REF_ROWS=2 REF_ITERS=20 \
  ./build/ref_host72 "$REF_CO" 2>&1 | grep timing        # both kernels: ms/iter + TFLOP/s
```

Kernel descriptors (VGPR/SGPR/LDS) are read from the code-object metadata; the fields are msgpack in the
AMDGPU metadata note:

```bash
python3 - "$REF_CO" <<'PY'
import struct, sys
data = open(sys.argv[1], "rb").read()
for key in (".vgpr_count", ".sgpr_count", ".group_segment_fixed_size", ".kernarg_segment_size"):
    i = data.find(key.encode()) + len(key)
    b = data[i]
    if b == 0xcc: print(key, data[i+1])
    elif b == 0xcd: print(key, struct.unpack(">H", data[i+1:i+3])[0])
    elif b == 0xce: print(key, struct.unpack(">I", data[i+1:i+5])[0])
    elif b < 0x80: print(key, b)          # positive fixint
PY
```

Profile environment (node 53): use `/shared/apps/ubuntu/opt/rocm-7.2.4/bin/rocprofv3` (v1.1.0) and
`unset GPU_DEVICE_ORDINAL CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES` before `HIP_VISIBLE_DEVICES=5`;
the ROCm 10.1.0 profiler (1.4.1) aborts on this node.
