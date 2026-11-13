# W5: reproducing the vendor MXFP8 FMHA through the safe DSL

Experiment W5 of [`two-weavers-plan.md`](two-weavers-plan.md). The target is the vendor forward kernel we benchmark against:
- the code object at `REF_CO` (`_ZN5aiter28fmha_fwd_hd128_mxfp8_gfx950E`), hd128;
- disassembled with the ROCm 10.1 `llvm-objdump -d --mcpu=gfx950`.

The question is whether the **safe, structured DSL** can express that schedule 1:1: FnBuilder, stages and the Weavers, plus counted safe-asm windows only where needed. For each remaining difference this document records why:
- expressible now;
- needs a back-end Weaver feature;
- needs a DSL feature;
- blocked by a safety / verification rule (and whether the rule is right).

Vendor code is never committed. Every vendor-derived listing is produced at run time from `REF_CO` under `/tmp` or `build/`.

## 1. Tools (deliverable 1)

| Tool | What it compares |
|---|---|
| [`tools/isa_diff.py`](../tools/isa_diff.py) `A B [--range-a L1:L2] [--range-b ..]` | The opcode stream without fillers (`s_waitcnt`, `s_nop`), aligned (longest matching blocks). Registers over the aligned pairs: identical, and identical up to one consistent renaming per register file. Waits / nops in front of each aligned pair: only in A / only in B / different counts / nop wait states |
| [`tools/isa_forms.py`](../tools/isa_forms.py) `kernel.dis corpus*.s` | Instruction forms (mnemonic + operand kinds + modifiers) of a kernel against everything our backend emits (the md5-manifest kernels): same form / mnemonic only / absent |
| [`tools/isa_cfg.py`](../tools/isa_cfg.py) `file.dis` | Basic blocks, back edges, natural loops with their exits, out-of-line loop blocks, join blocks |

## 2. The vendor kernel

| | vendor | ours (`fmha_fwd_hd128_mxfp8_v3_vs128`, Main `1706754f`) |
|---|---|---|
| VGPR / SGPR / AGPR, LDS | 246 / 99 / 0, 84 224 B | 252 / 48 / 0, 102 400 B |
| instructions without fillers | 8 267 | 8 893 |
| `s_waitcnt` / `s_nop` | 333 / 589 | 256 / 204 |
| partial waits `lgkmcnt(n > 0)` | 288 | 182 |
| distinct opcodes | 77 | — |

**Structure:**
- Fully unrolled code per wave group (A and B), two tiles per loop pass.
- A separate `dense_short` schedule for short grids.
- Four out-of-line rollback blocks (the lazy softmax-reference rebase).
- A shared write-out.

`isa_diff.py ours vendor` on the whole kernels: opcode alignment 18 % of ours / 19 % of the vendor. On the aligned pairs, registers are identical up to one renaming for 27 %. Today's kernel computes the same thing with a different algorithm and schedule (per-tile rebase, scaled PV, VGPR-resident constants), so the 1:1 question is answered by R2 / R3 below, not by this diff.

## 3. R2 — the diagnostic: what our pipeline adds, changes or rejects

R2 lifts the vendor stream onto our pipeline's vocabulary. Each finding is measured on the disassembly; class names as in the introduction.

### 3.1 Instruction forms (`isa_forms.py` against all 234 manifest kernels)

| Share of vendor instructions | Forms | Class |
|---|---|---|
| **81.5 %** (7 488) | 80 forms our backend already emits | expressible now |
| **12.8 %** (1 172) | 32 forms whose mnemonic we emit but not with these operand kinds. Almost all are **uniform values as SGPR operands of VALU instructions**: `v_cvt_scalef32_pk_fp8_bf16 … s` (576), `v_pk_fma_f32 … s[2]` (256), `v_fma_f32 … s` (130), `v_mul_f32 / v_sub_f32 / v_add_f32 / v_add_u32 … s`. Also `s_cselect` with inline constants / literals, `v_cndmask_b32_e64` with an inline constant and an SGPR-pair mask, `v_cmp_* s[2]` (e64 mask destinations) | **back-end Weaver / lowering**: keep wave-uniform operands in SGPRs and fold them into the VALU operand (one constant-bus slot; the safe-asm `Operand::BUS` rule already models it) |
| **5.8 %** (529) | 18 absent forms: `v_mov_b64 v[2], 0` (355, accumulator zeroing); `s_sub_u32`, `s_cmp_ge_u32`, `s_cmp_eq/le_u32`, `s_addk_i32`, `s_or_b32` (address / loop counters); `v_mul_i32_i24`; `v_cmp_nge_f32` (NaN-safe compare); `v_mfma_f32_32x32x8_bf16` (the helper MFMA); `s_mov_b64 exec, …` (exec masking, 21) | missing **MOps** (straightforward), except exec masking: **DSL feature** (lane-subset execution) |

### 3.2 Control flow (`isa_cfg.py`)

58 basic blocks.

- **Two-exit loops.** The A and B main loops each have two exits, one after each tile of the unrolled pair, to *different* drain blocks (`label_block_A_drain1` / `drain0`). Our DSL loops (`begin_loop` / `break_if` / `end_loop`) have one continuation after the loop. A second exit would have to become a flag plus an `if` after the loop, i.e. different code. → **DSL feature**: multi-exit loops with distinct continuations, or a loop form whose unrolled halves each own an exit.
- **Out-of-line cold blocks.** The rollback blocks are taken rarely; they are laid out after the loop and jump back into the loop body (`rollback_overflow_* → rollback_done_*`). A DSL `if` places its body inline (branch over it). Executed paths would be equal, but code layout and branch polarity differ. → **back-end Weaver**: block layout (cold-path placement, branch inversion).
- The rest (setup, the `dense_short` / main split, write-out joins) is structured.

### 3.3 Hazard padding (nops by producer → consumer class)

| Context | vendor: sites (states) | ours `_vs128`: sites (states) | Reading |
|---|---:|---:|---|
| VALU → VALU | 518 (524) | 102 (242) | Mostly `v_cmp_lt_i32 vcc` → `s_nop 0` (**1** state) → `v_cndmask_b32_e64 …, vcc` in the causal / tail masks (512 sites). Our strict hazard pass demands **2** states for a VALU VCC write → VALU VCC read (`hazards.rs:133`, taken from the LLVM hazard recognizer as the oracle). The two differ, and the vendor kernel is correct on hardware with 1. → **safety rule**: a replica gets one extra state per site unless the rule is re-derived. Open: check the requirement against `GCNHazardRecognizer` for gfx950 and a hardware probe. The ISA requirement must decide, not the vendor |
| VALU → permlane | 14 (42) | 10 (26) | Same rule on both sides (`v_permlane32_swap` source written by VALU) |
| trans → VALU | 14 (26) | 4 (20) | Both pad the transcendental result; ours on fewer sites (different instruction mix) |
| SALU → LDS DMA | — | 32 (48) | **Ours only**: the M0 write before each `buffer_load … lds`. The vendor sets M0 once per group of issues, with offsets in the instruction. Our `skip_m0` reuse covers only same-source runs → **back-end Weaver**: M0 hand-off planning across issues |
| MFMA / barrier → branch / VALU | — | 6 (112) | **Ours only**: block-end MFMA padding at loop ends and before branches (`edge_pad`). The vendor relies on the next consumer being far enough → **back-end Weaver**: cross-block hazard distance (pad on the edge that needs it, or not at all when the successor is independent) |

### 3.4 Waits

The vendor issues 288 partial `lgkmcnt(n > 0)` waits; ours issues 182. Both use counted waits, which our model and verifier support. The difference follows from the different read / MFMA order, not from a missing capability → expressible now.

### 3.5 What gpu_verify can say about a lifted kernel

`gpu_verify` works on the DSL IR (values, tiles, stage contracts). A kernel lifted to MIR, or embedded as safe-asm blocks, is **opaque** to it:
- an `Op::Asm` block counts in the certificate as a trusted assumption (`asm_opaque`);
- inside the block only the safe-asm verifier's clock model applies (wait-state distances, borrows), not the bounds, race or protocol proofs.

So the lifted kernel would carry a certificate of "N opaque blocks, 0 proven". Proofs are only available for what R3 expresses in the DSL.

### 3.6 Feature list for the back-end Weaver (W3) from R2

1. **SGPR operands for uniform values** in VALU instructions (constant-bus aware; 12.8 % of the vendor's instructions).
2. **Block layout:** cold blocks out of line, branch polarity.
3. **Multi-exit loops** at the machine level (with the DSL feature in §3.2).
4. **M0 planning across LDS-DMA issues** (base + instruction offset).
5. **Cross-block hazard distance** instead of block-end padding.
6. **Authored filler instructions** (`s_nop`) in pinned windows; and a re-derived VALU VCC write → VCC read distance (§3.3).
7. **Missing MOps:** `v_mov_b64`, `s_sub_u32`, `s_cmp_{ge,eq,le}_u32`, `s_addk_i32`, `s_or_b32`, `v_mul_i32_i24`, `v_cmp_nge_f32`, `v_mfma_f32_32x32x8_bf16`, `s_mov_b64 exec`.

## 4. R3 — the main question: the vendor schedule in the safe structured DSL (in progress)

Plan: a new kernel (`fmha_fwd_hd128_mxfp8_v3_replica`) built on the existing stages and pipeline, moved step by step onto the vendor's algorithm and schedule. Each step is measured with `isa_diff.py` (alignment, registers, fillers) and with port_gate ACC and same-run performance. Steps, in the vendor's order of importance (docs/fmha-vs-vendor-asm-diff.md, docs/fmha-vendor-stream-design.md):

1. **Algorithm:** the lazy rebase outside the loop (rollback blocks), unscaled PV (`v_mfma_f32_32x32x64_f8f6f4`, 8 per tile) with the P conversion through `v_cvt_scalef32_pk_fp8_bf16`, two `v_smfmac` row sums, the bf16 helper MFMA.
2. **Copy stream:** K 2 slots / V 3 slots, copies issued inside the QK window, completion at the end of the VALU phase.
3. **Registers:** the vendor's VGPR roles pinned (O v116–179, S v52–115, K / V v20–51, Q v4–19, P reusing S) through the DSL's register plan (`pin_vgpr_at`, shares).
4. **Order:** per phase, the vendor's instruction order through weave regions / authored order. Counted safe-asm windows only where the DSL cannot state the order.
5. **Two-exit loop / cold blocks:** as far as the DSL allows; the rest is listed against §3.6.

### 4.1 Starting point: the steady loops side by side

`isa_diff.py ours.s vendor.dis --range-a BB7:BB8 --range-b label_block_A_loop:label_block_A_drain1` compares group A's loop. Ours runs one tile per pass, the vendor two:

| per tile | ours | vendor (pass / 2) |
|---|---:|---:|
| instructions without fillers | 463 | 220 |
| VALU | 361 | 113 |
| scaled QK MFMA / unscaled PV MFMA | 16 / 0 (PV scaled) | 8 / 8 |
| `ds_read_b64_tr_b8` / `ds_read_b128` | 32 / 16 | 32 / 16 |
| LDS-DMA issues | 5 | 5 |
| `v_pk_add_f32` + `v_pk_mul_f32` (rebase) | 32 + 32 | 0 (out-of-line rollback blocks) |
| `v_perm_b32` + `v_pk_sub_u16` + `v_and_b32` (our P / scale path) | 32 + 32 + 32 | 0 |
| `v_max3_f32`, `v_cvt_pknorm_u16_f32`, `v_cvt_scalef32_pk_fp8_bf16` | 31, 32, 32 | 32, 32, 32 |
| `s_waitcnt` / `s_nop` | 20 / 11 | 16 / 1 |

What the table shows:
- **Same counts:** the reads, the copies and the softmax core are already the same per tile.
- **The rebase:** ours is our lazy recipe's rollback branch, laid out *inline* in the loop body (an `if`), whereas the vendor's rollback blocks are out of line. That is the block-layout item of §3.6, not an algorithm difference.
- **The real algorithm differences:**
  - PV through the scaled MFMA (the vendor: unscaled);
  - 96 VALU per tile of `v_perm` / `v_pk_sub_u16` / `v_and` that the vendor does not have (traced in §4.2: the LUT copy, not the long-sequence loop);
  - one tile per pass, against the vendor's two-tile unrolled pass with two exits.
- **Exact operand equality is 1.8 %:** registers are allocated differently, the gap §4 step 3 addresses.

### 4.2 Step 1: unscaled PV; where the 96 extra VALU come from

**Trace.** `_vs128` has two versions of each group's loop:
- the **LUT copy** (`FmhaBlock::lut_copy`: sequences of fewer than 4 key tiles keep the LUT-corrected fp8 conversion, `reduce.rs`: `perm` → `pk_sub_u16`), loops `BB7` / `BB26`;
- the **vendor-recipe copy** (`set_vendor_ref`, lazy rebase), loops `BB48` / `BB67`.

The 96 VALU are the LUT correction and live only in the short-sequence copy. The vendor has no such copy (it accepts the larger error on short sequences; our port gate measured 1.31× the vendor's mean error without it at S = 300, SK = 129). The comparison below uses the vendor-recipe loop.

**Change.**
- `AttnBody3::pv_unscaled`: PV through `v_mfma_f32_32x32x64_f8f6f4`, i.e. the DSL's `var_mfma_fp8_32x32x64`, as the vendor does;
- the row sum without its 1/16 (`OnlineSoftmax::l_total_unit`), so O / l is unchanged (powers of two).

Kernel `fmha_fwd_hd128_mxfp8_v3_replica` (registered, not dispatched; the md5 manifest gains this one entry). **Class: expressible now** — no new DSL or backend feature was needed.

| | `_vs128` (vendor-recipe loop) | replica step 1 | vendor (per tile) |
|---|---:|---:|---:|
| instructions / tile without fillers (incl. the inline rollback branch) | 337 | 335 | 220 (+ out-of-line rollback) |
| scaled / unscaled MFMA per tile | 16 / 0 | 8 / 8 | 8 / 8 |
| opcode alignment with the vendor loop | 36.8 % | 39.1 % | — |
| registers identical up to renaming | — | 3.8 % | — |

**Gates.**
- `gpu-verify --strict`: 0 errors.
- `port_gate` (replica as `_vs128`, STRESS skipped): BUILD, VERIFY, ACC16 15/15, ACC64 vendor parity on all paths (15/15, 15/15, 15/15, 13/13) and DET 5/5 all PASS.

**Performance, same run (normal data, TF/s, ratio to the vendor in the same run):**

| shape | `_vs128` | replica step 1 | change |
|---|---|---|---:|
| per-CU 64 blocks | 688.3 (0.941×) | 670.6 (0.925×) | −2.6 % |
| B4H8S2048 | 1883.9 (0.973×) | 1829.2 (0.948×) | −2.9 % |
| B1H8S8192 | 2484.0 (0.977×) | 2427.9 (0.957×) | −2.3 % |
| B2H8S2048 | 1247.5 (0.953×) | 1228.5 (0.952×) | −1.5 % |

**Reading.**
- The vendor's MFMA mix costs us 2–3 % with our schedule: the static counts are equal and the nops / waits nearly so (12 / 23 vs 13 / 20), so the difference is dynamic.
- Unscaled PV is kept in the replica (it is the vendor's form) but **not** for `_vs128`.
- Open: why the unscaled form runs slower in our order. Candidates are its source-C read timing / dependency rules in our hazard model and the scheduler's latency table; an ATT trace would settle it.

### 4.3 Step 2: the two-tile pass with static slots and one drain per exit

**Change.** `Pipeline::unroll2` (`TwoPhaseOffset`):

```
loop { tile it, static slot 0; idx = 1; break_if(it + 1 >= nkv - 1);
       tile it + 1, static slot 1; idx = 0; it += 2; break_if(it >= nkv - 1) }
if (idx == 1) { drain: last tile it + 1 in slot 1 } else { drain: last tile it in slot 0 }
```

Each drain keeps the full-tile / partial-tile `if`. Ring slot offsets become immediates, as in the vendor's alternating `0 / 0x4100`.

**Class.** Expressible now. The DSL's `break_if` already allows several exits from one loop; the `RegStaged` cadence used the same idiom.
- The vendor's layout (each exit edge jumps straight to its drain) is a **back-end Weaver** pass: jump threading on the per-edge constant `idx` plus drain layout. Agreed with backend-llvm-tail (W3): a plain MIR CFG pass, scheduled after the missing MOps.
- So no DSL multi-exit construct is needed.

**Static, group A loop vs the vendor's:**

| | replica step 2 | vendor |
|---|---:|---:|
| instructions without fillers | 635 | 439 |
| VALU / SALU | 442 / 24 | 226 / 46 |
| MFMA scaled / unscaled / smfmac / helper | 16 / 16 / 4 / 2 | 16 / 16 / 4 / 2 |
| `ds_read_b64_tr_b8` / `ds_read_b128` / LDS-DMA | 64 / 40 / 10 | 64 / 40 / 10 |
| `s_waitcnt` / `s_nop` | 40 / 23 | 32 / 2 |

The matrix, LDS and copy streams of the pass now have the vendor's exact counts. The VALU excess (+216) is the two inline rollback branches (about 100 each; out of line at the vendor, §3.2). Opcode alignment is 19 % because the inline rollback blocks interrupt the stream, and registers are still allocated differently (4 % equal up to renaming).

**Gates** (`port_gate`, the replica as `_vs128`, STRESS skipped): **OVERALL PASS**.
- BUILD, strict VERIFY (0 errors);
- ACC16 15/15;
- ACC64 vendor parity on all four paths;
- DET 5/5;
- the PERF rows.

**Performance, same session (normal data, ratio to the same-run vendor):**

| shape | `_vs128` | step 1 | **step 2** |
|---|---:|---:|---:|
| per-CU 64 blocks | 0.941× | 0.925× | **0.955×** (693.9 TF/s) |
| B4H8S2048 | 0.973× | 0.948× | **0.981×** (1894.9) |
| B1H8S8192 | 0.977× | 0.957× | **0.988×** (2508.1) |
| B2H8S2048 | 0.953× | 0.952× | **0.968×** (1266.9) |

**Reading.**
- The two-tile pass with static slots more than recovers step 1's loss: +1.0 to +1.6 % over `_vs128` in vendor ratio.
- Open: whether `unroll2` alone (with scaled PV) also helps `_vs128`. That is a separate, W4-gated proposal.

### 4.4 Step 3: the vendor's VGPR roles as register pins

The vendor's accumulator registers (group A loop) are:
- **S**, QK destinations: `v[52:67] v[68:83] v[84:99] v[100:115]`;
- **O**, PV destinations: `v[116:131] … v[164:179]`;
- K / V fragments: `v[20:51]`;
- Q: `v[4:19]`;
- P: `v[52:67]`, reusing S.

The DSL states them with `pin_vgpr_at` (`AttnBody3::pin_roles`). Pinning showed three things.

1. **A verification rule was wrong.** Our allocator checks required MFMA / SMFMAC operand tuples to sit at a multiple of their width (`v[52:67]` rejected: "not aligned to its width 16"). The ISA requirement on gfx90a+ is only 64-bit (even) alignment, and the vendor kernel runs `v_mfma_scale … v[52:67], v[20:27]` / `v_mfma_f32_32x32x64_f8f6f4 v[116:131]`. → **Safety rule: wrong, fixed** (`regalloc.rs`):
   - the checks require even alignment;
   - the allocator still places its own tuples at their width, but an explicit pin needs only even alignment;
   - md5 manifest unchanged.
2. **The allocator placed pins last.** The `TwoPhase` allocator ordered values by width then length, so a pinned V16 could find its registers already taken. → Pinned values are now allocated first (md5 unchanged), and the pin error says what blocks it: forbidding pins, or the occupancy.
3. **Pins cost performance with our schedule.**
   - **S pinned** (vendor numbers, 248 VGPRs, scheduled normally): port_gate correctness all PASS. But same-session performance drops to 0.889× / 0.974× / 0.946× / 0.897× the vendor (per-CU / B4H8S2048 / B1H8S8192 / B2H8S2048), against 0.955× / 0.981× / 0.988× / 0.968× unpinned (step 2).
   - **S and O pinned:** the scheduler runs out of VGPRs in two of the four body copies and falls back to the unscheduled order; loop invariants spill to LDS (16 KiB / 32 KiB). Result 0.863× / 0.917× / 0.928× / 0.885×.
   - **Q at `v[4:19]`:** not honourable. A value of ours is live there from the prelude on, because our prelude and other residents are not the vendor's.

**Reading.** Pinning a subset of the vendor's register numbers onto *our* schedule and *our* other residents is expressible, but it is not a replica: the vendor's numbers belong to its whole allocation and order. 1:1 registers need:
- the back-end Weaver's **schedule-aware register allocation**, i.e. allocate the remaining values around a full plan;
- the **full plan** stated in the DSL: K / V fragments, P reusing S, the scale words, all transient read results.

The replica therefore keeps `pin_roles` **off** (its asm is step 2's). The two allocator fixes stay; both are md5-neutral.

| Difference | Class |
|---|---|
| MFMA tuple at a non-width-aligned base | safety rule — was wrong, fixed |
| pins allocated after free values | back-end allocator — fixed (pins first) |
| a pinned role subset degrades our schedule; Q's registers taken by our prelude | **back-end Weaver**: schedule-aware RA around a complete register plan |
| K / V / P / scale-word registers (transient values, the vendor's rotation `v[20:51]`, P in S) | **DSL**: a register plan for transient fragments (`reg_share` exists; a rotation / ring plan for read results does not) |

### 4.5 Step 4a: the vendor's instruction forms (back-end lowering tokens)

**Change.**
- The replica turns on two of backend-llvm-tail's lowering tokens (Main `dfc305fc`):
  - `cmpneg`: the lazy rollback test written as `!(thr >= max)` (`OnlineSoftmax::nan_rollback`) becomes the vendor's NaN-taking `v_cmp_nge_f32` (12 sites);
  - `movb64`: fragment / accumulator zeroing as `v_mov_b64 v[..], 0` (136 sites).
- The drain selector is written `0 < idx`, a less-than compare that the jump-threading pass (`jthread`, next) folds per edge.

**Class:** expressible now. The forms were missing MIR operations (§3.1), and are now opt-in lowering patterns.

**Gates** (`port_gate`, the replica as `_vs128`, STRESS skipped): **OVERALL PASS**.

**Performance, same session (normal data, ratio to the same-run vendor):**

| shape | `_vs128` | replica step 4a |
|---|---:|---:|
| per-CU 64 blocks | 0.958× (690.1 TF/s) | **0.969×** (701.0) |
| B4H8S2048 | 0.943× | **0.949×** |
| B1H8S8192 | 0.980× | **0.991×** (2515.3) |
| B2H8S2048 | 0.962× | **0.981×** (1278.2) |

### 4.6 Step 4b: exit threading and out-of-line rollback blocks (back-end Weaver passes)

**Change.**
- The replica adds two lowering tokens, both from backend-llvm-tail, Main `720eab3c`:
  - `jthread`: each loop exit jumps straight to its drain; all 8 exit edges are threaded;
  - `coldlayout`: the blocks of an `if` marked unlikely go after all hot code, the hot path falls through, and the cold path jumps back.
- The lazy rollback is marked with `b.begin_if_unlikely` (`OnlineSoftmax::cold_rollback`).
- VGPRs unchanged (248); strict verify 0 errors.

**Static, group A loop (hot blocks only) vs the vendor's:**

| | replica 4b | vendor |
|---|---:|---:|
| instructions without fillers | **437** | 439 |
| VALU / SALU | 247 / 24 | 226 / 46 |
| MFMA, LDS reads, LDS-DMA | equal | equal |
| `s_waitcnt` / `s_nop` | 40 / 22 | 32 / 2 |
| opcode alignment | 28.4 % | — |

The loop now has the vendor's size and instruction mix. What is left:
- order: the alignment is 28 % (§4.7);
- 21 VALU more / 22 SALU fewer: the vendor keeps uniform values in SGPRs and does its address arithmetic on the scalar unit;
- waits and nops: 40 / 22 here vs 32 / 2 at the vendor.

**Gates.** port_gate OVERALL PASS.

**Performance, same session (× vendor):**

| shape | `_vs128` | replica 4b |
|---|---:|---:|
| per-CU 64 blocks | 0.952× | **0.971×** (700.2 TF/s) |
| B4H8S2048 | 1.012×¹ | **0.993×** (1920.6 TF/s vs 1830.4) |
| B1H8S8192 | 0.979× | **0.989×** |
| B2H8S2048 | 0.970× | **0.978×** |

¹ The vendor measured low in that run (1808.7 vs 1933.2 TF/s); the absolute TF/s favour the replica by 4.9 %.

### 4.7 Per-phase order: where the two hot loops differ (analysis)

The class timeline of one tile of group A's hot loop, written as runs (`v` VALU, `s` SALU, `t` `ds_read_b64_tr_b8`, `r` `ds_read_b128/b32`, `D` LDS-DMA issue, `Q` scaled QK MFMA, `P` unscaled PV MFMA, `R` smfmac row sum, `H` helper MFMA, `w` wait, `n` nop, `B` barrier, `p` setprio, `j` branch):

```
vendor   p v35 n j v64 s t8 w s p B | j w R2 P t4 w P t4 w P t4 w P t4 w P t4 w P t4 s4 v w P r2 s4 w v3 P r6 H w v6
         Q s2 v2 s2 v2 r2 Q s4 D s r2 w Q s D r2 w Q s D r2 w Q s D r2 w Q s D r2 w Q w Q p B
replica  s v3 j s v s v s v6 s n2 D s v6 s n2 D s n D s n D s n D p v33 n v3 n v3 j v50 w v15 w B |
         p H r R t4 w v R t4 r w P t4 w v r P t4 w v P r2 w P r3 w Q r2 w v Q r2 w Q t4 w Q t4 w P t4 w P t4 w P r2 w P r2
         w Q r2 w Q r2 w Q w Q w B
```

| Order difference | Vendor | Replica | Class |
|---|---|---|---|
| **Copy issue point** | the 5 LDS-DMA issues of the tile *two* ahead sit inside the QK window, one between consecutive QK MFMAs; completion at the end of the next VALU phase | the 5 issues of the *next* tile at the start of the VALU phase, each behind its own M0 write + nop; completion before the mid barrier | **DSL / pipeline**: a deeper copy ring (K 2 slots, V 3 slots, distance 2) and a stage issued from inside the matrix phase. Pipeline cadence and stage interleave exist (`rotate`, weave regions); the distance-2 ring for one buffer and DMA units woven between QK MFMAs do not |
| **V-fragment prefetch** | 8 `ds_read_b64_tr_b8` of the next PV at the end of the VALU phase, before the barrier | the V reads inside the M phase, 4 before each PV pair | expressible: the PV read units moved into the softmax stage's tail (a weave-order change) |
| **PV / QK order in the M phase** | row sums, the 8 PV MFMAs back to back (4 reads each), then the helper MFMA, then the 8 QK MFMAs | helper first, PV and QK interleaved in pairs (the weave's declaration order) | expressible: weave order of the merged `mma.pv` / `mma.qk` stage |
| **M0 per DMA issue** | one M0 per group, offsets in the instruction | `s_mov m0` + `s_nop` per issue | back-end Weaver (M0 planning, W3 B2) |
| **Scalar address arithmetic** | 46 SALU per pass | 24 SALU, more VALU | back-end Weaver / lowering (uniform values in SGPRs, W3 B2) |

The copy issue point is the largest structural difference. It is also the one the earlier copy-stream prototypes measured negative on `_vs128` ([`fmha-vendor-stream-design.md`](fmha-vendor-stream-design.md) §4–§6). The replica now has the vendor's tile pair, static slots and out-of-line rollback, so it is the right place to re-test it.

Next steps:
1. The two expressible weave-order changes (V prefetch, PV block before QK).
2. The distance-2 copy ring with in-window issue. This is a pipeline extension.

### 4.8 Step 4c: the vendor's matrix-phase order, authored (measured negative)

**Change.** `declare_region("m_phase", authored = true)` keeps the merged `mma.pv` + `mma.qk` stage in its declaration order: row sums, the PV block with its V reads, the helper MFMA, then the QK block with its K reads. That is the vendor's M-phase order. List scheduling, slot filling and ldshoist are off inside the region; waits are recomputed.

**Static.**
- Opcode alignment with the vendor's loop rises from 28 % to **55 %**.
- Instruction count unchanged (437).
- Waits / nops 46 / 30 (vendor 32 / 2).

**Performance (same session, × vendor):**

| shape | authored | step 4b in an earlier session |
|---|---:|---:|
| per-CU | 0.894× | 0.971× |
| B4H8S2048 | 0.938× | 0.993× |
| B1H8S8192 | 0.951× | 0.989× |
| B2H8S2048 | 0.918× | 0.978× |

That is −4 … −8 %. **Not adopted**; the replica keeps step 4b.

**Reading.** The vendor's matrix-phase order performs only together with what goes with it:
- the **V prefetch** before the barrier (§4.7: our first PV MFMA now waits on its own reads, 46 waits);
- its **register rotation**: K / V fragments in `v[20:51]`, so reads for the next MFMA never wait on the previous one's operands;
- its **copy issue points** inside the QK window.

Our list scheduler, given our registers and waits, finds a faster order than the vendor's order does on our registers. Making the authored order fast needs:
- the V prefetch (a DSL weave change, next);
- the **back-end Weaver's schedule-aware RA** for the fragment rotation, the same requirement as §4.4.

**Method note.** An environment toggle does not reach `amdc` inside `port_gate`, because the gate emits through the container. Same-session A/Bs therefore need two kernel names. The pin A/B numbers in §4.4 come from separate builds, not toggles.

### 4.9 Spin-off: the replica configuration as the shipped `_vs128`

Two A/Bs with `tools/perf_ab.sh` (`--families fmha_mxfp8 --release`, old = Main `e48ec5c8`, same session):

| new `_vs128` | geomean new / old | per case (1×8×2048 / 1×8×8192 / 2×32×8192 / 4×8×2048) | verdict |
|---|---:|---|---|
| `unroll2` + `jthread` only (scaled PV, inline rollback) | 0.996 | 0.980 / 0.998 / 1.001 / 1.004 | OK, no gain: not adopted |
| the full replica configuration (step 4b) | **1.017** (7 reps) | **1.021 (IMPROVED)** / 1.008 / 1.008 / 1.032 | OK, adopted |

The gain needs the combination. The two-tile pass alone is neutral, and unscaled PV alone loses 2–3 % (§4.2). Together, plus the out-of-line rollback, they win.

**port_gate of the shipped set with the new `_vs128`: OVERALL PASS** — BUILD, VERIFY, ACC16, ACC64 vendor parity on all paths, DET, STRESS (300 + 100 launches), PERF:
- per-CU 0.971× vendor (recorded 0.959×);
- B4H8S2048 0.983×;
- B1H8S8192 0.991× (2511 TF/s, recorded 2416);
- B2H8S2048 0.973×;
- layout median 0.972× (recorded 0.948×).

`_vs128` is now built with `Vs128Slots { replica: true }`; the md5 manifest changes for `_vs128` only.

**Verifier coverage, found by adopting the replica configuration.** The barrier mutation test (`gpu_tiles.rs`: remove each barrier of `_vs128` in turn; the verifier must notice at least 70 % of the removals) failed: 18 of 54 removals went unnoticed. Two gaps, both fixed (`gpu_verify` / `gpu_protocol`, md5 unchanged, every kernel still verifies with 0 errors):
1. The role barrier balance (`GV-BAR`) summed only the then-arm of each nested if. A barrier missing from an else arm (one of the per-exit drains, the full-tile arm of a tail test) was invisible. The two roles are now also compared on each nested if's barrier delta (else − then; ifs whose arms run the same barriers are ignored).
2. The tile-protocol simulation never took a runtime `break_if`, so the code after a mid-loop exit (the drain of a two-tile pass) was never simulated. In the last simulated iteration each runtime exit is now a decision, keyed by its ordinal so that the roles leave at the same exit.

After the fixes all 54 removals are detected; `cargo test` 548 / 0.

### 4.10 R3 step 5: copy distance 2 (K 2 / V 4 slots), with the B5 placement — correct, still slower (not adopted)

Kernel `fmha_fwd_hd128_mxfp8_v3_ring2`: experimental, registered, not dispatched.

**The ring.**
- `Pipeline::ahead2`: the body stages tile t + 2 in the matrix phase of tile t; the prologue stages tiles 0 and 1.
- `KvRing3::new_kv(.., 2, 4, ..)`. K goes into the slot of t: both groups' QK(t) readers are past the barrier that opens the phase. V goes into slot (t + 2) & 3, and PV reads V from slot t & 3.
- The masked slot keeps the LDS bounds provable. Strict verify: 0 errors, including `GV-PLAN-RING`, the check that the plan's slot claims agree with the simulated protocol.

**B5 plan.** In the matrix-phase merge:
- PV(t) units carry iter = h, QK(t + 1) units h + 1, and the DMA units of t + 2 (`UnitClass::Dma`) h + 2, with h the half of the two-tile pass;
- the phase is `Free`, with one interleave hint per QK MFMA unit;
- there are no explicit waits inside.

The back end (origin/b4ring) takes ring membership from the units' tile markers, orders after the auto-wait strip and recomputes the waits with the ring tokens. It places 16 DMAs between QK MFMAs per loop body.

**port_gate** (STRESS included). Correctness: ACC16 / ACC64 / DET / STRESS all PASS; ring reuse is what STRESS would catch. Performance, same session as the shipped `_vs128` (node 53, GPU 0):

| shape | `_vs128` (shipped) | ring2, copies at the phase start | ring2 + B5 placement | B5 vs `_vs128` |
|---|--:|--:|--:|--:|
| per-CU 64 (TF/s, ×vendor) | 704.9 (0.970×) | 0.885× (earlier session) | 659.4 (0.913×) | 0.935 |
| B4H8S2048 | 1906.6 (1.013×) | 0.937× | 1827.0 (0.954×) | 0.958 |
| B1H8S8192 | 2515.6 (0.993×) | 0.955× | 2452.7 (0.967×) | 0.975 |
| B2H8S2048 | 1286.0 (0.985×) | — | 1204.7 (0.924×) | 0.937 |

**Result.** The B5 placement recovers about half the loss of the hand-placed copies: per-CU 0.862× → 0.913× against the vendor. The distance-2 ring as a whole is still 2.5–6.5 % slower than the shipped distance-1 ring. **Not adopted.**

The §4 finding stands in a weaker form: copies landing during the M phase cost its LDS reads more than the earlier issue gains, even when they are woven into the QK MFMAs. The vendor's ring may only pay off together with its register plan (B6b: the K / V fragment rotation) and its M-phase order (§4.8).

**Follow-up: ring2 + the K / V fragment register rotation + the authored M-phase order (B6b), time-boxed.** Same session, node 53 GPU 0, × vendor:

| shape | `_vs128` | ring2 + B5 | ring2 + B5 + authored M-phase order (§4.8) |
|---|--:|--:|--:|
| per-CU | 706.9 TF/s (0.973×) | 656.4 (0.908×) | 582.3 (0.801×) |
| B4H8S2048 | 1877 (0.971×) | 1856 (0.968×) | 1715 (0.892×) |
| B1H8S8192 | 2515 (0.992×) | 2454 (0.967×) | 2278 (0.899×) |
| B2H8S2048 | 1278 (0.979×) | 1195 (0.920×) | 1081 (0.827×) |

- **The authored order without the rotation** pays the register WAR (§4.8): −11 … −12 %.
- **The rotation itself** cannot be expressed with `reg_ring`. It means "pairwise separate at any time": the 8 K + 8 V fragments of a matrix phase (V8 each, 64 VGPRs per ring) then never reuse registers, and the allocation fails both with whole-phase rings and with sliding windows of 4.
- **What is needed:** the vendor's rotation is a hard share per residue class (fragment i in buffer i mod 4) with the classes kept apart. backend-llvm-tail is adding `reg_rotate(name, 4, fragments)` (`RegMark::Rotate`). The ring2 + rotation + authored-order measurement waits for it.

**B6b, measured with `reg_rotate`** (branch `b6-rot`). The V fragments of PV(t) and the K fragments of QK(t + 1) of each matrix phase go into a rotation of 3 register buffers. With 4 buffers the allocation does not fit: peak 263 > 256 live dwords next to our S / O / P layout. Same session, node 53 GPU 0, × vendor:

| variant | per-CU | B4H8S2048 | B1H8S8192 | B2H8S2048 |
|---|--:|--:|--:|--:|
| ring2 + rotation (falls back to the unscheduled code) | 0.734 | 0.842 | 0.866 | 0.764 |
| ring2 + rotation + authored M-phase order | 0.829 | 0.915 | 0.913 | 0.851 |
| ring2 + B5 | 0.905 | 0.962 | 0.967 | 0.934 |
| `_vs128` (shipped) | 0.982 | 1.001 | 0.992 | 0.980 |

- The rotation removes part of the authored order's register WAR: 0.801 → 0.829.
- The vendor composition (distance-2 ring + rotation + its M-phase order) stays 7–25 % behind our shipped kernel with our register plan. Its v[20:51] rotation needs a register plan as small as the vendor's: Q in v[4:19] and P inside S.

**Not adopted.** The ring / rotation line is closed with the current register plan.

### 4.11 The W3 B2 tokens on the shipped configuration: `m0reuse`, `sgpr`, `sgpr2` — neutral

`lower_opts` of `_vs128` / `_replica` + `m0reuse` (one M0 write per group of DMAs with the same LDS base) + `sgpr` + `sgpr2` (uniform SGPR operands of VALU).

**port_gate:** all PASS.
- ACC16 / ACC64 / DET pass; strict verify reports 0 errors.
- Per-CU 700.9 TF/s (0.973× vendor), B4H8S2048 0.971×, B1H8S8192 0.992×, layout median 0.973×.

**perf_ab `fmha_mxfp8`** (old = Main `4f1e5a02`, 7 reps): geomean **1.004**. Per case: 0.994 / 0.999 / 1.007 / 1.014.

**Not adopted.** On this kernel, the M0 writes and the uniform-operand forms were not on the critical path.

## 5. Findings so far, per difference class (the deliverable)

| Difference from the vendor kernel | Class | State |
|---|---|---|
| Unscaled PV MFMA (8 + 8 per tile) | expressible now (DSL `var_mfma_fp8_32x32x64`) | in the replica (step 1) |
| Two tiles per loop pass, static ring slots, one drain per exit | expressible now (`Pipeline::unroll2`, `break_if` with a `which` variable) | in the replica (step 2) |
| Each exit edge jumping straight to its drain | back-end Weaver: jump threading on a per-edge constant | in the replica (`jthread`, step 4b) |
| Rollback blocks out of line (after the hot code, hot path falls through) | back-end Weaver: cold-block layout + a DSL hint (`begin_if_unlikely`) | in the replica (`coldlayout`, step 4b); VGPRs unchanged |
| `v_cmp_nge_f32`, `v_mov_b64`, `v_mfma_f32_32x32x8_bf16` | missing MIR ops → opt-in lowering patterns | the first two in the replica (step 4a); the bf16 helper is next |
| Scalar forms (`s_sub_u32`, `s_cmp_*_u32`, `s_addk_i32`, `s_or_b32`), `v_mul_i32_i24` | missing MIR ops / lowering patterns | low priority |
| Wave-uniform values as SGPR operands of VALU (12.8 % of the vendor's instructions) | back-end Weaver / lowering (constant-bus aware operand choice) | available as opt-in `sgpr` + `sgpr2` (W3 B2); on the replica configuration with `m0reuse`: correct, perf neutral (perf_ab 1.004), not adopted (§4.11) |
| Exec masking in the epilogue | DSL feature (lane subset) or a counted safe-asm window | not planned |
| Vendor nop after a VALU VCC write before a VCC read (1 state; we demanded 2) | safety rule: over-conservative | lowered to 1 (backend-llvm-tail, `vcc-rule`) |
| MFMA tuples at even, not width-aligned, bases (`v[52:67]`) | safety rule: wrong | fixed (even alignment, §4.4) |
| Pinned registers allocated after free values | back-end allocator | fixed (pins first, §4.4) |
| The vendor's register numbers (roles) | back-end Weaver: schedule-aware RA around a full plan; DSL: register plans for transient fragments | pins cost −4 … −12 % on our schedule; off (§4.4) |
| M0 written per LDS-DMA issue | back-end Weaver: M0 planning | available as opt-in `m0reuse` (54 → 18 M0 writes); neutral on the replica (§4.11), not adopted |
| Block-end MFMA padding | back-end Weaver: cross-block hazard distance | W3, before B4 |
| Copy issue point (vendor: distance-2 ring, DMA issues inside the QK window) | DSL / pipeline extension + back-end Weaver B5 | built (`_ring2`, §4.10): correct (port_gate incl. STRESS), the B5 merge places the copies between the QK MFMAs; still −2.5 … −6.5 % vs `_vs128` (not adopted; next with B6b and the M-phase order) |
| PV block before QK (the vendor's M-phase order) | expressible (authored region) | −4 … −8 % (§4.8), −11 … −12 % on top of ring2, and with the B6b rotation (`reg_rotate`, 3 buffers) still −7 … −15 % vs `_vs128` (§4.10): not adopted; needs the vendor's smaller register plan |
| V-fragment prefetch before the barrier | expressible (weave order) | **blocked on W3 B4+** (pays only with the authored order + schedule-aware RA) |
| The LUT-corrected copy for < 4 tiles (ours only) | accuracy choice: the vendor accepts the larger short-sequence error | kept |
| Verifier blind spots for multi-exit loops (else-arm barriers, mid-loop exits) | verification: incomplete | fixed (§4.9) |
| gpu_verify on lifted vendor code | proofs only for what the DSL expresses; lifted asm is opaque | — |

**Shipped result of W5:** the replica configuration (steps 1, 2, 4a, 4b) is the shipped `_vs128` (§4.9): perf_ab +1.7 % geomean, per-CU 0.971× the vendor.

## 6. R4 — the 1:1 exact replica (route B)

The vendor kernel, rebuilt instruction by instruction, compiled by our compiler and verified by gpu_verify. Route B (user decision): the repository holds only generic mechanisms. These are the lifter, exact mode ([`exact-mode.md`](exact-mode.md)), the E5 structured branch forms and the verifier forms. The vendor's instruction order, registers and the lifted IR are generated on the node at run time from `REF_CO` and are never committed. This section records counts and classes only.

### 6.1 Pipeline

1. `llvm-objdump -d` of `REF_CO` → [`tools/isa_lift.py`](../tools/isa_lift.py) `--kernel --exact`. Every instruction becomes a real DSL op or statement; the blocks become the structured tree:
   - ifs, the two natural loops with their exits as E5 exit-with-body arms;
   - the four rollback blocks as placed `if/else` with layout anchors.
   - Registers live across blocks are E5 scope variables; pointer and descriptor tuples become `Ptr` / `Rsrc` variables.
   - Registers whose every definition is in the prelude are carried by value, so descriptors keep their provenance.
   - Exact-mode pins (`reg_plan`) give every value the author's register.
2. `amdc emit --from-ir` with the normal verification:
   - exact mode keeps the author's order, registers, waits, nops, operand order and encodings, and only checks them (waits, hazards, clobbers);
   - gpu_verify proves the memory accesses.
3. Comparison: [`exact_isa_diff.py`](../tools/exact_isa_diff.py) (aligned instruction stream), [`isa_bytes_diff.py`](../tools/isa_bytes_diff.py) (encodings), [`isa_branch_targets.py`](../tools/isa_branch_targets.py) (each branch reaches the aligned target) and [`kd_fields.py`](../tools/kd_fields.py) (kernel descriptor). Then `ref_host` compares the outputs and timing against the vendor code object, in the same run.

Lifter options that turn the input's facts into IR facts:
- `--kernarg-size`, `--workgroup`, `--lds`, `--fp-mode`: the descriptor / metadata.
- `--contract-from <ir>`: the named arguments and the argument-level contract of OUR replica, which has the same 656-byte kernarg layout.
- `--wgmap`: the workgroup map.
- `--sym-range`: bounds of free size symbols.
- `--fix-waits` / `--fix-lanemask` / `--fix-m0`: the named deviations below.

### 6.2 Result

| | vendor | exact replica |
|---|---|---|
| Instructions | 9 189 | 9 264 (9 188 identical in order, 99.99 %) |
| Encodings of the aligned instructions | — | byte-identical except branch offsets |
| Branch targets | — | all identical by aligned position |
| VGPR / SGPR, LDS, FP mode | 246 / 99, 84 224 B, denorm32 flush / IEEE | identical |
| Kernel descriptor | — | identical except the `kernarg_size` field (656 vs 0; no effect, checked) |
| Output vs the vendor | — | bit-identical (worst diff 0) on all outputs at B1 S2048, B4 S2048, B1 S8192, B8 S4096, B2 S16384 |
| Same-run time vs the vendor (median of 9) | 1 | B4 S2048 1.003, B1 S8192 0.997, B8 S4096 1.001 |
| gpu_verify | — | 0 unproven: LDS 1 167, buffer 97, global 3, barrier-balanced ifs 6 |

### 6.3 Every remaining difference, with its reason

| Difference | Count | Reason |
|---|---|---|
| `s_waitcnt lgkmcnt(2)` added | 12 (6 sites × 2 copies) | The vendor reads LDS data the counter does not yet guarantee and relies on timing. The contention probe failed, so it is not accepted as a timing waiver. |
| `s_nop 0` between an M0 write and its LDS-DMA | 56 | Our hazard rule wants 1 wait state; the M0 probe is inconclusive, so it is a deviation until a probe settles it. |
| `s_nop 0` between a VALU lane-mask write and the next VALU's mask read | 6 | Our rule is 1 wait state (lead decision); the vendor uses 0. |
| A fall-through `s_branch` | 1 removed | The vendor jumps to the next block; exact mode lays out fall-throughs. |
| `s_nop` after `s_endpgm` | 2 | Code-end padding, never executed. |

The deviations cost nothing measurable: the timing above includes them. The vendor's own transcendental-result and `v_readfirstlane` distances are 1 wait state each. They are proven by hardware probes (`tools/trans_hazard_probe.sh`, `tools/readfirstlane_hazard_probe.sh`) and are target rows now, not deviations.

### 6.4 What is trusted

The certificate has 1 trusted assume and 30 trusted contracts:
- **Assume:** the workgroup map `(query tile, head)` is a bijection of the workgroup `(x, y)`. The vendor computes it with an 8-step restoring division; it is the same fact our replica states.
- **Contracts:** the argument-level host contract of our replica (7 extents, 9 kernarg ranges, 2 grid definitions, 11 kernarg identities), plus one size bound: heads ≤ 65 536 · 16. The vendor addresses the descale tensors with `v_mul_i32_i24`, which is exact only while 4 · heads < 2^23.
- **Not proven:** race freedom. The report lists the accesses as bounds-checked, not race-checked (1 167 LDS, 18 global writes), because the lift carries no tile-protocol facts.

### 6.5 What the replica found

Compiler and verifier gaps found by running the vendor's code through them, all fixed generically:
- **Exact-mode lowering:**
  - `s_mov_b64 exec, 0xffffffff` is a zero-extended 32-bit literal, i.e. half a wave issues the LDS-DMA (`ExecSet`). As `exec, -1` it corrupted 40 % of the rows of half the blocks and cost 2 %.
  - Authored operand order and encodings, `s_addk_i32`, `set_m0` of an add, negated compares.
  - Branches laid out as authored; author-pinned adds kept; the author's register counts and FP mode.
- **gpu_verify:**
  - E5 scope variables tracked flow-sensitively;
  - interval rules for the vendor's integer ops;
  - ring counters and their shadows;
  - the EXEC lane count of an LDS-DMA;
  - generic compare branch facts;
  - pointer equivalence through masked high words;
  - barrier balance per loop-exit path (E5 exits with bodies).
- **The lifter itself:** a conditional branch to an unnamed label, the MFMA scale-byte select, the accumulator read of `v_smfmac`, and the layout of the cold blocks.
