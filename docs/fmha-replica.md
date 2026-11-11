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
  - 96 VALU per tile of `v_perm` / `v_pk_sub_u16` / `v_and` that the vendor does not have. Which stage emits them (P packing or scale-word preparation) is the first thing R3 traces;
  - one tile per pass, against the vendor's two-tile unrolled pass with two exits.
- **Exact operand equality is 1.8 %:** registers are allocated differently, the gap §4 step 3 addresses.
