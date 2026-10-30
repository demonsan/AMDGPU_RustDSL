# Gen5 window spec: the hand-authored (safe-asm) inner tile window of `_vs128`

Status: **spec + scaffolding** (kernel-opt, 2026-10-30). Target kernel `fmha_fwd_hd128_mxfp8_vs5` (registered, builds, asm and output bit-identical to `_vs128`:
`diff` of the emitted `.s` = 0 lines, md5 faadf00c / c8b7a893 / d17778b0 on 1 8 2048 2048, 4 8 2048 2048, 3 3 768 1280). Everything below was extracted from the
emitted asm of `_vs128` (own backend, long/no-LUT recipe copy, `loop 2` = group A, `loop 3` = group B) and the DSL role `emit_fmha_vs128_role`
(`crates/amdgpu-dsl/src/kernels.rs`).

## 1. What a tile looks like today (per wave, measured from the asm)
One loop iteration = one 128-key tile per wave = two phases separated by two `s_barrier`s; the two wave groups (A = waves 0-3 staging K + K scales, B = waves 4-7
staging V) run the same code half a tile apart, so one group is in V while the other is in M.

| phase | VALU | MFMA | SMFMAC | DS reads | DMA | waits | notes |
|---|---|---|---|---|---|---|---|
| **V** (decide + convert + stage) | 251 (B) / 267 (A) | 0 | 0 | 0 | 4 (B) / 5 (A) | 2 (`vmcnt(0)`, `lgkmcnt(0)`) | all VALU, no matrix work |
| **M** (PV + QK(t+1) + row sums) | 25 (address adds) | **20** (8 PV + 8 QK scaled + 4 bias f16) | 4 | 52 | 0 | 16-18 partial `lgkmcnt` | all matrix, no VALU |
| tail (`loop` bookkeeping, `s_cbranch`) | ~1 | | | | | | |
Instruction mix of the V phase (long copy): `v_max3_f32` 31, `v_pk_add_f32` 32, `v_pk_mul_f32` 32, `v_cvt_pknorm_u16_f32` 32, `v_cvt_scalef32_pk_fp8_bf16` 32,
`v_mov_b32` 20-25, `v_add_u32` 19-23 (DMA address math), `v_cndmask_b32` 7-8, `v_cmp_lt_i32` 6-7, `v_mul_lo_u32` 4-6, `v_add_co_u32`/`v_addc_co_u32` 4-5, a short
rollback decision chain (`v_permlane32_swap_b32`, `v_cvt_f32_f16` x2, `v_cvt_f16_f32` x2, `v_exp_f32`, `v_cmp_lt_f32` x3). (The LUT copy used for < 9 tiles adds
33 `v_and_b32`, 33 `v_lshrrev_b32`, 32 `v_perm_b32`, 32 `v_pk_sub_u16`.)
Today's order in V: `[DMA issue interleaved with its address VALU] -> max3 x31 -> partner max (permlane32_swap) + rollback decision (compare) -> [rollback branch:
O/l rescale] -> pk_add x32 (bias shift) -> pk_mul x32 -> s_waitcnt vmcnt(0) (staged tile landed) -> pknorm x16 -> s_waitcnt lgkmcnt(0) -> pknorm x16 + scalef32 x16 ->
pknorm x16 + scalef32 x16 -> s_barrier`. Order in M: `bias MFMA(S0) -> [4 K-scale ds_read_b32 + V tr reads] -> QK(S0..S3) interleaved with the PV reads ->
PV e0..e3 (x2 halves) with 4 SMFMAC row sums -> s_barrier`.

## 2. Register map (own backend allocation of the `_vs128` loop, group A long copy)
| value | registers | produced by | consumed by |
|---|---|---|---|
| S tiles s0..s3 (16 fp32 scores per lane each) | v[128:143], v[112:127], v[96:111], v[80:95] | QK MFMAs of the M phase of the previous iteration (C/D accumulators, written by `v_mfma_scale`, C input = the bias MFMA result) | V phase: max3 chain, pk_add (in place), pk_mul, pknorm |
| O^T accumulators e0..e3 (16 fp32 each) | v[64:79], v[48:63], v[32:47], v[16:31] | PV MFMAs (in place, scale operands 127/123) | rescale in the rollback block, epilogue |
| row sums `la` (Frag4, dword 2 = 16 x sum) | v[192:195] | 4 in-place SMFMACs (`a_rs` = per-lane ones fragment) | epilogue |
| P words (16 dwords = `pa` (words 0-7) + `pb` (8-15), fp8 e4m3 pairs) | 16 VGPR (V-phase outputs, dead after the PV MFMAs) | scalef32 conversions | PV MFMA B operands, SMFMAC |
| K fragments (A of QK) | v[160:167]/v[184:191] pairs | `ds_read_b128` x2 per MFMA | QK MFMAs |
| V fragments (A of PV) | v[160:175] | `ds_read_b64_tr_b8` x8 per e-half | PV MFMAs |
| K scales | v241..v244 | `ds_read_b32` x4 (`kdpre`) | QK MFMAs (`sk0`/`sk1` via `v_lshrrev_b32`) |
| bias word (two f16) | 1 loop var (`bias2`) | rollback chain | bias MFMA (C of the first scaled QK MFMA) |
Q is register resident (`q0reg` + `q1reg`, 16 VGPR). Total 249-252 VGPR: there is no register headroom, so the target window must keep the S / O / P / Q live ranges
exactly as today (S(t+1) may only be written into an S tile after that tile's last pknorm read).

## 3. Target order (gen5 window), per wave, one tile
Goal: let the 12 QK-side MFMAs of tile t+1 (8 scaled + 4 bias) issue *while* the V-phase VALU of tile t is still running (they only need the S tile that the VALU
just freed), and keep PV of tile t (8 MFMA + 4 SMFMAC) as the only matrix work left for the M window. The window becomes three straight-line asm blocks around the one
wave-level branch (the rollback), exactly like the current DSL structure:
1. **Block A (`asm_max`)** - inputs S(t) (64 fp32 in 4 x 16 VGPR), bias word. `v_max3_f32` x31 (+1 `v_max_f32`), `v_permlane32_swap_b32` partner max, rollback compare
   inputs (`v_cmp_lt_f32 c_thr, mt`), bias chain (`v_cvt_f32_f16` x2, `v_add/mul/sub_f32`, `v_cvt_f16_f32` x2, `v_cndmask` x2). Outputs: `grow` (VCC / SGPR pair), delta, new bias word.
   Meanwhile (same block, independent): DMA issue for tile t+1 (4-5 `global_load_lds` + 8-10 VALU address ops + `s_mov_b32 m0`), `v_readfirstlane_b32`.
   Waits: none in the block (S(t) was written by MFMAs retired 20 states earlier: the entry wait clock must say so).
2. **DSL `begin_if(grow)`: rollback block** (unchanged): `v_exp_f32` + O rescale (4 x 8 `v_pk_mul_f32` on the O accumulators) + `la` rescale. Wave-level branch, so
   it stays outside the asm blocks; the loop-level contract must carry S(t), P(none yet), O, la, bias as live-in/out across it.
3. **Block B (`asm_conv_qk`)** - the interleave. For s in 0..3 (tile s of S(t)):
   a. `v_pk_add_f32` x8 (shift by delta, in place on S tile s), `v_pk_mul_f32` x8 (scale), `v_cvt_pknorm_u16_f32` x8 (4 results per 2-score pair, partial-write halves),
      `v_cvt_scalef32_pk_fp8_bf16` x4 -> 4 P words of tile s;
   b. as soon as step a of tile s is done (S tile s dead): issue **`bias MFMA` (`v_mfma_f32_32x32x16_f16`, ones x bias frag) into S tile s, then the two
      `v_mfma_scale_f32_32x32x64_f8f6f4` of QK(t+1) for tile s** (A = K frags of tile t+1 read by `ds_read_b128` x2 each + K scales `v241..244`, B = Q (resident), scale operands
      `sk0/sk1/sq0m/sq1m`), then continue with tile s+1's VALU (the MFMAs of tile s run in the XDL shadow of tile s+1's ~270/4 = ~65 VALU instructions).
   The K fragment `ds_read_b128` x16 + `ds_read_b32` x4 for QK(t+1) are issued ahead (one tile of lead), waits `s_waitcnt lgkmcnt(N)` with N = outstanding younger DS ops (exact counts
   from the builder clock, today 11/10/4/4/4/2/2/2/4 in the M phase).
   `s_waitcnt vmcnt(0)` for the staged V/K tile is placed here once, right before the first P word of tile 3 is needed for nothing (it only gates the barrier): after tile 2.
4. **`s_setprio(1)`, `s_waitcnt lgkmcnt(0)`, `s_barrier` (B1)** - publishes the DMA'd tile and aligns the two groups (today's barrier 1).
5. **Block C (`asm_pv`)** - PV of tile t: `ds_read_b64_tr_b8` x8 per e-half (V frags, ring slot of tile t), 8 `v_mfma_scale_f32_32x32x64_f8f6f4` (e0..e3 x halves, in place on
   O, scale operands c127/c123, B = P words) with the 4 in-place `v_smfmac_f32_32x32x32_fp8_fp8` row sums (`a_rs`, idx01/idx23) interleaved one per 2 PV MFMAs;
   waits `lgkmcnt(4/4/4/6/4/4/0)` exactly as emitted today; ends with the 20-state retire of the last accumulator before the next iteration's first reader (rescale / epilogue).
6. **`s_setprio(0)`, `s_waitcnt lgkmcnt(0)`, `s_barrier` (B2)** (ring recycle + complementary alignment, today's barrier 2).
Group B runs the identical code half a window later; the 2-slot ring, the K-scale DMA (`kddma`), the 2 barriers per tile and the pairing are unchanged.

## 4. Which ops are typed in `amdgpu-safeasm` today and which need new tokens
| needed | in safe-asm now? |
|---|---|
| `v_mfma_scale_f32_32x32x64_f8f6f4` (+ zero-C), `v_smfmac_f32_32x32x32_fp8_fp8` (in-place chain, 20-state `retire`) | **yes** (row-sum chain is byte-identical through `Stmt::VarAsm`) |
| `v_mfma_f32_32x32x16_f16` (bias MFMA, ones x bias frag) | no (new MFMA token, same `Mfma<'a>` borrow shape) |
| `v_add_f32`, `v_add_u32`, `v_mul_lo_u32`, `v_cmp_lt_i32`, `v_perm_b32`, `v_fma_f32`, `v_lshrrev_b32`, `global_store_dword` | yes |
| `v_max3_f32`, `v_max_f32`, `v_mul_f32`, `v_sub_f32`, `v_cmp_lt_f32`, `v_cndmask_b32` (e32 + e64), `v_cvt_f32_f16`, `v_cvt_f16_f32`, `v_exp_f32`, `v_mov_b32`, `v_and_b32`, `v_add_co_u32`/`v_addc_co_u32` | **new** (plain VALU templates; the constant-bus/`VgprOnly` rules apply) |
| `v_pk_add_f32`, `v_pk_mul_f32` (VOP3P, 64-bit operands: `Vgpr<2>` pairs), `v_pk_sub_u16` | **new** |
| `v_cvt_pknorm_u16_f32` with partial (16-bit-half) writes of a shared destination, `v_cvt_scalef32_pk_fp8_bf16` (scale operand + op_sel half writes) | **new** (the half-write destination needs a `lane_mut`-style borrow so that the two halves of the same dword are distinct `&mut` views) |
| `v_permlane32_swap_b32`, `v_readfirstlane_b32` (2 wait states after a VALU write: `Ready` token exists) | **new** (`readfirstlane` has the `Ready<'a,1>` rule) |
| `ds_read_b32`, `ds_read_b128`, `ds_read_b128` pairs (`b128x2` = two reads into 8 consecutive VGPR), `ds_read_b64_tr_b8` | **new** (+ `Pending` tokens on lgkmcnt with the *partial* count: `wait_lgkm(N)` must be derived from the builder clock) |
| `global_load_lds_dword(x4)` + `s_mov_b32 m0` (the DMA) and `s_waitcnt vmcnt(0)`, `s_nop` (hazard) | issue stays in the DSL (an `Op` outside the asm block) unless the block needs it interleaved: then **new** |
| `s_barrier`, `s_setprio` | **new** (block terminators: the asm block must end before a barrier; the contract carries the wait clock across it) |
| loop-level contract: live-in/out register sets + entry/exit wait clock for Block A/B/C and the DSL rollback in between | **new** (backend is adding it) |
The three blocks are straight-line, so only the `begin_if(grow)` and the loop back-edge stay in the DSL.

## 5. Expected win (cycle accounting) and what must be true
Inputs: per wave-iteration `_vs128` 1599 cycles (vendor 1156): XDL busy 1216 (1088), VALU 352 (251), WAIT_ANY 560 (413); kernel-occ's decomposition of the lost overlap is
issue-wait +190 and dependency-wait +97 per wave-iteration (287 cycles = 18% of 1599); the partner-free ablations show a solo group needs 4.1k cycles per tile against 5.34k
for the pair and 2.96k of pure compute (V 1.15k + M 1.5k + ~0.3k), the g2 experiment showed hiding QK under VALU is worth ~0.3-0.5k per tile when the windows stay balanced.
* **Upper bound** (all 287 lost-overlap cycles recovered): 5.34k -> ~4.4k per tile = **+21%** (387 -> ~470 TFLOP/s, 6.05 -> ~7.3 TFLOP/s/CU, vendor 8.4).
* **Credible range**: recover 35-60% of the lost overlap (0.35-0.55k per tile of the 5.34k): **+7% to +11%** per CU (387 -> 415-430 TFLOP/s at 64 blocks) - the gate is >= +5% (410).
* **Floor**: if the QK MFMAs issued from Block B fight the VALU of the partner group for the same XDL/VALU slots (the two groups already fill the SIMD), the gain can be ~0 (g2/g3 lost 6-9% when
  the merged window unbalanced the pairing).
What must be true: (1) the S / O / P / Q live ranges stay exactly as today (VGPR 249-252: no headroom for extra copies; QK(t+1) writes tile s only after tile s's last pknorm read);
(2) the window per group stays balanced: V-side VALU ~1.1k cycles, XDL ~1.2k cycles, so that the half-tile offset still lets one group's matrix work fill the other's VALU time (the target
moves ~0.5k of matrix work from the M window into the V window of the same group, i.e. both windows become ~1.35k and the barrier skew shrinks); (3) the waits come from the builder clock, not
from `lgkmcnt(0)` drains (every `lgkmcnt(0)` in the middle of a window costs ~100-200 cycles of the lost overlap); (4) the DMA issue stays inside Block A (not in the matrix-heavy window); (5) bit-identical
arithmetic: same instruction semantics and order of the floating-point operations as the DSL window (max3 chain order, pk_add/pk_mul/pknorm per word), so the md5 gate (faadf00c / c8b7a893 / d17778b0) is the
check at every step.

## 6. Scaffolding and integration plan
* `fmha_fwd_hd128_mxfp8_vs5` (this commit): same prologue / epilogue / DMA / barrier / rollback structure as `_vs128`, window still the DSL one (control point: identical asm, identical md5). Build with `amdc emit fmha_fwd_hd128_mxfp8_vs5 --backend own`.
* Step 1: replace Block C (PV + SMFMAC, the part that is already mostly typed) by `Op::Asm`; gate = md5 identical + timing >= `_vs128`.
* Step 2: Block A (max chain + decision) as asm; Step 3: Block B with the QK interleave (the actual win); each step gated by md5 + 15 shapes + determinism, the per-CU gain is measured at Step 3.
* Selection by `FMHA_OPT=gen5w` inside `emit_fmha_vs128_role` for `_vs5` only (so `_vs128`, the t128 / KV-split paths are untouched until the gate is passed).

## 7. Drain-closed boundary (explicit contract, for kernel-occ's validation) and the integration switch
**Rule**: the overlap lives WITHIN one tile window; every loop iteration boundary (the back edge, i.e. after barrier B2 and before Block A of the next tile) is drained:
* **No matrix instruction in flight and no unresolved DS load / VMEM load across the back edge.** Block C ends with the 20-state retire of the last PV accumulator and the last
  in-place SMFMAC (the builder's `retire(&mut seq)` padding; the `s_nop` padding sits in the shadow of the `s_waitcnt lgkmcnt(0)` + `s_barrier` that follow, so it costs ~0 on the critical path
  of the faster group), then `s_waitcnt lgkmcnt(0)`, then `s_barrier` (B2). The QK(t+1) MFMAs (the S(t+1) tiles) are issued in Block B, i.e. before barrier B1 and before the whole of Block C
  (8 PV MFMA + 4 SMFMAC = ~700 XDL cycles), so they are long retired (>> 20 states) at the back edge; their results are resolved values in VGPRs, not in-flight matrix results.
  (Today's DSL `_vs128` is NOT drain-closed in this sense: its QK(t+1) MFMAs are the last matrix instructions of the M phase and are in flight at barrier B2, covered by the MIR hazard pass;
  the gen5 order removes that.)
* The DMA of tile t+1 is issued in Block A and waited with `s_waitcnt vmcnt(0)` before barrier B1 (the 2-slot ring: the same as `_vs128`); the K-scale `global_load_lds_dword` is part of the same
  vmcnt group. At the back edge vmcnt = 0 and lgkmcnt = 0 (wait clock 0 for both counters).
* **The only cross-iteration state is VALUES in VGPRs and loop variables**: the O accumulators (4 x 16 fp32, already retired), the row sums `la` (Frag4, retired), the bias word `bias2` (loop var),
  the S(t+1) scores (4 x 16 fp32, produced by retired MFMAs), and the scalar loop counters; no P words (P(t) is produced in Block B and consumed in Block C of the SAME iteration), no K/V
  fragments, no K scales (re-read every tile), no pending Ready/Pending/Mfma token. Entry wait clock of every iteration = 0 for lgkmcnt and vmcnt and "all matrix results retired".
* Inside the window: Block B -> barrier B1 -> Block C is straight-line, only the rollback `begin_if(grow)` sits between Block A and Block B (it is a wave-level branch; its live-in/out set
  = S(t), O, la, bias word, delta/grow; at both of its edges the wait clock is 0 for lgkmcnt and the matrix state is "O accumulators retired" because the rescale `v_pk_mul_f32` reads O).
* Straight-line blocks end at every `s_barrier` and at the branch: the asm blocks never span a barrier (the loop contract carries the (drained) wait clock across it).
**Integration switch**: `_vs5` with `FMHA_OPT=winasm` (`vs5_winasm_requested()` in `kernels.rs`, called at the top of `emit_fmha_vs128_role`; today it stops with an explicit message because the ops do not exist
yet); without the flag `_vs5` is byte-identical to `_vs128` (control point). Swap order: Block C -> Block A -> Block B, each verified by md5 (same arithmetic) + 15 shapes + determinism.

## 8. Where the gain is expected, and how it is attributed
The win is expected from **Block B** (QK(t+1) issued under the V-side VALU, i.e. the in-wave overlap that the scheduler never produced). Block C (PV + SMFMAC) and Block A (max chain + decision + DMA issue)
swaps are *not* expected to move performance by themselves (they re-express instructions the scheduler already emits in about this order) and are gated separately - md5 (same arithmetic) + 15 shapes + determinism +
timing after EACH swap - so that any gain or loss is attributed to the block that caused it. The gen5 window is also not just a reordering: it changes the cross-barrier contract (section 7: `_vs128` has the QK(t+1) MFMAs
in flight at barrier B2, gen5 retires them inside the window), so a timing difference after Block B includes that effect. Reporting format for every step: asm md5 (emitted `.s`) AND the bf16 output-dump md5 (`REF_DUMP2`, 4194304
bytes for B1H8S2048) next to kernel-occ's frozen table.

## 9. Results so far (kernel-opt) and a correction to section 3 (Block B is not valid with the current ring)
**Measured, own backend, B1H8S2048 (`_vs128` 387.6-389.7 TFLOP/s in the same runs):**
| block | md5 (REF_DUMP2) | 15 shapes / det | TFLOP/s | verdict |
|---|---|---|---|---|
| C: PV + SMFMAC pinned (`winc`) | identical (6 shapes) | 15/15, 5/5 | 375.5 (1-step read lookahead) / 380 (2-step) | -3.3% / -2.5% (pinning the PV half alone removes the PV/QK interleave) |
| M: whole M phase pinned (`winm`, lookahead 1) | identical (4 shapes) | 15/15, parity 15/15, 5/5 x3 | 389.7-391.3 vs 389.1-389.7 | +0.5% (neutral base); B4H8S2048 / B1H8S8192 within +-0.8% |
| M with a 2-item read lookahead | identical | - | 375.5 | -3.1% (+8 VGPR of operand buffers: the list scheduler no longer fits and falls back to unscheduled code for the whole kernel) |
**Correction (found while writing Block B):** section 3 puts the 12 QK(t+1) MFMAs and the K-fragment reads of tile t+1 into the V window of tile t. With the `_vs128` 2-slot ring that is not
possible: the DMA of tile t+1 (K, K scales, V) is issued at the START of V(t) and only waited (`vmcnt(0)`) at the END of V(t), before barrier B1; K(t+1) is therefore not in LDS while the V-window VALU
runs, and the QK(t+1) reads/MFMAs can only issue after B1, i.e. in the M window (where Block M already has them, interleaved with PV). A valid Block B needs the K tile one tile EARLIER than today: K(t+1)
issued at V(t-1) (it can be: the K region of slot (t+1)%2 is free after QK(t-1), which would then have run in V(t-2)... i.e. the K ring runs one tile ahead of V), waited before V(t) starts, and K(t+2) issued at
V(t) start. That is exactly the merged-window structure that the earlier DSL experiments `_g2/_g3/_g3o` built (3-slot ring, QK(t+1) inside the V window, 1-2 barriers) and measured at -9% / -6% / -9%
(333.4-344.2 vs 367 TFLOP/s at that time), with the asm checked to contain the intended interleave (QK MFMAs between the conversion VALU, partial waits), so a hand-authored pinned version of the same order is
expected to reproduce that result rather than the +5..+11% of the optimistic range: what hurt those variants was the phase imbalance across the two wave groups (the V window grows by ~0.5k cycles of matrix work while the M window
shrinks, and the half-tile offset between the groups no longer lines the two windows up), not the instruction order inside the window. The conversion chain itself (per word `v_cvt_pknorm_u16_f32` x2 +
`v_cvt_scalef32_pk_fp8_bf16` lo/hi on a partial-write register; no pk_add/pk_mul: those 32+32 instructions belong to the rollback branch) is already available as typed ops.

**Verdict (closing the gen5 window track):** Block C is -2.5% (opt-in `winc`), Block M is neutral (+0.15..+0.5%, opt-in `winm`), both bit-identical (md5 `faadf00c95f4eaffff94ece7d1429b3d` on B1H8S2048, REF_DUMP2) with 15 shapes + determinism passing. Block B as specified in section 3 is not buildable under the current K ring (K(t+1) is not in LDS during V(t)); the only valid structure is the merged window, measured -6..-9% (g2/g3/g3o) with the phase-imbalance cause identified. The per-CU floor stands at ~6.11 TFLOP/s/CU (391 TFLOP/s at 64 blocks): the remaining gap to the vendor is not an instruction-authoring problem but the per-wave dependency chain and the phase balance of the two wave groups.