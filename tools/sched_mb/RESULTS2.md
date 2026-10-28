# gfx950 loop-control microbenchmarks (round 2)
Files: `gen2.py` (kernels, 123 incl. pads; writes mb2_{0,1,2}.s, tests2.json, groups2.json), `analyze2.py`, `run2_raw.txt` (raw, 4 reps x waves 1/4/8, rep 0 dropped),
`run2_analysis.txt` (all numbers), `host.cpp` reused unchanged. Node 53, GPU 1, ROCm 7.0.2. Same method as RESULTS.md (s_memtime, 2.39 GHz, 48-cycle timing overhead subtracted, 256-VGPR kernels, pad kernel first).
Loop kernels: 2000 iterations (500 for the 256-instr body), timed around the whole loop; iteration count verified in a register and stored: every loop kernel returned 0x7d0=2000 (0x1f4=500;
vccz-not-taken variants 0x7cf because the cmp precedes the count). All numbers are cycles/iteration per wave, body = N independent v_mul_f32 (4 cyc each; 16 -> 64).
Unrolled (u_*, t5_*) kernels run the block twice and time only the 2nd pass (warm I-cache; cold pass inflated per-instr cost ~1.5 cyc), minus a measured mark overhead (40-41 cyc).
Status: V = 3 reps within 5%. "E" = reps bimodal (see multi-wave note).
Deviation: the compiler's `v_add_u32 v22,v21,v20` is written in place (`v_add_u32 v21,v21,v20`, cmp reads v21) so the counter advances; cost identical (same VALU RAW chain).

## 1-3. Loop control cost, 1 wave (16-mul body = 64 cyc), V
| shape (per iteration) | cyc/iter | overhead vs 64 | notes |
|---|---|---|---|
| s_add_u32; s_cmp_lt_u32; s_cbranch_scc1 (taken back) | 96.0 | **32** | 8 issue (add,cmp) + ~24 taken-branch |
| v_add_u32; v_cmp_lt_i32; s_cbranch_vccnz (taken back), 0 VALU between | 112.0 | **48** | = SALU loop +16 (v_cmp->vcc->branch latency) |
| same, k=1/2/4 independent VALU between cmp and branch | 112/112/112 | 48 | extra VALU free |
| same, k=8 | 124 | 60 | => ~5 VALU hide the cmp->branch dependency (~20 cyc); each VALU beyond costs 4 |
| vccnz loop + extra forward `s_branch NEXT` | 112 (salu loop+fwd) | +16 | loop version |
| `s_cbranch_vccz EXIT`(not taken); `s_branch LOOP`(taken back) | 116 | 52 | NT cond branch costs only ~+4 after the cmp stall |
| compiler shape: vccz EXIT(NT); s_branch NEXT; NEXT: s_branch LOOP | 136 | **72** | forward-to-next s_branch +20 |
| salu loop + 2 forward s_branch to next | 132 | 68 | 1st fwd +16, 2nd +20 |
| unrolled (no loop), warm: 16 mul = 65.0; +s_add,s_cmp 74.8; + NT scc0 branch 86.1 (+11.3); + fwd s_branch 88.3 (+23 vs 65); + 2 fwd 110.5 (+22 each); + NT vccz 76.0 (+11) | | | V |
Per-instruction summary: taken backward branch ~20-24 cyc; forward `s_branch` to next label ~16-23 cyc each (fall-through = 0); not-taken s_cbranch ~4-11 cyc; cmp->vccnz dependency ~16-20 cyc (hidden by ~5 independent VALU).

## 4. Body size / alignment (1 wave, V)
* Overhead independent of body size: SALU loop 64/96/288/1056 for 8/16/64/256 muls -> overhead 32 always; VCC loop 80/112/304/1072 -> 48 always.
* `.p2align` 4/5/6/8 on the loop head: no effect (<0.5 cyc). Instruction-cache-line crossing of the body (256 instr = 1 KB) irrelevant.
* Byte offset of loop head mod 32 (sweep by s_nop shifts, body 8/16/64): offsets 0,4,8,12,16 (and 32..48) -> +0; **offset 20/24/28 (mod 32) -> +4/+8/+12 cycles per iteration** (same for all body sizes; vcc loop s5 +4, s15 +12). So a loop head in the last 12 bytes of a 32 B window costs up to +12; align loop heads to 32 B (or any of the first 20 bytes of the 32B window).

## Multi-wave behaviour (1 / 4 / 8 waves)
* 1 wave: perfectly deterministic as above.
* 4 waves (1/SIMD) and 8 waves: numbers equal the 1-wave ones in the "fast mode", BUT per launch the whole workgroup sometimes runs in a **slow mode where every taken branch costs ~+96 cycles** (iteration 96->192, 112->208, 288->384, 1056->1152, also 64->160 for the 8-mul body; constant +95.7, independent of body size). Frequency over all loop kernels: 39% of 4-wave runs, 19% of 8-wave runs, 0% of 1-wave runs; for 8 waves sometimes only a subset of waves (e.g. one SIMD pair) is slow. Depends on launch/code placement, not on body size; `.p2align 8` loops were mostly fast. Status: E (bimodal; cause unknown, plausibly instruction-fetch/I-cache sharing refetch on taken branch). Model: overhead 32 (SALU) / 48 (VCC) typical, with a ~+96/taken-branch tail risk when >=4 waves per CU.

## 5. v_cmp -> v_cndmask (unrolled, 128 pairs, warm pass; per instruction, or per group)
| test | 1 wave | 4 waves | 8 waves | status |
|---|---|---|---|---|
| v_mul only (reference) | 3.75 | 3.74 | 3.77 | V |
| v_cmp_lt_i32 (e32 vcc / e64 sgpr) alone | 3.84 | 4.58 | 3.84-4.6 | V (w8 E) |
| v_cndmask_b32 VOP2 (vcc, no writer in loop) alone, back-to-back | 5.3 (E) | **16.9** | **17.1** | w4 E(14%), w8 V |
| v_cndmask_b32_e64 (sgpr pair) alone | 3.75 | 3.74 | 3.74 | V |
| pair cmp(vcc) -> cndmask VOP2 (per pair) | 8.65 | 11.5 | 8.2 | V |
| pair cmp -> cndmask_e64 sgpr (per pair) | 8.75 | 9.5 | 8.3 | V |
| cmp + 1 indep mul + cndmask (vcc / e64) per group | 11.8 / 13.5 (E) | 11.8 / 14.3 | 11.8 / 14.3 | V / E |
| cmp + 2 indep mul + cndmask (vcc / e64) | 16.5 / 15.75 | 17.5 / 15.75 | 17.3 / 15.75 | V |
| cndmask VOP2 + 2 mul (vcc const) | 12.5 | 13.8 | 13.3 | V (= e64 13.3) |
| s_mov vcc,s[30:31] ; cndmask VOP2 | 7.9 | 16.5 | 34 (E) | E/V |
| s_mov s[32:33] ; cndmask_e64 | 8.6 | 9.1 | 9.2 | V |
Findings: (a) a cmp->cndmask pair costs ~8-9 (just the two issue slots, dependency adds <=1 cyc) -- **no gap needed**, dependency latency is hidden at 4-cyc issue cadence. (b) The ~17-cycle cost earlier seen at >=4 waves belongs to the
VOP2 vcc **cndmask itself when cndmasks are issued back-to-back** (cn,cn spacing ~17), not to the v_cmp: it disappears when other VALU are between them (cn + 2 mul = 13.3 for 3 instrs ~ same as e64) and when a v_cmp (fresh VALU vcc write) precedes (pair 8-11). It also appears when vcc is written by SALU
(s_mov vcc -> cndmask 16.5 at 4 waves, ~34 at 8 waves, E). _e64 with SGPR-pair mask: 3.75/instr at all wave counts. => for the machine model: charge ~4 for e64 cndmask; VOP2-vcc cndmask ~4 alone at 1 wave, ~17 issue-spacing for consecutive VOP2 cndmasks with >=4 waves.
