# Instruction scheduling (crates/amdgpu-compiler/src/sched/)

Branch `sched-xblock`. Default OFF; enable with `amdc emit --sched` or `AMDC_SCHED=1`
(`AMDC_SCHED_OPTS=rewait,xbar,noverify,stats,drain,limit=N,xbar=B,only=B1+B2`).
LLVM is not used for scheduling (its MachineScheduler/IGLP work per basic block); LLVM stays only a tail.

## Pipeline

`lower -> [sched::run] -> hazards (strict) -> mir::verify -> regalloc -> pad_physical_war -> tail (own | llvm)`.
With the flag off the pipeline is the old code path (default output unchanged: `fix_value_hazards` keeps its old behaviour; strict mode is a separate entry point).

`sched::run`: (1) strip `auto` waits; (2) per basic block: dependence graph, pressure-aware list scheduler (several window sizes, best estimate that stays within the pressure limit, else original order), independent verifier; (3) whole-function wait re-insertion; (4) wait verifier. If register allocation fails ("out of VGPRs") the pass is retried with lower pressure limits (-16/-32/-48), finally the unscheduled code.

Modules: `effects.rs` (reads/writes per dword slot of the canonical tuple, vcc/scc slots, memory descriptors, fence class), `order.rs` (memory/fence ordering rules shared by graph + verifier), `depgraph.rs`, `model.rs` (machine model table), `list.rs`, `verify.rs`, `waits.rs`, `liveness.rs`, `tests.rs`.

### Dependences
* Registers: RAW/WAR/WAW per dword slot; writes to disjoint dword ranges of one tuple are independent. Partial writers (`v_cvt_pk_fp8_f32` low/high) read their destination; `v_permlane32_swap` writes both operands; `VPtrAdd`/`v_add_co`/cmp write vcc, `VCndmask`/`addc` read it. vcc is an ordinary slot, so a compare and its cndmask are never separated by another vcc writer, but independent work may fill the gap (no bundling needed).
* Memory: LDS/global accesses conflict if one writes and they may alias; same base register value (versioned by reaching def) + constant offsets give exact byte ranges, otherwise may-alias. `s_barrier` and `s_setprio` are full fences for memory (and for ALU unless `xbar`); explicit `s_waitcnt` is one-way (earlier LDS/VMEM ops stay before it). Blocks with explicit partial waits (count>0) are not scheduled.
* The terminator's vcc read is part of the verified final state.

### Verifier (always on in tests and by default when the flag is on)
Value numbering over both orders: every read sees the same reaching definition, final definitions of all slots equal, plus pairwise memory/fence order. Independent of the graph builder.

### Wait re-insertion (own tail)
Scoreboard dataflow over the CFG (join = union, `after` = min). vmcnt queue (global loads/stores, in order), lgkmcnt queue (LDS in order, SMEM out of order => full drain and entries never retired by count). Loads stay in flight across `if` and loop edges (lowering used to drain at every edge: "rewait" alone gives 215.2 -> 218.5 TFLOP/s). **Bug found by hardware**: retiring SMEM entries after 15 younger lgkm ops (valid for in-order LDS only) produced stale kernarg registers and illegal memory accesses when 19+ scalar loads are hoisted together.

### Hazards found on the way (hardware / LLVM-oracle evidence)
Scheduled code is bit-exact through the LLVM tail but exposed gaps of our own tail. `hazards::fix_value_hazards_opts(strict)` adds: MFMA -> reader needs `latency` wait states *between* (existing code allowed one too few for the scaled MFMA; 19 is still stale, microbenchmark), `v_pk_add/mul_f32 -> v_exp/v_rcp` 1 wait state, vcc producer -> consumer 2 wait states, block-end padding for in-flight MFMA results; `pad_physical_war` (post-RA) pads writes to physical registers read by a running MFMA. Whether each of the last three is *required* on hardware is Experimental (the wrong results of a failing run were all explained by the SMEM-wait bug; the hazard additions are LLVM-oracle based).

## Machine model (model.rs; source tools/sched_mb/RESULTS.md)
Cycles per wave64 instruction, `s_memtime` (2.39 GHz). V = 3 reps within 5%, E = experimental.
| op | issue | note | grade |
|---|---|---|---|
| plain VALU (add/mul/mov/max3/pk_*/cvt_pk_bf16/cmp) | 4 | RAW latency hidden | V |
| v_cvt_pk_fp8_f32, v_permlane32_swap, v_exp/v_rcp | 8 | no separate trans pipe: exp + k adds = 8 + 4k | V |
| v_cndmask_b32 VOP2 (vcc), >=4 waves | ~19 | e64 form with SGPR mask stays 4 | V |
| 2nd wave on the SIMD | | doubles throughput only for plain fp32 add/mul/fma/mov | V |
| scaled MFMA 32x32x64 / 32x32x16 fp8 / 16x16x32 fp8 | 64 / 32 / 16 | dependent == independent; VALU issues freely underneath (~issue/4-1 per MFMA) | V |
| MFMA result readable (last D reg) | 20 / 11 / 7 wait states | no interlock: stale read | V |
| ds_read_b128 / b32 / tr_b8 / write_b128 | 8 / 4 / ~6 / ~26 | latency ~52 (write ~75) | E |
| global_load_dwordx4 | 14 | latency ~108 (L1) / ~230 (L2) | E |
| s_barrier | 4/8/12 (1/2-4/5-8 waves) | | V |
| s_setprio | 4 | | V |
Unknown: cache-bypass loads, SALU/VALU co-issue between waves beyond the above.

## Results (B=1,H=8,S=SK=2048, GPU 0, 30 iters, same session; vendor 518)
| variant | TFLOP/s | 15 shapes vs unscheduled build |
|---|---|---|
| unscheduled (base) | 215.2-216.3 | - |
| rewait only (cross-block scoreboard) | 218.5 | bit-exact |
| full in-block scheduler (own tail, strict hazards) | 215.6 | bit-exact |
| + ALU across barriers (`xbar`) | 215.4 | bit-exact |
| full, LLVM tail (drop-auto-waits) | 210.4 | bit-exact |
| xbar, LLVM tail | 214.8 | bit-exact |
The model estimates -25% cycles for the loop blocks (26046 -> 19481 per-block cycles summed), the hardware shows 0%. **Negative result**: in-block reordering does not help this kernel on hardware, so the loop is not bound by the dependence stalls the model thinks it removes.

Why (Experimental reading of the microbenchmarks): per iteration and wave the unmasked arm issues ~560 VALU instructions (128 `v_mul_f32` O rescale, 128 `v_exp`, 64 pk_add, adds/movs) = ~2.4-3.5k VALU cycles; two waves per SIMD share the VALU (only plain fp32 ops dual-issue), ~5-7k of the measured ~10.8k cycles/iteration are VALU issue, MFMA (1k per wave) already hides under it. The vendor kernel needs ~3.6k cycles/iteration with the same MFMA count because it issues far fewer VALU instructions (lazy rescale, MFMA row sums, unscaled PV). Reordering cannot remove VALU issue slots; software pipelining only helps if the VALU stream has slack, which the measured numbers suggest it barely has. The kernel-level lever is VALU instruction count (O rescale, exp throughput), not schedule order.

## Update 2: model validation, stage 2, stage 3 (branch `sched-xblock`)

Options: `AMDC_SCHED_OPTS=swp` (software pipelining), `taildup` (stage 2), `xbar`, `stats`, `nocfg`, `only=B1+B2`, `drain`, `rewait`, `limit=N`.
Pipeline: `cfg::cleanup` (remove the empty latch the lowering creates for each `break_if`, elide fall-through `s_branch`: a loop costs one taken branch instead of two; flag `MFunction::elide_fallthrough`, set only by the scheduler pipeline) -> strip auto waits -> dead `v_cndmask`/`v_cmp` exit scaffolding removal -> [taildup] -> [swp] -> per-block list scheduler (verified) -> waits -> strict hazards -> RA -> post-RA SrcC pad.

### (a) Why the model said -25% and the hardware 0 (FMHA)
The first model had one wave and no shared units. `sched/sim.rs` is now a multi-wave SIMD simulator (in-order issue, per-pipe wave gap, SIMD-shared unit occupancy: plain fp32 add/mul/fma/mov 2, other VALU 4, exp/cvt_fp8/permlane 8, MFMA 64/32/16; RAW latency; `s_waitcnt` counters; `s_nop`; barriers; fair arbitration) with measured loop-control costs (tools/sched_mb/RESULTS2.md, Verified at 1 wave): taken backward `s_cbranch` ~24, `s_branch` (even to the next label) ~18, not taken ~6, `v_cmp` -> `s_cbranch_vccnz` +16 (about 5 independent VALU hide it); 32 / 48 cycles loop overhead for SCC / VCC counters. Unknown/Experimental: with >=4 waves per CU some launches show +96 cycles per taken branch (bimodal; unmodelled); loop heads at byte offset 20/24/28 mod 32 cost +4/8/12 (alignment not done); VOP2 `v_cndmask` with vcc costs ~17 cycles at >=4 waves (e64 form with an SGPR-pair mask 3.75; emitter still uses VOP2).
`amdc simmodel <kernel> [--sched]` prints model cycles per iteration for 1 and 2 waves per SIMD; `tools/sched_model_vs_hw.sh` prints model vs hardware for the sched_* microkernels (cycles per original iteration, GPU 1, 4000 iterations, 64 threads = 1 wave, 512 threads = 2 waves per SIMD; 2.4 GHz assumed):

| kernel | build | model 1w | hw 1w | err | model 2w | hw 2w | err |
|---|---|---|---|---|---|---|---|
| valu_mul | plain / sched / swp | 152 / 106 / 91 | 159 / 111 / 93 | -4 / -5 / -2% | 170 / 118 / 97 | 257 / 113 / 95 | -34 / +4 / +2% |
| valu_max3 | plain / sched / swp | 152 / 106 / 91 | 179 / 111 / 99 | -15 / -5 / -8% | 232 / 182 / 161 | 226 / 184 / 165 | +3 / -1 / -2% |
| valu_exp | plain / sched / swp | 216 / 170 / 155 | 223 / 187 / 163 | -3 / -9 / -5% | 360 / 310 / 289 | 433 / 331 / 305 | -17 / -6 / -5% |
| mfma_chain | plain / sched / swp | 92 / 110 / 95 | 127 / 131 / 89 | -28 / -16 / +7% | 128 / 128 / 128 | 133 / 133 / 143 | -4 / -4 / -11% |
| mfma_valu16 | plain / sched / swp | 504 / 371 / 291 | 737 / 416 / 320 | -32 / -11 / -9% | 555 / 479 / 420 | 865 / 637 / 611 | -36 / -25 / -31% |
| mfma_valu4 | plain / sched / swp | 243 / 179 / 100 | 344 / 223 / 127 | -29 / -20 / -21% | 263 / 190 / 139 | 376 / 240 / 226 | -30 / -21 / -39% |

Agreement: VALU-only kernels within about 10% except the unscheduled build at 2 waves (the +96 branch slow mode). MFMA+VALU kernels: the model is optimistic by 10-25% at 1 wave and 20-40% at 2 waves; cause not isolated (candidates: MFMA/VALU interaction between the two waves of a SIMD, MFMA result latency beyond the wait-state count). Relative decisions (which schedule is better) were right in all 18 pairs at 1 wave; at 2 waves the model over-predicts the benefit of swp on the MFMA kernels. So the earlier FMHA mismatch is explained qualitatively: the FMHA loop is VALU-issue bound with two waves per SIMD, a regime where reordering a wave's stream cannot reduce the SIMD's VALU busy time.

### (b) Stage 2: tail duplication across if/else diamonds (`sched/taildup.rs`)
For a join block J reached only from the two arms (both `Goto J`) inside a loop, a prefix of J is copied to the end of both arms (each path executes exactly one copy; barrier counts per path unchanged); the prefix length (2..64 or all) is chosen by the simulator on the loop's hot path, accepted at >= 2% gain. Verified: GPU-free interpreter test (all prefix lengths, 10 trip counts) and hardware bit-exactness on `sched_diamond` (9 trip counts x 2 block sizes). Common-prefix hoisting and loop-boundary motion of multi-block loops are not implemented.

### (c) Stage 3: software pipelining (`sched/swp.rs`)
Single-block counted loops (`it+1 < n`, runtime n, do-while, m = max(1, n-it0)): the body is cut in program order into S0|S1; code = P: S0(0); K (2x unrolled, kt=(m-1)/2 trips): S1(i) S0(i+1) S1(i+1) S0(i+2) with the values crossing the cut renamed in the second half (tuple + alias vregs; modulo variable expansion by unrolling, no copies); E: S1 + one plain iteration if m-1 is odd. No speculation (nothing executes for a non-existent iteration), m=1,2 skip the kernel, loop-carried variables keep their registers. The cut is chosen by simulating candidate cuts (cheap scheduler, hot path only); pipelined only if the average over 1 and 2 waves improves by >= 3%. Verified: interpreter test over all cuts x trip counts 0..13; hardware: 5 microkernels x 15 trip counts (0..13, 100, 101) x {64, 512} threads, scheduled+swp build vs unscheduled build, 150 hash comparisons, 0 differences.

### (d) Microkernel results (hardware cycles per original iteration, 1 wave / 2 waves per SIMD)
| kernel | unscheduled | sched (in-block + cfg cleanup) | + swp / + taildup |
|---|---|---|---|
| mfma_valu4 (MFMA -> light VALU consumers) | 344 / 376 | 223 / 240 | swp **127 / 226** |
| mfma_valu16 | 737 / 865 | 416 / 637 | swp 320 / 611 |
| mfma_chain (in-place MFMA chain) | 127 / 133 | 131 / 133 | swp 89 / 143 |
| valu_mul | 159 / 257 | 111 / 113 | swp 93 / 95 |
| diamond (if/else with MFMA arms, independent VALU in the join) | 496 / 518 | 360 / 391 | taildup **283 / 407** |
Most of the gain over the unscheduled build at 1 wave is loop-control cleanup (branch elision, dead exit scaffolding, SrcC-only pad) and in-block scheduling; pipelining adds 1.76x on mfma_valu4 and 1.30x on mfma_valu16 at 1 wave, taildup 1.27x on the diamond. At 2 waves per SIMD the overlap gains mostly vanish (the second wave already fills the stalls): swp 240 -> 226 on valu4, taildup 391 -> 407 (slightly worse) on the diamond.

### FMHA (`fmha_fwd_hd128_mxfp8_t`, current Main defaults, B1H8S2048, GPU 1, 30 iters)
unscheduled 224.5-225.0 TFLOP/s; in-block scheduler 221.7; `swp,taildup` 221.2 (no FMHA loop is eligible for swp: multi-block with barriers; taildup finds no model gain on any of the 12 diamonds); bit-exact vs the unscheduled build on 14 of the 15 shapes. The 15th (B=2 H=8 S=700 SK=640 gqa=2) is not reproducible even for the unscheduled build (3 runs, 3 different outputs): a race in Main's kernel, reported to the lead. Conclusion (negative for FMHA): scheduling does not help a VALU-issue-bound two-waves-per-SIMD loop; it does help loops that run one wave per SIMD or have exposed MFMA/load latency (microkernels above).

## Not done
Pipelining of multi-block loop bodies (diamonds, barriers), common-prefix hoisting / speculation, loop-head 32-byte alignment, `v_cndmask_b32_e64` emission, modelling of the +96-cycle taken-branch slow mode, `s_setprio` ping-pong.

## `--sched=auto` (round 3)
`AMDC_SCHED=auto` / `--sched=auto`: model-gated scheduling, default scheduler still OFF. Per loop the multi-wave simulator decides (>= 3%); loops with `s_barrier` are not eligible (no cross-wave term in the model; FMHA stays unscheduled, as measured 221.7 scheduled vs 225.0 unscheduled). Hardware results and details: docs/llvm-tail-integration.md section 9, `tools/auto_vs_plain.sh`.