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

## Not done
Stage 2 (superblock/tail duplication across the `if tail` diamond) and stage 3 (modulo scheduling) are not implemented: given the stage-1 result they would only add MFMA/VALU overlap that the hardware already gets from the second wave. `s_setprio` ping-pong is untested.