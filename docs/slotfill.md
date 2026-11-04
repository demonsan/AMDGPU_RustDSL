# Slot-filling scheduler (`slotfill`, sched/slotfill.rs)

"Issue one, subtract one" with per-MFMA budgets derived by the compiler instead of authored knobs (`dma_gap`, `spread_reloads`, ...). Opt-in sched token `slotfill` (Function::sched_opts, e.g. GemmConfig.sched = "rewait,slotfill"; or the experiment switch `AMDC_SLOTFILL=1` / `AMDC_SLOTFILL=<kernel name substring>`). Off = no change at all (all 101 `amdc list` kernels have identical asm md5 against Main 1c3c9cf; full `cargo test` 325 passed, 8 of them new in tests/slotfill.rs).

## Algorithm

Regions are the stretches of a block between hard fences (`s_barrier`, explicit waits, `s_setprio`, cache maintenance); blocks with explicit partial waits and blocks with fewer than 16 MFMAs are left alone.

1. The MFMA sequence of a region is the fixed skeleton (author order). Slot `k` = the gap in front of MFMA `k`, slot `M` = after the last one. One slot = the measured MFMA issue interval (32 cycles, `docs/mfma-ceiling.md`: 32.25 cycles on gfx950), `SLOT_BUDGET` = 28 cycles for everything else (model issue costs from `sched/model.rs`).
2. Every other instruction (floater) gets a window `[lo, dl]`: `lo` from its producers (an MFMA reader of the old value, producer latency in slots, at most `HOIST_SLOTS` = 40 slots above its source position: the register-pressure cap), `dl` = DEADLINE from its consumers (an MFMA consumer at slot k minus the producer latency in slots, a floater consumer's deadline, a DMA without consumer in the region = region end minus `DMA_SLOTS` = 24, other loads region end minus their latency).
3. Memory operations (DMA, ds_read, ds_write, global load/store) are the anchors. Each class is spread evenly over the span of its windows (k-th of n at `lo_min + (k + 1/2)(dl_max - lo_min + 1)/n`, clamped to its own window); a slot whose budget is used up passes the instruction to the next slot with room (later first, then earlier). Address chains / `s_mov`s join their anchor's slot just before it; everything else keeps its authored slot clamped to its window. A consistency pass makes producers precede consumers slot-wise; if no consistent assignment exists the region is left alone.
4. Verified by `verify::check_perm` (always when `verify` is on; every result is proven legal against the dependence graph, M0 / SCC / VCC are slots of the graph so they stay exclusive and paired; in-order vm/lgkm queues: DMAs and global accesses keep their relative order, the wait pass recomputes every count; the M0 reuse pass runs after scheduling and works across other instructions).

LEGALITY CONTRACT (opt-in): inside one barrier interval an LDS-DMA never writes bytes a `ds_read` of the same interval reads (the 2-slot / ring protocol of `tile::Pipeline`, proven on the IR by the protocol verifier). `order::set_relax_dma_lds_read` (thread-local, set only inside the pass) drops the DMA <-> ds_read ordering edges; DMA vs ds_write / global accesses / other DMAs stay ordered. The flag must be thread-local: a first global version raced with concurrent compiles (tests).

## Experiment 1: GEMM v4 with the hand placement knobs neutralised

`gemm_mxfp8_bs_v4_plain` = v4 with `defer_dma: false` (all 16 DMAs in a burst after the barrier) and `spread_reloads: false` (B reloads in bursts of halves), sched "rewait" (the Gemm profile's ds_read hoisting stays); `_plain_sf` adds `slotfill`; `_plain_hc` = plain + hoistchain; `_sf` = the hand-tuned v4 + slotfill. GPU 4, 4096x4096x8192, alternating rounds (each round = every kernel once, REF_ITERS=100), `tools/ab_gemm.sh`, GEMM_OK in every run:

| kernel | A/B 1 (7 rounds) median | A/B 2 (9 rounds) median |
|---|---|---|
| v4 (hand-tuned default) | 3001 | 2996 |
| v4_plain (no knobs) | 2604 | 2579 |
| v4_plain_hc (no knobs + hoistchain) | 2601 | - |
| **v4_plain_sf (no knobs + slotfill)** | **3018** | **3007** |
| v4_sf (hand knobs + slotfill) | 3017 | 3008 |

Without the knobs the kernel loses 13-14 %; slotfill brings it back and ends 0.4-0.6 % above the hand placement (also when it runs on top of the hand placement). `tools/gemm_gate.sh gemm_mxfp8_bs_v4_plain_sf`: BUILD / VERIFY strict (0 errors, 0 warnings, 0 unproven) / ACC 3 shapes / DET 3 identical md5 / PERF 7 interleaved processes: ours 3296 vs vendor 3298 TF/s = 0.999, GATE PASS. What the pass did in the loop: the 16-DMA burst became one DMA every 2-3 MFMAs over slots 1..31 (the derived equivalent of dma_gap) and the reloads stayed at their window starts (in-place reload after the last reader).

## Experiment 2: FMHA v3 (port_gate, `AMDC_SLOTFILL=fmha_fwd_hd128_mxfp8_v3`)

`_vs128`: asm unchanged (no region qualifies); `_t128` (vgpr 255 -> 245) and `_kvsplit` change. `tools/port_gate.sh fmha_fwd_hd128_mxfp8_v3` PASS in both runs (BUILD, VERIFY strict, ACC16, ACC64 x 4 paths, DET, STRESS), PERF neutral within the run-to-run noise:

| shape / kernel | base | slotfill |
|---|---|---|
| b1h8s2048 kvsplit | 685.6 | 683.1 |
| b2h8s2048 _t128 | 1058.0 | 1063.7 |
| b2h8s2048 _t128 layout median | 1062.8 | 1060.8 |
| b4h8s2048 / b1h8s8192 / percu64 (_vs128, unchanged asm) | 1425.9 / 1982.1 / 446.6 | 1433.1 / 1985.7 / 443.9 |

So slotfill is correct on FMHA v3 but does not help it (those kernels' placement is not MFMA-slot-limited the way the GEMM's is).

## v4 default (this branch)

`GemmConfig::v4()` = the hand-knob-free form (`defer_dma: false`, `spread_reloads: false`, sched "rewait,slotfill"); the previous default is `GemmConfig::v4_hand()` / `gemm_mxfp8_bs_v4_hand`. Every other `gemm_mxfp8_bs_*` variant is defined relative to `v4_hand()` and its asm is byte-identical to before (checked against a Main 1c3c9cf build: v3, v4_gap2, v4_pre0, v4_ed, v4_roles, v3_nodma identical; `v4_hand` = old v4 up to the kernel-name lines).

- Alternating A/B, 9 rounds, GPU 4, GEMM_OK 9/9 each: new v4 **3022.8** vs v4_hand **2999.3** TF/s (+0.8 %).
- `tools/gemm_gate.sh gemm_mxfp8_bs_v4`: GPU 4 PASS, ours 3290.6 vs vendor 3299.5 = 0.997 (verify strict 0 errors / 0 warnings / 0 unproven, 3 shapes GEMM_OK, DET md5 identical); GPU 5 PASS, ours 3215.5 vs vendor 3245.5 = 0.991.
- Full `cargo test`: 326 passed, 0 failed.

## Register-pressure model and DMA window

- Pressure: `peak_live` counts live vector dwords (VGPR + AGPR tuples at full width, same rules as `hoist.rs`) of the authored and the new order of a region (region live-out from block liveness). The new order may raise the authored peak by at most `PRESSURE_SLACK` = 8 dwords; when it would, the hoist cap is halved (40, 20, 10, 5, 2, 1, 0 slots) and the region re-planned; if even cap 0 exceeds the budget the region is left alone. On GEMM v4: authored 182 -> 182 dwords (budget 190) at the full 40-slot cap, so the model does not bind there (it is a safety net for other kernels). `AMDC_SF_PRESSURE_SLACK` overrides.
- DMA window: `DMA_LATENCY_CYCLES / SLOT_CYCLES` slots before the region end. Swept `AMDC_SF_DMA_SLOTS` on v4 (alternating, GEMM_OK all): 8:3020, 16:3022, 24:2997 (7 rounds); 12:3026, 16:3020, 20:3011, 24:3013, hand 3002 (9 rounds); 32:2969. Flat for 12..24, worse at 32; the constant is 512 cycles = 16 slots. It is a swept constant, not derived from a measured DMA latency (the model's DMA latency of 240 cycles is too small and the real one varies with load).

## Limits / next

Spreading is per class and budget-driven, not hazard-aware beyond the graph (the hazard pass inserts nops afterwards); no occupancy / register-pressure model beyond `HOIST_SLOTS`; DMA latency window `DMA_SLOTS` is a constant. Enabled for `gemm_mxfp8_bs_v4` only (FMHA stays off: neutral). Other kernels: `sched: "rewait,slotfill"` or `AMDC_SLOTFILL`.