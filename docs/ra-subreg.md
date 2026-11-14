# Sub-register allocation (per-dword / 16-bit-half liveness)

Item 3 of the default-path lowering work: register allocation that tracks liveness below the tuple, the enabler for vendor-style register
plans. Three stages, each opt-in and gated on its own (md5 manifests + added table, perf_ab all families, port_gate); a stage becomes a
default only on a measured win.

| Stage | What | State |
|---|---|---|
| S1 `radword` | the two-phase packing tracks liveness per tuple dword (`regalloc::precise_slot_segments`): a tuple's dead dwords hold other values while its live ones stay put | opt-in; measured neutral (forced on for every kernel: 22 kernels change, the scheduler's out-of-VGPR fallbacks disappear, perf_ab 9 reps fmha_* / mla 0.998..1.001, port_gate PASS) |
| S2 affinity coalescing | lowering records affinities (tuple part -> tuple slot, loop init -> loop variable, set value -> variable, MFMA C -> D); the scan places a value at its affinity slot when that slot is free over the value's lifetime, and the copy becomes a same-register copy, dropped after allocation | lowering-level parts landed as opt-in tokens: `mfmadc`, `loopreuse`, `dupfold` (each measured neutral, docs/pass-catalog.md 2.2) |
| S3 16-bit halves | a half-register class for 16-bit values (cvt_pk / d16 loads / bf16 scale words) allocated into lo / hi of one VGPR; only operand positions that can address a half take one (op_sel, d16_hi writes, SDWA reads) | not started |

Soundness net, every stage: `crates/amdc/tests/register_check.rs` runs `exact::check_registers` (no instruction writes a physical register
dword whose value is still read later on some path, per dword and 16-bit half, across the CFG) on the allocation of every reachable
kernel; the two-phase packing keeps its reaching-writer check; the default path re-checks its waits on the physical-register view.

## Where the remaining copies are (decision record)

Copy removal moved no time on the FMHA kernels (MFMA / memory bound): `mfmadc`, `loopreuse`, `dupfold`, `radword` all measured neutral
(docs/pass-catalog.md 2.2). The kernels where a VALU instruction could matter are the VALU / SALU-bound ones; their back-end copies by
class (`amdc exact-report`, Main after lowopt-dup):

| Kernel | insts | copies | where |
|---|---|---|---|
| mla_decode_bf16_p16 | 2071 | 148 | 138 loop-entry (`BeginLoop` init, once per loop entry), 8 permlane16/32 swap operands (reductions, once per block), 2 loop-variable sets |
| pa_decode16_bf16_hd128_p16_c | 1032 | 66 | 52 loop-entry, 8 permlane swap operands, 5 loop-variable sets, 1 atomic |
| topk_softmax_renorm_bf16_e256_k8 | 746 | 54 | 36 DPP old-value operands, 18 permlane16 swap operands (a membound kernel) |

Almost all of them run once per loop entry or once per block, not per loop iteration: removing them cannot move these kernels' time either.
Decision (lead, 2026-11-14): S2 allocator affinity and S3 16-bit halves are stopped; `radword` and `dupfold` stay opt-in as building blocks
for a future register plan that needs them (e.g. vendor-style packing in an exact or planned kernel). The candidates left if a VALU-bound
kernel ever shows copy pressure in its loop: DPP with `bound_ctrl` instead of the old-value copy, and the in-loop loop-variable sets.
