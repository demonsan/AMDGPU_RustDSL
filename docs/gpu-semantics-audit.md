# gpu_verify v4 audit of record - four-axis protocol model

Auditor: `kernel-occ`. Branch: **`gpu-semantics-occ`** (this doc). Tested **`origin/Main` = `7b8c5e0`**
(v4 generalized protocol engine `gpu_protocol.rs`: arrival / completion / visibility / generation;
LDS instance + global regions + last-arriver pattern + counted trust ledger).
On the tested tree: **17/17 `gpu_protocol`**, **14/14 independent `gpu_protocol_audit`**,
**2/2 `contract_host_truth`**, and `--shipped --strict` is clean. Hardware: MI355X/gfx950, node 53,
GPU 6 (probes only).

## 1. Axis-separation attacks (independent `gpu_protocol_audit`)

| attack | expected | actual on 7b8c5e0 |
|---|---|---|
| (a) LDS barrier (arrival) but producer DS write **not completed** by `wait_lds(0)` | NO-COMPLETION | **REJECTED** `GV-TILE-STALE|GV-PROTO-NO-COMPLETION`: "the write ... is never completed (no wait_vm/wait_lds/fence_release between the write and the next barrier), so no barrier can publish it" |
| (b1) LDS producer wrote+waited, but **no barrier publication** | UNPUBLISHED | **REJECTED** `GV-PROTO-UNPUBLISHED`: "version 0 ... only published at phase never (no barrier after its completion)" |
| (b2) global wrote+waited but **no `fence_release`** | NOT-VISIBLE | **REJECTED** `GV-PROTO-NOT-VISIBLE`: "completed and a barrier followed, but the write was NOT released ... other waves may read stale lines" |
| (b3) global released but **no `fence_acquire`** before read | NOT-VISIBLE | **REJECTED** `GV-PROTO-NOT-VISIBLE`: "runs before the acquire ... may still hit stale cache lines (no acquire)" |
| (c) barrier present but consumer reads a **stale generation** | STALE-GEN | **REJECTED** `GV-PROTO-STALE-GEN`: expects version 1, latest published 0 |
| (d) visibility satisfied but **next generation written in the read window** (O3) | RACE | **REJECTED** `GV-PROTO-RACE`: "read of r[0] (version 0, phases 1..1) overlaps the write of version 1 ... no barrier separates them" |
| (e) roles derive **inconsistent generation** (producer version vs consumer version) | STALE-GEN | **REJECTED** both LDS and global `GV-PROTO-STALE-GEN` |

The axis separation holds: an arrival (barrier) never substitutes for completion; completion never
substitutes for publication; a published generation may still be stale if the consumer expects the
wrong generation; and a visible generation can still race with the next write.

## 2. Multi-producer / multi-consumer

| case | expected | actual |
|---|---|---|
| two producer roles fill the **same generation** (cooperative, each writes half) and two consumer roles read independently | allow | **clean**, `tile-reads: 20`, `region-cover: 4` |
| two producer roles write **different generations** of the same region in overlapping phases | RACE | **REJECTED** `GV-PROTO-RACE` (write-write, versions 0/1, phases 0..1 / 0..1) + `GV-PROTO-STALE-GEN` for both readers |
| readers in **all roles** (no same-role exemption; inherited from v16) | independent obligations | both role traces checked; same-generation cooperative case accepted, different-generation case rejected |
| fused KV-split combine (last-arriver) | pattern proven | shipped `probe_last_arriver`, `_lite`, `_plain_fenced` **clean**; negative controls `_nofence`, `_plain` rejected |
| missing release in fused pattern | NO-COMPLETION | **REJECTED** `GV-PROTO-NO-COMPLETION` |
| missing acquire after ticket | NOT-VISIBLE | **REJECTED** `GV-PROTO-NOT-VISIBLE` |
| ticket argument/region mismatch | ARRIVAL-MISMATCH | **REJECTED** `GV-PROTO-ARRIVAL-MISMATCH` |
| lite acquire + plain (non-sc1) loads | NOT-VISIBLE | **REJECTED** |
| generation not the block's split id | STALE-GEN | **REJECTED** `GV-PROTO-STALE-GEN` |

## 3. Trust ledger

Every escape is listed and counted by `Report::ledger()` and the certificate line. I built a
deliberately unsafe **global-region** kernel with `buf_trust("r", ...)` and **no completion, release,
acquire, or ticket**. It passes `--strict` (by design, the region is trusted), and the checker makes
the escape visible:

`certificate: 1 trusted assumes (0 assume_trusted, 0 tile_trust_periodic, 1 region_trust),
1 trusted contracts, 0 unproven`

with `[GV-ASSUME|GV-PROTO-TRUST] Info: buf_trust 'r': TRUSTED, the protocol of this region is not
checked (...)`. So a trust escape can hide a real protocol violation, but it is **not silent**: it
appears in `-v`, in the `GV-PROTO-TRUST` finding, and in the per-kernel/aggregate ledger. Same for
`assume_trusted`, `tile_trust_periodic`, aliases and host contracts. `--strict` still prints the
certificate line (Info/trust entries are not Errors).

## 4. Shipped certificate (`--shipped --strict`)

**rc=0, 0 errors 0 warnings on all 8 shipped kernels**, aggregate **44 trusted assumes
(44 assume_trusted, 0 tile_trust_periodic, 0 region_trust) / 216 trusted contracts / 0 unproven**.
Per-kernel: vs128 6/29/0, vs5 6/29/0, vs9 6/29/0, vs 4/29/0, t128 4/29/0, t 4/29/0, kvsplit 12/36/0,
combine8h 2/6/0. `tile-reads` vs128/vs5 **58880**, kvsplit **240640**; tile-cover 600/760. The v16
shared-tile visibility rule remains in force (barrier publication for every role, no same-role
exemption); v4 only generalizes the engine and adds global regions/last-arriver/ledger.

## 5. Host truth

`contract_host_truth.rs` on Main `7b8c5e0`: **2/2 pass** for B1H8S2048/B4H8S2048/B1H2S200; every
extent, `grid_is`, `ka_is`, `ka_range`, `sym_range` matches ref_host/attn_host; no overstatement.
Remaining contract gaps: `kvs_counters`@0x288 has no `arg_tensor`; split domain not constrained to
{2,4,8}.

## 6. Final soundness statement (generalized model, README-ready)

**"The compiler proves GPU memory protocols as four separate axes: arrival (barrier count), completion
(`wait_vm`/`wait_lds`/`fence_release`), visibility (LDS barrier; global release + barrier + acquire),
and generation (periodic slot/version), with multi-producer and multi-consumer obligations discharged
bidirectionally."** On the shipped FMHA set, `--shipped --strict` is **0 errors / 0 warnings / 0
unproven** under a countable trust budget of **44 trusted assumptions and 216 host contracts**;
cooperative same-generation fills by multiple waves are accepted, while missing completion, missing
publication, missing release/acquire, stale generations, next-generation read/write overlap,
different-generation write races, ticket/arrival mismatches and last-arriver fence omissions are all
rejected. Escapes (`assume_trusted`, `tile_trust_periodic`, `region_trust`/`buf_trust`, aliases, host
contracts) are listed, counted and printed; they are trust boundaries, not silent passes. **Scope
limits:** global regions are intra-workgroup plus the explicit last-arriver pattern (cross-kernel
split launches are out of scope); last-arriver checks "reads more than one slice", not
count==splits; only two roles and equal trip counts; byte-level partial tile overlap is conflated;
safe-asm interiors are not checked; i32 wrap is assumed away; interactive hardware visibility
(`buffer_wbl2`/`buffer_inv`) is supported by the `tests/gpu/` probes, not proved. Within these stated
limits, v4 generalizes the v1.6 LDS certificate to multi-producer/multi-consumer LDS and intra-WG
global protocols with an explicit, countable trust ledger.
