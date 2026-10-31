# Last-arriver (fused KV-split combine): memory-ordering primitives and the required sequence

## DSL surface (all additive; own backend and both LLVM dialects)
| DSL | MOp | assembly (gfx950) |
|---|---|---|
| `b.store_f32_wt(base, off, v)` / `b.store_f32x4_wt(base, off, frag4)` | `GlobalStoreF32Sc1` / `GlobalStoreF32x4Sc1` | `global_store_dword[x4] ... sc1` (device-scope write-through) |
| `b.load_f32_coherent(base, off)` / `b.load_b128_coherent(base, off)` | `GlobalLoadF32Sc1` / `GlobalLoadB128Sc1` | `global_load_dword[x4] ... sc1` (device scope: bypasses non-coherent levels) |
| `b.atomic_add_u32(base, off, v) -> u32` | `GlobalAtomicAddRet` | `global_atomic_add vOLD, vaddr, vdata, off sc0` (sc0 = return pre-op value, sc1=0 = device scope) |
| `b.fence_release()` / `b.fence_release_lite()` | `SWaitVm 0; SWaitLgkm 0; BufferWbl2; SWaitVm 0` / `SWaitVm 0; SWaitLgkm 0` | `s_waitcnt vmcnt(0)`, `s_waitcnt lgkmcnt(0)`, `buffer_wbl2 sc1`, `s_waitcnt vmcnt(0)` |
| `b.fence_acquire()` / `b.fence_acquire_lite()` | `SWaitVm 0; BufferInv` / `SWaitVm 0` | `s_waitcnt vmcnt(0)`, `buffer_inv sc1` |

Atomic semantics are PER LANE: every active lane adds ITS value at the address and receives its own old value. For one ticket per wave pass `1` in lane 0 and `0` in the other lanes (the other lanes' adds are no-ops) and `read_first_lane` lane 0's result; or count in units of 64 and use `old >= total-1` (probe_last_arriver does the latter: exactly one lane in the whole grid sees `nblocks*64-1`). The result is an ordinary VGPR (the own wait pass treats the atomic as a vmcnt queue entry whose destination is the old value, so reading it inserts the right `s_waitcnt vmcnt(n)`; stores take a queue slot, so partial counts stay exact).

## The required sequence (what the ISA says and what we emit)
Sources: kernel-wiki `mi350-isa-memory-ordering` (CDNA4 ISA 9.1.10, tables 49/50): load/store scope SC[1:0] 0=wave 1=group 2=device 3=system; **device scope on a multi-L2 part = coherent cache bypass**; for atomics SC0 = return the pre-op value, SC1 = 0 device / 1 system. VM_CNT for stores drops when the write reaches L2 (not a host-visibility proof); scope bits are not an acquire/release protocol. MI355X has one L2 per XCD, so plain stores/loads are NOT guaranteed coherent between workgroups on different XCDs.
Writer (every split):  partial stores `sc1` -> `fence_release()` [wait all stores; `buffer_wbl2 sc1`; wait] -> `atomic_add_u32` ticket.
Last arriver (`old == total-1`): `fence_acquire()` [`s_waitcnt vmcnt(0)` for the ticket; `buffer_inv sc1`] -> `*_coherent` loads of the partials.
This is exactly the LLVM AMDGPU gfx942 agent-scope release/acquire model (release: `buffer_wbl2 sc1; s_waitcnt vmcnt(0)`; acquire: `s_waitcnt vmcnt(0); buffer_inv sc1`). The scheduler treats `BufferWbl2`/`BufferInv` as `Fence::Barrier`: no memory operation can cross them, ALU may.
`_lite` variants drop the L2 write-back / invalidate; they are only sufficient if EVERY shared store uses `*_wt` and every shared load `*_coherent` (then the cache levels are bypassed by the access itself) - empirically 0 failures, but the architectural guarantee is the full sequence. Default to the full sequence unless the measured cost matters.

## Hardware evidence (MI355X, GPU 4; `tools/last_arriver_probe.sh`, kernels `probe_last_arriver*`)
N workgroups x 64 lanes write `(bid+1)*1000+tid+seed` (a new seed every run, so stale lines cannot match by accident) with the chosen store kind, take the ticket, and the last arriver sums all partials. Block counts 2..2432, 12 counts x 1000 runs = 12000 runs per kernel, result and counter checked every run:
| kernel | stores/loads | fences | failures |
|---|---|---|---|
| `probe_last_arriver` | sc1 / sc1 | full (wait, wbl2, wait / wait, inv) | 0 / 12000 |
| `probe_last_arriver_lite` | sc1 / sc1 | waits only | 0 / 12000 |
| `probe_last_arriver_nofence` | sc1 / sc1 | none | 0 / 12000 |
| `probe_last_arriver_plain_fenced` | plain / plain | full | 0 / 12000 |
| `probe_last_arriver_plain` | plain / plain | none | **1 / 12000** |
The last row shows the probe can detect a violation (a stale partial read on another XCD, about 1 in 10^4 runs); the others did not fail, i.e. both the write-through/coherent accesses and the L2 fences are individually sufficient on this part and the full sequence has both. A weaker variant passing a finite test is not a proof: keep the full sequence as the contract.

## Backends
All five new ops (plus `BufferWbl2` / `BufferInv`) have rows in BOTH `llvm20.tbl` and `llvm24.tbl` (cpol 16 = sc1; the returning atomic uses cpol 1 = sc0 and `syncscope("agent")` so LLVM does not add the system-scope `sc1`), so the LLVM tail emits the same instructions (tail_invariance covers `probe_last_arriver*`); `tools/last_arriver_probe.sh` runs with `BACKEND=llvm-tail` too.

## 8-wave (512-thread) probe: the real fused-combine shape (review round 2)

`s_barrier` does NOT drain VM_CNT, so the contract is: every wave that wrote a partial drains its OWN stores (`s_waitcnt vmcnt(0)`, `fence_release_lite`) BEFORE the block barrier; only after the barrier does thread 0 take the ticket (value 1, the other threads add 0), the ticket is broadcast through LDS, and in the last block EVERY wave does the acquire drain before its coherent loads. `fence_acquire` (full) is `s_waitcnt vmcnt(0); buffer_inv sc1; s_waitcnt vmcnt(0)` (the inv returns an ACK through VM_CNT). Static check for the lite contract: `asm_check::check_lite_contract` (every shared global access `sc1`, ticket `sc0` without `sc1`, no `buffer_wbl2/inv`, a store must be drained before the next `s_barrier` and before the atomic) - covered by tests/lite_contract.rs.
Probe: kernels `probe_la8*` (512 threads, per-element hash(i+seed) partials, new seed every run, block-dependent stagger loop, odd block counts 3..2432, counter on its own page, concurrent 256 MB device-to-device copy on a second stream, exact per-element verification by `probe_la_verify` on the device). Results (MI355X, GPU 4):
| variant | accesses | fences | result |
|---|---|---|---|
| `probe_la8_lite` | sc1/sc1 | per-wave drain before the barrier, acquire drain | **0 bad runs of 400000** |
| `probe_la8` (full) | sc1/sc1 | drain + `buffer_wbl2` / drain + `buffer_inv` + drain | 0 of 20000 (smoke; the long run was killed by an external process on the shared node, see below) |
| `probe_la8_plain` | plain/plain | none | **10966 bad runs of 20000 (55%)** |
The 8-wave probe is ~3 orders of magnitude more sensitive than the 1-wave one (plain/no-fence failed 1/12000 there). Honest limitation: the longer full / plain_fenced / nodrain runs (>= 1e5 each) were terminated repeatedly by an external SIGTERM on the shared node (other sessions' guard scripts run on the same account/GPU), so only the lite 4e5 and the 20000-run smoke numbers above are final; `tools/last_arriver8_probe.sh` reproduces any of them (`RUNS=... THRASH=1 STAGGER=4`).

## Final status (Main v2.2-round75-final)

Primitives validated as documented: **lite contract 0 failures / 400000 runs**, **full-fence 0 / 20000**
(smoke; long runs externally terminated), and the **plain no-fence probe fails 10966 / 20000 = 55%**, so the
probe is sensitive and the lite/full results are meaningful. The fused-combine kernel using these primitives
was measured **negative: best 36.6 us vs 26.8-27.0 us for the two-launch split+combine path**, so it stays
**opt-in** and is not used by the shipped `_kvsplit` + `combine8h` path. See `docs/current-status.md` for the
authoritative snapshot and `docs/vendor-spec.md` section 7.8 for the memory-order contract.
