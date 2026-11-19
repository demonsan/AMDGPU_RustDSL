# gfx1250 hang-safety rules (G5)

A kernel that hangs a GPU takes the device away from everybody who shares it, and a hung device may need an administrator. So the compiler treats every behaviour that can leave a wave, a work-group or the whole device
waiting forever as an **error**: the kernel is not emitted. There is no warning level and no environment switch.

The rules live in [`crates/amdgpu-codegen/src/gpu_verify_1250.rs`](../crates/amdgpu-codegen/src/gpu_verify_1250.rs) and run inside `gfx1250::compile_with` for both tails (own and LLVM):

- the **IR layer** (`verify_ir`) reads the kernel IR right after lowering and allocation: tensor descriptors built from constants or kernarg words, barrier operations (including init / join / leave and the cluster-barrier form) and the control flow around them, and the launch facts the kernel declares;
- the **asm layer** (`verify_asm`) reads the final instruction text the assembler will see, whatever produced it (the own emitter, the LLVM tail, an exact replay with raw words);
- `enforce` / `enforce_asm` turn any violation into a compile error naming the rule id (`GV-1250-*`).

A kernel the rules reject is never run on a GPU: the mutation tests below stop at the compiler, and a new probe for a hazard is written as a compile-time test first.

## Launch facts a kernel declares

| fact | how | used by |
|---|---|---|
| workgroups per cluster | `Function::kd` entry `cluster_dims` (accepted by the emitter, not a descriptor field) | `MCAST`, `BAR-ID` |
| allocated named barriers | `Function::kd` entry `named_barrier_count` | `BAR-ID` |
| signals per work-group of a barrier the verifier cannot count | `FnBuilder::barrier_signal_count(id, per_wg)` (a `g5:signals_per_wg:<id>` entry; trusted, listed by `trusted_items`) | `BAR-COUNT` |
| host-built descriptor words | the host contract on a kernarg scalar (`FnBuilder::ka_one_of`) | `MCAST` |
| tensor descriptors built by the host (evidence / probe kernels) | `FnBuilder::tdm_host_descriptor()` (`g5:tdm_host_descriptor`; trusted, listed by `trusted_items`) | `GV-TDM-PROV`, `GV-GLB-*` |
| the outermost loop is the tile loop of a persistent kernel | `FnBuilder::persistent_tile_loop()` (`g5:persistent_tile_loop`; CHECKED, not trusted: the tile body is simulated once and must end with every tensor operation drained and a barrier after its last LDS access) | tile protocol (`GV-TILE-UNSUPPORTED` when the body is not quiescent) |
| wave-uniform runtime ifs inside loops hold markers / waits | `FnBuilder::loop_ifs_explored()` (`g5:loop_ifs_explored`; trusted and listed: each occurrence in each simulated iteration is a decision, at most 6 per path, only the simulated iterations are covered) | tile protocol (`GV-TILE-UNSUPPORTED` without it) |

## Rules

| rule id | layer | what it keeps from happening | mutation test | positive twin |
|---|---|---|---|---|
| `GV-1250-TDM-CLAUSE` | asm | a tensor operation inside an instruction clause | `mut_tdm_clause` | `pos_tdm_clause` |
| `GV-1250-SLEEP` | asm | an indefinite sleep (including the raw-word encoding): a wave that sleeps forever may never wake | `mut_sleep` | `pos_sleep` |
| `GV-1250-MCAST` | IR | a tensor operation whose multicast mask is not provably zero in a launch that declares no cluster of at least two workgroups (provable by constants, by known-zero bits through clears / sets of bits above 15 / shifts / sums and loop-carried variables, or by the host contract) | `mut_mcast` | `pos_mcast` |
| `GV-1250-MCAST-REQ` | IR | a cluster multicast whose request is not provably the same in every workgroup its mask names: a multicast is a **rendezvous of identical requests** (one wave per workgroup issues the same request with the same mask; the data goes to every requesting workgroup once all have asked, or on timeout), so a descriptor word (global address, LDS address, tile / stride words; everything but the mask) that depends on the workgroup's cluster position along an axis the mask does not depend on makes the workgroups of the group ask different things and nothing is delivered (each waits for the timeout; found by a probe series where a per-workgroup-slice design delivered nothing to the peer). The dependence is tracked through the field extractions of TTMP6 (x = bits 3:0, y = bits 7:4; the cluster ids and shape fields are the same for the whole cluster) | the first MXFP4 design (a row mask with a per-x slice, a column mask with a per-y request) | a row mask with the row's own tile, a column mask with the column's own tile |
| `GV-1250-BAR-ID` | IR + asm | a barrier id out of range, the cluster barrier (or the cluster-barrier form) outside a cluster launch, a named barrier beyond the allocated count in a signal / wait / init / join (an unallocated barrier is silently a no-op, so the synchronization is missing), the first-arriver form on the cluster barrier | `mut_bar_id`, `mut_bar_id_forms`, `mut_bar_id_asm` | `pos_bar_id`, `pos_bar_id_forms`, `pos_bar_id_asm` |
| `GV-1250-NBAR-COUNT` | IR | a named barrier whose member count is not a constant in the field's range (1..=63); completion is count-based, so the count is the number of signals per phase, joined or not (signal-only producers count) | `mut_nbar_count` | `pos_nbar_count` |
| `GV-1250-BAR-JOIN` | IR | a break in the named-barrier join discipline: a second join before the leave (one joined barrier at a time), a leave with nothing joined, a wait on a named barrier while not joined to it (any wave may signal any barrier), paths of a branch or a loop body that end with different join states | `mut_bar_join` | `pos_bar_join` |
| `GV-1250-BAR-INIT` | IR | a named barrier used before it is initialized, or used with no work-group barrier between its initialization and its first use (a wave that re-initializes after another wave signalled loses the count) | `mut_bar_init` | `pos_bar_init` |
| `GV-1250-BAR-COUNT` | IR | the cluster barrier signalled by other than exactly one wave per work-group per wait, proved by **alternation along every path**: exactly one wave of a work-group can reach a signal of -3 (each wave's path decided by its own evaluable conditions: the wave id, TTMP8[29:25] = waveIDinGroup, variables copied once, wave-uniform unknowns taken both ways), and for that wave a dataflow over the control-flow graph (loop back edges included) tracks the signals since the last wait in {0, 1, 2+}: at every wait the state must be exactly 1 (kernel entry to the first wait, and between consecutive waits). Sound for a split barrier (signal early, wait late) in any region. Waves that wait in different loops need those loops to iterate alike: proved by a small scalar-evolution comparison (the counter's entry value as a symbolic base plus a constant, its constant-step top-level updates, one loop-invariant bound, and each exit's phase offset; two loops with the same first-wait / later-pass thresholds, step, bound and type wait equally often; an off-by-one, a different step or a bound the loop writes do not prove) and only otherwise by the named launch fact `barrier_wait_trips_match(-3)`. Only a path that branches on a condition that is neither evaluable nor wave-uniform needs the named launch fact `barrier_signal_count(id, per_wg)`, a **trusted item** (`gpu_verify_1250::trusted_items`), never an implicit pass | `mut_bar_count` | `pos_bar_count` |
| `GV-1250-BAR-WAIT-IMM` | asm | a barrier wait whose operand is not an inline constant | `mut_bar_wait_imm` | `pos_bar_wait_imm` |
| `GV-1250-TRAP-BAR` | IR + asm | the barriers reserved for the trap handler used by a kernel | `mut_trap_bar` | `pos_trap_bar` |
| `GV-1250-BAR-REACH` | IR | a barrier that not every wave of its group reaches: a barrier under a condition (a branch, a goto, a loop exit) that differs between waves with different barrier arrivals on its sides (path-sensitive over the control-flow graph, gotos included), and a wait on a barrier that nothing in the kernel signals (completion is count-based and any wave may signal, so the signals may come from other waves) | `mut_bar_reach`, `mut_bar_goto` | `pos_bar_reach`, `pos_bar_goto` |
| `GV-1250-TDM-DESC` | IR | a tensor descriptor whose compile-time-known fields are invalid for a kernel: the context-restore / store / volatile bits set, a type other than 2, a valid count above 1, a tile that does not fit in the declared LDS, or (computed addresses: slot x stride + wave x slice over every slot / wave) can lie outside the declared LDS or outside the ring the marker names (words the host supplies are the host contract's business) | `mut_tdm_desc` | `pos_tdm_desc` |
| `GV-1250-TDM-STORE-PAD` | IR | a tensor store from LDS whose descriptor enables LDS padding (or whose padding field is not a compile-time constant): the hardware reads a store's LDS source unpadded (MI450, 2026-11-18: an LDS-staged epilogue with 16-byte row padding stored rows 1.. from shifted addresses), so the staging of a tensor store is laid out unpadded | hard error | `mut_tdm_store_pad` / `pos_tdm_store_pad` in `crates/amdgpu-kernels/tests/hang_rules_1250.rs` |
| `GV-1250-TDM-DRAIN` | asm | a tensor operation that can issue with more than 10 tensor operations outstanding: a sound static bound over every path (an issue adds one, a wait for the tensor counter to drop to N caps the bound at N, joins take the maximum, loops iterate to a fixpoint; a loop that issues without draining is unbounded) | `mut_tdm_drain` | `pos_tdm_drain` **Also the end of the program**: `s_endpgm` reachable with a tensor operation (or an async copy into LDS, on its own counter) still outstanding on any path is a violation: the write would land in LDS the next workgroup of the CU may own (the last steady iteration of a pipeline stages a tile nothing waits for; the stage templates now drain after their last window, `PipeBody::finish`). |
| `GV-1250-DPP` | asm | a DPP modifier together with a literal operand or on a 64-bit integer operation | `mut_dpp` | `pos_dpp` |
| `GV-1250-PERMLANE-CMPX` | asm | a lane permute directly after a `v_cmpx` | `mut_permlane_cmpx` | `pos_permlane_cmpx` |
| `GV-1250-DVGPR` | asm | dynamic VGPR allocation (the feature does not exist on this target) | `mut_dvgpr` | `pos_dvgpr` |
| `GV-1250-PRIV` | asm | a write of privileged state from a kernel (halt, privileged registers) | `mut_priv` | `pos_priv` |
| `GV-1250-SCHED-MODE` | asm | selecting the full relaxed dependency-checking mode or the reserved value of the scheduling-mode register, or writing it with a value the verifier cannot see; the normal mode and the partial mode (the program owns two dependency classes) are accepted, because exact replays use the partial one | `mut_sched_mode` | `pos_sched_mode` |
| `GV-1250-SETREG-MSB` | asm | a mode-register write directly followed by a VGPR MSB write | `mut_setreg_msb` | `pos_setreg_msb` |

Tests are in [`crates/amdgpu-kernels/tests/hang_rules_1250.rs`](../crates/amdgpu-kernels/tests/hang_rules_1250.rs).

- **Mutation test** (`mut_<rule>`): the faulty kernel or instruction stream is rejected with the rule's id, and **accepted when that rule alone is switched off** (`with_rules_disabled`, available only to tests, thread-local). A test that does not depend on its rule cannot pass, so removing or weakening a rule turns its test red.
- **Positive twin** (`pos_<rule>`): the nearest legal variant is accepted, so a rule cannot hide behind a blanket rejection.
- `every_rule_is_covered` fails the build when a rule in `RULES` has no `mut_` / `pos_` pair; `the_probes_pass_the_rules` checks that every shipped gfx1250 probe passes all rules (IR and asm).

## Hazards inserted by the compiler vs rules checked here

Two layers deal with "the hardware must not be asked for this", and they must agree:

- **G2 hazards** ([`gfx1250-hazards.md`](gfx1250-hazards.md), `target/gfx1250/hazards.rs`) INSERT the waits / `v_nop`s the architecture needs, in kernels the compiler schedules. They can only help where the compiler owns the instruction order.
- **G5 rules** (this page) CHECK the final instruction stream, whoever produced it: the own tail, the LLVM tail, or an exact replay (an author-placed stream the compiler may not touch, so it inserts nothing).
  Where the two overlap (a lane permute after `v_cmpx`, a mode write before a VGPR MSB write, the tensor drain), G2's output always passes G5, and G5 is what stops an exact replay that lacks what G2 would have inserted.

The tensor drain is the instructive case. The public compiler's hazard pass for this target (the feature "needs to keep tensor count low") inserts `s_wait_tensorcnt 10` before any tensor operation that an earlier one can reach on some path without a
wait of at most 10 in between. The source states that purpose and gives no reason for the number and no consequence of a violation (stall, corruption or hang). What the insertion guarantees is a **count invariant**: at most 10 tensor operations are
outstanding when another one issues (so at most 11 in flight). The instruction pattern is only the compiler's cheap way to guarantee it: it inserts without counting, so with one earlier operation outstanding the inserted wait is satisfied at once.

`GV-1250-TDM-DRAIN` therefore checks the **invariant**, not the pattern: a sound static upper bound of the outstanding tensor operations at every tensor issue, path-sensitive over the control-flow graph (an issue adds one, a wait for the counter to drop to N
caps the bound at N, joins take the maximum, loops iterate to a fixpoint and saturate when they issue without draining). A tensor operation that can issue with a bound above 10 is an error. The invariant holds in everything the compiler's own tails emit
(their insertion maintains it) and is at least as strong as what the compiler enforces, so it also stops an exact replay that lacks the wait; it does not rely on any distance, timing or count of instructions between operations.

**Vendor code objects, "pattern differs, invariant holds".** Of the 91 shipping vendor code objects scanned (`tools/mi450/hangscan.sh`; `tools/mi450/drainhist.sh` lists every consecutive tensor-operation pair with its instruction distance and whether a wait of
at most 10 sits between), all satisfy the invariant. **12 of them (48 sites) differ from the compiler's pattern**: the persistent (`_ps`) MLA decode kernels with 96 or 128 query heads issue two tensor loads about 35 instructions apart with no tensor wait
between them, with at most two operations outstanding there. They are not deviations from the invariant, and an exact replay needs no wait at those sites; the other 79 files also satisfy the compiler's pattern as it stands. (A rule written from the
pattern, or from a distance window inferred from the corpus, would have been a timing assumption; the count form needs neither.)

## Planned (needs the forms G3 introduces)

These rules are staged after the forms they need exist; each lands with its mutation test and positive twin. (Implemented since this list was written: the tensor-descriptor checks are `GV-1250-TDM-DESC` / `GV-TDM-PROV` / `GV-GLB-*` above, and LDS ring-slot reuse before the producer's completion and the consumers' release is the tile protocol, `GV-TILE-*`.)

| planned rule | what it will keep from happening |
|---|---|
| `GV-1250-TDM-OUTSTANDING` | (the in-flight limit itself is `GV-1250-TDM-DRAIN` above) a wait that depends on an operation queued behind a stalled issue, and the async-copy counterpart |
| `GV-1250-BLOCK-LOAD` | a block load into its own source registers; a transpose load with a partial exec mask |
| `GV-1250-SCALAR-ALIGN` | scalar memory data operands wider than 32 bits that are not aligned; a scaled matrix multiply in a clause that is too short or ends badly |

## Adding a rule

1. Add the rule to `RULES` (id, layer, one generic sentence) and implement it in `verify_ir` or `verify_asm`, reporting through `viol` so `with_rules_disabled` can switch it off in tests.
2. Add `mut_<stem>` and `pos_<stem>` to `hang_rules_1250.rs` and the line to `COVERAGE`; `every_rule_is_covered` enforces the pair.
3. Never run the faulty kernel on a GPU: the mutation test must stop at the compiler.

## An incident: a device hang during kernel development, and what it taught

What ran. While a GEMM kernel family was being tuned, a sweep of kernel shapes ran on one idle GPU. Some shapes (a second K sub-step per ring slot) returned wrong results; a later shape, an 8-wave variant that had already run correctly once that day, then left the process blocked inside the driver (a register-write wait of the micro-engine scheduler) and the device unusable until an administrator reset it. All runs used hard polled timeouts, which cannot fire once a process is blocked in the driver.

What was found. The wrong results were a **code-generation bug**: the shared wait pass used the previous architecture's counter range (15) for the gfx1250 LDS counter (63), so with many LDS loads in flight the oldest ones were treated as complete and their WMMA readers received no `s_wait_dscnt`. Bisecting one work-group with a per-K-tile mask (`ONLY_K`) showed exactly the loop-body windows wrong, and the compile now carries the missing waits (an independent re-check of the final waits with the gfx1250 counters turns such a drop into a compile error). The LDS-above-64-KiB hypothesis was tested separately by micro-probes and is not the cause (an interim guard rule was added and retired).

What is not known. The hang itself was **not reproduced**: the same 8-wave kernel, rebuilt with the fixed waits, runs correctly from one work-group to the full grid. The most likely reading is a device-state effect of the preceding runs that consumed garbage operands, but this is not established, and the kernel is kept as a regression kernel (`gemm_g1250_tdm_stage_f4_8w_256x64`): it is run on one work-group first, watchdogged, whenever the back end's wait or ring handling changes. No data-dependent loop or branch exists in it, so wrong data alone cannot steer control flow.

What changed in the process. Sweeps of experimental kernels run only on GPUs and shapes cleared in advance, one work-group before the grid, and a kernel family whose results are wrong is not run again until the cause is found; the verifier gap (a wait model shared by the generator and its own checker) was closed by an independent check, and the rules in this document keep their mutation tests.

## Tensor bounds (global provenance and extent of every tensor descriptor)

A tensor operation reads (or writes) a global tile whose address sits in the descriptor's group 0. A descriptor built from zeroed or unrelated registers points at no buffer and can hang the device (an incident: a code-generation bug zeroed the pointer registers of a GEMM's descriptors). Every `tensor_load_to_lds` / `tensor_store_from_lds` of a kernel that is not an exact replay is therefore checked, as a **compile error**, by the global-bounds prover that also checks the plain loads and stores:

- **Provenance** (`GV-TDM-PROV`): the descriptor's address words are the low and high halves of one pointer value that derives from a kernarg pointer by additions; a constant, an unrelated register or the halves of two pointers are rejected (`GV-GLB-PROV` for a pointer without kernarg provenance).
- **Extent** (`GV-GLB-NOEXT`): that kernarg pointer has a declared extent in bytes, written in terms of the launch contract (kernarg ranges and definitions, grid dimensions, free symbols): `arg_tensor`.
- **Range** (`GV-GLB-OOB`): the tile's last byte, `offset + (rows - 1) x row stride + row bytes`, is proven inside the extent, and its first byte not before it, over every wave (cooperative and multicast slices), workgroup, cluster position and loop iteration the offset ranges over; the facts used are the contract's (and the counted trusted ones such as a cluster position lying inside the declared cluster). A tile the prover cannot bound is rejected.
- A kernel whose descriptors are **built by the host** (an evidence / probe kernel) names that as the launch fact `g5:tdm_host_descriptor` (`FnBuilder::tdm_host_descriptor`): a trusted item listed by `trusted_items`, never silent. Exact replays of vendor kernels get the findings as warnings.

Tests: `crates/amdgpu-kernels/tests/tdm_bounds.rs` (a constant base, unrelated registers, halves of different pointers, no declared extent, a tile past the extent, an unbounded offset: all rejected; a legal tile and a contract-bounded offset compile), and every shipped gfx1250 kernel passes the rule in `gfx1250_emit`.

**Paged / indirect buffers** (a base that comes from memory, like a KV cache addressed through a page table). Declare the pool with `FnBuilder::arg_tensor_paged(name, pages_sym, page_bytes, max_pages)`: its extent is `pages_sym x page_bytes` for a free symbol that is not a kernarg, with a declared range. State the host contract on the loaded (wave-uniform: `ReadFirstLane`) page index with `FnBuilder::page_index_below(idx, pages_sym)`: a trusted, counted and listed fact (the host's validate must check every table entry, like the slot mapping of the gfx950 paged kernels). The proof then covers `index x page_bytes + offset + tile` inside the pool, so a tile larger than a page, a missing contract or a missing extent are compile errors (tests in `tests/tdm_bounds.rs`).

**LDS bounds.** Every plain `ds_read` / `ds_write` and every async copy's LDS side must lie inside the declared LDS segment, and the declared + spill LDS inside the target's per-workgroup limit (gfx1250: 320 KiB): `GV-LDS-OOB`, a compile error on every gfx1250 compile (the same range prover as the global bounds; it tracks the alignment of an address, so `(4 t + 128) & 511` is at most 508). A masked address that can reach the end of the segment is rejected (tests in `tests/tdm_bounds.rs`).

**Async copies and prefetches.** (1) Two async global -> LDS copies in flight in one marker region (no completing `s_wait_asynccnt` between them; a partial wait retires all but the n most recent, the counter being in order) must write provably disjoint LDS bytes, because the async LDS writes may land in any order even between two loads of one wave: `GV-LDS-ASYNC-OVERLAP`. The proof is either apart intervals or the same linear address form `uniform + k tid + c` whose constants differ by a distance no two work-items of the workgroup can bridge. Copies of different tile events are covered by the ring-slot rules of the tile protocol. (2) `global_prefetch_b8` (`FnBuilder::prefetch_global`): a bad address is a translation fault reported to the host, so its address is proven inside a declared extent like a global load (`GV-GLB-*`). (3) `s_barrier_signal_isfirst` on a named barrier is rejected by `GV-1250-BAR-ID` (its SCC result is removed in future products). Tests: `tests/tdm_bounds.rs`, `mut_bar_id_isfirst_named` in `tests/hang_rules_1250.rs`.

**Register reads.** `GV-1250-GETREG-IDLE`: an `s_getreg_b32` of STATUS, STATE_PRIV, EXCP_FLAG_PRIV / _USER, SCRATCH_BASE (the gfx12 FLAT_SCR ids) or the XNACK state needs `s_wait_idle` (or `s_wait_alu 0`) directly before it, because the instruction does not wait for idle and would return a stale value. The ids are those of the public LLVM `Hwreg::ID_*` table (`llvm/lib/Target/AMDGPU/SIDefines.h`, names in `Utils/AMDGPUAsmUtils.cpp`): 2, 4, 17, 18, 20, 21, 33, 34. A read of MODE, a clock or the hardware ids needs no wait.

**Scalar loads in flight.** `GV-1250-SMEM-PENDING`: an SGPR written by an `s_load` / `s_buffer_load` may not be mentioned (read, or written again) by any instruction before an `s_wait_kmcnt 0` (or `s_wait_idle`) has completed the load on every path. SMEM returns out of order, so a counted `s_wait_kmcnt N` above 0 proves nothing. The tensor operations are the dangerous readers (their descriptor words are SGPR tuples: a descriptor read before its load landed points at no buffer). The IR layers prove each descriptor's provenance (`GV-TDM-PROV`, `GV-GLB-*`) and the lowering inserts the waits; this rule is the independent check on the final text, path-sensitive over the control-flow graph (joins take the union, loops iterate to a fixpoint), with mutation tests for a missing wait, a counted wait, a wait on one path only, a wait after the next iteration's use and a late-overwritten register.
