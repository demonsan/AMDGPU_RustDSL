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
| host-built descriptor words | the host contract on a kernarg scalar (`FnBuilder::ka_one_of`) | `MCAST` |

## Rules

| rule id | layer | what it keeps from happening | mutation test | positive twin |
|---|---|---|---|---|
| `GV-1250-TDM-CLAUSE` | asm | a tensor operation inside an instruction clause | `mut_tdm_clause` | `pos_tdm_clause` |
| `GV-1250-SLEEP` | asm | an indefinite sleep (including the raw-word encoding): a wave that sleeps forever may never wake | `mut_sleep` | `pos_sleep` |
| `GV-1250-MCAST` | IR | a tensor operation whose multicast mask is not provably zero in a launch that declares no cluster of at least two workgroups | `mut_mcast` | `pos_mcast` |
| `GV-1250-BAR-ID` | IR + asm | a barrier id out of range, the cluster barrier (or the cluster-barrier form) outside a cluster launch, a named barrier beyond the allocated count in a signal / wait / init / join (an unallocated barrier is silently a no-op, so the synchronization is missing), the first-arriver form on the cluster barrier | `mut_bar_id`, `mut_bar_id_forms`, `mut_bar_id_asm` | `pos_bar_id`, `pos_bar_id_forms`, `pos_bar_id_asm` |
| `GV-1250-NBAR-COUNT` | IR | a named barrier whose member count is not a constant in the field's range (1..=63); completion is count-based, so the count is the number of signals per phase, joined or not (signal-only producers count) | `mut_nbar_count` | `pos_nbar_count` |
| `GV-1250-BAR-JOIN` | IR | a break in the named-barrier join discipline: a second join before the leave (one joined barrier at a time), a leave with nothing joined, a wait on a named barrier while not joined to it (any wave may signal any barrier), paths of a branch or a loop body that end with different join states | `mut_bar_join` | `pos_bar_join` |
| `GV-1250-BAR-INIT` | IR | a named barrier used before it is initialized, or used with no work-group barrier between its initialization and its first use (a wave that re-initializes after another wave signalled loses the count) | `mut_bar_init` | `pos_bar_init` |
| `GV-1250-BAR-WAIT-IMM` | asm | a barrier wait whose operand is not an inline constant | `mut_bar_wait_imm` | `pos_bar_wait_imm` |
| `GV-1250-TRAP-BAR` | IR + asm | the barriers reserved for the trap handler used by a kernel | `mut_trap_bar` | `pos_trap_bar` |
| `GV-1250-BAR-REACH` | IR | a barrier that not every wave of its group reaches: a barrier under a condition that differs between waves with unequal arrivals in the arms, a barrier in a loop whose exit differs between waves, a wait with no signal of that barrier | `mut_bar_reach` | `pos_bar_reach` |
| `GV-1250-TDM-DESC` | IR | a tensor descriptor whose compile-time-known fields are invalid for a kernel: the context-restore / store / volatile bits set, a type other than 2, a valid count above 1, a tile that does not fit in the declared LDS (words the host supplies are the host contract's business) | `mut_tdm_desc` | `pos_tdm_desc` |
| `GV-1250-TDM-DRAIN` | asm | a tensor operation that can issue with more than 10 tensor operations outstanding: a sound static bound over every path (an issue adds one, a wait for the tensor counter to drop to N caps the bound at N, joins take the maximum, loops iterate to a fixpoint; a loop that issues without draining is unbounded) | `mut_tdm_drain` | `pos_tdm_drain` |
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

These rules are staged after the typed tensor descriptor, the named-barrier operations and the async-LDS forms exist; each lands with its mutation test and positive twin.

| planned rule | what it will keep from happening |
|---|---|
| `GV-1250-TDM-DESC` | tensor descriptors with inconsistent paired fields, misaligned fields, a tile extent outside the LDS allocation or the tensor outside the declared buffer, a gather with unordered indices |
| `GV-1250-TDM-OUTSTANDING` | (the in-flight limit itself is `GV-1250-TDM-DRAIN` above) a wait that depends on an operation queued behind a stalled issue, and the async-copy counterpart |
| `GV-1250-TDM-RING` | reuse of an LDS ring slot before both the producer's completion and the consumers' release |
| `GV-1250-BLOCK-LOAD` | a block load into its own source registers; a transpose load with a partial exec mask |
| `GV-1250-SCALAR-ALIGN` | scalar memory data operands wider than 32 bits that are not aligned; a scaled matrix multiply in a clause that is too short or ends badly |

## Adding a rule

1. Add the rule to `RULES` (id, layer, one generic sentence) and implement it in `verify_ir` or `verify_asm`, reporting through `viol` so `with_rules_disabled` can switch it off in tests.
2. Add `mut_<stem>` and `pos_<stem>` to `hang_rules_1250.rs` and the line to `COVERAGE`; `every_rule_is_covered` enforces the pair.
3. Never run the faulty kernel on a GPU: the mutation test must stop at the compiler.
