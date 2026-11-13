# Exact authored mode (design note)

**Goal:** 1:1 replication of a vendor kernel's asm. For a kernel or a region declared **exact**, the author fixes the complete instruction
order and every physical register, including the explicit `s_nop` / `s_waitcnt` they place. The back end does NOT schedule, allocate
registers, or insert waits / nops / copies. It only VERIFIES, and any violation is a compile error that names the instruction pair, the
register or the rule.

Item (2), the DSL register plan for the replica, is kernel-opt's; this note is item (1).

Context:
- the pass list: [pass-catalog.md](pass-catalog.md) section 9;
- authored regions and pins: [weaver-regs.md](weaver-regs.md);
- safe-asm windows: [safe-asm-design.md](safe-asm-design.md);
- the replica findings: [fmha-replica.md](fmha-replica.md) sections 4.4 and 4.8.

Opt-in only. A kernel without an exact declaration compiles exactly as today (md5 manifest unchanged).

## 1. Scope: region first, then kernel

**`declare_region(name, exact)`** (extending today's `authored`). Inside the region every instruction and register is the author's. Outside
it, the normal pipeline runs.
- Values crossing the boundary must be pinned: they are the region's hand-offs.
- The rest of the kernel is allocated around the pinned ranges. This uses the existing pins-first allocation (`regalloc` TwoPhase, pins
  first).
- The replica's matrix phase is the first target. A whole kernel is "one exact region around the body" plus an exact prologue and
  epilogue.

## 2. What the author must be able to express

| Need | Today | Exact mode |
|---|---|---|
| physical VGPR of a value | `pin_vgpr_at(r, base)` (HARD) | required for EVERY value defined in the region, temporaries included; a missing pin is an error |
| physical SGPR / AGPR | no SGPR pin; AGPR via `pin_agpr` (file, not number) | `pin_sgpr_at`, `pin_agpr_at` (new) |
| register reuse (P in S, the K / V rotation) | `reg_share` (hint), `reg_rotate` (branch) | the pins themselves: two values with the same pin and disjoint lives are the reuse, checked (section 4) |
| wait states | hazard passes insert `s_nop` | `b.s_nop(n)` statement (new; lowers to `SNop`, never merged or moved) |
| counter waits | `wait_vm(n)`, `wait_lds(n)` (explicit, kept) | the same, plus `b.waitcnt(vm, lgkm)` for the combined form the vendor uses; lowering emits no automatic waits |
| M0 | written by lowering / `m0-explicit` per DMA (or `m0reuse` groups) | `b.set_m0(v)` statement (new); an LDS-DMA in the region uses the current M0 and never writes it itself |
| instruction choice | DSL op -> lowering pattern (+ tokens) | every DSL op in the region must have a FIXED expansion (table below); patterns that pick forms by context (`sgpr`, `imm`, `addrfold`, `cmpfold`) are on inside exact regions, and the result is checked, not chosen |
| exact instruction text | safe-asm `Seq` windows (`Op::Asm`, pinned) | allowed inside exact regions; their own register / hazard rules are verified as today |
| block layout | structured `if` / `loop`; `jthread`, `coldlayout` transform the CFG | the CFG passes are off inside exact regions; the author writes the exits (multi-exit loops via `break_if`) and the layout is the statement order |
| VCC / SCC | implicit | implicit, verified (section 4 item 5) |

## 3. Lowering in exact mode: no hidden instructions

Every machine instruction gets an origin when lowering emits it: the statement index that produced it, plus `synthetic` when lowering
added it on its own. In an exact region a synthetic instruction is an error. The error names the statement and the instruction, and says
which DSL form avoids it.

| Lowering source of extra instructions | Exact-mode behaviour |
|---|---|
| constant materialization and its cache (`consts`, hoisted `v_mov` / `s_mov` of immediates) | a constant is an instruction at its own statement (no cache, no hoist), pinned like any value; inline / literal operands where the form allows, checked |
| SGPR -> VGPR copies (`VMovFromS`), readfirstlane for a uniform operand | allowed only where the author wrote the op; an operand class mismatch is an error ("operand 1 of stmt N needs a VGPR: the value is in s12") |
| `set_var` copies (loop variables), `coalesce` | a loop variable and the values written to it must share the pin, so no copy is needed; otherwise error |
| descriptor pseudos (`SBufferRsrc`: 4 SALU), `VPtrAdd` (2 VALU), other fixed multi-instruction expansions | allowed: the expansion is fixed, and its registers come from the pins; listed by `amdc exact-report` |
| M0 writes per DMA, `m0-reuse`, `m0-explicit`, `m0-pad` | off inside exact regions; M0 must be set by `set_m0`; a missing / wrong M0 at a DMA is an error |
| automatic waits (`MInst::auto`), `waitcnt-recompute` | none emitted; verified (section 4) |
| hazard `s_nop`s (`hazard-value`, post-RA hazards, `hz-fill`) | none inserted; verified |
| remat / LDS spill / retry ladders | off: an exact region that does not fit is an error, not a retry |
| scheduler, `bweave-order` merges, `ldshoist`, `slotfill`, `wsink`, `ldahead` | off inside exact regions (today's `authored` already turns the scheduler off per region) |

`amdc exact-report <kernel>` lists per statement the instructions it produced, synthetic ones marked. Authors use it while converting a
region; it is the step-1 deliverable that needs no hardware.

## 4. Verification list (all errors, none fixes)

The idea: run each fixing pass on a COPY of the MIR in "check mode". Any instruction it would insert, or any register it would change, is
a violation, reported with its position and the rule. This reuses today's passes, so their rules are checked, not re-implemented.

1. **Hazards:**
   - `hazard-value`, `hazard-inflight-waw`, `hazard-phys-war`, `hazard-valu-sgpr`, `hazard-soft-clause` and the M0 hazard
     (`m0-lds-dma`), all against the target hazard table (`amdgpu-target`);
   - error format: "stmt 812 v_mfma … -> stmt 815 v_add_f32 reads v[52:55]: needs 18 wait states (mfma-dst-write), has 11";
   - the test-time hazard oracle (`tools/hazard_oracle.sh`, LLVM's recognizer) runs on exact kernels like on every other.
2. **Waits:**
   - `verify_waits` over the given order (vm / lgkm scoreboard, operand waits incl. DMA operands, the ring-slot tokens of a plan with
     rings);
   - error: "stmt 640 ds_read_b128 v[20:23] overwrites a register the load at stmt 601 is still writing (vmcnt needs <= 3, last wait
     vmcnt(5))".
3. **Registers:**
   - every value in the region is pinned, and the class / width / alignment rules apply (even tuple alignment for MFMA operands, the ISA
     rule);
   - liveness: two values whose live ranges overlap may not overlap in registers. Error names both values, their defining statements and
     the register;
   - pins outside the file limits (VGPR count, AGPR offset, SGPR budget) are errors;
   - the launch budget is checked as today.
4. **Implicit effects:** each instruction's emitted text against its table row (`tests/emit_effects.rs` as a per-kernel check). Between a
   producer of SCC / VCC / M0 / EXEC and its consumer, no other writer of that state may sit. Error names the three instructions.
5. **Semantics:** `gpu_verify --strict` on the IR (LDS bounds, tile protocol, barriers, global regions, GV-PLAN-RING), as for every kernel.
6. **Fixed expansions:** emission of every MIR instruction succeeds, and the region's instruction count equals what `exact-report` listed
   (no pass added or dropped anything).

Mutation tests (one per rule): remove an authored `s_nop`, a `waitcnt`, a `set_m0`; give two live values one pin; put an SCC writer
between an `s_cmp` and its `s_cselect`. Each must fail with the named rule.

## 5. Steps

| Step | Content | Gate |
|---|---|---|
| E1 | instruction origins + `synthetic` flag in lowering; `amdc exact-report` | md5 identical (analysis only) |
| E2 | check-mode runs of the hazard / wait / M0 passes (copy + compare); the register legality check of a fully pinned region | md5 identical; mutation tests |
| E3 | DSL: `declare_region(.., exact)`, `pin_sgpr_at`, `pin_agpr_at`, `s_nop`, `waitcnt(vm, lgkm)`, `set_m0`; the exact pass body for marked regions (scheduler, auto waits, nops, M0 passes, CFG passes, ladders off; checks on) | md5 identical for every existing kernel; unit tests on a small exact kernel; hardware probe (a pinned window that runs) |
| E4 | the replica's matrix phase in exact mode with kernel-opt's register plan (2) | port_gate (correctness), then perf vs the vendor in-session |

## 5.1 Forms the replica needs (kernel-opt, from the vendor disassembly)

1. **Per-dword liveness inside tuples, in-place sub-register writes.** The vendor's P is written into S dword by dword while the rest of S is
   live (`v_cvt_pknorm_u16_f32 v52, v52, v53`, `v_cvt_scalef32_pk_fp8_bf16 v52, v52, s89 op_sel:[0,0,1]`).
   - The register check is per dword: a dword may be redefined after its last reader, while other dwords of the tuple are still live.
   - A 16-bit half write (`op_sel` dst hi) is a read-modify-write of its dword.
   - The scheduler's effects model already tracks dwords (`RegFx::SubWrite`, the precise segments of the two-phase allocator); the exact
     check uses the same model.
2. **Tuple assembly from pinned parts without copies.** `frag_cat` / pack of values pinned at consecutive registers lowers to nothing; a
   gap or a misaligned base is an error.
3. **Loop-variable pins.** `pin_vgpr_at(RegRef::Var)` exists; `pin_sgpr_at` accepts variables too (the vendor's s68 loop key with
   `s_addk_i32 s68, 0x80`, s89 / s90 / s92 softmax scale and threshold).
4. **VCC.** `break_if` / `begin_if_unlikely` on a compare lowers to exactly `v_cmp_*_e32` + `s_cbranch_vccnz` (no extra `s_and` / `s_cmp`);
   the author's `s_nop` between them is checked against the VCC distance.
5. **`s_setprio n`** as a statement.

Not needed for the replica: AGPRs, EXEC masking (EXEC stays full in the hot path).

## 5.2 Hazard rules the vendor's code undercuts (running list)

In exact mode a vendor distance below our table surfaces as a named violation. A rule is relaxed only with evidence: a hardware probe,
LLVM's hazard recognizer, or the ISA document. There are no silent allowances.

| Rule (target row) | Our distance | Vendor | Evidence | Status |
|---|---|---|---|---|
| `valu-vcc-valu` (VALU VCC write -> VALU VCC read) | 2 -> 1 | 1 | hardware probe correct at 0 / 1 / 2 (`tools/vcc_hazard_probe.sh`); LLVM asks 2 | relaxed to 1 (user decision) |
| FP4 / FP6 16x16x128 MFMA -> VALU / VMEM read | 8 / 20 | - | `tools/mfma_lat_probe.sh`: 7 / 1 measured | not relaxed (oracle allowance documents LLVM's) |

(More rows as exact-mode checks on the replica name them.)

## 6. Coordination with kernel-opt (item 2)

The register plan must give a pin to every value of the exact region. That includes temporaries: read results, scale words, P inside S,
the K / V rotation as explicit pins, and the loop-carried values.

My side: the exact region tells which values lack one ("stmt N defines %v.. without a pin"), so the plan can be completed iteratively
with `exact-report`.

## 7. As built

| Step | State | Where |
|---|---|---|
| E1 | live | `MInst::stmt`, `amdc exact-report`, `tests/exact_report.rs` |
| E2 | live | `exact::verify_final` (the wait and hazard passes run in check mode on a copy; every insertion is a named violation), `exact::check_registers` (per-16-bit-half clobber check across the CFG; hardware-initialised values defined at entry); `tests/exact_verify.rs`: the compiler's own output passes for every reachable kernel, plus mutation tests |
| E3 | live | `FnBuilder::exact()` / irv1 `exact`, `SNop`, `SetM0`, `PinS` (values and loop variables), `MovV` / `MovS`, `NegF32` (source-negate modifier), `MfmaBf16Wide`; `exact::finalize` |
| E3 hardware | PASS | `tools/exact_probe.sh`: a fully pinned kernel with in-place register reuse runs verbatim |
| E4 / E5 | in progress | exact straight-line sections in a structured shell (kernel-opt's lifter); E5 branch forms and scope variables (kernel-opt) |

The exact lowering rules:
- **Accumulator:** an MFMA reads C where it is (D and C may differ).
- **Constants:** a pinned constant is an authored move at its statement.
- **Scalar values:** a value is scalar exactly when it is pinned to an SGPR and provably wave-uniform; generic integer compares and selects take their SALU form.
- **M0:** a DMA forms no M0 source (the author's `set_m0` is the M0).
- **Free copies:** a copy between values pinned to the same register is no instruction (tuple assembly, views, bitcasts).
- **Hardware registers:** a hardware register read into a value pinned at that register is no instruction; v0 / s0..s7 may be reused once their values are dead.

Still open:
- the combined `s_waitcnt vmcnt lgkmcnt` form;
- 64-bit and EXEC moves;
- the buffer descriptor from a kernarg pointer-add chain (kernel-opt's item n);
- a check that an M0 write reaches every DMA.

## 8. E5: structured exact branch forms and scope variables (implemented, kernel-opt)

The vendor's control flow, lifted by `tools/isa_lift.py --structure`, has three shapes:
- ifs whose then-branch falls through;
- loops whose exits carry their own bodies (the drains);
- branches into out-of-line blocks that jump back (the rollback-overflow paths).

Registers merge at joins that are not loop headers. E5 expresses all of this without gotos. The statements are layout and scope only: the structure, and with it every analysis, stays the plain `if` / `loop`.

| Statement (`FnBuilder`) | Meaning | Lowering |
|---|---|---|
| `begin_if_at(cond, anchor)` (`Stmt::BranchPlace` + `BeginIf`) | the then-branch is laid out out of line at the layout anchor `anchor` | its blocks are tagged with the anchor; the condition's taken edge enters them; the end jumps to the join |
| `layout_anchor(name)` (`Stmt::LayoutAnchor`) | where the branches placed at `name` go | a block boundary, no instruction; pass `lower-place` moves the tagged blocks in front of it |
| `break_if(c)` inside `begin_if_at(c, ..)` | a loop exit with its own body | one jump to the loop exit, with no second conditional branch |
| `declare_vars(&[(ty, init)])` (`Stmt::DeclareVars`) | variables without a loop, assignable with `set_var` in the rest of the function | registers allocated and initialised like loop variables (shared helper `alloc_vars`) |

**Lowering and layout.**
- `crates/amdgpu-codegen/src/lower_cf.rs` (`lower-place`, a no-op without anchors) does the layout.
- After it, jumps to blocks that only jump on are retargeted, and blocks no edge reaches are dropped.
- So a loop exit reaches its join in one branch.
- The fall-through `s_branch` of the unscheduled emitter is the exact pipeline's to elide (`MFunction::elide_fallthrough`).

**Verification.**
- `verify` (IR): a placement must precede an `if` and name an existing anchor; an anchor is declared once; scope variables follow the loop-variable init / type rules.
- `gpu_verify`: scope variables are flow-insensitive. They are unbounded unless an `assume` restricts them, and each `set_var` is then checked against the assumption (GV-ASSUME). The variable-init, uniformity and taint scans treat them like loop variables.
- `gpu_protocol`: their initialisation is part of the taint / cone scans.

**Gates.**
- md5 manifest identical: 240 kernels, 0 differences. `cargo test` and `run_all` pass.
- `crates/amdgpu-kernels/tests/e5_branches.rs`:
  - the layout test: exits after the back edge, the cold path at its anchor, one conditional branch per placed if, no forwarding block;
  - the plain form for comparison;
  - strict `gpu_verify` 0 errors;
  - irv1 round trip;
  - mutation tests: placement not before an `if`, unknown anchor, anchor declared twice — each rejected with its message.
- Hardware probe `tools/e5_probe.sh`: the placed and the plain kernel, n = 1..8, every lane right (E5_PROBE_OK, node 53 GPU 0).
