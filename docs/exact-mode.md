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

| WAITS: LDS read consumed with no wait, older than two newer reads (`lgkmcnt(2)` needed; the vendor's A-loop bf16-init sequence, 6 sites) | counter-guaranteed wait required | none (relies on the reads completing within the issue time of a 32x32x64 MFMA + ~14 ALU + 2 newer reads) | `tools/timing_probe.sh` (generic pattern, not the vendor's exact code): 8 waves with LDS traffic. With the counter-guaranteed wait, 6 / 6 runs are exact. Without it, at a gap of 16 VALU instructions, 4 of 6 runs read stale data (80-384 of 512 threads), and gaps of 0 / 2 / 8 also fail | **refuted**: the replica adds the 6 waits ("1:1 except 6 added `s_waitcnt lgkmcnt(2)`, under full proof"). `TrustWait` stays available for a "1:1 with 6 trusted waits" variant; its reason must cite this probe as refuting |

| VALU writes a lane mask (VOP3 `v_cmp_*_e64 s[..]`, VOP2 `v_cmp_*_e32 vcc`) -> `v_cndmask_b32` reads it as the mask (`hazard-value` / `hazard-valu-sgpr`) | 2 | 0 or 1 (~1100 sites in the lifted kernel) | `tools/lanemask_hazard_probe.sh` (node 53 GPU 2): e64 into four SGPR pairs and e32 into vcc, read by cndmask e64 / e32, are correct at 0, 1 and 2 wait states (512 threads, 5 runs). LLVM asks 2 | relaxed to 1 (`valu-sgpr-mask`; lead decision: the margin is deliberate). The vendor's ~6 back-to-back sites are named deviations (`s_nop 0`) |
| VALU VGPR write -> `v_readfirstlane` read (`valu-readfirstlane`) | 2 | 1 | `tools/readfirstlane_hazard_probe.sh`: stale at 0 wait states in every configuration, correct at 1 and 2 (512 threads, 4 VGPRs, 3 writer kinds, 5 runs) | relaxed to 1 (W4: 144 kernels' md5 changed, perf_ab all families OK, port_gate PASS) |
| transcendental result -> plain VALU read (`trans-valu`; other readers keep `trans-result` = 5) | 5 | 1 / 2 | `tools/trans_hazard_probe.sh`: v_exp / v_rcp / v_log / v_sqrt read by v_mov / v_add_f32 / v_cndmask_e64: stale at 0, correct at 1-3 (512 threads, 5 runs); LLVM TransForwardingHazard: 1 | relaxed to 1 (same W4 gate) |
| SALU M0 write -> LDS-DMA (`m0-pad`) | 1 | 0 (56 sites) | `tools/m0_hazard_probe.sh`: inconclusive (failures at every distance, so the probe's LDS layout is at fault, not M0 timing) | not relaxed; the replica's 56 sites are named deviations (`s_nop 0`) |

The waits row is a finding about the vendor kernel itself: it relies on LDS read timing, and the probe shows that timing can fail under contention.

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
| E4 / E5 | live | the whole vendor kernel lifted into one structured exact IR (kernel-opt's lifter), E5 branch forms and scope variables |
| replica | verified | `gpu-verify --strict` proves the lifted kernel (section 9); same output as the vendor, same-run performance 0.998-1.001x |

The exact lowering rules:
- **Accumulator:** an MFMA reads C where it is (D and C may differ).
- **Constants:** a pinned constant is an authored move at its statement.
- **Scalar values:** a value is scalar exactly when it is pinned to an SGPR and provably wave-uniform; generic integer compares and selects take their SALU form.
- **M0:** a DMA forms no M0 source (the author's `set_m0` is the M0).
- **Free copies:** a copy between values pinned to the same register is no instruction (tuple assembly, views, bitcasts).
- **Hardware registers:** a hardware register read into a value pinned at that register is no instruction; v0 / s0..s7 may be reused once their values are dead.

Forms added for the whole-kernel lift: `WaitCnt`, `ExecSave` / `ExecAll` / `ExecSet { mask }` / `ExecRestore`, `TrustWait`, `Undef`,
`Permlane32SwapA` / `B`, `PtrLo` / `PtrHi` / `PtrCat` / `RsrcCat`, `LoadScalar`, `KernargSegment`, `AddK`, `EncodeE64`, Ptr / Rsrc
variables, lane masks pinned at an SGPR pair / VCC (PinS 106) / SCC (PinS 253), SADDR global loads, the `fp_mode` header. In exact mode the
lowering keeps the author's operand order and encodings, never removes or folds an authored instruction (no DCE of loads, no coalescing,
no retargeting, no address folding through a pinned value), lays branches out as the layout says (fall-through elided), and the waits are
checked on the physical registers.

Still open:
- a check that an M0 write reaches every DMA;

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

## 9. The vendor FMHA replica: proof, trust and deviations

The lifted vendor kernel (kernel-opt's `isa_lift.py --kernel --exact`, ~22k statements) compiles through the normal verified emit:
`gpu-verify --strict` proves every global, buffer and LDS access in bounds, every descriptor's provenance and the barrier balance of the
wave roles. Certificate: 1 trusted assume, 30 trusted contracts, 0 unproven.

The analysis forms this needed (all proofs, nothing trusted):
- **Ring counters** (`ring_facts`): a variable set only to constants or to `v + Select(Cmp(v, K), a, b)` is enumerated to its reachable
  values; with an expression start (a per-wave base) the start interval's orbits are enumerated. A shadow (a variable updated by the same
  step) is its init minus the ring's init plus the ring. A step computed ahead of its `set_var` is recognised by the ring's rule.
- **EXEC lanes**: an LDS-DMA writes only the lanes of the author's EXEC (`ExecSet 0xffffffff` = 32 lanes).
- **Pointer words**: descriptors and pointers assembled from words (`RsrcCat`, `PtrCat`) are recognised through PtrAdd chains and the
  masked or unmasked hi word (a canonical 48-bit address).
- **Branch facts** for the generic integer `Cmp` (either side constant, `NotI1` on the else arm).
- **Interval rules** for the vendor's integer ops (`v_mul_i32_i24` with symbolic bounds, mbcnt, add3, or); a ceil-division grid is
  `(P + r) / k` with a remainder symbol (exact kernels); a third non-negativity test in `amdgpu-sym` (bounded symbols written from their
  upper bound).
- **Barrier balance** over E5 loop exits with bodies (kernel-opt): each exit path is compared between the wave roles.

Trust (each counted in the certificate):
- 1 assume: the workgroup map `(query tile, head)` is a bijection of the workgroup `(x, y)` (the vendor's XCD remap computes it by an
  8-step restoring division the intervals do not prove);
- 30 contracts, the host / ABI facts of the replica's 656-byte kernarg layout: 7 argument extents, 9 kernarg ranges, 2 grid definitions,
  11 kernarg definitions, and `HK` in [1, 65536] (the vendor's 24-bit address multiplies are exact only below that).

Race freedom is proven too (kernel-opt's tile plan for the lift: K 2 x 16640 at 0, V 3 x 16640 at 33280 by ring offset, S 2 x 512 at
83200): every LDS access lies in its tile (1167 / 1167), the protocol simulation proves 120880 tile reads (period 3, no stale / race /
unpublished read), and all 18 global stores are disjoint across workgroups. The output and LSE stores are clipped by their descriptors'
`num_records` (the tail tile's rows >= S are dropped by the hardware): `records_cap` turns that clip into a row cap. The lift adds 1 contract
for this: the LSE head stride `ka464 = 4 ka96`. Analysis forms added for it: `slot_by_offset` tiles, ring periods for the periodicity gate,
branch facts that re-evaluate earlier pure values, exact-mode ghost values (marker-only values are no instruction), the ceil grid kept as a
symbol (`ka96 = 256 grid_x - r`), linear `v_mul_i32_i24`, and caps with symbolic coefficients.

Named deviations from the vendor's instruction stream (everything else is identical and in order):
- 12 `s_waitcnt lgkmcnt(2)`: the 6 LDS timing-reliant sites (two copies each), refuted by `tools/timing_probe.sh`;
- 56 `s_nop 0` between an M0 write and its LDS-DMA (`m0-pad`; no conclusive probe);
- 6 `s_nop 0` at back-to-back lane-mask write -> v_cndmask (`valu-sgpr-mask` kept at 1);
- 1 fall-through `s_branch` of the vendor elided by the layout.

Hardware: output bit-identical to the vendor (B1 / B4 S2048, B1 S8192), same-run performance 0.998-1.001x.
