# Safe ASM: authoring gfx950 instruction sequences in safe Rust

`crates/amdgpu-safeasm` (`#![forbid(unsafe_code)]`, no dependencies beyond our own compiler crate) lets a hot instruction sequence be written
as ordinary Rust calls whose **types encode the ISA rules this project was bitten by**. It records MIR (`MInst`/`MOp`), so register
allocation, scheduling, hazard passes and emission stay the existing ones; `Seq::to_asm()` / `Display` goes through them for tests.

## 1. Why

The session's bugs were all "a consumer read something that was not what the author assumed": SGPR declaration, a dead load aliasing a live
SGPR, `v_readfirstlane` of a VGPR written by the previous VALU (2 wait states), the SMFMAC implicit accumulator, an SGPR source in
`v_cndmask_b32_e32` (vcc uses the gfx9 constant bus), a re-issued compare reading a mutated register. Each was found late (hardware run,
assembler error, or a plausible-but-wrong result). The goal: make the *shape* of those bugs unrepresentable, and where that is impossible,
make the sequence builder do the exact arithmetic so it cannot be forgotten.

## 2. The rules and how they are encoded

| ISA rule | Encoding | Failure mode |
|---|---|---|
| Register widths (V1/V2/V4/V8/V16, S1/S2) | `Vgpr<const N>` / `Sgpr<const N>`; only the ISA's widths can be created (`Wd<N>: VectorWidth`, sealed) | `Wd<3>: VectorWidth is not satisfied` |
| A fragment operand has a fixed width (e.g. MFMA A/B = 8 dwords) | const generic in the signature: `Ready<'a, 8>` | `expected Ready<'_, 8>, found Ready<'_, 4>` |
| Lane/sub-register views | `lane::<I>()`, `lane_mut`, `half8::<HI>()` keep the parent borrowed; bounds checked by an associated const | `lane index out of range for this register tuple` |
| e32 forms: src0 any operand, src1 VGPR only | `A: Operand`, `B: VgprOnly` (sealed traits) | `&Sgpr<1>: VgprOnly is not satisfied` |
| gfx9 constant bus: ONE scalar/literal source per VALU instruction, no literal in VOP3 | `Operand::BUS` / `VOP3_OK` summed in an associated `const` assertion | `error[E0080] ... gfx9 VOP3: at most one SGPR/literal source` |
| `v_cndmask_b32_e32` reads vcc through the bus: no SGPR source | data operands are `VgprOnly` | `&Sgpr<1>: VgprOnly is not satisfied` |
| `v_cndmask_b32_e64` mask is an SGPR pair (uses the bus): data operands bus-free | `Bus2` assertion | `... reads its SGPR-pair mask through the constant bus` |
| Inline constants -16..=64 | `Imm<const K: i32>` with a const assertion; `Lit` is a separate type allowed only in e32 src0 | `inline constants are -16..=64` |
| One `vcc`; compare then consume | `Vcc<Free>` / `Vcc<Live>` type-state tokens (affine values) | `expected Vcc<Free>, found Vcc<Live>` |
| VALU write -> matrix operand / `v_readfirstlane` needs 2 wait states | those instructions take `Ready<'a, N>`; the only way to get one is `Seq::ready(&reg)`, which pads *exactly* the missing wait states from the builder's clock | `expected Ready<'_, 8>, found &Vgpr<8>` |
| MFMA/SMFMAC result needs 20 wait states; in-place same-opcode chain is interlocked | the instruction returns `Mfma<'a>`/`Smfmac<'a>` holding the `&mut` borrow of the accumulator and the borrows of every operand until `.retire(&mut seq)` (pads to 20 states); `.chain(..)` issues the next same-opcode in-place op with no wait | `cannot borrow o as immutable because it is also borrowed as mutable` |
| Do not overwrite a register an in-flight instruction still reads (WAR) | the same borrows: writing needs `&mut` | `cannot borrow a as mutable because it is also borrowed as immutable` |
| SMFMAC accumulator C is implicit | `acc: &mut Vgpr<4>` is both C and D; emitted with C as `srcs[3]` (the MIR rule) | n/a |
| Users cannot add their own operand kinds | `Sealed` supertrait, not nameable outside the crate | `the trait bound Mine: Sealed is not satisfied` |

The type system alone cannot count instructions, so hazards that are *distances* (2 / 20 wait states) use a two-layer design: the type
forces you through `ready()` / `retire()` (you cannot forget), and the builder computes the exact padding from how many instructions were
already issued in between (so independent work placed before the `ready()` is used instead of being wasted on `s_nop`).

## 3. Example (the real port: `examples::pv_rowsum_phase`)

```rust
let mut o = s.vgpr::<16>();
let (rpa, rv0, rsa, rsb) = (s.ready(&pa), s.ready(&v0), s.ready(&sa), s.ready(&sb));
let t = s.v_mfma_scale_f32_32x32x64_f8f6f4_zero(&mut o, rpa, rv0, rsa, rsb);
let (rpb, rv1) = (s.ready(&pb), s.ready(&v1));
let t = t.chain(s, rpb, rv1, rsa, rsb);          // in place, same opcode: back to back
t.retire(s);                                      // 20 wait states, nothing else may touch `o` before this
let mut r = s.zeroed::<4>();
// ... four chained v_smfmac_f32_16x16x128_fp8_fp8, then retire, then
let lane = s.lane::<4, 2>(&r);
s.v_add_f32(&mut l_new, &l_old, lane);
```
Emitted (own backend, after register allocation; `tests/pv_phase.rs::show_phase_asm`): operand setup, then
`v_mfma_scale_f32_32x32x64_f8f6f4 v[48:63], v[8:15], v[24:31], 0, v2, v3` / `... v[48:63], v[16:23], v[32:39], v[48:63], v2, v3`,
`s_nop 7; s_nop 7; s_nop 3` (the 20 states), the accumulator zeroing, `s_nop 1` (the 2 VALU->SMFMAC states, already partly covered by the
movs), four back-to-back `v_smfmac_f32_16x16x128_fp8_fp8 v[24:27], ...`, `s_nop 7; s_nop 7; s_nop 3`, `v_add_f32_e32 v1, v42, v26`.

## 4. What was verified

* `forbid(unsafe_code)` plus a test that no source line contains `unsafe` (`crate_forbids_unsafe_and_contains_none`).
* The emitted asm assembles with the real assembler (`pv_phase_assembles`).
* Hazard padding is exact: chained SMFMACs adjacent, >= 20 states before the first reader, >= 2 before the first SMFMAC (`waits_are_inserted_by_construction`).
* The matrix/VALU instruction sequence, with registers renumbered by first appearance, is IDENTICAL to what the existing kernel path
  (DSL `mfma_scale_fp8` / `smfmac_fp8_16x16x128` -> `lower::lower`) produces for the same phase (`matrix_sequence_equals_the_existing_kernel_path`).
* 15 compile-fail cases (`tests/ui/fail_*.rs`, run by `tests/compile_fail.rs`), each must fail with a diagnostic containing the text on its first
  line; `pass_sequence.rs` must compile. They cover: wrong fragment width, SGPR in a VGPR slot (cndmask and compare src1), reading an in-flight
  MFMA D, overwriting an operand of an in-flight MFMA, matrix operand / `v_readfirstlane` without `Ready`, a second compare while vcc is live, two SGPRs
  or a literal in VOP3, an SGPR data operand with an SGPR-pair mask, lane/immediate range, unsupported width, user-implemented operand kinds.
  (`trybuild` could not be used: its transitive dependencies need edition 2024, the pinned cargo is 1.83; the runner is a 60-line
  `rustc` driver against the built rlib with the same expectations.)
* `cargo miri` was not run: the crate is `forbid(unsafe_code)` with no unsafe dependency of its own, so miri adds nothing beyond std; no nightly toolchain is pinned.

## 5. What cannot be caught statically (and what to do about it)

* **Distances across control flow.** `Seq` is straight-line: a loop back edge or a branch joins two clocks. Contracts at block boundaries
  (what is in flight at entry/exit) are not yet modelled; today the hazard passes that run on MIR after the sequence (our `hazards.rs`, the LLVM
  tail) remain the safety net there.
* **LDS / VMEM completion (`lgkmcnt`, `vmcnt`).** Not modelled yet: `ds_read`/`global_load` would return `Pending` tokens whose `wait(n)` carries the scoreboard
  count, exactly like `Mfma::retire`; designed, not built.
* **Same-SGPR reuse.** gfx9 allows the same SGPR twice in one instruction; the const check conservatively counts two scalar operands as two.
* **Distinct physical registers.** MFMA operand/accumulator overlap is a register-allocation property (`regalloc::check_mfma_operand_aliasing`);
  the types guarantee distinct *virtual* registers because `&mut acc` cannot alias `&a`.
* **Scheduler interaction.** The inserted `s_nop` are MIR instructions; the list scheduler treats them as fences, so the padding it was given cannot be moved
  across. Running the sequence through the reordering scheduler may still add its own, never fewer, waits.
* **Values.** Nothing here proves arithmetic correctness (scale semantics, index patterns); it removes the *encoding and hazard* classes only.

## 6. Roadmap to authoring kernel code with it

1. A `Stmt::Asm(Seq)` bridge in the DSL so a safe sequence is spliced into a kernel with named virtual-register inputs/outputs.
2. `Pending` tokens for DS/VMEM loads (scoreboard counts) and `s_waitcnt` derived from them.
3. Block-boundary contracts (`Seq<Entry, Exit>`) for software-pipelined loops.
4. More opcode templates, generated from `backend/llvm24.tbl` plus a small per-opcode operand-class table (the legality rules live in one place).

## Adversarial review - what holds and what does not

Independent review (kernel-occ) of `crates/amdgpu-safeasm` @ Main `a50369d`. Reproducers live in
`scratch/safeasm-review/` (not a workspace member; `run_cases.sh` compiles each case with rustc against the
crate rlibs, `hw/run_hw.sh` runs raw-asm ISA probes on GPU 5). The crate's own suite passes (5 ui/queue
tests, 18 compile-fail cases, `pv_phase`).

### Coverage table: (a)-(l) from the review brief

| # | attack | result | evidence |
|---|---|---|---|
| a | read in-flight MFMA D, token alive | **caught** (E0502 cannot borrow acc as immutable while mutably borrowed) | `cases/fail_read_inflight_mfma_d.rs` |
| a' | read in-flight MFMA D after **dropping the token** (`let _ = t;` / `mem::forget`) and reading via `lane()` as a plain `Operand` | **BYPASS - compiles, 0 wait states, wrong data** | `cases/pass_read_drop.rs`: emitted `v_mfma...` immediately followed by `v_add_f32 v1, v32, v32`; `hw/run_hw.sh` `rawread_nopad` reads **0** where the correct value is **64** (`rawread_pad20`), i.e. a real hardware stale read |
| b | overwrite a register an in-flight MFMA reads, token alive | **caught** (E0502) | `cases/fail_write_operand_of_inflight_mfma.rs` |
| b' | same after dropping the token | **BYPASS - compiles, no padding** | `cases/pass_war_drop.rs` (and WAW variant `pass_waw_drop.rs`); raw-asm `war_nopad` did not change the result in this probe, so the hazard is ISA-illegal but not observed here |
| c | forget the 2-wait rule before `v_readfirstlane` / matrix operand | **caught by construction** (`Ready` is the only path; plain register is `E0308`) | `cases/fail_matrix_operand_without_ready.rs`; crate ui case |
| d | SMFMAC implicit C written by VALU with no padding | **holds - the builder pads automatically** (the accumulator is passed through `make_ready_pub`); emitted `s_nop 1` (2 states) right before the SMFMAC | `cases/pass_smfmac_implicit_c.rs` |
| e | SGPR in a VGPR-only slot / constant-bus violations (`v_cndmask_b32_e32` vcc, e64 mask, two SGPRs in VOP3) | **caught** (E0277 `VgprOnly`, E0080 constant-bus const assert) | `cases/fail_cndmask_e32_literal.rs`, `fail_two_sgprs_vop3.rs`; crate ui cases |
| f | two compares with a live vcc | **caught** (E0308 `Vcc<Free>` vs `Vcc<Live>`); a second `vcc()` after dropping the token compiles but **panics at run time** ("vcc already taken") - safe but a DoS, not a compile error | `cases/fail_second_compare_while_mask_live.rs`, `cases/pass_second_vcc.rs` |
| g | wrong-width fragment/operand | **caught** (`Wd<N>` sealed; `Operand` only for V1/Sub<1>/Sgpr<1>) | `cases/fail_wrong_operand_width.rs`; crate ui cases |
| h | literal in VOP3 / inline constant out of range | **caught** (E0277/E0080 with readable messages) | `cases/fail_literal_in_vop3.rs`, `fail_inline_constant_range.rs` |
| i | hazard across a Seq boundary | **not expressible in the safe API** (no splice/merge method; `Seq` owns its register table) - but the raw-MIR escape hatch below makes it possible | `cases/fail_cross_seq_splice.rs`, `pass_no_cross_boundary.rs` |
| j | loop-carried wait-distance violation | **not expressible** (no loop construct in `Seq`) - same escape hatch caveat | as (i) |
| k | DS/VMEM partial-wait misuse | **not expressible yet** (no memory ops in the crate); the Pending-token design is not implemented, so this rule is simply absent rather than enforced | as (i) |
| l | raw `MOp` escape hatch | **BYPASS - full**: `amdgpu-compiler` is a public dependency and `Seq::to_mfunction` returns a mutable `MFunction`, so a caller can push arbitrary `MInst` (any `MOp`) after the checked sequence and emit it; this also provides loops, memory ops and cross-boundary sequences | `cases/pass_escape_mfunction.rs` (raw `v_add_u32` injected and emitted) |

### Findings (ranked)

1. **F1 - dropping a matrix token releases its borrows without adding the 20 wait states.** `Mfma`/`Smfmac`
   are `#[must_use]` but Rust only warns; `let _ = t;` (or `mem::forget(t)`) ends the borrows, and any
   *VALU* read of the result (`Seq::lane`/`half8` -> `Operand`) needs no `Ready`, so the sequence compiles
   and emits with zero padding. Hardware probe: reading the D of a fresh scaled MFMA immediately gives
   **0 instead of 64** (`hw/rawread_nopad` vs `rawread_pad20`). This is exactly the bug class the crate
   claims to eliminate. **Fix:** share a drop flag (`Rc<Cell<bool>>`) with the token, set it in
   `impl Drop`, and make `to_asm`/`to_mfunction` return `Err("unretired matrix result")` when set; or make
   the token hold `&'a mut Seq` (linear session type) so no other sequence operation is possible until
   `retire`. The same drop hole gives WAR/WAW bypasses (F2).
2. **F2 - WAR/WAW on in-flight operands/accumulator via the same drop hole** (`pass_war_drop`,
   `pass_waw_drop`). A raw-asm WAR probe (`war_nopad`) did not change the result at this latency, so it is
   ISA-illegal but not demonstrated to corrupt on gfx950; still the same fix as F1.
3. **F3 - the raw-MIR escape hatch defeats every type rule** (`pass_escape_mfunction`): `MFunction` fields
   are public and `to_mfunction` hands one out, so a caller can append arbitrary `MInst { MOp, ... }`.
   This is arguably "the compiler crate is the raw layer", but the doc should say so explicitly, and the
   public API should not invite it: return an opaque wrapper from `to_mfunction`, or make
   `MFunction`/`Block`/`MInst` construction/fields non-public in `amdgpu-compiler`.
   Recommended strongest fix: a **hazard verifier** in `to_asm`/`to_mfunction` that re-walks the MIR with
   the same wait-state model and rejects the sequence if any rule is violated - it catches F1/F2 and any
   future API hole in one place, and is cheap (the crate already owns the clock).
4. **F4 - the token does not encode the opcode**, so `chain` can follow a `_zero`-variant MFMA with the
   *different* accumulate opcode back to back (`examples::pv_rowsum_phase` itself does this). The type
   comment says the interlock is same-opcode only. Raw-asm probe: different-opcode back-to-back gives the
   correct result (`zerochain_nopad == zerochain_pad20 == 128`), i.e. hardware tolerated it here, but the
   type-level invariant is still weaker than the documented rule. **Fix:** `Mfma<'a, const OP: Opcode>`
   (or two token types) and make `chain` generic over a single opcode.
5. **F5 - `VALU_TO_MATRIX_READ = 2` is empirically sufficient for the A/B operand path** (raw-asm
   `valu2mfma_pad1 == valu2mfma_pad7 == 64`), but it is only 2 clocks while the CDNA4 table (p74/PASS=4)
   suggests the required wait is larger; it held here, but the evidence should be pinned with a hardware
   regression probe (or raised to 8 if further probes show otherwise; it costs 6 states per matrix op).
6. **F6 - a second `vcc()` after a dropped `Vcc<Live>` panics** (`seq.rs:212`) instead of failing to
   compile; panic is safe but user-hostile. Return `Option`/`Result` or use a type-state method.
7. **F7 - PoC is sequence-level only:** the ported phase emits the *same matrix/VALU ops* in the same
   order as the kernel path (6 MFMA/SMFMAC + 1 add; assembly-checked after register renaming), but the
   safe asm contains **51 `s_nop` wait states vs 4** in the kernel path (it pads 20 states after the PV
   chain even though the row-sum SMFMAC does not read D, plus 20 after the row-sum chain). Both assemble;
   the safe sequence is therefore *more conservative*, not instruction-identical, and it cannot be run as
   an observable kernel because `Seq` has no loads/stores and the PoC operands are all zero. The "bit
   identical results" claim is true at the matrix-op level (same operands/opcodes) but is not testable
   end-to-end until memory ops exist.

### Residual risks

* `Safe` here means "cannot express the ISA-hazard classes through the checked builder"; it does not
  cover the compiler crate, which remains a fully safe Rust API that can emit illegal sequences.
* Value-level mistakes (using a register before writing it, wrong fragment *contents*, wrong scale
  operand) are not addressed; only shape/hazard classes are.
* Releasing a token is not linear (F1/F2). This is the single most important gap.
* The wait-state constants are only hardware-evidenced for the matrix-result rule (20, confirmed earlier);
  VALU->matrix (2) and vcc->cndmask (2) still rest on the ISA table, not on a regression probe.
* `Sealed`/`#[forbid(unsafe_code)]` hold: no `unsafe` in the crate and no way to invent operand kinds.

## Adversarial review, round 2 - the hardened API (Main `7aecae5`)

Independent re-review (kernel-occ) after token ownership, the poison flag + `Seq::finish() -> Result`, the hazard
verifier (`verify.rs`), opcode-typed tokens, `vcc() -> Option`, the `AsmBlock`/`Op::Asm` bridge and the DS/VMEM
`Pending` tokens. Reproducers: `scratch/safeasm-review2/` (`run_cases.sh` compiles each case against the crate rlibs
in the pinned container; `hw/run_hw.sh` runs the raw-asm wait-state probes on GPU 5). The crate's own suite (15
tests incl. the attack tests) and the compiler's `tests/safeasm.rs` (15 tests) pass.

### Coverage: closed vs still open

| # | round-1 attack | round-2 result | evidence |
|---|---|---|---|
| 1 | drop the MFMA/SMFMAC token, then read D | **CLOSED.** The accumulator is *moved* into the token: reading it later is `E0382 borrow of moved value`; aliasing it before the op blocks the move (`E0505`); an `Option` slot cannot be taken while borrowed (`E0502`); `drop`/`let _` sets the poison flag and `finish()` reports "unretired matrix result"; `mem::forget` skips `Drop` but the verifier's end-of-sequence rule still refuses; `exit()` without `output_exit` is refused | `cases/fail_drop_read_d.rs`, `fail_alias_before_token.rs`, `fail_option_take_while_borrowed.rs`, `pass_war_poison.rs`, `pass_forget_token.rs`, `pass_exit_dropped.rs` |
| 2 | WAR/WAW with the new ownership | **CLOSED.** Dropping the token releases the operand borrows, but the write then trips both the poison flag and the verifier's WAR rule ("writes r2, an operand instruction 21 is still reading (WAR, issued 1 wait states ago)"); `finish()` returns both reasons | `cases/pass_war_poison.rs` |
| 3 | chain across opcodes | **CLOSED.** `chain` exists only on `Mfma<'a, OP_MFMA_SCALE>`; the `_zero` form is `Mfma<'a, OP_MFMA_SCALE_ZERO>` and only `retire` exists: `E0599 Mfma<'_, 1> is not an iterator` (method not found). The ported phase retires the `_zero` form before the accumulate form and gets the full 20 states | `cases/fail_chain_after_zero.rs`; compiler test `zero_form_then_accumulate_form_gets_the_full_latency` |
| 4 | `AsmBlock` without `finish()`/the verifier | **MOSTLY CLOSED, two holes.** `forge_for_tests` is `pub` (only `#[doc(hidden)]`), so it *is* callable from a normal build (verified: `pass_forge_reachable`), but it needs an existing verified block and every forged block is re-verified by `AsmBlock::verify`, `block_to_asm`, `block_to_mfunction` and the compiler's `Op::Asm` lowering - hazard violations are rejected with precise per-instruction messages. Holes: (a) the verifier **panics** on a forged `SNop` with negative `imm` (`attempt to add with overflow`, `verify.rs:105`; in release the `slot +=` wraps and can corrupt the state accounting); (b) the verifier checks only hazards, not encoding legality: a forged 3-source `VAddF32` passes `verify()` and `block_to_asm` silently emits `v_add_f32_e32 v1, v1, v1`; a forged `VReadFirstLane` with a VGPR destination passes and is silently re-typed to an SGPR | `cases/pass_forge_emit.rs`, `pass_forge_reachable.rs` |
| 5 | wait states on hardware: is 2 states enough (CDNA4 p74 suggests PASS = 4)? | **2 is safe, and conservative**: VALU -> MFMA A/B operand, VALU -> `v_readfirstlane`, VALU -> SMFMAC operand, compare -> `v_cndmask` mask all produced the *fresh* value already with **1** wait state and stayed correct at 2, 3, 4 (four probes each). The matrix-result rule is conservative too: reading D 1 state after the MFMA gives stale 0, but **8 states already gives the correct 64** (12/16/20 also correct), while the verifier/crate require 20. So the constants can stay as they are | `hw/run_hw.sh`, GPU 5 |
| 6 | remaining escape into raw MIR from a safe path | only `forge_for_tests` (4) and the compiler crate itself (raw layer by design); `Pending` tokens cannot be used as registers (`E0308`), a load destination cannot be used before `wait` (`E0382`), `Exit` has no public constructor (`E0451`) | `cases/fail_pending_as_operand.rs`, `fail_load_dest_before_wait.rs`, `fail_forge_exit.rs` |

### New findings from round 2 (all reproduced from the *safe* API unless noted)

1. **F8 - input ports are writable and silently clobber the caller's register (hardware-confirmed wrong result).**
   `Seq::input::<N>()` returns an ordinary owned `Vgpr<N>`, so `&mut x` type-checks; `finish()`/the verifier do not
   enforce the documented read-only contract, and the DSL lowering maps the port straight onto the caller's register.
   Reproducer: block writes its input (= 2), kernel stores the DSL value 7.0 before the block and after it; GPU
   output: before `0x40E00000` (7.0), after **2**. A safe-authored sequence can therefore corrupt a live kernel value.
   **Fix:** return a read-only handle from `input()` (no `Dst` impl), and/or a verifier rule "no instruction may
   define a register listed in `inputs`" plus a lowering check. This is the most important remaining issue.
2. **F9 - outputs are not checked to be defined.** `let x = s.vgpr::<4>(); s.output(x); s.finish()` succeeds with a
   0-instruction block, so an undefined register is handed to the kernel. **Fix:** verifier rule "every output must be
   written before the end (or declared through `output_exit`)".
3. **F10 - `ds_write_b32` can never be used: it always poisons the sequence.** `ops::ds_write_b32` does
   `let _ = self.issue_load(Queue::Lgkm, ());`, and the immediately dropped `Pending<()>` reports "a load was
   abandoned without wait()" in `finish()` - even if the caller later calls `wait_lgkm_all`. **Fix:** track a value-less
   scoreboard entry (or return a `Pending<()>` that must be waited) instead of dropping a `Pending` internally.
4. **F11 - model/lowering asymmetry on constant moves (no wrong result escaped).** `Seq`/the verifier count a
   materialised constant (`splat`) as one wait state, while the embedding lowering's constant cache can drop such
   moves. A block that puts 19 dropped `splat`s between an MFMA and its reader passes `finish()`; `block_to_asm` does
   not const-cache and emits all 19, and the kernel path's own MIR hazard pass re-pads the reader end-to-end (GPU
   result was the correct 64.0), so the safety net held - but the two models should be aligned (e.g. the verifier should
   re-run on the *lowered* MIR, or the lowering must keep the instruction count).
5. **F12 - the verifier's malformed-input behaviour is a panic, not a diagnostic** (see 4a). Use checked arithmetic and
   reject `SNop` `imm` outside `0..=7` (the only values emission produces).

### Hardware-measured minimum wait states (gfx950, MI355X, GPU 5)

| rule | states tried | first state count giving the fresh value | crate/verifier requires | verdict |
|---|---|---|---|---|
| VALU write -> MFMA A/B operand | 1,2,3,4 | **1** | 2 | 2 is safe (conservative) |
| VALU write -> `v_readfirstlane` | 1,2,3,4 | **1** | 2 | 2 is safe (conservative) |
| VALU write -> SMFMAC operand | 1,2,3,4 | **1** | 2 | 2 is safe (conservative) |
| compare (vcc) -> `v_cndmask` mask | 1,2,3,4 | **1** | 2 | 2 is safe (conservative) |
| MFMA result -> VALU reader | 1,8,12,16,20 | **8** (1 state = stale 0 vs 64) | 20 | 20 is safe (conservative) |

### Verdict

For the bug classes it was built for - dropped matrix tokens (F1), WAR/WAW on in-flight operands (F2), reading a
matrix result too early, the 2-state VALU rule, unretired loads, and opcode-mismatched chains (F4) - the hardened
crate **can be quoted as a guarantee, but only "inside its own API"**: the type discipline is backed by an independent
verifier, the poison flag and the MIR hazard pass, and every round-1 bypass we found is now a compile error or a
`finish()` error. It is **not yet a guarantee for "a safe sequence cannot corrupt the surrounding kernel"**: F8
(writable input ports) and F9 (undefined outputs) are real safe-path holes with a hardware-confirmed wrong result, and
`forge_for_tests` is reachable in normal builds with a verifier that panics on malformed input and does not validate
encoding legality. F8/F9 should be fixed (and `forge_for_tests` feature-gated) before the sentence "a safe-asm
sequence cannot produce a wrong kernel" is used.

**Final integration claim (`_vs128` row-sum behind the safe-asm flag): NOT YET VERIFIABLE** - as of Main/`backend-tail`
`7aecae5` the bridge exists (`Op::Asm`, `examples::rowsum_chain_block`, compiler tests pass incl. identical asm vs
the DSL chain) but the kernel does not use it behind a flag yet, so the 15-shape md5 + determinism comparison could
not be run. Re-run `scratch/safeasm-review2` + the standard 15-shape harness as soon as the flag lands.
