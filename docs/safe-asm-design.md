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