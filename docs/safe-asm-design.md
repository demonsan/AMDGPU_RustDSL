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

## Round-2 fixes (F8-F12) and what the safe-path guarantee is now

* **F8 (inputs are read-only):** `Seq::input` returns `In<N>`, which has no `Dst` implementation and no writable view; the verifier rejects any instruction that defines an input register and `lower_asm` re-checks the lowered MIR. In/out registers exist only as `Seq::inout` (bound to a loop variable by `Stmt::VarAsm`).
* **F9:** every output port must be written (or exported in flight). **F10:** `ds_write_b32` takes a scoreboard slot without a token (`wait_lgkm_all` covers it). **F11:** the constant cache in `lower_asm` is only applied if the verifier still accepts the stream with the folded moves removed, otherwise the moves are kept. **F12:** the verifier uses saturating arithmetic and reports malformed immediates as errors; `forge_for_tests` exists only under `cfg(test)` / the `test-support` feature.
* **Guarantee:** a block that `Seq::finish` returns, and that the compiler lowers through `Op::Asm` / `Stmt::VarAsm`, (1) uses only encodable operand forms, (2) satisfies the 2-wait VALU->matrix, 20-wait matrix-result and DS/VMEM scoreboard rules and has no WAR/WAW against in-flight matrix ops, (3) writes only registers it owns or its declared in/out port, (4) produces only defined outputs. It does not prove value correctness, and `amdgpu-compiler`'s MIR remains an unchecked IR for everything that is not a safe block.

### First real use: the `_vs128` row-sum chain

(Historical: the `FMHA_OPT` switch and the legacy kernel were removed in the 2026-11 cleanup; the bridge itself remains.) `FMHA_OPT=<default recipe>,safeasm` routed the four in-place SMFMACs of the per-tile row sum (`smfv`) through `examples::rowsum_var_block()` and `Stmt::VarAsm` (12 statements replace 48 `VarSmfmac`). Result: the emitted assembly is BYTE-IDENTICAL to the DSL path (md5 equal), 15 shapes pass 3 times (REF_OK 15/15; at 64 rows the same 12/3 as the normal path), 358.2 TFLOP/s. Default behaviour is unchanged (the flag is opt-in).


## Generated opcode templates and coverage (round 3)

The VALU templates are GENERATED by `crates/amdgpu-safeasm/build.rs` from the operand-class table `crates/amdgpu-safeasm/opspec.tbl` (`method | MOp key | form`). The forms are `vop1`, `vop2` (src0 `Operand`, src1 `VgprOnly`), `vop2_imm0` (shift amount is an `Imm<K>`), `vop3_2` / `vop3_3` (compile-time constant-bus assertion `Bus1`/`Bus3`, no literal), `cmp` (`Vcc<Free>` -> `Vcc<Live>`) and `store_vm`. `build.rs` fails the build if a row's key is not in `backend/llvm24.tbl` or its arity disagrees with the table's signature, so the template set cannot drift from the backend table. The hand-written ops that were replaced (`v_add_u32`, `v_add_f32`, `v_mul_lo_u32`, `v_cmp_lt_i32`) now come from the generator and the whole ui suite (all diagnostics) passes unchanged on the generated code, which is the equivalence proof.

**Coverage** (`cargo test -p amdgpu-kernels --test safeasm coverage_matrix -- --nocapture`; 91 rows in the backend table): 26 generated typed templates, 14 hand-written (tokens/type-state: v_mov, v_cndmask (vcc and SGPR-pair forms), v_readfirstlane, scaled MFMA (accumulate/zero), SMFMAC, ds_read_b32/b128, ds_write_b32, global_load_dwordx4, in-place add, register init), and 54 raw-only. Raw-only on purpose: scalar ops (S*), branches/barriers/waits/markers, transcendental ops (v_exp, v_rcp) and packed fp32 (v_pk_*_f32) whose result hazards the verifier does not model yet, partial-write converts (v_cvt_pk_fp8 lo/hi), v_permlane32_swap, carry ops (v_add_co/addc), pointer add, v_readsub, the f16/fp8 16x16 and 32x32x8 MFMAs, ds_write_b128, ds_read_b64_tr_b8, and the scalar-address global loads. Raw-only ops stay available through the MIR/`MOp` layer (not through `Seq`), i.e. without the safe-path guarantee.

## LDS / VMEM tokens and the runnable fragment

`ds_read_*`, `global_load_dwordx4` return `Pending` (they own the destination); `Pending::wait` emits the partial `lgkmcnt(k)` / `vmcnt(k)` from the scoreboard (nothing if an earlier wait already covered it); stores (`ds_write_b32`, `global_store_dword`) take a queue slot; `wait_lgkm_cnt(k)` / `wait_vm_cnt(k)` are explicit partial waits that return an error when the count waits for nothing; a sequence ending with outstanding operations is a `finish()` error. ui cases: read before wait, double wait (moved token), scalar store data.
`examples::global_probe_block` + the kernel `probe_safeasm_e2e` (`tools/safeasm_e2e_probe.sh`, host `tests/gpu/safeasm_e2e_host.cpp`) is a complete safe-Rust kernel fragment: global load (vmcnt in flight) -> scaled MFMA (ones x ones, every D element 64.0) while the load is outstanding -> `vmcnt(0)` -> VALU sums -> global store. Hardware (MI355X, GPU 4): 64/64 lanes exact (`out[tid] = sum(in[4*tid..4*tid+4]) + 64`).


## Round-3 final verification (independent, GPU 5)

### Byte-identical integration, reproduced

`_vs128` emitted with `--backend own` in three configurations - `FMHA_OPT` unset (base), the explicit default recipe
(`cvt4,pksub,lazy,msum,permlane,kdlds,dbuf,lacc1,psh2,svpk,svcorr,swz`) and that recipe plus `safeasm`:

* `.s` md5 is **bc0590617995ebb13c79afd31efcca33** in all three cases (9825 lines each), `.co` md5
  **57f6bc1a4ee650e29faf27cf30e36ff7** in all three, vgpr=248 sgpr=45;
* `diff base.s safeasm.s` = **0 lines**, `diff default.s safeasm.s` = **0 lines**;
* the DSL dump shows the route is actually taken: default has 48 `varN += smfmac(...)` statements and 0 asm blocks,
  `safeasm` has **12 `varN <- asm `safeasm` (...)`** blocks and 0 `VarSmfmac` (48 SMFMACs -> 12 `Stmt::VarAsm`, 36
  fewer dump lines). The flag is opt-in; the default build is unchanged.

### Gates, determinism and timing on the safeasm build (GPU 5)

* 15 reference shapes at **16 rows**: **15/15 REF_KERNEL_OK, 0 FAIL** (identical per-shape result lines to the default build).
* 15 reference shapes at **64 rows**: **12 OK / 3 FAIL - exactly the same 12/3 as the default build** (this is the known
  worst-element noise at the 64-row sample, not a safe-asm regression; the asm is byte-identical).
* determinism 3 runs x 5 shapes: **5/5 DET_OK**, identical md5 per shape.
* timing, same vendor reference and same run: default **358.88 / 358.99 / 359.33 TFLOP/s**, safeasm
  **359.39 / 359.21 / 359.17 TFLOP/s** (vendor 516.2 - 517.0 in the same runs) - identical within noise, matching the
  reported ~358 TFLOP/s.

### Final closed/open table

| issue | status | evidence |
|---|---|---|
| F1 dropped matrix token / early read of D | **closed** | `In`... see round 2; `E0382`, poison, verifier; unchanged |
| F2 WAR/WAW on in-flight operands | **closed** | poison + verifier WAR diagnostics |
| F3 verifier in `finish()` / `Op::Asm` lowering | **closed** | runs in all three places; forged/rewritten streams rejected |
| F4 chain across opcodes | **closed** | `chain` only on `Mfma<OP_MFMA_SCALE>`, `E0599` for the `_zero` form |
| F5 2-state wait rule | **closed/validated** | hardware: fresh value already at 1 state for all four 2-state classes; 2 is safe |
| F6 `vcc()` | **closed** | returns `Option`, second call is `None` (no panic) |
| F8 writable input ports (round-2 hole, hardware-confirmed) | **closed** | `Seq::input` returns `In<N>`: writing is `E0277 &mut In<1>: Dst<1>`, `lane_mut` `E0308`, matrix accumulator `E0308`; verifier rejects any instruction defining an input register and `lower_asm` re-checks the lowered MIR. An in/out block fully bound through `Op::Asm` passes DSL verify but is rejected at lowering ("6 inputs bound for 5 read-only ports (in/out port bound: false)"). Positive hardware check: a block that *reads* an input, embedded in a kernel that stores the same DSL value before and after, leaves it at 7.0 (lane0 = [7, 7]) |
| F9 undefined output | **closed** | `finish()` -> "output port r0 is never written by the sequence (an undefined value would flow into the kernel)" |
| F10 `ds_write` unusable | **closed** | `ds_write_b32` takes a scoreboard slot without a token; `finish()` Ok, listing `[... LdsWriteB32, SWaitLgkm ...]` |
| F11 const-folded stream vs wait states | **closed** | the folded stream is re-verified; the round-2 "19 dropped splats" block now emits **20 wait states** between the MFMA and its reader (needs 20) |
| F12 forge/malformed input | **closed** | `forge_for_tests` is **unreachable in a normal build** (`E0425 cannot find value`), `AsmBlock` fields are private (`E0451`), `verify_insts` only returns a diagnostic (cannot manufacture a block), malformed `s_nop` immediates and saturating arithmetic are verifier errors |
| input-through-loop / `inout` misuse | **closed** | `var_asm` with a wrong loop-variable type -> "in/out port V4 does not match var0"; wrong input count -> "4 inputs bound for 5 remaining ports"; correct loop use verifies clean |

### Hardware-measured minimum wait states (gfx950, GPU 5) - carried forward, unchanged

| rule | states tried | first correct | crate requires | verdict |
|---|---|---|---|---|
| VALU -> MFMA A/B | 1,2,3,4 | **1** | 2 | 2 safe |
| VALU -> `v_readfirstlane` | 1,2,3,4 | **1** | 2 | 2 safe |
| VALU -> SMFMAC operand | 1,2,3,4 | **1** | 2 | 2 safe |
| compare -> `v_cndmask` mask | 1,2,3,4 | **1** | 2 | 2 safe |
| MFMA result -> reader | 1,8,12,16,20 | **8** (1 stale: 0 vs 64) | 20 | 20 conservative/safe |

### Final guarantee verdict

**Yes - with a precisely scoped sentence.** For a sequence obtained from `Seq::finish()` and embedded through
`Op::Asm` / `Stmt::VarAsm` into a kernel built by `amdgpu-compiler`, the crate's guarantee can now be quoted as:

> A safe-authored sequence cannot corrupt the kernel's data and cannot emit an illegal or under-padded instruction
> hazard: it can only write registers it owns or the in/out port it was given (inputs are read-only at the type,
> verifier and lowering levels), every output is defined, and the emitted instruction stream satisfies the 2-wait
> VALU->matrix, 20-wait matrix-result and DS/VMEM scoreboard rules with no WAR/WAW against in-flight matrix ops,
> independently of the type-level discipline (poison flag + verifier + the MIR hazard pass).

What this does **not** claim: arithmetic/value correctness (scale semantics, index patterns, wrong fragment contents);
hazards across arbitrary control flow other than the `VarAsm` in/out contract (block-boundary contracts for
software-pipelined loops are still not modelled); and anything outside a safe block (`amdgpu-compiler`'s MIR remains
an unchecked IR for hand-written code). The independent round-3 evidence above (byte-identical `_vs128` assembly,
15/15 + 12/3 gates matching the default, 5/5 determinism, equal timing) is what upgrades the claim from "inside its
own API" to "usable in the real kernel".

## Loop-level contract: adversarial opinion (kernel-occ, validator)

**Verdict: a loop-level contract can be enforced well enough to trust a hand-authored window, but the
current design does not yet do it - for the cross-iteration part where the window's win lives, the trust
boundary today is still "the author plus the MIR hazard pass". Ship the window opt-in; make it default only
after the three conditions below.**

Why the current guarantee does not cover a loop:
1. **The verifier is straight-line and starts its clock at 0.** `verify_block` walks one `Seq` with its own
   clock, requires no outstanding DS/VMEM loads at the end and allows matrix writes in flight only if
   declared via `output_exit`/`exit_in_flight`. It has no notion of a back edge, fixed point, or entry state.
   `Stmt::VarAsm` carries **values** through in/out ports, not scoreboard in-flight state. A window that
   overlaps tile N's VALU with tile N-1's MFMA (exactly what recovers issue overlap) needs a non-empty
   entry state; the verifier cannot check the first/last iteration distances or loop-carried WAR/WAW. The
   MIR hazard pass is then the only net, and it does not know the author's intended in-flight set.
2. **Coverage is partial by design.** The coverage matrix has 54 raw-only ops out of 91 backend-table rows:
   `v_exp`/`v_rcp`, `v_pk_*_f32`, `v_permlane32_swap`, carry ops, the 16x16/32x32x8 f16/fp8 MFMAs,
   `ds_write_b128`, scalar-address global loads, etc. A real inner window needs several of these; using
   them through the raw MOp escape hatch leaves the safe-path guarantee, and the verifier pattern-matches
   only the op classes it knows (`VMfma`/`VSmfmac`/`V*`/loads), so packed-result hazards, multi-dword DS
   store hazards and transcendental wait classes are not modelled. The crate's own doc admits this.
3. **The static-contract precedent is linear, not CFG-aware.** `asm_check::check_lite_contract` explicitly
   says "a drain on any path before the use counts" and "control flow is not followed"; exact for the
   straight-line release, but a loop contract cannot be enforced by that style of checker on the loop back
   edge.

What an enforceable loop contract needs (all feasible, none free):
* **Drain-closed block boundaries or explicit state.** Either require the loop-bound block to enter and
  exit with no matrix ops in flight and no outstanding DS/VMEM loads (all outputs retired; forbid
  `output_exit` in loops) - then the per-iteration verifier composes and trust reduces to arithmetic
  intent - or add `Seq<Entry, Exit>` state tokens and a fixed-point check that the entry state equals the
  exit state each iteration (and that first-iteration entry matches the loop prologue, last-iteration exit
  matches the epilogue).
* **Op whitelist for loop-bound blocks**: only generated typed templates (extend `opspec.tbl`/verifier to
  cover the ops the window actually needs), or an explicit verifier rule per raw op used; reject unknown
  raw MOps in a loop context.
* **Verify at the right level**: run the verifier on the final lowered + scheduled MIR/asm with CFG
  (including the back edge), not only on the `Seq`; the list/SWP scheduler is allowed to add waits, never
  remove them, but the final asm must be re-checked because block boundaries can move. Add the CFG-aware
  loop check alongside `asm_check` (a linear checker is not enough).
* **Independent validation is still mandatory**: md5 bit-identity vs `_vs128`, the counter direction
  (WAIT_INST_ANY down, XDL busy up, wave cycles down), same-run per-CU/filled timing, and the 15-shape
  gate/determinism through the auto path. A contract can prove legality, not that the window implements
  the right arithmetic.

Residual "trust the author" even with the contract: fragment contents/scale semantics/index patterns
(value correctness), and anything outside a safe block. Those cannot be typed away by a loop contract;
they are covered by the bit-identical md5 gate and the CPU/vendor correctness gates.

**Recommendation:** opt-in first. The minimal defensible contract for this window is a **drain-closed
per-tile block** (start/end no matrix in flight, no outstanding loads, all ops typed or verifier-covered,
`VarAsm` in/out ports only); if the window must pipeline across tiles, require the explicit `Seq<Entry,Exit>`
fixed-point contract before it can be a default path. Until then a green hardware run is evidence about the
tested schedules, not a proof.

## Round 4: window enablers (DS ops, conversions, pinning, loop-body contract, final-MIR check)

* **DS ops** (`ds_read_b32/b128/b64_tr_b8`, `ds_write_b32/b128`): offsets are const generics with a compile-time alignment/range check; reads return `Pending<D>` for any `Dst<N>` destination (owned tuple or a writable view), partial `lgkmcnt(k)` from the scoreboard. Writable views: `pair_mut`, `half8_mut`, `quarter16_mut` and the disjoint splitters `split_pairs_mut` / `split_halves8_mut` / `split_quarters16_mut` (one mutable borrow, several views in flight).
* **Conversions/packs**: `v_pk_add_f32`/`v_pk_mul_f32` (pair operands, pk->trans hazard), `v_exp/rcp/log_f32` (`Trans` token, 4 wait states), `v_permlane32_swap_b32`, and the half-write converts `v_cvt_scalef32_pk_fp8_bf16` / `v_cvt_pk_fp8_f32` lo/hi as a type-state (`Pw<L,H>`: the dword can only be taken back when both halves are written; the verifier tracks halves on the stream as well).
* **Loop contract**: `Seq::finish_loop_body()` = drain-closed (no matrix result in flight, no outstanding DS/VMEM, no `output_exit`); op whitelist (anything not typed/hand-written is rejected as raw-only).
* **Pinning** (`Seq::pin()`): the block's instructions are barrier nodes for the list scheduler (order preserved verbatim, nothing moves into/out of/across), the hazard pass's stall filling and the lowering passes that move/merge instructions skip pinned instructions. Waits for the loads a pinned block's inputs depend on are made explicit at the start of the run.
* **Final-MIR check** (`pin_check`, runs on the scheduled MIR of every kernel with pinned instructions, an error stops the compile): re-derives the value-hazard padding and the DS/VMEM waits with the compiler's own models and rejects any padding/wait that would have to be inserted INSIDE a pinned run (at its boundary it is normal), and any pinned read of a half-written dword.
* **Own-tail hazard audit**: VALU writer classes now include the scalef32 half-write converts, the permlane swap and the register-read ops; an explicit `s_nop N` counts N+1 wait states. No change to the shipped kernels' asm (checked).
* **Hardware evidence**: `tools/cvtchain_probe.sh` (kernels `probe_cvtchain_dsl` / `probe_cvtchain_asm`, the asm one embeds the pinned `examples::cvt_chain_block`: pk_add/pk_mul/pknorm/svcorr/scalef32 lo+hi/ds_write + transposed ds_read): 50 data sets, bit-identical outputs; `cvt_chain_instruction_sequence_matches_the_dsl_path` checks the chain instruction sequence is identical (9 chain mnemonic groups in both kernels).

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

## Status: complete

The safe-asm line is finished: the crate, the compiler bridge and the contract are done and validated on hardware; the window experiments that motivated the last rounds are closed.

**Coverage.** 29 generated + 27 hand-written typed templates (`opspec.tbl` + `build.rs` for the VALU/store families; tokens and type-states by hand), 51 of 102 backend rows raw-only
(`cargo test -p amdgpu-kernels --test safeasm coverage_matrix`), and 29 ui cases (28 compile-fail + 1 pass control; one `rustc` driver, no `trybuild` because the container cargo is 1.83).
Typed families: VALU int/fp, `pk_add/mul/sub`, `perm`, `permlane32_swap`, `pknorm`, half-write converts (`Pw<L,H>` type-state), transcendental (`Trans` token, 4 wait states), all MFMA forms
(scaled accumulate / zero-C / explicit-C, unscaled accumulate / zero-C / explicit-C, f16 zero-C with its 12-state result), `smfmac`, DS (`ds_read_b32/b128/b64_tr_b8`, `ds_write_b32/b128`
with const-generic aligned offsets and writable views as destinations), `global_load_dwordx4`, `global_store_dwordx2/x4`.

**The four adversarial review rounds (independent, kernel-occ) and what each closed.**

| round | found | closed by |
|---|---|---|
| 1 (`a50369d`) | dropped matrix token / early read of D, WAR/WAW against in-flight operands, verifier not run on `finish`, chain-opcode mixups, 2-wait VALU->matrix, `vcc` reuse | ownership tokens + poison-on-drop, finish-time verifier, per-opcode chain rules, `Vcc<Free/Live>` type-state |
| 2 (`7aecae5`) | **F8** writable input ports, **F9** undefined outputs, F10 `ds_write` without a scoreboard slot, F11 constant folding that broke the verified stream, F12 verifier panics / unvalidated encodings | read-only `In<N>`, every output must be written, stores take queue slots, fold only if the verifier still accepts, saturating verifier, `forge_for_tests` gated |
| 3 (final verification, hardware) | integration claim unverifiable until a flag existed | generated templates, `Pending` tokens, e2e probe (64/64 exact), `FMHA_OPT=...,safeasm`: the `_vs128` row-sum chain is **byte-identical** to the DSL path (15 shapes x3 REF_OK, determinism) |
| 4 (loop-level contract opinion) | a straight-line verifier is not sound across loop iterations; raw-only ops could sneak into a window; scheduler/RA could rewrite hand order | drain-closed boundaries, op whitelist, instruction pinning + the final-MIR `pin_check` |

**The loop contract (drain-closed boundary).** A loop-body `Seq` is finished with `finish_loop_body()`: it must enter and exit with NO matrix op in flight and NO outstanding DS/VMEM loads, and
`output_exit` (an in-flight matrix result at the exit) is rejected. The entry/exit wait clock is therefore defined, and the straight-line verifier is sound for every iteration.
**Op whitelist.** The verifier accepts only generated templates, hand-written templates and internal bookkeeping ops; a raw-only `MOp` forged into a stream is rejected ("not modelled").
**Pin + `pin_check`.** `Seq::pin()` marks the block's instructions as scheduler barriers (order preserved verbatim, nothing moves into, out of or across; see `docs/scheduling.md`); after
scheduling, `pin_check` re-derives value-hazard padding, DS/VMEM waits and half-written dwords with the COMPILER's models and fails the compile if any padding or wait would have to be inserted
inside a pinned run. The waits for the loads a pinned block's inputs depend on are made explicit at the start of the run.

**Bridge.** `Op::Asm` (value form), `Stmt::VarAsm` (one in/out loop variable), `Stmt::VarAsmN` (several, in declaration order); writable views (`pair_mut`, `half8_mut`, `quarter16_mut`
and the disjoint splitters) let several DS reads fill one MFMA operand.

**Hardware-validated reference blocks.** `rowsum_var_block` (the four in-place SMFMACs of the `_vs128` row sum, byte-identical asm), `pv_rowsum_phase` (PV + row sum), `cvt_chain_block`
(pk_add/pk_mul -> pknorm -> svcorr (lshr/and/perm/pk_sub) -> two fp8 P words with `v_cvt_scalef32_pk_fp8_bf16` lo/hi -> `ds_write_b32` -> `ds_read_b64_tr_b8`: bit-identical outputs vs the DSL path on
50 data sets, identical chain instruction sequence), `global_probe_block` (`probe_safeasm_e2e`: load in flight -> scaled MFMA -> `vmcnt(0)` -> VALU -> store, 64/64 exact).

**Real windows and the closure of the gen5 line.** Two real hand-authored windows of `_vs128` were built on top of the finished crate: Block C (PV + row sums, pinned) **-2.5%** and Block M
(the whole M phase as one pinned `VarAsmN` block) **neutral**; both bit-identical to the DSL path, both correct under the full gates. Pinning removes the scheduler's PV/QK interleave and the
shipped kernels are not instruction-issue limited (`docs/scheduling.md`), so the gen5 window line is **closed**; the crate stays as a verified tool and a hazard-model oracle, not as a performance lever.

**Hazards found with it (fixed in the compiler, regression-tested).** Physical WAW against in-flight MFMA destinations, the VMEM wide-store data WAR (2 wait states for `dwordx4` store data), the
missing VALU-writer classes (`scalef32` half-write converts, `permlane32_swap`, register reads), an explicit `s_nop N` counting as one state, and the two-phase RA reaching-writer false positive.