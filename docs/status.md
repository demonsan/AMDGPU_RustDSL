# Implementation status

Updated after the first end-to-end milestone on gfx950 (`mi355-gpu-53`).

## Verified working

| Step | Evidence |
|---|---|
| Rust toolchain only inside a container | `tools/container/Containerfile` (ubuntu 22.04 + rustup 1.83.0), built with rootless podman; host and shared home untouched |
| Workspace builds and tests | `tools/container/cargo.sh build` clean, `cargo test` → 8 backend tests pass |
| Typed IR + verifier | `amdgpu-dsl`: SSA values, loop variables, scope and type checking |
| Lowering + own register allocation | `amdgpu-compiler`: MIR, linear-scan RA, copy coalescing, wait-state insertion |
| Assembly emission | `amdc emit attention_fwd` → gfx950 asm accepted by the ROCm assembler |
| Code object loads and dispatches | HIP module load + launch from `tests/gpu/attn_host.cpp` |
| ABI + memory path | `probe_store` kernel: `o[tid] = q[tid]` plus a kernarg scalar, `PROBE_OK` |
| Hardware transcendental behaviour | `probe_math`: `exp(1) = 2.0`, `exp(2) = 4.0`, `rcp(3) = 0.333333313` |
| **Attention kernel numerics** | `attention_fwd` on gfx950 vs CPU reference: `H=1 S=64` max_abs_err `9.4e-05`; `H=2 S=128` max_abs_err `6.0e-05`; `bad = 0` in both |
| MFMA infrastructure | `probe_mfma`: emitted `v_mfma_f32_16x16x16_f16 v[12:15], v[4:5], v[6:7], v[8:11]` with 4/2/2/4-dword tuples; all 256 accumulator elements `16.0` → `PROBE_MFMA_OK` |
| LDS, barriers, `lgkmcnt` | `probe_lds`: 64 lanes write their id to LDS, barrier, read the neighbour's value → `PROBE_LDS_OK`; descriptor declares 256 B of LDS |
| Wide MFMA (16-dword accumulator) | `probe_mfma_wide`: `v_mfma_f32_32x32x8_f16` with a 16-dword zeroed accumulator; every element `8.0` → `PROBE_MFMA_WIDE_OK` |
| FP8 MFMA | `probe_mfma_fp8`: all-ones E4M3 operands; `v_mfma_f32_16x16x32_fp8_fp8` → `32.0` (K=32), `v_mfma_f32_32x32x16_fp8_fp8` → `16.0` (K=16) → `PROBE_MFMA_FP8_OK` |
| Block-scaled fp8 MFMA | `probe_mfma_scale`: `v_mfma_scale_f32_32x32x64_f8f6f4` with E8M0 scales; `(127,127)` → `64.0`, `(127,128)` → `128.0`, `(127,0x8000)` → `0.0`, `(127,0x7f00)` → `0.0` → `PROBE_MFMA_SCALE_OK` |
| Wave-role branching (8 waves) | `probe_roles`: waves 0-3 store `111.0`, waves 4-7 store `222.0` (`marker_bad = 0`) and exchange LDS data across waves (`bad = 0`) |
| FMHA control skeleton | `probe_roles_loop`: 8 waves, role A/B, 4 iterations, two paired barriers per iteration, cross-role LDS exchange → `PROBE_ROLES_LOOP_OK` |
| FMHA data path | `probe_fmha_datapath`: distinct A and B tiles staged in LDS (`ds_write_b128`), read back (`ds_read_b128`), concatenated into `Frag8` operands and multiplied with `v_mfma_scale_f32_32x32x64_f8f6f4` → every value is `64.0` → `PROBE_FMHA_DATAPATH_OK` |
| A/B paired loops | `probe_ab_loops`: 8 waves, role A/B, outer loop with two sequential inner phase loops, paired barriers, LDS read-modify-write accumulator → `PROBE_AB_LOOPS_OK` (closed form over 512 lanes) |
| Nested loops, register accumulator | `probe_nested_acc`: inner loop updates the outer loop variable, accumulator read after the loops → `PROBE_NESTED_ACC_OK` |
| MFMA accumulation loop | `probe_loop_mfma`: `Frag16` loop variable accumulated by `v_mfma_scale_f32_32x32x64_f8f6f4` over two K=64 iterations → every element `128.0` → `PROBE_LOOP_MFMA_OK` |
| FMHA core loop | `probe_fmha_core`: A/B phase staging with paired barriers, `ds_read_b128` → `Frag8` → in-place MFMA accumulation in a loop → every value `128.0` → `PROBE_FMHA_CORE_OK` |
| fp8 matmul (real data) | `probe_fp8_matmul`: identity and per-lane-scale tests through the real fragment layout -> `PROBE_FP8_MATMUL_OK` |
| fp8 load path | `probe_fp8_load`: address arithmetic and the ISA fragment mapping for the real-data fp8 matmul probe -> `PROBE_FP8_LOAD_OK` |
| Pack instructions | `probe_pack`: `v_cvt_pknorm_u16_f32(1.0, 0.5)` = `0x8000FFFF`; `v_cvt_scalef32_pk_fp8_bf16` low half = `0xAAAA3838`, high half = `0x3838AAAA` → `PROBE_PACK_OK` |
| LDS self-read | `probe_lds_self2`: own-slot read after a barrier → `SELF2_OK` |
| Two-wave LDS | `probe_waves2`: own + partner-wave slot → `PROBE_WAVES2_OK` |
| `v_max3_f32`, fp8 fragments | `v_max3_f32` in the ISA of the compiler, `Frag8`/`V8` register class for the `32x32x64` operands |

`attention_fwd` is a real dense attention kernel (one lane per query row, online
softmax over the key loop, f32 accumulation, f16 output) that is fully generated
by this compiler: typed DSL → internal MIR → own RA → gfx950 assembly → `.co` →
executed and numerically verified on hardware.

## Bugs found and fixed on the way

Compiler/IR:

1. `ValueId`/`VarId` were both `u32` → conflicting `From` impls; `VarId` is a newtype now.
2. Nested DSL builder calls had to be split into locals (Rust borrow rules).
3. The allocator did not mark assigned slots → every virtual register got the same physical register.
4. Constant moves carried the value in `imm` but the emitter only read `srcs` → **every constant was emitted as 0**.
5. `max_v`/`max_s` were computed from the occupancy bitmap *after* allocation, so reclaimed high slots were not declared; code used `v208` while metadata said 140 VGPRs.
6. `BreakIf` branch polarity was inverted.

Code object / ABI:

7. Missing `.amdhsa_user_sgpr_count` → no kernarg pointer was passed.
8. Kernel descriptor not 64-byte aligned → the runtime read every field shifted; surfaced as "illegal instruction" / "memory aperture violation".
9. `kernarg_size` not 8-byte aligned → the assembler rejected the metadata.
10. Single-letter argument names (`n`) parse as YAML booleans → argument names are quoted.

Hardware behaviour that the compiler now models:

11. **`v_exp_f32` computes `2^x`, not `e^x`.** Natural exp is lowered to
    `exp2(x * log2(e))`. This was invisible to the constant-input test (`exp(0) = 1`
    either way) and only showed up with random data.
12. **`v_mfma_scale_*_f8f6f4` scales are E8M0 exponents in the low byte of a
    VGPR**, one scalar per matrix: the result is
    `2^(a-127) * 2^(b-127) * (A @ B) + C`, and A/B bytes are decoded as E4M3.
    Passing the scales in SGPRs (as the syntax also accepts) silently produced
    zero, so the DSL/lowering now always materialises them as VGPRs.
13. **LDS read results need an `lgkmcnt` wait before use.** The lowerer originally
    only tracked VMEM and kernarg loads, so `ds_read_b32` results were consumed
    too early and read as zero. `pending_lds` now forces `s_waitcnt lgkmcnt(0)`.
14. **Transcendental results are not interlocked.** A consumer issued in the
    instruction slot right after `v_exp_f32`/`v_rcp_f32` reads the stale register
    value. `hazards::fix_value_hazards` models the latency and pads with `s_nop`;
    `attention_fwd` relies on it (it computes `l_new` immediately after `p` on
    purpose).

Host harness:

13. `hipModuleLaunchKernel` needs one pointer per argument, not one pointer to the whole kernarg struct.

## Regression tests

`crates/amdgpu-compiler/tests/backend.rs` (no GPU needed):

* every registered kernel compiles and has descriptor + metadata;
* the descriptor covers every physical register the instruction stream touches;
* the kernel descriptor is emitted with `.p2align 6`;
* natural exp lowers through `v_exp_f32` with the `log2(e)` constant;
* transcendental results get wait states;
* the IR verifier rejects ill-typed programs;
* the attention IR verifies and really contains a loop.

## Resolved: the "LDS self-read / multi-wave exchange" anomaly

Earlier rounds reported that reading a lane's own LDS slot, or exchanging LDS
data across waves, "read back zero". That was **not** a hardware or compiler
problem: the kernels store raw 32-bit values (`bitcast_f32` is an alias, so the
stored word is the integer bit pattern), while those probe checks reinterpreted
the buffer as `float` and compared against `(float)tid`. Small integers as float
bits are denormal values that print as `0.0`, so the checks failed while the
hardware behaved correctly. Checks now compare raw bit patterns.

Verified after the fix (all on gfx950):

| probe | meaning |
| --- | --- |
| `probe_lds` | neighbour-slot read after a barrier (`bad=0`) |
| `probe_lds_self2` | own-slot read after a barrier (`bad=0`) |
| `probe_waves2` | two waves: own slot + partner-wave slot (`own_bad=0 partner_bad=0`) |
| `probe_roles` | eight waves: role A/B ids exchanged across waves (`bad=0`) |
| `probe_roles_loop` | 8 waves, roles, 4 iterations, two paired barriers per iteration, cross-role LDS exchange (`bad=0`, 2048 values) |

`probe_roles_loop` is the control skeleton of the FMHA reference core loop, so
the structural pieces needed for the 1:1 reproduction are in place.

## Resolved: the 16 KB LDS probe failure

The 32 B/lane variant was writing its B tile at `8192 + tid*32`, which reaches
`24560` B — past the declared 16 KB and over the A tile. It was a probe bug, not
an LDS capacity or compiler problem. With both tiles inside the declaration the
A/B data path is exact.

## IR capabilities added for the loop (round 9)

1. `SetVar` now accepts **any enclosing loop's** variable, so an inner loop can
   update an outer loop accumulator.
2. Values and loop variables stay visible **after `EndLoop`**; a loop variable
   then holds its last value (undefined if the body never ran).
3. Loop variables can have fragment types (`Frag2/4/8/16`), so a `Frag16` MFMA
   accumulator can live in registers across iterations. Copies are emitted as
   `VReadSub`/`VWriteSub` pairs.
4. Every fragment operand materialisation now calls `touch()`, so `ds_read_*`
   results feeding `cat_frag8`/MFMA get their `lgkmcnt` wait.

Still unsupported: pointer-typed loop variables.

Risk to watch: the MFMA accumulation loop produced `64.0` instead of `128.0`
in an earlier register allocation and `128.0` after the probe gained a
diagnostic store (same IR semantics). That points at a register-aliasing hazard
for MFMA D/C versus A/B that the allocator does not model; the condition needs
to be found and enforced in MIR verification.

## Hardware facts confirmed from the CDNA4 ISA (kernel wiki)

Source: `MI_Kernel_Agent/skills/kernel-wiki/references/mi300-sources/cdna4-isa`
(chapter 7, MFMA).

* The matrix core has its own register file, the **AccVGPRs**. The `ACC` field of
  an MFMA instruction selects VGPRs (0) or AccVGPRs (1) for A (low bit) and B
  (high bit), and `ACC_CD` does the same for C and D. In the code object the
  boundary between the two files is `.amdhsa_accum_offset`.
* Operands must be **contiguous and the base register aligned to the operand
  width** (a 4-register A needs a base that is a multiple of 4). Our raw/linear
  allocator aligns tuples to their width, but this is not verified yet.
* For the F16/BF16/I8 MFMA group the sources and destination must additionally be
  **even aligned**.
* MFMA ignores the execution mask (forced to 1).
* `V_MFMA_SCALE_F32_32X32X64_F8F6F4` is a **4-dword instruction**: two back to
  back VOP3Ps, the first carrying `0xD3AC` plus the two scale sources (SRC0 = A
  scale, SRC1 = B scale). `OP_SEL[0]`/`OP_SEL[1]` pick **which byte** of the
  scale VGPR is used (00 -> `[7:0]`, 01 -> `[15:8]`, 10 -> `[23:16]`, 11 ->
  `[31:24]`). There is one scale byte per lane per 32 K-values and the scale
  applies to that instruction only.
* The F8F6F4 MFMA takes 32 cycles when either A or B is F8 and 16 otherwise
  (64/32 for the 32x32x64 form).

## MFMA accumulator is in place

`lower.rs` now materialises the incoming accumulator into the destination
registers and emits `C == D` (the form the aiter reference kernel uses), and
`regalloc` rejects any MFMA where C and D are not the same range or where A/B
overlap D. All MFMA probes still pass with this form: `PROBE_MFMA_SCALE_OK`,
`PROBE_MFMA_FP8_OK`, `PROBE_MFMA_WIDE_OK`, `PROBE_LOOP_MFMA_OK` (128.0),
`PROBE_FMHA_DATAPATH_OK`.

## Resolved: MFMA result reads need independent instructions

CDNA4 ISA section 7.6 ("Dependency Resolution: Required Independent
Instructions") states that an MFMA result written to a VGPR must not be read by a
VALU/VM/LDS instruction until 5/8/12/20 independent instructions have been
issued, depending on the opcode (there is no forwarding path). Chained in-place
MFMA accumulation of the same opcode needs 0. `hazards.rs` now models this: every
MFMA result gets a ready cycle and consumers are padded with `s_nop`.

That was the cause of the zeros from `probe_fmha_core` (the accumulator copy
right after the MFMA read a partially written VGPR). The same padding also makes
the other MFMA probes deterministic instead of timing-dependent.

Verified after the fix:

| probe | result |
| --- | --- |
| `probe_fmha_core` | A/B phase staging, three paired barriers per iteration, in-place MFMA accumulation over two K=64 iterations -> every value `128.0` -> `PROBE_FMHA_CORE_OK` |
| `probe_loop_mfma` | `PROBE_LOOP_MFMA_OK` (`128.0`) |
| `probe_mfma_scale` / `probe_mfma_fp8` / `probe_mfma_wide` | all still OK |
| `attention_fwd` | `bad=0/16384`, `max_abs_err=5.996e-05` |

Follow-ups: the wait counts are the conservative end of the ISA table (12 for the
16x16/32x32 F16/FP8 shapes, 20 for F8F6F4) and the pass is per basic block, so
cross-block producers are not covered yet. Both should be tightened/validated
once the FMHA core loop is wired to real data.

## Pack instructions and MFMA alignment checks (round 12)

* `v_cvt_pknorm_u16_f32` is wired through DSL/MIR/emit and verified on hardware:
  `(1.0, 0.5)` produces `0x8000FFFF` (low half = S0 as unorm16, high = S1).
* `v_cvt_scalef32_pk_fp8_bf16` is wired with the low-half form
  (`v_cvt_scalef32_pk_fp8_bf16 vD, vS0, vS1`), verified to replace only bits
  `[15:0]` of the destination and keep the rest (`0xAAAA3838`). Its OPSEL[3]
  high-half encoding could not be pinned down (`op_sel:[0,0,0,1]` and
  `op_sel:[0,0,1]` both produce wrong values), so the DSL builds the high half
  explicitly with `v_lshlrev_b32` + `v_and_b32` + `v_or_b32`; verified to give
  `0x3838AAAA`. The native high-half form should be found later to save the VALU
  ops.
* MFMA operand alignment is now enforced after allocation: each operand base must
  be aligned to its width (and even), as the CDNA4 ISA requires.

## Real-data block-scaled fp8 matmul is numerically verified

`probe_fp8_matmul` feeds a genuine 32x64 fp8 A and 64x32 fp8 B (stored
transposed) from global memory through `v_mfma_scale_f32_32x32x64_f8f6f4` and
compares against an analytic CPU reference:

* test 1 (identity): `A[i][k] = (k == i)`, `B[k][j] = (j == k)` -> `D` must be the
  32x32 identity,
* test 2 (scales): all-ones A and B with per-lane E8M0 scales
  `127 + (row & 1)` -> `D[i][j] = 64 * 2^((i & 1) + (j & 1))`.

Both pass on hardware (`PROBE_FP8_MATMUL_OK` for `FP8_TEST=1` and `FP8_TEST=2`),
using the fragment mapping from CDNA4 ISA 7.1.3 on both the kernel and the host
side: `A[i][k]` in lane `i + 32 * (k / 32)` at byte `k % 32`, `B[k][j]` in lane
`j + 32 * (k / 32)`, lane `l`'s scale byte covering row `l % 32` and K block
`l / 32`, and `D[i][j]` in lane `j + 32 * ((i / 4) % 2)` at register
`4 * (i / 8) + (i % 4)`.

### Root cause found by bisection

`probe_fp8_min` (constant `Frag8` operands) passed all 64.0, while
`probe_fp8_minload` (same MFMA with the operands loaded from global memory)
failed, so the MFMA itself and the store side were fine. The emitted assembly
showed no `s_waitcnt vmcnt(0)` between the scale loads and the MFMA: `as_v1` and
`as_v2` were the only operand helpers that did not call `touch`, so a V1/V2 value
produced by a memory load was consumed without waiting (the scale registers read
0, which multiplied the whole result to zero, and in other layouts produced
garbage addresses). Both helpers now touch their operand, which also covers the
V2 A/B operands of the F16 MFMA shapes.

## Next: FMHA reference core loop

1. Resolve the 16 KB LDS question above (needed before scaling the core loop).
2. Build the paired A/B loop skeleton: two loops with rollback exits, four
   paired barriers per outer iteration, exactly as in the reference kernel.
3. Wire the data path into the loop: LDS staging -> `ds_read_b64/b128` -> MFMA
   accumulation (accumulator lives in the fragment registers).
4. Add the softmax pack instructions
   (`v_cvt_pknorm_u16_f32`, `v_cvt_scalef32_pk_fp8_bf16`) with their
   low/high write-mask semantics.
5. Compare the emitted instruction stream against the reference witness
   section by section.