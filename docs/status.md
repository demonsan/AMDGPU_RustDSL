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

## Open issue: probe_fmha_core reads zero

`probe_fmha_core` (A phase / B phase staging, three paired barriers per
iteration, in-place MFMA accumulation) currently stores `0.0` in every element,
while `probe_fmha_datapath` (same staging, no loop) and `probe_loop_mfma` (loop +
accumulation, one staging phase) both pass. The emitted sequence is

```
BB1: ds_write_b128 v3, v[4:7] ; bar ; ds_write_b128 v8, v[4:7] ; bar
     ds_read_b128 v[12:15], v3 ; ds_read_b128 v[32:35], v8
     ... Frag8 build ...
     v_mfma_scale_f32_32x32x64_f8f6f4 v[64:79], v[40:47], v[48:55], v[64:79], v12, v12
```

so the operands are built from the LDS reads, but the result is zero. Next step is
to store the built Frag8 lane values to global memory inside the loop to see
whether the operands or the MFMA are zero.

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