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
| FMHA data path | `probe_fmha_datapath`: `ds_write_b128` → `ds_read_b128` → `Frag8` (`frag4` concatenation) → `v_mfma_scale_f32_32x32x64_f8f6f4` → every one of 2048 values is `64.0` → `PROBE_FMHA_DATAPATH_OK` |
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

## Open question: 32 bytes per lane (16 KB LDS) probe

A variant of `probe_fmha_datapath` that stages 32 B per lane (16 KB of LDS with
512 threads, two `ds_read_b128` feeds) produced wrong MFMA results for lanes
with `tid >= 388` (`57.5` and `955.0` instead of `64.0`), while the 16 B/lane
(8 KB) version is exact. The MFMA and the read path are identical, so this is
either an LDS-capacity limit for this launch shape or a register/aliasing issue
in the two-fragment concatenation. Worth isolating before the FMHA core loop,
which needs >= 8 KB and the reference kernel uses ~84 KB.

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