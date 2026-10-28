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
| Wave-role branching (8 waves) | `probe_roles` marker phase: waves 0-3 store `111.0`, waves 4-7 store `222.0`, `marker_bad = 0` — DSL `if` + wave-id from `v_readfirstlane` verified on hardware |
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

## Open issue: multi-wave LDS exchange

`probe_lds` (one wave, 64 threads) passes on hardware, but the cross-wave exchange
probes (`probe_roles`, `probe_waves2`) read garbage back from LDS even for a
lane's own slot. What has been ruled out so far:

* probe arguments (the harness always passes one output pointer to probes),
* host buffer sizes and output indices,
* constant vs. per-lane data, presence of `v_readfirstlane`,
* one wave (block 64) vs. two (block 128) vs. eight waves (block 512),
* LDS declaration of 256 vs. 512 bytes, declared workgroup size 64 vs. 128,
* the `lgkmcnt` wait before consuming a `ds_read_b32` result (fixed, but not sufficient).

Next experiment: run the hand-written `tools/lds_probe.s` (assembled directly, no
DSL) as a multi-wave kernel. If that also fails, the problem is in how the code
object/descriptor is consumed; if it works, the difference to the generated
kernel is the bug.

## Next: FMHA reference core loop

The 1:1 reproduction target is the core loop of
`aiter/hsa/gfx950/fmha_v4_fwd/fwd_hd128_mxfp8.co` (facts in
[docs/fmha-reference.md](fmha-reference.md)). Landed so far: MFMA
(`v_mfma_f32_16x16x16_f16`, `32x32x8_f16`, `16x16x32_fp8_fp8`,
`32x32x16_fp8_fp8`), 4/16-dword accumulators, LDS read/write, `s_barrier` and
explicit `lgkmcnt` waits — every one of those verified on hardware.

Instruction forms confirmed to assemble on gfx950 (operand shapes measured, not
yet wired into the DSL): `v_mfma_f32_16x16x128_f8f6f4` and
`v_mfma_f32_32x32x64_f8f6f4` (A/B eight dwords), the 16-byte
`v_mfma_scale_f32_32x32x64_f8f6f4` with its two extra scale operands,
`v_max3_f32`, `v_cvt_pknorm_u16_f32`, `v_cvt_scalef32_pk_fp8_bf16`,
`ds_read_b64`, `ds_read_b64_tr_b8`, `ds_read_b128`, `s_waitcnt lgkmcnt(4/2/0)`.
`v_smfmac_f32_16x16x128_fp8_fp8` still needs its exact operand triple pinned
down before it can be used.

Still needed, in order:

1. The scaled fp8/fp4 MFMA forms (`v_mfma_scale_f32_32x32x64_f8f6f4`) and the
   wide `f8f6f4` shapes, including the 16-byte scaled encoding.
2. Vector LDS widths (`ds_read_b64`/`b128`, `ds_read_b64_tr_b8`) and
   automatic `lgkmcnt` accounting for partial readiness.
3. Softmax helpers as first-class ops: `v_max3_f32`, `v_cvt_pknorm_u16_f32`,
   `v_cvt_scalef32_pk_fp8_bf16` (with the low/high write-mask semantics).
4. Multi-wave blocks: role split by wave id and the paired-phase barriers.
5. Then the loop structure: A/B loop pairing, rollback exits, and the
   witness-based comparison against the reference instruction stream.

## Not started

* MFMA/PV tiling beyond the micro-kernels, LDS staging.
* MORI-based cross-card fusion (LSA, SDMA, GDA).
* gfx1250 Expert2 dependency scheduling.
* JIT adapters for DeepInfer / SGLang.
