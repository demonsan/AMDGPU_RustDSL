# Implementation status

Updated after the first end-to-end milestones on gfx950 (`mi355-gpu-53`).

## Verified working

| Step | Evidence |
|---|---|
| Rust toolchain only inside a container | `tools/container/Containerfile` (ubuntu 22.04 + rustup 1.83.0), built with rootless podman; host and shared home untouched |
| Workspace builds | `tools/container/cargo.sh build` → `amdc` binary, no warnings |
| Typed IR + verifier | `amdgpu-dsl`: SSA values, loop variables, scope/type checking |
| Lowering + own register allocation | `amdgpu-compiler`: MIR, linear-scan RA, copy coalescing, `s_waitcnt` insertion |
| Assembly emission | `amdc emit attention_fwd` → 1.1k lines of gfx950 asm, accepted by the ROCm assembler (`clang -mcpu=gfx950 -mcode-object-version=6`) |
| Code object loads and dispatches | `hipModuleLoad` + `hipModuleLaunchKernel` from `tests/gpu/attn_host.cpp` |
| ABI + memory path correct | `probe_store` kernel: `o[tid] = q[tid]` plus a kernarg scalar, `PROBE_OK` on hardware |

## Bugs found and fixed on the way

1. `ValueId`/`VarId` were both `u32` → conflicting `From` impls; `VarId` is now a newtype.
2. Nested DSL builder calls needed to be split into locals (Rust borrow rules).
3. Register allocator did not mark assigned slots as used → every virtual register got the same physical register.
4. Constant moves carried their value in `imm`, but the emitter only read `srcs` → **every constant was emitted as 0**.
5. `BreakIf` branch polarity was inverted.
6. `max_v`/`max_s` were computed from the occupancy bitmap *after* allocation, so reclaimed high slots were not declared; code used `v208` while metadata said 140 VGPRs.
7. Missing `.amdhsa_user_sgpr_count` → no kernarg pointer was passed.
8. Kernel descriptor was not 64-byte aligned; the runtime read every field shifted, which surfaced as "illegal instruction" / "memory aperture violation".
9. `kernarg_size` not 8-byte aligned → the assembler rejected the metadata.
10. Single-letter argument names (`n`) are parsed as YAML booleans → argument names are now quoted.
11. `hipModuleLaunchKernel` needs one pointer per argument, not one pointer to the whole kernarg struct.

## Open bug: `attention_fwd` numerics

The kernel launches and finishes (no faults) but produces wrong values:

* Failure is **independent of the data**: with `q = k = 0`, `v = 0.5`, `scale = 0` the output is still wrong (expected 0.5 everywhere).
* Failure is **independent of the key length** (`ATTN_SK=1` and `64` behave the same).
* Bad rows follow `tid % 8 < 4` (groups of four lanes), i.e. half the lanes are wrong; matches a hardware/resource-granularity effect or a violated implicit operand rule, not ordinary math.
* Row 0 element 0 reads `0.0`, the rest `±inf`, which points at a zeroed/lost normaliser rather than a rounding issue.

Reproduce with `tests/gpu/run.sh`, e.g.

```bash
ATTN_H=1 ATTN_S=64 ATTN_SK=64 ATTN_CONST=1 bash tests/gpu/run.sh
```

### Next steps

1. Add a `probe_softmax` kernel (single element per key) so the loop-carried
   accumulator/normaliser can be inspected on its own.
2. Check the implicit operand rules for the emitted instructions against the
   ISA again, especially `v_add_co_u32`/`v_addc_co_u32` pairing and the
   `global_load_*` address pair alignment, at the register counts this kernel
   needs (212 VGPRs).
3. If the lane pattern survives a kernel using far fewer registers, treat it as
   a descriptor/resource problem rather than an instruction problem.

## Not started

* 1:1 reproduction of the FMHA reference core loop (`docs/fmha-reference.md`).
* MFMA/PV tiling, LDS staging, asynchronous read contracts.
* MORI-based cross-card fusion (LSA, SDMA, GDA).
* gfx1250 Expert2 dependency scheduling.
* JIT adapters for DeepInfer / SGLang.
