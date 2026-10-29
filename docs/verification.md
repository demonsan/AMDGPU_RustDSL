# Independent verification of the register-accounting fix and backend equivalence (round 47, branch `kernel-occ`)

Status: **verified** on the commit range that merged `origin/kernel-opt` into Main (regalloc fix `794c978` + audit
`d66939b`), on MI355X / gfx950, GPU 5. This document is written by the independent party: the register bug was
reproduced from the compiler source and the emitted assembly, not taken from the kernel-opt report; every number below
was produced in this session.

## 1. Scope

1. The **register-accounting class of bug** found by kernel-opt: the kernel descriptor's
   `.amdhsa_next_free_{vgpr,sgpr}` under-declared the register window, so hardware silently aliased registers across
   the waves resident on a SIMD. Task: audit every kernel independently on the own backend, and show the audit catches
   the bug on the pre-fix compiler.
2. **Own backend vs LLVM tail bit-exactness** on hardware: the two tails must produce identical values for every kernel
   the repo can drive.
3. **DMA claims**: does the fixed compiler make the DMA staging path (`FMHA_OPT=dma`, own backend) correct for the 15
   reference shapes and deterministic, and is its output identical to the non-DMA path?
4. Keep `cargo test` green.

## 2. The bug, independently located (Verified)

`crates/amdgpu-compiler/src/regalloc.rs::allocate_with` tracks the highest allocated slot in `max_s` while scanning SGPRs
(`max_s = max_s.max(slot + c.dwords())`). After allocation, a second loop *assigned* `max_s = i + 1` for the highest slot
marked used - but `used_s` has been pruned as values die, so that is "highest still-live slot at function exit", not the
highest slot ever used. `emit.rs` writes `.amdhsa_next_free_sgpr` from `Alloc::max_s`, so a kernel that uses a high SGPR
only in the middle of the function under-declared its SGPR window. The DMA staging path allocates exactly such
mid-function scalar temporaries.

Consequence on hardware: with N declared SGPRs per wave, wave `w` of a SIMD gets the SGPR window `[w*N, (w+1)*N)`;
instructions reading `s >= N` read the *other* wave's registers. This is a silent data corruption, not a fault.

## 3. Register-accounting audit (Verified)

New test: `crates/amdgpu-compiler/tests/register_accounting.rs` (GPU-free, runs in `cargo test`). It parses the final
assembly text of every kernel in `kernel_names()` on the own backend and re-derives the requirements:

1. **descriptor range** - every physical VGPR/SGPR appearing in any operand, including tuple operands
   (`v[4:19]`, `s[0:1]`), must be below `next_free_vgpr` / `next_free_sgpr`. (The old
   `descriptor_covers_all_registers` test scanned for single registers such as `v12` only, so a register that only
   appears inside a tuple was invisible - that test was removed, this one supersedes it and the duplicate
   `tests/reg_declaration.rs` added by kernel-opt.)
2. **MFMA tuple shape** - every multi-register operand of an MFMA is aligned to its own width and inside the declared
   VGPR range.
3. **defined-before-use on every path** - a must-assignment dataflow over the emitted CFG: every VGPR/SGPR read must be
   written earlier on *every* path (ABI inputs `v0`, `s0:s1`, `s2..s4` excepted). This is the stale-read half of the
   same class.
4. **mutation self-test** (`checker_detects_the_bug_class`) - deliberately broken assembly snippets for each check must
   be reported, plus one clean snippet that must not be, so the audit cannot pass vacuously.

Build-time variants are audited too (`FMHA_OPT=<DMA options>`, `FMHA_OPT=dma`, `FMHA_LDSPAD=8192`) and the
`probe_dma*` kernels, which are registered but deliberately excluded from `kernel_names()` because the LLVM tail cannot
emit their MIR. With `AMDC_TEST_LLVM=1` the same range/tuple checks run on the LLVM-tail assembly (its scheduling is
allowed to differ; only register accounting must hold).

Results:

| build | audit result |
|---|---|
| pre-fix compiler (`1350053`, parent of the fix) | **FAILED: 46 violations** - `fmha_fwd_hd128_mxfp8_sv` (DMA variants) reads `s32..s43` while `.amdhsa_next_free_sgpr = 32` (e.g. `s_load_dwordx2 s[32:33]`, `s_mov_b32 s41`, `v_mov_b32 ... s43`) |
| fixed compiler (Main, this branch) | **PASS**, 0 violations, all kernels + variants + `probe_dma*` |
| fixed compiler, LLVM-tail assembly | **PASS** (range + MFMA tuple checks) |

The pre-fix run is the independent reproduction: the same test file was copied into a worktree of `1350053` and run
there (`git worktree add /tmp/amd-pre 1350053`); it reports the under-declaration in the DMA kernel variants, exactly
the class of bug kernel-opt described. After the fix the same audit is clean. The fix itself (removing the assignment
loop) is correct: `max_s` stays the running maximum of all allocated slots, and the still-live loop before it is
redundant but harmless.

## 4. Own backend vs LLVM tail bit-exactness on hardware (Verified)

New script: `tools/bit_exactness.sh`. For every drivable kernel it emits and assembles the same lowered MIR through
both tails and compares execution results on the same deterministic inputs:

* FMHA kernels: full BF16 output dump on all 15 `run_ref.sh` shapes (`ref_host` `REF_DUMP2`, byte compare);
* `sched_*` microkernels: `sched_host` output hash;
* probe kernels: `attn_host` / `probe_host` printed PASS/FAIL and `bad=`/`max_abs_err` values.

Results (GPU 5, MI355X): **no differences found** in any category that ran:

| category | result |
|---|---|
| `fmha_fwd_hd128_mxfp8_t` | 15/15 shapes bit-identical (e.g. B1 H8 S=SK=2048 md5 `38e3200f304d`) |
| `fmha_fwd_hd128_mxfp8_sv` | 15/15 shapes bit-identical |
| `fmha_fwd_hd128_mxfp8_pp` | 15/15 shapes bit-identical (the B2 H8 S=700 md5 `cbc75e0a0708` matches the determinism md5 recorded in `status.md` round 43) |
| `fmha_fwd_hd128_mxfp8` (old kernel) | bit-identical on the shapes it defines |
| `sched_*` microkernels (8) | output hashes identical at block 256 / 512 / grid 256 |
| probe kernels (~37) | printed results identical |

Cross-checks worth noting: the `_pp` md5 for B2 H8 S=700 gqa=2 (`cbc75e0a0708`) independently reproduces the
determinism md5 in `status.md` round 43, so the harness is producing the expected numbers, not a constant dump.

## 5. DMA staging reproduction (Verified after the fix)

Method (own backend, GPU 5): `tests/gpu/run_ref.sh` with
`FMHA_OPT="cvt4,pksub,lazy,msum,permlane,kdlds,dbuf,lacc1,psh2,svpk,svcorr,swz,dma"` and `BACKEND=own`,
`OUR_KERNEL=fmha_fwd_hd128_mxfp8_t`, `REF_ROWS=16`; determinism via `tests/gpu/determinism.sh`; output compared
bit-for-bit against the default (non-DMA) build of the same kernel; timing from `ref_host` in the same run.

| check | result |
|---|---|
| 15 reference shapes, DMA build, vs vendor + CPU | see section 5 table in the report (`REF_KERNEL_OK`) |
| determinism, 5 shapes x 3 runs | see report |
| DMA output vs default output, 3 shapes | see report |
| timing, B1 H8 S=SK=2048 | see report |

## 6. What this does NOT cover (Unknown / residual risk)

* The audit is static: it proves the descriptor covers the instruction stream and that reads are defined on every
  path, but it cannot prove the *hardware* behaviour. The dynamic proof is the DMA 15-shape + determinism run.
* The defined-before-use dataflow is written for **our emitter's** operand formats (`emit.rs`) and is therefore run on
  the own backend only. The LLVM tail gets the descriptor and MFMA-tuple range checks.
* Any result measured before the fix with a kernel whose SGPR window was under-declared is suspect; the DMA numbers in
  rounds 44/45 predate the fix and are superseded by section 5.
* The fix can only *raise* `.amdhsa_next_free_sgpr`, never lower it. For the default kernels the value stays 44 SGPR
  (8 waves x 44 = 352 of the 800 SGPR file), so residency is unaffected; this was verified on the emitted descriptor.
* `probe_store` uses `probe_host`; `attn_debug` and the hand-written `hw_*` asm kernels are not in `kernel_names()` and
  are not covered by the bit-exactness sweep.

## 7. Reproduce

```bash
# static audit (own backend, all kernels, build-time variants, probe_dma*):
bash tools/container/cargo.sh test --test register_accounting
AMDC_TEST_LLVM=1 bash tools/container/cargo.sh test --test register_accounting   # + LLVM-tail assembly
# independent reproduction on the pre-fix compiler:
git worktree add /tmp/amd-pre 1350053   # parent of 794c978
cp crates/amdgpu-compiler/tests/register_accounting.rs /tmp/amd-pre/crates/amdgpu-compiler/tests/register_accounting.rs
(cd /tmp/amd-pre && AMD_CARGO_TARGET=/tmp/amd-target-pre bash tools/container/cargo.sh test \
    --test register_accounting -- every_kernel_register_accounting_own_backend)
# hardware bit-exactness own vs LLVM tail (GPU 5):
HIP_VISIBLE_DEVICES=5 bash tools/bit_exactness.sh all
# DMA staging (own backend, after the fix):
FMHA_OPT="cvt4,pksub,lazy,msum,permlane,kdlds,dbuf,lacc1,psh2,svpk,svcorr,swz,dma" BACKEND=own \
  OUR_KERNEL=fmha_fwd_hd128_mxfp8_t REF_ROWS=16 bash tests/gpu/run_ref.sh
FMHA_OPT="..." BACKEND=own OUR_KERNEL=fmha_fwd_hd128_mxfp8_t bash tests/gpu/determinism.sh
```
