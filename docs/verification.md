# Independent verification: register accounting, backend equivalence, DMA (round 47, branch `kernel-occ`)

Author: `kernel-occ`, the independent party. Everything below was produced in this session on MI355X / gfx950
(GPU 5, `aac16` node 53) from the compiler source and the emitted assembly/hardware; the register bug was located
from `regalloc.rs` and the assembly before reading any teammate report. Base commit: `origin/Main` = `ae003a9`
(merges `kernel-opt` incl. the regalloc fix `794c978`, the new DMA staging `68d03aa`, the dead-load DCE fix
`7934e4e`; round-46 docs `d6f2d9d`).

## 1. Scope

1. **Register-accounting bug class** (found by kernel-opt): the kernel descriptor's
   `.amdhsa_next_free_{vgpr,sgpr}` under-declared the per-wave register window, so hardware silently aliased
   registers between the waves resident on a SIMD. Independent audit of *every* kernel + independent reproduction on
   the pre-fix compiler.
2. **Own backend vs LLVM tail bit-exactness** on hardware for every kernel the repo can drive.
3. **DMA staging** (`FMHA_OPT=...,dma`, own backend): 15 reference shapes, determinism, output equivalence vs the
   default build, timing, and the option-string sensitivity.
4. Keep `cargo test` green.

## 2. The bug, located independently (Verified)

`crates/amdgpu-compiler/src/regalloc.rs::allocate_with` tracks the highest allocated slot in `max_s` while scanning
SGPRs (`max_s = max_s.max(slot + c.dwords())`). After allocation, a second loop **assigned**
`max_s = i + 1` for the highest slot marked used - but `used_s` is pruned as values die, so this is "highest
still-live slot at function exit", not the highest slot ever used. `emit.rs` writes
`.amdhsa_next_free_sgpr` from `Alloc::max_s`. A kernel that uses a high SGPR only in the middle of the function
therefore under-declares its SGPR window; with N declared SGPRs per wave the wave gets `[w*N, (w+1)*N)` and any
access to `s >= N` reads the *other* wave's registers. Silent data corruption, no fault. The DMA staging variant
allocates exactly such mid-function scalar temporaries, which is why only that build exposed it.

## 3. Register-accounting audit (Verified)

`crates/amdgpu-compiler/tests/register_accounting.rs` (GPU-free, runs in `cargo test`; this is the canonical test
and **supersedes the duplicate `tests/reg_declaration.rs`**, which is deleted on this branch). It parses the final
assembly text of every kernel in `kernel_names()` on the own backend and re-derives the requirements:

1. **descriptor range** - every physical VGPR/SGPR index in any operand, tuple operands (`v[4:19]`, `s[0:1]`)
   included, must be below `next_free_vgpr` / `next_free_sgpr`. The old test only understood single-register
   operands such as `v12`, so a register used only inside a tuple was invisible to it.
2. **MFMA tuple shape** - every multi-register operand of an MFMA is aligned to its width and inside the declared
   VGPR range (the post-RA check in `regalloc.rs` checks the same on MIR; this checks the emitted text).
3. **defined-before-use on every path** - must-assignment dataflow over the emitted CFG: every VGPR/SGPR read must
   be written earlier on *every* path from entry (ABI inputs `v0`, `s0:s1`, `s2..s4` excepted). This is the stale-
   read half of the same class.
4. **mutation self-test** (`checker_detects_the_bug_class`) - deliberately broken snippets for each check must be
   reported and one clean snippet must not be, so the audit cannot pass vacuously.

Coverage: all `kernel_names()`, the build-time variants (`FMHA_OPT=<DMA options>`, `FMHA_OPT=dma`,
`FMHA_LDSPAD=8192`) and the `probe_dma*` kernels (registered but not in `kernel_names()` because the LLVM tail
cannot emit their MIR). With `AMDC_TEST_LLVM=1` the descriptor/tuple checks also run on the LLVM-tail assembly.

Results:

| build | audit result |
|---|---|
| **pre-fix compiler `1350053`** (parent of `794c978`), same test file copied into a worktree | **FAILED: 46 violations** - `fmha_fwd_hd128_mxfp8_sv` DMA variants read `s32..s43` while `.amdhsa_next_free_sgpr = 32` (e.g. `s_load_dwordx2 s[32:33]`, `s_mov_b32 s41`, `v_mov_b32 ... s43`) |
| fixed Main (`ae003a9`, this branch) | **PASS**, 0 violations, all kernels + variants + `probe_dma*` |
| fixed Main, LLVM-tail assembly | **PASS** (descriptor range + MFMA tuple checks) |

The pre-fix failure is the independent reproduction: the identical test catches the under-declaration that
corrupted the DMA kernel's head-1 blocks. The fix (removing the assignment loop) is correct: `max_s` stays the
running maximum of all allocated slots.

Descriptor budget after the fix (own backend, default build): `fmha_fwd_hd128_mxfp8_t` vgpr=256 sgpr=48,
`_sv` 256/48, `_pp` 252/44, old `fmha_fwd_hd128_mxfp8` 176/43. Eight waves x 48 SGPR = 384 of the 800 SGPRs per CU,
so residency is unchanged (measured 292.96 TFLOP/s for `_t` on llvm-tail, 286.84 own, vendor 515.87 in the same run,
B1 H8 S=SK=2048).

## 4. Own backend vs LLVM tail bit-exactness on hardware (Verified)

`tools/bit_exactness.sh` emits and assembles the same lowered MIR through both tails and compares execution results
on deterministic inputs: full BF16 output dumps on the 15 `run_ref.sh` shapes for the FMHA kernels, output hashes
for the `sched_*` microkernels, printed results for the probe kernels.

| category | result |
|---|---|
| `fmha_fwd_hd128_mxfp8_t` / `_sv` / `_pp` | **15/15 shapes bit-identical each (60/60)**; e.g. `_t` B1H8S2048 md5 `38e3200f304d`; `_pp` B2H8S700 gqa=2 md5 `cbc75e0a0708` matches the determinism md5 in `status.md` round 43 |
| `fmha_fwd_hd128_mxfp8` (old kernel) | bit-identical on its 13 in-spec shapes; see the nondeterminism note below for the other 2 |
| `sched_*` microkernels (8) | output hashes identical at block 256 / 512 / grid 256 |
| probe kernels | 20 pass on both tails and print identical results; see below for the FAIL cases and the `probe_store` red flag |

**Deterministic vs nondeterministic kernels (documented limitation).** Several A/B "diffs" in this experiment were
run-to-run nondeterminism in the *kernel*, not a tail difference; future comparisons must not treat them as tail
evidence:

* `fmha_fwd_hd128_mxfp8` (the **old, non-default** kernel) on the two out-of-spec shapes of the 15-shape set
  (`B=2 H=8 S=700 SK=640 gqa=2` and `B=2 H=4 S=300 SK=129`; the kernel documents S multiple of 256, kv_seq_len
  multiple of 128, gqa=1) produces a different dump on every run on **both** tails (4 distinct md5 in 4 runs). Its
  13 in-spec shapes are deterministic and tail-identical.
* Probe kernels: `probe_roles_loop` produced 4 distinct outputs in 4 runs (both tails); `probe_fp8_fmha_trab` is
  stable on own but one of two LLVM-tail runs differed; `probe_waves2` and `probe_ab_loops` showed one earlier diff
  and were stable 4/4 on re-run. These probes also FAIL in both tails with the harness default `ATTN_BLOCK=64`
  (pre-existing harness configuration, not tail evidence).
* Everything else measured is deterministic and identical: the three FMHA default kernels (all 15 shapes), the 8
  `sched_*` kernels, and the 20 passing probes.

**RED FLAG (found, fix handed to backend-llvm-tail): the LLVM-tail `probe_store` code object crashes the host at
`hipModuleLaunchKernel` (SIGSEGV); the own-backend object prints `PROBE_OK`.** Root cause: the LLVM tail lists
kernarg *layout padding* as kernel arguments. `probe_store` has a 20-byte argument area rounded to 24 bytes; the
exported IR adds a dummy `%pad0` parameter (`crates/amdgpu-compiler/src/backend/llvm_mir.rs::kernarg_ir`), which
llc turns into a fourth HSA argument (`pad0`, offset 20, by_value). HIP's kernel-parameter launch API then reads a
fourth parameter from a three-entry array (out of bounds). The own metadata has only the three real arguments and
works; patching the descriptor fields does not change the crash. Scope: any LLVM-tail kernel whose DSL kernarg
layout has internal gaps or trailing padding when launched via the `kernelParams` API (`ref_host`/`sched_host` use
the flat-buffer API and are unaffected; `attention_fwd`'s 8 contiguous args are unaffected). Fix: do not emit pad
parameters as HSA args (strip `padN` from the generated `.amdgpu_metadata` `.args` list; `kernarg_segment_size`
already includes the padding).

## 5. DMA staging on the new implementation (Verified)

New in Main: `68d03aa t kernel: DMA staging (FMHA_OPT=dma, own backend)` + round-46 fixes. Method (own backend,
GPU 5, same session): `FMHA_OPT=cvt4,pksub,lazy,msum,permlane,kdlds,dbuf,lacc1,psh2,dma` (the option string used
for kernel-opt's verification), `OUR_KERNEL=fmha_fwd_hd128_mxfp8_t`, `tests/gpu/run_ref.sh` with `REF_ROWS=16`,
`tests/gpu/determinism.sh`, and a same-run comparison against the default build.

| check | result |
|---|---|
| 15 reference shapes vs vendor kernel + CPU | **15/15 `REF_KERNEL_OK`**, mean_err 1.07e-3 - 1.95e-3 vs vendor 1.50e-3 - 2.83e-3, `bad=0` everywhere |
| determinism, 5 shapes x 3 runs | **5/5 `DET_OK`** (identical md5 three runs per shape) |
| DMA output vs default-build output, 3 shapes (B1H8S2048, B2H4S1024, B2H8S700 gqa=2) | **bit-identical** on all 3 |
| timing B1 H8 S=SK=2048 (same run, 30 iters) | default own **287.14**, DMA own **287.93**, vendor **516.24/516.34** TFLOP/s - DMA is not faster with the default scheduler, matching round 46 (288.5 vs 287-288) |
| option-string sensitivity, benchmark shape | exact string, exact+`svpk` and bare `dma` all `REF_KERNEL_OK` (mean_err 1.25e-3 / 1.25e-3 / 1.07e-3) - **no option-string sensitivity on the new implementation** |

Historical note (superseded): **before** `68d03aa`, i.e. on Main at `7237b6f`/`9b269c1` and on kernel-opt's
experimental branch `1350053`, the DMA build produced NaN on 14/15 shapes, was nondeterministic, and failed with
every option string/scheduler tried (including the string above). That was the stale DMA implementation, not the
regalloc bug; the regalloc fix alone (verified in section 3) predates the new DMA code.

## 6. Not covered / residual risk

* The audit is static; dynamic proof comes from the 15-shape + determinism run. The defined-before-use dataflow is
  written for our emitter's operand formats and runs on the own backend only; the LLVM tail gets descriptor/tuple
  checks (plus `tail_invariance.rs`).
* Results measured with an under-declared SGPR window (before `794c978`) are suspect for kernels whose high SGPRs
  were only live mid-function; the DMA numbers in rounds 44/45 predate both fixes and are superseded.
* `probe_store` remains broken on the LLVM tail until the pad-argument metadata fix lands (section 4).
* The old `fmha_fwd_hd128_mxfp8` kernel is nondeterministic on out-of-spec inputs (section 4); do not use it for
  tail comparisons outside its documented requirements.
* `attn_debug` and the hand-written `hw_*` asm kernels are outside the bit-exactness sweep (`probe_store` is
  covered as the red-flag repro).

## 7. Reproduce

```bash
# static audit (own backend, all kernels, build-time variants, probe_dma*):
bash tools/container/cargo.sh test --test register_accounting
AMDC_TEST_LLVM=1 bash tools/container/cargo.sh test --test register_accounting   # + LLVM-tail assembly
# independent reproduction on the pre-fix compiler (parent of 794c978):
git worktree add /tmp/amd-pre 1350053
cp crates/amdgpu-compiler/tests/register_accounting.rs /tmp/amd-pre/crates/amdgpu-compiler/tests/register_accounting.rs
(cd /tmp/amd-pre && mkdir -p build && AMD_CARGO_TARGET=/tmp/amd-target-pre bash tools/container/cargo.sh test \
    --test register_accounting -- every_kernel_register_accounting_own_backend)     # 46 violations
# hardware bit-exactness own vs LLVM tail (GPU 5):
HIP_VISIBLE_DEVICES=5 bash tools/bit_exactness.sh all
# DMA staging on the new implementation (own backend):
FMHA_OPT="cvt4,pksub,lazy,msum,permlane,kdlds,dbuf,lacc1,psh2,dma" REF_CO=<vendor.co> BACKEND=own \
  OUR_KERNEL=fmha_fwd_hd128_mxfp8_t REF_ROWS=16 bash tests/gpu/run_ref.sh
FMHA_OPT="..." REF_CO=<vendor.co> BACKEND=own OUR_KERNEL=fmha_fwd_hd128_mxfp8_t bash tests/gpu/determinism.sh
```

## Loop-variable semantics and the re-read lint

A *loop variable* (`begin_loop` / `set_var`, `Opnd::Var`) is loop-carried state. Contract: a read of a variable **before** the first `set_var` of that variable in an iteration sees the value the previous iteration left (the initial value in the first iteration). A read **after** a `set_var` of the same variable in the same iteration is **not part of the contract**: our lowering keeps the variable in one mutable register and often retargets the producer of the `set_var` operand straight into it, so such a read usually sees the newly written value, but that depends on the producer, on in-place `Var*` statements, on nested `if`s and on instruction movement, and kernel-opt measured stale data (39% wrong rows, varying from run to run) in one such pattern. For an intra-iteration producer/consumer pair use the plain SSA value that was passed to `set_var`.
`amdgpu_dsl::lint(&Function)` reports every such read conservatively (a `set_var` inside an `if` counts); `amdc emit` prints the reports as warnings; `FnBuilder::allow_var_rereads()` opts a kernel out. Cross-iteration reads (read before the set) and loop conditions expressed on SSA values are not reported. `tests/var_lint.rs`: the lint precision test, and a guard that only the kernels listed in `KNOWN_REREADERS` (fmha_*, sched_diamond; plus the `probe_*` research kernels) re-read after set - any other kernel must be clean.