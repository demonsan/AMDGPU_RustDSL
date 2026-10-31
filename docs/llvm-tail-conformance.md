# LLVM tail conformance (pinned LLVM: ROCm 10.1.0, AMD LLVM 24.0.0git)

> **Current status (final).** Pinned LLVM: ROCm 10.1.0 / AMD LLVM 24.0.0git (dialect `llvm24`, default; `llc` built by `tools/llvm_build_llc.sh`); the `llvm20` dialect (ROCm 7.0.2) is also supported.
> `tail_invariance` guarantees that the llvm-tail and own backends emit the same instruction streams modulo nops / waits / `delay_alu`. `cargo test`: 111 tests / 28 targets green
> (numbers as in `docs/current-status.md`). 101 MOps have rows in both `llvm20.tbl` and `llvm24.tbl`; the DMA, LDS-direct and `buffer_load ... lds` forms are own-only (kernels using them auto-select the own backend).
> Known limitations: (1) the `llvm20.tbl` rows are copies of the `llvm24.tbl` rows and are machine-verified only by the llvm24 `llc`; (2) the hazard-oracle test skips the S4 buffer-descriptor ops (a 128-bit SGPR live-in
> is not expressible in its per-block live-ins), which are covered by the one-row-per-op machine-verifier test and the hardware probes. Design context: `docs/design.md` §2 and §8.

Question: does "Rust RA + Rust scheduler -> LLVM tail passes" pass LLVM's existing hazard / waitcnt /
pre-emit compiler tests? Evidence grades: **Verified** (measured here), **Experimental** (observed, not root-caused),
**Unknown**. LLVM source is never modified.

## 1. Pin and test infrastructure (Verified)

* Pin: ROCm 10.1.0 (`/shared/apps/ubuntu/opt/rocm-10.1.0`, ROCm/llvm-project `d6f6cb691863e9258034ab48c1dfce21689d5606`,
  `AMD LLVM version 24.0.0git`). Recorded in `backend.lock` (dialect `llvm24`); the previous pin (ROCm 7.0.2, `llvm20`) in `backend.llvm20.lock`.
  Both dialects live in the tree (`backend/llvm20.*`, `backend/llvm24.*`), selected by the probed `llc --version` (override `AMDC_DIALECT`).
* ROCm 10.0.0 / 10.1.0 / 10.1.0-20260902 ship **no `llc`** (only `libLLVM.so`, FileCheck, not, count, split-file, opt, llvm-mc, ...) and no llvm-lit.
  `tools/llvm_build_llc.sh <rocm> <llvm-project> <out>` compiles `llvm/tools/llc/{llc,NewPMDriver}.cpp` of the SAME commit against the installed headers and links
  the installed `libLLVM.so`: 2 TUs, seconds. Result: `/shared/amdgpu/home/blyu_qle/llvm-tools/rocm-10.1.0/bin` (llc + symlinks to all ROCm 10.1 tools); this is the default tool dir of `amdc`.
* Sources: `git init; git remote add origin https://github.com/ROCm/llvm-project.git; git fetch --depth 1 origin <sha>`, sparse checkout `/llvm/ /cmake/ /third-party/ /libc/`
  into `/shared/amdgpu/home/blyu_qle/llvm-src/rocm-10.1.0`.
* lit: `cmake -G Ninja llvm -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" -DLLVM_INCLUDE_TESTS=ON -DLLVM_ENABLE_ASSERTIONS=OFF` (configure only, nothing built) in
  `.../llvm-src/build-rocm-10.1.0` generates `test/lit.site.cfg.py`; `llvm_tools_dir` is patched to the tool dir above and `llvm_shlib_dir` to `<rocm>/lib/llvm/lib`.
  Run: `python3 llvm/utils/lit/lit.py -j8 -q -o res.json <build>/test/CodeGen/AMDGPU` (no psutil => no `--timeout`).
* Docker/podman: `tools/container/cargo.sh` accepts `AMD_ROCM_MOUNT="<rocm> <tools dir>"` and `AMD_LLVM_BIN`; GPU scripts take `ASM_ROC` (assembler tree) and `BACKEND`.

### Baseline of the pinned LLVM itself (test/CodeGen/AMDGPU, 4996 tests, 78 s with 8 workers) - Verified
PASS 4923, UNSUPPORTED 64 (non-assert build), XFAIL 7, FAIL 2 (`amdgpu-llvm-debuginfo-analyzer.ll`: needs a tool we do not have; `lto-lower-module-lds.ll`), both unrelated to the tail.
Raw results: `/shared/amdgpu/home/blyu_qle/llvm-src/results/baseline-10.1.0.json`.

## 2. Definition of conformance

LLVM's CHECK lines encode LLVM's own RA / scheduler output, so they cannot be matched for tests that start from virtual registers. We define, per test class:

**(A) Post-RA MIR tests whose RUN line exercises a tail pass** (`post-RA-hazard-rec, si-insert-waitcnts, amdgpu-insert-delay-alu, amdgpu-wait-sgpr-hazards,
si-memory-legalizer, si-pre-emit-peephole, si-insert-hard-clauses, si-mode-register, amdgpu-lower-vgpr-encoding, gcn-create-vopd, branch-relaxation, si-late-branch-lowering`):
the importer (`backend/mir_roundtrip.rs`, `amdc mir-roundtrip`) parses every `body: |` section into structured instructions (defs, instruction flags, opcode, operands with register flags,
verbatim suffix = MMOs / debug locations, BUNDLE children; unparsable lines stay verbatim and are counted `opaque`), prints them back at an information level, and the test is run again.
Levels:
* `identity` - structure round trip, everything kept. Pass = the unchanged RUN + FileCheck passes.
* `flags` - additionally drop the register flags our exporter never writes (`killed dead renamable internal`; `undef` is kept, it states intent).
* `all` - additionally drop instruction flags, memory operands and debug locations = the information level of the MIR our exporter writes.
For `flags`/`all` the original CHECK lines are not expected to hold (they name the flags), so the pass criterion is **differential**: `tools/filecheck_diff.py` replaces FileCheck, records the
checked output of the unmodified test, and requires the normalised output (flags/MMOs/debug locs/liveins/comments removed) of the transformed test to be identical, i.e. LLVM's tail makes the same insertions
(nops, waits, folds) when it is fed only what our exporter provides.

**(B) Our scheduler.** The importer produces raw MIR; our scheduler operates on `MFunction` (our MOps) and cannot move arbitrary LLVM opcodes, so a "scheduler level" on LLVM's tests is **not defined and not run**.
The scheduler is covered where it can be: kernels exported with `AMDC_SCHED=1` (`swp`, `taildup`) go through (C) and the hardware regression (section 5).

**(C) Pre-RA MIR tests (virtual registers) and the real path.** Their CHECK lines cannot be satisfied by any other RA, and our RA/scheduler consume only DSL kernels (our MOps), not arbitrary MIR.
What is checkable is the invariants of OUR path on all 48 kernels (41 shipped + `sched_*` micro kernels): the exported MIR passes `-verify-machineinstrs` through the tail, and the tail is idempotent
(a second `llc -start-before=si-insert-waitcnts -stop-after=amdgpu-insert-delay-alu` over the post-tail MIR changes nothing). "Pass" there means exactly these two properties, nothing about LLVM's CHECK lines.

**Bottom line on the instruction "passes all of LLVM's existing hazard tests":** met in the strongest form that is well-defined for class (A) at the `identity` level (every selected test passes with the original CHECK lines), met
in the differential sense for the `flags` level except 5 tests, partially for `all`; NOT satisfiable by definition for pre-RA tests and for tests of unsupported targets (gfx6-gfx12 variants the DSL never emits). Details below.

## 3. Results (tools/llvm_conformance.sh all; Verified)

Census of `test/CodeGen/AMDGPU/*.mir`: 838 files = 490 pre-RA (virtual registers) + 348 post-RA. Tail-pass RUN lines: 172 post-RA + 6 pre-RA = **178 selected tests**.

| level | pass criterion | PASS / 178 | importable (169: bodies>0, no opaque line) PASS |
|---|---|---|---|
| baseline | original | 178 | - |
| identity | original RUN+CHECK | **178** | 169 / 169 |
| identity | differential | **178** | 169 / 169 |
| flags | differential | **173** | 164 / 169 |
| all | differential | **140** | 131 / 169 |

Importer coverage: 10722 instructions, 10703 parsed (19 opaque lines kept verbatim: INLINEASM etc.), 51 bundles round-tripped, 23 lines changed by printing (whitespace normalisation only; tests still pass).

### Failures grouped by cause

`flags` (5):
* kill/dead liveness flags missing -> an LLVM peephole is skipped (conservative, semantically valid): `hazard-pass-ordering` (pre-emit peephole does not fold `s_mov/s_andn2/s_cbranch_vccnz`, the `s_nop 0` differs), `insert-skip-from-vcc`, `si-pre-emit-peephole-preserve-loop-info` (3). Our exporter does not write `killed/dead`; implementing them from the liveness we already compute is a candidate improvement (Experimental: the verifier would check the flags). No kernel of ours has the `s_and vcc,exec,x; s_cbranch_vccz` shape.
* verifier "Using an undefined physical register" when `killed`/`dead` are dropped: `waitcnt`, `waitcnt-agpr` (2; Experimental, not root-caused: the tests read registers the verifier only accepts with the original flags).

`all` adds 33 more:
* 19 "missing memory operand from image instruction": image (MIMG) opcodes need an MMO; the DSL emits no image instructions (not applicable).
* 10 conservative wait differences: 7 `memory-legalizer-*` / `wait-before-stores-with-scope_sys` (atomic orderings and nontemporal bits live in MMOs: our exporter writes address-space/size only), `lds-dma-waitcnt` (LDS-DMA is recognised through the MMO), `wait-xcnt`, `waitcnt-back-edge-loop` (without MMOs LLVM waits on both counters). Real limitation as soon as the DSL gets atomics / LDS-DMA: the exporter must then emit atomic ordering, syncscope and the LDS-DMA marker in the MMO.
* 1 parse error `branch-relaxation-debug-info` (removing the `debug-location` suffix breaks the test's own `DBG_` operand syntax: importer artefact), 1 `memory-legalizer-invalid-addrspace` (test expects errors keyed to MMOs), 1 `gfx11-sgpr-hazard-latency` (its RUN line starts at `machine-scheduler`: flags change LLVM's scheduling decisions, not our path), 1 `promote-constOffset-to-imm-gfx12` (cpol bits from MMO).

No failure was caused by an operand-kind or bundle limitation of the importer (all 51 bundles round-trip); no failure is attributable to our scheduler (not run, see B).

### Opcode table vs LLVM's own MIR (`amdc mir-table-report`, 10754 instructions of the selected tests)
matched exactly 4395 (46 table opcodes seen); 453 explicit operands match but the implicit-operand set differs (target variants: `$m0` for gfx6 DS, `$vcc_lo`, `$flat_scr`, `$mode` on other generations;
our rows are the gfx950 forms that the pinned llc accepts, validated by the opcode probe); 154 explicit shape differs (hard-coded `$sgpr0_sgpr1` kernarg base, gfx10 `$vcc_lo` carry forms); 5752 instructions use opcodes our table does not contain (expected: the table only has what the DSL emits).

### Exporter-side invariants on our own kernels (Verified)
48 kernels: `-verify-machineinstrs` clean and tail idempotent, with `AMDC_SCHED` off and with `AMDC_SCHED=1 AMDC_SCHED_OPTS=swp`.

## 4. Not covered, and why
* Pre-RA MIR tests (490 files; 6 with tail-pass RUN lines): CHECK lines encode LLVM's RA by construction; our RA cannot ingest them (no MOp mapping). Only (C) applies.
* `.ll` tests (about 7 mention tail passes): need LLVM's whole pipeline.
* Tests for gfx6-gfx12 variants inside the 178: they run (they are in the identity/differential numbers) but exercise hazards the DSL can never produce.
* A scheduler level on raw MIR (B). Unknown: how much of LLVM's hazard-rule coverage is reachable from gfx950 DSL code; `tools/hazard_oracle` (pairwise matrix) answers that question for the rules that matter, from the other direction.

## 5. Hardware with the ROCm 10.1 tail (GPU 4, Verified)
`BACKEND=llvm-tail ASM_ROC=<rocm-10.1.0>`: `cargo test` 33 passed; `attention_fwd` ATTENTION_FWD OK; fmha_t 15/15 REF_KERNEL_OK and determinism 5/5 DET_OK with no scheduler, with `AMDC_SCHED=1 AMDC_SCHED_OPTS=swp`, and with `taildup,swp` (Rust RA + Rust scheduler -> LLVM tail).

## 6. Reproduce
```
bash tools/llvm_build_llc.sh /shared/apps/ubuntu/opt/rocm-10.1.0 <llvm-project checkout> <tools dir>     # llc for ROCm 10
# sparse checkout + cmake configure + lit.site.cfg.py patch: section 1
bash tools/container/cargo.sh build --bin amdc   # with AMD_ROCM_MOUNT / AMD_LLVM_BIN as in section 1
bash tools/llvm_conformance.sh [lit|census|table|idem|all]     # env: LLVM_SRC LLVM_BUILD LLVM_TOOLS LLVM_ROCM AMDC WORK
python3 tools/conformance_classify.py $WORK/results/lit-diff-all.log     # failures by cause
```

## 7. Keeping it green on the next LLVM update
1. Probe: `amdc backend probe` (pass list, forbidden passes after the start); `amdc backend write-lock` only after steps 2-5.
2. `tools/llvm_build_llc.sh` for the new tree (or use its `llc` if shipped), regenerate `llvmNN_header.txt` with `llc -stop-before=gcn-create-vopd` on the IR shell; copy the table, fix the rows the opcode probe rejects (`cargo test --test llvm_tail`).
3. `tools/llvm_conformance.sh all`: baseline of the new LLVM first (its own failures are not ours), then compare the identity / flags / all tables with section 3; any new failure is an exporter/table regression.
4. Hazard diff: rerun `hazard_matrix` (`AMDC_ORACLE_OUT`) and diff against the previous matrix to get the new erratum changelog; rerun `tools/hazard_oracle.sh` for static nop/wait counts.
5. Hardware: attention_fwd, 15 shapes, determinism, scheduler variants (section 5).

## 8. Default backend and the retained-capability guarantee (round 3)

The LLVM tail is now the default backend (`docs/llvm-tail-integration.md` section 9). What is **proven** about "RA and scheduling remain ours" (Verified, `cargo test --test tail_invariance`, 48 kernels, scheduler off / `1` / `swp` / `taildup,swp` / `auto`): the instruction stream after LLVM's tail equals ours modulo nops, waits, fallthrough branches and operand spelling, the exported MIR is physical-register only, and the descriptor sizes agree. What is **not** claimed: that LLVM never adds instructions (it may add hazard workarounds such as `s_mov_b64 vcc, vcc`).
Open conformance items (unchanged from section 3): kill/dead flags, MMO atomic / LDS-DMA information, the 2 verifier cases (`waitcnt.mir`, `waitcnt-agpr.mir`).

## 9. Round 5: kill/dead flags exported, the 2 verifier cases explained (pinned ROCm 10.1.0, `tools/llvm_conformance.sh all`)

**Exporter change (Verified).** `llvm_mir::annotate_flags` writes `killed` / `dead` on the exported MIR from the per-instruction backward liveness we already compute (whole-operand granularity; only the last use of a register in an instruction carries the kill; flags are on by default, `AMDC_MIR_FLAGS=0` disables). All 50 kernels pass `-verify-machineinstrs` with them (`exported_flags_pass_the_machine_verifier`, 36 `cargo test` tests green), `tail_invariance` is green (default and `AMDC_SCHED=ld`), FMHA is unchanged: `fmha_fwd_hd128_mxfp8_t` B1H8S2048 292.1-292.8 TFLOP/s, 15/15 shapes bit-identical to the unscheduled plain build, LLVM `s_nop` count unchanged (171).

**The 2 "undefined physical register" cases (`waitcnt.mir`, `waitcnt-agpr.mir`) - root cause (Verified).** Not an exporter bug. The tests contain a `BUNDLE` whose second instruction reads a register defined inside the bundle and marked `internal` (`FLAT_STORE_DWORD $vgpr1_vgpr2, internal $vgpr0, ...`). The round-trip `flags` level dropped `internal` together with `killed/dead/renamable`; without it the verifier sees a read of `$vgpr0` that is not live-in. Our exporter never produces bundles, so there is nothing to emit; the `flags` level now keeps `undef` and `internal` (they state intent / bundle membership, not liveness).

**Final coverage table (178 post-RA tail-pass MIR tests, 169 fully importable; all numbers from one run):**

| level | criterion | PASS / 178 | importable PASS / 169 |
|---|---|---|---|
| identity | original RUN + FileCheck | 178 | 169 |
| identity | differential | 178 | 169 |
| flags (drop `killed dead renamable`) | differential | **175** (was 173) | 166 |
| reflag (flags, then recompute `killed/dead` with our annotation) | differential | 173 | 164 |
| all (also drop instruction flags, MMOs, debug locations) | differential | 140 | 131 |

Remaining failures:
* `flags` (3): `hazard-pass-ordering`, `insert-skip-from-vcc`, `si-pre-emit-peephole-preserve-loop-info` - the SIPreEmitPeephole folds need the kill/dead flags that this level removes from LLVM's own input.
* `reflag` (5): recomputing the flags with our annotation recovers `hazard-pass-ordering`. `insert-skip-from-vcc` and `si-pre-emit-peephole-preserve-loop-info` still differ *because the pass now optimises MORE than LLVM's recorded output* (e.g. the dead `$vcc = S_MOV_B64 0` / `S_AND_B64 $exec, -1` are removed: LLVM's input has no kill flags, so its expected output keeps them); a differential comparator cannot count an improvement as a pass. `hazard-shift64` (verifier) and `vopd-combine`, `vopd-combine-gfx1250` differ because tiny test functions never consume their results: our exact liveness marks those defs `dead`, which LLVM's own tests never do, and the hazard / VOPD passes then act differently. These are test-function artefacts of the foreign-MIR level, not exporter defects (the same annotation is verifier-clean on all kernels).
* `all` (38): unchanged: 19 MIMG needs MMO (DSL emits no image ops), 12 wait differences from missing MMO atomic / LDS-DMA information (`waitcnt.mir` and `waitcnt-agpr.mir` moved here: without MMOs LLVM waits conservatively), 7 other (importer artefact, error-expectation test, `machine-scheduler` start, cpol from MMO).
Opcode table vs LLVM MIR: 4405 exact, 453 implicit-operand differences, 154 explicit-shape differences, 5742 opcodes not in the table (48 table opcodes seen). Tail idempotence + verifier: 50 kernels, 0 failing.

**What stays unsatisfiable by definition:** pre-RA MIR tests (490 files, 6 with tail-pass RUN lines; CHECK lines encode LLVM's register allocation) and a scheduler level on LLVM-MIR (our scheduler works on our MOps). **Open items:** MMO atomic / LDS-DMA information when the DSL needs it; sub-register-granular kill flags (only whole-operand flags are written); tracking of `exec/m0/agpr/ttmp` liveness for foreign MIR in the `reflag` level.

**Reproduce:** `bash tools/llvm_build_llc.sh /shared/apps/ubuntu/opt/rocm-10.1.0 <llvm-project checkout> <tools dir>` (llc for ROCm 10); sparse checkout + `cmake -G Ninja llvm -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" -DLLVM_INCLUDE_TESTS=ON` + patch of `test/lit.site.cfg.py` as in section 1; `bash tools/container/cargo.sh build --bin amdc`; `bash tools/llvm_conformance.sh all` (levels `lit|census|table|idem`; env `LLVM_SRC LLVM_BUILD LLVM_TOOLS LLVM_ROCM AMDC WORK`); failures by cause: `python3 tools/conformance_classify.py $WORK/results/lit-diff-<level>.log`; FMHA sanity: `AB_VARIANTS=default bash tools/opt_ab.sh`.