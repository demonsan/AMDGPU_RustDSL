# L4 pass catalogue and pass-manager design (gfx950)

Survey of the backend as it exists on branch `sched-refactor` (based on origin/Main a5fc452d), written for phase **P1** of
[refactor-4layer-plan.md](refactor-4layer-plan.md): "named L4 passes and a pass manager with one pipeline per target, each pass carrying its
evidence (test / probe / LLVM oracle)", plus `amdc passes --target gfx950`. Nothing here changes code. Line numbers refer to this checkout;
they are anchors for the refactor, not a stable API. Crate shorthands: `compiler` = `crates/amdgpu-codegen/src`, `dsl` = `crates/amdgpu-dsl/src`,
`tile` = `crates/amdgpu-tile/src`, `amdc` = `crates/amdc/src/main.rs`, `k-tests` = `crates/amdgpu-kernels/tests`, `c-tests` =
`crates/amdgpu-codegen/tests`.

Evidence vocabulary: **test** (cargo test file or unit test), **probe** (hardware probe: `crates/amdgpu-kernels/src/probes/*.rs`, `tools/*probe*.sh`),
**oracle** (compared against the pinned LLVM `llc` / assembler: `tools/hazard_oracle.sh`, `tools/llvm_compare.sh`, `tools/llvm_conformance.sh`,
`c-tests/hazard_oracle.rs`), **doc** (measured notes under `docs/`). "none found" means `git grep` over `crates/*/tests`, the module's own
`#[cfg(test)]`, `tools/` and `docs/` found nothing specific to the pass; indirect coverage (every kernel compiles) is stated as such.

Two facts about the repo that matter for P1 and are easy to miss:

* There is **no `.co` step inside `amdc`**. `amdc emit` writes `.s`; the `.co` is produced by `clang -x assembler -target amdgcn-amd-amdhsa
  -mcpu=gfx950` from `python/amdgpu_jit/_compile.py:195` (and `tools/ab_fmha.sh:24`, `tools/md5_kernels.sh` for md5s of the `.s`).
* There is **no golden md5 manifest file** in the tree, only the generator `tools/md5_kernels.sh` (md5 of `emit <k> --backend=own` for every
  `amdc list` kernel) and `tools/md5_compare.py`. P0 / P1 acceptance needs the manifest to be created first (see §7).

---

## 1. The compilation pipeline as it runs today

Entry points: `compile` `compiler/lib.rs:47` -> `compile_with` `lib.rs:52`; `compile_mir_alloc` `lib.rs:61` and `compile_mir` `lib.rs:66`
(model reports, `amdc mops|nops|simmodel|print-ir --stage mir-ra|regplan`) call the same `lower_and_allocate`; `export_mir` `lib.rs:270` stops
before the tail (`amdc emit --mir-out`). CLI: `amdc emit` `amdc:976` resolves the backend (`resolve_backend` `lib.rs:379`: `auto` = `llvm-tail`
unless `uses_own_only_ops` `lib.rs:276`, which lowers the kernel once just to look for `Backend::OwnOnly` MOps = LDS-DMA) then calls `compile_with`.

Numbered pipeline (function name, file:line). "per rung" = re-executed by a retry ladder (§5.3).

| # | Step | Function | Location |
|---|---|---|---|
| 0 | (CLI only) load kernel: registry `kernel_by_name` or `--from-ir` text parse | `load_kernel` / `ir_text::parse` | `amdc:58`, `dsl/ir_text.rs:686` |
| 0a | (CLI only) shape specialization `--specialize` | `specialize::check/close/specialize` | `dsl/specialize.rs:91,166,221` |
| 0b | (CLI only) strict certificate, for `--from-ir` / specialized kernels unless `--no-verify` | `certify_specialized` -> `gpu_verify::analyse` | `amdc:119`, `amdc:985` |
| 1 | **Outer ladder** over lowering-option extras `[], remat1, remat2, remat3, remat4, imm+ldsspill{8,16,24,32,48,64}`; retries only when the error contains `"out of VGPRs"`; then rejects `lds_bytes > 160 KiB` | `lower_and_allocate` | `lib.rs:73-90` |
| 2 | Resolve the compile policy from the kernel NAME / `CompileProfile` (`Function::compile_policy`; the by-name `policy_name` lookup was removed: the AMDC_SCHED_ADD experiment switch matches the kernel name only), build the lowering option list (profile presets: vs/kvsplit/g2/t128, Gemm, other fmha, default `dce,cmpfold`) + `Function::lower_opts` + ladder extra | `lower_and_allocate_with` | `lib.rs:97-138` (`policy_name` `lib.rs:34`) |
| 3 | IR verify (`amdgpu_dsl::verify`), `any_lane` strip, `uniform_analysis` (if `scalarize`) | `lower::lower` | `lower.rs:84-111`, `dsl/lib.rs:3625` |
| 4 | Lower DSL IR to virtual-register MIR (`Lowerer::run`; also inserts `auto` waits, stamps `MInst::region`) | `Lowerer::run` | `lower.rs:1639` |
| 5 | Lowering clean-ups inside `Lowerer::finish`: AGPR pin marks, `resolve_reg_plan`, `cmpfold` group, dce, remat, LDS spill, coalesce, final dce | `Lowerer::finish` | `lower.rs:373-456` |
| 6 | `kabatch` kernarg batching (option) | `batch_kernarg_loads` | `lower.rs:113,122` |
| 7 | AGPR assignment / repair | `agpr::assign` | `lib.rs:142`, `agpr.rs:119` |
| 8 | Scheduler options: profile preset, `Function::sched_opts`, `AMDC_SCHED(_OPTS)`, declared regions (+ `AMDC_SCHED_REGION`), `region_strategies`, then `AMDC_SCHED_ADD` | `scheduler_options`, `declared_regions`, `region_strategies` | `lib.rs:145-166, 291-375` |
| 9 | **Inner ladder** `attempts = [(cut 0, hoist None)] + ldshoist:(0,24),(0,8),(0,0) + reorder:(16,0),(32,0),(48,0) + launch-budget:(c,0),(c+16,0),(c+32,0)`; each rung clones `base` (the post-AGPR MIR) | `lower_and_allocate_with` | `lib.rs:181-237` |
| 9.1 | `sched::run`: cfg cleanup, strip auto waits, dead-select dce, ldshoist, slotfill, sched_group_barrier phases, list scheduler (`schedule_blocks`), wsink, ldsink/ldahead32, **`insert_waits`**, `verify_waits` | `sched::run` | `sched/mod.rs:246-291` |
| 9.2 | M0: reuse (Gemm profile only), explicit M0 writes | `m0::reuse` / `m0::explicit` | `lib.rs:209-212`, `sched/m0.rs:17,60` |
| 9.3 | Pinned (safe-asm) block check on the scheduled MIR | `pin_check::check` | `lib.rs:213`, `pin_check.rs:14` |
| 9.4 | Value-latency hazards, `strict = true` (also pads block ends) | `hazards::fix_value_hazards_opts` | `lib.rs:217`, `hazards.rs:106` |
| 9.5 | MIR structural verify | `mir::verify` | `lib.rs:218`, `mir.rs:765` |
| 9.6 | Register allocation (+ `regplan::check` when a plan exists) | `regalloc::allocate` | `lib.rs:222`, `regalloc.rs:31` |
| 9.7 | Launch-budget check (error text gets the `"out of VGPRs for the launch"` prefix so the ladder retries) | `regalloc::check_workgroup_budget` | `lib.rs:224`, `regalloc.rs:1135` |
| 9.8 | Post-RA pads: in-flight-load WAW, physical WAR, `hzfill` (strict), M0 wait state, VALU-SGPR hazards | `pad_inflight_waw`, `pad_physical_war`, `hzfill::fill_with`, `m0::pad`, `pad_valu_sgpr` | `lib.rs:225-229`, `hazards.rs:758,469,999`, `hzfill.rs:217`, `sched/m0.rs:209` |
| 9.9 | Any other error: abort. All rungs fail: print `sched: falling back ...` on stderr and continue at 10 | | `lib.rs:232-238` |
| 10 | **Unscheduled fallback / no-scheduler path** (`base` moved): `phase::strip`, M0 reuse, M0 explicit, `pin_check`, `fix_value_hazards` (non-strict), `mir::verify`, `verify_waits`, `allocate`, then the same post-RA pads with `hzfill(strict=false)` | | `lib.rs:240-266` |
| 11 | `check_workgroup_budget` again (outside the ladder, hard error) | | `lib.rs:54` |
| 12 | `gpu_verify::enforce`: IR + MIR verification per `AMDC_GPU_VERIFY=off|warn|strict|default` (default: only MIR register / wait errors fail) | `gpu_verify::enforce` | `lib.rs:55`, `gpu_verify.rs:2835` |
| 13 | Tail backend `finish`: **own** = `emit::emit` (text with `.amdhsa_*` descriptor and `amdhsa.kernels` metadata); **llvm-tail** = probe `backend.lock`, `llvm_mir::export_opts`, `llc -start-before=<pass>`, `strip_pad_args` | `OwnTail::finish`, `LlvmTail::finish` | `lib.rs:56`, `backend/mod.rs:31,145`, `emit.rs:12`, `backend/llvm_mir.rs:159`, `backend/mod.rs:556` |
| 14 | (outside amdc) assemble to `.co` with clang | `_compile.py` | `python/amdgpu_jit/_compile.py:195` |

Today's pipeline has **two copies** of the post-allocation sequence (steps 9.2-9.8 and 10); they differ in exactly four places: `strict` flag of
the hazard pass, wait handling (`sched::run` re-derives waits, the fallback trusts lowering's `auto` waits and only `verify_waits` them),
the budget check inside the rung, and the `hzfill` `strict` flag. A pass manager must reproduce this, not "unify" it (§5.3).

Global / hidden state read by the pipeline: thread-local lowering option override (`lower.rs:32-66`, `with_opts`), thread-local
`LAST_ORIGINS` (`lower.rs:110`, read by `diag`), thread-local `order::set_relax_dma_lds_read` (`sched/order.rs:47`), thread-local mutation hook
`hazards::without_valu_sgpr` (`hazards.rs:1118`), and process environment (`AMDC_*`, listed in §2.9).

---

## 2. Pass / analysis catalogue

Columns: **ID** = stable proposed name; **Where** = entry function; **Pos** = stage index (`PRE` pre-lowering IR, `LOW` lowering,
`MIR` MIR pre-RA, `RA` allocation, `POST` post-RA, `EMIT`, `VER` verification) with order index; **In -> Out**; **Target dependence**
(`generic` / `g950-param` = algorithm generic, constants or tables are gfx950 parameters / `g950-only` = rules or ISA names are gfx950-specific) with
the reason; **Switches**; **Evidence**. Target facts quoted from the code are the candidates for `Target` (P0).

### 2.1 Pre-lowering IR and IR analyses (layer: amdgpu-dsl, L4 IR)

| ID | Where | Pos | In -> Out | Target dep. | Switches | Evidence |
|---|---|---|---|---|---|---|
| `ir-specialize` | `dsl/specialize.rs:221` (`check` 91, `close` 166) | PRE.0 (CLI only) | `Function`, bindings -> `Function` | generic | `--specialize` | test: `crates/amdc/tests/jit.rs`; doc: `docs/jit.md`, `docs/sym.md` |
| `ir-parse-text` / `ir-print-text` | `dsl/ir_text.rs:686,697` / `print` | PRE.0 (CLI) | text <-> `Function` | generic | `--from-ir` | test: `k-tests/ir_text.rs` (print -> parse fixpoint + identical asm for every kernel); `tools/ir_roundtrip_check.sh`; doc: `docs/ir-text.md`, `docs/ir-text-roundtrip-design.md` |
| `ir-check` | `dsl/ir_text.rs:870` | PRE | `Function` -> `Vec<String>` | generic | `amdc ir-check` | test: `k-tests/ir_text.rs` |
| `verify-ir-structure` | `dsl/lib.rs:3625` (called `lower.rs:85`) | PRE.1 (always) | `Function` -> errors | generic (typing, wave-uniform branch rule `IR-UNIFORM-BRANCH`; not audited for gfx950-only ops) | none | test: `k-tests/` + `c-tests/` files calling `amdgpu_dsl::verify`: `bringup_guards`, `dpp`, `lds_b64`, `permlane_swap`, `mfma16x16x128`, `buffer_b128x2`, `regplan`, `safeasm`, `tileops_ir`, `uniform_select`, `var_slice`, `expr_layer`, `backend`, `capture` |
| `analysis-uniform-values` | `dsl/lib.rs:4678` (`uniform_values`); lowering's own copy `lower.rs:4139` (`uniform_analysis`, used by `scalarize`) | PRE (analysis) | `Function` -> uniform value / var sets | g950-param (wave-uniform = no workitem-id dependence; wave64 not hard-coded here) | `scalarize` lowering option | test: `c-tests/uniform_select.rs`, `scalarize2.rs` |
| `analysis-facts` | `dsl/facts.rs` (`Facts`, `must_order`, `stale_view_roots` 601) | PRE (analysis, used by Weaver and lowering) | `Function`/units -> def/use/effect facts | generic | none | test: `crates/amdgpu-codegen/tests/weave*.rs`, `c-tests/var_views.rs`; doc: `docs/weave-capture.md` |
| `strip-any-lane` | `lower.rs:4985` | PRE.2 | `Function` -> `Function` (markers substituted) | generic | none (automatic when marker present) | none found (indirect: every kernel using `any_lane`) |

### 2.2 Lowering (L4, `compiler/lower.rs`)

`lower` is one 5 k-line module; the steps below are its separable internal passes (all in `Lowerer::finish` order, `lower.rs:373-456`). Proposed
split: `lower-mir` stays one `Function -> MFunction` pass; the post-lowering clean-ups become MIR passes in the same order (pure refactor).

| ID | Where | Pos | In -> Out | Target dep. | Switches | Evidence |
|---|---|---|---|---|---|---|
| `lower-mir` | `lower.rs:84` (`Lowerer::run` 1639) | LOW.0 | `Function` -> `MFunction` (vregs, `auto` waits, region ids) | **g950-only**: MOp set = gfx950 ISA (scaled MFMA f8f6f4, `v_cvt_scale*` fp4/fp6, `v_permlane32_swap`, `v_pk_*_f32`, LDS-DMA with M0), inline-constant table `is_inline_const` `lower.rs:70`, VOP3-literal rule `imm_ok` `lower.rs:76`, wave64 (`workgroup_size+63)/64`) | lowering option list (`dce,cmpfold` default; `scalarize scalarize2 waveuni sgpr imm shl addrfold saddr saddrlds retarget2 zeroinit lgkmn edgepad salubr kabatch remat1-4 ldsspill*`): profile presets, **kernel-name based** (`lib.rs:105-138`) | test: `k-tests/mir_invariants.rs` (side-effect sequence and definedness with/without the default options), `register_accounting.rs`, `sgpr_legality.rs`, `tileops_ir.rs`, `tail_invariance.rs`, `dispatch_compile.rs`; `c-tests/bringup_guards.rs`, `var_semantics.rs`, `var_slice.rs`, `var_views.rs`, `scalarize2.rs`, `uniform_select.rs`, `dpp.rs`, `lds_b64.rs`, `buffer_b128x2.rs`, `permlane_swap.rs`, `mfma16x16x128.rs`; oracle: `tools/llvm_compare.sh` + `tools/llvm_compare/cases.tsv` (instruction choice vs llc), `docs/llvm-tail-conformance.md`; probe: `crates/amdgpu-kernels/src/probes/{mod,tileops,handoff}.rs`, `tools/isa_probes.sh` |
| `lower-mfma-dc` (token `mfmadc`, opt-in) | `Lowerer::mfma_acc` (`lower.rs`) | LOW.0 | MFMA accumulator form: when C stays live after the MFMA, D is a new register and the MFMA reads C where it is (gfx950 `v_mfma vdst, a, b, srcC`, D and C identical or disjoint) instead of copying C into D | g950-only | **measured neutral**: on by default it removed 2628 copies over 12 kernels (vs128 -256), same-run perf_ab fmha_mxfp8 1.003 / fmha_fp8pt 1.002 (9 reps), port_gate PASS; kept opt-in (exact mode always uses this form) | none |
| `lower-movb64` (token `movb64`, opt-in) | `run_movb64` (`lower.rs`) | LOW.0 | two adjacent writes of zero into the dwords 2k, 2k+1 of one tuple (fragment / accumulator initialization) become one `v_mov_b64 v[..], 0` | g950-only | **measured neutral**: `AMDC_LOWER_ADD=movb64` changes 70 of the 240 kernels (VALU -1950 over 19 `fmha_fwd`, -194 `fmha_bwd`, -50 `gemm_a8w8`, -19 `gemm_a16w16`, ...; every saving is in the accumulator zeroing prologue), same-run `perf_ab --quick` over decode / fmha_bf16 / fmha_bwd / fmha_fp8pt / fmha_hd192 / fmha_mxfp8 / gemm / mla (3 reps): all families inside their allowed drop (decode 0.999, fmha_bf16 1.011, fmha_bwd 1.001, fmha_fp8pt 1.000, fmha_hd192 1.016, fmha_mxfp8 1.008, gemm 1.006, mla 0.998); the apparent +1 .. +3 % on four small cases did not reproduce with 9 reps (bf16 1.001 / 1.006, hd192 0.997 / 1.001, mxfp8 0.998 / 0.986). 11 probe kernels that resolve to `llvm-tail` fail to emit (`llvm20.tbl has no row for VZeroSub2`, kinds `-`). Kept opt-in (the FMHA replica configuration uses it) | `tests/mops_w5.rs` |
| `lower-jthread` (token `jthread`, opt-in) | `cfg_thread.rs` (`run_jthread`, `lower.rs`) | LOW.0 | jump threading on flags known per incoming edge (a branch on a compare whose outcome one incoming edge fixes goes straight to its target) | g950-only | **no candidates outside its configuration**: added to every kernel (`AMDC_LOWER_ADD=jthread`) the static screen changes 0 of the 240 manifest kernels (only the FMHA replica configuration, which already carries it, has the pattern); nothing to measure | `tests/jthread.rs` |
| `lower-coldlayout` (token `coldlayout`, opt-in) | `cfg_thread.rs` (`run_coldlayout`, `lower.rs`) | LOW.0 | the blocks of branches hinted unlikely (`begin_if_unlikely`) move after every hot block | g950-only | **no candidates outside its configuration**: `AMDC_LOWER_ADD=coldlayout` and `jthread,coldlayout` change 0 of the 240 kernels (no kernel but the replica has an unlikely-hinted branch) | `tests/jthread.rs` |
| `lower-kabatch` | `lower.rs:122` | LOW.1 | `MFunction` -> `MFunction` (kernarg `s_load` batched to `s_load_dwordx4`) | g950-only (SMEM x4, 16-byte windows) | `kabatch` | none found (GEMM profile only) | **not a default candidate** (`AMDC_LOWER_ADD=kabatch`, static screen over the 240 manifest kernels): 37 kernels change by -1 .. -21 SALU instructions (`qknorm_rope`, `add_rmsnorm`, `reshape_cache`, ...), but the 77 kernels that resolve to the `llvm-tail` backend fail to emit (`llvm20.tbl has no row for SLoadDwordx4`); GEMM has it through its profile; no hardware A/B (the gain is a few scalar loads in the prologue) |
| `lower-select-fold` | `fold_select_cmp` 3953, `select_minmax` 5045, `dce_cmp_pairs` 4000, `fuse_minmax3` 5146 | LOW.2 | `MFunction` -> `MFunction` | g950-only (v_min/max/med3, vcc hazard avoidance) | `cmpfold` (default on) | `k-tests/mir_invariants.rs` | `cmpfold` is a DEFAULT token: `AMDC_LOWER_ADD=cmpfold` changes 0 of the 240 manifest kernels (nothing to A/B) |
| `lower-dce` | `dce_full` 4041 (loads-only always; full when `dce`), `dce` 3902 (final) | LOW.3 / LOW.6 | `MFunction` -> `MFunction` | generic | `dce` (default on) | `k-tests/mir_invariants.rs` |
| `lower-remat` | `remat_consts` 4242 (`remat1..3`), `remat_exprs` 4316 (`remat4`) | LOW.4 | `MFunction` -> `MFunction` | generic (pressure transform) | ladder rung, `remat1..4` | none found specific (exercised by the outer ladder; `register_accounting.rs` indirect) |
| `lower-lds-spill` | `lds_spill` `lower.rs:4452` | LOW.5 | `MFunction` -> `MFunction` (+ `insert_waits`) | g950-param (LDS 160 KiB limit `lib.rs:80`) | `ldsspill8..64`, env `AMDC_RA_LDS_SPILL(_TARGET)` documented at `lower.rs:4446` (read by probes only, `probes/mod.rs:2106`) | test: `c-tests/wait_operands.rs` (regression where the ladder reached LDS spill) |
| `lower-coalesce` | `coalesce` `lower.rs:3853` | LOW.6 | `MFunction` -> renames | generic | `AMDC_COALESCE_DEBUG` (debug print) | none found specific |

### 2.3 AGPR assignment (MIR pre-RA)

| ID | Where | Pos | In -> Out | Target dep. | Switches | Evidence |
|---|---|---|---|---|---|---|
| `assign-agpr` | `agpr.rs:119` | MIR.0 | `MFunction` -> `MFunction` (+ `agpr_pin`) | **g950-only**: unified 512-register file, which MOps may read/write AGPR (`agpr_srcs`/`agpr_dst` in `op_attrs`), `v_accvgpr_*`, MFMA vdst/srcC same-file rule | DSL `pin_agpr` / `agpr` marks | test: `k-tests/agpr.rs`; probe: `tools/agpr_probe.sh`; doc: `docs/agpr.md` (assembler-verified rules) |

### 2.4 Scheduling (MIR pre-RA; `compiler/sched/`, driver `sched::run` `sched/mod.rs:246`)

Whole subsystem is entered only when `scheduler_options(f)` is `Some`. Analyses used: `sched-liveness` (`sched/liveness.rs:40`, `successors` 14),
`sched-depgraph` (`depgraph.rs:47` over `effects.rs:68` and `order.rs:59` `mem_must_order`), `sched-cost-model` (`model.rs:76`: per-opcode pipe / issue /
latency table, "data, every row carries an evidence grade"), `sched-sim` (`sim.rs:272`, reporting only, `amdc simmodel` / `amdc summary`).

| ID | Where | Pos | In -> Out | Target dep. | Switches | Evidence |
|---|---|---|---|---|---|---|
| `sched-cfg-cleanup` | `sched/cfg.rs:16` | MIR.1 | `MFunction` -> `MFunction` (empty latch removal, sets fall-through elision) | generic | `nocfg` | none found specific |
| `waits-strip-auto` | `sched/waits.rs:224` | MIR.2 | -> `MFunction` without `MInst::auto` waits | generic | - | `c-tests/wait_operands.rs` (indirect) |
| `sched-dce-dead-selects` | `sched/cfg.rs:55` | MIR.3 | -> `MFunction` | generic | - | none found |
| `sched-ldshoist` | `sched/hoist.rs:17` | MIR.4 | `MFunction`, `Options` -> `MFunction` (ds_read moved up, pressure-budgeted, `check_perm` verified) | g950-param (LDS-read latency / pressure budget) | `ldshoist`, `hoist=N`; GEMM preset, ladder rungs `(0,24),(0,8),(0,0)` | test: `k-tests/sched_kernels.rs`, `sched_preset.rs`; doc: `docs/scheduling.md` |
| `sched-slotfill` | `sched/slotfill.rs:329` | MIR.5 | `MFunction` -> `MFunction` (MFMA skeleton, floaters in deadline windows) | **g950-only**: `SLOT_CYCLES 32` (MFMA issue interval measured 32.25, `tools/mfma_ceiling.py`), `SLOT_BUDGET 28`, `HOIST_SLOTS 40`, `DMA_LATENCY_CYCLES 512` (`slotfill.rs:32-43`) | `slotfill`; env `AMDC_SLOTFILL`, `AMDC_SF_DMA_SLOTS`, `AMDC_SF_PRESSURE_SLACK` appear only in docs | test: `c-tests/slotfill.rs`, `sched_regions.rs`; doc: `docs/slotfill.md` |
| `sched-phase` | `sched/phase.rs:169` (`strip` 55) | MIR.6 | `MFunction` -> `MFunction` (`sched_group_barrier` markers consumed; always stripped) | g950-param (LLVM-style class mask `phase.rs:30`) | DSL `sched_group_barrier` / `sched_pattern` | test: `c-tests/sched_phase.rs` |
| `sched-list` | `sched/mod.rs:151` (`schedule_blocks`; `list::schedule_best` `list.rs:238`) | MIR.7 | `MFunction` -> `MFunction`; each block checked by `verify::check_perm` (`verify.rs:47`) | g950-param: algorithm generic, machine model data `model.rs` (gfx950 cycle table), pressure limit default 240 (`Options::default`), `_t128` 248 | `reorder`/`rewait`, `xbar`, `limit=N`, `noverify`, `stats`; `AMDC_SCHED=1|ld|0`, `AMDC_SCHED_OPTS`, `AMDC_SCHED_ADD` (**experiment**), `AMDC_SCHED_REGION` (**experiment**) | test: `sched/tests.rs` (unit, `sched/mod.rs:293`), `k-tests/sched_kernels.rs` (every kernel schedules + verifies, with/without xbar, rewait), `sched_preset.rs`, `c-tests/sched_regions.rs`, `sched_regions_lower.rs`; doc: `docs/scheduling.md`; probe/measure: `tools/sched_mb/` (`RESULTS2.md`), `tools/sched_model_vs_hw.sh` |
| `sched-wsink` | `sched/ldahead.rs:144` | MIR.8 | `MFunction` -> `MFunction` | g950-only (lgkmcnt explicit waits) | `wsink` | test: `c-tests/ldahead.rs`; doc: `docs/lgkm-lookahead.md` |
| `sched-ldahead` | `sched/ldahead.rs:18` (`ldsink=D`, `ldahead32=D`) | MIR.9 | `MFunction` -> `MFunction` | g950-only (LDS read latency, MFMA distance) | `ldsink=D`, `ldahead32=D` | test: `c-tests/ldahead.rs`, `sched_regions_lower.rs`; doc: `docs/lgkm-lookahead.md` |
| `waitcnt-recompute` | `sched/waits.rs:178` (`insert_waits`) | MIR.10 | `MFunction` -> `MFunction` with scoreboard waits | **g950-only**: vm / lgkm counter model, `vmcnt` 6 bit (63), `lgkmcnt` 4 bit (15), partial lgkm waits only while no SMEM is queued (`waits.rs:5-15`) | always (inside `sched::run`); fallback path never runs it | test: `c-tests/wait_operands.rs`, `gpu_atomic.rs`, `inflight_waw.rs`; k-tests `sched_kernels.rs`; doc: `docs/scheduling.md` |
| `verify-waits` | `sched/waits.rs:236` | VER (also MIR.11) | `MFunction` -> `Result` | g950-only (same counter model; re-simulates with waits present) | `Options::verify` (inside `sched::run`); unconditional on the fallback path `lib.rs:256`; reused by `gpu_verify::verify_mir` as `GV-WAIT` | same as `waitcnt-recompute`; `k-tests/gpu_semantics.rs` |
| `m0-reuse` | `sched/m0.rs:17` | MIR.12 | `MFunction` -> `MFunction` (`skip_m0`) | **g950-only** (M0 base for LDS-DMA, mandatory `s_nop 0`) | only for `CompileProfile::Gemm` (`m0_reuse_on` `lib.rs:93`); the lowering token `m0reuse` turns it on for any kernel (opt-in, **measured neutral**: the static screen over the 240 manifest kernels changes 10 (9 `fmha_fwd`, `probe_dma_kread`), -521 SALU instructions, VALU unchanged, moe / mla / decode / gemv unchanged; same-run `perf_ab` `fmha_mxfp8` 0.998 / `fmha_fp8pt` 1.000 (5 reps), every case inside its allowed drop; kept opt-in) | none dedicated; M0 appears in `c-tests/dma_hazards.rs`, `slotfill.rs`, `sched_regions.rs` |
| `m0-explicit` | `sched/m0.rs:60` | MIR.13 | `MFunction` -> `MFunction` (explicit `s_mov m0` hoisted; folds `s_add` into m0) | g950-only | `AMDC_M0_EXPLICIT=0`, `AMDC_M0_FOLD=0`, `AMDC_M0_HOIST=0` (kill switches, `m0.rs:61-65`; **experiments**) | none dedicated (see `m0-reuse`) |

### 2.5 MIR checks and hazards before allocation

| ID | Where | Pos | In -> Out | Target dep. | Switches | Evidence |
|---|---|---|---|---|---|---|
| `verify-pinned-blocks` | `pin_check.rs:14` | MIR.14 (VER) | `MFunction` -> `Vec<String>` (error aborts, non-"VGPR" so no retry) | g950-only (runs the value-hazard and wait models on a copy) | none | test: `c-tests/pin_check.rs`, `pinning.rs`; `k-tests/safeasm.rs` |
| `hazard-value` | `hazards.rs:106` (`fix_value_hazards_opts`; `fix_value_hazards` 99 = non-strict) | MIR.15 | `MFunction` -> `MFunction` (`s_nop` / in-block fill) | **g950-only**: `TRANS_LATENCY 5`, `FILL_WINDOW 48` (`hazards.rs:19-22`), MFMA result latency from `op_attrs mfma.latency` (CDNA4 ISA 7.6), VALU->MFMA/permlane/readfirstlane/DPP 2 states (4 after carry-out adds), cvt dst-select 1 state | `strict` = scheduled path (also pads block end); `AMDC_NO_FILL_VCC` (A/B **experiment** `hazards.rs:24`); `edgepad` lowering option carries windows across blocks | test: `c-tests/hazard_classes.rs` (every VALU writer class), `hazard_oracle.rs`, `smfmac_hazards.rs`, `dma_hazards.rs`, `k-tests/hzfill.rs`; oracle: `tools/hazard_oracle.sh` (own tail vs llc `s_nop` counts), `c-tests/hazard_oracle.rs` (writes `$AMDC_ORACLE_OUT` matrix); probe: `tools/mfma_dep.sh`, `docs/mfma-dep-probe.md` |
| `verify-mir-structure` | `mir.rs:765` | MIR.16 (VER) | `MFunction` -> `Vec<String>` | g950-param | none | test: `k-tests/sched_kernels.rs`, `c-tests/smfmac_hazards.rs`, `k-tests/hzfill.rs` |

### 2.6 Register allocation

| ID | Where | Pos | In -> Out | Target dep. | Switches | Evidence |
|---|---|---|---|---|---|---|
| `regalloc` (`regalloc-scan` -> `regalloc-scan-top` -> `regalloc-two-phase` fallbacks) | `regalloc.rs:31` (`allocate_any` 39, `allocate_policy` 77, `Policy::Scan/TwoPhase`) | RA.0 | `MFunction` -> `Alloc` (`phys`, `max_v`, `max_s`, `scalar`) | **g950-param**: `VGPR_COUNT 256`, `AGPR_COUNT 256`, `UNIFIED_FILE 512`, `SGPR_COUNT 102`, `V_RESERVED 1` (v0 = workitem id), `S_RESERVED 8` (kernarg ptr, wg ids) (`regalloc.rs:20-29`), even-aligned tuples; policy tries 3 allocators in sequence, falls to next only on `out of VGPRs` | none (`AMDC_RA_FORCE_TWOPHASE`, `AMDC_RA_DEBUG_FILE` exist: debug) | test: `c-tests/ra_reaching.rs`; `k-tests/register_accounting.rs`, `sgpr_legality.rs`; `c-tests/regplan.rs`; internal gate `check_reaching` `regalloc.rs:914` (two-phase result rejected on violations) |
| `regplan-check` | `regplan.rs:208` (called from `allocate` `regalloc.rs:34`) | RA.1 (VER) | `MFunction`, `Alloc` -> `Result` (pins exact, separate groups disjoint, caps) | g950-param | active only when DSL `RegMark`s exist | test: `c-tests/regplan.rs`, `k-tests/` `regplan` users; doc: `docs/weaver-regs.md` |
| `check-workgroup-budget` | `regalloc.rs:1135` | RA.2 (VER) | -> `Result` | **g950-only**: 4 SIMDs per CU, wave64, 512 unified file, allocation granule 8 | none; failure text must contain `out of VGPRs` inside the ladder (`lib.rs:224`) | `k-tests/register_accounting.rs`; unit `launch_vgpr_budget` `sched/mod.rs:298` |
| `analysis-pressure` | `regalloc.rs:847` (`pressure_profile`); `regplan::region_report` `regplan.rs:100` | RA (analysis) | -> peak per position / region | g950-param | `amdc regplan` | `c-tests/regplan.rs` |

### 2.7 Post-RA hazards and fills

| ID | Where | Pos | In -> Out | Target dep. | Switches | Evidence |
|---|---|---|---|---|---|---|
| `hazard-inflight-waw` | `hazards.rs:758` (`pad_inflight_waw`) | POST.0 | `MFunction`, `Alloc` -> waits inserted | g950-only (vmcnt / lgkmcnt counters at physical level; scalar loads return out of order) | none (correctness) | test: `c-tests/inflight_waw.rs`; env `AMDC_WAW_DEBUG` debug |
| `hazard-phys-war` | `hazards.rs:469` (`pad_physical_war`) | POST.1 | `MFunction`, `Alloc` -> `s_nop` | **g950-only**: MFMA write-completion wait states (`mfma_dst_wait` 455: 18+1 for 16-pass, 21 for scaled 32x32x64 / sparse), VMEM wide store data 2 states (`STORE_DATA_WAIT 3`, `hazards.rs:431`) | none (`AMDC_NO_PHYS_WAR` / `_WAW` / `_STORE` are only mentioned in docs; not read by code) | test: `c-tests/phys_waw.rs`, `k-tests/agpr.rs`, `hzfill.rs`; doc comment says "only scheduled code" but it runs on both paths (`lib.rs:262`) |
| `hz-fill` | `hzfill.rs:217` (`fill_with`; `fill` 196) | POST.2 | `MFunction`, `Alloc`, strict, `Options` -> `MFunction` (instructions moved into nop windows; re-runs hazard passes on a copy, `check_perm`) | g950-only (reuses hazard passes; physical dependence incl. vcc, scc) | scheduler token `hzfill` (per region) or `AMDC_SCHED_ADD=<k>:hzfill` (**experiment**), `AMDC_SCHED_VERBOSE` | test: `k-tests/hzfill.rs`; doc: none beyond module header |
| `m0-pad` | `sched/m0.rs:209` | POST.3 | `MFunction` -> `s_nop 0` where the explicit M0 write is adjacent to its DMA | g950-only | none | none dedicated |
| `hazard-valu-sgpr` | `hazards.rs:999` (`pad_valu_sgpr`) | POST.4 | `MFunction`, `Alloc` -> `s_nop` (VALU-written SGPR then VMEM reader 5 states / VALU reader 2) | **g950-only** (LLVM GCNHazardRecognizer rules, gfx940 hazards) | none ("not switchable"); mutation-test hook `without_valu_sgpr` `hazards.rs:1118` | test: `k-tests/hazard_valu_sgpr.rs`; probe: `probe_valu_sgpr` (named in `hazards.rs:998`); oracle: llc tail via `tools/hazard_oracle.sh` |
| `analysis-nop-classes` | `hazards.rs:919` (`classify_nops`) | POST (report) | `MFunction` -> `Vec<NopSite>` | g950-only | `amdc nops` | none found (tool, `tools/nop_attrib.py`) |

### 2.8 Emission / tail

| ID | Where | Pos | In -> Out | Target dep. | Switches | Evidence |
|---|---|---|---|---|---|---|
| `emit-own` | `emit.rs:12` (`OwnTail` `backend/mod.rs:25`) | EMIT.0 | `MFunction`, `Alloc` -> asm text + `.amdhsa_kernel` descriptor + metadata | **g950-only**: `amdhsa.target: amdgcn-amd-amdhsa--gfx950`, `.amdhsa_accum_offset`, `next_free_vgpr/sgpr`, user SGPR count 2, wg id x/y/z, denorm modes (`emit.rs:65-128`), DPP / op_sel / cbsz encodings | `--backend=own`; env `AMDC_BACKEND` | test: `k-tests/asm_assembles.rs` (real assembler accepts every kernel), `tail_invariance.rs`, `bringup_guards.rs`, `dispatch_compile.rs`; md5 oracle `tools/md5_kernels.sh` |
| `emit-llvm-tail` | `backend/mod.rs:145` (`LlvmTail::finish`); `llvm_mir::export_opts` `backend/llvm_mir.rs:159`; `run_llc` `backend/mod.rs:98` | EMIT.0 (alternative) | `MFunction`, `Alloc` -> post-RA MIR text -> `llc` asm | g950-param: `LLC_ARGS = -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -O2`, start pass preference `gcn-create-vopd` / `si-memory-legalizer` (`backend/llvm20.rs:44-45`); op table `backend/llvm20.tbl` / `llvm24.tbl` | `--backend=llvm-tail|auto`; `AMDC_LLC_ARGS`, `AMDC_KEEP_MIR`, `AMDC_DIALECT`, `AMDC_BACKEND_LOCK`, `AMDC_DROP_AUTO_WAITS` (**experiment**), `ROCM_LLVM_BIN`; pinned by `backend.lock` (`backend/probe.rs:122-183`) | test: `k-tests/llvm_tail.rs`, `backend_auto.rs`, `backend.rs`, `tail_invariance.rs`, `register_accounting.rs`; unit tests in `backend/llvm_mir.rs`, `backend/mir_roundtrip.rs`; oracle: `tools/llvm_conformance.sh`, `docs/llvm-tail-conformance.md`, `docs/llvm-tail-integration.md`, `amdc backend probe|check` |
| `emit-asm-check-lite` | `asm_check.rs:12,18` (`check_lite_contract[_ex]`) | EMIT -> VER (asm text) | asm text -> `Result` | g950-only (`sc1`, `buffer_wbl2`, `s_barrier` does not drain vmcnt) | none; not called from the compile pipeline (only tests) | test: `k-tests/lite_contract.rs`, `c-tests/mfma16x16x128.rs`; doc: `docs/last-arriver-memory-model.md` |
| `report-mir-roundtrip` | `backend/mir_roundtrip.rs:283-440` | tool | MIR text <-> text | g950-only (table) | `amdc mir-roundtrip`, `mir-table-report` | unit tests in the module |

### 2.9 Environment switches (all are experiments / debug unless stated)

`AMDC_SCHED` (`lib.rs:288,341`, `sched/mod.rs:71`; `1`, `ld`, `0`; also set by `--sched` / `--sched=ld`, `amdc:1005-1008`, `amdc:748`), `AMDC_SCHED_OPTS`
(`sched/mod.rs:82`), `AMDC_SCHED_ADD` (`lib.rs:154,170`), `AMDC_SCHED_REGION` (`lib.rs:310`), `AMDC_SCHED_VERBOSE` (`lib.rs:176,238`),
`AMDC_M0_EXPLICIT|FOLD|HOIST` (`sched/m0.rs:61-65`), `AMDC_NO_FILL_VCC` (`hazards.rs:24`), `AMDC_WAW_DEBUG` (`hazards.rs:829,888`),
`AMDC_RA_DEBUG_FILE` (`regalloc.rs:1021`), `AMDC_COALESCE_DEBUG` (`lower.rs:744`), `AMDC_GPU_VERIFY` (`gpu_verify.rs:2818`; **policy**, not an
experiment), `AMDC_GW_DEBUG`, `AMDC_NO_BREAK_FACTS` (`gpu_verify.rs:2056`, disables loop-break facts: negative testing), `AMDC_BACKEND` (`amdc:992`; **policy**),
tail: `AMDC_LLC_ARGS`, `AMDC_KEEP_MIR`, `AMDC_DIALECT`, `AMDC_BACKEND_LOCK`. Weaver (kernel-build time): `AMDC_WEAVE_AUTO`, `AMDC_WEAVE_AUTO_STMT`,
`AMDC_WEAVE_AUTO_VERBOSE|PREFIX` (`tile/weave_auto.rs:405-434`, allow-listed by `k-tests/no_env_knobs.rs`: kernels must not read the
environment). `AMDC_LOWER_OPTS` and `AMDC_RA_FORCE_TWOPHASE`, `AMDC_NO_PHYS_*`, `AMDC_MFMAPAIR`, `AMDC_PIN_VMWAIT`, `AMDC_DPP_REDUCE`, `AMDC_SF_*`,
`AMDC_SLOTFILL`, `AMDC_SOFTMAX_RECOMPUTE_EXP`, `AMDC_WAITS_SKIP_FENCE_OPERANDS` occur in docs / comments / probes only. P1 should fold the live ones
into one `Options` struct and reject unknown `AMDC_*` names in `amdc` (opt-in).

---

## 3. Verification passes

All live in `compiler/gpu_verify.rs` (3046 lines) and `compiler/gpu_protocol.rs` (1443 lines); they are L4 **verification** (target-independent
IR level, gfx950 register / wait rules at MIR level).

| ID | Where | Pos | In -> Out | Target dep. | Gating | Evidence |
|---|---|---|---|---|---|---|
| `verify-gpu-ir` (v1 bounds, LDS budget, global / buffer bounds with symbolic intervals and the kernel `Contract`; GV-GLB-OOB / -NOEXT / -PROV, GV-BUF-OVER / -WRAP, GV-LDS-OOB, GV-CONTRACT, GV-ASSUME*, GV-ASM) | `gpu_verify.rs:2583` (`verify_ir`; `An::run`) | VER.0 | `Function` -> `Report` | g950-param: `LDS_LIMIT = 160 KiB` (`:2656`); address / descriptor semantics are gfx950 MUBUF | `AMDC_GPU_VERIFY` (`enforce` `:2835`); default mode ignores IR-level errors (kernels need their contract first) | test: `k-tests/gpu_semantics.rs` (a negative case per error class), `verifier_cases.rs`, `trust_inventory.rs`, `gpu_tiles.rs`, `lds_tracked.rs`, `rows.rs`, `eltwise.rs`, `gemm_mxfp8.rs`, `rope_cache.rs`, `tile_l123.rs`, `tile_l4.rs`, `tile_l56.rs`; `c-tests/gv_bits.rs`, `gpu_buffer_lds.rs`, `gpu_atomic.rs`, `sym_branch.rs`, `uniform_select.rs`, `bringup_guards.rs`, `dpp.rs`, `lds_b64.rs`, `permlane_swap.rs`, `mfma16x16x128.rs`, `buffer_b128x2.rs`; doc: `docs/gpu-semantics-verifier.md`, `docs/trust-inventory.md`, `docs/sym.md` |
| `verify-barrier-balance` (GV-BAR: wave-divergent if/else with different barrier counts) | `gpu_verify.rs` `barrier_balance` (ends `:2580`), called `:2619` | VER.1 | `Function` -> findings | generic (s_barrier semantics) | same | `k-tests/gpu_semantics.rs` |
| `verify-stages` (GV-STAGE: stage contracts live_in/live_out, element tables, barrier phases, outstanding vm/lgkm) | `gpu_verify.rs:2882`, called `:2620` | VER.2 | `Function` -> findings | generic (uses linear vm/lgkm in-flight counters) | same | test: `c-tests/stage_ir.rs`, `crates/amdgpu-stage/tests/contract_stage.rs`; doc: `docs/fusion-layout-handoff.md` |
| `verify-protocol-tiles` (v4 protocol engine: LDS tiles / rings / private regions and global-memory regions along ARRIVAL, COMPLETION, VISIBILITY, GENERATION axes; GV-TILE-* aliased to GV-PROTO-*, `proto_alias` `gpu_protocol.rs:78`; also the pub alias `gpu_tiles`) | `gpu_protocol.rs:1160` (`verify_tiles`) | VER.3 | `Function` -> findings | g950-param (wave roles, barrier/wait semantics; unrolls 1..=4 iterations) | same; trusted escapes counted in the ledger | test: `c-tests/gpu_tile_attacks.rs`, `gpu_tile_horizon.rs`, `gpu_tile_inflight.rs`, `gpu_tile_tripcount.rs`, `k-tests/gpu_protocol.rs`, `gpu_protocol_audit.rs`, `gpu_tiles.rs`; doc: `docs/gpu-semantic-protocol.md` |
| `verify-last-arrivers` (buffer / global protocol: last-arriver counters, release -> barrier -> acquire) | `gpu_protocol.rs:1322` | VER.4 | `Function` -> findings | g950-only (`sc1`, `buffer_wbl2` / `buffer_inv` memory model) | same | test: `c-tests/gpu_atomic.rs`, `k-tests/lite_contract.rs`; doc: `docs/last-arriver-memory-model.md`; probes: `tools/last_arriver_probe.sh`, `last_arriver8_probe.sh`, `lgab.sh` |
| `verify-gpu-mir` (register class / alignment / limits, MFMA overlap and file, unified budget, occupancy, LDS, `GV-WAIT` / `wait-paths` via `verify_waits`) | `gpu_verify.rs:2665` (`verify_mir`) | VER.5 | `Function`, `MFunction`, `Alloc` -> `Report` | **g950-only**: `VGPR_LIMIT 256`, `SGPR_LIMIT 102`, `LDS_LIMIT 160 KiB`, AGPR 256, even-aligned tuples, s0..s7 / v0 reserved (`:2654-2656`, `:2685-2760`; alignment rules "assembler-measured") | **runs in `compile_with` always** (`enforce`); `Default`: errors here fail the compile; `Warn` / `Strict` / `Off` via `AMDC_GPU_VERIFY` | test: `k-tests/gpu_semantics.rs` (`verify_mir` negative cases), `register_accounting.rs`, `sgpr_legality.rs`; assembler-measured notes in-code |
| `ledger` / trust ledger (counts `assume_trusted`, `tile_trust_periodic`, `region_trust` = `buf_trust`, `asm_opaque`; plus trusted contracts: extents, ka_ranges, grid_defs, ka_defs, ka_sets, sym_ranges) | `gpu_verify.rs:30` (`Ledger`), `Report::ledger` `:99` | VER (summary) | `Report` -> `Ledger` | generic | printed by `amdc gpu-verify` (`amdc:456`), `certify_specialized` (`amdc:126`), `amdc summary` (`diag.rs:265`) | test: `k-tests/trust_inventory.rs` (each escape is an error, a ledger entry, or a counted "bounds-checked, not race-checked" access); doc: `docs/trust-inventory.md` |
| `certificate` (= `analyse` = lower + `verify_ir` + `verify_mir`, strict reading) | `gpu_verify.rs:2827` | VER | `Function` -> `Report` | as above | `amdc gpu-verify <k>|--all|--shipped [--strict] [--json]` (`amdc:399`); `emit` / `jit-info` call it for `--from-ir` and `--specialize` kernels unless `--no-verify` (`must_verify` `amdc:79`; `amdc:985`, `amdc:950`); `amdc ir-check --verify` (`amdc:852`) | test: `crates/amdc/tests/jit.rs`, `contract_host_truth.rs`; `k-tests/gpu_semantics.rs` |
| `verify-schedule-perm` | `sched/verify.rs:47` (`check_perm`) | VER (inside `sched-*`, `hz-fill`) | `MFunction` insts, perm -> `Result` | generic (value-numbering of register dwords incl. vcc / scc + `order::mem_must_order`) | `Options::verify` (default on) | `sched/tests.rs`; `c-tests/slotfill.rs` |
| `verify-asm-lite-contract` | `asm_check.rs:12` | VER (text) | see `emit-asm-check-lite` | g950-only | tests only | `k-tests/lite_contract.rs` |
| `verify-functional-proofs` (online-softmax rebase / rollback algebraic proof) | `functional_proofs.rs:156` (`softmax_invariant_proof`) | VER (stand-alone) | none -> `Result` | generic | not a pipeline pass; test-only | `c-tests/functional_proofs.rs`; doc: `docs/functional-proofs.md` |
| `verify-diagnostics` (statement-localised JSON, `srcmap`, `summary`) | `diag.rs:106-331` | tool | `Report` / error -> `Diag` | generic | `--json`, `amdc srcmap|summary` | test: `k-tests/diagnostics.rs`; doc: `docs/diagnostics.md` |
| `verify-ir-roundtrip` | `amdc ir-check` `amdc:840`; `ir_text::print/parse` | CLI + test | `Function` -> text -> `Function` | generic | `amdc ir-check` | `k-tests/ir_text.rs`, `tools/ir_roundtrip_check.sh` |
| safe-asm block verifier (own wait-state clock) | crate `amdgpu-safeasm` (used through `asm.rs:33-74`, `pin_check`) | VER (IR build time) | asm block -> verdict | g950-only | compile-fail / ui tests | tests: `crates/amdgpu-safeasm/tests/{compile_fail,ui}`; docs `docs/safe-asm-design.md`, `docs/rust-safety.md`; `c-tests/rust_safety.rs` |

Where each runs in the CLI: `emit` -> steps 12 (always, default policy) + optional certificate (0b); `jit-info` -> same via `compile_with`;
`gpu-verify` -> `analyse` (compiles to final MIR, then `verify_ir` + `verify_mir`); `ir-check --verify` -> `analyse` on a parsed file; `check-launch`
(`amdc:499`) evaluates the contract facts for a concrete launch (host truth, `crates/amdc/tests/contract_host_truth.rs`).
**Gated:** all `verify_*` by `AMDC_GPU_VERIFY`; scheduler verifiers by `Options::verify` (`noverify` token); the strict certificate by `--no-verify`.
There is no `emit --verify` flag today; `emit` verifies strictly only for `--from-ir` / `--specialize` inputs.

---

## 4. The Weaver and relatives

Per the plan (L4: "Weaver (region ordering and register organisation)") these move to `amdgpu-codegen`. Today they are split over three crates.

| Piece | Location | Takes | Produces | Tile-level types today? |
|---|---|---|---|---|
| Units / capture | `tile/weave.rs:158-233` (`Units::capture` 177, `anchor` 191, `split` 203, `append` 222) | closures run on a `FnBuilder`; statement facts (`amdgpu_dsl::facts::{Facts, ResClass, must_order}`) | `Unit { class, hint, latency, deps, regs: RegProfile, iter, stmts, summary }` (`:57`) | No: imports only `amdgpu_dsl` (`weave.rs:12-13`). `UnitClass` (Mfma / Valu / LdsRead / Dma / Salu / Branch / Setup) are instruction classes |
| Weaver | `tile/weave.rs:247-346` (`Weaver::order` 279, `seq` 274, `spread` 286, `plan` 309, `emit` 321, `check_order` 346) | `Units`, author order or skeleton + fillers | IR statements in a topologically checked order, ordinary region IR (labelled `weave_region_begin`) | No. Env hook `AMDC_WEAVE_AUTO` consulted inside `plan` (`weave.rs:311-325`) - violates the "no env in kernels" rule; allow-listed |
| Rotation | `tile/weave.rs:366-406` (`rotate`, `RotStage`, `Part`) | parts of a software-pipelined loop | prologue / steady / epilogue IR | No |
| weave-auto | `tile/weave_auto.rs` (`Model::gfx950()` 68, `simulate_items` 186, `plan_items` 238, `auto_order` 406) | units, machine `Model` (issue gap 4, MFMA result 84 cycles, ds_read 125-145, LDS-DMA issue 24 - "calibrated on the warm ATT of the 4-wave FMHA"), register-pressure guard | cost-minimal topological order | No tile types; **gfx950-only constants**; "not a performance oracle below ~3800 simulated cycles per iteration", opt-in, never default (doc: `docs/weaver-auto.md`) |
| Region strategies | DSL `declare_region(name, authored, tokens)` `dsl/lib.rs:1773`, `RegionDecl` `:1611`, `weave_region_begin/end` `:2212,2217`, `Stmt::RegionMark`; compiler: stamping `MInst::region` (`lower.rs:588,2225-2232`), `declared_regions` / `region_strategies` (`lib.rs:308,324`), `sched/region.rs` (`for_region` 30, `runs` 77, `block_allows` 97), `AMDC_SCHED_REGION` | region names -> scheduler token strings | per-run options for `sched-list`, `sched-ldshoist`, `sched-slotfill`, post-passes; `authored` = order untouchable, waits recomputed | No tile types, but tokens are the scheduler vocabulary (so `sched` option tokens are part of the L3->L4 API); keyed by NAME |
| Stage-level hints | `Hint` / `UnitClass` (`weave.rs:19,35`), `Function::compile_profile` (`dsl/lib.rs:1621`: `Default, Gemm, Vs128, T128, KvSplit, Combine`), `Function::sched_opts` / `lower_opts` (`dsl/lib.rs:1550`), `sched_group_barrier` markers (-> `sched-phase`), `StageBegin { name, contract }` (`dsl/lib.rs:1175`) | | per-kernel scheduler / lowering policy | `policy_name` (`compiler/lib.rs:34`) maps a profile back to a **shipped kernel name** and `lower_and_allocate_with` then branches on name prefixes / suffixes (`lib.rs:105-138`) - a layering leak: policy should be a profile -> option-set table in L4 (or in `Target`), selected explicitly |
| Register organisation (`weaver-regs`) | DSL `RegMark` `dsl/lib.rs:1335` (`pin_v`, `share`, `separate`, `reg_cap`, `reg_cap_agpr`, `ring` builders `:2178-2208`); `resolve_reg_plan` `lower.rs:459` -> `MFunction::reg_hints` (`mir.rs:727`); allocator honours pins / separation (`regalloc.rs:1039 Plan`); `regplan::check` `regplan.rs:208`, `report` 180, `region_report` 100; AGPR marks `AgprMark` `:1320` -> `assign-agpr` | register marks | pinned / disjoint / capped allocation, per-region pressure report (`amdc regplan`) | No; doc: `docs/weaver-regs.md` |
| Stage contracts | `StageContract` `dsl/lib.rs:1150`, checked by `verify-stages` | | | Lives in the DSL (L4) already, authored by L2 stages |

Layering conclusion: `weave.rs` + `weave_auto.rs` are self-contained over `amdgpu-dsl`; the tile-level callers are `tile/stage.rs:10,91`,
`pipeline.rs:20`, `dma.rs:137` (`issue_buffer_units`) and kernel code (`fmha/*`, `gemm`, `mlp`, `norm`). Moving them to L4 is a file move plus
re-pointing those imports. They run at **kernel-build time** (IR -> IR), so in the pass manager they are not stages of the compile pipeline but
"builder-time utilities" registered under a separate heading (§6). Four independent machine models exist and should converge on
`Target::sched_model`: `sched/model.rs` (cycle table), `weave_auto::Model::gfx950`, the `slotfill` constants (`slotfill.rs:32-43`), and the hazard
constants (`hazards.rs:19-22,431`); today they are mutually inconsistent by design (each calibrated for its own purpose).

---

## 5. Planned pass-manager pipeline (gfx950)

### 5.1 Data structures (implementable as a pure refactor)

```rust
pub struct Target { /* P0: wave size, register files, LDS, counters, hazard table, latency model, atoms, descriptor format, LLC settings */ }

pub struct CompileOptions {        // replaces scattered env / name logic; one parse of env at the CLI
    pub lower: Vec<String>,        // lowering tokens (profile preset + Function::lower_opts + ladder extra)
    pub sched: Option<sched::Options>,   // scheduler_options() + AMDC_SCHED_ADD applied
    pub m0_reuse: bool,            // m0_reuse_on()
    pub hzfill: bool,
    pub verify_mode: gpu_verify::Mode,   // AMDC_GPU_VERIFY
    pub tail: TailKind,            // own | llvm-tail (resolve_backend)
}

pub struct PassCtx<'a> {
    pub target: &'a Target,
    pub opts: &'a CompileOptions,
    pub func: &'a amdgpu_dsl::Function,      // read-only source IR (verification, diag)
    pub diag: Diagnostics,                   // warnings / notes / stats / stderr lines (today eprintln!)
    pub analyses: AnalysisCache,             // see 5.2
    pub attempt: AttemptInfo,                // which rung of which ladder (for stats / the stderr fallback line)
}

pub enum PassError { Resource(String) /* "out of VGPRs..." -> retryable */, Fatal(String) }   // keeps today's message text byte-for-byte

pub trait Pass<S> {                          // S = Function | MirState
    fn name(&self) -> &'static str;          // stable kebab-case id from §2
    fn evidence(&self) -> &'static [Evidence];   // Evidence::{Test(&str), Probe(&str), Oracle(&str), Doc(&str)}
    fn target_dep(&self) -> TargetDep;       // Generic | Param(&'static str) | Only(&'static str)
    fn enabled(&self, cx: &PassCtx) -> bool { true }   // option gate (skippable by option)
    fn requires(&self) -> &'static [Analysis]; fn invalidates(&self) -> Invalidates;   // §5.2
    fn run(&self, s: &mut S, cx: &mut PassCtx) -> Result<(), PassError>;
}
pub struct MirState { pub m: MFunction, pub alloc: Option<Alloc> }   // alloc = None before RA
pub struct Pipeline { stages: Vec<Stage> }   // fn gfx950_pipeline(opts: &CompileOptions) -> Pipeline  (one constructor per target)
pub enum Stage { Seq(Vec<Box<dyn AnyPass>>), Retry(Ladder), Fallback { primary: Box<Stage>, on: ErrClass, fallback: Box<Stage> } }
```

Rules: a pass sees `&mut MirState`; analyses are computed on demand and cached keyed by `(analysis, mir_generation)`; every mutating pass
bumps the generation unless it declares `preserves` (today there is **no** caching: `liveness::compute`, `liveness::successors` are recomputed per
pass, e.g. `sched/mod.rs:155`, `hazards.rs:110`; keeping recomputation is the byte-safe default, caching is an optimisation to gate behind md5 equality).
`Diagnostics` must reproduce the current stderr (`lower.rs:430` "lds-spill", `lib.rs:176` hzfill line, `lib.rs:238` "sched: falling back") because
tests and Python tooling may parse it; it does not affect asm bytes.

### 5.2 Proposed ordered pipeline

Names are the IDs of §2. `[opt]` = skippable by option; `{...}` = options selecting variants. "Analyses" = results consumed.

**Stage A - source IR (`Function`)**
1. `ir-specialize` [CLI only; before the pipeline] -> `Function`.
2. `verify-ir-structure` (always; today inside `lower-mir`, `lower.rs:85`) - consumes: none.
3. `strip-any-lane` (automatic when markers present).
4. `analysis-uniform-values` -> `Analysis::Uniform` consumed by `lower-mir` when `{scalarize}`; invalidated by IR changes (none in the pipeline).

**Stage B - lowering (`Function -> MFunction`)**
5. `lower-mir` {lowering tokens: profile preset + `lower_opts` + ladder extra}; consumes `Uniform`; produces `MFunction` with `auto` waits, regions, `reg_hints`.
6. `lower-kabatch` [opt `kabatch`], `lower-select-fold` [opt `cmpfold`], `lower-dce` (loads-only always; full [opt `dce`]), `lower-remat` [opt `remat1-4`], `lower-lds-spill` [opt `ldsspill*`], `lower-coalesce`, `lower-dce` (final). Note the current order inside `finish`: kabatch runs in `lower()` **after** `finish` (`lower.rs:112-115`), i.e. after the final `dce`; keep that.
7. `assign-agpr` (always; no-op when no marks) - invalidates all MIR analyses.

**Stage C - policy resolution (not a pass on `MFunction`; computed once per ladder rung)**
8. `resolve-sched-options` (`scheduler_options`, `declared_regions`, `region_strategies` on the post-AGPR MIR, `AMDC_SCHED_ADD`, `hzfill` detection) -> `CompileOptions.sched`, `.hzfill`. Needs `MFunction::reg_hints.regions`, hence after `lower-mir`.

**Stage D - scheduled body (inside the inner ladder rung; on a clone of the Stage-B/C output)**
9. `sched-cfg-cleanup` [opt `nocfg` disables], 10. `waits-strip-auto`, 11. `sched-dce-dead-selects`,
12. `sched-ldshoist` [opt: any region has `ldshoist`; `{hoist=N}`; rung may force `hoist_dist`], 13. `sched-slotfill` [opt `slotfill`],
14. `sched-phase` (markers; always strips), 15. `sched-list` [opt `reorder` or region `reorder`; `{limit=N}`; `{xbar}`], consumes `Liveness` (block-level), `DepGraph` (per block, built inside),
16. `sched-wsink` [opt `wsink`], 17. `sched-ldahead` [opt `ldsink`/`ldahead32`], 18. `waitcnt-recompute`, 19. `verify-waits` [opt `noverify`] (Options::verify).
   Exactly the sequence of `sched::run`; propose to flatten it into 11 passes of one `sched` stage while keeping `sched::run` as a compatibility wrapper for the unit tests (`sched/tests.rs`, `k-tests/sched_kernels.rs` call `sched::run` directly).
20. `m0-reuse` [opt: Gemm profile], 21. `m0-explicit`, 22. `verify-pinned-blocks`,
23. `hazard-value` {strict = true}, 24. `verify-mir-structure`,
25. `regalloc` -> sets `MirState.alloc`; internal policy ladder scan -> scan-top -> two-phase is **inside** the pass (one pass, three strategies; reported as `regalloc{scan,scan-top,two-phase}`), 26. `regplan-check` (inside `allocate` today; separate pass gated on `reg_hints.active()`), 27. `check-workgroup-budget` -> `PassError::Resource` (prefix `"out of VGPRs for the launch: "`).
28. `hazard-inflight-waw`, 29. `hazard-phys-war`, 30. `hz-fill` {strict=true} [opt `hzfill`], 31. `m0-pad`, 32. `hazard-valu-sgpr`. Fixed order: `hz-fill` re-runs the hazard passes on a copy and must see the others' padding; `m0-pad` and `hazard-valu-sgpr` come after `hz-fill`, so `hz-fill` cannot see them.

**Stage E - unscheduled fallback body (same shape, different parameters; entered when Stage D exhausts its ladder, or when `sched` is `None`)**
33. `sched-phase{strip}` (just `phase::strip`), 34. `m0-reuse` [opt], 35. `m0-explicit`, 36. `verify-pinned-blocks`, 37. `hazard-value` {strict = false},
38. `verify-mir-structure`, 39. `verify-waits` (hard error; not a retry class), 40. `regalloc` (+ `regplan-check`), 41-45 = 28-32 with `hz-fill{strict=false}`.
   Pass manager: Stage D and E share the **same pass objects** for 20-24 and 28-32; they differ by `PassCtx::path = Scheduled | Unscheduled`, which
   selects `hazard-value.strict`, `hz-fill.strict`, whether Stage D's 9-19 precede, and whether `verify-waits` runs before or after RA (D: inside 19; E: after 24).

**Stage F - final checks and tail**
46. `check-workgroup-budget` (hard), 47. `verify-gpu-ir` + `verify-barrier-balance` + `verify-stages` + `verify-protocol-tiles` + `verify-last-arrivers` + `verify-gpu-mir` under `enforce` [opt `AMDC_GPU_VERIFY=off`; policy chooses which findings are errors],
48. `emit-own` | `emit-llvm-tail` {tail}.

**Report-only (not in the compile pipeline, listed by `amdc passes --reports`)**: `analysis-pressure`, `analysis-nop-classes`, `sched-sim`, `verify-diagnostics`.

### 5.3 Entangled spots and how the pass manager represents them

1. **Two nested retry ladders that re-run a prefix** (the main one).
   * Outer ladder (`lib.rs:77`): 11 rungs; each rung re-executes **everything from `lower-mir`** with extra lowering tokens; retryable iff the error contains `out of VGPRs`.
     The ladder is not just "try again": rung `k` changes `CompileOptions.lower` and the `LDS spill > 160 KiB` check (`lib.rs:80`) runs after the winning rung.
   * Inner ladder (`lib.rs:185-237`): up to 1 + 3 + 3 + 3 = 10 rungs, built from the options (`ldshoist` adds 3, `reorder` adds 3 + up to 3), each rung
     clones the post-`assign-agpr` `base` and re-runs Stage D; a rung changes only `max_vgprs` (cut) and `hoist_dist` / `ldshoist`.
     When all rungs fail, Stage E runs on the original `base` (moved, not cloned), after one stderr line.
   * Representation: `Stage::Retry(Ladder { rungs: Vec<Rung>, retry_on: ErrClass::Resource, snapshot: SnapshotPolicy::CloneInput, body })`. `Rung` is a pure patch
     of `CompileOptions` (outer: `lower_extra: &[&str]`; inner: `{max_vgprs_cut, hoist_dist}`); rungs are generated by `Ladder::for_gfx950(opts)` **using the exact
     lists in `lib.rs:77,185-196`**, including the launch-budget rungs that depend on `launch_vgpr_budget(workgroup_size)` (`sched/mod.rs:298`).
     `Stage::Fallback { primary: Retry(inner), on: Resource, fallback: Stage E }` expresses "scheduler gives up -> unscheduled". The nesting is
     `Retry(outer, [B..F-without-tail] )` with the inner `Fallback` inside it; a `Resource` error in Stage E propagates to the outer ladder.
   * `PassError::Resource` replaces the string test `e.contains("out of VGPRs")`; both `regalloc` and `check-workgroup-budget` (which today prefixes `out of VGPRs for the launch`) return it.
     The error text must stay identical because `diag::from_compile_error` (`diag.rs:165`) parses it.
2. **Policy by kernel name** (`policy_name`, `lib.rs:34,105-138`): option sets are chosen from `f.name` prefixes/suffixes (`fmha_fwd_hd128_mxfp8_vs*`, `*_kvsplit`, `*_g2`, `*_t128`, `v3_` names) in addition to `compile_profile`. A refactor must keep the table (a `PolicyTable`), not infer it from profile alone, or md5s change for kernels without a profile.
3. **Duplicated post-RA tail with different parameters** (Stage D vs E, §1): keep one implementation, parameterised by `path`; the `strict` flag changes `hazard-value` output (block-end padding) and therefore asm.
4. **`hz-fill` re-runs hazard passes** (`hzfill.rs:10-13`, "after the moves of a block the value-hazard and physical-WAR passes are re-run on a copy"): `hz-fill` calls other passes as functions. The pass manager must expose those two as library calls (not only registered passes) or give `hz-fill` a sub-pipeline.
5. **`regalloc` contains a pass** (`regplan::check`) and a retry (three allocators) and a verifier (`check_reaching` for two-phase). Split `regplan-check` out only after md5 equality is proven; the allocator's internal fallback stays inside `regalloc`.
6. **Lowering contains its own pipeline** (`finish`, `lower.rs:373-456`) and calls `insert_waits` after LDS spill (`lower.rs:431`), i.e. a post-scheduler-style pass run mid-lowering. Splitting `lower-*` into MIR passes must keep that call attached to `lower-lds-spill`.
7. **`verify-waits` runs twice with different meaning**: inside `sched::run` (before the m0 / hazard passes, on scheduled MIR) and on the fallback path after `mir::verify`; and again as `GV-WAIT` in `verify-gpu-mir` on the final MIR (post-padding). Three call sites, one function.
8. **Thread-local mutable state**: `lower::with_opts` override (the ladder extras travel through it, `lower.rs:53-58`), `order::set_relax_dma_lds_read` (slotfill sets and unsets it), `LAST_ORIGINS`. The pass manager's `PassCtx` should own these; until then passes must keep the same set/unset discipline.
9. **Pass ordering dependencies that are implicit**: `edgepad` (lowering option -> `MFunction::edge_pad`) influences `hazard-value` and the emitter; `elide_fallthrough` is set by `sched-cfg-cleanup` and read by `emit-own` only; `MInst::skip_m0` produced by `m0-*`, consumed by `emit-own` and `llvm_mir`; `MInst::auto` produced by `lower-mir`, stripped by `waits-strip-auto`, optionally ignored by `LlvmTail::drop_auto_waits`. Record these as `PassContract` fields (produces / consumes MIR flags) in `evidence()`-adjacent metadata so `amdc passes` can print them.

---

## 6. `amdc passes --target gfx950` output sketch

New subcommand in `amdc` (next to `list`, `amdc:393`): no kernel needed for the default listing; with a kernel it lists the **resolved** pipeline
(which passes are enabled and which rungs exist for that kernel). Columns: index, stage, pass, target dependence, gate, evidence counts.

```
$ amdc passes --target gfx950
pipeline gfx950  (SKETCH - counts and evidence numbers below are illustrative, not measured)
  #   stage   pass                    dep           gate                          evidence
  1   source  verify-ir-structure     generic       always                        test:14
  2   source  strip-any-lane          generic       auto                          test:0
  3   lower   lower-mir               gfx950-only   lowering tokens               test:20 probe:3 oracle:2 doc:2
  4   lower   lower-select-fold       gfx950-only   cmpfold (default on)          test:1
  ...
 10   mir     assign-agpr             gfx950-only   always                        test:1 probe:1 doc:1
 11   sched   sched-cfg-cleanup       generic       sched (+ not `nocfg`)         test:0
 ...
 18   sched   sched-list             gfx950-param  sched.reorder                 test:6 doc:1 probe:2
 21   sched   waitcnt-recompute       gfx950-only   sched                         test:4 doc:1
 24   mir     hazard-value            gfx950-only   always, strict=sched path     test:5 oracle:2 probe:1
 26   ra      regalloc                gfx950-param  always                        test:4
 29   post    hazard-phys-war         gfx950-only   always                        test:3
 30   post    hz-fill                 gfx950-only   token `hzfill`                test:1
 34   verify  verify-gpu-mir          gfx950-only   AMDC_GPU_VERIFY != off        test:3
 41   emit    emit-own | emit-llvm-tail   gfx950-only | param   --backend       test:5 oracle:3

retry ladders
  L1 lower-extra   [-, remat1, remat2, remat3, remat4, imm+ldsspill8, ... ldsspill64]  retry on: out of VGPRs
  L2 sched-attempt [(cut 0, hoist -), (0,24), (0,8), (0,0), (16,0), (32,0), (48,0), launch-budget x3]  then: unscheduled fallback
$ amdc passes --target gfx950 --kernel fmha_fwd_hd128_mxfp8_v3_t128 --evidence
  lower-mir        tokens: dce cmpfold scalarize retarget2 imm sgpr zeroinit addrfold saddrlds edgepad salubr
  sched-list       limit=248 xbar  (preset T128)
  ...
  hazard-value     tests: crates/amdgpu-codegen/tests/hazard_classes.rs, hazard_oracle.rs, smfmac_hazards.rs
                   oracle: tools/hazard_oracle.sh   probe: tools/mfma_dep.sh
$ amdc passes --target gfx950 --json     # machine-readable (name, stage, target_dep, gate, evidence[])
$ amdc passes --check                    # fails when a pass has no evidence entry (see §7, none found list)
```

Additions the output should support: `--json`; `--check` (every pass must list >= 1 evidence entry or an explicit `waived: <reason>`); `--print-ladder`; a
`--target gfx1250` stub that prints `pipeline not implemented (spec only)` (phase G is spec-only).

---

## 7. Open questions and risks

1. **Golden md5 manifest does not exist in the repo.** Only `tools/md5_kernels.sh` (own backend, `amdc list` kernels). The plan also asks for "every dispatch-reachable instantiation"
   and certificates; `k-tests/dispatch_compile.rs` compiles them but does not hash. Needed before step 1 of P1: manifest of (kernel, instantiation) -> md5 of `.s` **and** per-kernel certificate text; include `--backend=llvm-tail` md5s only where the pinned llc is available (it is not on this machine).
2. **Environment is part of the compiler's input.** `AMDC_SCHED`, `AMDC_SCHED_ADD`, `AMDC_SCHED_REGION`, `AMDC_M0_*`, `AMDC_NO_FILL_VCC`, `AMDC_GPU_VERIFY` change output; the manifest must be generated with a scrubbed environment, and P1 should collect them into `CompileOptions` (and keep the same names for back-compat).
3. **Name-based policy** (`policy_name`, `lib.rs:34`, 105-138) means some kernels get presets via their name, not their profile; an explicit policy table must reproduce it exactly (includes `v3_` name prefix special case at `lib.rs:119`).
4. **Retry ladders mutate shared state through thread-locals** (`lower::with_opts`); a pass manager that runs passes on multiple threads (the compiler is currently single-threaded per kernel; `cargo test` runs tests in parallel threads, which is why `with_opts` is thread-local) must keep that property.
5. **No tests found** for: `sched-cfg-cleanup`, `sched-dce-dead-selects`, `lower-kabatch`, `lower-remat`, `lower-coalesce`, `strip-any-lane`, `m0-reuse` / `m0-explicit` / `m0-pad` (only incidental M0 mentions), `analysis-nop-classes`, `sched-sim` (reporting; `amdc simmodel`). `amdc passes --check` would flag these; the evidence request should be answered by adding md5-pinned tests, not by editing the passes.
6. **Probes are evidence of rules, not of passes.** `crates/amdgpu-kernels/src/probes/` (`mod.rs`, `tileops.rs`, `handoff.rs`) and `tools/*probe*.sh` validate ISA facts; mapping them to a specific pass (e.g. `probe_valu_sgpr` -> `hazard-valu-sgpr`) is only stated in comments (`hazards.rs:998`). The `evidence()` lists need a curated mapping; this document gives the best-effort one from `git grep`.
7. **Target boundary for `Target` (P0).** What clearly belongs in `Target`: register files and reservations (`regalloc.rs:20-29`), limits in `gpu_verify.rs:2654-2656`, `launch_vgpr_budget` (`sched/mod.rs:298`), counters (`waits.rs:14`), hazard tables (`hazards.rs:19-60,431,455`; `op_attrs` `mfma.latency`), `sched/model.rs`, slotfill and weave-auto models, `emit.rs` descriptor, `backend/llvm20.rs` LLC settings. What does **not** fit a flat table: wait handling (gfx1250 has split counters LOAD / STORE / DS / KM / ASYNC / TENSOR / X, so `waitcnt-recompute` and `hazard-inflight-waw` are target-specific *algorithms*, not parameters), M0 handling, MFMA vs WMMA hazards, wave32. Recommendation: `Target` carries a `trait TargetPasses { fn pipeline(&self, opts) -> Pipeline }` and a data part; passes tagged `gfx950-only` stay in `target/gfx950`, `generic` / `g950-param` stay in `amdgpu-codegen`.
8. **`MOp` is the ISA.** `mir.rs` / `op_attrs.rs` (1166 lines) / `emit.rs` (1234 lines) / `backend/llvm20.tbl` are gfx950's instruction set. For gfx1250 the enum cannot be shared as-is; this is a P0 `Target` / atoms decision that blocks every `g950-only` row.
9. **`hzfill` and the post-RA ordering** (§5.3 item 4) make any reordering of `hazard-*` passes unsafe even where they look commutative; `pad_physical_war`'s comment ("Only run for scheduled code") is stale (it runs on both paths) - fix the comment, not the behaviour.
10. **`gpu_verify::enforce` default mode** only fails on MIR errors; IR-level errors are collected silently. If `amdc passes` advertises "verified", state which mode. Changing the default would break kernels without a contract (not a pure refactor).
11. **Where the `.co` step lives.** The plan's "from Function to assembly text / .co" has no `.co` code in Rust; if P1 wants a `finalize` pass (`clang`), it is a new capability (Python JIT does it today), not a refactor.
12. **Size of `lower.rs` (5243 lines) and `gpu_verify.rs` (3046 lines)**: splitting them is the bulk of P1's mechanical work; each split must be a move-only commit with the md5 manifest unchanged (plan rule: no behaviour change mixed into a move).

## 8. As built (branch `pass-mgr`)

- `crates/amdgpu-codegen/src/pass/mod.rs`: `PassInfo` (name, stage, target dependence, gate, evidence), runnable MIR passes `Pass { info, run }`,
  the gfx950 bodies `SCHEDULED` / `UNSCHEDULED` (the same pass objects, `Path` selects `hazard-value` / `hz-fill` strictness), the descriptions
  `FRONT` (source + lowering + `assign-agpr`) and `BACK` (final checks, tail), the ladders `OUTER_LADDER` / `inner_ladder` and `retryable`;
  `lower_and_allocate` runs them. A unit test requires an evidence entry for every pass.
- `amdc passes [--target gfx950] [--evidence]` prints the pipeline and the ladders (`--target gfx1250`: not implemented, specification only).
- The compile policy is declared data: `amdgpu_dsl::CompileProfile::policy()` → `CompilePolicy { lower_base, lower, m0_reuse, sched_preset }`
  (`Function::compile_policy`); the compiler no longer inspects kernel names (the only name-specific extras, `saddrlds,edgepad,salubr` of the v3
  `_t128`, are declared by that kernel as `lower_opts`). `policy_name` remains only for the `AMDC_SCHED_ADD` experiment matching.
- Every step: md5 manifest 0 differences (234 kernels), certificates unchanged.

## 9. Current pipeline (Main ed666ee4, `amdc passes --target gfx950`)

63 rows: 16 front (3 source, 12 lowering incl. `lower-movb64`, `lower-jthread`, `lower-coldlayout`, then `assign-agpr`), 25 scheduled, 19
unscheduled, 3 back. Changes since section 8:
- **scheduled body:** `sched-cfg-cleanup`, `waits-strip-auto`, then `bweave-order` (the back-end Weaver: authored order plus the interleave
  hints of Free phases, docs/two-weavers.md section 7), then the scheduler passes as before; `hazard-soft-clause` closes the body.
- **unscheduled body:** `bweave-order`, `sched-phase-strip`, M0 / pin / `hazard-value` / `verify-mir-structure` / `verify-waits`, `regalloc`
  with lowering's waits in place, then `waits-strip-auto`, `waitcnt-recompute`, `hazard-value` and `verify-waits` again on the allocated order
  (W3 B2), then the post-RA hazards.
- **wait pass:** with a plan whose Free phase moves an LDS-DMA across iterations, `waitcnt-recompute` / `verify-waits` add the ring-slot
  dependences (`RingModel`, docs/w3-b5-b6.md).
- **retry ladders:** unchanged (outer: `remat1..4`, `ldsspill8..64`; inner: pressure cuts / hoist caps / launch budget); before the
  unscheduled fallback, a rung that keeps the schedule except in the Free phases (B6c).
- **verification:** `tests/emit_effects.rs` (each expansion's implicit SCC / VCC effects declared), GV-PLAN-RING.

## Appendix: counts

Catalogued: 7 pre-lowering IR / analysis rows, 7 lowering rows, 1 AGPR row, 13 scheduling rows (+4 scheduler analyses), 3 MIR check / hazard rows,
4 allocation rows, 6 post-RA rows, 4 emission / tail rows, 13 verification rows. Entangled spots: 9 (§5.3). Two retry ladders (11 outer rungs, up to 10 inner).
