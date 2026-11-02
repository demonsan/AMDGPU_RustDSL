# Independent adversarial audit: gpu_verify v1.2 (periodicity gate + tile lifetime)

Auditor: `kernel-occ`. Branch: `gpu-semantics-occ`. Verifier commit: **Main `bb4fce0`** (v1.2
periodic-counter gate, certificate line). Host-truth integration: Main `bb4fce0` + kernel-opt
annotations, branch **`truth-final` @ `080f406`**. Hardware: MI355X / gfx950, node 53, GPU 6.

On `bb4fce0`: **28/28 `gpu_semantics` + 15/15 `gpu_tiles` pass**. On `truth-final`:
`contract_host_truth` **2/2 pass**.

## 1. Assume layer (unchanged, re-verified)

| attack | result on bb4fce0 |
|---|---|
| `gv_attack_assume_loop` (assume `[0,0]`, `set_var`=+1) | **REJECT** `GV-ASSUME-VIOLATED`, `--strict` rc=1 |
| `gv_attack_assume_trusted` (same, trusted reason) | **pass**, 2 `GV-ASSUME` Info `TRUSTED` lines with reason, counted |
| `gv_attack_false_contract` (1 MiB declared, 64 B real) | **pass**, 2 globals "proven" - host-contract trust boundary |
| `gv_attack_lds_const`, `gv_attack_desc_over`, `gv_attack_clamp_path` | **all REJECT** with the expected codes |

`certificate: N trusted assumes, M trusted contracts, K unproven` is printed per kernel and for the
whole list; on pure Main (no annotations) `_vs128` shows `0 trusted assumes, 0 trusted contracts,
106 unproven`, which is the honest state.

## 2. v1.2 tile-periodicity gate - new results

**The old `late_race` is fixed:** a marker using `Select`/`Cmp` is rejected with
`GV-TILE-UNROLL-LIMIT` ("not an affine/periodic function ... at most 10 iterations"), and the
simulation (now 10 iterations) also finds the race:
`[GV-TILE-RACE] read of buf[1] ... overlaps write version 6 ...`, plus `GV-TILE-STALE`; `--strict`
fails. Good.

**New finding - the affine gate accepts periods longer than the 10-iteration horizon.**
`gpu_tile_horizon::affine_period_256_is_missed`: producer slot =
`(nxt & 1) + ((nxt >> 8) & 1)`, version = `nxt`; this is affine/bitwise of the counter, so the gate
accepts it, and the simulator sees only the first 10 iterations. At iteration 255 the extra bit flips
the producer slot onto the consumer's slot with the next version (real race), yet `bb4fce0` returns
**no tile error** (`tile-reads: 55`, no `GV-TILE-UNROLL-LIMIT`), so a launch with >=257 iterations is
certified although it races. **This is a conditional soundness hole:** the gate checks the FORM of
the marker, not that the simulation horizon covers its period. A rigorous gate must bound the
period/cone (e.g. reject shifts/masks that can have period > `nmax`, or simulate an iteration count
exponential in the expression's bit-width) or state the horizon as a trusted assumption.
Non-affine-but-periodic forms (`Select` with period 2) are conservatively rejected - a false positive,
but safe.

**`tile_trust_periodic(reason)` escape:** it skips the gate, is listed once as
`GV-ASSUME` Info `tile_trust_periodic: TRUSTED ... reason ...`, and is counted. With it a
`Select`-based marker whose race starts at iteration 256 is **accepted with 0 errors**, i.e. the
escape can hide a late race. It is a documented trust boundary, like `assume_trusted` and host
contracts; reviewers must inspect the TRUSTED list, `--strict` cannot.

**Correct v1.2 tile rejections:** non-counter marker version -> `GV-TILE-INDEX`; DMA producer without
`wait_vm(0)` -> `GV-TILE-STALE` in all 1..10 unrolls, with wait clean; slots=1 overwrite ->
`GV-TILE-RACE`; read-before-publish, role-divergent barrier, alias without `lds_alias`, marker
escape -> rejected. Remaining documented limits: same-tile partial-byte overlap conflated; untagged
LDS `Warn`; wave-uniform runtime `if` with barriers/markers `GV-TILE-UNSUPPORTED` (Warn); >2 roles
and cross-workgroup out of scope.

## 3. Host-truth check and final verdict table

`tests/contract_host_truth.rs` on `truth-final` (Main `bb4fce0` + annotations) evaluates every
`Function.contract` extent, `grid_is`, `ka_is`, `ka_range`, `sym_range` for 3 shapes against
`ref_host.cpp`/`attn_host.cpp`: Q/K/V/O, Q/K scale logical+zeroed slack, V scale, split
`grid_z = B*splits`, Ohat/lse `splits*rows_total*128*4`/`*4`, combine `out=rows_total*128*2`.
**2/2 pass; no declared logical/alloc extent overstates the host allocation; all grid/ka relations
match.** Remaining contract gaps: `kvs_counters`@0x288 still has no `arg_tensor`; split domain is not
constrained to {2,4,8}; `rows_total` is now tied to B*S*H.

With the annotated kernels + the pending kernel-opt verifier fixes, `amdc gpu-verify` gives:

| kernel | errors | certificate | notes |
|---|---:|---|---|
| `_vs128` | **0** | 6 trusted assumes, 29 trusted contracts, 0 unproven | LDS600 + tile-cover600 + barriers18 |
| `_t128` | **0** | 4 trusted assumes, 29 trusted contracts, 0 unproven | LDS336 |
| `_kvsplit` | **0** | 12 trusted assumes, 36 trusted contracts, 0 unproven | LDS760 + tile-cover600 |
| `_kvsplit_combine8h` | **0** | 2 trusted assumes, 6 trusted contracts, 0 unproven | global3 |
| `_t` | **1** | 4 trusted assumes, 29 trusted contracts, 1 unproven | one `GV-LDS-OOB`: `ds_write_b32` at address `[67584 | 66450 | 67582 | 66448, 69756 | 362364 | 657276 | 70268]` vs 69632 B - a certified-safe kernel; analysis precision, not a contract error |

The `_t` case is the only remaining shipped false positive; it is a real LDS address built from
`Lshr/Shl/Xor` of loop values and is safe (kernel certified), but the interval engine cannot close the
bound. All other shipped kernels are clean on the integration branch. Earlier plain Main without
annotations had `_vs128` 106 unproven (all `GV-GLB-*`), showing why the contract layer is required.

## 4. Hardware confirmations (unchanged, GPU 6)

* MUBUF OOB (`buffer_probe.sh both`): **BUFFER_OK** rw/lds/dyn; zero-fill and dropped OOB stores hold.
* LDS OOB (`probe_lds_oob`, 3 runs): read +256 = 0, read +1024 = **0x3f800000 stale non-zero** (not
  zero-filled), write +4096 not observable, no fault. LDS has no hardware bounds check.
* MFMA alignment: Main v1.1+ uses the assembler-measured even-tuple rule; earlier v52/v4 accepted /
  v1 rejected evidence matches.

## 5. Final soundness statement (v1.2)

**Provably rejects:** LDS DS/DMA segment overflow/wrap; unbounded ring/loop vars; global
extent/no-extent/provenance and one-path clamps; descriptor `num_records` overstatement; barrier
balance; VGPR/SGPR limits, tuple/descriptor alignment and MFMA overlap; path-sensitive/loop-carried
wait-before-use; tile overlap/escape/kind/stale/race/DMA-completion and non-counter markers; non-affine
or non-periodic marker forms (`GV-TILE-UNROLL-LIMIT`); non-inductive `assume` (`GV-ASSUME-VIOLATED`).
`certificate` counts every trusted assume/contract and every unproven fact.

**Trusted / still not proven:** host contracts (`arg_tensor`/`ka_*`/`grid_is`); `assume_trusted` and
`tile_trust_periodic` escapes (listed/countable, but can hide real OOB/race); the v1.2 periodicity
gate's 10-iteration horizon is not a soundness guarantee for affine markers whose period exceeds it
(my `affine_period_256` attack is certified although it races at it=255); raw safe-asm interiors
(`GV-ASM` Info); untagged LDS lifetime (Warn); partial-byte tile overlap conflated; runtime if with
barriers `Warn`.

**Out of scope:** cross-workgroup races/memory model; i32 arithmetic wrap-around; hardware faults for
OOB LDS/MUBUF; protocols beyond two wave roles. Consequently v1.2 is a strong finding engine and
certifies the shipped FMHA kernels except `_t` (precision), but **the affine long-period tile gap must
be closed (or the horizon made explicit/trusted) before "any loop trip count" can be claimed**. The
periodicity gate fixed the original `late_race` but not this larger-period variant.
