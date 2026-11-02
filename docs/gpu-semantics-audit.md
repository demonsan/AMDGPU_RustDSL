# Independent adversarial audit: gpu_verify v1.3 - audit of record

Auditor: `kernel-occ`. Branch: **`gpu-semantics-occ`** (this doc). Tested verifier: **Main `2c8b27b`**
(v1.3: period/horizon gate, path exploration, tile markers in shipped kernels, certificate line).
Full workspace per lead: **165 tests / 31 targets pass**; on the tested tree the verifier suites are
**28/28 `gpu_semantics` + 20/20 `gpu_tiles`**, and my audit suites (`gpu_tile_attacks`,
`gpu_tile_tripcount`, `gpu_tile_horizon`, `contract_host_truth`) pass. Hardware: MI355X/gfx950,
node 53, GPU 6 (probes only).

## 1. Attack verdict table

| attack | expected | actual on 2c8b27b | verdict |
|---|---|---|---|
| `gv_attack_lds_const` (declared 256 B, read at 512) | reject | `GV-LDS-OOB`, `[512,512]` vs `[256,256]` | PASS |
| `gv_attack_desc_over` (64 B extent, `num_records=4096`) | reject | `GV-BUF-OVER` | PASS |
| `gv_attack_clamp_path` (clamp only on one branch) | reject | `GV-GLB-OOB`, `[0,1008]` vs `[512,512]` | PASS |
| `gv_attack_assume_loop` (assume `[0,0]`, `set_var`=+1) | reject | `GV-ASSUME-VIOLATED` (1 error), `--strict` rc=1 | PASS |
| `gv_attack_assume_trusted` (same, trusted reason) | pass but listed | pass, 2 `GV-ASSUME` Info `TRUSTED` lines with reason, counted in findings | trust boundary |
| `gv_attack_false_contract` (1 MiB declared, 64 B host pointer) | flag/unproven | pass, 2 globals "proven" | trust boundary (host contract) |
| `gpu_tile_tripcount::late_race` (`Select` marker, race at it>=5) | reject | `GV-TILE-UNROLL-LIMIT` **and** `GV-TILE-RACE`/`GV-TILE-STALE`; `--strict` fails | PASS |
| `gpu_tile_horizon::affine_period_256_is_now_rejected` (slot `(nxt&1)+((nxt>>8)&1)`) | reject | `GV-TILE-UNROLL-LIMIT`: period up to **512** (shift bit 8) needs 1026 iterations, horizon capped **70**; `tile-reads` not certified | PASS |
| same with `tile_trust_periodic` escape (`Select` race at it=256) | trusted, listed | pass 0 errors, one `GV-ASSUME` Info `tile_trust_periodic: TRUSTED ... reason ...`, counted | trust boundary |
| non-counter marker version | reject | `GV-TILE-INDEX` | PASS |
| DMA producer without `wait_vm(0)` | reject | `GV-TILE-STALE` in all simulated iterations; with wait: clean | PASS |
| slots=1 overwrite, read-before-publish, role-divergent barrier, alias without `lds_alias`, marker escape | reject | `GV-TILE-RACE`/`GV-TILE-STALE`/`GV-TILE-OVERLAP`/`GV-TILE-ESCAPE`/`GV-BAR` | PASS |

**Period-bound attack result (new):** the bound is conservative for the slot dimension - the period-256
slot attack is now correctly gated. Two *version-only* forms are still under-bounded:
`version = nxt ^ 256` has real period 512, but `period_bits` reports "period dividing 2" (marker XOR
constants are not counted); `version = nxt << 8` / `nxt * 256` have no finite period, yet are accepted
and also claim period 2 (`SHL`/`MUL` are not counted by `period_bits`). In the protocols I built the
simulator still caught an early stale read (XOR) or the protocol was correct despite the false period
claim (SHL/MUL), so I did **not** demonstrate a certified OOB from this under-approximation; it is a
soundness-of-claim gap to close (count `XorI32` constants and `Shl`/`MulI32` in `period_bits`, or
restrict version forms). The escape `tile_trust_periodic` is listed and countable but can hide a real
late race, exactly like `assume_trusted`.

## 2. Shipped strict certificate (Main 2c8b27b, `--shipped --strict`)

`amdc gpu-verify --shipped --strict` exits **0: 0 errors on all 8 shipped kernels**, with LDS/tile
coverage and a countable trust budget:

| kernel | errors | tile-reads | certificate (trusted assumes / contracts / unproven) |
|---|---:|---:|---|
| `_vs128` | 0 | 16640 | 6 / 29 / 0 |
| `_vs5` | 0 | 16640 | 6 / 29 / 0 |
| `_vs9` | 0 | - | 6 / 29 / 0 |
| `_vs` | 0 | - | 4 / 29 / 0 |
| `_t128` | 0 | - | 4 / 29 / 0 |
| `_t` | 0 | - | 4 / 29 / 0 |
| `_kvsplit` | 0 | 66560 | 12 / 36 / 0 |
| `_kvsplit_combine8h` | 0 | - | 2 / 6 / 0 |
| **all listed** | **0** | | **44 / 216 / 0** |

`tile-cover` is 600 for vs128/vs5/kvsplit; `GV-TILE-UNTAGGED` is a **Warn** on kvsplit (160 LDS
statements outside a marker scope, lifetime not checked) and does not gate strict. The earlier `_t`
false-positive LDS error is gone after the +128 B LDS fix. The 44 trusted assumptions come from 10
reason instances (two families: tile counter bounded by `nkv`/loop exit, split counter bounded by
`splits`); the 216 contracts are the host `arg_tensor`/`ka_is`/`grid_is` declarations. ASM md5
invariants reported by the lead: vs128 `00d34086`, t128 `f83c02ce`, kvsplit `148b329d`, combine8h
`19bdd5df`, `_t` `2d7db16e` with identical outputs.

## 3. Host-truth check

`tests/contract_host_truth.rs` runs directly on Main `2c8b27b` (annotations are now merged): **2/2
pass**. For B1H8S2048, B4H8S2048 and B1H2S200 it evaluates every `Function.contract` extent,
`grid_is`, `ka_is`, `ka_range` and `sym_range` against the `ref_host.cpp`/`attn_host.cpp` formulas:
Q/K/V/O, Q/K scale logical + zeroed slack, V scale, split `grid_z = B*splits`, Ohat/lse
`splits*rows_total*128*4`/`*4`, combine `out=rows_total*128*2`. **No declared logical/alloc extent
overstates the host allocation; all grid/ka relations match the launch config.** Remaining declared
contract gaps: `kvs_counters`@0x288 has no `arg_tensor` (fused path); the split contract does not
state `splits in {2,4,8}`; the 160 untagged kvsplit LDS accesses are outside the tile lifetime model.

## 4. Hardware confirmations (GPU 6, unchanged)

* MUBUF OOB (`tools/buffer_probe.sh both`): **BUFFER_OK** rw/lds/dyn - OOB loads zero, OOB stores
  dropped, runtime descriptor range check holds.
* LDS OOB (`probe_lds_oob`, 3 runs): +256 reads 0, +1024 reads **0x3f800000 stale non-zero** (not
  zero-filled), write +4096 not observable, no fault -> no hardware LDS bounds check; the verifier
  must reject OOB by construction.
* MFMA alignment: assembler-measured even-tuple rule in v1.1+; v52/v4 accepted, v1 rejected.

## 5. Final soundness statement (for README)

**"The compiler is GPU-semantics-aware and proves the properties it claims."** On gfx950 (MI355X),
`amdc gpu-verify --shipped --strict` certifies all 8 shipped FMHA kernels with **0 errors** and proves
every LDS/DS/DMA access in its declared segment (16,640 tile reads for vs128/vs5, 66,560 for kvsplit,
600 tile-covered accesses), every global buffer access against a host-truth-checked contract
(2/2 contract-host-truth tests, 216 declared allocations, no overstatement), every VGPR/SGPR
allocation against the assembler-measured limits/alignment (including MFMA tuple rules), all
barrier/role balance, and all path-sensitive/loop-carried wait-before-use; the LDS tile lifetime model
proves overlap/escape/kind/stale/race/DMA-completion and rejects non-counter or non-periodic markers,
with a period/horizon bound (2^M capped at 70 iterations) that gates the period-256 attack. The proof
is **conditional on a small, explicit, countable trust budget**: 44 `assume_trusted` facts (two
documented exit-counter reason families), 216 host allocation/shape contracts, and the
`tile_trust_periodic` escape - all printed under `-v` and counted in the per-kernel certificate line
(`N trusted assumes, M trusted contracts, K unproven`; K=0 on every shipped kernel). Known honest
gaps: raw safe-asm interiors are not bounds-checked (`GV-ASM` Info); untagged LDS accesses are
Warn-only (160 in kvsplit); same-tile partial-byte overlap is not distinguished; wave-uniform runtime
`if` with barriers/markers is Warn; `period_bits` under-approximates version-only `^`/`<<`/`*`
markers (no certified exploit found); `kvs_counters` still lacks an `arg_tensor`; i32 wrap-around,
cross-workgroup races and >2 wave roles are out of scope. Within those limits, the v1.3 certificate is
sound and the shipped FMHA set is clean.
