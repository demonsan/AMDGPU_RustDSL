# Independent adversarial audit: gpu_verify v1.1 (tile lifetime + assume soundness)

Auditor: `kernel-occ`. Branch: `gpu-semantics-occ`. Re-audited verifier: **Main `4c4909a`**
(v1.1 tile model + `GV-ASSUME-VIOLATED` + `assume_trusted` + assembler-measured alignment).
Host-truth integration: Main + kernel-opt annotations (`gpu-semantics-opt@5f923d8`, branch
`truth-main`). Hardware: MI355X / gfx950, node 53, GPU 6.

Verifier tests on `4c4909a`: **28/28 `gpu_semantics` + 11/11 `gpu_tiles` pass** (full Main per lead:
156 tests /31 targets). Host-truth tests on `truth-main`: **2/2 pass**.

## 1. Soundness attacks - assume layer

| attack | expected | actual on Main 4c4909a | verdict |
|---|---|---|---|
| `gv_attack_assume_loop` - var assumed `[0,0]`, `set_var`=+1, load at updated index | reject | **reject** `GV-ASSUME-VIOLATED` (stmt 8), 1 error, `--strict` rc=1 | **fixed** |
| `gv_attack_assume_trusted` - same loop but `assume_trusted(..., "loop exits before i=1")` | pass but listed | **pass strict** (0 errors); `-v` prints 2 `GV-ASSUME` **Info** lines, both marked `TRUSTED` with the reason; summary counts them as 2 findings | **documented trust boundary; listed and countable** |
| `gv_attack_false_contract` - declares 1 MiB for a 64 B host pointer, loads +4096 | flag/unproven | **pass** (0 errors; 2 global accesses proven) | **trust boundary (host contract)** |
| `gv_attack_lds_const` - declared 256 B, read at 512 | reject | **reject** `GV-LDS-OOB`, `[512,512]` vs `[256,256]` | correct |
| `gv_attack_desc_over` - 64 B extent, `num_records=4096` | reject | **reject** `GV-BUF-OVER` | correct |
| `gv_attack_clamp_path` - clamp only one branch | reject | **reject** `GV-GLB-OOB`, `[0,1008]` vs `[512,512]` | correct |

**`assume_trusted` is a real escape:** a kernel can use it to certify an access the loop actually
violates (the attack is an OOB on iteration 2). It is not silent: every use is `Info`/`TRUSTED` with
the reason, and the finding count includes it. That is acceptable only if the trust list is reviewed;
`--strict` alone cannot distinguish it from a proof. Same class as false contracts.

## 2. Tile-lifetime attacks (v1.1 model)

| attack | expected | actual | verdict |
|---|---|---|---|
| `gpu_tile_attacks::non_counter_version` - consume version from a kernarg | reject | **reject** `GV-TILE-INDEX` | correct |
| `gpu_tile_attacks::dma_without_wait_vm` - DMA producer, barrier, no `wait_vm(0)` | reject | **reject** `GV-TILE-STALE` in all 1..4 unroll scenarios ("published at phase never") | correct |
| same with `wait_vm(0)` | accept | **clean**, `tile-reads: 4` | correct |
| ring slots=1 with next-iteration overwrite (`gpu_tiles` negative test) | reject | **reject** `GV-TILE-RACE` | correct |
| read before the publishing barrier, role-divergent barrier, alias without `lds_alias`, escape from marker (`gpu_tiles` negatives) | reject | **all rejected** (`GV-TILE-STALE`/`GV-TILE-RACE`/`GV-TILE-OVERLAP`/`GV-TILE-ESCAPE`/`GV-BAR`) | correct |

**Open soundness gap - loop unrolling is 1..=4 only.** `gpu_tile_tripcount::late_race` is a
double-buffered two-role ring that is correct for iterations 0..4 and races only when `it >= 5`
(producer overwrites the consumer's current slot version). On Main 4c4909a the verifier returns
**no tile errors** (`tile-reads: 10`, `tile-cover: 3`) and would certify it, yet a host launcher with
`n >= 6` executes the race. This is the documented trip-count<=4 assumption, but it is a real
conditional soundness hole: a >4-iteration loop must either be proven bounded by 4 (or the model must
unroll/abstract all iterations), otherwise the certificate is not valid. This should be reported in
the verifier doc's honest-gaps section and gated in `--strict` (e.g. `GV-TILE-UNROLL-LIMIT` when the
trip count cannot be proven <=4).

Other model limits (documented, not soundness bugs): same-tile partial-byte overlap is not
distinguished; untagged LDS accesses are `Warn` only; wave-uniform runtime `if` containing
barriers/markers is `GV-TILE-UNSUPPORTED` (Warn); cross-workgroup races and more than two roles are
out of scope; all loops are assumed to run the same iteration count in every role.

## 3. Host-truth check (`tests/contract_host_truth.rs`)

Implemented on `truth-main` (Main `4c4909a` + annotations `5f923d8`); **2/2 pass**. For three shapes
(B1H8S2048, B4H8S2048, B1H2S200) it evaluates every `Function.contract`:
`arg_tensor` alloc/logical extents, `grid_is`, `ka_is`, `ka_range`, `sym_range`, and compares them with
`tests/gpu/ref_host.cpp`/`attn_host.cpp` formulas. It covers Q/K/V/O, Q/K scale logical+zeroed slack,
V scale, split `grid_z = B*splits`, Ohat/lse `splits*rows_total*128*4` / `*4`, and combine
`out=rows_total*128*2`. **No declared extent exceeds the host allocation; all grid/ka relations match.**
Remaining contract gaps: `kvs_counters`@0x288 still has no `arg_tensor`; the split contract does not
state `splits in {2,4,8}` (auto uses2/4; `REF_SPLIT` could force other values); assumptions added to
shipped kernels are currently **not inductive** (next item).

## 4. False positives after Main + annotations

Safe shipped kernels (certified) on `truth-main`:

| kernel | errors | codes | proven |
|---|---:|---|---|
| `_vs128` | 34 | 30 `GV-GLB-OOB` + 4 `GV-ASSUME-VIOLATED` | LDS 600, global 76, barriers 18 |
| `_t128` | 66 | 48 `GV-GLB-OOB` + 16 `GV-LDS-OOB` + 2 `GV-ASSUME-VIOLATED` | LDS 320, global 44 |
| `_kvsplit` | 64 | 60 `GV-GLB-OOB` + 4 `GV-ASSUME-VIOLATED` | LDS 760, global 18 |
| `_combine8h` | 1 | `GV-ASSUME-VIOLATED` | global 3 |
| `_t` | 3 | 1 `GV-LDS-OOB` + 2 `GV-ASSUME-VIOLATED` | LDS 225, global 88 |

These are **all false positives on certified kernels**:
* `GV-ASSUME-VIOLATED` on kernel-opt's new assumes: e.g. `set_var var0 = [1, 255/128 +
  1/128*ka(0x110)]` outside assumed `[0, 127/128 + 1/128*ka(0x110)]`; exit-test-bounded counters need
  `assume_trusted` (with a reason) or a tighter inductive range.
* `GV-GLB-OOB` global/DMA lower bounds are `[-inf,+inf]` for K/V/Q-scale because the split/loop
  variable is still not related to the contract; LDS/DS side is proven for vs128/kvsplit, but t128/t
  have 16/1 DMA-LDS-side false positives with M0 expressions involving `Select`/loop vars.
* `_t128` grid rows_per_block is 128, and its contract is correct; the remaining errors are analysis
  precision, not contract errors (host-truth test passes).

Earlier `4e6d58f` run had 93 kernels / 2 clean / 2926 errors (2906 `NOEXT`); after annotations the
global count drops sharply (e.g. vs128 106 -> 34). The 141-run sweep is still not a certificate: it
contains all probes and option combinations without contracts; do not gate on `--strict` over it.

## 5. Hardware confirmations (unchanged, GPU 6)

* MUBUF OOB (`tools/buffer_probe.sh both`): **BUFFER_OK** for rw / lds / runtime-sized dyn (zero-fill
  / dropped stores hold).
* LDS OOB (`probe_lds_oob`, 3 runs): valid sentinel OK; read at +256 = `0x00000000`, read at +1024 =
  **`0x3f800000` stale non-zero (not zero-filled)**; write at +4096 not observable; no fault. LDS has
  no hardware bounds check; the verifier must reject OOB by construction.
* MFMA register alignment: Main 4c4909a now models the assembler-measured **even VGPR tuple** rule
  (v1.1 fix); my earlier finding v52/v4 accepted / v1 rejected matches the new model. S4 alignment
  covered by tests.

## 6. Final soundness statement

**Provably catches (Main 4c4909a):** LDS DS/DMA segment overflow/wrap; unbounded ring/loop vars;
global extent/no-extent/provenance and one-path clamps; descriptor `num_records` overstatement;
role-barrier balance; VGPR/SGPR limits, tuple/descriptor alignment and MFMA overlap; path-sensitive
and loop-carried wait-before-use; tile overlap/escape/kind/stale/race/DMA-completion for simulated
1..=4-iteration, two-role protocols; non-counter tile versions; `GV-ASSUME-VIOLATED` for
non-inductive `assume`.

**Trusted / not proven:** host contract extents (`arg_tensor`/`ka_*`/`grid_is`) are assumptions; the
new `assume_trusted(reason)` escape (listed, countable, but can hide a real OOB); loop trip counts
are assumed <=4 and equal across roles (my `late_race` attack certifies a >4-iteration race - the
main open gap); raw safe-asm interiors are not checked (`GV-ASM` Info); untagged LDS lifetime is
Warn-only; partial-byte tile overlap is conflated.

**Out of scope:** cross-workgroup races/memory model, i32 arithmetic wrap-around (the analysis assumes
no wrap; a wrapping index can invalidate a proof), hardware faults for OOB LDS/MUBUF (MUBUF is
hardware-guarded, LDS is not), and multi-role protocols beyond two wave roles. Therefore v1.1 is a
strong *finding engine* and a useful certificate for bounded (<=4 iteration) two-role LDS protocols,
but not yet an unconditional end-to-end "no OOB / no race" guarantee until the unroll gap and the
trusted-assume list are gated.
