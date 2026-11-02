# Independent adversarial audit: gpu_verify v1

Auditor: `kernel-occ`. Branch: `gpu-semantics-occ`. Tested verifier commit: **`4e6d58f`**
(`origin/gpu-semantics`; analyzer `818838c`). Hardware: MI355X / gfx950, node 53, GPU 6.
The verifier's own `cargo test -p amdgpu-compiler --test gpu_semantics` passes **22/22** on this commit.

## 1. Soundness attacks (DSL test kernels in `gv_attack_*`)

Each attack is a real DSL kernel, checked with `amdc gpu-verify <kernel> --strict`; the attack is
`PASS` (unsound) if the checker exits 0, `REJECT` if it finds an Error.

| attack | class | expected | actual | verdict |
|---|---|---|---|---|
| `gv_attack_lds_const` (declared 256 B, `ds_read` at 512) | LDS bounds | reject | **reject** `GV-LDS-OOB`, address `[512,512]` vs limit `[256,256]` | correct |
| `gv_attack_desc_over` (64 B extent, `buffer_desc(q,4096)`) | descriptor num_records | reject | **reject** `GV-BUF-OVER` (4096 vs allocation `[64,64]`) | correct |
| `gv_attack_clamp_path` (min-clamp on one branch, raw index on the other) | clamp provenance | reject | **reject** `GV-GLB-OOB`, address `[0,1008]` vs `[512,512]` | correct |
| `gv_attack_false_contract` (declares 1 MiB for a host 64 B pointer, then loads +4096) | host contract | flag/unproven | **PASS** (`0 errors`; `global: 2` proven) | **trust boundary - not a proof of host truth** |
| `gv_attack_assume_loop` (loop var assumed `[0,0]`, `set_var`=+1, global load at the updated index) | assume induction | reject | **PASS** (`0 errors`; only 2 `GV-ASSUME` **Info** lines) | **UNSOUND** |

**Critical finding:** `assume` induction violations are recorded as `Severity::Info`, not `Error`. A
kernel that assumes a loop variable stays in a range but updates it outside that range still passes
`--strict`; the access is then justified by the assumption. The attack is a real OOB on the second
iteration. Recommended fix: an unproved `set_var` against an assumed range must be an **Error**
(or the kernel must be marked non-certified), not Info.

**False contract is the second trust boundary:** `arg_tensor`/`arg_tensor_slack` are HOST declarations.
If they overstate the allocation, every affected access is "proven" and `--strict` passes. This is not a
checker implementation bug, but it means the certificate is only as good as the host-truth check. A
false-contract attack is included as `gv_attack_false_contract`.

## 2. False-positive audit

`amdc gpu-verify --all` on the tested commit: **93 kernels, 2 clean, 2926 errors** -
`GV-GLB-NOEXT` 2906, `GV-LDS-OOB` 18, `GV-GLB-OOB` 1, `GV-BUF-OVER` 1. The 2 clean kernels are the two
false-contract/assume attacks; every unannotated production/probe kernel gets `GV-GLB-NOEXT` by design.

Shipped kernels (all safe/certified; all rejections are global-contract false positives on `4e6d58f`,
which has no shape equalities yet):

| kernel | findings | proven | notes |
|---|---:|---|---|
| `_vs128` | **106 errors** | LDS 600, barriers 18, waits clean | all `GV-GLB-OOB`; Q/K/V/Q-scale/K-scale/out |
| `_t128` | 108 | LDS 320, barriers 8 | same |
| `_kvsplit` | 78 | LDS 760, barriers 18 | same |
| `_kvsplit_combine8h` | 3 | waits clean | loop-index bounds on lse/ohat/out |
| `_t` | 89 | LDS 225, barriers 11 | same |

Representative exact residual (Kernel-opt's own diagnostic, matches my run):
`Q` load upper bound is
`ka(0x60)*ka(0x70) - ka(0x70) - ka(0x90) + ka(0x90)*grid_y - ka(0xA0) + ka(0xA0)*grid_z`
vs limit `ka(0xA0)*grid_z`; the missing relation is the host ABI equality
`0xA0 = 0x60*0x70`, `0x70 = grid_y*0x90`, i.e. `ka_is(0xA0, ka(0x60)*ka(0x70))`,
`grid_is(1, ka(0x140))`. `K` DMA lower bounds are `[-inf,+inf]` because the split/ring loop variable
is not yet related to the contract; the LDS accesses are fully proven (600/760), so this is purely
a missing contract, not a checker failure. After `ka_is` equalities land these should be re-run.

**141-run sweep:** not fully certified in this audit because the sweep includes every probe and option
combination and the verifier branch has no contracts for them; the `--all` run above is the same
universe (93 kernels) and shows the expected `NOEXT` pattern. Do not treat `--strict` on the sweep as
a correctness signal until all kernels carry contracts.

## 3. Hardware confirmations (GPU 6)

* **MUBUF OOB read/write** (`tools/buffer_probe.sh both`, own backend): **BUFFER_OK** for `rw`, `lds`,
  and the runtime-sized `dyn` descriptor. OOB loads return zero, OOB stores are dropped, LDS-DMA forms
  are byte-identical to `global_load_lds`; the model's "hardware-guarded descriptor" assumption holds.
* **LDS OOB has no bounds check and is not zero-filled.** Probe `probe_lds_oob` declares a 256-byte
  segment, writes/reads a sentinel at lane offsets, then reads at +256, +1024 and reads back a write at
  +4096. Results (3 launches identical): valid sentinel read OK; **+256 = 0x00000000**;
  **+1024 = 0x3f800000 (stale non-zero data, not zero-fill)**; write at +4096 is not observable on
  readback (dropped or aliased), no fault. Conclusion: hardware does **not** zero-fill OOB LDS; the
  verifier must reject all out-of-segment accesses by construction (as it does).
* **MFMA destination alignment:** the assembler accepts `v[52:67]` and `v[4:19]` (both multiples of 4)
  and rejects `v[1:16]`; vendor hot-loop MFMA destinations are e.g. `v52:67`, `v100:115`. The verifier's
  generic rule `p % dwords == 0` requires a 16-dword tuple to start at a multiple of **16**, which is
  **stricter than the observed ISA/toolchain requirement (multiple of 4)**. This is a false-positive
  rule, not a soundness hole; it should be corrected to the ISA alignment (or justified if a stricter
  allocator invariant is intended). S4 descriptor alignment is covered by the unit tests; I did not
  find a hardware counterexample.

## 4. Host-contract truth and coverage gaps

The annotation formulas themselves match `tests/gpu/ref_host.cpp` for Q/K/V/O, Q/K scales (logical +
zeroed slack), V scale, split Ohat/lse, and combine Ohat/lse/out:
`Q=B*S*H*D`, `K=V=B*SK*HK*D`, `O=B*S*H*D*2`, `Qd=B*S*H*4 + 512*H*4`, `Kd=B*SK*HK*4 + 640*H*4`,
`Vd=4096`, `Ohat=splits*B*S*H*128*4`, `lse=splits*B*S*H*4`, combine `out=B*S*H*128*2`.
No **overstated** declared extent was found in the existing annotation set. Gaps:

1. **`kvs_counters` (0x288) has no `arg_tensor`** in the fused path; contract coverage gap.
2. **Split contract does not constrain `splits` to {2,4,8}** (`ka_range(0x280,1,MAX)` only); the
   `bi=z>>lg`, `sp=z&(splits-1)` assumes are valid only for power-of-two splits. Host auto uses2/4,
   but `REF_SPLIT` can force other values; the contract should state the domain or the assume is not
   host-guaranteed.
3. **`rows_total` (0x284) is not tied to B*S*H**; the Ohat/lse extents use the supplied value. A wrong
   host value is a false proof; contract should state `rows_total == B*S*H` or the verifier should
   derive it from shape scalars.
4. **Raw safe-asm is explicitly unchecked**: `verify_ir` records `GV-ASM` Info "internal memory
   accesses are NOT bounds-checked by v1"; any `Op::Asm`/`Stmt::VarAsm` can hide an OOB. This is a
   documented scope limit, not a regression, but it means "no OOB" is not end-to-end for asm blocks.
5. **Contract declarations are trusted** (false-contract attack): the checker cannot know
   `malloc` sizes. A host-truth check is required (below).

## 5. Recommended host-truth check (executable)

Cheap version to add as `crates/amdgpu-compiler/tests/contract_host_truth.rs` (not committed here because
the annotated kernels are on `gpu-semantics-opt`, not on the tested verifier commit):
evaluate each `Function.contract.extents[*].alloc/logical` with the test's shape values and assert
equality to the host formulas above for B1H8S2048, B4H8S2048, B1H2S200; also assert
`grid_defs`/`sym_ranges` imply the split assumptions. This turns the documented host contract into a
checked one and catches overstatement. A `ref_host` launch-time assertion `alloc_bytes >= logical`
would cover the C++ side; the Rust test covers the DSL side.

## 6. Overall soundness statement

* **Provably catches now:** constant/derived LDS segment overflow and negative/wraparound addressing;
  unbounded ring/loop variables; global extent/no-extent/provenance and clamp-path bounds; descriptor
  `num_records` overstatement; role/barrier balance; VGPR/SGPR limits, tuple/descriptor alignment and
  MFMA destination overlap; path-sensitive and loop-carried wait-before-use; 22 unit tests pass.
* **Does not yet guarantee:** host allocation truth (declarations are trusted); `assume` induction
  violations (Info, not Error - **critical gap**); safe-asm interior accesses (Info/skipped); fused
  counter buffer and split-domain/shape relations not annotated; MFMA alignment rule stricter than the
  observed hardware requirement.
* **Verdict for v1:** useful as a *finding engine* over IR/MIR; it is **not yet a sound end-to-end
  "no OOB" certificate** because of the assume-severity gap, trusted contracts and asm scope. The first
  two must be fixed before strict mode gates shipping; the contract/`ka_is` work is already in flight.
