# gpu_verify v16 audit of record - shared-tile visibility fix

Auditor: `kernel-occ`. Branch: **`gpu-semantics-occ`** (this doc). Tested Main:
**`origin/Main` = `a645265`** (v16 fixed shared-tile visibility = barrier publication for every role;
`a742a09` + docs 13.1). Independent audit tests are in this branch:
`gpu_tile_self_visibility.rs` (7/7), `shipped_ring_layout.rs` (1/1), plus the earlier
`gpu_tile_attacks` / `gpu_tile_tripcount` / `gpu_tile_horizon` suites. On the tested tree:
**28/28 `gpu_semantics` + 33/33 `gpu_tiles` + 7/7 self-visibility + 2/2 host-truth pass.**

## 1. The corrected rule (user-visible)

**Shared LDS tiles are produced by OTHER waves (including waves of the same role) and consumed
independently by every wave. Visibility is barrier/`publish`-based for every role; a wave's own
`wait_lds(0)` only completes its own writes and does NOT make the tile visible to itself.
"Self-produced, self-consumed" is valid only for private per-wave regions (`lds_private`).**
The v16 `gpu_tiles::check_pair` no longer has the `if p.role == c.role { p.done < c.start }` shortcut;
for shared tiles `visible = p.publish <= c.ord`, and read/write overlap is checked across same-role
waves too.

## 2. Independent attack table (`gpu_tile_self_visibility`, all pass)

| attack | expected | actual on v16 | verdict |
|---|---|---|---|
| (a) same-role: every wave writes its slice, `wait_lds(0)`, reads the whole tile, **no barrier** | reject | **`GV-TILE-STALE`** "produced by the waves of THIS role too, but a shared tile is only visible after a barrier publishes it (own wait_vm/wait_lds(0) is not enough ...), published at phase never" | SOUND (old rule accepted) |
| (b) barrier only on the producer role's path | reject | **`GV-BAR`** (then 1 / else 0 barriers) + **`GV-TILE-STALE`** | SOUND |
| (c1) consumer before the publishing barrier | reject | **`GV-TILE-STALE`** (published phase 1, consumer phase 0; evidence: version 0 not visible) | SOUND |
| (c2) consumer at the publish phase (immediately after barrier) | accept | **clean**, `tile-reads: 20` | SOUND boundary |
| (d) `ds_write` then barrier, **no `wait_lds(0)`** | reject | **`GV-TILE-STALE`** "own wait ... is not enough", published at phase never | SOUND |
| (e1) same-version cooperative fill by both roles (P low half, C high half), wait, barrier, consume version 0 | accept | **clean**, `barrier-balanced-ifs: 2`, `tile-reads: 20` | SOUND |
| (e2) different versions of the same slot written by the two roles in overlapping phases | reject | **`GV-TILE-RACE`** (write-write versions 0/1, phases 0..1 / 0..1) + **`GV-TILE-STALE`** | SOUND |

Additional: (f) non-counter marker version -> `GV-TILE-INDEX`; DMA producer without `wait_vm(0)` ->
`GV-TILE-STALE`; slots=1 overwrite/read-before-publish/alias/escape -> `GV-TILE-RACE`/`STALE`/
`OVERLAP`/`ESCAPE`; period-256 attack -> `GV-TILE-UNROLL-LIMIT`. `tile_trust_periodic` and
`assume_trusted` remain listed/counted trust boundaries.

## 3. Shipped certificate (`--shipped --strict`, 0 errors AND 0 warnings)

`amdc gpu-verify --shipped --strict -v`: **rc=0, 0 Error lines, 0 Warn lines on all 8 shipped
kernels**, aggregate **44 trusted assumes / 216 trusted contracts / 0 unproven**. **No shipped kernel
fails the new visibility rule**, so there is no candidate missing-barrier kernel bug.

| kernel | errors | tile-reads | tile-cover | certificate |
|---|---:|---:|---:|---|
| `_vs128` | 0 | 58880 | 600 | 6/29/0 |
| `_vs5` | 0 | 58880 | 600 | 6/29/0 |
| `_kvsplit` | 0 | 240640 | 760 | 12/36/0 |
| `_t128` | 0 | - | - | 4/29/0 |
| `_t` | 0 | - | - | 4/29/0 |
| `_vs9` | 0 | - | - | 6/29/0 |
| `_vs` | 0 | - | - | 4/29/0 |
| `_combine8h` | 0 | - | - | 2/6/0 |

Counter note: v16 reports more `tile-reads` than the v15 certificate (vs128/vs5 58880 vs 16640;
kvsplit 240640 vs 66560; kvsplit tile-cover 760 vs 600). This is the stricter visibility/path
enumeration, not a regression: every event is proven, there are 0 errors and 0 warnings, and K=0
unproven. The v15 numbers were produced under the old same-role shortcut and are not comparable.

## 4. Block-level shipped K/V ring check (independent of the verifier)

`shipped_ring_layout.rs` splits the shipped kernel's top-level role branch and, **inside each role
branch independently**, checks for a barrier between the K/V/KD tile producer and first matching
consumer. On `a645265`:

| kernel | branch | K/V/KD producers | consumers | publishing barrier stmt |
|---|---|---:|---:|---:|
| `_vs128` | roleA | 6 | 80 | **506** |
| `_vs128` | roleB | 6 | 80 | **6820** |
| `_kvsplit` | roleA | 6 | 80 | **559** |
| `_kvsplit` | roleB | 6 | 80 | **7516** |

Both roles have their own publishing barrier in the marker/barrier layout; no role-local missing
barrier was hidden.

## 5. Host truth

`contract_host_truth.rs` on v16: **2/2 pass** for B1H8S2048/B4H8S2048/B1H2S200; every extent,
`grid_is`, `ka_is`, `ka_range`, `sym_range` matches ref_host/attn_host; no overstatement. Remaining
contract gaps: `kvs_counters`@0x288 has no `arg_tensor`; split domain not constrained to {2,4,8}.

## 6. Final soundness statement (README-ready)

**"The compiler is GPU-semantics-aware: shared LDS tile visibility is barrier-publication based for
every wave, never self-produced."** v16 certifies all **8 shipped FMHA kernels** with
`--shipped --strict` at **0 errors and 0 warnings**, 0 unproven facts, and a countable trust budget of
**44 trusted assumes / 216 host contracts**. It rejects a wave that writes/waits/reads a shared tile
without a barrier (`GV-TILE-STALE`), a barrier on only one role's path (`GV-BAR`), reads strictly
before the publishing phase, producers crossing a barrier without completing their writes, and
different-version overlap (`GV-TILE-RACE`); it accepts consumer-at-publish-phase, same-version
cooperative fills, and private per-wave regions. The same certificate covers LDS/DMA bounds, global
buffers, VGPR/SGPR legality, barrier/role balance, path-sensitive/loop-carried waits and the tile
protocol. Trust is explicit: `assume_trusted`, `tile_trust_periodic`, and host
`arg_tensor`/`ka_is`/`grid_is` (44/216, printed per kernel as `N trusted assumes, M trusted contracts,
K unproven`). Documented limits remain: raw safe-asm interiors, untagged LDS, partial-byte tile
overlap, runtime-if outcomes, `period_bits` under-approximation for version-only `^`/`<<`/`*`,
`kvs_counters` contract gap, i32 wrap, cross-workgroup races and >2 wave roles. Within those limits,
this is the audit of record for "no OOB / wrong lifetime for global/LDS/VGPR/SGPR" on the shipped set.
