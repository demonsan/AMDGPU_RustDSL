# KV-split design study (sequence-parallel attention with an exact cross-block merge)

> Status: ADOPTED - shipped as one of the default dispatch paths (`_kvsplit` + `_kvsplit_combine8h` selected by `_auto`); see docs/kvsplit-results.md, docs/current-status.md and rounds 60-62, 69 of docs/status.md.

Status: **design study, no kernel implementation** (2026-10-30). Baseline: branch `final-auto`, validated build
`6e9e988` (auto dispatch certified in `docs/vendor-spec.md` §7.9.5c), GPU 5, MI355X (gfx950). Numbers marked
*measured* were re-run for this study with the warm protocol (`REF_WARMUP` 80-150, `REF_ITERS` 60-100, best/median
of 5). Everything else is a model and is labelled as such.

## 0. Summary and recommendation

**The lever is real but small: expected 1.3-1.8x over the current `t128`, not 3-4x.** The wave arithmetic is
correct (splitting the key dimension 4 ways gives 2048 waves, 256 blocks of 8 waves = 256 CUs at 2 waves/SIMD),
but the per-CU rate at 2 waves/SIMD is the existing `_vs128` rate (5.9 TFLOP/s/CU, *measured*), so the
compute-only ceiling is 256 x 5.9 = **1.51 PFLOPS, i.e. 11.4 us for B1H8S2048** (17.18 GFLOP), and the split
partials add 33.6 MB of writes (7.0 us standalone, *measured*) plus a **5.1 us combine kernel** (*measured*).
Net modelled total 18-25 us = **690-950 TFLOP/s** (central ~800), vs today's `t128` 509.7 TFLOP/s (33.7 us,
*measured*) and vendor 537.3 (32.0 us, *measured*): **1.35-1.87x over `t128`, 1.28-1.78x over the vendor**, on the
benchmark shape only. Filling shapes keep the current dispatch and do not regress.

**Recommendation: GO, as a two-step prototype, gated on measured split=4 total wall <= 21 us.**
Step A first: split=2 on the existing `_vs128` block shape (minimal kernel change; expected 540-610 TFLOP/s,
1.06-1.20x over `t128`). If that works, Step B: split=4 (the 690-950 TFLOP/s case). Do **not** start with split=8
(partial traffic doubles, measured combine cost doubles, no extra CU fill). Prototype size: ~200-400 DSL LOC in
the FMHA kernel (key range + partial epilogue), ~80-150 LOC combine kernel, ~150-250 LOC harness + dispatch;
2-3 days kernel + 1-1.5 days independent validation (see §8).

## 1. The wave arithmetic (why 2/4/8 splits, and which block shape)

Measured baseline on B1H8S2048 (B=1, H=8, S=SK=2048, D=128):

| kernel | rows/block | threads | waves/block | waves/SIMD | LDS B | blocks | CUs used | waves total | TFLOP/s | TFLOP/s/CU |
|---|---|---|---|---|---|---|---|---|---|---|
| vendor | 256 | 512 | 8 | 2 | 84480 | 64 | 64 (25%) | 512 | **537.3** | 8.40 |
| `_vs128` | 256 | 512 | 8 | 2 | 102400 | 64 | 64 (25%) | 512 | 383.4 | 5.99 |
| `_t128` | 128 | 256 | 4 | 1 | 135168 | 128 | 128 (50%) | 512 | **509.7** | 3.98 |

`rows_total = B*S*H = 16384`; one wave covers 32 rows (2 lanes per row); 1024 SIMDs (256 CUs x 4). Both kernels
have exactly 512 waves, so they can only fill 64 CUs at 2 waves/SIMD or 128 CUs at 1 wave/SIMD - the grid
underfill established in §7.9. To fill 256 CUs at 2 waves/SIMD we need 2048 waves, i.e. **4x the workgroups**.

**Carrier choice: the `_vs128` block shape (256 rows, 512 threads, 8 waves/block), not the `_t128` shape.**
An 8-wave block occupies 2 waves/SIMD on one CU; with LDS 102400 B (2 x > 163840) it is one block/CU, so
`blocks` workgroups exactly fill `min(256, blocks)` CUs at the desired 2 waves/SIMD with **no LDS or VGPR
change**: VGPR 249 x 2 waves/SIMD = 498 <= 512. The `_t128` shape has 4 waves = 1 wave/SIMD and would need two
co-resident blocks (LDS <= 81920 B; the shipped 135168 B does not co-reside, and a 2-slot variant would still be
99840 B unless the dead ones/Q staging is dropped to 69632 B) plus phase-interference experiments. Not worth it
for the first prototype.

Splitting the key dimension by `s` multiplies the workgroups by `s` but leaves total FLOPs and total K/V traffic
unchanged:

| s | workgroups | waves | CU fill (1 block/CU) | waves/SIMD | compute-only time (17.18 GFLOP / 5.9 TFLOP/s/CU) |
|---|---|---|---|---|---|
| 1 (today's `_vs128`) | 64 | 512 | 64 (25%) | 2 | 45.5 us (measured 44.8) |
| 2 | 128 | 1024 | 128 (50%) | 2 | 22.8 us |
| **4** | **256** | **2048** | **256 (100%)** | **2** | **11.4 us** |
| 8 | 512 | 4096 | 256 (100%, 2 sequential block waves) | 2 | 11.4 us (no extra CU gain) |

So 2 and 4 are the useful points; 8 only adds traffic (see §5) and is kept as a diagnostic knob.

## 2. The scheme

**Row tiling (unchanged).** Row blocks stay `ceil(S/256)` per (b,h) of 256 query rows; each workgroup owns one row
block and one key split. Rows are never split, so every output row is produced by exactly one combine. Grid:
`(ceil(S/256), H, B*s)`; `split = blockIdx.z % s`, `b = blockIdx.z / s`; the workgroup loops only keys
`[split*SK/s, (split+1)*SK/s)` (tail handling: the existing 128-key tile loop already masks the last partial tile;
splits with <128 keys are allowed, but a split must contain at least one tile - so require `SK >= 128` and
`s <= floor(SK/128)` if we want no empty split; the prototype can also let empty splits write lse=-inf).

**Online softmax partials (per split s, per row r).** Using the kernel's existing base-2 domain
(`x = QK * softmax_scale * log2(e)`, `P = 2^(x-m)`):
```
m_s(r) = max over keys in split s of x
l_s(r) = sum over keys in split s of 2^(x - m_s)          # the kernel's row sum
O_s(r,:) = sum over keys in split s of 2^(x - m_s) * v   # fp32 accumulation, 128 floats
```
Epilogue stores the **normalized** partial and the **log2-domain LSE**:
```
Ohat_s(r,:) = O_s(r,:) / l_s(r)          # fp32, divided once per row element
lse_s(r)    = m_s(r) + log2(l_s(r))      # fp32 (one v_log2_f32 per row)
```
Two implementation details agreed with kernel-opt (they preserve the combine algebra):
* **Key ranges are 128-key-tile aligned**, not byte-balanced: `ntiles = ceil(SK/128)`,
  `tps = ceil(ntiles/splits)`, split `i` owns tiles `[i*tps, min(ntiles,(i+1)*tps))`. Every split must own at
  least one tile, i.e. `(splits-1)*tps < ntiles`; the harness dispatch guard enforces this (Step A writes no
  `-inf` empty split). For the benchmark (ntiles=16): s=2 -> 8+8 tiles, s=4 -> 4+4+4+4.
* **`lse` is stored in the kernel's own base-2 domain** where `P = 2^(x + 512 b - 127)` with the carried
  reference bias `b`, so `lse_s = log2(l_s) - 512 b_final_s` up to a split-independent additive constant. The
  absolute values are **not** the textbook `m + log2(l)`; only differences between splits of the same row are
  meaningful, and `w_s = 2^(lse_s - max lse)` is unchanged. The harness treats lse as opaque (it only compares
  per-row weights), and Ohat already includes the V descale exactly as today's epilogue does.
Storage: `Ohat`: `[s][rows_total][128] fp32` (or fp16 option, §4); `lse`: `[s][rows_total] fp32`;
row index = `((b*S + row)*H + h)`, i.e. the same linear order as the output, so the combine writes coalesced
bf16 output with no transpose. For B1H8S2048 `rows_total = 16384`: Ohat = 8.39 MB per split, lse = 64 KB per split
(s=2: 16.8 MB, s=4: 33.6 MB, s=8: 67.1 MB).

**Combine kernel.** One thread per `float4` of a row: load `lse_s(r)` for all splits, `m = max_s lse_s`,
`w_s = 2^(lse_s - m)`, `O(r,:) = (sum_s w_s * Ohat_s(r,:)) / (sum_s w_s)`, write bf16. Traffic
`s*rows_total*128*4` read + `s*rows_total*4` read + `rows_total*128*2` write (s=4: 38.0 MB). Grid: 8192 blocks
x 256 threads reaches 7.4 TB/s measured (§5). No atomics, no cross-block sync, deterministic.

**Why the merge is exact (real arithmetic).**
```
sum_s w_s * Ohat_s = sum_s 2^(lse_s - m) * (sum_{i in s} 2^(x_i - m_s) v_i) / l_s
                   = 2^(-m) * sum_s 2^(lse_s) * (1/l_s) * sum_{i in s} 2^(x_i - m_s) v_i
lse_s = m_s + log2(l_s)  =>  2^(lse_s)/l_s = 2^(m_s)
=> sum_s w_s * Ohat_s = 2^(-m) * sum_s sum_{i in s} 2^(x_i) v_i = 2^(-m) * sum_all 2^(x_i) v_i
sum_s w_s           = 2^(-m) * sum_all 2^(x_i)
=> O = sum_all 2^(x_i) v_i / sum_all 2^(x_i)      (independent of m_s and of the split boundaries)
```
The current single-block kernel computes the same quantity with a running max and per-tile rescale; in real
arithmetic the two are identical. `lse` and `Ohat` are exactly what the split blocks already compute for their
local max, so the merge needs no extra state (`m_s` cancels, which is why storing `lse_s` is enough).

**Storage precision / bit-identity.** The merged result is **not bit-identical** to `_vs128`/`_t128`: the split
blocks quantize `P` against different max references (split-local vs running per-tile), and the merge adds one
weighted-sum and one division rounding per element. A numpy emulation of the same algebra (384 rows, SK=2048,
D=128; fp32 accumulators, P quantized through an fp16 proxy; exact fp64 reference) gives:

| splits | mean |diff| vs single | max |diff| | bf16 words differing | max bf16 ULP | p99 bf16 ULP |
|---|---|---|---|---|
| 2 | 3.24e-6 | 2.51e-5 | 0.19% | 1 | 0 |
| 4 | 4.33e-6 | 2.86e-5 | 0.25% | 1 | 0 |
| 8 | 4.78e-6 | 2.66e-5 | 0.24% | 1 | 0 |

For scale: the single-block simulation is 3.98e-6 from the exact fp64 result, so **splitting adds about the same
rounding magnitude as the existing kernel's own per-tile rescale**, i.e. mean 4e-6 on an output whose max |ref| is
2.0 (2e-6 relative), ~0.6% of the current kernel-vs-CPU mean error (7.3e-4) and 1/500 of a bf16 ULP. A pessimistic
model that quantizes P to fp8 without the kernel's correction recipe gives split-vs-single ~2.7e-3, the same order
as that model's own single-vs-exact error - the shipped two-word/LUT correction is exactly what keeps this small.
Conclusion: **the combined output cannot be bit-identical; it should be within 1-2 bf16 ULP on <=0.3% of elements,
far inside the REF_KERNEL tolerance (0.05 + 5%) and the parity margin.** The prototype must measure the real
histogram (test in §4); if fp32 partials are kept, expect 4e-6-level diffs, not gate failures.

## 3. ABI and harness impact

**Buffers and ABI.** Both kernel families already live behind the 656-byte kernarg ABI in `ref_host`; the split
kernel is our own code and can extend its own args (the vendor args stay untouched, `ref_host` already patches
per-kernel fields). Add four fields: `Ohat` pointer at 0x270, `lse` pointer at 0x278, `u32 splits` at 0x280 and
`u32 rows_total = B*S*H` at 0x284 (the per-split stride; the kernel cannot derive B from the base args),
kernarg segment 720 bytes; the vendor/`_vs128`/`_t128` 656-byte ABI is untouched. Vendored ABI sizes stay at 656 for vendor; ours grows to 0x280/0x290. Buffers are allocated once,
outside the timed region: `Ohat` `s*rows_total*128*4` B, `lse` `s*rows_total*4` B (`rows_total = B*S*H`; gate
maximum is 16384 rows -> 33.6 MB at s=4, trivial against 192 GB HBM).

**Second launch.** The candidate is now two dependent launches on one stream: phase 1 = split kernel, phase 2 =
combine. The headline wall must be **one hipEvent pair around both launches** (plus the inter-launch gap). Report
`timing ours main:`, `timing ours combine:`, `timing ours total:` and compute TFLOP/s from the total; the vendor
stays one launch. Warm-up includes both launches; the gate's REF_ITERS loop times both.

**Reference comparison and the §7.9.0 protocol.** For the auto/dispatch path the current two-line protocol is
kept: (a) per-CU efficiency on the same grid (both at 2 waves/SIMD; the counter normalisation is now per wave:
`waves = blocks * 8`, still clock-free, usable for "split vs `_vs128` per-CU" comparisons); (b) absolute
throughput at each kernel's best legal tiling, with both launches of ours counted. Do not compare a main-kernel-only
number with the vendor's single launch.

**Harness changes (ref_host; attn_host only if the e2e launcher is used).**
- `REF_SPLIT=s` (2/4/8; 0 = single-kernel path), `REF_CO2_SPLIT`/`REF_KNAME2_SPLIT` and
`REF_CO2_COMBINE`/`REF_KNAME2_COMBINE` for the new objects; `REF_SPLIT_DUMP` for the diff test.
- Allocate `Ohat`/`lse`, extend our args, launch split kernel then combine; one event window; print the three
  timing lines + `[grid]` (chosen kernel, rows, threads, blocks, CU fill, split count, key range).
- Gate: the existing `REF_KERNEL_OK/REF_KERNEL_FAIL` and `PARITY_OK` read the final bf16 output, so they work
  unchanged once the combine writes the same buffer; run the 15 shapes at REF_ROWS=16 and 64.
- New check: `REF_DUMP2` from the split path vs a `_vs128` dump, then a tiny script (or a `REF_DIFF_REF` mode)
  reporting mean/max abs diff and bf16 ULP histogram; accept max <= 2 bf16 ULP, mean_abs <= 1e-4 (planned gate).
- `run_ref.sh`/`determinism.sh`: auto build `_kvsplit` + `_combine` when `OUR_KERNEL=..._kvs`, dispatch
  `REF_SPLIT` per shape, same env inheritance as the current `_auto` path; `attn_host` needs the same two-launch
  site only if it launches the FMHA kernel.
- Counters (`vendor_counters.sh`) become two dispatches; collect phase 1 for the main-kernel counters and phase 2
  separately so the split kernel's work can be compared per wave.

**Dispatch integration.** Keep the current `_auto` fallback for filling shapes (`N256 >= 256` -> `_vs128`;
`N256 <= 128` -> `_t128`), and add the KV-split candidate only when all of:
1. `N256 <= 128` and `N256*s <= 256` (one block wave; no oversubscription - measured `_t128` loses 0.41-0.80x
   exactly when its grid needs a second block wave, `docs/vendor-spec.md` §7.9.5c);
2. `s = 2..4` (s=8 is a diagnostic, not a default);
3. the model estimate `T_kvsplit(s) < 0.95 * T_current` (formula below), to avoid tiny shapes where the fixed
   ~5 us combine dominates.
Model used by the dispatcher (all quantities per shape):
```
T_current = F / (4.03e12 * min(256, 2*N256))                    # _t128, or 5.9e12 * min(256,N256) for _vs128
T_kvsplit(s) = max(F / (5.9e12 * min(256, N256*s)),             # compute-bound phase 1
                   partials_write_bytes(s) / 4.8e12)            # measured write BW, partially hidden
             + combine_bytes(s) / 7.4e12 + 1.5e-6              # measured combine BW + launch gap
F = 4*B*H*S*SK*D
```
For the 15 gate shapes `N256` is 1,4,32,64,4,27,8,48,16,2,16,16,8,4,2 - all <= 64, so the KV-split candidates
are the benchmark (N=64 -> s=4), B2H8S700 (N=48 -> s=4, 192 blocks), B2H4S1024 (N=32 -> s=4 or 8), and the small
ones only if the model says so (e.g. B1H16S256 SK=1024: t128 ~16.6 us, kvsplit-4 ~8 us -> wins; B1H1S256 SK=256:
~4 us both -> no). The dispatcher must therefore evaluate the formula per shape; this is a table in the prototype
and a hard gate test (all 15 shapes must choose a winner and the gate must pass).

## 4. Traffic and bandwidth: is the extra partial traffic a wall?

**K/V traffic is invariant under splitting; Q is re-read s times; partials add 2*s*rows*D*4 bytes.**
For B1H8S2048 (`rows_total = 16384`, `N256 = 64`):

| s | blocks | K/V read | Q read | Ohat write | lse write | phase-1 traffic | combine traffic | combine (measured) |
|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 33.55 MB | 2.10 MB | 0 (bf16 out 4.19) | 0 | 39.8 MB | - | - |
| 2 | 128 | 33.55 MB | 4.19 MB | 16.78 MB | 0.13 MB | 54.7 MB | 21.1 MB | **3.90 us** (5405 GB/s) |
| 4 | 256 | 33.55 MB | 8.39 MB | 33.55 MB | 0.26 MB | 75.8 MB | 38.0 MB | **5.10 us** (7389 GB/s) |
| 8 | 512 | 33.55 MB | 16.78 MB | 67.11 MB | 0.52 MB | 118.0 MB | 71.8 MB | ~9.7 us (extrapolated) |

*Derivation:* each split block reads `(SK/s)*D*2` bytes of K+V and there are `N256*s` blocks, so total K/V =
`N256*SK*D*2` = 33.55 MB, independent of s. Q = `rows_total*D` per block = `2.1 MB * s`.

**Measured achievable streaming bandwidth on GPU 5** (HIP probes written for this study, 8192 blocks x 256
threads, best of 200):
| probe | traffic | time | GB/s |
|---|---|---|---|
| combine s=4 (38.0 MB read+write) | 38.0 MB | **5.10 us** | **7389** |
| combine s=2 (21.1 MB) | 21.1 MB | 3.90 us | 5405 |
| pure partial write, 33.6 MB fp32 | 33.6 MB | 7.00 us | 4816 |
| *reference:* MI350 HBM peak (kernel-wiki `mi350-arch-topology` table 1.1, physical 13) | - | - | **8192 GB/s** |

kernel-wiki checks: `mi350-arch-topology` gives **8192 GB/s peak** (MI350X, 8 XCD); `mi350-arch-l2-paths` gives
**4 MB L2 per XCD** (8 XCD -> 32 MB aggregate), L2 client-side read 2048 B/cycle and write 1024 B/cycle per
16-channel group (table 6.1, physical 37) - the L2 fabric is not the wall for this pattern, HBM is.
`mi350-bottleneck-memory-bound` explicitly warns that its HBM/fabric PMCs are GC-side proxies and that a
bandwidth conclusion needs a timing A/B - hence the measured probes above rather than counter claims.

**Is 33.6 MB of partial writes a wall?** A partial write of 33.6 MB at 4.8 TB/s takes 7.0 us; the phase-1 compute
is 11.4 us, so the writes can be hidden if they overlap the MFMA stream (the kernel writes only in the epilogue,
once per row block, so the write burst lands at the end and only partly overlaps). With 256 blocks writing
33.6 MB over ~11-14 us the write bandwidth demand is 2.4-3.1 TB/s = 50-64% of the measured write-only bandwidth
(4.8 TB/s) and 29-38% of HBM peak -> co-limiting, not a hard wall. **L2 residency:** aggregate L2 is 32 MB, so s=4
partials (33.6 MB) just exceed it and most combine reads come from HBM (the measured 7.4 TB/s is consistent with
that); s=2 partials (16.8 MB) fit if distributed evenly (2.1 MB per XCD < 4 MB) and could be partly L2-resident,
which is why s=2's combine is already at 3.9 us. s=8 doubles the partial traffic for no CU gain - the main reason
it is a diagnostic.
**Verdict: bandwidth is not a blocker for s=2, and it is the main uncertainty for s=4 (0-7 us of the 18-25 us
model).** The prototype must counter this (TCC_EA0_WRREQ/read counters, and an fp16-partial variant, below).

**Optional traffic halving: fp16 partials.** Ohat is bounded by max|V| (1.875 in the gate data) and has fp16
mantissa 10 bits (0.05% relative); storing Ohat as fp16 halves phase-1 writes and combine reads:
partial write 16.8 MB (~3.5 us), combine 21.2 MB (measured **4.10 us** with a deliberately naive scalar probe;
a float4/float8-vectorized version should match the fp32 path's efficiency). Expected gain at s=4: phase-1
memory drops from 7.0 to 3.5 us and combine from 5.1 to ~4.0 us -> total ~16-17 us (up to ~1.05 PFLOPS). Risk:
fp16 rounding of the partial (0.05%) is comparable to the current kernel-vs-CPU error (7.3e-4 on output up to 2.0);
must be validated against the parity gate and the ULP test before adoption. Keep fp32 as the prototype default.

## 5. Expected absolute TFLOP/s on B1H8S2048 and on filling shapes

Model: `T(s) = max(11.4 us compute, phase-1 write time partially hidden) + combine + 1.5 us launch gap`; the range
spans "writes fully hidden" (optimistic) to "writes fully serialized after the MFMA stream" (pessimistic):

| s | blocks | CUs | phase-1 compute | partial write (hidden .. serial) | combine | total range | TFLOP/s range (central) | vs `t128` | vs vendor |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 128 | 128 | 22.8 us | 3.5 us (0-3.5) | 3.9 us | 26.7-31.7 us | 542-644 (**600**) | 1.06-1.26x | 1.01-1.20x |
| **4** | **256** | **256** | **11.4 us** | **7.0 us (0-7.0)** | **5.1 us** | **18.0-25.0 us** | **687-954 (800)** | **1.35-1.87x** | **1.28-1.78x** |
| 8 | 512 | 256 (2 waves) | 11.4 us | 14.0 us (0-14) | 9.7 us | 22.6-36.6 us | 469-760 (650) | 0.92-1.49x | 0.87-1.41x |

Central case = 50% of the write time hidden. The "1.5 PFLOPS minus combine" estimate in the brief is the
compute-only ceiling (11.4 us = 1.51 PFLOPS); the net number is lower because (a) the per-CU rate at 2 waves/SIMD
is the measured `_vs128` rate 5.9 TFLOP/s/CU (not the vendor's 8.4), (b) phase-1 writes cost 0-7 us, and (c) the
combine is a real 5.1 us serial second launch. Best case still gives **~950 TFLOP/s = 1.87x `t128` / 1.78x
vendor**; the honest central estimate is **~800 TFLOP/s = 1.57x `t128` / 1.49x vendor**.

**Filling shapes (must not regress).** For `N256 >= 256` the GPU is already full with one `_vs128` launch; KV-split
cannot add CUs (its blocks would be `N256*s > 256`, i.e. two block waves) and adds partial+combine traffic, so the
dispatcher keeps `_vs128` (as the measured table in §7.9.5c shows: B4H8S2048 vendor 1852.8 / `_vs128` 1302.5;
B1H8S8192 vendor 2422.5 / `_vs128` 1832.8; `_t128` loses 0.77/0.70x there). The same fallback covers
`129 <= N256 <= 255` (where `_t128` already loses and KV-split's `s>=2` oversubscribes). The only new dispatch
risk is a shape at exactly `N256 <= 128` where the model fails to identify a win; the prototype gate must include
all 15 shapes and prove the dispatch choice + no regression.

## 6. Risk list

| # | risk | likelihood / impact | mitigation |
|---|---|---|---|
| 1 | Phase-1 partial writes do not overlap and cost the full 7 us (s=4) -> total 25 us, only 1.35x | medium / high | fp16 partials (halves), smaller s, write-combining/layout so epilogue bursts overlap the last tile; measure TCC_EA0_WRREQ + time A/B in the prototype |
| 2 | Combine fixed cost ~5.1 us + 1.5-3 us launch gap = 25-30% of the target total; smaller shapes can be dominated by it | high / medium | dispatch condition (4) above; combine is already at 7.4 TB/s (90% of HBM peak), so only fp16 partials reduce it; could fuse the last split's combine with an atomic-free "last block writes" trick only with a grid sync (unavailable) |
| 3 | Merge is not bit-identical; fp16/bf16 partials could break parity | medium / medium | prototype with fp32 partials first (model says ~4e-6 diff, <=0.3% elements at 1 bf16 ULP); explicit split-vs-single ULP test; keep parity gate |
| 4 | Occupancy/regression: two blocks per CU or >512 VGPR if block shape is changed | low | keep the `_vs128` block shape (8 waves, LDS 102400, VGPR 249) - one block/CU gives exactly 2 waves/SIMD, no LDS/VGPR work |
| 5 | Dispatch complexity / filling-shape regression | medium / high | keep `_vs128` for `N256 >= 129`; add `T_kvsplit < 0.95*T_current` guard; gate the dispatcher on all 15 shapes + the filling shapes |
| 6 | Small shapes / key tails: splits with <128 keys, empty splits, `SK < 128*s` | medium / low | require `SK/s >= 128` or let empty splits write lse=-inf and skip; reuse the existing masked last-tile loop |
| 7 | Counter/profiling becomes two dispatches; attribution confusion | low / low | per-phase counters, normalise per wave (`waves = blocks*8`); report totals for the headline |
| 8 | Determinism/races: none expected (no atomics, no cross-block communication) | low | determinism 3x5 on the split path; bit-identical repeats |
| 9 | Effort/opportunity cost: 3-5 dev days for 1.3-1.8x on under-filled shapes only | certain / medium | two-step prototype; stop after step A if <1.1x; do not touch the certified filling-shape path |

## 7. Go / no-go

**GO for the two-step prototype, with the explicit target 1.35-1.8x over `t128` (not 3-4x), and a hard stop if the
measured split=4 total is > 21 us (i.e. < ~0.82 PFLOPS / < 1.4x `t128`).** The lever (CU fill 25/50% -> 100%) is
the only remaining big one and the wave arithmetic is sound; the risks are cost, not feasibility. The filling-shape
path is untouched, so the downside is bounded to the prototype effort.

- **Step A (1 day): split=2 on the `_vs128` block shape.** Expected 542-644 TFLOP/s (1.06-1.26x `t128`), minimal
  kernel diff (key range + epilogue store), full harness/combine stack, gate + determinism + ULP test. This
  validates the two-launch timing, the combine numerics and the dispatch plumbing at the smallest traffic.
  Go to Step B only if Step A measures >= 1.1x `t128` and the ULP histogram matches the model.
- **Step B (2-3 days): split=4** (the real target): 687-954 TFLOP/s. First measure phase-1 write overlap with
  counters; if the total is > 21 us, try fp16 partials (expected +10-15%) before deciding.
- **No split=8** beyond a diagnostic run.

**Prototype size.**
- Kernel (`crates/amdgpu-dsl/src/kernels.rs`): new `fmha_fwd_hd128_mxfp8_kvsplit` entry reusing
  `emit_fmha_vs128_role` with two new parameters: (i) key range `[split*SK/s, (split+1)*SK/s)` in the key loop and
  the `nkv` bound, (ii) a "partial epilogue" mode that writes `Ohat = O/l` fp32 + `lse = m + log2(l)` instead of
  the bf16 output. Plus the 3 new kernarg fields and the grid-z decode. ~200-400 LOC, no ISA/scheduler work.
- Combine kernel: a new tiny kernel `fmha_hd128_kvsplit_combine` (DSL or a HIP C++ prototype): one thread per
  float4, `s` streams + lse, one exp2/max/div per row; ~80-150 LOC; measured 5.1 us at s=4 already (HIP probe).
- Harness: `ref_host.cpp` two-launch path, buffers, `REF_SPLIT`/`_SPLIT`/`_COMBINE` envs, extended args, three
  timing lines, `[grid]` split info, dump/diff mode; `run_ref.sh`/`determinism.sh` auto-build + dispatch;
  ~150-250 LOC. `attn_host` only if the e2e launcher uses FMHA.
- Validation: 15 shapes 16/64 rows, determinism, split-vs-single ULP, dispatch table, filling-shape no-regress:
  ~1-1.5 days (independent, as usual).

## 8. Reproduce this study

```bash
# baseline numbers (already in §7.9.5c, warm protocol):
REF_WARMUP=150 REF_ITERS=100 REF_B=1 REF_H=8 REF_S=2048 REF_SK=2048 REF_HK=8 \
  REF_CO2=build/fmha_fwd_hd128_mxfp8_t128.co REF_KNAME2=fmha_fwd_hd128_mxfp8_t128 \
  REF_ROWS=2 ./build/ref_host "$REF_CO"      # vendor + t128 in the same run
# bandwidth probes used in §4 (HIP, GPU 5):
#   combine s=4 (fp32 Ohat [4][16384][128] + lse [4][16384] -> bf16 out): 38.0 MB in 5.10 us
#   combine s=2: 21.1 MB in 3.90 us; pure 33.6 MB fp32 partial write: 7.00 us (4816 GB/s)
```
Kernel-wiki references: `mi350-arch-topology` (8192 GB/s peak, 256 active CUs/package, 4 MB L2/XCD),
`mi350-arch-l2-paths` (L2 4 MB/XCD, 16 channels, 2048 B/cycle client read, 1024 B/cycle write),
`mi350-bottleneck-memory-bound` (PMC caveats; bandwidth conclusions require timing A/B).
