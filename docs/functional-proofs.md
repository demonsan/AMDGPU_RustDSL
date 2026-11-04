# Functional proofs (SF-Tensor-inspired, zero dependencies)

Two machine-checked layers, no Z3/SMT and no new dependencies. Both are executable Rust tests; a
failure prints a concrete offending coordinate and the tests are part of `cargo test`.

## E1 — GEMM index-level proof (`amdgpu-kernels/tests/functional_gemm.rs`)

The proof composes the layouts the MXFP8 GEMM actually builds:

| kernel object | layout / proof |
|---|---|
| CTA / waves | 4 waves 2x2; `wave_m=cta_m/2`, `wave_n=cta_n/2`; every CTA `(m,n)` belongs to exactly one wave block |
| A global image | `a_glob = [cta_m,128] : [K,1]`; every `(row, global K)` byte is unique; padded A image uses the same logical row/K map |
| B preshuffle | `b_glob = ([16,cta_n/16],[16,8]) : ([16,16K],[1,256])`; checked against the kernel formula `(n/16)·16K + (k/16)·256 + (n%16)·16 + k%16`, including the per-K-block advance `2048` |
| MFMA atom | `v_mfma_scale_f32_16x16x128_f8`, kernel orientation `swap_ab`; `MmaAtom` A/B/C TV layouts checked injectively by `proofs::check_tv` |
| accumulator | the transposed C TV layout `e = n' + 16 m'`; 8x8 atom blocks per wave cover each 256x256 CTA exactly once |
| K schedule | v3 `Cadence::TwoSlot`, v4 `Cadence::RegStaged`; the compute sequence contains every K tile exactly once, and 2-slot reuse is `generation(t+2)` after `generation(t)` |
| scales | A row `(m,kb)` and B 128-column `(n/128,kb)` ownership; the v4 DPP/permute panel transposition is checked by `verify_sfa_transpose()` (the same proof the v4 builder runs at build time) |
| epilogue | every `(m,n)` maps to one C address `2·(m·N+n)`; duplicate stores are rejected |

The proof counts `(m,n,kb)` contributions and requires exactly one for every output and every
K-block, and requires `A[m, kb·128..kb·128+127]` / `B[n, ·]` plus the corresponding scale rows to
exist. It runs for both `GemmConfig::v3()` and `GemmConfig::v4()`.

**Regression counterexample.** `prove(cfg, drop_k_tile=Some(1))` models the historical "tile 1
swallowed" bug by omitting the second K block from the compute schedule. The proof fails with:

```
missing/duplicate contribution C[0,0] K block 1 (k=128): 0 times
```

That is a concrete `(m=0,n=0,k=128)` counterexample, not just a boolean failure.

Assumptions: index/layout level (not IEEE floating point), one CTA of the global tiling per proof
run (grid tiling is the same map shifted by `(by·cta_m,bx·cta_n)`), and the schedule is the
cadence's declared compute order; the actual generated IR was separately verified by `gpu_verify`
and the GEMM gate. The test is a proof of the layout compositions, not a replacement for the
runtime CPU reference.

## E2 — online-softmax rebase/rollback recurrence (`amdgpu-compiler/tests/functional_proofs.rs`)

For one row with arbitrary length, real arithmetic. Streaming state after a prefix:

```
m = max score seen
d = Σ exp(s_i - m)          o = Σ exp(s_i - m) v_i
```

Invariant: `o·exp(m) = Σ exp(s_i) v_i` and `d·exp(m) = Σ exp(s_i)`.

For a new score `x`, new frame `m' = max(m,x)`, the rebase/rescale step is
`o ← o·exp(m-m')`, `d ← d·exp(m-m')`, then `o += exp(x-m')·v`, `d += exp(x-m')`.
`functional_proofs::softmax_invariant_proof` proves the induction identities symbolically

```
(S·exp(m-m') + exp(x-m')·v)·exp(m') = S·exp(m) + exp(x)·v
(D·exp(m-m') + exp(x-m'))·exp(m')   = D·exp(m) + exp(x)
exp(a-b)·exp(b-a) = 1
```

using only `exp(a)exp(b)=exp(a+b)`, additive cancellation and constant folding, for arbitrary
symbols `S,D,v,m,m',x`. Rollback/rebase to an earlier frame is the third identity, so thresholds
only change the reference frame and not the mathematical result.

**Counterexample.** Setting `rescale=false` when the max increases (the broken recipe) is not
proved; the numeric host test uses `scores=[0,10,2]`, `vals=[1,-2,0.5]`:
naive value `-1.9997868688`, broken recipe `-0.9998322687` — a concrete mismatch. The good recipe
matches the naive reference for lengths 1,2,7,64 in the same test.

Run:

```bash
cargo test -p amdgpu-kernels --test functional_gemm -- --nocapture
cargo test -p amdgpu-compiler --test functional_proofs -- --nocapture
```

## Relationship to the design doc

This is the first two steps of the SF Tensor direction in `docs/ir-review-tirx-sftensor.md`:
index-level composition proofs (E1) and a real-arithmetic loop invariant (E2), kept as small
in-tree checkers rather than an SMT stack. `gpu_verify` remains the memory/synchronization proof;
these layers add functional evidence without changing the shipped kernels or their assembly.


## E3 status (actual-IR symbolic execution, region-IR tree)

The E3 prototype is on branch `ir-proof`:

- `crates/amdgpu-kernels/src/ir_proof.rs` interprets the **actual built `Function::body()` tree**
  (region IR), enumerates the 4 waves/64 lanes, unrolls the pipeline loop to 3 K tiles, maps
  actual global DMA/LDS addresses to element tags and executes actual MFMA/store statements.
- The proof-only `GEMM_PROOF_DROP_KB` hook is in the real GEMM builder (a real IR `if` around the
  MMAs of one tile generation), so setting it to 1 changes the kernel Function, not a replica.
- Current observed counts for the v4 direct-C variant: 1,084,416 executed IR statements, 49,152
  actual MFMA statements, 65,536 actual C stores, K generations {0,1,2}. The E3 theorem is not yet
  green: operand provenance through the LDS register-staged path remains incomplete (most MFMAs
  see empty A/B tags in the current tag map; see `mfma_no_tags` diagnostics). The two E3 tests are
  marked `#[ignore]` with this reason. This is the remaining work for E3; no counterexample is
  claimed until operand provenance is sound.
