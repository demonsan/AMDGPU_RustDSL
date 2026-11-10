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
| K schedule | v4 `Cadence::RegStaged` (the v3 `TwoSlot` cadence was removed with the v3 GEMM); the compute sequence contains every K tile exactly once, and 2-slot reuse is `generation(t+2)` after `generation(t)` |
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

## E2 — online-softmax rebase/rollback recurrence (`amdgpu-codegen/tests/functional_proofs.rs`)

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
cargo test -p amdgpu-codegen --test functional_proofs -- --nocapture
```

## Relationship to the design doc

This is the first two steps of the SF Tensor direction in `docs/ir-review-tirx-sftensor.md`:
index-level composition proofs (E1) and a real-arithmetic loop invariant (E2), kept as small
in-tree checkers rather than an SMT stack. `gpu_verify` remains the memory/synchronization proof;
these layers add functional evidence without changing the shipped kernels or their assembly.


## E3: symbolic execution of the built kernel Program (region-IR tree)

`crates/amdgpu-kernels/src/ir_proof.rs` executes the **actual `Function::body()` tree** of the v4
builder at a small symbolic configuration: one 256x256 CTA, K=384 (three K tiles), 256 threads
enumerated with concrete lane/wave ids (`WorkitemId` 0..255, lane=`tid&63`, wave=`tid>>6`).
Global loads/DMA produce element tags `A[m,k]`, `B[k,n]`, `Sa[m,kb]`, `Sb[block,kb]` from the
emitted address expressions; LDS is a byte-value map (`M0 + imm + lane*size` destination per
`gpu_verify`'s `BufferLoadLds` semantics); reads and MFMAs execute lockstep over all threads, so
tile-generation correctness is not assumed. MFMA operands are classified by logical tag kind
because the atom is built with `swap_ab` (hardware src A carries the logical B fragment and src B
the logical A fragment; the scale slots are swapped the same way). The epilogue's accumulator
provenance (`Tag::Acc(var, dword)`) maps each stored C element to its K contributions, for both the
shipped LDS-transpose epilogue and the direct-C variant.

On the **shipped `gemm_mxfp8_bs_v4()` Function** (LDS epilogue) the trace is:

- 616,960 executed IR statements, 49,152 actual `VarMfmaScale16` statements, 65,536 actual C
  stores, 196,608 C contributions = exactly 3 per output (K generations 0,1,2);
- every C[m,n] has exactly one contribution per K generation and is stored exactly once;
- every K generation's A/B operand tags cover the full 256x128 tile and the scale tags match the
  output rows/blocks.

The proof-only `GEMM_PROOF_DROP_KB` hook is a real IR `if` in the reg-staged compute path of the
kernel (`Body::compute_tile`): it drops one tile generation from the emitted Function, not from a
replica. Setting it to 1 makes the same checker fail with a concrete counterexample on the shipped
v4 Function: `missing C[0,0] K block 2; duplicated K blocks [1]` (one generation is swallowed and
the shifted pipeline duplicates another). The theorem and the kernel-code counterexample are
un-gated tests in `crates/amdgpu-kernels/tests/functional_gemm_ir.rs`; the direct-C v4 variant is
checked by the same checker as well.
