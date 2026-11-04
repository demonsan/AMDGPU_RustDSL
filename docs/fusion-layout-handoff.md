# Fusion hand-off modeling (program item G)

When two MMA stages are fused, the first stage's result is handed to the second in registers. Its TV layout (which lane and register holds which
element) generally differs from the layout the second stage's operand needs. This document covers how the conversion is planned and lowered,
how the MMA orientations are chosen to make it free, and (step 3, pending region IR) the region-boundary resource contracts.

## 1. `convert_layout`

`amdgpu_layout::convert::plan_convert(src_tv, dst_tv, elems_per_reg) -> ConvPlan` takes two static TV layouts of the same tensor
(thread = mode 0, value = mode 1). It computes `M = dst⁻¹ ∘ src` per slot and classifies it:

| class | condition | lowering (`amdgpu_tile::handoff::convert_layout`) | cost |
|---|---|---|---|
| Identity | every element stays in its slot | none (a view) | 0 |
| Register | same lane, whole registers move | register renaming (`reg_map`) | 0 |
| RegisterBytes | same lane, sub-dword elements change byte position | `v_perm_b32` (cost model only) | ⌈sources/2⌉ per register |
| Quad | `lane >> 2` kept | one `dpp_quad_perm` per (dst reg, src reg) + `v_cndmask` merges on `lane % 4` | as emitted |
| Half | lane `t` or `t ^ 32` | today the LDS round trip (`v_permlane32_swap` not exposed by the DSL) | — |
| CrossLane | anything else | wave-private LDS round trip: write at `4 (64 v + t)`, read at `base(r') XOR lin(lane bits)` | 2 · regs + 1 |

When the thread and value counts are powers of two, `ConvPlan::gf2` holds the GF(2) matrix of the slot-index bits. `bit_moves()` then names
the moved bits, e.g. `lane bit 5 -> value bit 0`.

**Build-time proofs.** The quad lowering is simulated on element ids and compared with the plan (`simulate_quad`). For the LDS round trip,
the XOR-linear read address is derived from the plan and checked for all 64 lanes.

**Hardware probes.** `handoff_probe_{reg,quad,cross}` (one self-checking wave each, `tests/gpu/handoff_host.cpp`) were run on MI355X GPU 5,
with 0 bad slots out of 1024 in every case:

| probe | conversion | instructions emitted |
|---|---|---|
| reg | register renaming | 0 |
| quad | lane bits 0 ↔ 1 | 16 `v_mov_b32_dpp` |
| cross | 32x32 MFMA C → its transpose | 16 `ds_write_b32` + 16 `ds_read_b32` |

## 2. MMA orientation search

`amdgpu_tile::orient::search_attention(atoms)` enumerates every candidate for the attention hand-off S → softmax → P → P·V:

- **Producer atom and orientation:** `S` (A = Q, B = K) or `S^T` (A = K, B = Q).
- **Consumer atom.**
- **Role of P:** the A operand (`O = P V`) or the B operand (`O^T = V^T P^T`).
- **Key permutation σ:** a free permutation of the consumer's K inside one K block, absorbed by V's read layout.

The cost of a candidate has two parts:

- **S → P conversion.** It is zero when a σ exists under which every lane's multiset of (query, σ(k)) equals the multiset of (query, key) it
  holds in S; the conversion is then a register renaming, because the fp8 packing instruction takes any four registers. Keys and k slots are
  matched by their (lane, query) incidence signatures. Otherwise the cost is the `plan_convert` cost with σ = identity.
- **Softmax statistics → O alignment.** It is zero when every O element's query is held by its own lane in S. The row max and row sum live
  per query in S's lanes, and the O rescale consumes them.

Ties prefer the larger atom: fewer MFMAs and operand registers per FLOP.

Result over the scaled fp8 32x32x64 and 16x16x128 atoms (`crates/amdgpu-tile/tests/orient.rs`), sorted by cost:

| producer | S^T | consumer | P | conversion | statistics |
|---|---|---|---|---|---|
| **32x32x64** | **yes** | **32x32x64** | **B** | **0** | **0** |
| 16x16x128 | yes | 16x16x128 | B | 0 | 0 |
| 32x32x64 | yes | 32x32x64 | A | 0 | 64 |
| 32x32x64 | no | 32x32x64 | A | 65 (cross-lane) | 0 |
| 32x32x64 | no | 32x32x64 | B | 65 | 32 |
| … | | | | | |

The winner is the shipped FMHA design: S^T = K Q^T, then O^T = V^T P^T. The test also checks the FMHA's own packing. P word slot j of lane
l comes from S accumulator j / 16, value j % 16, which is key `32 (j / 16) + 8 (r / 4) + 4 (l / 32) + r % 4`. That packing is one of the
zero-cost permutations: every lane's (query, key) multiset is exactly its S multiset.

**In the FMHA.** `amdgpu_tile::fmha::attention_orientation()` runs the search and asserts that the winner is the orientation the builder
implements. The QK, Q-load and O-store atoms are taken from it.

**Acceptance check.** `amdc emit` of the four v3 FMHA kernels gives assembly byte-identical to Main 1c3c9cf:

| kernel | md5 (both builds) |
|---|---|
| vs128 | 68d2832b0bf4 |
| kvsplit | e4fc18b14aac |
| kvsplit_combine8h | 24a95a51a69e |
| t128 | e4c7e75d140f |

**Limitation.** The V side of σ (the `ds_read_b64_tr_b8` register mapping) is not modeled as a layout. The search proves that the P side is a
zero-cost renaming and that the FMHA's packing is a valid σ. That the transposed V reads realize the same σ is still covered only by the
FMHA's end-to-end correctness, not by a layout proof.

## 3. Stage-boundary resource contracts and the fused MLP example

### 3.1 `StageBoundary`

`amdgpu_tile::contract_stage` defines the contract model. Each side of a stage boundary is a `StageBoundary` with these fields:

| field | content |
|---|---|
| `live` | the fragments handed over (`LiveFrag`): name, element table over the tensor after any K permutation, element bits, storage (VGPR / AGPR / an LDS region) |
| `pinned` | register ranges that must survive the boundary untouched |
| `lds_freed` / `lds_reused` | LDS regions the producer frees and the consumer reuses |
| `vm` / `lgkm` | outstanding memory counts: the producer's worst case at the boundary, or the consumer's tolerance |
| `barrier_phase` | how many barriers have been passed (cross-wave stages must agree) |
| `roles` | the wave roles of each side |

`check_boundary(prod, cons)` proves the producer satisfies the consumer:

- every consumed fragment is handed over with the same shape and element width;
- the storage is compatible;
- every reused LDS region was freed, and the pinned ranges are provided;
- the outstanding counts are within the consumer's tolerance;
- the barrier phase and the roles are equal.

It returns the `plan_convert` movement and cost of every hand-off. A boundary emits no instruction and is not a scheduling barrier.

**Status before item B.** The checks run as build-time assertions in the kernel builder. When region IR lands, the boundary moves onto
the Region node and the verifier checks it. Tests: `crates/amdgpu-tile/tests/contract_stage.rs` covers an identity hand-off, a transposed
expectation reported as cross-lane, and four error cases.

### 3.2 Fused GEMM → elementwise → GEMM (`crates/amdgpu-kernels/src/mlp_fp8.rs`)

The chain is `Y = fp8(relu(0.125 · X W1)) W2`:

| tensor | shape and type |
|---|---|
| X | [M][256] fp8 |
| W1 | [256][256] fp8 |
| W2 | [256][256] fp8 |
| Y | [M][256] bf16 |

All MFMAs are the scaled 32x32x64 fp8 atom with unit scales. One CTA is 4 waves × 32 rows.

**Orientation.** The kernel uses the orientation the search derives, the same one as the FMHA S → P hand-off:

- GEMM 1 computes `H^T = W1 X^T` (A = W1 rows, B = X rows), so each lane owns one row m.
- relu, scaling and fp8 packing happen in accumulator order.
- The packed words are GEMM 2's B operand (`Y^T = W2 H^T`) as they are, up to the key permutation
  `sigma(k) = 32 (k/32) + 8 ((k%16)/4) + 4 ((k%32)/16) + k%4` inside every 64-wide n1 block.
- `sigma` is absorbed by a pre-permuted copy of W2, done once offline like the vendor's weight preshuffle.

The builder constructs the producer and consumer `StageBoundary` of each of the 4 k-blocks and asserts `check_boundary` → `Register`/`Identity`
with cost 0: the hand-off is free.

**Unfused pair.** `mlp_fp8_g1` (GEMM 1 + elementwise, H written as fp8 to HBM) and `mlp_fp8_g2` (GEMM 2 reading H, natural n1 order) are
built from the same code.

**Gate (MI355X GPU 5, `tests/gpu/mlp_host.cpp`).**

- **Correctness:** compared against a host reference that recomputes the same fp8 rounding of H, on 131072 sampled outputs: 0 bad, worst
  relative error 1.6e-3 for both the fused and the unfused path.
- **Fused vs unfused:** the outputs are bitwise equal for 99.75% of elements. The rest differ because `sigma` reorders the MFMA's
  k accumulation.
- **Strict verify:** `amdc gpu-verify --strict` gives 0 findings and 0 unproven for all three kernels.

**Timing** (interleaved, medians of 7 × 50 launches):

| M | fused | unfused pair (2 launches) | speedup |
|---|---:|---:|---:|
| 65536 | 43.6 µs (394 TF/s) | 58.8 µs (292 TF/s) | **1.35×** |
| 16384 | 21.4 µs | 27.3 µs | **1.28×** |

**Scope of the speedup.** All three kernels use direct global-to-register operand loads, with no LDS staging and no software pipelining. The
speedup is fusion on equal building blocks: H no longer goes through HBM, and there is one launch fewer. It is not a tuned MLP: absolute
throughput is far below the GEMM v4 kernel.
