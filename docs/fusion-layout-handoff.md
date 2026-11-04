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

## 3. Region-boundary resource contracts (pending item B)

Not started: waits for backend-llvm-tail's region IR.
