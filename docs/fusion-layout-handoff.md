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

**In the FMHA.** `amdgpu_kernels::fmha::attention_orientation()` runs the search and asserts that the winner is the orientation the builder
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

### 3.2 Fused GEMM → elementwise → GEMM (`crates/amdgpu-kernels/src/mlp/mod.rs`)

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

### 3.3 Weight-stationary, persistent kernels (`mlp_fp8_ws_{fused,g1,g2}`)

The building blocks of §3.2 are tuned along the structure the shape calls for. The weights are tiny next to the activations (2 × 64 KB
vs MBs of X / Y) and K = 256, so the GEMM v4 K-loop ring would barely run. Instead:

- **Weights in LDS once per CTA.** Both weight images are DMA'd into LDS (`buffer_load_dwordx4 … lds`, 136 KB) and stay there.
  - The images are host-prepared with 272-byte rows (256 data + 16 pad), so the 16 rows of a `ds_read_b128` phase hit 16 distinct bank
    windows.
  - Every fragment read is a lane base plus an immediate, via `load_lds_b128x2`.
  - The fused kernel's W2 image is the σ-permuted copy.
- **Persistent CTAs.** 256 CTAs loop over 128-row blocks (c, c + G, …). Row operands are direct 16-byte loads, and the next block's X is
  prefetched during GEMM 2.
- **Same hand-off.** The `StageBoundary` is checked exactly as in §3.2.

**Accuracy.** The same host reference and checks pass: 0 bad, worst relative error 1.6e-3, H checked separately.

**Timing** (MI355X GPU 5, interleaved, medians of 7 × 50 launches):

| M | ws fused | ws unfused pair | fusion speedup | vendor GEMM M×256×256 (one launch) | vendor pair lower bound (2 launches, no activation / requant) |
|---|---:|---:|---:|---:|---:|
| 65536 | **22.3 µs** (772 TF/s) | 38.0 µs | **1.71×** | 11.4 µs | 22.8 µs |
| 262144 | **62.1 µs** (1107 TF/s) | 118.6 µs | **1.91×** | 41.2 µs | 82.4 µs |

**No comparable vendor kernel.** aiter has no fused fp8 MLP with an fp8 intermediate. The closest reference is the aiter/opus blockscale
GEMM (PR #6696 `.co`) run twice at M×256×256. That is a lower bound for the vendor path: it omits the relu and the fp8 requantisation of H
(which would need a third kernel or epilogue support). Against that bound:

- the fused kernel matches it at M = 65536 (22.3 vs 22.8 µs);
- it is 1.33× faster at M = 262144.

Our unfused building blocks are still slower than the vendor GEMM at these shapes (38 vs ≈ 23 µs for the pair).

**Remaining limiter.** The row-operand fragments are built with `cat_frag8(load, load)`, which the lowering materialises with `v_mov`
right after the loads. Each load pair is therefore followed by `s_waitcnt vmcnt`, which serialises the X prefetch. A copy-free
`buffer_load_b128x2` has been requested from backend-llvm-tail.

### 3.4 The boundary on the region IR (`Node::Stage`, GV-STAGE)

Rebased on region IR (origin/region-ir 7c53884).

**IR.** The stage is a separate node kind beside `Region { Produce | Consume }`, which keeps its tile protocol meaning.
- **Node:** `Node::Stage { name, contract: Box<StageContract>, body }`. Flat markers: `Stmt::StageBegin { name, contract }` / `Stmt::StageEnd`.
- **Nesting:** `TreeBuilder` opens and closes stages like regions. A stage opened inside a tile region is rejected at build time; tile
  regions may nest inside a stage.
- **Builder:** `stage_begin` / `stage_end`, the closure `stage(name, contract, |b| ..)`, and `stage_live_out(v)` to add hand-over values
  computed inside the stage.
- **`StageContract` (amdgpu-dsl, IR level):** `live_in` / `live_out`: `StageVal` { name, IR values, element-table fingerprint from the tile
  layer, element bits, storage }, `lds_freed` / `lds_reused`, `barrier_phase`, `roles`.

**Lowering.** The lowering walks into the stage body; the markers emit nothing, so a stage is transparent to lowering and scheduling.

**`gpu_verify::verify_stages` (GV-STAGE, errors).** Checks at the stage edge:
- every `live_in` is handed over by an earlier stage's `live_out` with the same values, the same table and width, and the same roles;
- `live_out` values are defined;
- a value defined in one stage and read in another must be declared `live_in`;
- an LDS tile is reused only if an earlier stage freed it, and a freed tile is written again only by a stage that declares the reuse;
- `barrier_phase` equals the number of barriers that precede the stage.

Proven hand-offs are counted under `stage-handoff` in the report.

**In the fused MLP.** The fused kernel's loop body is two stages, `gemm1_act` (live-out P0..P3: producer tables) and `gemm2` (live-in
P0..P3: consumer tables), with the next block's X prefetch between them.
- **Verifier:** `gpu-verify --strict` reports `stage-handoff: 4` proven, 0 errors, 0 warnings.
- **Transparency:** the assembly is byte-identical to the pre-stage build.

**Tests.** `crates/amdgpu-compiler/tests/stage_ir.rs` covers:
- tree / flat round trip;
- a layout mismatch, an undeclared cross-stage read and a wrong barrier phase, each reported;
- a stage inside a tile region panicking.

`region_ir.rs` is extended.

**md5.** All 107 `amdc list` kernels, plus the 4 FMHA v3 kernels and GEMM v3 / v4, emit byte-identical assembly to region-ir 7c53884.

**Tests.** Full `cargo test`: 73 targets, 334 passed.

### 3.5 Copy-free row-operand loads and slotfill

**`buffer_load_b128x2`** (backend-llvm-tail, origin/buf-b128x2) replaces `cat_frag8(load, load)`. The two loads land directly in the halves
of one 8-dword operand, so there are no `v_mov` copies and no immediate `vmcnt` waits. The X prefetch now overlaps.

**Gate** (GPU 5, interleaved, medians of 7 × 50 launches; 0 bad against the host reference, H checked; strict verify 0 / 0):

| M | ws fused | ws unfused pair | speedup | vendor two-launch lower bound |
|---|---:|---:|---:|---:|
| 65536 | **20.2 µs** (852 TF/s) | 33.3 µs | 1.65× | 22.8 µs |
| 262144 | **54.1 µs** (1270 TF/s) | 96.6 µs | 1.79× | 82.4 µs |

**slotfill** (sched, origin/slotfill, via `AMDC_SLOTFILL=mlp_fp8_ws` on a temporary merge): the assembly of all three ws kernels is
unchanged, and timing is identical (55.2 vs 55.1 µs fused at M = 262144). The weight-stationary loop body already interleaves its operand
reads with the MFMAs, and the X prefetch is the only VMEM traffic in the loop. Not adopted.
