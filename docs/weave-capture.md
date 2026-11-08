# Capture mode and dependency facts (weave2 DSL foundation)

`FnBuilder` can **capture** the statements a piece of builder code emits instead of appending them to the current block, and **splice**
them back later in a chosen order. Together with per-statement **dependency facts** this lets a weaver (amdgpu-tile `weave2`) build
units with ordinary builder code and then interleave them at statement granularity, legally. Code: `crates/amdgpu-dsl/src/lib.rs`
(capture API), `crates/amdgpu-dsl/src/facts.rs` (facts and the ordering rule); tests: `crates/amdgpu-kernels/tests/capture.rs`.

## API

```rust
let (cap, r) = b.capture(|b| { ...builder code... });   // or: let id = b.capture_begin(); ...; let cap = b.capture_end(id);
b.splice(cap);                                          // append in captured order
b.splice_nodes(nodes);                                  // append nodes of one or more captures in any order
let facts: Vec<Facts> = b.facts(&cap);                  // one per top-level node
let edges = cap.order_edges(&b.f);                      // (i, j): node i must stay before node j
let groups = cap.groups();                              // node ranges that must not be split (weave-region mark pairs)
let unit = cap.summary(&b.f);                           // facts of the whole capture as one unit
```

- Ids (values, loop variables) are allocated as usual; a value defined inside a capture is usable by later code once the capture is spliced
  before it. Constants materialised by the expression layer are shared across captures as in a direct build (so splicing in capture order
  reproduces the direct build exactly); the sharing is visible as an ordinary def -> use dependency in the facts.
- Regions opened **and** closed inside a capture (`loop`, `if`, tile regions, stages) stay one node; a capture that ends with an open region
  panics. Captures nest (LIFO); `finish()` panics while a capture is open. A tile region cannot be spliced into an open tile region.
- **Identity**: capturing and splicing every capture once in capture order gives exactly the function of the direct build (test
  `capture_and_splice_in_order_is_the_identity`; kernels migrated to weave2 are checked by their asm md5).

## Facts (`facts::Facts`)

| field | meaning |
|---|---|
| `class` | `ResClass`: `Mfma { passes }`, `Valu`, `Trans`, `Salu`, `LdsRead`, `LdsWrite`, `VmemLoad`, `VmemStore`, `Dma`, `Sync`, `Marker`, `Region`, `Asm`; `latency()` / `issue()` are nominal cycles (MFMA passes cross-checked against the `amdgpu-isa` atoms) |
| `defs` / `uses` | SSA values defined / read (a region: defined inside / read from outside) |
| `var_reads` / `var_writes` | loop variables read / written with `set_var` semantics (in-place `Var*` updates read and write; `set_var` and the zero-C / C-operand MFMA forms only write) |
| `eff` | `Effects`: LDS read / write, global read / write, DMA, barrier, `wait_lds(n)` / `wait_vm(n)`, fences, markers (tile / stage / weave-region), directives (`set_prio`, `sched_group`, timer), branch region, asm; `exec_change` is always false (DSL branches are wave-uniform) |
| `tile` | `(tile, slot)` of a tile region node |
| `footprint` | result dwords of the defined values (vector-register footprint; `Rsrc` / `I1` count 0) |

Classification is an exhaustive `match` over `Op` / `Stmt`; operands are collected with the generated id visitor, so a new variant or field
cannot be silently missed.

## Legal reordering (`facts::must_order(a, b)`, `a` originally before `b`)

1. **Data**: `b` uses a value `a` defines; loop variables write -> read, read -> write, write -> write.
2. **Anchors** (barrier, waits, fences, markers, directives, asm) keep their order with every effectful statement and with each other; a
   pure statement may cross an anchor (the compiler re-inserts the waits a consumer needs; barriers order memory, not ALU work).
3. **Memory**: LDS write vs LDS read / write (DMA writes LDS) unless the two are tile regions of the same tile with different constant slots;
   global write vs global read / write (no disjointness is assumed).
4. **Branch regions** (`if` / `loop`) are ordered against every effectful statement.

Anything not ordered by these rules may be interleaved; the result still goes through the IR verifier, lowering, the scheduler and
`gpu_verify --strict` like any other kernel.
