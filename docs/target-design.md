# `amdgpu-target`: target description (P0 design)

Part of the four-layer refactor ([`refactor-4layer-plan.md`](refactor-4layer-plan.md)). P0 moves every gfx950 fact behind one
description without changing any output: the golden md5 manifest (`tests/golden/md5_manifest.tsv`), the certificates and the tests
stay identical. gfx1250 (MI450 B0) is described against the same fields as a specification only (target-gfx1250.md on branch `gfx1250-spec`,
phase G).

## 1. Split: data vs. per-instruction tables

| Part | Where | Content |
|---|---|---|
| `amdgpu-target` (new crate, no dependencies) | `src/lib.rs` (types), `src/gfx950.rs`, later `src/gfx1250.rs` | Plain data: wave, register files, LDS, counters, hazard rule table (numbers + citations), latency classes, matrix atoms, descriptor / emission format, LLVM tail settings, verifier limits |
| Per-instruction tables | `amdgpu-compiler/src/target/gfx950/` (moved from `op_attrs.rs`, `emit.rs`, `backend/llvm*.tbl`) | `MOp` → attributes (pipe, cost, counter class, hazard roles, mnemonic / encoding), emit templates, LLVM MIR rows. They name `MOp`, which lives in the codegen crate, so they cannot live in `amdgpu-target` without a cycle |
| Behaviour (passes) | `amdgpu-compiler` (P1: `amdgpu-codegen`) | Unchanged algorithms (scheduler, RA, hazard / wait insertion, verifier) that read their numbers from the `Target` instead of constants |

A compilation carries `&'static Target` (from `Function::target`, default `gfx950`); every pass gets it from the `MFunction` /
compile context. There is no global "current target".

## 2. `Target` fields

```rust
pub struct Target {
    pub name: &'static str,                 // "gfx950"
    pub wave: Wave,                         // lanes: 64 (gfx1250: 32)
    pub regs: RegFiles,
    pub lds: Lds,
    pub counters: Counters,
    pub hazards: &'static [HazardRule],
    pub latency: LatencyModel,
    pub atoms: Atoms,
    pub emit: Emission,
    pub llvm: LlvmTail,
    pub verify: VerifyLimits,
}
```

- **`Wave`**: `lanes`, `exec_bits` (the width of an exec / lane-mask SGPR tuple: 2 SGPRs on wave64, 1 on wave32).
- **`RegFiles`**: `vgpr: RegFile`, `agpr: Option<RegFile>` (gfx1250 has no AGPR segment), `sgpr: RegFile`; `RegFile { count, alloc_granule, tuple_align: &[(dwords, align)], reserved }`, plus:
  - `unified: Option<Unified { total, accum_granule }>`. gfx950: arch VGPR + AGPR ≤ 512, `accum_offset` rounded to 4, `next_free_vgpr` to 8.
  - the SGPR rules: usable count (102), the special registers outside it (vcc, exec, m0), the user / system SGPR layout (kernarg pointer in s[0:1], workgroup ids).
  - `occupancy: Occupancy` — a model, not just a table:
    - `Steps(&[(regs, waves_per_simd)])`: gfx950, waves per SIMD from the VGPR / AGPR (unified) and SGPR counts.
    - `BlockPool { block_vgprs, blocks }`: gfx1250, a pool of 64 blocks of 16 VGPRs shared by the waves; SGPRs are fixed and not an occupancy term.
- **`Lds`**: `bytes_per_workgroup` (160 KiB), `banks`, the DMA forms and their M0 rule (`m0_base_plus_imm`).
- **`Counters`**: a generic counter set.
  - `counters: &[Counter { name, max, in_order }]`.
  - `classes: &[(MemClass, CounterId, OrderKind)]`. `MemClass` = vmem load / vmem store / lds / lds-dma / smem / msg / tensor / async / ….
    `OrderKind` is per class, not per counter: `InOrder` or `OutOfOrder` (SMEM on both targets, so a wait for it is always 0). On gfx1250 one counter
    can be in order for one class and out of order for another (XCNT: VMEM vs SMEM translation order).
  - `wait: WaitForm`, one of:
    - `Combined { mnemonic, fields }`: gfx950 `s_waitcnt vmcnt(n) lgkmcnt(n) expcnt(n)`.
    - `PerCounter(&[(CounterId, mnemonic)])`: gfx1250 `s_wait_loadcnt` / `storecnt` / `dscnt` / `kmcnt` / `asynccnt` / `tensorcnt` / `xcnt`.

  gfx950: `vmcnt` (63; loads and stores, in issue order), `lgkmcnt` (15; LDS in order, SMEM out of order, so an SMEM wait is always 0), `expcnt`.
  The wait verifier and the wait insertion iterate over the set instead of the two named fields.
- **`HazardRule`**:
  - `{ name, producer: Class, consumer: Class, role: Operand, waits: Waits, scope: Value | Physical, source }`.
  - `Waits { count: Count, unit: Unit }`.
    - `Count` is a number, a function of the producer (MFMA passes: `passes + 3 + extra`), or `Instr` for a required instruction (gfx1250 `s_wait_*`).
    - `Unit` is the distance unit: `WaitStates` (`s_nop` / any instruction issued, gfx950) or `IndependentValu` (only independent VALU instructions count, as in the gfx1250 WMMA `v_nop` rule, where co-execution state changes the count).
  - `source` cites the evidence: LLVM `GCNHazardRecognizer` function, probe name, or ISA page.

  The gfx950 table is the audit table of [`trust-inventory.md`](trust-inventory.md) §5.1. `hazards.rs` keeps its algorithms (value-level ready map, post-RA physical passes) and reads the numbers per rule.
- **`LatencyModel`**: per pipe (`Valu / Trans / Mfma / Lds / Vmem / Salu / Smem / Ctrl`) issue and occupancy, the MFMA pass table, the transcendental latency, and `trans_latency` and `mfma_dst_wait` as rules rather than constants. The per-MOp cost entries reference these classes.
- **`Atoms`**: MFMA / WMMA / SWMMAC shapes with lane layouts per operand (what `amdgpu-isa` holds today), the copy atoms (DMA widths, `tr_b8` / `tr_b16` reads), and the scale operand forms.
- **`Emission`**:
  - kernel descriptor: the field list with limits and rounding (`next_free_sgpr` ≤ 102 rounded to 8 and capped, `next_free_vgpr`, `accum_offset`, LDS);
  - `.amdhsa_*` user / system SGPR flags, code-object version, directives;
  - `mcpu`.
- **`LlvmTail`**: `mcpu`, dialect (`llvm20` / `llvm24` tables), llc arguments.
- **`VerifyLimits`**: the `gpu_verify` constants (`VGPR_LIMIT`, `SGPR_LIMIT`, `LDS_LIMIT`, buffer `num_records` rules).

## 3. Inventory of gfx950 facts to move (P0)

| Today | Becomes |
|---|---|
| `regalloc.rs` `VGPR_COUNT` / `AGPR_COUNT` / `UNIFIED_FILE` / `SGPR_COUNT` / reserved, `unified_budget` | `target.regs` |
| `emit.rs` descriptor block (rounding, `.amdhsa_*`, accum offset) and the mnemonic match | `target.emit` + `target/gfx950/emit.rs` |
| `op_attrs.rs` (cost, pipe, latency, `wdelay`, hazard roles, counter class) | `target/gfx950/op_attrs.rs` over `target.latency` / `target.counters` |
| `hazards.rs` numbers (vcc 2, trans 5, MFMA tables, M0, VALU→SGPR 5 / 2, wide-store 2, DPP / permlane 2) | `target.hazards` rows |
| `sched/model.rs`, `sched/waits.rs` counter names / maxima | `target.latency`, `target.counters` |
| `agpr.rs` unified-file rules | `target.regs.unified` |
| `gpu_verify.rs` limits | `target.verify` |
| `backend/llvm20.tbl`, `llvm24.tbl`, `llvm_mir.rs` headers, mcpu strings | `target/gfx950/llvm/*`, `target.llvm` |
| `amdgpu-isa` atoms | `target.atoms` (the crate stays as L3's atom API over them) |
| `dispatch_compile.rs` descriptor check, `-mcpu=gfx950` in tools / tests | read from `target.emit` / `target.llvm` |

Each row is one move commit: behaviour-neutral, md5 manifest identical, certificates identical, `cargo test` + `run_all.sh` green.

## 4. How gfx1250 fits (check, not implementation)

- **Wave and registers:** wave32; VGPR allocation granule; fixed SGPRs. These are data in `wave` / `regs`.
- **Split counters:** `PerCounter` wait form. Classes: TDM → TENSOR, async LDS DMA → ASYNC, scalar memory → KM.
- **Hazards:** WMMA / SWMMAC software hazards and VALU dependency distances are `HazardRule` rows. Rows that need an instruction rather than wait states use `Waits::Instr`.
- **Matrix atoms:** WMMA lane layouts go in `atoms`.
- **Not described:** VOPD pairing and clusters are not described by P0 fields. They become `Target` extensions when a gfx1250 backend exists. Phase G records them in prose.
