# gfx1250 target description — phase G specification (public sources only)

Phase G of [`refactor-4layer-plan.md`](refactor-4layer-plan.md): a documented *description* of the AMDGPU target **gfx1250** against the
`Target` fields of the P0 design (`docs/target-design.md`, branch `target-p0`). It is a specification check, not an implementation:
no pass, no code generation, no hardware run, no change to any gfx950 output.

## 0. Sources and citation rules

Only two public sources are used:

1. **[ISA]** AMD *"CDNA5" Instruction Set Architecture Reference Guide*, 27-July-2026.
   - Public URL: <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna5-instruction-set-architecture.pdf>, linked from <https://gpuopen.com/amd-gpu-architecture-programming-documentation/>.
   - Cited as `CDNA5 ISA §section (p. printed page)`.
   - The document may not be duplicated, so facts are paraphrased. Instruction and field names, numbers and very short phrases are kept as they are.
2. **[LLVM]** Upstream llvm-project at commit `274ab686e241322e1acdef07a7e6964c83d47911` (2026-11-09).
   - Files used: `llvm/lib/Target/AMDGPU/*`, `llvm/lib/TargetParser/AMDGPUTargetParser.cpp` and `llvm/include/llvm/TargetParser/AMDGPUTargetParser.h` (rows marked **[TP]**), `llvm/docs/AMDGPUUsage.md`, and the MC tests `llvm/test/MC/AMDGPU/gfx1250*.s`, `gfx12_asm_*.s`.
   - Cited as `LLVM <path>:<line> @274ab686`, with the path relative to `llvm/` or to `llvm/lib/Target/AMDGPU/` where the section says so.

Rules for every row:
- **Source:** each row cites one of the two sources. A value we inferred is marked **derived**.
- **No public source:** the row says "unknown (no public source)".
- **Disagreement:** where the ISA and LLVM disagree, or one is silent, the row says so. The value this description uses is given with the reason.

**Products and steppings:**
- No public source ties CDNA5, gfx1250 or `gfx1250-strict` to a product name or a silicon revision. AMDGPUUsage lists the product names as "TBA" (§7, `emit`). This description therefore covers *gfx1250 as the public ISA and upstream LLVM describe it*.
- The correspondence between the public CDNA5 ISA and the LLVM target `gfx1250` (rather than `gfx1250-strict`) is **derived** from matching opcodes and gfx1250-only instructions (§7, "Correspondence").

The assembler smoke checks are in [`tests/gfx1250_smoke/`](../tests/gfx1250_smoke/run.sh); see §8.

Contents:
- §1: summary of the proposed `Target`.
- §2: gfx950 ↔ gfx1250 differences and the L4 pass classification (our own analysis).
- §3–§7: the fields, each with its source table and Open list.
  - §3: `wave` / `regs` / `lds`.
  - §4: `counters` / memory model / `sync`.
  - §5: `hazards` / `latency`, with co-execution and VOPD as prose.
  - §6: `atoms` / `mem`.
  - §7: `emit` / `llvm` / `verify` / products / correspondence.
- §8: assembler smoke checks.

## 1. Summary: the proposed `Target` for gfx1250

The detail and the exact citations for each row are in the subsection named in the last column. Subsections keep their own numbering (P1–P5).

| Field | gfx1250 value | Main sources | Detail |
|---|---|---|---|
| `wave` | `lanes: 32`, `exec_bits: 1`; no wave64 | ISA §3; LLVM feature set | P1 §1 |
| `regs.vgpr` | 1024 per wave, allocated in blocks of 16, even-aligned tuples (≥ 64-bit).<br>v0–v255 are encodable directly. Higher registers take address bits [9:8] from MODE, set by `s_set_vgpr_msb`.<br>A few operands are limited to v0–v255: VIMAGE VADDR3-4, the WMMA scale operands and `v_wmma_ld_scale*` | ISA §3.3.2 (pp. 16–18), §7.12.6; LLVM `AMDGPUBaseInfo.cpp`, `GCNFeatures.td` | P1 §2.1 |
| `regs.agpr` / `unified` | `None` / `None` | LLVM `AMDGPU.td:587`; the ISA lists only SGPR and VGPR | P1 §2.2 |
| `regs.sgpr` | 106 (s0–s105), fixed, not an occupancy term.<br>VCC is 106/107, TTMP0–15 are 108–123, NULL 124, M0 125, EXEC 126/127.<br>Alignment: even for 64-bit, quad for wider operands.<br>Up to 32 user SGPRs. Work-group IDs are architected in TTMP9 / TTMP7 | ISA §3.3.1, §3.5.3; LLVM `AMDGPUUsage.md`, [TP] | P1 §2.3–2.4 |
| `regs.occupancy` | `BlockPool { block_vgprs: 16, blocks: 64 }`, at most 16 waves per SIMD. The ISA is silent on both numbers | LLVM `GCNFeatures.td:1907`, [TP] `getTotalNumVGPRs` | P1 §2.5 |
| `lds` | 320 KiB per work-group, granule 2048 B, 64 banks × 4 B.<br>Async and cluster loads take the LDS address from a per-lane **VGPR**; TDM takes it from the descriptor. There is no M0 LDS base, and the gfx950 `m0_base_plus_imm` rule does not apply.<br>Conflict: LLVM `TCP_SPLIT` lists LDS up to 384 kB; the ISA says 320 KB + 64 KB WGP$ | ISA §3.3.4, §10.7–10.8, §10.11, §11; LLVM `FLATInstructions.td`, `GCNFeatures.td` | P1 §3 |
| `counters` | LOADcnt, STOREcnt, DScnt, ASYNCcnt, TENSORcnt, XCNT (6-bit, max 63); KMcnt (5-bit, max 31).<br>`WaitForm::PerCounter`: `s_wait_loadcnt`, `_storecnt`, `_dscnt`, `_kmcnt`, `_asynccnt`, `_tensorcnt`, `_xcnt`, plus `s_wait_loadcnt_dscnt` / `s_wait_storecnt_dscnt` and `s_wait_idle`.<br>`s_waitcnt`, `s_wait_expcnt`, `_samplecnt`, `_bvhcnt` and `s_wait_event` are rejected | ISA §4 (wait instructions), §15; LLVM `SIInsertWaitcnts.cpp`; MC `gfx1250_asm_unsupported.s` | P2 §1–3 |
| order / wait model | Order is per class. KMcnt and the SMEM side of XCNT are out of order; async loads vs async stores are out of order on ASYNCcnt.<br>LLVM waits a counter to N > 0 only while a single event type is pending (otherwise 0).<br>LLVM never inserts ASYNC / TENSOR waits automatically | LLVM `SIInsertWaitcnts.cpp:1568-1605`, `:224-229`; ISA | P2 §2, §5 |
| ALU waits | `s_wait_alu` (depctr fields) and `s_delay_alu`. The ISA does not describe the `s_wait_alu` fields; LLVM defines them. Proposed as `Count::Instr` rows | LLVM `SIDefines.h`, `AMDGPUBaseInfo.cpp`; MC | P2 §4 |
| memory model | Scopes `SCOPE_CU` / `SE` / `DEV` / `SYS` (0–3); cluster maps to `SCOPE_SE`. Acquire / release sequences per AMDGPUUsage GFX12.5 (`global_inv` / `global_wb` on LOAD / STORE counters) | LLVM `AMDGPUUsage.md`, `SIMemoryLegalizer.cpp`; ISA §10 | P2 §6 |
| `hazards` | gfx1250 sets `FeatureNoDataDepHazard`, so the classic `s_nop` wait-state checks are off.<br>Active rows: WMMA→WMMA and WMMA→VALU co-execution (count per instruction group, unit = independent VALU / `V_NOP`), TRANS co-execution, `V_CMPX`→`V_PERMLANE`, `V_PERM_PK16`, getreg wait-idle, scratch-base forwarding, setreg MODE, DS async-barrier-arrive, the TDM drain, and ISA-only rules.<br>Conflict: for 16x16x128 FP8 / F8F6F4 / 32x16x128 F4 the ISA gives 4 / 3, LLVM 5 / 4 | ISA §7.12 / co-execution sections; LLVM `GCNHazardRecognizer.cpp` | P3 §0–1 |
| `latency` | LLVM `GFX1250SpeedModel`: VALU 5, TRANS 8, F64 37, WMMA 4 / 8 / 16 by shape, plus a blocking-cycle table. The ISA gives almost no cycle counts | LLVM `SISchedule.td`; ISA | P3 §2 |
| `atoms` | Every gfx1250 WMMA / SWMMAC: shapes, A/B/C/D formats, register counts, sparse index, scale / scale16 (VOP3PX2), `v_wmma_ld_scale*`, modifier encodings.<br>**A/B/C/D lane layouts restated from ISA pp. 99–107**; two internal ISA inconsistencies are flagged.<br>`gfx1250-strict` lacks 9 of them (16x16x128 FP8, 32x16x128 F4) | ISA §7.12 (pp. 95–107); LLVM `VOP3PInstructions.td`; MC `gfx1250_asm_wmma_w32.s` | P4 §1–4 |
| `mem` (proposed) | Buffer V#: 57-bit base, 45-bit `num_records` (bytes), 14-bit stride. Global / flat / scratch forms and offsets.<br>Async LDS (ASYNCcnt), cluster multicast (M0 mask).<br>TDM: descriptor groups 0–3 in SGPRs, completion on TENSORcnt.<br>Cache policy: SCOPE / TH / NV / SCALE_OFFSET codes | ISA §9–10; LLVM `AMDGPUUsage.md`, `SIDefines.h` | P4 §5–8 |
| `emit` / `llvm` | `-mcpu=gfx1250` (subarch `amdgpu12.50`). ELF mach 0x049 (`gfx1250-strict` 0x0eb). Code-object V4 / V5 (default) / V6.<br>Descriptor: VGPRs in blocks of 16; SGPR field 0; LDS in 2048 B units; `.amdhsa_named_barrier_count`.<br>Product names "TBA" | LLVM `AMDGPUUsage.md`, `GCNProcessors.td`, `GCNFeatures.td` | P5 §1–2 |
| `verify` | VGPR 1024 (256 direct), SGPR 106, LDS 327 680 B, 16 named barriers, 32 user SGPRs, buffer base / `num_records` widths | ISA; LLVM | P5 §3 |
| `sync` (prose) | Split barriers (`s_barrier_signal` / `s_barrier_wait`). Barrier IDs −4 … 16: cluster trap / user, work-group trap / work-group, named 1–16.<br>Work-group clusters of up to 16 WGs, IDs in TTMP6 / 7 / 9.<br>The assembler rejects `s_barrier_signal_isfirst -3` (`AsmParser/AMDGPUAsmParser.cpp:5585`) | ISA §2.3, §5 (barriers); LLVM | P2 §7–8 |

## 2. gfx950 ↔ gfx1250 differences and the L4 passes (own analysis)

How each L4 component of today's backend (`crates/amdgpu-compiler`) relates to a gfx1250 target. The gfx1250 column states only facts
cited in §3–§7. Classes:
- **generic**: algorithm and numbers unchanged;
- **re-parameterized**: same algorithm, numbers or tables from `Target`;
- **new**: needs a new pass or a substantially new algorithm;
- **n/a**: does not apply.

| Area | gfx950 (today) | gfx1250 | L4 component | Class |
|---|---|---|---|---|
| Wave | 64 lanes; exec = 2 SGPRs | 32 lanes; exec = 1 SGPR | lane / exec lowering, `gpu_verify` lane model | re-parameterized (`wave`) |
| VGPR file | 512 unified (arch ≤ 256 + AGPR), granule 8 | 1024 per wave in blocks of 16; above v255 through the MODE MSBs (`s_set_vgpr_msb`) | `regalloc.rs` | re-parameterized. The MSB windowing is **new**: insert and minimise `s_set_vgpr_msb` and keep the v0–v255-only operands low. LLVM does this in `AMDGPULowerVGPREncoding.cpp` |
| AGPR | 256; MFMA C/D may live there (`agpr.rs`) | none | `agpr.rs`, `pin_agpr` | n/a |
| SGPR | 102 usable, allocated (rounded to 8) | 106, fixed (descriptor field 0) | `regalloc.rs`, `emit.rs` | re-parameterized |
| Occupancy | step table from VGPR / AGPR / SGPR | `BlockPool` (VGPR only), 16 waves / SIMD cap, LDS | `regplan.rs`, `sched/model.rs`, `sched/sim.rs` | re-parameterized |
| LDS | 160 KiB, 64 banks; DMA `buffer_load … lds` with M0 base + imm | 320 KiB per WG, 64 banks; async DMA with a per-lane VGPR LDS address; TDM | `gpu_protocol.rs`, `sched/m0.rs` | rings: re-parameterized; M0 reuse: n/a. The async / TDM completion protocol is **new**: ASYNCcnt / TENSORcnt, and async writes to LDS may land out of order (ISA §10.8) |
| Counters / waits | combined `s_waitcnt vmcnt / lgkmcnt / expcnt` | 7 split counters, `PerCounter` waits, XCNT, KMcnt wait 0 only | `sched/waits.rs`, `pin_check.rs`, `gpu_verify` wait axis | re-parameterized (`counters`). **New:** per-class order on a shared counter, and the XCNT rules (P2 §5) |
| ALU dependency waits | none (interlock + `s_nop`) | `s_wait_alu` / `s_delay_alu`; rows that need an instruction | `hazards.rs` | **new** row kind (`Count::Instr`). An `s_delay_alu` hint pass is optional |
| Matrix hazards | MFMA pass-based wait states (`s_nop`) | WMMA / SWMMAC co-execution padding counted in independent VALU / `V_NOP` | `hazards.rs`, `hzfill.rs` | re-parameterized rows with a **new** unit (`IndependentValu`). `hzfill` maps naturally |
| VALU / TRANS distances | VALU→SGPR 5 / 2, trans 5, DPP 2 (software) | `FeatureNoDataDepHazard`: no classic checks; TRANS co-execution rule | `hazards.rs`, `sched/model.rs` | re-parameterized |
| VOPD | none | `v_dual_*` (VOPD, VOPD3) dual issue | — | **new** (pairing pass, optional). Prose in P3 §4 |
| Matrix atoms | MFMA 32x32 / 16x16 | WMMA / SWMMAC 16x16xK (32x16 for F4), layouts from ISA pp. 99–107 | `amdgpu-isa`, `atoms` | re-parameterized (`atoms`) |
| Scaled MMA | `v_mfma_scale_*` | `v_wmma_scale*` / `scale16`, `v_wmma_ld_scale*` | `atoms`, `hazards` | re-parameterized |
| Memory ops | buffer / global with `sc0 sc1 nt`; `buffer_load … lds` | `th:` + `scope:` + `nv` + `scale_offset`; async LDS; cluster loads; TDM; block loads | memory lowering, `emit.rs` | cache bits: re-parameterized. Async / cluster / TDM lowering: **new** |
| Barriers / sync | `s_barrier` | split signal / wait, named barriers 1–16, cluster barriers, clusters | `sched/order.rs`, `gpu_protocol.rs`, lowering | re-parameterized (barrier = signal + wait). Named / cluster barriers: **new** |
| Scheduler model | MFMA pipe, `s_setprio` ping-pong | WMMA pipe and co-execution, VOPD | `sched/model.rs`, `list.rs`, `slotfill.rs`, `phase.rs` | re-parameterized (`latency`). The co-execution model is **new** |
| Weave regions, liveness, depgraph, CFG cleanup | — | — | `sched/region.rs`, `liveness.rs`, `depgraph.rs`, `cfg.rs` | generic |
| LDS-read placement (`ldsink`, `ldahead32`, `ldshoist`) | MFMA-relative | WMMA-relative | `sched/ldahead.rs`, `hoist.rs` | re-parameterized |
| Emission | gfx950 mnemonics, `.amdhsa_*`, accum_offset, SGPR rounding | VGPR blocks of 16, SGPR field 0, LDS 2048 B units, named-barrier count, no accum_offset | `emit.rs` | re-parameterized + new per-instruction tables |
| LLVM tail | `llvm20` / `llvm24` MIR tables, `-mcpu=gfx950` | `-mcpu=gfx1250` | `backend/` | re-parameterized (new table set) |
| Verifier | `gpu_verify` limits and protocol | new limits, split counters, async / TDM completion | `gpu_verify.rs`, `gpu_protocol.rs` | limits: re-parameterized. Async / TDM axes: **new** |

**The five extensions the P0 types need** (for backend-llvm-tail; each with its public justification):

| # | Extension | Public justification |
|---|---|---|
| 1 | `RegFile` gains a direct-encoding window (256) and a list of v0–v255-only operands | ISA §3.3.2.3 (pp. 17–18), §7.12.6 (p. 106); LLVM `VOP3PInstructions.td:1894-1895`, `MIMGInstructions.td:566`, `AMDGPULowerVGPREncoding.cpp` (P1 §2.1) |
| 2 | `Counters`: order is per class. A counter with more than one event type pending is out of order and can only be waited to 0 | LLVM `SIInsertWaitcnts.cpp:1568-1605` (P2 §5); ISA (P2 §2) |
| 3 | A description of the `s_wait_alu` fields for `Count::Instr` rows | LLVM `SIDefines.h` / `AMDGPUBaseInfo.cpp` and MC `gfx12_asm_sopp.s` (P2 §4). The ISA does not describe the fields |
| 4 | `Lds.dma` with a per-lane VGPR LDS address (no M0 rule) | ISA §10.8 (pp. 135–136); LLVM `FLATInstructions.td:391-418` (P1 §3) |
| 5 | `mem` / `sync` extensions | ISA §2.3, §5, §10 (P2 §7–8, P4 §5–8) |

Plus `HazardRule.kind: Software | Interlock`. Public support:
- the TRANS co-execution and WMMA rules are software-enforced (`GCNHazardRecognizer.cpp`);
- the ISA's co-execution pairing rules distinguish cases the hardware stalls from cases software must pad (P3 §3).

## 3. gfx1250 target description — P1: `wave`, `regs`, `lds`, topology

Scope: the `Wave`, `RegFiles` (`vgpr`, `agpr`, `sgpr`, `unified`, `occupancy`) and `Lds` fields of `Target`
(names from `target-design.md` §2), plus topology as prose only. Public sources only:

- **ISA** = AMD "CDNA5" Instruction Set Architecture Reference Guide (27-July-2026). Cited as *CDNA5 ISA §x (p. N)*, where N is the printed page ("N of 823").
- **LLVM** = upstream llvm-project @274ab686 (2026-11-09). Cited as *LLVM path:line @274ab686*.
  Most paths are in the local sparse checkout (`llvm/lib/Target/AMDGPU`, `llvm/docs/AMDGPUUsage.md`, `llvm/test/MC/AMDGPU`).
  Two files are outside the sparse paths: `llvm/lib/TargetParser/AMDGPUTargetParser.cpp` and `llvm/include/llvm/TargetParser/AMDGPUTargetParser.h`.
  They were read from the public GitHub web view of the same commit (`raw.githubusercontent.com/llvm/llvm-project/274ab686…/`) and are marked **[TP]**.
  They define the per-GPU numbers (total VGPRs, max waves, addressable SGPRs) that `AMDGPUBaseInfo.cpp` calls.

"derived" marks my own inference from the cited facts. "unknown (no public source)" means neither source states the fact.

---

### 1. `wave: Wave`

```rust
pub const WAVE: Wave = Wave {
    lanes: 32,
    exec_bits: 1,          // one SGPR (exec_lo) is the lane mask
    // wave64: not supported on gfx1250 (keep as a doc comment / assert, no field)
};
```

| fact | value | source | notes |
|---|---|---|---|
| Wave size | 32 work-items | CDNA5 ISA §1.1 (p. 3, "Wave" term); Ch. 2 intro (p. 9) | The ISA says the device only runs wave32. Wording about wave64 elsewhere in the ISA does not apply to this device (Ch. 1 intro, p. 3). |
| wave64 support | none | CDNA5 ISA Ch. 1 intro (p. 3); LLVM `llvm/lib/Target/AMDGPU/GCNFeatures.td:1881-1908` (FeatureGFX12 lists `FeatureSupportsWave32` only) and `:2494-2586` (12_50 common set adds `FeatureWavefrontSize32` at :2507 and no `FeatureSupportsWave64`) @274ab686 | gfx1200 gets `FeatureSupportsWave64` from `FeatureISAVersion12` (`GCNFeatures.td:2423-2426`); gfx1250 does not use that set. `AMDGPURemoveIncompatibleFunctions.cpp:205-209` removes wave64 functions when the GPU lacks the feature. The ISA still has a STATUS.WAVE64 bit (§3.4.1, p. 21) and wave64 branches in the pseudocode; treat them as legacy. |
| LLVM `-mcpu=gfx1250` target features | `xnack` only (no `wavefrontsize64`, no `cumode` option listed) | LLVM `llvm/docs/AMDGPUUsage.md:1189-1204` @274ab686 | gfx1200 lists `cumode` and `wavefrontsize64` (`AMDGPUUsage.md:1165-1166`). |
| EXEC width | 64-bit register, but wave32 uses only bits [31:0]; hardware ignores the upper half | CDNA5 ISA §3.1 Table 4 (p. 12); §3.2.2 (p. 14) | EXECZ looks only at EXEC[31:0] (p. 14). |
| Lane-mask SGPR tuple (`exec_bits`) | 1 SGPR (`exec_lo`); VCC is `vcc_lo` | CDNA5 ISA §3.4.7 (p. 24): wave32 compares write only the low 32 bits of VCC, EXEC or one SGPR. LLVM `SIRegisterInfo.cpp:4498`, `SIRegisterInfo.h:407-408` (wave-mask class is a 32-bit SGPR class when wave32), `AMDGPULaneMaskUtils.h:48` @274ab686 | ISA and LLVM agree. |
| VCCZ | summary of VCC[31:0] only | CDNA5 ISA §3.4.7 (p. 24) | |
| Assembler wave-size requirement | async/TDM/load-tr instructions require wavesize 32 | LLVM `FLATInstructions.td:1241,1252` (`WaveSizePredicate = isWave32`); `test/MC/AMDGPU/gfx1250_asm_load_tr.s:36` ("requires wavesize=32") @274ab686 | |

---

### 2. `regs: RegFiles`

```rust
pub const REGS: RegFiles = RegFiles {
    vgpr: RegFile {
        count: 1024,                       // per wave, logical v0..v1023
        alloc_granule: 16,                 // VGPRs per allocation block (wave32)
        tuple_align: &[(1,1),(2,2),(3,2),(4,2),(5,2),(6,2),(7,2),(8,2),
                       (9,2),(10,2),(11,2),(12,2),(16,2),(32,2)],
        reserved: &[],                     // no VGPR reserved by hardware
    },
    agpr: None,                            // no AccVGPR file on gfx1250
    sgpr: RegFile {
        count: 106,                        // s0..s105 usable
        alloc_granule: 0,                  // fixed allocation, not an occupancy term
        tuple_align: &[(1,1),(2,2),(3,4),(4,4),(8,4),(16,4)],
        reserved: &[],                     // specials (vcc, m0, exec, null, ttmp) sit outside 0..105
    },
    unified: None,                         // no arch-VGPR + AGPR unified file
    occupancy: Occupancy::BlockPool { block_vgprs: 16, blocks: 64 },  // + max 16 waves/SIMD
    // proposed extension (not in P0 types): high-VGPR window
    // vgpr_msb: Some(VgprMsb { direct_window: 256, setter: "s_set_vgpr_msb",
    //                          entry_state: 0, low_only: &[/* see table 2.1 */] }),
};
```

#### 2.1 VGPRs

| fact | value | source | notes |
|---|---|---|---|
| Max VGPRs per wave | 1024 (v0..v1023) | CDNA5 ISA §3.1 Table 4 (p. 12); §3.3.2.1 (p. 16); §3.3.2.3 (p. 17). LLVM `Utils/AMDGPUBaseInfo.cpp:1336-1341` (`FeatureVGPRMSBs` + wave32 gives 1024); `SIRegisterInfo.td:667-668` (VGPR_32 = VGPR0..1023) @274ab686 | Agree. [TP] `AMDGPUTargetParser.cpp:456-469`: addressable = min(2^(8+2), total). |
| Allocation granule | blocks of 16 VGPRs (16 × 32 lanes = 512 dwords) | CDNA5 ISA §3.3.2.1 (p. 16). LLVM `Utils/AMDGPUBaseInfo.cpp:1313-1314` (GFX10.3+ wave32 gives 16; gfx1250 inherits `FeatureGFX10_3Insts` via FeatureGFX12, `GCNFeatures.td:1888`); [TP] `AMDGPUTargetParser.cpp:425-434` @274ab686 | Agree. |
| Descriptor VGPR count encoding | `GRANULATED_WORKITEM_VGPR_COUNT = max(0, ceil(vgprs_used/16) - 1)`, `max_vgpr` 1..1024 | LLVM `docs/AMDGPUUsage.md:7474-7476`; `Utils/AMDGPUBaseInfo.cpp:1328-1329` (encoding granule 16 when `FeatureVGPRMSBs`) @274ab686 | The ISA does not describe the kernel descriptor (ISA silent). |
| Minimum allocation | a wave cannot be created with 0 VGPRs | CDNA5 ISA §3.3.2.1 (p. 16) | derived: the smallest allocation is 1 block (16 VGPRs), matching LLVM `max(1u, NumRegs)` (`AMDGPUBaseInfo.cpp:1286-1289`). |
| Tuple alignment | any VGPR operand ≥ 64 bits must start on an even VGPR (VALU and memory) | CDNA5 ISA §3.3.2.1 (p. 16); §7.2 (p. 69). LLVM `GCNFeatures.td:483-487` (`vgpr-align2`) and `:2498` (enabled for 12_50); `AMDGPU.td:587-588` (HwMode "alignment requirement but no AGPRs") @274ab686 | Agree. derived: `tuple_align = 2` for every width ≥ 2; no 4-alignment rule for VGPRs in either source. WMMA scale pairs ("2 VGPRs") must also be even (§7.12.6, p. 106). |
| Out-of-range VGPR | dest out of range turns the instruction into a NOP (VMEM is issued with EXEC=0 so counters stay correct); source out of range reads v0 (VOPD: addr % 4) | CDNA5 ISA §3.3.2.2 (pp. 16-17) | Useful for `VerifyLimits`: index + width must stay ≤ allocated size. |
| Direct-encoding window | instruction fields hold 8 bits, so v0..v255 are reachable without help | CDNA5 ISA §3.3.2.3 (p. 17) | |
| High-VGPR mechanism | 8 MODE bits (2 per operand slot: DST, SRC0, SRC1, SRC2) supply address bits [9:8]. They are set by `S_SET_VGPR_MSB imm` (SOPP; imm[1:0]=src0, [3:2]=src1, [5:4]=src2, [7:6]=dst) or by `S_SETREG MODE`, and stay in effect until changed | CDNA5 ISA §3.3.2.3 (pp. 17-18); §3.4.3 Table 7 (p. 22: MODE[13:12]=dst, [15:14]=src0, [17:16]=src1, [19:18]=src2); §15.5 S_SET_VGPR_MSB (pp. 281-282). LLVM `SOPInstructions.td:1946-1951` (`HasVGPRMSBs`), `:3061` (SOPP opcode 0x006); `test/MC/AMDGPU/gfx1250_asm_sopp.s:41-51` (encodings `…,0x86,0xbf`) @274ab686 | Agree on the bit layout. LLVM `AMDGPULowerVGPREncoding.cpp:474-481` notes that MODE stores the same 4 fields rotated left by 2. |
| `S_SET_VGPR_MSB` imm[15:8] | ISA: the operand is described only through imm[7:0]. LLVM stores the previous mode in imm[15:8] for bookkeeping and the assembler accepts 0xffff | CDNA5 ISA §3.3.2.3 (p. 18). LLVM `AMDGPULowerVGPREncoding.cpp:230-231,262`; `gfx1250_asm_sopp.s:49-51` @274ab686 | ISA silent on [15:8]. Emit 0 in the high byte unless matching LLVM output byte-for-byte. |
| A tuple may cross v255→v256 | yes; out-of-range checks use the 10-bit logical address | CDNA5 ISA §3.3.2.3 (p. 18) | |
| MSB mapping per encoding | VOP1/2/C/3/3P: DST, SRC0..2 map to vdst/src0..2. VOPD: X and Y operands. VDS: vdst, addr, data0, data1. FLAT/GLOBAL/SCRATCH (incl. async LDS): vdst, addr, vsrc. VBUFFER: vdata, vaddr. VIMAGE: vdata, vaddr0..2 | CDNA5 ISA §3.3.2.3 table (p. 18); §15.5 (p. 282) | Special cases: FMAAK uses the SRC0/SRC1 MSBs, FMAMK uses SRC0/SRC2, and accumulate ops (e.g. V_FMAC) use the DST MSBs for both vdst and src2 (p. 18). |
| Operands restricted to v0..v255 (MSBs ignored) | VIMAGE VADDR3-4 (column "Not offset"); WMMA scale SRC0/SRC1 fields; `V_WMMA_LD_SCALE*` (treats MSBs as 0) | CDNA5 ISA §3.3.2.3 (p. 18); §7.12.6 (p. 106). LLVM `VOP3PInstructions.td:1894-1895,2322-2323` (`VCSrc_b32/b64_Lo256`); `MIMGInstructions.td:566` (`Addr3RC = VGPROp_32_Lo256`); `SIRegisterInfo.td:684-686,1151-1152,1182-1183,1211` (Lo256 classes, `RsrcReg32`) @274ab686 | Agree. The VBUFFER "RSRC" entry in that ISA column is an SGPR operand. pdftotext layout of p. 18 is ambiguous for the VBUFFER/VIMAGE rows (see Open). |
| 16-bit VGPR halves reachable | with MSBs: v0..v127.[l/h] and v256..v383.[l/h] | CDNA5 ISA §3.3.2.3 (p. 18) | LLVM `SIRegisterInfo.td:644-656` (VGPR_16_Lo128 for VOP1/2/C true16). |
| Compiler convention for MSBs | all 4 MSB fields are 0 at function entry and at the start of every non-fall-through block. The pass runs late and inserts `s_set_vgpr_msb` before VGPR users | LLVM `AMDGPULowerVGPREncoding.cpp:10-39` @274ab686 | ISA silent (this is an ABI choice). derived: adopt the same convention for interoperability. |
| MSB setter hazards | `S_SETREG MODE` directly followed by `S_SET_VGPR_MSB` can give a wrong result, so put one instruction (e.g. `s_nop`) between them. LLVM also models "S_SETREG to MODE clobbers MSB bits" | CDNA5 ISA §5.7.2 (p. 56). LLVM `AMDGPULowerVGPREncoding.cpp:200-213,254-259`; `GCNFeatures.td:1396-1399,2635` (`FeatureSetregVGPRMSBFixup` on gfx1250) @274ab686 | The ISA states only the adjacency hazard. The clobber fix-up is LLVM-only (ISA silent). Belongs in `hazards` (P-hazard section). |
| Indexed VGPR access | `V_MOVREL{D,S,SD}_B32` index from M0[31:0]; `V_MOVRELSD_2_B32` / `V_SWAPREL_B32` use M0[9:0] (src) and M0[25:16] (dst). The MSBs also apply to MOVREL | CDNA5 ISA §7.11 (p. 94); §3.3.2.3 (p. 17) | |
| Block VGPR load/store | `GLOBAL/SCRATCH_{LOAD,STORE}_BLOCK`: up to 32 consecutive VGPRs; an M0 bitmask selects which ones; counts once in LOADcnt/STOREcnt | CDNA5 ISA §10.6 (pp. 133-134) | Spill / CWSR helper, listed here because it is a VGPR-file mechanism. |
| Early VGPR release | `S_SENDMSG DEALLOC_VGPRS` (0x03) frees all VGPRs and scratch before `S_ENDPGM`. Wait for outstanding scratch stores first | CDNA5 ISA §3.3.2.1 (p. 16); §5.4 Table 18 (p. 44) | Returns blocks to the pool early. |
| Dynamic VGPR mode | LLVM: attribute `amdgpu-dynamic-vgpr-block-size` (16 or 32), RSRC3 `ENABLE_DYNAMIC_VGPR` for GFX125\*, at most 8 blocks, and `s_alloc_vgpr` assembles for gfx1250. ISA: no description, and `S_ALLOC_VGPR` is not in the ISA instruction list | LLVM `docs/AMDGPUUsage.md:3474-3480,8006-8013`; `Utils/AMDGPUBaseInfo.cpp:2377-2385,1303-1304,1349-1352`; `Utils/AMDGPUBaseInfo.h:253`; `SOPInstructions.td:438-444,2372`; `test/MC/AMDGPU/gfx12_asm_sop1.s:3,6-19` (RUN for amdgpu12.50) @274ab686 | **Conflict / ISA silent.** Recommendation: the Target models static allocation only. |

#### 2.2 `agpr: Option<RegFile>` and `unified`

| fact | value | source | notes |
|---|---|---|---|
| AccVGPR file | absent (`agpr: None`) | LLVM `AMDGPU.td:587-588` ("alignment requirement but no AGPRs"); `GCNFeatures.td:2494-2661` (no `FeatureAGPRAlloc` / MAI for 12_50); `SIRegisterInfo.cpp:711-723` (AGPRs all reserved when there are no MAI instructions) @274ab686 | The ISA register list has only SGPR and VGPR (§3.1 Table 4, p. 12). The ISA never mentions AGPRs, which is consistent. |
| Unified arch-VGPR/AGPR file | none (`unified: None`) | derived from the row above. The [TP] `AMDGPUTargetParser.cpp:441-450,456-461` 512-unified path is gfx90a-only | No `accum_offset` in the gfx1250 descriptor (AMDGPUUsage `.amdhsa_accum_offset` is GFX90A/GFX942 only, `docs/AMDGPUUsage.md:17851-17855`). |

#### 2.3 SGPRs and special scalar registers

| fact | value | source | notes |
|---|---|---|---|
| Usable SGPRs | 106 (s0..s105), fixed per wave | CDNA5 ISA §3.1 Table 4 (p. 12); §3.3.1.1 (p. 14). [TP] `AMDGPUTargetParser.cpp:383-393` (addressable 106 for major ≥ 10); LLVM `SIRegisterInfo.td:415-448` (SGPR tuples stop at index 105) @274ab686 | Agree. |
| VCC | `vcc_lo`/`vcc_hi` = SGPR 106/107; normal dependency checks apply | CDNA5 ISA §3.3.1.1 (p. 14); §3.3.1.1.1 (p. 15). LLVM `SIRegisterInfo.td:181` (vcc_hi enc 107) @274ab686 | LLVM `getMaxNumSGPRs` without the addressable clamp = 108 (`Utils/AMDGPUBaseInfo.cpp:1245-1246`). |
| Trap temporaries | TTMP0..15 = scalar encodings 108..123. The user shader can read them; writes outside the trap handler are ignored | CDNA5 ISA §3.3.1.3 (p. 15); §3.1 Table 4 (p. 13) | LLVM reserves all TTMPs for allocation (`SIRegisterInfo.cpp:620-627`) but reads TTMP6/7/9 as architected IDs (§2.4 below). |
| Other scalar encodings | 124 = NULL, 125 = M0, 126/127 = EXEC_LO/HI | CDNA5 ISA §3.3.1.3 (p. 15). LLVM `SIRegisterInfo.td:206-207,227-232` (gfx11+: m0 = 125, null = 124) @274ab686 | Agree. |
| NULL | reads 0. A write is discarded, but SALU still updates SCC. Not allowed as a descriptor | CDNA5 ISA §3.4.5 (p. 23) | |
| M0 | 32-bit, written only by SALU. Uses: MOVREL index, DS ADDTID offset, multicast WG mask, barrier ID / member count, sendmsg data, block-load mask, temporary | CDNA5 ISA §3.4.4 Table 8 (p. 23); §10.6 (p. 134); §10.7 (p. 134) | |
| Alignment | 64-bit operands start on an even SGPR (including PC moves and SMEM base pairs). >64-bit operands, and SMEM data of >2 dwords, need quad alignment. Misalignment is illegal (unpredictable results) | CDNA5 ISA §3.3.1.2 (p. 15). LLVM `SIRegisterInfo.td:415` (64-bit tuples step 2), `:418-448` (96/128/256/512-bit tuples step 4); `test/MC/AMDGPU/gfx1250_asm_vimage_err.s:52-56` and `gfx1250_asm_load_tr.s:34-35` ("invalid register alignment") @274ab686 | Agree. A 32-bit SGPR used as a VALU source may sit at any index (p. 15). |
| Region crossing | an indexed or multi-dword operand must stay inside one region: [s0..s105 + VCC] or [TTMP0..15]. Out-of-range reads return 0 and writes are dropped | CDNA5 ISA §3.3.1.3 (pp. 15-16) | Verifier rule. |
| Descriptor SGPR count | `GRANULATED_WAVEFRONT_SGPR_COUNT` is reserved/0 on GFX10-12 because 128 SGPRs are always allocated | LLVM `docs/AMDGPUUsage.md:7496-7497` @274ab686 | ISA: every wave gets 106 + 16 TTMP (p. 12). Not an occupancy term: LLVM `Utils/AMDGPUBaseInfo.cpp:1255-1259` (`isSGPROccupancyLimited` = false for major ≥ 10). |
| LLVM-reserved SGPRs | EXEC, M0, NULL, TTMPs, src_* aperture regs, and anything above `getMaxNumSGPRs` except VCC; plus per-function SP/FP/scratch-rsrc/base-pointer/exec-copy when used | LLVM `SIRegisterInfo.cpp:575-688` @274ab686 | derived: for a leaf kernel with no stack, `reserved = &[]` inside s0..s105. |

#### 2.4 User / system SGPR layout at wave launch (compute)

```rust
// proposed shape inside RegFiles / Emission (field names illustrative)
sgpr_abi: SgprAbi {
    max_user_sgprs: 32,
    user_order: &[PrivSegBuf(4) /*not on gfx1250*/, DispatchPtr(2), QueuePtr(2),
                  KernargPtr(2), DispatchId(2), PrivSegSize(1), PreloadKernargs(n)],
    tg_info_after_user: true,   // optional {ordered_append_term, wg_size_in_waves}
    workgroup_id: Architected { x: TTMP9, y: TTMP7_lo16, z: TTMP7_hi16 },
    cluster_info: TTMP6,        // nibbles wg_x/y/z, nwg_x/y/z, flat_nwg
    vgpr0: PackedWorkItemId { x: 0..9, y: 10..19, z: 20..29 },
}
```

| fact | value | source | notes |
|---|---|---|---|
| User SGPRs | up to 32, loaded into s0.. in order. Only enabled values are loaded, packed densely with no alignment padding | CDNA5 ISA §3.5.3 (p. 29); §3.5.3.1 Table 9 (p. 30). LLVM `Utils/AMDGPUBaseInfo.cpp:2420-2424` (32 on GFX1250+); `docs/AMDGPUUsage.md:7733-7740` (USER_SGPR_COUNT is 6 bits [6:1] on GFX125\*) @274ab686 | **Doc conflict inside LLVM**: generic text `AMDGPUUsage.md:8173-8178` still says "up to 16". Code and the rsrc2 table say 32. |
| Order of user SGPRs | dispatch ptr, queue ptr, kernarg segment ptr, dispatch id, (flat-scratch-init not on GFX11+), private segment size, preloaded kernargs | LLVM `docs/AMDGPUUsage.md:8183-8232`; `:17772-17799` (which directives apply to GFX12) @274ab686 | ISA lists only "user data registers" generically (p. 30). derived: with only the kernarg pointer enabled it is s[0:1]. Because the ISA adds no alignment padding (p. 29), an odd number of preceding user SGPRs would put a 64-bit pointer on an odd SGPR (illegal per §3.3.1.2). Keep 64-bit user values on even offsets. |
| Kernarg preload | supported (`FeatureKernargPreload`), data goes into user SGPRs after the last enabled non-preload user SGPR | LLVM `GCNFeatures.td:2561`; `docs/AMDGPUUsage.md:8359-8387` @274ab686 | ISA silent. |
| Word after user SGPRs | optional `{ordered_append_term[11:0], work_group_size_in_waves[5:0]}`, enabled by `COMPUTE_PGM_RSRC2.tg_size_en` | CDNA5 ISA §3.5.3.1 Table 9 (p. 30) | LLVM generic table "Work-Group Info" has a slightly different bit list (`AMDGPUUsage.md:8245-8248`, ordered_append_term[10:0] and a first_wavefront bit). Minor doc mismatch. |
| Work-group IDs | architected, not in user SGPRs: X = TTMP9[31:0]; Y = TTMP7[15:0]; Z = TTMP7[31:16] (TTMP7 loaded only if Y or Z is used; Z = 0 if only Y is used) | CDNA5 ISA §3.5.3.1 Table 9 (p. 30). LLVM `SIISelLowering.cpp:2834-2843,2863` (`hasArchitectedSGPRs`); `GCNFeatures.td:1563-1566,2510` @274ab686 | Agree. With clusters, TTMP9/TTMP7 hold the cluster ID within the grid (p. 30). |
| TTMP8 | {DebugMark, GridYZvalid, waveIDinGroup[4:0], dispatchIndex[24:0]} | CDNA5 ISA §3.5.3.1 Table 9 (p. 30) | derived: wave-in-group ID is available without an extra SGPR. |
| Cluster position (TTMP6) | 4-bit fields: wg_x/y/z [11:0], nwg_x/y/z −1 [23:12], flat_NWG −1 [27:24]. Not initialized when no cluster is used | CDNA5 ISA §2.3.1 (p. 11); §3.5.3.1 (p. 30). LLVM `SIISelLowering.cpp:2844-2857` (same nibble masks) @274ab686 | Agree. |
| Scratch | SCRATCH_BASE is an architected register (0 if no scratch); flat scratch is architected | CDNA5 ISA §3.5.2 (p. 29). LLVM `GCNFeatures.td:1558-1561,2509`; `docs/AMDGPUUsage.md:1194` @274ab686 | No private-segment-buffer user SGPRs on gfx1250 (derived from `AMDGPUUsage.md:17772-17775`, GFX6-GFX10 only). |
| VGPR0 at launch | packed work-item ID: X [9:0], Y [19:10], Z [29:20] | CDNA5 ISA §3.5.4 (p. 31). LLVM `docs/AMDGPUUsage.md:8298-8329`, `:1195` ("Packed work-item IDs") @274ab686 | Agree. |

#### 2.5 `occupancy: Occupancy`

```rust
occupancy: Occupancy::BlockPool { block_vgprs: 16, blocks: 64 },
// waves_per_simd(v) = min(16, 64 / ceil(v / 16))   (v >= 1)
// SGPRs, LDS and barriers are separate limits (SGPRs never limit)
```

| fact | value | source | notes |
|---|---|---|---|
| Max waves per SIMD (EU) | 16 | LLVM `GCNFeatures.td:1907` (`FeatureMaxWavesPerEU16` in FeatureGFX12); [TP] `AMDGPUTargetParser.cpp:584-587` @274ab686 | **ISA silent** (no waves-per-SIMD figure). |
| Physical VGPRs per SIMD (wave32) | 1024 | [TP] `AMDGPUTargetParser.cpp:441-450` (GFX10+ without `1536_PHYSICAL_VGPRS` gives 1024 for wave32); LLVM `GCNFeatures.td:2471` (1536 only in the gfx1200 set, not 12_50) @274ab686 | **ISA silent** on the per-SIMD file size. derived: 1024 / 16 = **64 blocks**, matching target-design.md §2 `BlockPool`. |
| Occupancy formula | waves = min(MaxWaves, Total / alignTo(NumVGPRs, granule)), ≥ 1 | LLVM `Utils/AMDGPUBaseInfo.cpp:1357-1374` @274ab686 | derived table: ≤64 VGPRs → 16; 80 → 12; 96 → 10; 112 → 9; 128 → 8; 144 → 7; 160 → 6; 176-192 → 5; 208-256 → 4; 272-336 → 3; 352-512 → 2; 528-1024 → 1. |
| SGPR effect | none | LLVM `Utils/AMDGPUBaseInfo.cpp:1255-1259,1386-1396` @274ab686; CDNA5 ISA §3.3.1.1 (p. 14, fixed allocation) | |
| Pool sharing | blocks are freed at wave end or by DEALLOC_VGPRS | CDNA5 ISA §3.3.2.1 (p. 16) | Whether the pool is per SIMD (LLVM model) or shared more widely is unknown (no public source). See Open. |
| LDS effect on occupancy | LLVM uses `getLocalMemorySize` = 320 KiB (full-SIMD, not halved) as the occupancy capacity, with 2048-byte allocation granule | LLVM `Utils/AMDGPUBaseInfo.cpp:1143-1157`; [TP] `AMDGPUTargetParser.h:200-221` @274ab686 | derived: WGs per WGP by LDS = floor(327680 / roundup(lds, 2048)). |
| Work-groups per WGP | ISA: up to 32 (single-wave WGs not counted and get no barrier). LLVM: `getMaxWorkGroupsPerCU` = min(waveSlots / wavesPerWG, 16), because gfx1250 has `FeatureCuMode` | CDNA5 ISA §2.2 (p. 10). LLVM `Utils/AMDGPUBaseInfo.cpp:1166-1185`; `GCNFeatures.td:2499` @274ab686 | **Conflict** (32 vs 16). Both exempt single-wave WGs. Wave-slot cap: 16 × 4 = 64 waves per WGP (derived). |

---

### 3. `lds: Lds`

```rust
pub const LDS: Lds = Lds {
    bytes_per_workgroup: 327_680,      // 320 KiB
    alloc_granule: 2048,               // bytes
    banks: 64,                         // 4 bytes wide
    // DMA forms and how the LDS address is supplied:
    dma: &[
        LdsDma::AsyncLoad   { lds_addr: VgprPerLane /*VDST*/ , plus_ioffset: true, m0: None,          counter: ASYNC },
        LdsDma::AsyncStore  { lds_addr: VgprPerLane /*VSRC*/ , plus_ioffset: true, m0: None,          counter: ASYNC },
        LdsDma::ClusterLoad { lds_addr: VgprPerLane /*VDST*/ , plus_ioffset: true, m0: Some(WgMask),  counter: ASYNC },
        LdsDma::Tensor      { lds_addr: DescriptorField /*D# group0 [63:32]*/,   m0: None,          counter: TENSOR },
    ],
    // legacy gfx950 rule `m0_base_plus_imm` (M0 = LDS base for global_load_lds / buffer ... lds): not used on gfx1250
    ds_addtid_m0_offset: true,         // DS_{LOAD,STORE}_ADDTID_B32: M0 + imm + lane*4
};
```

| fact | value | source | notes |
|---|---|---|---|
| Max LDS per work-group (or single wave) | 320 KiB (0-320 KiB) | CDNA5 ISA §2.2 (p. 10); §3.1 Table 4 (p. 12); §3.3.4 (p. 19); §11.1 (p. 148). LLVM `GCNFeatures.td:2627` (`FeatureAddressableLocalMemorySize327680`); `Utils/AMDGPUBaseInfo.cpp:1126-1127,1148` @274ab686 | Agree. `VerifyLimits.LDS_LIMIT = 327680`. |
| Physical LDS per WGP | one memory per WGP shared between LDS (320 KB) and WGP$ (64 KB); the split can be configured | CDNA5 ISA §1.2.2.1 (p. 7); §11.1 (p. 148) | LLVM RSRC3 `TCP_SPLIT` for GFX125\* lists splits up to LDS=384 kB / VC=64 kB out of 448 kB (`docs/AMDGPUUsage.md:8018-8023`). **Conflict**: the ISA states 320 KB LDS + 64 KB WGP$. |
| Per-CU vs per-WGP | LDS belongs to the WGP. A WG always spans the 4 SIMDs, so the full 320 KiB is usable (no half-mode) | CDNA5 ISA §1.2.2.1 (p. 7); §3.3.4 (p. 19, every SIMD of the WGP can access LDS). LLVM `GCNSubtarget.h:362-364`, `Utils/AMDGPUBaseInfo.cpp:2507-2509` (gfx1250 always full-SIMD); [TP] `AMDGPUTargetParser.h:221` ("gfx1250 320 320/320 n/a") @274ab686 | Agree. |
| Allocation granule | 2048 bytes | CDNA5 ISA §3.3.4 (p. 19). LLVM `GCNFeatures.td:2628-2629` (alloc and encoding 2048); `docs/AMDGPUUsage.md:7808-7809` (GRANULATED_LDS_SIZE = roundup(size / (512×4))) @274ab686 | Agree. |
| Banks | 64 banks × 4 bytes; consecutive dwords go to consecutive banks; same-bank conflicts are serialized for indexed/atomic ops | CDNA5 ISA §1.2.2.1 (p. 8); §11.1 (p. 148); §11.2 (p. 149). LLVM `GCNFeatures.td:430,2630` (`FeatureLDSBankCount64`) @274ab686 | Agree. derived: bank = (byte_addr / 4) % 64. ISA does not give the exact bank hash. |
| DS addressing | addr = LDS_BASE + VGPR[ADDR] (byte) + 16-bit unsigned imm {offset1,offset0}. 2ADDR forms use two 8-bit offsets × 4 (×8 for 64-bit); STRIDE64 forms ×64 | CDNA5 ISA §11.2 Table 68 (p. 149); §11.2 (p. 150); §10.2 (p. 130) | Address is relative to the wave/WG allocation. Range check is on the zero-extended 32-bit address (p. 131). |
| DS ADDTID | addr = LDS_BASE + imm16 + lane×4 + M0 (M0 dword-aligned; no VGPR address) | CDNA5 ISA §11.2 (p. 150); §3.4.4 Table 8 (p. 23) | **ISA internal inconsistency** on the M0 width: Table 8 (p. 23) says lds_offset[19:0], Table 68 (p. 149) says Offset[15:0], and the DS_LOAD/STORE_ADDTID pseudocode (§15.15, p. 745) uses M0[19:0]. derived: 20 bits are needed for 320 KiB. |
| Async global↔LDS (DMA-style) | `GLOBAL_LOAD_ASYNC_TO_LDS_B{8,32,64,128}` / `GLOBAL_STORE_ASYNC_FROM_LDS_B*`: the per-lane LDS byte address comes from a VGPR (VDST for loads, VSRC for stores). INST_OFFSET is added to both the LDS and the global address. Global address uses GV or GVS. Tracked by ASYNCcnt | CDNA5 ISA §10.8 (pp. 135-136); §10.8.1 (p. 136). LLVM `FLATInstructions.td:391-418` (vdst holds the DS address, "will not use M0"; `Uses = [EXEC, ASYNCcnt]`), `:1252-1263`; `test/MC/AMDGPU/gfx1250_asm_vflat.s:3528-3534` @274ab686 | Agree. **M0 is not used for the LDS address** (unlike gfx950 `m0_base_plus_imm`). Async ops may write LDS out of order (p. 135). An out-of-allocation LDS address drops the write (p. 19). |
| Cluster multicast to LDS | `CLUSTER_LOAD_ASYNC_TO_LDS_B{8,32,64,128}`: LDS address as for async loads. M0[15:0] = target-WG mask, M0[16] = early timeout. M0[15:0]=0 means only the requesting WGP receives the data. Outside a cluster it acts as a plain global load | CDNA5 ISA §10.7 (pp. 134-135). LLVM `FLATInstructions.td:1239-1246` (`Uses = [M0, EXEC, ASYNCcnt]`) @274ab686 | ISA lists the multicast forms as B32/64/128 (p. 135). LLVM also defines a B8 form. Minor mismatch. |
| Tensor DMA (TDM) | `TENSOR_LOAD_TO_LDS` / `TENSOR_STORE_FROM_LDS`: the LDS address is the D# field `lds_addr` (group 0 bits [63:32], bytes). D# groups sit in SGPR blocks (VADDR0..3). No VGPRs, EXEC ignored, TENSORcnt | CDNA5 ISA §10.11.1 (p. 140); §10.11.2 (p. 141); §10.11.4 (p. 144). LLVM `MIMGInstructions.td:2318-2320,2361`; `test/MC/AMDGPU/gfx1250_asm_vimage_err.s:52-56` (D# SGPR blocks must be 4-aligned) @274ab686 | Optional LDS padding (pad_interval / pad_amount) is applied on load only (p. 141). |
| Legacy LDS-DMA (`global_load_lds_*`, `buffer_* lds`, M0 = LDS base) | no gfx1250 evidence: no gfx1250 MC test uses it, and the ISA lists only async/TDM/DS as LDS access methods | CDNA5 ISA §11.1.1 (p. 148). LLVM `FLATInstructions.td:1217-1225` (pseudos with M0), real encodings only at `:2838-2845` (GFX9/950) and `:3156-3160` (GFX10) @274ab686 | derived: absent on gfx1250. Confirm with an llvm-mc probe (Open). |
| LDS out-of-range | loads return 0, out-of-range store bytes are dropped, MEMVIOL is reported. Checked per byte (unaligned mode) or per dword | CDNA5 ISA §3.3.4.1 (p. 19) | |
| LDS alignment | natural alignment B8:1, B16:2, B32:4, B64:8, B96/B128:16 unless SH_MEM_CONFIG is unaligned. Atomics must be naturally aligned | CDNA5 ISA §3.3.4.1 (pp. 19-20); §3.3.3 (p. 19) | LLVM additionally sets `FeatureNeedsAligned2addrDS` on gfx1250-strict only (`GCNFeatures.td:2616`). |

---

### 4. Topology (prose only; no `Target` field yet)

- **WGP / CU / SIMD.** A WGP ("compute unit pair") holds 2 CUs. Each CU has 2 SIMD32s that share one memory path, so a WGP has 4 SIMD32s. CDNA5 ISA §1.1 Table 2 (p. 4); §2.2 (p. 10). Each SIMD pair connects to the WGP$ and to one Tensor Data Mover. derived: 2 TDMs per WGP (§1.2.2.1, p. 7).
- **Work-group placement.** All waves of a work-group go to one WGP and are spread over its 4 SIMDs (§2.2, p. 10). LLVM: gfx1250 always uses full-SIMD mode with 4 SIMDs per WG (`GCNSubtarget.h:362-364`; [TP] `AMDGPUTargetParser.h:256-258`), and `supportsWGP` returns false for gfx1250 (`Utils/AMDGPUBaseInfo.cpp:2521-2525`).
- **"CU mode" naming.** `FeatureCuMode` is set for gfx1250 (`GCNFeatures.td:2499`), so `.amdhsa_workgroup_processor_mode` defaults to 0 (`docs/AMDGPUUsage.md:17586-17587,17911-17914`). `AMDGPUUsage.md:2868-2870` says WGP mode is unsupported on gfx12+. In practice this does **not** halve LDS or SIMDs on gfx1250 (full-SIMD override above). derived: treat the WGP as the scheduling unit.
- **Workgroup size.** At most 1024 work-items (§2.2, p. 10; LLVM [TP] `AMDGPUTargetParser.h:277-281`). derived: at most 32 waves per WG, so at most 8 waves per SIMD from a single WG. The WG size in waves is 6 bits in the tg_size word (§3.5.3.1, p. 30).
- **Work-groups per WGP.** ISA: 32. LLVM: 16. See 2.5 (conflict).
- **Clusters.** Up to 16 WGs, each on a separate WGP of one shader engine. They synchronize with cluster barriers and share data through multicast loads (§2.3, p. 10). LLVM `FeatureClusters` (`GCNFeatures.td:2575`). Prose only per target-design §4.
- **Named barriers.** Up to 16 per WG (`NAMED_BAR_CNT`, granule 4). LLVM `docs/AMDGPUUsage.md:7995-8001`; ISA STATE_PRIV.NAMED_BARRIER_ID 0-16 (§3.4.2, p. 22).

---

### 5. `VerifyLimits` contributions (derived)

| constant | gfx1250 value | basis |
|---|---|---|
| `VGPR_LIMIT` | 1024 (≥ 1 block; next_free_vgpr rounded to 16) | §2.1 rows 1-3 |
| `SGPR_LIMIT` | 106 usable + VCC at 106-107 | §2.3 rows 1-2 |
| `LDS_LIMIT` | 327 680 bytes, granule 2048 | §3 rows 1, 4 |
| VGPR tuple rule | even start for ≥ 64-bit | §2.1 "Tuple alignment" |
| Low-VGPR-only operands | VIMAGE vaddr3-4, WMMA scale src0/1, `v_wmma_ld_scale*` | §2.1 |

---

### Open

1. **Physical VGPRs per SIMD and pool scope**: 1024 (64 × 16-VGPR blocks) and 16 waves/SIMD come only from LLVM ([TP] `AMDGPUTargetParser.cpp:441-450,584-587`; `GCNFeatures.td:1907`). The ISA is silent. Settle with a public AMD CDNA5 whitepaper or the HIP occupancy API, or by checking that `llc -mcpu=gfx1250` reports occupancy 16/8/4/1 for 64/128/256/1024 VGPRs (`.amdhsa`/`; Occupancy:` remarks, no GPU needed).
2. **Work-groups per WGP**: ISA 32 (§2.2 p. 10) vs LLVM 16 (`AMDGPUBaseInfo.cpp:1180-1184` with `FeatureCuMode`). This needs an AMD statement. LLVM may just be conservative.
3. **LDS/WGP$ split**: ISA 320 KB + 64 KB (p. 148) vs LLVM `TCP_SPLIT` options up to 384 kB LDS of 448 kB (`AMDGPUUsage.md:8018-8023`). Is more than 320 KiB per WG ever allowed? Check whether the rocm-10.1 assembler accepts `.amdhsa_group_segment_fixed_size 393216` for gfx1250.
4. **Dynamic VGPR mode** on gfx1250 (LLVM RSRC3 bit, `s_alloc_vgpr` assembles; ISA has neither). Probe: `llvm-mc -triple=amdgcn -mcpu=gfx1250 <<< 's_alloc_vgpr 16'` with the rocm-10.1 assembler, and wait for an ISA revision.
5. **Legacy M0-based LDS DMA** (`global_load_lds_dword`, `buffer_load_b32 … lds`) on gfx1250: expected absent. Probe with `llvm-mc -mcpu=gfx1250` (expect "instruction not supported").
6. **High-VGPR assembly syntax**: no gfx1250 MC test writes `v256`+ directly. LLVM prints low numbers plus an explicit `s_set_vgpr_msb` (`AMDGPULowerVGPREncoding.cpp:19-30`). Probe whether llvm-mc accepts `v_mov_b32 v300, v1` or requires `s_set_vgpr_msb 0x40` + `v_mov_b32 v44, v1`. Also check whether an odd-aligned `v[1:2]` 64-bit operand is rejected (no gfx1250 MC test found).
7. **`S_SET_VGPR_MSB` imm[15:8]**: the ISA documents only [7:0]. Confirm that hardware ignores the high byte LLVM writes (needs AMD statement). Safe choice: emit 0.
8. **DS ADDTID M0 width** (16 vs 20 bits; ISA pp. 23 / 149 / 745 disagree). Needs an ISA erratum. Use ≤ 16-bit offsets until settled, or 20-bit per pseudocode.
9. **VBUFFER/VIMAGE rows of the "not offset" table** (ISA p. 18): pdftotext layout is ambiguous. VIMAGE VADDR3-4 low-only is corroborated by LLVM `MIMGInstructions.td:566`. Read the rendered PDF table to confirm that VBUFFER has no low-only VGPR operand.
10. **User-SGPR doc mismatch** in AMDGPUUsage (generic "up to 16" at :8173 vs 32 at :7733-7740 / `AMDGPUBaseInfo.cpp:2420-2424`), and the tg_size word layout (ISA p. 30 vs `AMDGPUUsage.md:8248`). Probe: assemble a kernel with `.amdhsa_user_sgpr_count 32` for gfx1250.
11. **LDS bank mapping** beyond "64 × 4 B, consecutive addresses go to different banks" (exact hash, 64-bit/128-bit access conflict rules): unknown (no public source).
12. **Cluster multicast B8 form**: present in LLVM (`FLATInstructions.td:1242`) but not listed in ISA §10.7 (p. 135). Probe: `llvm-mc -mcpu=gfx1250 'cluster_load_async_to_lds_b8 v1, v[2:3], off'`.

## 4. P2_counters_sync — gfx1250 dependency counters, waits, memory model, synchronization

Public sources only:
- **ISA** = AMD "CDNA5" Instruction Set Architecture Reference Guide (27-July-2026), cited as `CDNA5 ISA §x (p. N)` with the printed page number.
- **LLVM** = upstream llvm-project @274ab686 (2026-11-09), cited as `LLVM <path>:<line> @274ab686`.

Conventions: "derived" means our own inference from the cited facts. "unknown (no public source)" means neither source settles the point.
Field names follow `target-design.md` §2 (`Counters{counters, classes, wait}`, `HazardRule`, `Waits{count, unit}`, `LatencyModel`). Barriers and clusters are not P0 `Target` fields, so §7–§8 are prose with sketches in comments only.

---

### 1. `counters.counters`: the dependency counters

```rust
// gfx1250: wave32, split per-counter waits. `in_order` is only a summary. The real ordering is
// per class (see §2): XCNT and ASYNCCNT are in order for one class and out of order across classes.
counters: &[
    Counter { name: "loadcnt",   max: 63, in_order: true  }, // CounterId 0
    Counter { name: "storecnt",  max: 63, in_order: true  }, // 1
    Counter { name: "dscnt",     max: 63, in_order: true  }, // 2
    Counter { name: "kmcnt",     max: 31, in_order: false }, // 3: SMEM returns out of order, so wait only for 0
    Counter { name: "xcnt",      max: 63, in_order: false }, // 4: in order for VMEM, out of order for SMEM
    Counter { name: "asynccnt",  max: 63, in_order: false }, // 5: loads in order, stores in order, load vs store not
    Counter { name: "tensorcnt", max: 63, in_order: true  }, // 6
],
// Not present on gfx1250: vmcnt / lgkmcnt / expcnt (pre-GFX12 combined form), samplecnt, bvhcnt.
```

| fact | value | source | notes |
|---|---|---|---|
| Counter set on CDNA5 | LOADcnt, STOREcnt, DScnt, KMcnt, ASYNCcnt, TENSORcnt, XCNT | CDNA5 ISA §3.1 (p. 13) | XCNT appears in the §3.1 state table. It is **missing** from the §5.7.1 counter table (p. 54), so its width comes only from §3.1, the S_WAIT_XCNT entry (p. 293) and LLVM. |
| LOADcnt width | 6 bits (max 63) | CDNA5 ISA §3.1 (p. 13); §5.7.1 (p. 54) | Same in LLVM: `getLoadcntBitWidth` = 6 for major ≥ 12, LLVM llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp:102-104 @274ab686 |
| STOREcnt width | 6 bits | CDNA5 ISA §5.7.1 (p. 54) | LLVM llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp:125-127 @274ab686 |
| DScnt width | 6 bits | CDNA5 ISA §5.7.1 (p. 54) | LLVM llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp:117-119 @274ab686 |
| KMcnt width | 5 bits (max 31) | CDNA5 ISA §5.7.1 (p. 54); S_WAIT_KMCNT reads SIMM16[4:0], §15.5 (p. 293) | LLVM llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp:130-132 @274ab686 |
| ASYNCcnt width | 6 bits | CDNA5 ISA §5.7.1 (p. 54) | LLVM width is 6 only when the version is 12.5: llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp:140-142 @274ab686 |
| TENSORcnt width | 6 bits | CDNA5 ISA §5.7.1 (p. 54); S_WAIT_TENSORCNT reads SIMM16[5:0], §15.5 (p. 294) | LLVM has no `getTensorcntBitWidth`. `HardwareLimits::get` returns 0 for ASYNC_CNT and TENSOR_CNT (no case for them), LLVM llvm/lib/Target/AMDGPU/AMDGPUWaitcntUtils.cpp:82-108 @274ab686. LLVM never saturates these two counters (it does not track them for automatic waits, §5). |
| XCNT width | 6 bits | CDNA5 ISA §3.1 (p. 13); S_WAIT_XCNT reads SIMM16[5:0], §15.5 (p. 293) | LLVM width is 6 only for version 12.5: llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp:135-137 @274ab686 |
| Counting unit | instructions, not lanes | CDNA5 ISA §5.7.1 (p. 53) | The exception is KMcnt, which counts dwords for SMEM (+1 or +2, see §2) |
| Overflow behaviour | issue stalls before a counter would overflow | CDNA5 ISA §5.7.1 (p. 53) | A compiler therefore needs no software cap for correctness. "derived": LLVM still clamps waits at max via `HardwareLimits`. |
| Wait semantics | `s_wait_Xcnt N` sleeps the wave until counter ≤ N | CDNA5 ISA §5.7 (p. 52), Table 25 (p. 53) | |
| Data return vs counter | VGPR writeback may be out of order, but the decrement reflects in-order completion | CDNA5 ISA §5.7.1 (p. 53) | Same statement in LLVM llvm/docs/AMDGPUUsage.md:14398-14404 @274ab686 |
| Ordering across types | in order within a type (except SMEM); out of order between types | CDNA5 ISA §5.7 (p. 53); §5.7.1 (p. 53) | LLVM llvm/docs/AMDGPUUsage.md:14395-14396 @274ab686 |
| LLVM enum | LOAD, DS, EXP, STORE, SAMPLE, BVH, KM, X, ASYNC, TENSOR (+ expert-mode VA_VDST_RD/WR, VM_VSRC) | LLVM llvm/lib/Target/AMDGPU/AMDGPUWaitcntUtils.h:28-46 @274ab686 | X, ASYNC and TENSOR are marked "gfx1250". EXP, SAMPLE and BVH exist in the enum but have no gfx1250 wait instruction (see §3). |
| Subtarget features | `FeatureWaitXcnt`, `FeatureAsynccnt`, `FeatureTDMInsts` in the 12.50 common set | LLVM llvm/lib/Target/AMDGPU/GCNFeatures.td:2535,2553,2554 @274ab686 | gfx1250 = `FeatureISAVersion12_50`, LLVM llvm/lib/Target/AMDGPU/GCNProcessors.td:315-316 @274ab686 |

---

### 2. `counters.classes`: which instruction classes use which counter

```rust
// (MemClass, CounterId, OrderKind). Rule for the wait pass (matches LLVM counterOutOfOrder):
// a counter may be waited on with N > 0 only while every pending class on it is InOrder and only
// ONE class is pending. Otherwise the only safe wait is 0.
classes: &[
    (VmemLoad /*global/buffer/scratch/flat load, atomic-with-return*/, LOADCNT,   InOrder),
    (CacheInv /*global_inv*/,                                          LOADCNT,   InOrder),   // in order with loads
    (ClusterLoad /*cluster_load_b32/64/128 (multicast to VGPR)*/,      LOADCNT,   InOrder),
    (VmemStore /*store, atomic-no-return*/,                             STORECNT,  InOrder),
    (CacheWb /*global_wb, global_wbinv*/,                               STORECNT,  InOrder),
    (Lds /*ds_* incl. ds_atomic_barrier_arrive_rtn_b64*/,               DSCNT,     InOrder),
    (FlatLdsPart /*flat_* also bumps DSCNT*/,                           DSCNT,     InOrder),   // see FLAT rule
    (Smem /*s_load*, s_buffer_load*, s_dcache_inv*/,                    KMCNT,     OutOfOrder),
    (Msg /*s_sendmsg, s_sendmsg_rtn_*, s_sendmsghalt*/,                 KMCNT,     OutOfOrder), // rtn in order among themselves only
    (BarrierQuery /*s_get_barrier_state, s_barrier_leave, s_barrier_signal_isfirst (WG)*/, KMCNT, OutOfOrder),
    (AsyncLoad /*global_load_async_to_lds_*, cluster_load_async_to_lds_*, ds_atomic_async_barrier_arrive_b64*/, ASYNCCNT, InOrder),
    (AsyncStore /*global_store_async_from_lds_*/,                       ASYNCCNT,  InOrder),   // load vs store: out of order
    (Tensor /*tensor_load_to_lds, tensor_store_from_lds*/,              TENSORCNT, InOrder),
    (VmemXlate /*every VMEM issue*/,                                    XCNT,      InOrder),
    (SmemXlate /*every SMEM issue*/,                                    XCNT,      OutOfOrder),
],
```

| fact | value | source | notes |
|---|---|---|---|
| LOADcnt classes | VMEM loads and atomics with return (VIMAGE, VBUFFER, FLAT/Scratch/Global), GLOBAL_INV | CDNA5 ISA §5.7.1 (p. 54); §5.7.1.2 (p. 55); §10.1.1 (p. 128) | +1 per instruction. Decremented when that load and all earlier ones are done. |
| STOREcnt classes | VMEM stores, atomics without return, GLOBAL_WB, GLOBAL_WBINV | CDNA5 ISA §5.7.1 (p. 54); §5.7.1.2 (p. 55) | Decrement happens when data reaches the level named by SCOPE (§4.1.1, p. 36). Stores are in order with other stores. A store then a load to the same address from one wave stays ordered (p. 55). |
| STOREcnt ordering subtlety | stores from one wave to *different* addresses are not kept in order with each other | CDNA5 ISA §5.7.1 (p. 53) | The counter still decrements in order (in-order "type" rule, p. 53). This is a memory-ordering fact, not a counter fact (derived). |
| DScnt classes | every LDS instruction; every FLAT instruction (LDS part) | CDNA5 ISA §5.7.1 (p. 54); §5.7.1.4 (p. 56) | An LDS load decrements when data is in VGPRs; an LDS store when the data is written to LDS. |
| FLAT rule | one FLAT op increments DScnt **and** LOADcnt or STOREcnt. The two halves complete independently, which is a WAW risk when two FLAT loads target the same VGPR | CDNA5 ISA §5.7.1.3 (p. 55); §10.1.1 (p. 128) | LLVM forces wait 0 on LOAD/DS while a FLAT is pending, unless `hasFlatLgkmVMemCountInOrder`: LLVM llvm/lib/Target/AMDGPU/SIInsertWaitcnts.cpp:1359-1364 @274ab686 |
| KMcnt: SMEM | +1 for loads of ≤ 32 bits, +2 for larger; S_DCACHE_INV adds 1 per scalar-cache bank. Returns out of order even to the same address, so only `s_wait_kmcnt 0` is meaningful | CDNA5 ISA §5.7.1 (p. 54); §5.7.1.1 (p. 55); §8.2 (p. 110-111) | §8.2 says "+1 per single-dword fetch or cache invalidate", while §5.7.1.1 says invalidates count per bank. Minor wording difference. |
| KMcnt: messages | +1 per S_SENDMSG, decremented when the message leaves the WGP. S_SENDMSG_RTN_* return in order among themselves but out of order with other KM users | CDNA5 ISA §5.7.1.5 (p. 56); §15.3 S_SENDMSG_RTN_B32/B64 (p. 258) | |
| KMcnt: barriers | S_GET_BARRIER_STATE (+1/−1), S_BARRIER_LEAVE (SCC result), S_BARRIER_SIGNAL_ISFIRST (SCC result, for **workgroup** barriers only; not for cluster barriers) | CDNA5 ISA §5.6.3 Table 22 (p. 48); §15.3 (p. 260); §15.5 S_BARRIER_LEAVE (p. 287) | LLVM models these SCC writes as event `SCC_WRITE` on KM_CNT (§5). |
| Scalar prefetch | S_PREFETCH_* do not use KMcnt | CDNA5 ISA §8.5 (p. 112) | They need `s_wait_xcnt 0` before them (§4 below). |
| ASYNCcnt classes | async loads and stores between memory and LDS; CLUSTER_LOAD_ASYNC_TO_LDS_*; DS_ATOMIC_ASYNC_BARRIER_ARRIVE_B64 | CDNA5 ISA §5.7.1 (p. 54); §10.7 (p. 135); §10.8 (p. 135-136); §11.2 (p. 152) | LLVM: `FLAT_Global_Load_LDS_Pseudo` with IsAsync sets `ASYNC_CNT=1`, `VM_CNT=0`, LLVM llvm/lib/Target/AMDGPU/FLATInstructions.td:402-404 @274ab686. Async store sets `ASYNC_CNT=1`, `VM_CNT=0`, :434-435. The barrier-arrive op sets `ASYNC_CNT=1, LGKM_CNT=0`, LLVM llvm/lib/Target/AMDGPU/DSInstructions.td:904-905 @274ab686. |
| ASYNCcnt ordering | loads complete ("done") in order with loads, stores in order with stores, **loads vs stores out of order**. LDS writes may land out of order even between two loads. The barrier-arrive op stays in order with async loads | CDNA5 ISA §10.8 (p. 135-136); §11.2 (p. 152) | **Conflict / gap:** LLVM has a single event `ASYNC_ACCESS` for all of these (LLVM llvm/lib/Target/AMDGPU/AMDGPUHWEvents.cpp:131-132 @274ab686), so `counterOutOfOrder` would treat mixed async loads and stores as in order. In practice LLVM never auto-waits ASYNC (§5), so this does not matter there. Our wait pass must model two classes (derived). |
| TENSORcnt | +1 per TDM instruction (TENSOR_LOAD_TO_LDS / TENSOR_STORE_FROM_LDS), with one "done" per instruction. Loads and stores are in order with each other within a wave, and unordered against other memory types | CDNA5 ISA §5.7.1 (p. 54); §10.11.1 (p. 140) | LLVM `TENSOR_CNT = 1` on the tensor pseudos, LLVM llvm/lib/Target/AMDGPU/MIMGInstructions.td:2301 @274ab686 |
| XCNT | +1 per memory instruction issued; −1 when its address translation is answered (XACK/XNACK). Decrements in issue order for VMEM, out of order for SMEM | CDNA5 ISA §3.1 (p. 13); Table 25 (p. 53) | LLVM: AMDGPUUsage describes the same mechanism and adds that waiting on any memory counter for N also implies xcnt ≤ N, LLVM llvm/docs/AMDGPUUsage.md:14388-14390 @274ab686. LLVM tracks it only when XNACK is enabled (§5). |
| XCNT when XNACK replay is absent | S_WAIT_XCNT acts as a NOP | CDNA5 ISA §15.5 S_WAIT_XCNT (p. 293) | gfx1250 supports XNACK on/off modes: LLVM llvm/lib/Target/AMDGPU/GCNFeatures.td:2658-2659 @274ab686 |
| Multicast cluster loads | stay in order with other loads of the same counter (LOADcnt or ASYNCcnt), unordered with other kinds | CDNA5 ISA §10.7 (p. 135) | Only the requesting wave's counter decrements (p. 135). |

---

### 3. `counters.wait`: wait instructions (`WaitForm::PerCounter`)

```rust
wait: WaitForm::PerCounter(&[
    (LOADCNT,   "s_wait_loadcnt"),   // SOPP op 64, SIMM16[5:0]
    (STORECNT,  "s_wait_storecnt"),  // SOPP op 65, SIMM16[5:0]
    (XCNT,      "s_wait_xcnt"),      // SOPP op 69, SIMM16[5:0]
    (DSCNT,     "s_wait_dscnt"),     // SOPP op 70, SIMM16[5:0]
    (KMCNT,     "s_wait_kmcnt"),     // SOPP op 71, SIMM16[4:0]
    (ASYNCCNT,  "s_wait_asynccnt"),  // SOPP op 74, SIMM16[5:0]
    (TENSORCNT, "s_wait_tensorcnt"), // SOPP op 75, SIMM16[5:0]
]),
// Proposed extension (not in P0): combined forms the emitter may fuse
//   s_wait_loadcnt_dscnt  (op 72): loadcnt  in SIMM16[13:8], dscnt in SIMM16[5:0]
//   s_wait_storecnt_dscnt (op 73): storecnt in SIMM16[13:8], dscnt in SIMM16[5:0]
// Full drain: s_wait_idle (op 10).  ALU waits: s_wait_alu (op 8), s_delay_alu (op 7), see §4.
// NOT available: s_waitcnt, s_waitcnt_{vmcnt,vscnt,expcnt,lgkmcnt}, s_wait_expcnt,
//                s_wait_samplecnt, s_wait_bvhcnt, s_wait_event.
```

| fact | value | source | notes |
|---|---|---|---|
| Opcodes | WAIT_LOADCNT 64, STORECNT 65, XCNT 69, DSCNT 70, KMCNT 71, LOADCNT_DSCNT 72, STORECNT_DSCNT 73, ASYNCCNT 74, TENSORCNT 75 | CDNA5 ISA §14.1 SOPP opcode table (p. 169-170); §15.5 (p. 293-294) | LLVM real encodings 0x40/0x41/0x46/0x47/0x48/0x49/0x4a/0x4b (LLVM llvm/lib/Target/AMDGPU/SOPInstructions.td:3047-3054,3069-3070 @274ab686) and XCNT 0x45 (:3062). All agree. |
| Immediate fields | each single wait uses SIMM16[5:0], except KMCNT uses [4:0] | CDNA5 ISA §15.5 (p. 293-294) | The LLVM assembler accepts any 16-bit value (e.g. `s_wait_asynccnt 0x1234`, LLVM llvm/test/MC/AMDGPU/gfx1250_asm_sopp.s:5-7 @274ab686). Hardware reads only the low bits (derived). |
| Combined forms field layout | LOAD/STORE count in SIMM16[13:8], DS count in SIMM16[5:0] | CDNA5 ISA §15.5 (p. 293-294) | **ISA-internal conflict:** Table 25 (p. 53) gives load/store as SIMM16[15:8]. LLVM uses shift 8 (LLVM llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp:145-147 @274ab686) and width 6, which matches p. 293. |
| LLVM fusing | if DS and LOAD are both needed it emits `s_wait_loadcnt_dscnt`, else DS and STORE give `s_wait_storecnt_dscnt`; the rest are emitted singly | LLVM llvm/lib/Target/AMDGPU/SIInsertWaitcnts.cpp:2243-2290 @274ab686 | |
| `s_wait_idle` | waits until every dependency and memory counter is zero | CDNA5 ISA §15.5 S_WAIT_IDLE (p. 286) | LLVM gfx11/12 encoding SOPP 0x00a (LLVM llvm/lib/Target/AMDGPU/SOPInstructions.td:3257 @274ab686). Accepted for 12.50 (LLVM llvm/test/MC/AMDGPU/gfx12_asm_sopp.s:3,281-282 @274ab686). |
| `s_waitcnt` unsupported | not on gfx1250 | LLVM llvm/lib/Target/AMDGPU/SOPInstructions.td:1710-1713 @274ab686 (`isNotGFX1250Plus`); LLVM llvm/test/MC/AMDGPU/gfx1250_asm_unsupported.s:223-226 @274ab686 | The ISA does not list S_WAITCNT in the SOPP opcode table (p. 169-170). |
| `s_wait_expcnt`, `s_wait_event` unsupported | rejected for gfx1250 | LLVM llvm/test/MC/AMDGPU/gfx1250_asm_unsupported.s:195-199 @274ab686 | No export unit (predicate `HasExportInsts`, LLVM llvm/lib/Target/AMDGPU/SOPInstructions.td:1879,1908 @274ab686) |
| `s_wait_samplecnt`, `s_wait_bvhcnt` unsupported | rejected for gfx1250 | LLVM llvm/test/MC/AMDGPU/gfx1250_asm_unsupported.s:215-221 @274ab686 | |
| SOPK `s_waitcnt_vmcnt/vscnt/expcnt/lgkmcnt` | only gfx10/gfx11 encodings exist | LLVM llvm/lib/Target/AMDGPU/SOPInstructions.td:2769-2772,2792-2795 @274ab686 | So they are unavailable on gfx12/gfx1250 (derived: no gfx12 real). |
| Clause legality | `S_WAIT_*CNT`, `S_WAIT_IDLE`, `S_DELAY_ALU` are allowed inside an `s_clause` but not as its first instruction | CDNA5 ISA §5.3 (p. 42-43) | For S_DELAY_ALU there is conflicting text; see §4. |

---

### 4. ALU waits and instruction-required waits (`HazardRule` rows with `Waits{count: Instr}`)

```rust
// s_delay_alu: an optional performance hint. Put it in LatencyModel as a "delay hint" class, not a hazard.
// SIMM16 = InstID0[3:0] | InstSkip[6:4] << 4 | InstID1[10:7] << 7
//   InstID: 0 none, 1..4 VALU 1..4 back, 5..7 TRANS32 1..3 back, 8 reserved (FMA accum), 9..11 SALU 1..3 cycles
//   InstSkip: 0 same, 1 next, 2..5 skip 1..4, 6..7 reserved
// s_wait_alu (op 8) = LLVM S_WAITCNT_DEPCTR, fields (LLVM AsmUtils):
//   sa_sdst[0] va_vcc[1] vm_vsrc[4:2] hold_cnt[7] va_ssrc[8] va_sdst[11:9] va_vdst[15:12]
hazards: &[
  HazardRule { name: "xcnt_before_atomic_store_rmw", producer: AnyVmemOrSmem, consumer: GlobalFlatAtomicOrStoreAtomic,
               role: Operand::None, waits: Waits{ count: Instr("s_wait_xcnt 0"), unit: WaitStates }, scope: Physical,
               source: "AMDGPUUsage.md:14392-14393; SIMemoryLegalizer.cpp:2251-2256" },
  HazardRule { name: "xcnt_before_prefetch", producer: AnyMem, consumer: SPrefetch,
               role: Operand::None, waits: Waits{ count: Instr("s_wait_xcnt 0"), unit: WaitStates }, scope: Physical,
               source: "CDNA5 ISA §8.5.1 p.112" },
  HazardRule { name: "xcnt_before_sleep_forever", producer: AnyMem, consumer: SSleepForever,
               role: Operand::None, waits: Waits{ count: Instr("s_wait_xcnt 0"), unit: WaitStates }, scope: Physical,
               source: "CDNA5 ISA §5.2 Table 15 p.41" },
  HazardRule { name: "idle_before_getreg_status", producer: Any, consumer: SGetregStatusStatePriv,
               role: Operand::None, waits: Waits{ count: Instr("s_wait_idle"), unit: WaitStates }, scope: Physical,
               source: "CDNA5 ISA §5.7.2 p.56 (LLVM uses s_wait_alu 0 instead)" },
  HazardRule { name: "tdm_drain", producer: Tensor, consumer: Tensor, role: Operand::None,
               waits: Waits{ count: Instr("s_wait_tensorcnt 10"), unit: WaitStates }, scope: Physical,
               source: "GCNHazardRecognizer.cpp:4539-4562 (LLVM-only, FeatureNeedsTDMDrain)" },
],
```

| fact | value | source | notes |
|---|---|---|---|
| S_DELAY_ALU opcode and fields | op 7; InstID0 SIMM16[3:0], InstSkip [6:4], InstID1 [10:7] | CDNA5 ISA Table 25 (p. 53); §15.5 (p. 283) | LLVM gfx11/12 SOPP 0x007, LLVM llvm/lib/Target/AMDGPU/SOPInstructions.td:3254 @274ab686. Assembler syntax `instid0(VALU_DEP_1) \| instskip(NEXT) \| instid1(...)`, LLVM llvm/test/MC/AMDGPU/gfx12_asm_sopp.s:89-108 @274ab686 |
| InstID codes | 0 none; 1-4 VALU N back; 5-7 TRANS32 N back; 8 reserved; 9-11 SALU wait 1-3 cycles | CDNA5 ISA §5.8 Table 26 (p. 58); §15.5 (p. 283-284) | p. 284 names code 8 "FMA_ACCUM_CYCLE_1 (reserved)" |
| InstSkip codes | 0 same instruction; 1 next; 2-5 skip 1-4 instructions; 6 reserved | CDNA5 ISA Table 26 (p. 58); §15.5 (p. 284-285) | |
| Counting rules | InstID counts issued VALUs (EXEC-skipped VALUs count; branched-over ones do not). Skip counts all instructions except S_SET_VGPR_MSB and S_WAIT_ALU. A new S_DELAY_ALU replaces unconsumed info. Optional for correctness | CDNA5 ISA §5.8 (p. 57) | |
| TRANS class for delay | 16/32-bit SIN, COS, SQRT, RCP, RSQ, TANH, EXP, LOG. 64-bit trans count as VALU. **XDL WMMA (≤16-bit data) is tracked as TRANS**. LD_SCALE+WMMA counts as one instruction | CDNA5 ISA §5.8 (p. 58) | Matches LLVM: XDL WMMA is the TRANS delay type on gfx1250 and F64 trans is excluded, LLVM llvm/lib/Target/AMDGPU/AMDGPUInsertDelayAlu.cpp:69-75 @274ab686 |
| S_DELAY_ALU in clauses | §5.8 (p. 58) says not to use it inside VALU clauses, and §15.5 (p. 285) calls it illegal inside an S_CLAUSE clause | CDNA5 ISA §5.8 (p. 58); §15.5 (p. 285) | **ISA-internal conflict** with §5.3 (p. 43), which lists S_DELAY_ALU as legal inside a clause (not first). Conservative choice (derived): never place it inside a clause; put it before `s_clause` (p. 43). |
| S_WAIT_ALU | opcode 8 exists | CDNA5 ISA §14.1 (p. 169) | **The ISA gives no field description.** LLVM: S_WAITCNT_DEPCTR is renamed `s_wait_alu` on gfx12, SOPP 0x008, LLVM llvm/lib/Target/AMDGPU/SOPInstructions.td:3255 @274ab686 |
| S_WAIT_ALU fields | depctr_hold_cnt bit 7 (1 bit), sa_sdst bit 0 (1), va_vdst bits 15:12 (4), va_sdst bits 11:9 (3), va_ssrc bit 8 (1), va_vcc bit 1 (1), vm_vsrc bits 4:2 (3) | LLVM llvm/lib/Target/AMDGPU/Utils/AMDGPUAsmUtils.cpp:66-74 @274ab686; LLVM llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp:154-200 @274ab686 | Semantics (what each sub-counter counts) are unknown (no public ISA source). Syntax example: LLVM llvm/test/MC/AMDGPU/gfx12_asm_sopp.s:242-255 @274ab686 (run for 12.50 at :3). |
| LLVM expert scheduling mode | opt-in attribute `amdgpu-expert-scheduling-mode`. Writes SCHED_MODE = 2 via `s_setreg_imm32_b32`, then tracks VA_VDST / VM_VSRC and emits `s_wait_alu` | LLVM llvm/docs/AMDGPUUsage.md:3488-3489 @274ab686; LLVM llvm/lib/Target/AMDGPU/SIInsertWaitcnts.cpp:2969-2976,2292-2309,3491-3497 @274ab686 | The ISA documents only SCHED_MODE bit 2 (DISABLE_XDL_ARB_STALL), §5.7.2.1 (p. 56). The meaning of SCHED_MODE[1:0]=2 is unknown (no public source). Not needed for P2 (derived). |
| `s_wait_xcnt 0` before flat/global atomic store or RMW | required so the atomic stays atomic across an XNACK replay | LLVM llvm/docs/AMDGPUUsage.md:14392-14393 @274ab686; LLVM llvm/lib/Target/AMDGPU/SIMemoryLegalizer.cpp:2251-2256 @274ab686 | LLVM also adds it before every **volatile** VMEM access (SIMemoryLegalizer.cpp:2226-2231). The ISA is silent on this rule. |
| `s_wait_xcnt 0` before S_PREFETCH_* | required | CDNA5 ISA §8.5.1 (p. 112) | LLVM handling not found in the checked files (Open). |
| `s_wait_xcnt 0` before "sleep forever" | required for S_SLEEP / S_MONITOR_SLEEP with SIMM16[15]=1 | CDNA5 ISA §5.2 Table 15 (p. 41) | LLVM (FeatureNoSleepForever) rewrites a monitor-sleep-forever to 127 instead, LLVM llvm/lib/Target/AMDGPU/AMDGPULowerIntrinsics.cpp:247-263 @274ab686; feature in gfx1250 set, LLVM llvm/lib/Target/AMDGPU/GCNFeatures.td:2655 @274ab686 |
| Wait before S_GETREG | ISA: wait-idle before S_GETREG of STATUS, STATE_PRIV, SCRATCH_BASE, EXCP_FLAG_PRIV, EXCP_FLAG_USER; also before xnack_state_priv / xnack_mask access | CDNA5 ISA §5.7.2 (p. 56) | **Conflict:** LLVM inserts `s_wait_alu 0` (S_WAITCNT_DEPCTR 0), not `s_wait_idle`, and only for STATUS, STATE_PRIV, EXCP_FLAG_PRIV and EXCP_FLAG_USER (no SCRATCH_BASE): LLVM llvm/lib/Target/AMDGPU/GCNHazardRecognizer.cpp:4417-4436 @274ab686, LLVM llvm/lib/Target/AMDGPU/GCNSubtarget.h:990-992 @274ab686 |
| TDM drain (LLVM) | before a TDM op, if an earlier TDM op has not been covered by `s_wait_tensorcnt ≤ 10`, insert `s_wait_tensorcnt 10` | LLVM llvm/lib/Target/AMDGPU/GCNHazardRecognizer.cpp:1973-1974,4539-4562 @274ab686; feature "Needs to keep tensor count low", LLVM llvm/lib/Target/AMDGPU/GCNFeatures.td:1683-1686,2654 @274ab686 | The ISA is silent; the reason is unknown (no public source). Adopt it (derived: harmless). |

---

### 5. How LLVM `SIInsertWaitcnts` models gfx1250

| fact | value | source | notes |
|---|---|---|---|
| Generator | GFX12+ path: `WaitcntGeneratorGFX12Plus`, `MaxCounter = NUM_EXTENDED_INST_CNTS` (or expert) | LLVM llvm/lib/Target/AMDGPU/SIInsertWaitcnts.cpp:3491-3497 @274ab686 | `hasExtendedWaitCounts` = generation ≥ GFX12, LLVM llvm/lib/Target/AMDGPU/GCNSubtarget.h:670 @274ab686 |
| Event → counter (GFX12+) | LOAD ← VMEM_READ_ACCESS, VMEM_INV_ACCESS. DS ← LDS_ACCESS, GDS_ACCESS. STORE ← VMEM_WRITE_ACCESS, SCRATCH_WRITE_ACCESS. KM ← SMEM_ACCESS, SQ_MESSAGE, SCC_WRITE. X ← VMEM_GROUP, SMEM_GROUP. ASYNC ← ASYNC_ACCESS. TENSOR ← TENSOR_ACCESS. (EXP/SAMPLE/BVH events exist but cannot occur on gfx1250.) | LLVM llvm/lib/Target/AMDGPU/SIInsertWaitcnts.cpp:294-315 @274ab686 | Counter→instruction table at :101-107 |
| Instruction → event | DS (LGKM) → LDS_ACCESS. FLAT → VMEM_GROUP (if XNACK) + simplified VMEM event + LDS_ACCESS (if it may hit LDS) + ASYNC_ACCESS (async). TENSOR_CNT instr → TENSOR_ACCESS. VMEM → simplified event (+VMEM_GROUP if XNACK). SMRD → SMEM_ACCESS (+SMEM_GROUP if XNACK). S_BARRIER_LEAVE and S_BARRIER_SIGNAL_ISFIRST_* → SCC_WRITE. S_SENDMSG* → SQ_MESSAGE. S_GET_BARRIER_STATE_* and S_MEMTIME → SMEM_ACCESS | LLVM llvm/lib/Target/AMDGPU/AMDGPUHWEvents.cpp:107-193 @274ab686; LLVM llvm/lib/Target/AMDGPU/SIInstrInfo.h:909-913 @274ab686 | GLOBAL_INV → VMEM_INV_ACCESS (LOAD); GLOBAL_WB/WBINV → VMEM_WRITE_ACCESS (STORE), LLVM llvm/lib/Target/AMDGPU/AMDGPUHWEvents.cpp:62-68 @274ab686. Matches the ISA (p. 55, p. 128). |
| DS_ATOMIC_ASYNC_BARRIER_ARRIVE_B64 | has `LGKM_CNT=0` and is not FLAT/VMEM, so `getEventsForImpl` returns NONE (derived) | LLVM llvm/lib/Target/AMDGPU/DSInstructions.td:904-905 @274ab686; LLVM llvm/lib/Target/AMDGPU/AMDGPUHWEvents.cpp:110-192 @274ab686 | The ISA says it uses ASYNCcnt (p. 152). LLVM does not auto-track ASYNC anyway. |
| Out-of-order rule | a counter is OOO if (a) it is the SMEM counter (KM) with SMEM pending, (b) it is X with SMEM_GROUP pending, (c) for LOAD: >1 event type pending, ignoring VMEM_INV_ACCESS, (d) otherwise when mixed event types are pending. An OOO counter can only be waited to 0 | LLVM llvm/lib/Target/AMDGPU/SIInsertWaitcnts.cpp:1568-1605,1365-1369,1535-1537 @274ab686 | This is the "per-class OrderKind + single-class" rule proposed in §2. |
| XCNT tracking | only when `hasWaitXcnt() && isXNACKEnabled()`. Scores are set on all *use* operands of the memory instruction. A wait is needed before an instruction **defines** (overwrites) such a register, i.e. a WAR on address/data sources until translation is done | LLVM llvm/lib/Target/AMDGPU/SIInsertWaitcnts.cpp:961-973,2553-2554 @274ab686; LLVM llvm/lib/Target/AMDGPU/AMDGPUHWEvents.cpp:123-124,147-148,157-158 @274ab686 | "derived": with XNACK off, LLVM emits no XCNT waits except the memory-legalizer soft waits. |
| XCNT SMEM/VMEM interleave | LLVM assumes hardware inserts an implicit xcnt wait between interleaved SMEM and VMEM, so only one group is outstanding at a time | LLVM llvm/lib/Target/AMDGPU/SIInsertWaitcnts.cpp:962-971 @274ab686 | **The ISA is silent** (Open) |
| XCNT implied by other waits | `kmcnt 0` with SMEM pending implies xcnt 0. A loadcnt N wait with only VMEM loads pending (no stores) implies xcnt ≤ N. Before a VMEM instruction, the XCNT wait is dropped because translation is in order | LLVM llvm/lib/Target/AMDGPU/SIInsertWaitcnts.cpp:1297-1319,1544-1557,2597-2603 @274ab686 | AMDGPUUsage states the general implication, LLVM llvm/docs/AMDGPUUsage.md:14390 @274ab686 |
| XCNT drain points | branches, calls, returns, S_ENDPGM*, S_TRAP, S_GETREG/S_SETREG*, S_SENDMSG*, S_BARRIER_WAIT, S_BARRIER_SIGNAL*_{M0,IMM} are treated as resetting XCNT to 0 | LLVM llvm/lib/Target/AMDGPU/SIInstrInfo.cpp:3709-3736 @274ab686; LLVM llvm/lib/Target/AMDGPU/SIInsertWaitcnts.cpp:2769-2772 @274ab686 | **The ISA is silent** on implicit XCNT drains (Open) |
| ASYNC / TENSOR not auto-waited | blanket waits (calls, all-zero) leave ASYNC and TENSOR at "no wait". They are only waited for through `llvm.amdgcn.wait.asyncmark` or explicit builtins | LLVM llvm/lib/Target/AMDGPU/SIInsertWaitcnts.cpp:224-229,1891-1895,2759-2763 @274ab686; LLVM llvm/docs/AMDGPUUsage.md:14414-14416 @274ab686 | Our pass must insert them itself (derived: our DSL owns TDM/async). |
| Function entry (non-kernel) | emits `s_wait_loadcnt_dscnt 0` and `s_wait_kmcnt 0` (+ expert `s_wait_alu`); STORE, X, ASYNC and TENSOR are not waited | LLVM llvm/lib/Target/AMDGPU/SIInsertWaitcnts.cpp:3523-3549 @274ab686 | |
| SCC from barrier ops | SCC_WRITE goes on KM_CNT. A later `s_barrier_wait` with the same immediate as a pending `s_barrier_signal_isfirst` clears it (no kmcnt wait needed) | LLVM llvm/lib/Target/AMDGPU/SIInsertWaitcnts.cpp:1074-1077,1500-1516,2430-2431 @274ab686 | Consistent with the ISA: read SCC only after S_BARRIER_WAIT (p. 48) |
| Barrier and outstanding memory | gfx1250 has BackOffBarrier, so no forced all-zero wait before barriers | LLVM llvm/lib/Target/AMDGPU/SIInsertWaitcnts.cpp:2560-2575 @274ab686; LLVM llvm/lib/Target/AMDGPU/GCNFeatures.td:2497 @274ab686 | ISA: a wave trapped in S_BARRIER_WAIT re-executes it after RFE (p. 49) |

---

### 6. Memory model: scopes, acquire/release, cache control

```rust
// Prose-only sketch: memory-model data a gfx1250 backend needs.
// scope field (2 bits): 0 WGP ("SCOPE_CU" in asm) | 1 SE | 2 DEV | 3 SYS
// LLVM syncscope -> scope: system/one-as:SYS, agent:DEV, cluster:SE, workgroup/wavefront/singlethread:CU
// acquire(scope >= cluster, global): s_wait_{load|store}cnt 0 (+dscnt) ; global_inv scope:SE ; [agent+] s_wait_loadcnt 0
// release(workgroup|cluster): s_wait_storecnt 0 ; s_wait_loadcnt 0 ; s_wait_dscnt 0 ; s_wait_xcnt 0 (atomics)
// release(agent|system):     as above; global_wb only when MTYPE NC may be used (not on gfx1250 per LLVM features)
```

| fact | value | source | notes |
|---|---|---|---|
| SCOPE encoding | 0 WGP, 1 SE, 2 DEV, 3 SYS | CDNA5 ISA §4.1.1 (p. 35) | LLVM asm names `SCOPE_CU/SE/DEV/SYS`, LLVM llvm/docs/AMDGPUUsage.md:14423-14426 @274ab686 |
| SCOPE semantics | if the instruction's scope ≤ the cache's scope, the access hits or completes there; otherwise it is forwarded to the next level | CDNA5 ISA §4.1.1 (p. 35) | LLVM llvm/docs/AMDGPUUsage.md:14434-14443 @274ab686 |
| Release primitive | "WB" op with a scope: caches whose scope is smaller than the requested scope flush their dirty lines | CDNA5 ISA §4.1.1 (p. 35) | |
| Acquire primitive | "INV" op with a scope: caches whose scope is smaller than the requested scope are invalidated | CDNA5 ISA §4.1.1 (p. 36) | |
| Store completion | STOREcnt decrements when the store is done at its SCOPE level | CDNA5 ISA §4.1.1 (p. 36); §5.7.1 (p. 54) | |
| WB/INV effect table | scope 0: no-op; scope 1-2: WGP$ only; scope 3: WGP$ and L2 | CDNA5 ISA §4.1.1 Table 13 (p. 37) | AMDGPUUsage: affects caches whose scope is strictly smaller, LLVM llvm/docs/AMDGPUUsage.md:14454-14456 @274ab686 |
| Cache-control instructions | GLOBAL_INV (LOADcnt), GLOBAL_WB (STOREcnt), GLOBAL_WBINV (STOREcnt); FLAT-encoded, with `scope:` | CDNA5 ISA §5.7.1.2 (p. 55); §10.1 (p. 128) | Encodings: LLVM llvm/test/MC/AMDGPU/gfx1250_asm_vflat.s:2081-2106 @274ab686. AMDGPUUsage: wbinv = inv then wb; these are ordered with other memory ops of the wave, LLVM llvm/docs/AMDGPUUsage.md:14458-14473 @274ab686 |
| NV bit | WB/INV with NV=0 touch only NV=0 lines; with NV=1 they touch all lines | CDNA5 ISA §4.1.2 (p. 38-39) | LLVM sets NV on non-volatile accesses for GFX12.5, LLVM llvm/lib/Target/AMDGPU/SIMemoryLegalizer.cpp:2189-2198 @274ab686 |
| Scalar caches | the K$ (scalar data cache) is read-only and not coherent with vector caches; S_DCACHE_INV + `s_wait_kmcnt 0` | CDNA5 ISA §5.7.1.1 (p. 55); §8.1.4 (p. 110) | LLVM uses SMEM only for data that cannot change during the dispatch, so it never invalidates, LLVM llvm/docs/AMDGPUUsage.md:14487-14494 @274ab686 |
| SMEM scope | the SMEM SCOPE field uses the same encoding. WGP scope is not coherent with VMEM stores | CDNA5 ISA §4.1.1 (p. 37) | |
| syncscope map | none/system → SYS; agent → DEV; cluster → SE; workgroup/wavefront/singlethread → CU (default) | LLVM llvm/docs/AMDGPUUsage.md:14514-14537 @274ab686; LLVM llvm/lib/Target/AMDGPU/SIMemoryLegalizer.cpp:2300-2321 @274ab686 | Workgroup gets SE only in WGP mode. gfx1250 sets `FeatureCuMode`, LLVM llvm/lib/Target/AMDGPU/GCNFeatures.td:2499 @274ab686 |
| `cluster` syncscope | synchronizes threads of one cluster. On targets without clusters it acts as agent | LLVM llvm/docs/AMDGPUUsage.md:2195-2202 @274ab686; LLVM llvm/lib/Target/AMDGPU/SIMemoryLegalizer.cpp:63,202 @274ab686 | |
| Acquire, workgroup | load atomic then `s_wait_loadcnt 0` (+`s_wait_dscnt 0` for generic). No cache invalidate (shared WGP$) | LLVM llvm/docs/AMDGPUUsage.md:14713-14747 @274ab686; LLVM llvm/lib/Target/AMDGPU/SIMemoryLegalizer.cpp:2076-2088 @274ab686 | |
| Acquire, cluster | wait for the load or RMW (loadcnt / storecnt 0, + dscnt for generic), then `global_inv` at SE scope | LLVM llvm/docs/AMDGPUUsage.md:14748-14765,14896-14922 @274ab686; LLVM llvm/lib/Target/AMDGPU/SIMemoryLegalizer.cpp:2073-2075 @274ab686 | |
| Acquire, agent/system | as for cluster, but `global_inv` uses the table's scope, or SE when MTYPE NC is not used, then `s_wait_loadcnt 0` after the inv | LLVM llvm/docs/AMDGPUUsage.md:14766-14787 @274ab686; LLVM llvm/lib/Target/AMDGPU/SIMemoryLegalizer.cpp:2059-2072,2105-2114 @274ab686 | gfx1250 (non-strict) has `FeatureNoMTypeNC`, so the inv uses SCOPE_SE (LLVM llvm/lib/Target/AMDGPU/GCNFeatures.td:2656 @274ab686). gfx1250-strict lacks it (:2588-2622) and uses DEV/SYS. |
| Fence acquire | workgroup: `s_wait_storecnt 0; s_wait_loadcnt 0; s_wait_dscnt 0`. Cluster/agent/system: same plus `global_inv` (+ `s_wait_loadcnt 0` for agent/system) | LLVM llvm/docs/AMDGPUUsage.md:15022-15088 @274ab686 | |
| Release, workgroup/cluster | `s_wait_storecnt 0; s_wait_loadcnt 0; s_wait_dscnt 0`, then `s_wait_xcnt 0`, then the store/RMW at the table scope | LLVM llvm/docs/AMDGPUUsage.md:15107-15132,15203-15228 @274ab686 | No `global_wb` for cluster or workgroup, LLVM llvm/lib/Target/AMDGPU/SIMemoryLegalizer.cpp:2158-2162 @274ab686 |
| Release, agent/system | waits, then `global_wb` (only if MTYPE NC may be used; omitted otherwise), then waits again, `s_wait_xcnt 0`, and the op | LLVM llvm/docs/AMDGPUUsage.md:15149-15189,15245-15286 @274ab686; LLVM llvm/lib/Target/AMDGPU/SIMemoryLegalizer.cpp:2137-2181 @274ab686 | For gfx1250 (NoMTypeNC) no `global_wb` is emitted (derived from :2149-2156). |
| Fence release | workgroup/cluster: the three waits. Agent/system: waits + optional `global_wb` + waits | LLVM llvm/docs/AMDGPUUsage.md:15293-15347 @274ab686 | |
| seq_cst | acquire/release sequences + leading waits on load atomics | LLVM llvm/docs/AMDGPUUsage.md:15706-15765 @274ab686 | |
| Async/tensor excluded | ASYNC-LDS and tensor operations are outside the LLVM memory model | LLVM llvm/docs/AMDGPUUsage.md:14414-14416 @274ab686 | The section is marked incomplete (multicast, barriers and NV are not covered), LLVM llvm/docs/AMDGPUUsage.md:14362-14367 @274ab686 |
| Cache structure (LLVM) | write-through WGP$ shared by LDS and vector L0 (clean data only), with 2 request queues (one per SIMD pair); SE-level L1 buffer; possibly several L2s per agent | LLVM llvm/docs/AMDGPUUsage.md:14317-14358 @274ab686 | Cooperative atomics are forced to ≥ SCOPE_DEV, LLVM llvm/lib/Target/AMDGPU/SIMemoryLegalizer.cpp:2287-2298 @274ab686 |

---

### 7. `sync` (prose-only): barriers

```rust
// Prose-only (no P0 field). Split barrier: signal then wait. IDs are signed inline constants.
// BarrierId: -4 ClusterTrap(priv) | -3 ClusterUser | -2 WgTrap(priv) | -1 Workgroup | 0 Null | 1..=16 Named
// s_barrier_signal  <id|m0>     SOP1 op 78    (m0: id = M0[4:0]; named member count = M0[22:16] if != 0)
// s_barrier_signal_isfirst <id> SOP1 op 79    SCC result; KMcnt for WG barriers; NOT for id -3
// s_barrier_wait    <imm16>     SOPP op 20    inline constant only
// s_get_barrier_state sD,<id|m0> SOP1 op 80   KMcnt
// s_barrier_init <id|m0> 81, s_barrier_join <id|m0> 82, s_barrier_leave (SOPP 21), s_wakeup_barrier <id|m0> 87
// Workgroup barrier idiom (LLVM): s_barrier_signal -1 ; s_barrier_wait -1
// Cluster barrier idiom (LLVM): s_barrier_signal_isfirst -1 ; s_barrier_wait -1 ; if SCC: s_barrier_signal -3 ; s_barrier_wait -3
```

| fact | value | source | notes |
|---|---|---|---|
| Model | split barrier: S_BARRIER_SIGNAL (arrive) then S_BARRIER_WAIT. The barrier completes when the signal count reaches the member count, then resets | CDNA5 ISA §5.6 (p. 45); §5.6.1 (p. 46) | No monolithic `s_barrier` on gfx12/gfx1250 in LLVM: the intrinsic is lowered to signal −1 + wait −1, LLVM llvm/lib/Target/AMDGPU/AMDGPULowerIntrinsics.cpp:218-231 @274ab686. S_BARRIER real encodings exist only up to gfx11, LLVM llvm/lib/Target/AMDGPU/SOPInstructions.td:3025,3120 @274ab686 |
| Workgroup size | ≤ 1024 work-items = 32 wave32 | CDNA5 ISA §5.6 (p. 45); §2.2 (p. 10) | |
| Single-wave WG | all workgroup-barrier instructions are S_NOP; no barrier resource is allocated | CDNA5 ISA §2.2 (p. 10); §5.6 (p. 46) | LLVM downgrades them to `wave_barrier`, and isfirst gives true, LLVM llvm/lib/Target/AMDGPU/AMDGPULowerIntrinsics.cpp:202-216 @274ab686 |
| Barrier IDs | −4 cluster trap, −3 cluster user, −2 WG trap, −1 workgroup, 0 null, 1-16 named | CDNA5 ISA §5.6.3 Table 21 (p. 47) | LLVM `AMDGPU::Barrier` enum, LLVM llvm/lib/Target/AMDGPU/SIDefines.h:1358-1369 @274ab686 |
| Trap barriers | −2 and −4 are privileged: ignored unless PRIV=1 | CDNA5 ISA §15.3 S_BARRIER_SIGNAL (p. 259); §5.6 (p. 46) | |
| Early exit | a wave may terminate instead of signaling. Signaling and then exiting is not allowed | CDNA5 ISA §5.6 (p. 46) | |
| Validity | WG barrier completes only after all waves of the WG are created. Named barriers do not need this | CDNA5 ISA §5.6 (p. 45-46) | |
| Opcodes | SIGNAL 78, SIGNAL_ISFIRST 79, GET_BARRIER_STATE 80, INIT 81, JOIN 82, WAKEUP_BARRIER 87 (SOP1); WAIT 20, LEAVE 21 (SOPP) | CDNA5 ISA §15.3 (p. 259-262); §15.5 (p. 286-287) | LLVM SOP1 0x4e/0x4f/0x50/0x51/0x52/0x57 (LLVM llvm/lib/Target/AMDGPU/SOPInstructions.td:2362-2379 @274ab686) and SOPP 0x014/0x015 (:2988-2989). Agree. |
| Operand forms | ID from an inline constant or M0[4:0]. S_BARRIER_WAIT takes only an inline constant. WG/trap barriers only via inline constant | CDNA5 ISA §5.6.3 (p. 47) | **Conflict:** §15.3 (p. 259-260) says the M0 operand form is "reserved for other architectures", yet LLVM accepts `s_barrier_signal m0` etc. for 12.50 (LLVM llvm/test/MC/AMDGPU/gfx12_asm_sop1.s:3,717-727 @274ab686) |
| M0 member count | ISA: named-barrier member count in M0[22:16] (7 bits); ID in M0[4:0] | CDNA5 ISA §5.6.3 Table 22 (p. 48); §15.3 (p. 259) | **Conflict:** LLVM puts the ID in M0[5:0] and the count in a 6-bit field at bit 16 (`getBarrierMemberCountBitWidth` = 6 for major < 13), LLVM llvm/lib/Target/AMDGPU/SIISelLowering.cpp:13169-13180 @274ab686, LLVM llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp:150-152 @274ab686. "derived": the two agree for counts ≤ 63 and IDs ≤ 16. |
| Wait mapping | S_BARRIER_WAIT on any positive ID waits on the currently **joined** named barrier. ID 0, or a positive ID with nothing joined, is a NOP. Negative IDs select WG/trap/cluster/cluster-trap completion bits | CDNA5 ISA §5.6.3 (p. 48); §15.5 S_BARRIER_WAIT (p. 286-287) | |
| `s_barrier_signal_isfirst` | WG barrier: SCC=1 for the first signaler, 0 for others, returned via KMcnt. Read SCC only after S_BARRIER_WAIT. M0 form not allowed | CDNA5 ISA §5.6.3 Table 22 (p. 48) | LLVM accepts `s_barrier_signal_isfirst m0` in the assembler (LLVM llvm/test/MC/AMDGPU/gfx12_asm_sop1.s:726-727 @274ab686), which conflicts with p. 48 |
| isfirst on cluster barriers | §5.6.6 (p. 51): writes SCC=1 only for the first workgroup and leaves others untouched, so the user must zero SCC first. §15.3 (p. 260): illegal with the cluster-user barrier (−3) | CDNA5 ISA §5.6.6 (p. 51); §15.3 (p. 260); §5.6.6.1 (p. 52) | **ISA-internal conflict.** LLVM follows p. 260 and rejects isfirst on −3, LLVM llvm/lib/Target/AMDGPU/AMDGPULowerIntrinsics.cpp:189-196 @274ab686. The CWSR flow uses isfirst only on trap barriers (p. 52). Rule (derived): no isfirst on −3. |
| Named barriers | 0-16 per WG, allocated at dispatch in groups of 4 from 64 per WGP. A wave joins at most one at a time. Member count set by S_BARRIER_INIT or signal-with-M0. JOIN does not change counts; LEAVE decrements the member count and returns SCC (KMcnt) | CDNA5 ISA §5.6 (p. 46); §5.6.2 (p. 47); §5.6.5.2-5.6.5.5 (p. 49-50) | Out-of-range IDs are ignored (p. 50). The count of allocated named barriers comes from the dispatch; the code-object field for it is not in this sparse checkout (Open). |
| S_BARRIER_LEAVE operand | ISA Table 22 lists a Bar# argument; the SOPP pseudocode uses the joined ID | CDNA5 ISA Table 22 (p. 48); §15.5 (p. 287) | LLVM gfx12 form takes no operand (`isGFX12Only`), LLVM llvm/lib/Target/AMDGPU/SOPInstructions.td:1684-1690 @274ab686; LLVM llvm/test/MC/AMDGPU/gfx12_asm_sopp.s:17-18 @274ab686 |
| S_WAKEUP_BARRIER | wakes sleeping waves associated with a named barrier; it does not release S_BARRIER_WAIT | CDNA5 ISA §5.6.3 (p. 48); §15.3 (p. 262) | gfx1250-only in LLVM (LLVM llvm/test/MC/AMDGPU/gfx1250_asm_sop1.s:74-84 @274ab686) |
| LDS (memory) barriers | DS_ATOMIC_BARRIER_ARRIVE_RTN_B64 (DScnt) and DS_ATOMIC_ASYNC_BARRIER_ARRIVE_B64 (ASYNCcnt, ordered after earlier async loads) | CDNA5 ISA §11.2 (p. 152) | LLVM `FeatureLdsBarrierArriveAtomic`, LLVM llvm/lib/Target/AMDGPU/GCNFeatures.td:2570 @274ab686. LLVM's pipe-bug workaround (`s_wait_alu` around it) applies to generation GFX12, LLVM llvm/lib/Target/AMDGPU/GCNSubtarget.h:1000-1004 @274ab686. Whether gfx1250 really needs it: unknown (no public source). |
| Barrier ≠ fence | a barrier does not order memory by itself. For LDS sharing, a release (`s_wait_dscnt 0` etc.) before signal and an acquire after wait are needed | derived from LLVM llvm/docs/AMDGPUUsage.md:15022-15038,15293-15310 @274ab686 | AMDGPUUsage says barriers are not yet documented in the GFX125x model (:14366) |
| Traps | a wave waiting in S_BARRIER_WAIT that traps re-executes the wait after return | CDNA5 ISA §5.6.4 (p. 49) | |

---

### 8. Workgroup clusters and multicast loads (prose-only)

```rust
// Prose-only (no P0 field).
// cluster: 1D/2D/3D, <= 16 workgroups, one WGP per workgroup, all workgroups the same size
// ids: IB_STS2.CLUSTER_ID[9:6] (0 = not in a cluster), IB_STS2.WG_IN_CLUSTER[24:21]
//      TTMP9 = cluster id X, TTMP7 = {Z[31:16], Y[15:0]},
//      TTMP6 = {flat_nwg-1[27:24], nwg_z-1[23:20], nwg_y-1[19:16], nwg_x-1[15:12], wg_z[11:8], wg_y[7:4], wg_x[3:0]}
// cluster barrier: one signal per *workgroup* on -3, every wave waits on -3
// multicast: cluster_load_b{32,64,128} (LOADcnt), cluster_load_async_to_lds_b{8,32,64,128} (ASYNCcnt)
//            M0[15:0] = workgroup mask (0 => plain load), M0[16] = early timeout
```

| fact | value | source | notes |
|---|---|---|---|
| Cluster definition | workgroups co-scheduled on one SE. 1D/2D/3D, up to 16 workgroups, each on its own WGP. All workgroups and clusters in a dispatch have the same size. Hierarchy: grid → cluster → workgroup → wave → item | CDNA5 ISA §2.3 (p. 10) | LLVM `FeatureClusters` in 12.50, LLVM llvm/lib/Target/AMDGPU/GCNFeatures.td:1657-1660,2575 @274ab686 |
| Single-WG cluster | §2.3 (p. 10): not considered a cluster; cluster-barrier ops become NOP when not in a cluster | CDNA5 ISA §2.3 (p. 10) | **ISA-internal conflict:** the §3.5.3.1 table (p. 30) shows a 1-workgroup cluster with ClusterID ≥ 1 and cluster barriers available. Open. |
| Cluster ID state | IB_STS2.CLUSTER_ID bits 9:6 (0 = none, 1-15 = physical id, not CWSR invariant); IB_STS2.WG_IN_CLUSTER bits 24:21 (0..15, CWSR invariant) | CDNA5 ISA §2.3.1 (p. 11); §3.4.9 (p. 25) | LLVM reads IB_STS2 offset 6 width 4 (cluster id) and offset 21 width 4 (flat WG id in cluster), LLVM llvm/lib/Target/AMDGPU/SIISelLowering.cpp:2811-2816,11509-11511 @274ab686 |
| Grid position | TTMP9 = cluster X, TTMP7[15:0] = Y, TTMP7[31:16] = Z (in workgroup units); TTMP6 holds per-dimension counts−1 and positions as in the sketch | CDNA5 ISA §2.3.1 (p. 11) | LLVM uses the same TTMP6 masks, LLVM llvm/lib/Target/AMDGPU/SIISelLowering.cpp:2834-2857 @274ab686. With clusters, the TTMP9/7 "workgroup id" is the **cluster id**, and the real WG id = cluster_id × (max+1) + wg_in_cluster: :2787-2802 |
| Compile-time cluster dims | attribute `amdgpu-cluster-dims` = "x,y,z" (0,0,0 = off; 1024,1024,1024 = on but unknown); metadata `.cluster_dims` | LLVM llvm/docs/AMDGPUUsage.md:3481-3487,7072-7075 @274ab686 | |
| Cluster barriers | user (−3) and trap (−4). The member count is the number of workgroups and decrements when a WG terminates. Recommended: WG barrier first, then one wave per WG signals −3, then all waves wait on −3. Signal cannot set the member count | CDNA5 ISA §5.6.6 (p. 51-52) | LLVM `s_cluster_barrier` lowering does exactly this, using WG `isfirst` to pick the signaling wave, LLVM llvm/lib/Target/AMDGPU/AMDGPULowerIntrinsics.cpp:136-176 @274ab686 |
| Cluster barrier state | valid (1 bit), member count (5), signal count (5), trap signal count (5). Per wave: cluster-done and cluster-trap-done bits in STATE_PRIV | CDNA5 ISA §5.6.6 Tables 23-24 (p. 51) | Readable through `s_sendmsg_rtn_b32/b64 sendmsg(MSG_RTN_GET_CLUSTER_BARRIER_STATE)` (msg 136), LLVM llvm/lib/Target/AMDGPU/SIDefines.h:700 @274ab686; LLVM llvm/test/MC/AMDGPU/gfx1250_asm_sop1.s:50-56 @274ab686 |
| Multicast loads | one wave per WG issues the same request with the same M0 mask. Data goes to every requesting WG when all have asked or on timeout. Bypasses (force-misses) WGP$ regardless of scope | CDNA5 ISA §10.7 (p. 134-135) | |
| Multicast M0 | M0[15:0] = WG-in-cluster mask (bit i = WG i receives the data); M0[16] = early timeout; mask 0 means a normal load returning only to the requester | CDNA5 ISA §10.7 (p. 135); §3.4.4 M0 table (p. 23) | |
| Multicast instructions and counters | CLUSTER_LOAD_B32/64/128 → LOADcnt (to VGPRs); CLUSTER_LOAD_ASYNC_TO_LDS_* → ASYNCcnt (to LDS). Only the requester's counter decrements, and others must learn of completion via barrier or atomics. Outside a cluster they become global loads | CDNA5 ISA §10.7 (p. 135) | The §10.7 table lists ASYNC widths B32/64/128, but the opcode list (§14.7, p. 206) and LLVM also have **B8** (LLVM llvm/test/MC/AMDGPU/gfx1250_asm_vflat.s:3388-3410 @274ab686). Feature `FeatureMcastLoadInsts`, LLVM llvm/lib/Target/AMDGPU/GCNFeatures.td:2577 @274ab686 |
| Cluster TDM | the TDM descriptor carries a workgroup mask for multicast; it must be zero outside a cluster | CDNA5 ISA §10.11 (p. 140) | Details belong to the TDM section (other part) |
| Cluster memory scope | `cluster` syncscope → SCOPE_SE; acquire at cluster scope uses `global_inv` SE | LLVM llvm/docs/AMDGPUUsage.md:14527-14528 @274ab686; LLVM llvm/lib/Target/AMDGPU/SIMemoryLegalizer.cpp:2073-2075,2313-2315 @274ab686 | "derived": a cluster lives in one SE (ISA p. 10), so SE scope is sufficient |

---

### Open

1. **XCNT in the ISA counter table.** §5.7.1 (p. 54) omits XCNT, and the only width evidence is §3.1 (p. 13) and S_WAIT_XCNT (p. 293). Also, what increments XCNT exactly (LDS-only DS ops? TDM? async?). A public answer could come from a later ISA revision. An `llvm-mc` probe cannot settle hardware behaviour.
2. **Implicit XCNT drains and the SMEM/VMEM interleave.** LLVM assumes these (SIInsertWaitcnts.cpp:962-971; SIInstrInfo.cpp:3709-3736); the ISA is silent. Until confirmed, emit explicit `s_wait_xcnt` (derived conservative choice).
3. **ASYNCcnt load-vs-store ordering.** The ISA (p. 135-136) says loads vs stores are out of order. LLVM uses one event class. Our pass should use two classes; whether LLVM should too is open.
4. **Combined-wait field width.** SIMM16[15:8] (Table 25, p. 53) vs [13:8] (p. 293-294). LLVM uses shift 8, width 6. Probe: `llvm-mc -triple=amdgpu12.50 -show-encoding` on `s_wait_loadcnt_dscnt` with values >63 in the high byte (e.g. with the rocm-10.1 assembler), and compare the disassembly.
5. **Barrier M0 operand.** §5.6.3 (p. 47-48) allows the M0 form; §15.3 (p. 259-260) says it is reserved for other architectures. Member count is M0[22:16] (ISA, 7 bits) vs 6 bits (LLVM). An `llvm-mc` probe shows only that the assembler accepts the form, not that hardware supports it.
6. **`s_barrier_signal_isfirst` on clusters.** p. 51 (SCC set only for the first WG) vs p. 260 (illegal on −3). Is isfirst on −4 allowed (CWSR flow, p. 52)? We follow LLVM: never use it on −3.
7. **Single-workgroup clusters.** p. 10 (not a cluster) vs §3.5.3.1 table (p. 30: ClusterID ≥ 1, cluster barriers present).
8. **S_DELAY_ALU inside clauses.** p. 43 (legal, not first) vs p. 58 and p. 285 (do not use / illegal).
9. **S_WAIT_ALU semantics.** Opcode only in the ISA. Field layout comes from LLVM AsmUtils; the meaning of va_vdst, va_sdst, vm_vsrc, sa_sdst, va_ssrc, va_vcc and hold_cnt on CDNA5 is unknown (no public source). Also unknown: SCHED_MODE[1:0]=2 (LLVM expert mode) vs ISA SCHED_MODE bit 2 (DISABLE_XDL_ARB_STALL, p. 56).
10. **Wait before S_GETREG.** ISA: wait-idle and SCRATCH_BASE included (p. 56). LLVM: `s_wait_alu 0` without SCRATCH_BASE. Use `s_wait_idle` (derived conservative choice).
11. **TDM drain (`s_wait_tensorcnt 10`).** LLVM-only (FeatureNeedsTDMDrain); the hardware reason is unknown (no public source).
12. **TENSORcnt / ASYNCcnt limits in LLVM.** `HardwareLimits::get` returns 0 for both (AMDGPUWaitcntUtils.cpp:82-108). Harmless today because LLVM does not auto-wait them, but it should be confirmed before relying on LLVM's limits.
13. **`s_wait_xcnt 0` before S_PREFETCH_*** (ISA p. 112). Not found in the inspected LLVM files; check SIInsertWaitcnts / GCNHazardRecognizer for prefetch handling in a fuller checkout.
14. **Named-barrier count in the kernel descriptor / dispatch** (how a kernel requests 4/8/12/16 named barriers) is not in this sparse checkout. AMDGPUUsage kernel-descriptor tables in a full checkout may settle it.
15. **DS_ATOMIC_ASYNC_BARRIER_ARRIVE_B64 workaround.** LLVM applies `s_wait_alu` around it to all generation-GFX12 targets (GCNSubtarget.h:1000-1004). Whether gfx1250 needs it is unknown (no public source).
16. **AMDGPUUsage GFX125x memory-model gaps.** The section itself says multicast loads, barriers, NV sequences and cooperative atomics are not covered (AMDGPUUsage.md:14362-14367). A future LLVM revision is the only public source.

## 5. P3_hazards_latency — gfx1250 hazard rules, latency model, co-execution, VOPD / VOPD3

Public sources only:
- **ISA**: AMD "CDNA5" Instruction Set Architecture Reference Guide (27-July-2026). Cited as `CDNA5 ISA §x (p. N)`, where N is the printed page number.
- **LLVM**: upstream llvm-project @274ab686 (2026-11-09). Cited as `LLVM <path>:<line> @274ab686`. Paths are relative to `llvm/lib/Target/AMDGPU/` unless they start with `llvm/`.

Conventions:
- "derived" marks our own inference from the cited facts.
- "unknown (no public source)" means neither source settles the point.
- "agree" / "disagree" / "ISA silent" / "LLVM silent" compare the two sources on that row.
- ISA content is paraphrased. Tables restate facts as compact rules and do not copy the ISA tables.

Field names follow `target-design.md` §2: `HazardRule { name, producer, consumer, role, waits: Waits { count, unit }, scope, source }`, `Count = N | Fn | Instr`, `Unit = WaitStates | IndependentValu`, and `LatencyModel`. VOPD and co-execution are prose only (target-design.md §4).

---

### 0. Which LLVM hazard machinery is live on gfx1250

```rust
// gfx1250 (LLVM processor "gfx1250", sched model GFX1250SpeedModel)
// generation = GFX12, plus FeatureNoDataDepHazard => the classic s_nop wait-state checks are skipped.
// Live: WMMA / TRANS co-execution, VPERM_PK16, VCMPX->PERMLANE, getreg wait-idle, scratch-base
//       forwarding, setreg(MODE) v_nops, DS async-barrier-arrive, TDM drain, SMEM soft clause (xnack).
```

| fact | value | source | notes |
|---|---|---|---|
| gfx1250 is in the GFX12 generation | `FeatureISAVersion12_50` = `_Common` (which starts with `FeatureGFX12`) + extras | LLVM GCNFeatures.td:2494-2496, 2624-2661 @274ab686 | Predicates such as `getGeneration() == GFX12` therefore include gfx1250. |
| GFX12 carries `FeatureNoDataDepHazard` | "does not need SW waitstates" | LLVM GCNFeatures.td:1275-1278, 1896 @274ab686 | — |
| Effect of NoDataDepHazard | `PreEmitNoopsCommon` / `getHazardType` return before the VMEM, VALU, DPP, DivFMas, RWLane, MAI, getreg, setreg, RFE, M0 and permlane-forwarding checks | LLVM GCNHazardRecognizer.cpp:574-575, 734-735 @274ab686 | So `getSetRegWaitStates()` = 2 (GCNSubtarget.h:301-303) is **not** applied on gfx1250. ISA agrees in spirit: hardware resolves most non-memory dependencies, and explicit wait states are rarely needed, CDNA5 ISA §5.7 (p. 52). |
| Checks that run before that early return | SMRD soft clause; NSA→VMEM bug (not gfx1250); FP-atomic→denorm (GFX10 only); WMMA co-execution (scheduler mode only) | LLVM GCNHazardRecognizer.cpp:557-572, 726-732 @274ab686 | — |
| `fixHazards` (always runs post-RA) | Calls each `fix*` and gates each one on a subtarget feature | LLVM GCNHazardRecognizer.cpp:1945-1975 @274ab686 | §1.3 lists the fixes that are live on gfx1250. |
| gfx1250 hazard features | `VcmpxPermlaneHazard`, `VPermPk16Hazard`, `VOPD3F64OPYSrcHazard`, `SetregVGPRMSBFixup`, `TransCoexecutionHazard`, `GFX1250VALUBlockingCycles`, `WMMACoexecutionHazards`, `RequiresInitialUnclausedVmem`, `NeedsTDMDrain`, `VOP3PX2IncrementsVaVdstTwice`, `SupportsXNACK` | LLVM GCNFeatures.td:2526, 2536, 2626, 2634-2635, 2649-2654, 2659 @274ab686 | `VOPDInterlockHazard` (GCNFeatures.td:1292-1296) and `VALUTransUseHazard` (GCNFeatures.td:1298-1301, used by GFX11 at :2341, :2355) are **not** in the gfx1250 list. |
| Window model restricted to gfx1250 | `hasCoExecWindowModel()` is true when WMMA and TRANS co-execution hazards are both set **and** the target is gfx1250. gfx1251 and gfx12.5-generic are excluded because their WMMA latencies differ. | LLVM GCNHazardRecognizer.cpp:204-219 @274ab686 | — |

---

### 1. `hazards: &'static [HazardRule]`

#### 1.1 Sketch

```rust
// Units: IndependentValu = V_NOP or a VALU that does not touch the hazard registers
//        (LLVM counts every VALU-pipe op, WMMAs included).
//        WaitStates = any issued instruction / s_nop.
pub static GFX1250_HAZARDS: &[HazardRule] = &[
  HazardRule { name: "wmma_to_wmma_raw",     producer: XdlWmma,   consumer: XdlWmma,
               role: Operand::ConsumerAorBorIndex_overlaps_ProducerD,
               waits: Waits { count: Count::Fn(wmma_wmma_slots /* §1.2: 2,3,4|5,5,9 */), unit: IndependentValu },
               scope: Physical, source: "ISA §7.12.1 p.98; LLVM checkWMMACoexecutionHazards" },
  HazardRule { name: "wmma_to_valu",         producer: XdlWmma,   consumer: Valu /*non-WMMA, non-LDS-DMA*/,
               role: Operand::RAW_D | WAW_D | WAR_AorBorIndex /* LLVM adds C */,
               waits: Waits { count: Count::Fn(wmma_valu_slots /* §1.2: 1,2,3|4,4,8 */), unit: IndependentValu },
               scope: Physical, source: "ISA §7.12.1 p.98-99; LLVM checkWMMACoexecutionHazards" },
  HazardRule { name: "wmma_to_wmma_nocoexec", producer: Wmma|Swmmac, consumer: Wmma|Swmmac,
               role: Operand::ConsumerAorB_or_SwmmacIndex_overlaps_ProducerD,
               waits: Waits { count: Count::N(1), unit: IndependentValu }, scope: Physical,
               source: "ISA §7.12.1 p.98 (co-exec disabled); LLVM fixWMMAHazards" },
  HazardRule { name: "trans_to_valu",        producer: Trans,     consumer: Valu /*non-trans*/,
               role: Operand::RAW_TransDst | WAR_TransSrc,
               waits: Waits { count: Count::N(1), unit: IndependentValu }, scope: Physical,
               source: "ISA §5.7.2 p.56; LLVM fixVALUTransCoexecutionHazards" },
  HazardRule { name: "vcmpx_to_permlane",    producer: VCmpxWritesExec, consumer: Permlane,
               role: Operand::Exec, waits: Waits { count: Count::N(1), unit: IndependentValu /*LLVM: not V_NOP*/ },
               scope: Physical, source: "ISA §7.2.7 p.75; LLVM fixVcmpxPermlaneHazards" },
  HazardRule { name: "vperm_pk16_follower",  producer: VPermPk16, consumer: NextInstr,
               role: Operand::Any, waits: Waits { count: Count::N(1) /*immediately next must be a safe VALU*/, unit: IndependentValu },
               scope: Physical, source: "LLVM fixVPermPk16Hazard (ISA silent)" },
  HazardRule { name: "getreg_wait_idle",     producer: Any, consumer: SGetreg(STATUS|STATE_PRIV|EXCP_FLAG_*|SCRATCH_BASE*),
               role: Operand::HwReg, waits: Waits { count: Count::Instr("s_wait_alu 0x0" /*ISA: wait-idle*/), unit: WaitStates },
               scope: Physical, source: "ISA §5.7.2 p.56; LLVM fixGetRegWaitIdle" },
  HazardRule { name: "sethalt_setprio_to_getreg", producer: SSethalt|SSetprio, consumer: SGetreg(STATE_PRIV|STATUS),
               role: Operand::HwReg, waits: Waits { count: Count::N(1), unit: WaitStates }, scope: Physical,
               source: "ISA §5.7.2 p.56 (LLVM silent)" },
  HazardRule { name: "setreg_mode_to_set_vgpr_msb", producer: SSetreg(MODE), consumer: SSetVgprMsb,
               role: Operand::Mode, waits: Waits { count: Count::N(1), unit: WaitStates }, scope: Physical,
               source: "ISA §5.7.2 p.56; LLVM AMDGPULowerVGPREncoding" },
  HazardRule { name: "setreg_mode_vnops",    producer: AnyValu, consumer: SSetreg(MODE),
               role: Operand::Mode, waits: Waits { count: Count::N(2), unit: IndependentValu /*v_nop x2*/ },
               scope: Physical, source: "LLVM fixSetRegMode (ISA silent)" },
  HazardRule { name: "scratch_base_forwarding", producer: Salu|Valu writes s102|s103,
               consumer: reads src_flat_scratch_base_lo|hi or s_getreg FLAT_SCR_LO|HI, role: Operand::Sgpr,
               waits: Waits { count: Count::Instr("s_wait_alu depctr_sa_sdst(0) depctr_va_sdst(0)") /*within 10 SGPR writes*/, unit: WaitStates },
               scope: Physical, source: "LLVM fixScratchBaseForwardingHazard (ISA silent)" },
  HazardRule { name: "ds_async_barrier_arrive_b64", producer: Any, consumer: DsAtomicAsyncBarrierArriveB64,
               role: Operand::Any, waits: Waits { count: Count::Instr("s_wait_alu depctr_vm_vsrc(0) before AND after"), unit: WaitStates },
               scope: Physical, source: "LLVM fixDsAtomicAsyncBarrierArriveB64 (ISA silent)" },
  HazardRule { name: "tdm_drain",            producer: TensorOp, consumer: TensorOp,
               role: Operand::Counter(TENSORcnt), waits: Waits { count: Count::Instr("s_wait_tensorcnt 10"), unit: WaitStates },
               scope: Physical, source: "LLVM fixTDM (ISA silent)" },
  HazardRule { name: "smem_soft_clause_xnack", producer: Smem, consumer: Smem /*same group, defs∩uses or store*/,
               role: Operand::Sgpr, waits: Waits { count: Count::N(1), unit: WaitStates }, scope: Physical,
               source: "LLVM checkSoftClauseHazards (ISA silent)" },
  HazardRule { name: "xnack_regs_wait_idle", producer: Any, consumer: Access(xnack_state_priv|xnack_mask),
               role: Operand::HwReg, waits: Waits { count: Count::Instr("s_wait_idle"), unit: WaitStates }, scope: Physical,
               source: "ISA §5.7.2 p.56 (LLVM silent)" },
  HazardRule { name: "entry_unclaused_vmem", producer: KernelEntry, consumer: FirstInstr,
               role: Operand::Any, waits: Waits { count: Count::Instr("s_mov_b64 s[64:65],0; v_nop; global_prefetch_b8 v0, s[64:65] scope:SE th:RT"), unit: WaitStates },
               scope: Physical, source: "LLVM SIInsertWaitcnts (ISA silent)" },
];
```

#### 1.2 WMMA co-execution categories

LLVM puts each XDL WMMA / SWMMAC producer into a category by its scheduling latency (4 / 8 / 16 / 32). Each category has two counts: the number of V_NOPs or independent VALUs required before a dependent **WMMA**, and before a dependent **VALU**. On gfx1250 the ISA table gives the WMMA count as "1 NOP + k co-execution slots" and the VALU count as "k co-execution slots".

| fact | value | source | notes |
|---|---|---|---|
| Category from latency | 4 → cat 6; 8 → cat 0 (WMMA) / 2 (SWMMAC); 16 → cat 1 (WMMA) / 3 (SWMMAC); cats 4, 5 are gfx1251/generic only | LLVM GCNHazardRecognizer.cpp:2717-2787 @274ab686 | Latency comes from the sched model (§2). |
| LLVM required counts (WMMA consumer) | cat 0..6 = {5, 9, 3, 5, 9, 17, 2} | LLVM GCNHazardRecognizer.cpp:2802 @274ab686 | — |
| LLVM required counts (VALU consumer) | cat 0..6 = {4, 8, 2, 4, 8, 16, 1} | LLVM GCNHazardRecognizer.cpp:2803 @274ab686 | WMMA count = VALU count + 1 in every category, consistent with the ISA "1 NOP + k slots" form. |
| Dense F16 / BF16 WMMA (16x16x32) | ISA: WMMA 5 (1+4), VALU 4. LLVM: latency 8 → cat 0 → 5 / 4. | CDNA5 ISA §7.12.1 (p. 98); LLVM SISchedule.td:570 @274ab686 | **agree** |
| Dense FP8/BF8 16x16x64, and F8F6F4 with both A and B in F4 | ISA: 2 (1+1) / 1. LLVM: latency 4 → cat 6 → 2 / 1. | CDNA5 ISA §7.12.1 (p. 98); LLVM SISchedule.td:497-500, 491-495, 569 @274ab686 | **agree** |
| Dense FP8/BF8 16x16x128; F8F6F4 not both F4; F4 32x16x128 | ISA: 4 (1+3) / 3. LLVM: latency 8 → cat 0 → **5 / 4**. | CDNA5 ISA §7.12.1 (pp. 98-99); LLVM SISchedule.td:571, 573-574 @274ab686 | **disagree**: LLVM is one higher (conservative). LLVM's own slot pattern for these (`0EEIEEIIVV`, 3 I-slots) matches the ISA's 3, AMDGPUCoExecInfo.h:408-430 (derived). |
| Dense IU8 WMMA (16x16x64) | ISA: 9 (1+8) / 8. LLVM: latency 16 → cat 1 → 9 / 8. | CDNA5 ISA §7.12.1 (pp. 98-99); LLVM SISchedule.td:572 @274ab686 | **agree** |
| SWMMAC F16 / BF16 / FP8 / BF8 | ISA: 3 (1+2) / 2. LLVM: latency 8 → cat 2 → 3 / 2. | CDNA5 ISA §7.12.1 (pp. 98-99); LLVM SISchedule.td:570-571 @274ab686 | **agree** |
| SWMMAC IU8 | ISA: 5 (1+4) / 4. LLVM: latency 16 → cat 3 → 5 / 4. | CDNA5 ISA §7.12.1 (pp. 98-99); LLVM SISchedule.td:572 @274ab686 | **agree** |
| gfx1250-strict (not gfx1250) | 16x16x64 FP8 → latency 8; F8F6F4 with any F8 input → 16 | LLVM SISchedule.td:476-500 @274ab686 | The categories then shift, because they are computed from latency. Out of scope, noted only. |
| "If co-exec is disabled" | WMMA consumer: 1 V_NOP; VALU consumer: none | CDNA5 ISA §7.12.1 (pp. 98-99) | LLVM always applies the co-exec counts when `WMMACoexecutionHazards` is set, plus the separate 1-V_NOP rule (§1.3, `wmma_to_wmma_nocoexec`). |
| Non-XDL WMMA (`v_wmma_f32_16x16x4_f32`) | Not "XDL" in LLVM (`is_wmma_xdl` is set only for the block in VOP3PInstructions.td:2260-2316), so it is excluded from the co-exec tables | LLVM VOP3PInstructions.td:2318-2319; SIInstrInfo.cpp:12077-12085 @274ab686 | The ISA table names only XDL WMMA/SWMMAC. For F32/F64 WMMA: **unknown (no public source)** (Open). |

#### 1.3 HazardRule rows

| fact | value | source | notes |
|---|---|---|---|
| **wmma_to_wmma_raw** | producer: XDL WMMA/SWMMAC (cat c). consumer: XDL WMMA/SWMMAC. role: consumer A or B (or SWMMAC Index) overlaps producer D, partial overlap included. waits: Fn(c), unit IndependentValu. scope: Physical (post-RA register overlap). | CDNA5 ISA §7.12.1 (p. 98); LLVM GCNHazardRecognizer.cpp:2789-2840 @274ab686 | Both sources. Counts per §1.2 (disagree on one group). C overlapping the previous D is **not** a hazard: hardware stalls, LLVM GCNHazardRecognizer.cpp:2684-2685; ISA lists only A/B/Index (derived agree). |
| **wmma_to_valu** | producer: XDL WMMA/SWMMAC. consumer: VALU that is not WMMA/SWMMAC and not LDS-DMA (`isCoexecutableVALUInst`). role: RAW (VALU source ∩ D), WAW (VALU dst ∩ D), WAR (VALU dst ∩ A/B/Index). waits: Fn(c), IndependentValu. | CDNA5 ISA §7.12.1 (pp. 98-99); LLVM GCNHazardRecognizer.cpp:2712-2715, 2842-2868 @274ab686 | Both. LLVM's WAR set also includes WMMA src2 (C), which is conservative. **LLVM superset.** |
| Distance counting (both rows above) | Only VALU-pipe instructions count (V_NOP or VALU). LLVM keeps a separate list of emitted VALU/WMMA instructions (up to 18) and counts entries; s_nop does **not** help. | CDNA5 ISA §7.12.1 (p. 98); LLVM GCNHazardRecognizer.h:65-73; GCNHazardRecognizer.cpp:825-838, 1219-1249 @274ab686 | → `Unit::IndependentValu`. LLVM counts an intervening WMMA as a filler (`IsVALUOrWMMA`). Whether the ISA's "unrelated VALU" includes WMMA: Open. |
| Mitigation emitted | V_NOPs before the consumer (`emitVNops`). Optionally hoisted to a loop preheader when no hazard source exists inside the loop (`wmma_vnop_hoisting` option). In scheduler mode it is reported as `Hazard` (not `NoopHazard`) so the scheduler can fill the gap. | LLVM GCNHazardRecognizer.cpp:566-572, 1937-1943, 2886-2956 @274ab686 | LLVM only (implementation choice). |
| **wmma_to_wmma_nocoexec** | producer: any WMMA/SWMMAC. consumer: WMMA/SWMMAC whose A or B (SWMMAC: also Index) overlaps producer D. waits: N(1), unit IndependentValu (expired by any VALU). Emits one V_NOP. | LLVM GCNHazardRecognizer.cpp:2658-2710 @274ab686; CDNA5 ISA §7.12.1 (p. 98) | Matches the ISA "co-exec disabled: 1 V_NOP" rows. LLVM runs it unconditionally; it is subsumed by `wmma_to_wmma_raw` when co-exec is on (derived). |
| **trans_to_valu** | producer: TRANS (incl. pseudo-scalar trans `v_s_*`, which are `TRANS = 1`). consumer: non-TRANS VALU. role: RAW (VALU reads trans vdst) or WAR (VALU writes a trans source). waits: N(1), IndependentValu (any VALU in between expires it). Emits one V_NOP. | CDNA5 ISA §5.7.2 (p. 56); LLVM GCNHazardRecognizer.cpp:2616-2656; VOP3Instructions.td:2074, 2086 @274ab686 | **agree** (ISA: one independent op or V_NOP, because TRANS takes 2 cycles). Trans→trans is not covered by this row: LLVM excludes TRANS consumers (line 2619). |
| **multicycle_coexec (generic)** | ISA: hardware does not check dependencies between co-executing multicycle instructions of one wave, so software covers them with V_NOPs or independent VALUs. | CDNA5 ISA §5.7.2 (p. 56) | LLVM has data-hazard rules only for WMMA and TRANS. Multicycle VALUs (F64 = 32 blocking cycles, §2) get **no** data-hazard rule, only the scheduler stall model (§3). **Possible gap / conflict**: Open. |
| **vcmpx_to_permlane** | producer: VOPC/VOP3 compare that writes EXEC (V_CMPX). consumer: any V_PERMLANE* (16, X16, 64, *_VAR, 16/32_SWAP, BCAST, UP, DOWN, XOR, IDX_GEN). waits: N(1), unit: a non-NOP VALU. LLVM inserts `v_mov_b32 vX, vX` (vX = permlane src0). | CDNA5 ISA §7.2.7 (p. 75); LLVM GCNHazardRecognizer.cpp:504-520, 1984-2017; GCNFeatures.td:2526 @274ab686 | **disagree on V_NOP**: the ISA names V_NOP as an acceptable filler, but LLVM says SQ discards V_NOP and does not count it (lines 1994-2007). Use LLVM's v_mov form (safe under both). |
| **vperm_pk16_follower** | producer: V_PERM_PK16_B4/B6/B8_U4. consumer: the next issued instruction (meta skipped). It must be a "safe" VALU: an XDL op (WMMA/SWMMAC/DOT), or a VALU with GFX1250 blocking cycles < 2, or a TRANS with 2. Pseudo-scalar trans and multicycle VALUs are not safe. Otherwise LLVM inserts V_NOP. | LLVM GCNHazardRecognizer.cpp:4390-4415; SIInstrInfo.h:1260-1288; GCNFeatures.td:1313-1316, 2536 @274ab686 | **ISA silent** (ISA only gives the operand rule: src0/src1 both VGPR or both SGPR, CDNA5 ISA §7.9 (p. 89)). LLVM also keeps `s_cbranch_execz` around it so EXEC ≠ 0 (SIInstrInfo.cpp:4963-4967) and relies on the entry prologue for the cross-wave part. |
| **entry_unclaused_vmem** | At a hardware entry point LLVM emits `s_mov_b64 s[64:65], 0`; `v_nop`; `global_prefetch_b8 v0, s[64:65] scope:SCOPE_SE th:TH_LOAD_RT` as the first VMEM, unclaused. Count: Instr. | LLVM SIInsertWaitcnts.cpp:3758-3774; GCNFeatures.td:1417-1420, 2653 @274ab686 | **ISA silent.** LLVM ties the V_PERM_PK16 cross-wave mitigation to this prologue (GCNHazardRecognizer.cpp:4391-4396). The asm parser also checks it (AsmParser/AMDGPUAsmParser.cpp:7051). |
| **getreg_wait_idle** | consumer: `s_getreg_b32` of STATUS, STATE_PRIV, EXCP_FLAG_PRIV, EXCP_FLAG_USER (ISA also lists SCRATCH_BASE). waits: Instr. LLVM emits `S_WAITCNT_DEPCTR 0` (printed `s_wait_alu 0`, i.e. every depctr field 0). | CDNA5 ISA §5.7.2 (p. 56); LLVM GCNHazardRecognizer.cpp:4417-4436; GCNSubtarget.h:990-992; llvm/test/MC/AMDGPU/gfx12_asm_sopp.s:242-243 @274ab686 | **Partial disagreement.** (1) SCRATCH_BASE is in the ISA list but not in LLVM's. (2) The ISA says "wait-idle"; `S_WAIT_IDLE` waits for all counters (CDNA5 ISA §15.5 (p. 286)), while LLVM uses only `s_wait_alu 0`. Open. |
| **sethalt_setprio_to_getreg** | `S_SETHALT` → `S_GETREG` (STATE_PRIV or STATUS) and `S_SETPRIO` → `S_GETREG` STATE_PRIV need one S_NOP or other instruction in between. waits: N(1), WaitStates. | CDNA5 ISA §5.7.2 (p. 56) | **LLVM silent** (no such rule in `fixHazards`, GCNHazardRecognizer.cpp:1945-1975). |
| **xnack_regs_wait_idle** | `S_WAIT_IDLE` before reading or writing xnack_state_priv / xnack_mask. Count: Instr. | CDNA5 ISA §5.7.2 (p. 56) | **LLVM silent.** Trap-handler territory; not needed by kernels (derived). |
| **setreg_mode_to_set_vgpr_msb** | `S_SETREG*` to MODE immediately followed by `S_SET_VGPR_MSB` may lose the MSB update. waits: N(1) of any instruction; LLVM uses `s_nop 0`. | CDNA5 ISA §5.7.2 (p. 56); LLVM AMDGPULowerVGPREncoding.cpp:558-563 @274ab686 | **agree.** LLVM also rewrites imm bits [12:19] of `s_setreg_imm32_b32 MODE` to keep the current VGPR MSBs (setreg clobbers them), AMDGPULowerVGPREncoding.cpp:499-556; GCNFeatures.td:1396-1399. ISA: MODE bits 12-19 hold the VGPR MSBs, CDNA5 ISA §3.4.3 (p. 22) (derived consistency). |
| **setreg_mode_vnops** | Two `V_NOP`s before `S_SETREG_B32` / `S_SETREG_IMM32_B32` that targets MODE. waits: N(2). | LLVM GCNHazardRecognizer.cpp:4529-4537; GCNSubtarget.h:640-642 @274ab686 | **ISA silent.** Derived caveat: the code compares the whole encoded simm16 operand with `ID_MODE`, so it seems to fire only for offset 0 / size 1. Verify with llc (Open). |
| **scratch_base_forwarding** | producer: SALU or VALU writing s102 or s103. consumer: an instruction reading `src_flat_scratch_base_lo/hi`, or `s_getreg` of FLAT_SCR_LO/HI. The window ends after 10 SGPR-writing SALU/VALU instructions, or at an `s_wait_alu` with sa_sdst = 0 and va_sdst = 0. Fix: `s_wait_alu depctr_sa_sdst(0) depctr_va_sdst(0)`. Count: Instr. | LLVM GCNHazardRecognizer.cpp:4453-4527; GCNSubtarget.h:1006-1010 @274ab686 | **ISA silent** about s102/s103. The ISA describes FLAT_SCRATCH_BASE as a constant derived from SCRATCH_BASE, CDNA5 ISA §6.10 (p. 66), §10.2 (p. 132). The LLVM comment says "8 SGPR writes by VALU", but the code uses 10 for both (line 4461-4462 vs 4508). Post-RA only. |
| **flat_scratch_hi_b64_salu** (verifier) | An SALU with a 64-bit SGPR destination (or `s_bitcmp0/1_b64`) must not read `src_flat_scratch_base_hi`. | LLVM SIInstrInfo.cpp:6331-6340; GCNSubtarget.h:1012-1016 @274ab686 | **Different rules.** ISA: HI as a 32-bit source is undefined, and LO/HI as a 64-bit source both give the full 64-bit value, CDNA5 ISA §6.10 (p. 66). Use only `_lo` in 64-bit contexts and never `_hi` as a 32-bit source (derived union). |
| **ds_async_barrier_arrive_b64** | `s_wait_alu depctr_vm_vsrc(0)` immediately before **and** after `DS_ATOMIC_ASYNC_BARRIER_ARRIVE_B64`. The LLVM comment says it must not be claused. Count: Instr. | LLVM GCNHazardRecognizer.cpp:4438-4451; GCNSubtarget.h:1000-1004 @274ab686 | **Tension**: the ISA lists DS_ATOMIC_ASYNC_BARRIER_ARRIVE in the Async clause type, CDNA5 ISA §5.3 (p. 42). Open. |
| **tdm_drain** | Before a TENSORcnt-using op (not `s_wait_tensorcnt`): if an earlier TDM op is reachable with no `s_wait_tensorcnt ≤ 10` in between, insert `s_wait_tensorcnt 10`. Count: Instr. | LLVM GCNHazardRecognizer.cpp:4539-4562; GCNFeatures.td:1683-1686, 2654 @274ab686 | **ISA silent** (TENSORcnt is 6 bits, CDNA5 ISA §5.7.1 (p. 54)). Cross-reference P2 counters. |
| **smem_soft_clause_xnack** | With XNACK on: in a run of consecutive SMEM, if any def in the group overlaps a use, or the new SMEM is a store, break the group with one wait state. | LLVM GCNHazardRecognizer.cpp:1303-1358 @274ab686 | **ISA silent** (it defines SMEM groups, CDNA5 ISA §8.3 (p. 111)). Runs before the NoDataDep early return, so it is live on gfx1250 (`FeatureSupportsXNACK`, GCNFeatures.td:2659). |
| **wmma_exec_all_ones** (constraint) | WMMA/SWMMAC require EXEC = all 1s; WMMA is never EXEC-skipped. | CDNA5 ISA §7.12 (p. 97); §3.2.2.1 (p. 14) | No hazard rule in LLVM GCNHazardRecognizer (LLVM silent here). |
| **wmma_reuse_bits** (constraint) | The A/B reuse hint may be 1 only when the current instruction equals the previous one; otherwise the result is undefined. | CDNA5 ISA §7.12 (p. 97) | **LLVM silent**: no check found in GCNHazardRecognizer.cpp; the intrinsic lowering passes 0 (AMDGPULowerIntrinsics.cpp:327-328). Our emitter should set reuse only for identical back-to-back WMMAs (derived). |

#### 1.4 Rules that are not active on gfx1250

| fact | value | source | notes |
|---|---|---|---|
| VALU→SGPR-read hazard (GFX12 `VALUReadSGPRHazard`) | Disabled for gfx1250 | LLVM GCNSubtarget.h:636-638 @274ab686 | Matches the ISA statement that hardware resolves non-memory dependencies, CDNA5 ISA §5.7 (p. 52). |
| s_nop before `s_sendmsg MSG_DEALLOC_VGPRS` | Not required on gfx1250 | LLVM GCNSubtarget.h:986-988 @274ab686 | ISA silent. |
| LDS-direct VALU / VMEM hazards | Not applicable: no LDSDIR instructions on gfx1250 (`isGFX12PlusNot12_50`) | LLVM DSDIRInstructions.td:144-159; GCNHazardRecognizer.cpp:1951-1954 @274ab686 | — |
| VALU partial-forwarding, VALU trans-use, VALU mask-write | GFX11 only | LLVM GCNSubtarget.h:614-616, 634; GCNFeatures.td:1298-1301 @274ab686 | — |
| Trans forwarding, dst-sel forwarding, permlane forwarding, cvt-scale forwarding | GFX940 / GFX950 only | LLVM GCNSubtarget.h:588-600, 620-622 @274ab686 | gfx950 rows of trust-inventory §5.1 do not carry over. |
| VMEM→scalar write, SMEM→vector write, vcmpx-exec WAR, LDS-branch-VMEM WAR | GFX10 features, not set; asserted off with extended wait counts | LLVM GCNFeatures.td:2243-2248; GCNHazardRecognizer.cpp:2019-2022, 2231-2232 @274ab686 | — |
| M0 read hazards (movrel, sendmsg, LDS DMA, LDS direct) | GFX9 only | LLVM GCNSubtarget.h:557-572 @274ab686 | — |
| Setreg / getreg / RFE wait states | Not applied (after the NoDataDep early return) | LLVM GCNHazardRecognizer.cpp:734-735, 760-767 @274ab686 | The ISA still says S_ROUND_MODE / S_DENORM_MODE avoid a setreg "wait state penalty", CDNA5 ISA §15.5 (p. 286). We read that as a performance stall, not a correctness rule (derived). |
| Required export priority | No exports on gfx1250 | LLVM GCNHazardRecognizer.cpp:4284-4301; GCNSubtarget.h:384-386 @274ab686 | — |
| VOPD interlock (parity) hazard | GFX11-style; not in the gfx1250 feature list | LLVM GCNFeatures.td:1292-1296; Utils/AMDGPUBaseInfo.h:668-671 @274ab686 | — |

---

### 2. `latency: LatencyModel`

#### 2.1 Sketch

```rust
latency: LatencyModel {
    // LLVM GFX1250SpeedModel: IssueWidth 1, MicroOpBufferSize 1, in-order model.
    pipes: &[
        // (pipe, latency_cycles, issue_occupancy_cycles)   occupancy = GFX1250 "blocking cycles"
        (Valu,  5, 1),        // Write32Bit / FloatCvt / FloatFMA
        (Valu,  6, 1),        // QuarterRate32;  Write64Bit = 6 (occ 2 for 64-bit int ops)
        (Valu,  8, 4),        // IntMul (v_mul_lo/hi_u32, v_mad_u32, v_mad_u64_u32)
        (Valu, 37, 32),       // F64 arithmetic (Double/DoubleAdd/DoubleCvt)
        (Trans, 8, 2),        // f32/f16/bf16 TRANS on HWTransVALU; pseudo-scalar trans 8 (on HWVALU)
        (Trans,38, 32),       // f64 rcp/rsq/sqrt (HWVALU + HWTransVALU)
        (Salu,  2, 1), (Sfpu /*SALU float*/, 4, 1),
        (Smem, 20, 1), (Lds, 20, 1), (Vmem, 320, 1),
        (Ctrl /*branch*/, 32, 1), (Barrier, 2000, 1),
    ],
    wmma: &[ // (group, latency = XDL occupancy, co-exec window pattern)
        ("16x16x64 fp8/bf8; f8f6f4 both-F4", 4,  "0EEIVV"),
        ("16x16x32 f16/bf16; swmmac f16/bf16 (16x16x64)", 8, "0EIIEEIIV" /*swmmac: permissive*/),
        ("16x16x128 fp8/bf8; swmmac fp8/bf8", 8, "0EEIEEIIVV"),
        ("16x16x128 f8f6f4 (not both F4)", 8, "0EEIEEIIVV" /*scaled: 0EEIEEISVV*/),
        ("32x16x128 f4", 8, "0EEIEIEIVV" /*scaled: 0EEIEIESVV*/),
        ("16x16x64 iu8; swmmac iu8", 16, "0EIIEEIIEEIIEEIIV" /*swmmac: permissive*/),
        ("16x16x4 f32 (non-XDL, on VALU)", 16, "-"),
    ],
    trans_latency: Rule::Fn(trans_dep /* ISA: 2-cycle issue => 1 independent VALU; sched: 8 */),
    wmma_dst_wait: Rule::Hazard("wmma_to_valu" /* §1.3 */),
    delay_alu: DelayAlu { valu_back_max: 4, trans_back_max: 3, salu_cycles_max: 3 },
}
```

#### 2.2 Facts

| fact | value | source | notes |
|---|---|---|---|
| Processor → sched model | `gfx1250` uses `GFX1250SpeedModel` | LLVM GCNProcessors.td:315 @274ab686 | — |
| Machine model | IssueWidth 1, MicroOpBufferSize 1, PostRAScheduler on | LLVM AMDGPUSchedModels.td:13-26, 36 @274ab686 | — |
| Resources | HWBranch, HWExport, HWLGKM, HWSALU, HWVMEM, HWVALU, HWTransVALU, HWRC, each 1 unit with buffer 1; HWXDL with buffer 0 | LLVM SISchedule.td:100-126 @274ab686 | HWXDL with buffer 0 means a WMMA blocks the next XDL op for `ReleaseAtCycles` (derived). |
| VALU 32-bit / cvt / FMA latency | 5 | LLVM SISchedule.td:530-531, 534 @274ab686 | ISA gives no VALU latency. ISA S_DELAY_ALU can describe up to 4 VALU back (p. 58). |
| Quarter-rate 32-bit | 6 | LLVM SISchedule.td:533 @274ab686 | — |
| 64-bit int (Write64Bit) | 6 | LLVM SISchedule.td:559 @274ab686 | Blocking 2 (v_add/sub_nc_u64, v_lshl*_b64, v_lshl_add_u64, v_mov_b64, i64/u64 compares): VOP2Instructions.td:946-948, 1518; VOP3Instructions.td:394, 825; VOP1Instructions.td:241-243; VOPCInstructions.td:816, 853 |
| Int multiply | 8, blocking 4 | LLVM SISchedule.td:560; VOP3Instructions.td:191, 216, 443 @274ab686 | `v_mul_u64` is WriteDouble with blocking 32 (VOP2Instructions.td:953-955). |
| F64 arithmetic | 37 (Double, DoubleAdd, DoubleCvt), blocking 32 | LLVM SISchedule.td:561-563; VOP2Instructions.td:1513-1514, 1525; VOP3Instructions.td:200, 232, 262, 328, 370, 383; VOPCInstructions.td:616, 1211 @274ab686 | ISA: f64 fma/add/mul/min/max are multicycle ("in certain products"), CDNA5 ISA §7.8 (p. 86). **agree** that F64 is multicycle; the ISA gives no cycle count. |
| TRANS 32-bit (f32/f16/bf16) | latency 8 on HWTransVALU, blocking 2 | LLVM SISchedule.td:532; VOP1Instructions.td:354, 370, 546 @274ab686 | ISA: TRANS takes 2 cycles, CDNA5 ISA §5.7.2 (p. 56). **agree** on issue occupancy 2. |
| TRANS 64-bit (rcp/rsq/sqrt f64) | latency 38 on HWVALU + HWTransVALU, blocking 32 | LLVM SISchedule.td:564; VOP1Instructions.td:363-368 @274ab686 | ISA: 64-bit trans counts as "VALU" for S_DELAY_ALU, CDNA5 ISA §5.8 (p. 58). |
| Pseudo-scalar trans (`v_s_exp/log/rcp/rsq/sqrt_f32/f16`) | latency 8 on HWVALU, blocking 2, TRANS = 1 | LLVM SISchedule.td:535; VOP3Instructions.td:2073-2090 @274ab686 | ISA: they run on the VALU pipe and ignore EXEC, CDNA5 ISA §7.10 (p. 93). |
| BF16 transcendental (`v_rcp/sqrt/rsq/log/exp/sin/cos_bf16`) | Read-modify-write; no co-execution support | CDNA5 ISA §15.8 (pp. 364-366); §15.14 (p. 535) | LLVM gives them the generic TRANS class (derived; not separately modelled). |
| Other VALU blocking (multicycle) | DPP16 / DPP8 forms: 2; v_swap*, v_swaprel, permlane16/32_swap: 2; dot IU: 4; v_perm_pk16_*: 4; qsad / mqsad: 4; cvt_scale pk8: 4; pk16: 8; scalef32 pk32: 16 | LLVM VOPInstructions.td:1043, 1262; VOP1Instructions.td:653, 899, 928, 950; VOP3PInstructions.td:848; VOP3Instructions.td:378, 436, 1471, 1481, 1495, 1927-1985, 2019 @274ab686 | Used only by the co-exec scheduling strategy: `classifyFlavor` marks MultiCycleVALU when blocking > 1 (AMDGPUCoExecSchedStrategy.cpp:404-408, 554-565; SIInstrInfo.cpp:11361-11372). The hazard recognizer's repeat rate comes from sched-model `ReleaseAtCycle` (SIInstrInfo.cpp:12052-12068), which is set only for XDL WMMA (derived). |
| XDL WMMA latency / occupancy | 1-pass 4, 2-pass 8, 4-pass 16 (8-pass 32: gfx1251 only); `ReleaseAtCycles` equals latency | LLVM SISchedule.td:517-524 @274ab686 | — |
| gfx1250 WMMA latency assignment | 16x16x64 FP8/BF8 → 4. F16/BF16 (WMMA and SWMMAC) → 8. 16x16x128 FP8/BF8 (WMMA and SWMMAC) → 8. IU8/IU4 → 16. 16x16x128 F8F6F4: both F4 → 4, else 8. 32x16x128 F4 → 8. | LLVM SISchedule.td:566-574, 491-500 @274ab686 | ISA gives no cycle counts, only co-exec slot counts (§1.2), plus a "16-cycle WMMA" example in CDNA5 ISA §5.7.2.1 (p. 56). Derived: k_valu + 1 = occupancy/2 for 8 and 16, and 1 for 4 (4 → 1, 8 → 4 or 3, 16 → 8). |
| Non-XDL WMMA `v_wmma_f32_16x16x4_f32` | latency 16 on HWVALU | LLVM SISchedule.td:526, 547 @274ab686 | — |
| SALU / SALU-float / SMEM / LDS / VMEM / branch / barrier | 2 / 4 / 20 / 20 / 320 / 32 / 2000 | LLVM SISchedule.td:537-544 @274ab686 | Model values, not measurements. ISA: SALU read of the shader clock about 8 cycles, CDNA5 ISA §3.4.11 (p. 28). `S_MUL_U64` / `S_MUL_*64` take 5 cycles (an SALU follower needs only an `INSTID_SALU_CYCLE_1` delay; a VALU follower may need S_NOP for performance), CDNA5 ISA §15.1 (p. 230). |
| ECC | With ECC, VALU writes of ≤ 16 bits run at half rate (RMW via V_MOV_PRSV_B32) | CDNA5 ISA §13.1.1, §13.1.6 (pp. 159-160) | Whether gfx1250 parts have VGPR ECC enabled: **unknown (no public source)**. LLVM silent. |
| s_nop range | `s_nop N` gives N+1 wait states, N in 0..127 | CDNA5 ISA §15.5 (p. 278) | LLVM emits at most `s_nop 7` per instruction (GCNHazardRecognizer.cpp:636-644). Either is legal. |
| S_DELAY_ALU (optional, performance only) | DEP codes: 1-4 = VALU 1-4 back; 5-7 = TRANS 1-3 back; 9-11 = SALU 1-3 cycles. SKIP 0-5. Two deps per instruction. A new S_DELAY_ALU replaces pending info. Not allowed inside an S_CLAUSE clause. | CDNA5 ISA §5.8 (pp. 57-58); §15.5 (pp. 283-285) | LLVM limits: VALU_MAX 5, TRANS_MAX 4, SALU_CYCLES_MAX 4 ("one larger than encodable"). AMDGPUInsertDelayAlu.cpp:87-98 @274ab686. **agree.** |
| XDL WMMA in S_DELAY_ALU | Tracked as TRANS. LD_SCALE + WMMA (VOP3PX2) counts as one instruction. | CDNA5 ISA §5.8 (p. 58) | LLVM: XDL WMMA → TRANS delay type, AMDGPUInsertDelayAlu.cpp:68-81. **agree.** LLVM also skips the delay for a back-to-back WMMA C-reuse chain (AMDGPUInsertDelayAlu.cpp:400-414). LLVM counts VOP3PX2 twice for the va_vdst counter (SIInsertWaitcnts.cpp:868-876; GCNFeatures.td:1679-1681). That is a different counter, so it does not conflict. |
| s_wait_alu (S_WAITCNT_DEPCTR) fields | sa_sdst [0], va_vcc [1], vm_vsrc [4:2], hold_cnt [7], va_ssrc [8], va_sdst [11:9], va_vdst [15:12] | LLVM Utils/AMDGPUBaseInfo.cpp:154-200; llvm/test/MC/AMDGPU/gfx12_asm_sopp.s:242-255 @274ab686 | ISA lists S_WAIT_ALU only as SOPP opcode 8, CDNA5 ISA §14.1.5 (p. 169). No field description: **ISA silent** (see P2). |

---

### 3. Co-execution rules (prose; no P0 field)

**ISA view.**
- A wave that issues a multicycle VALU op is normally held by the instruction arbiter until that op has executed. For example, a 16-cycle WMMA blocks the wave for 16 cycles, and other waves co-execute in the meantime. SCHED_MODE bit 2 (DISABLE_XDL_ARB_STALL) lets a wave issue WMMAs back to back and then run independent work, CDNA5 ISA §5.7.2.1 (p. 56).
- Hardware does not check dependencies between co-executing multicycle ops of one wave, so software covers them with V_NOP or independent VALU, CDNA5 ISA §5.7.2 (p. 56). §7.12.1 (pp. 98-99) gives the WMMA slot counts. TRANS needs 1 slot.
- S_DELAY_ALU treats XDL WMMA like TRANS so that normal VALU can co-execute with WMMA, CDNA5 ISA §5.8 (p. 58).
- BF16 trans ops do not co-execute, CDNA5 ISA §15.8 (p. 364).

**LLVM view (gfx1250 only, `hasCoExecWindowModel`).** Each WMMA opens a window with one stage per cycle, LLVM AMDGPUCoExecInfo.h:13-21, 45-70, 404-434 @274ab686:
- `0` (E0): issue cycle. Control only (`s_delay_alu`, `s_set_vgpr_msb`).
- `E`: SALU and memory (DS / VMEM / SMEM), no VALU.
- `I`: VALU, TRANS, SALU and memory.
- `S`: an `I` stage where the next scaled WMMA may also issue; its LD_SCALE uses the I cycle.
- `V`: SALU, memory or the next WMMA, but no VALU or TRANS.

Instruction classes come from `classifyFlavor`/`getCoExecMask` (AMDGPUCoExecInfo.h:131-158; AMDGPUCoExecSchedStrategy.cpp:374-424). LDS-DMA and tensor-DMA count as VALU, and waits / barriers count as SALU. In scheduler mode, a stage mismatch is a stall `Hazard`, not a V_NOP (GCNHazardRecognizer.cpp:331-368, 542-555). It is a throughput model, not a correctness rule. The data hazards in §1.3 are separate. Other scheduler-only stalls:
- **TRANS shadow**: after a TRANS, the next cycle cannot take another TRANS or a multicycle VALU (`CyclesUntilTRANS = 2`; GCNHazardRecognizer.cpp:251-268, 299-313).
- **Multicycle VALU shadow**: for repeat-rate − 1 cycles, only off-pipe ops issue (GCNHazardRecognizer.cpp:270-291, 315-329).
- **Multi-shadow**: a VALU caught under both a WMMA window and a TRANS shadow waits for both (GCNHazardRecognizer.cpp:370-394).

| fact | value | source | notes |
|---|---|---|---|
| IU8 16x16x64 window | occupancy 16, window 17: `0EIIEEIIEEIIEEIIV` (8 I-slots) | LLVM AMDGPUCoExecInfo.h:404-405 @274ab686 | 8 I-slots = ISA "8 co-exec slots", CDNA5 ISA §7.12.1 (p. 99). Derived **agree**. |
| F16/BF16 16x16x32 window | 8 / 9: `0EIIEEIIV` (4 I) | LLVM AMDGPUCoExecInfo.h:420-422 @274ab686 | ISA 4 slots (p. 98). **agree.** |
| FP8/BF8 16x16x64, both-F4 F8F6F4 | 4 / 6: `0EEIVV` (scaled both-F4: `0EESVV`) (1 I) | LLVM AMDGPUCoExecInfo.h:411-418 @274ab686 | ISA 1 slot (p. 98). **agree.** |
| FP8/BF8 16x16x128, F8F6F4 | 8 / 10: `0EEIEEIIVV` (scaled: `0EEIEEISVV`) (3 I) | LLVM AMDGPUCoExecInfo.h:407-409, 424-426 @274ab686 | ISA 3 slots (p. 99) agrees with the pattern. LLVM's hazard table uses 4 (§1.2). |
| F4 32x16x128 | 8 / 10: `0EEIEIEIVV` (scaled: `0EEIEIESVV`) | LLVM AMDGPUCoExecInfo.h:428-430 @274ab686 | ISA 3 slots. **agree.** |
| SWMMAC windows | FP8/BF8 SWMMAC reuse the 16x16x128 pattern (3 I). F16/BF16 and IU8 SWMMAC (gfx1250 w32) have no variant, so the window is permissive `AAAAAAAAA`. | LLVM VOP3PInstructions.td:2185-2200, 2193-2198; AMDGPUCoExecInfo.h:432-434 @274ab686 | ISA hazard slots for SWMMAC: 2 (F16/BF16/FP8), 4 (IU8). The pattern is a throughput hint only. |
| Window cap | `MaxCoExecStages` 32 | LLVM AMDGPUCoExecInfo.h:220-221 @274ab686 | — |
| Same-wave vs cross-wave co-exec | ISA: the default arbiter stall gives co-execution to *other* waves, while the hazard table counts same-wave slots and has a "co-exec disabled" column | CDNA5 ISA §5.7.2.1 (p. 56); §7.12.1 (pp. 98-99) | How these fit together, and which mode is the default for a lone wave: **unknown (no public source)** (Open). |
| EXEC-skipped VALU | VALUs may be skipped when EXEC = 0 (timing-dependent). V_NOP is never skipped. | CDNA5 ISA §3.2.2.1 (p. 14) | Derived: use V_NOP rather than "independent VALU" when EXEC may be 0. LLVM emits V_NOP for the WMMA / TRANS rules (GCNHazardRecognizer.cpp:1937-1943, 2654). |

---

### 4. VOPD / VOPD3 dual issue (prose; no P0 field)

**Availability.** VOPD exists on GFX12 (`FeatureVOPDInsts`, LLVM GCNFeatures.td:1889). VOPD3 exists only with gfx1250 instructions (`hasVOPD3`, LLVM GCNSubtarget.h:740). Both are wave32 only: CDNA5 ISA §7.8 (p. 86); LLVM GCNVOPDUtils.cpp:585-588.

**Pairing (LLVM).** The DAG mutation `VOPDPairingMutation` (GCNVOPDUtils.cpp:575-643) greedily makes eligible pairs adjacent. It refuses a pair when the fusion would serialize independent loads (GCNVOPDUtils.cpp:480-568). `tryMatchVOPDPair` tries VOPD first and then VOPD3 (GCNVOPDUtils.cpp:427-439). The second op may not RAW-depend on the first (GCNVOPDUtils.cpp:384-386). This matches the ISA rule that the two ops are independent (§7.8, p. 86).

| fact | value | source | notes |
|---|---|---|---|
| Opcodes | VOPD OPX: 12 ops (fmac, fmaak, fmamk, mul, add, sub, subrev, mul_dx9_zero, mov, cndmask, max_num, min_num). OPY: those plus add_nc_u32, lshlrev, sub_nc_u32, lshrrev, ashrrev, max_i32, min_i32. VOPD3 adds fma_f32, bitop2 (Y), and f64 fma / add / mul / max_num / min_num (X only), and drops fmaak / fmamk. | CDNA5 ISA §7.8, §7.8.1 (pp. 86-88) | LLVM also maps `v_dot2_f32_f16/bf16` without modifiers, with src2 = dst, onto VOPD (GCNVOPDUtils.cpp:83-122). `v_bitop3` with src2 = 0 becomes DUAL_BITOP2 (GCNVOPDUtils.cpp:289-292). |
| DPP | not allowed | CDNA5 ISA §7.8 (p. 86); LLVM GCNVOPDUtils.cpp:194-195 @274ab686 | **agree** |
| Literals | VOPD: at most one unique 32-bit literal, which X and Y may share; no 64-bit literal. VOPD3: no literal. | CDNA5 ISA §7.8 (p. 86), §7.8.1 (p. 88); LLVM GCNVOPDUtils.cpp:238-263, 322-327; llvm/test/MC/AMDGPU/gfx1250_asm_vopd_errs.s:8 @274ab686 | **agree.** LLVM can move one VOPD3 src0 immediate into an SGPR with `s_mov_b32` (only when liveness is tracked and the function is not optsize). |
| Scalar bus | At most 2 scalar values across the pair (SGPRs plus literal), with one SGPR or literal per op. VCC used by `cndmask` and EXEC as data each count as one SGPR. | CDNA5 ISA §7.8 (p. 86), §7.8.1 (p. 88); LLVM GCNVOPDUtils.cpp:272-274, 298-301, 328-330 @274ab686 | **agree.** VOPD3 `v_dual_cndmask_b32` may use any SGPR as the select (ISA p. 88). |
| src0 / src1 VGPR banks | X and Y sources in the same slot must be in different banks (bank = VGPR mod 4), or be the same VGPR with the same size. src2 (VOPD: fmamk's second input and the tied accumulator): one even, one odd. | CDNA5 ISA §7.8 (p. 86); LLVM Utils/AMDGPUBaseInfo.h:664-667; AMDGPUBaseInfo.cpp:985-1036; GCNVOPDUtils.cpp:389, 408-411 @274ab686 | **agree.** Masks: VOPD {dst 1, src0 3, src1 3, src2 1}; VOPD3 {1, 3, 3, 3}. LLVM allows the same VGPR (GFX12+) unless the pair is an anti-dependency swap. 64-bit operands check both halves. |
| V_MOV_B32 + V_MOV_B32 | OPY then reads through the SRC2 port, so LLVM skips source bank checks (except MSB equality) | CDNA5 ISA §7.8 (p. 86); LLVM GCNVOPDUtils.cpp:348-357 @274ab686 | **agree** |
| Destinations | VOPD: one even, one odd (vdstY encodes bits [7:1], LSB = ~vdstX[0]). VOPD3: any bank, but the two must not be equal or overlap. | CDNA5 ISA §7.8 (pp. 86-87), §7.8.1 (pp. 87-88); LLVM AMDGPUBaseInfo.cpp:1023-1030; gfx1250_asm_vopd_errs.s:157, 174 @274ab686 | **agree** |
| VGPR MSBs | X and Y operands in the same slot must have equal VGPR MSBs | LLVM Utils/AMDGPUBaseInfo.cpp:1016-1018 @274ab686 | Derived from ISA: one S_SET_VGPR_MSB value per operand slot, CDNA5 ISA §3.3.2.3 (p. 17). The ISA §7.8 text does not state it. |
| Multicycle OPX | If OPX is multicycle (f64 fma / add / mul / min / max), it must not overwrite OPY's sources | CDNA5 ISA §7.8 (p. 86); LLVM SIInstrInfo.cpp:11297-11303; GCNVOPDUtils.cpp:407-413 @274ab686 | **agree** (LLVM uses `IsDPMACCInstruction` as "multicycle"). |
| VOPD3 modifiers | NEG only. No abs, clamp, omod or op_sel. src1 (and src2 unless bitop3) must be registers. | CDNA5 ISA §7.8.1 (p. 87, NEGX/NEGY fields); LLVM GCNVOPDUtils.cpp:276-319; gfx1250_asm_vopd_errs.s ("ABS not allowed in VOPD3") @274ab686 | **agree** |
| VOPD3 F64 OPY last-VGPR hazard | When OPX is 64-bit, an OPY VGPR source must not be the wave's last allocated VGPR. LLVM refuses the pair when the register sits just below an allocation-granule boundary and at or above the function's VGPR use. | LLVM GCNVOPDUtils.cpp:124-165, 185-186; GCNFeatures.td:1323-1327, 2626 @274ab686 | **ISA silent.** Our pairing pass should copy this rule (derived). |
| Exceptions | A VOPD pair raises at most one exception | CDNA5 ISA §7.8 (p. 87) | — |
| Errors checked by llvm-mc for gfx1250 | literal count, constant bus, dst parity, distinct dst, src0/1/2 banks, VOPD3 ABS | llvm/test/MC/AMDGPU/gfx1250_asm_vopd_errs.s:1-242 @274ab686 | Can be used as an encoding-level probe (Open). |

---

### Open

1. **WMMA counts for 16x16x128 FP8/BF8, F8F6F4 (not both F4) and 32x16x128 F4.** ISA: 4 / 3. LLVM: 5 / 4 (§1.2). To settle: compile a MIR test with `llc -mcpu=gfx1250 -run-pass=post-RA-hazard-rec`, count the V_NOPs, and check for an ISA errata. Until then, use LLVM's larger value.
2. **Non-XDL WMMA (`v_wmma_f32_16x16x4_f32`, F64 WMMA if present) hazards.** Neither source has a rule. **unknown (no public source)**.
3. **Generic multicycle co-execution (ISA §5.7.2, p. 56) vs LLVM.** LLVM has no data-hazard rule for F64 / int-mul / cvt-scale multicycle VALUs (blocking 4-32). Either hardware interlocks them or LLVM has a gap. **unknown (no public source)**.
4. **Default co-execution mode.** It is unclear how the default arbiter stall (§5.7.2.1, which gives co-exec to other waves) relates to the same-wave slot table and its "co-exec disabled" column. Does a single wave without DISABLE_XDL_ARB_STALL need the slot counts? **unknown (no public source)**.
5. **V_CMPX → V_PERMLANE filler.** ISA allows V_NOP; LLVM says V_NOP is discarded and uses `v_mov_b32 vX, vX`. Use the v_mov form.
6. **"Wait-idle" before S_GETREG.** Is `s_wait_alu 0` enough (LLVM), or is `S_WAIT_IDLE` needed? SCRATCH_BASE is in the ISA list only. Use `s_wait_idle` until settled (derived safe choice).
7. **LLVM-only rules with no ISA text**: V_PERM_PK16 follower and entry prologue; scratch-base forwarding (window 10, comment says 8 for VALU); two V_NOPs before `s_setreg` MODE; TDM drain at 10; DS_ATOMIC_ASYNC_BARRIER_ARRIVE_B64 waits and no-clause (the ISA lists it in the async clause type); VOPD3 F64 OPY last VGPR; SMEM soft clause under XNACK. A revision of the public ISA document would settle these (LLVM's comments refer to a document section that is not public).
8. **`fixSetRegMode` match.** It compares the encoded hwreg simm16 with `ID_MODE` (GCNHazardRecognizer.cpp:4531), so partial MODE writes may not get the V_NOPs. Check with llc on `s_setreg_imm32_b32 hwreg(HW_REG_MODE, 0, 4)`.
9. **ISA-only rules that LLVM ignores**: S_SETHALT / S_SETPRIO → S_GETREG one-instruction gap; S_WAIT_IDLE around xnack registers; the WMMA reuse-bit rule.
10. **Fillers.** Does an intervening WMMA count as an "unrelated VALU" filler (LLVM: yes)? Does an EXEC = 0-skipped VALU count? **unknown (no public source)**. Prefer V_NOPs, which are never skipped.
11. **S_WAIT_ALU / depctr field semantics** are not described in the ISA. LLVM gives only the layout. Probe: assemble `s_wait_alu depctr_*` with the rocm-10.1 llvm-mc `-mcpu=gfx1250 -show-encoding` and compare with gfx12_asm_sopp.s:242-255.
12. **Latency numbers.** LDS 20, VMEM 320, SMEM 20, branch 32 and barrier 2000 are LLVM model values. The ISA gives none. Settling them needs measurement, which is out of scope (no GPU runs).
13. **VGPR ECC on gfx1250 parts and the half-rate 16-bit writes**: **unknown (no public source)**.
14. **SWMMAC co-exec windows.** LLVM uses the 3-slot FP8 pattern for FP8 SWMMAC, while the ISA hazard row says 2. F16/BF16/IU8 SWMMAC windows are permissive. These only affect throughput modelling.
15. **Encoding-level VOPD probes** (banks, parity, literals, VOPD3 ABS) can be run without a GPU: `llvm-mc -triple=amdgcn -mcpu=gfx1250` from rocm-10.1 on llvm/test/MC/AMDGPU/gfx1250_asm_vopd_errs.s and gfx1250_asm_vopd3.s. Hazard insertion needs `llc` on MIR. The sparse checkout has no `llvm/test/CodeGen/AMDGPU` hazard tests to compare against.

## 6. gfx1250 target description — P4: `atoms` (WMMA / SWMMAC / transpose loads) and memory (`mem`)

Scope: the `Atoms` field of `Target` (matrix atoms, operand lane layouts, scale forms, copy atoms) and the memory facts that the
P0 design spreads over `atoms` (copy atoms), `lds` (DMA forms), `emit` (descriptor formats) and `counters` (counter class per form).
There is no `mem` field in `target-design.md` §2. This section proposes one sub-struct, `MemModel`, to hold them. Field names follow `target-design.md`.

Sources (public only):

- **ISA** = AMD "CDNA5" Instruction Set Architecture Reference Guide, 27-July-2026. Cited as *CDNA5 ISA §x (p. N)*. N is the printed page ("N of 823"); printed page = PDF page − 9.
- **LLVM** = upstream llvm-project @274ab686. Path abbreviations: `AMDGPU/` = `llvm/lib/Target/AMDGPU/`, `MC/` = `llvm/test/MC/AMDGPU/`,
  `docs/` = `llvm/docs/`. Cited as *LLVM path:line @274ab686*.

"derived" marks my own inference from the cited facts. "unknown (no public source)" means neither source states the fact. All layout
and table content is restated as compact rules; nothing is copied from the ISA tables.

Notation for layouts: `l` = lane 0..31, `g = l / 16` (lane half), `i = l % 16`, `r` = VGPR index inside the operand tuple, `h` = 16-bit
half (0 = [15:0], 1 = [31:16]), `b` = byte (0..3, at bits [8b+7:8b]), `n` = nibble (0..7, at bits [4n+3:4n]).

---

### 1. `atoms.matrix`: WMMA / SWMMAC instruction set

```rust
pub struct MatrixAtom {
    mnemonic: &'static str, opcode: u8, enc: Enc,           // Enc::Vop3p (8 B) | Enc::Vop3px2 (16 B = LD_SCALE + WMMA)
    m: u8, n: u8, k: u16,
    a: Fmt, b: Fmt, c: Fmt, d: Fmt,                          // for SWMMAC: c == d (accumulator in VDST)
    vgprs: Regs { a: u8, b: u8, c: u8, d: u8 },              // VGPRs per lane (wave32)
    sparse: Option<SparseIdx { idx_vgprs: u8, index_key_bits: u8 }>,
    scale: Option<ScaleForm { block_k: u8 /*32|16*/, scale_regs: u8 /*1|2*/ }>,
    mods: Mods,                                              // see §2
    avail: Avail,                                            // Gfx1250 | Gfx1250AndStrict
}
pub const GFX1250_WMMA: &[MatrixAtom] = &[
  // dense, VOP3P (8-byte), wave32 only
  wmma("v_wmma_f32_16x16x4_f32",        0x5d, 16,16,  4, F32,  F32,  F32, F32, Regs{a:2, b:2, c:8, d:8}, Both),
  wmma("v_wmma_f32_16x16x32_f16",       0x60, 16,16, 32, F16,  F16,  F32, F32, Regs{a:8, b:8, c:8, d:8}, Both),
  wmma("v_wmma_f16_16x16x32_f16",       0x61, 16,16, 32, F16,  F16,  F16, F16, Regs{a:8, b:8, c:4, d:4}, Both),
  wmma("v_wmma_f32_16x16x32_bf16",      0x62, 16,16, 32, BF16, BF16, F32, F32, Regs{a:8, b:8, c:8, d:8}, Both),
  wmma("v_wmma_bf16_16x16x32_bf16",     0x63, 16,16, 32, BF16, BF16, BF16,BF16,Regs{a:8, b:8, c:4, d:4}, Both),
  wmma("v_wmma_bf16f32_16x16x32_bf16",  0x64, 16,16, 32, BF16, BF16, F32, BF16,Regs{a:8, b:8, c:8, d:4}, Both),
  wmma("v_wmma_{f32,f16}_16x16x64_{fp8,bf8}_{fp8,bf8}", 0x6a..=0x71, 16,16, 64, 8-bit, 8-bit, ..,  Regs{a:8, b:8, c:8|4, d:8|4}, Both),
  wmma("v_wmma_i32_16x16x64_iu8",       0x72, 16,16, 64, IU8,  IU8,  I32, I32, Regs{a:8, b:8, c:8, d:8}, Both),
  wmma("v_wmma_{f32,f16}_16x16x128_{fp8,bf8}_{fp8,bf8}",0x80..=0x87,16,16,128, 8-bit, 8-bit, ..,  Regs{a:16,b:16,c:8|4,d:8|4}, Gfx1250Only),
  wmma("v_wmma_f32_16x16x128_f8f6f4",   0x33, 16,16,128, F8|F6|F4 (matrix_a_fmt), F8|F6|F4 (matrix_b_fmt), F32, F32,
       Regs{a:16|12|8, b:16|12|8, c:8, d:8}, Both),
  wmma("v_wmma_f32_32x16x128_f4",       0x88, 32,16,128, FP4,  FP4,  F32, F32, Regs{a:16,b:8, c:16,d:16}, Gfx1250Only),
  // block-scaled, VOP3PX2 (16-byte): dword0 = LD_SCALE (opcode byte 0x35 = SCALE, 0x3a = SCALE16), dword1 = the WMMA
  scaled("v_wmma_scale_f32_16x16x128_f8f6f4",   0x33, ScaleForm{block_k:32, scale_regs:1}, Both),
  scaled("v_wmma_scale16_f32_16x16x128_f8f6f4", 0x33, ScaleForm{block_k:16, scale_regs:2}, Both),
  scaled("v_wmma_scale_f32_32x16x128_f4",       0x88, ScaleForm{block_k:32, scale_regs:1}, Gfx1250Only),
  scaled("v_wmma_scale16_f32_32x16x128_f4",     0x88, ScaleForm{block_k:16, scale_regs:2}, Gfx1250Only),
  // sparse (A is 2:4 structured along K), VOP3P; SRC2 = index VGPR(s); C/D in VDST
  swmmac("v_swmmac_f32_16x16x64_f16",      0x65, 16,16, 64, Regs{a:8, b:16, c:8, d:8}, SparseIdx{idx_vgprs:1, index_key_bits:1}),
  swmmac("v_swmmac_f32_16x16x64_bf16",     0x66, ..),  swmmac("v_swmmac_f16_16x16x64_f16",  0x67, .. c/d:4),
  swmmac("v_swmmac_bf16_16x16x64_bf16",    0x68, .. c/d:4), swmmac("v_swmmac_bf16f32_16x16x64_bf16", 0x69, .. c/d:8 (LLVM)),
  swmmac("v_swmmac_{f32,f16}_16x16x128_{fp8,bf8}_{fp8,bf8}", 0x73..=0x7a, 16,16,128, Regs{a:8, b:16, c:8|4, d:8|4},
         SparseIdx{idx_vgprs:2, index_key_bits:1}),
  swmmac("v_swmmac_i32_16x16x128_iu8",     0x7b, 16,16,128, Regs{a:8, b:16, c:8, d:8}, SparseIdx{idx_vgprs:2, index_key_bits:1}),
];
// LLVM-only helper encodings (no ISA entry of their own): v_wmma_ld_scale_paired_b32 (0x35), v_wmma_ld_scale16_paired_b64 (0x3a)
// Not on gfx1250: v_wmma_f64_16x16x4_f64 (gfx1251 only), all gfx12 16x16x16 WMMA and 16x16x32/x64 SWMMAC, wave64 forms.
```

| fact | value | source | notes |
|---|---|---|---|
| Operation | D = A×B + C; one matrix spread across the 32 lanes; VOP3P encoding | CDNA5 ISA §7.12 (p. 94) | |
| Dense WMMA list, shapes and formats | 16x16x4 F32; 16x16x32 F16/BF16 (F32, F16, BF16 or BF16-from-F32-C results); 16x16x64 FP8/BF8 (4 A/B pairs, F32 or F16 result) and IU8 (I32); 16x16x128 FP8/BF8 (4 pairs × F32/F16) | CDNA5 ISA §7.12 Table 43 (p. 95–96); LLVM `AMDGPU/VOP3PInstructions.td:2259-2319` @274ab686 | ISA and LLVM agree on the set. |
| BF16F32 variant | C is F32, D is BF16 | CDNA5 ISA §7.12 Table 43 (p. 95); LLVM `AMDGPU/VOP3PInstructions.td:2158-2160` (profile D=v8bf16, C=v8f32) | |
| F8F6F4 non-scaled form | `v_wmma_f32_16x16x128_f8f6f4`, opcode 51 (0x33); A/B formats from OPSEL / OPSEL_HI | CDNA5 ISA §14.3.6 Table 96 (p. 190); §15.10 (p. 440–441); LLVM `AMDGPU/VOP3PInstructions.td:2277,2819` @274ab686 | Missing from ISA Table 43 (p. 95–96), present in the opcode table and in Chapter 15. |
| F4 32x16 non-scaled form | `v_wmma_f32_32x16x128_f4`, opcode 136 (0x88) | CDNA5 ISA §14.3.6 Table 96 (p. 190); §15.10 (p. 464); LLVM `AMDGPU/VOP3PInstructions.td:2294,2815` | Missing from Table 43; present elsewhere in the ISA. |
| Scaled forms | SCALE (block 32) and SCALE16 (block 16) for 16x16x128 F8F6F4 and 32x16x128 F4 | CDNA5 ISA §7.12 Table 43 (p. 96); §7.12.6 (p. 103); §15.11 (p. 465–468); LLVM `AMDGPU/VOP3PInstructions.td:2278-2279,2296-2297,2820-2826` | |
| SWMMAC list | 16x16x64 F16/BF16 (F32, F16, BF16, BF16F32 results); 16x16x128 FP8/BF8 (4 pairs × F32/F16); 16x16x128 IU8 | CDNA5 ISA §7.12 Table 43 (p. 95–96); LLVM `AMDGPU/VOP3PInstructions.td:2300-2315,2830-2845` | Table 43 gives the A size after expansion: 16x64 for 16-bit, 16x128 for 8-bit. |
| Opcodes (VOP3P OP field) | 93 f32_16x16x4; 96–100 16-bit dense; 101–105 16-bit SWMMAC; 106–113 x64 FP8; 114 IU8; 115–123 x128 SWMMAC; 128–135 x128 FP8; 51 F8F6F4; 136 F4 | CDNA5 ISA §14.3.6 Table 96 (p. 189–190); LLVM `AMDGPU/VOP3PInstructions.td:2791-2845` | All values agree (0x5d=93 … 0x88=136). |
| VOP3PX2 (scaled) encoding | 128 bits. Bits [63:0] are the load-scale part: `SCLOP` [19:16] = 0b0101 (SCALE) or 0b1010 (SCALE16), ENCODING [31:20] = 0xCC3. Bits [127:64] are the WMMA part: OP 0x33. | CDNA5 ISA §14.3.7 Tables 97–98 (p. 190–191) | LLVM writes byte 2 as 0x35 / 0x3a and the WMMA opcode as 0x33 or 0x88 (`AMDGPU/VOP3PInstructions.td:2664-2706,2820-2826`). Same bits, derived. ISA §14.3.7 lists only the 0x33 forms, but p. 97 and §15.11 (p. 465–468) also use VOP3PX2 for the 32x16 F4 forms (opcode 136). |
| Scaled op counts as 2 instructions; clause rules | A clause with VOP3PX2 needs ≥3 instructions (the 16-byte op counts 2), must not end with VOP3PX2, and needs EXEC≠0 throughout | CDNA5 ISA §7.12.6 (p. 104) | Verifier rule (`VerifyLimits`), derived. |
| LD_SCALE in S_DELAY_ALU | the LD_SCALE + WMMA pair counts as one instruction; LD_SCALE is not scoreboarded | CDNA5 ISA §5.8 (text near p. 57–58; ISA txt line 3697) | Belongs to the latency/hazard sections. Noted here for the atom. |
| Standalone `v_wmma_ld_scale_paired_b32` / `v_wmma_ld_scale16_paired_b64` | LLVM defines them as separate VOP3P ops, opcodes 0x35 / 0x3a, with operands src0, src1 (+ scale sel, fmt, reuse) | LLVM `AMDGPU/VOP3PInstructions.td:2241-2251,2321-2324,2947-2948`; `MC/gfx1250_asm_wmma_w32.s:966-998,1051-1068` | ISA has no separate entry. It only mentions "V_WMMA_LD_SCALE" (§3.3.2, p. 18). LLVM-only spelling of the first VOP3PX2 dword. Do not emit them alone (derived). |
| VGPR counts per lane | as in the sketch: A/B 2 (F32 x4), 8 (16-bit x32, 8-bit x64), 16 (8-bit x128), F8F6F4 A/B = 16/12/8 for F8/F6/F4; C/D 8 (32-bit) or 4 (16-bit); 32x16 F4: A 16, B 8, C/D 16 | LLVM `AMDGPU/VOP3PInstructions.td:2140-2238` (profiles); `MC/gfx1250_asm_wmma_w32.s:6,41,81,161,441,901-943,1686`; CDNA5 ISA §7.12.6.1 (p. 105: 16/12/8 VGPRs for F8/F6/F4) | Dense counts: derived = M·K·bits / (32 lanes · 32). |
| SWMMAC register counts | A (packed) 8; B 16; C/D 8 or 4; index 1 VGPR (16-bit forms) or 2 VGPRs (8-bit forms) | LLVM `AMDGPU/VOP3PInstructions.td:2185-2200` (index type i32 vs i64); `MC/gfx1250_asm_wmma_w32.s:556,646,806` | ISA: the 16-bit index fits one VGPR (§7.12.4, p. 101). For the 8-bit index the ISA describes only the first 32 bits (§7.12.5, p. 102). A 2-VGPR index is derived and agrees with LLVM. |
| bf16f32 SWMMAC accumulator size | LLVM: VDST is 8 VGPRs (F32 profile) | LLVM `AMDGPU/VOP3PInstructions.td:2303,2835`; `MC/gfx1250_asm_wmma_w32.s:616` | ISA: C=F32, result=BF16, and SWMMAC uses VDST as both C and D (§7.12, p. 95; Table 44, p. 101). How BF16 is stored in an 8-VGPR VDST: unknown (no public source). |
| Operand roles | SRC0=A, SRC1=B; WMMA SRC2=C, VDST=D; SWMMAC SRC2=index, VDST=C and D (read-modify-write) | CDNA5 ISA §7.12.2 Table 44 (p. 101) | |
| Register alignment | SRC0, SRC1, VDST even VGPR; SRC2 even VGPR or inline 0 | CDNA5 ISA §7.12 (p. 96); LLVM `AMDGPU/GCNFeatures.td:483,2498` (FeatureRequiresAlignedVGPRs in the 12_50 set) | **Conflict:** LLVM accepts other inline constants as C, e.g. `1.0` (`MC/gfx1250_asm_wmma_w32.s:11-13,951-953`) and integer `1` for iu8 (`:448`). |
| Wave size / EXEC | wave32 only; EXEC must be all ones | CDNA5 ISA §7.12 (p. 97); LLVM `MC/gfx1250_asm_wmma_w32.s:3,9` ("requires wavesize=32") | Pseudocode forces EXEC to all ones (§15.10, p. 444). |
| Numerics | RNE only; denorms kept (MODE.denorm ignored); NaN in → NaN out; no exceptions; no DPP; FP16_OVFL=1 → saturate to ±max (16-bit and smaller floats), IU8 saturates | CDNA5 ISA §7.12 (p. 97) | |
| Clamp (CM) | must be 0 for float results; for integer results, clamp to [INT_MIN, INT_MAX] | CDNA5 ISA §7.12 (p. 97); LLVM `MC/gfx1250_asm_wmma_w32.s:473,838` (clamp on iu8), `MC/gfx1250_asm_wmma_w32_err.s:372-373` (rejected on f8f6f4) | |
| gfx1250 vs gfx1250-strict | strict lacks `FeatureWMMAF4Insts` and `FeatureWMMA16x16x128FP8Insts`. So strict rejects: `v_wmma_f32_32x16x128_f4`, `v_wmma_scale{,16}_f32_32x16x128_f4` and the 8 `v_wmma_{f16,f32}_16x16x128_{fp8,bf8}_{fp8,bf8}`. Everything else (incl. F8F6F4 scaled, all SWMMAC, f32 16x16x4, ld_scale) is common. | LLVM `AMDGPU/GCNFeatures.td:2588-2622` (strict) vs `:2624-2661` (gfx1250); `AMDGPU/VOP3PInstructions.td:2282-2298`; `MC/gfx1250-strict_err.s:2-36`; `AMDGPU/GCNProcessors.td:310-317` | The ISA describes the non-strict set: it lists these ops and their hazard rows (p. 98–99), derived. Strict also lacks `FeatureFP8E5M3Insts` and `FeatureBlock16ConversionScaleInsts`. So it rejects the `clamp` (E5M3) forms of v_cvt_*fp8 used to build E5M3 scales (`MC/gfx1250-strict_err.s:217-224`), and `scale_sel` ranges for v_cvt_scale_pk* differ (`:82-215`). |
| Not available on gfx1250 | gfx12 16x16x16 WMMA and 16x16x32/x64 SWMMAC (rejected); `v_wmma_f64_16x16x4_f64` (only with `FeatureGFX1251GEMMInsts` = gfx1251) | LLVM `MC/gfx1250_asm_unsupported.s:22-52`; `MC/gfx1250-strict_err.s:38-80`; `AMDGPU/VOP3PInstructions.td:2327-2329`; `AMDGPU/GCNFeatures.td:1166-1168,2663-2678` | ISA has no F64 WMMA, but its NEG/OPSEL tables have an "F32 / F64" row (p. 96–97). |

---

### 2. `atoms.matrix.mods`: modifiers and their encodings

```rust
pub struct Mods {
    // VOP3P bit positions (dword-relative): OPSEL[2:0]=[13:11], OPSEL_HI[1:0]=[60:59], OPSEL_HI[2]=[14],
    // NEG[2:0]=[63:61], NEG_HI[2:0]=[10:8], CM=[15]
    reuse_a: Some(Bit(13)), reuse_b: Some(Bit(14)),               // OPSEL[2], OPSEL_HI[2]; not on non-scaled F8F6F4 / 32x16 F4
    c_neg: Bit(63), c_abs: Bit(10),                               // NEG[2], NEG_HI[2] on SRC2 (float forms)
    iu8_signed_a: Bit(61), iu8_signed_b: Bit(62),                 // NEG[0], NEG[1] for IU8
    matrix_a_fmt: Bits(13..=11), matrix_b_fmt: (Bit(14), Bits(60..=59)), // F8F6F4 only: 0 FP8,1 BF8,2 FP6,3 BF6,4 FP4
    index_key: Bit(11),                                           // SWMMAC, LLVM only (0..1)
    // VOP3PX2 load-scale dword: A scale sel [11], B scale sel [59], A scale fmt [62:61], B scale fmt [9:8],
    // reuse A [13], reuse B [14]; scale fmt 0 E8M0, 1 E5M3, 2 E4M3
}
```

| fact | value | source | notes |
|---|---|---|---|
| NEG / NEG_HI use | IU8: NEG[0]=A signed, NEG[1]=B signed, rest 0. FP8/BF8, F16/BF16, F32, F8F6F4/F4: only NEG[2]=negate C and NEG_HI[2]=abs C. LD_SCALE: all 0. | CDNA5 ISA §7.12 VOP3P field table (p. 96) | |
| LLVM NEG acceptance | gfx1250 16-bit dense WMMA: no A/B neg; FP8/F4/F8F6F4/F32x4: no A/B neg; IU8: `neg_lo:[1,0,0]`/`[0,1,0]` encode signedness | LLVM `AMDGPU/VOP3PInstructions.td:1779,1809-1813` (NoABMods), `:2142-2160`; `MC/gfx1250_asm_wmma_w32.s:453-458`; `MC/gfx1250_asm_wmma_w32_err.s:360-370,496-524` | **Conflict:** LLVM accepts `neg_lo:[1,0,0] neg_hi:[1,0,0]` on 16-bit **SWMMAC** (`MC/gfx1250_asm_wmma_w32.s:566-573`). ISA says NEG[1:0]/NEG_HI[1:0] must be 0 for FP16/BF16 (p. 96). |
| OPSEL / OPSEL_HI use | SWMMAC and the plain IU8/FP8/16-bit/F32 WMMA: OPSEL[2]=reuse A, OPSEL_HI[2]=reuse B, others 0. F8F6F4: OPSEL[2:0]=A type, OPSEL_HI[2:0]=B type. | CDNA5 ISA §7.12 OPSEL table (p. 97) | Bit positions from VOP3P format: OPSEL [13:11], OPSEL_HI[1:0] [60:59], OPSEL_HI[2] [14] (§14.3.6, p. 188–189). |
| Reuse encodings in LLVM | `matrix_a_reuse` → bit 13, `matrix_b_reuse` → bit 14 | LLVM `AMDGPU/VOP3PInstructions.td:2578-2584`; `MC/gfx1250_asm_wmma_w32.s:31-38` (bytes 0x20 / 0x40) | Agrees with ISA. |
| Reuse rule | hint only; set only if the previous instruction is the same instruction, else result undefined | CDNA5 ISA §7.12 (p. 97) | Scheduler / verifier constraint, derived. |
| Reuse on scaled ops | in the LD_SCALE dword: OPSEL[2] (bit 13) = reuse A, OPSEL_HI[2] (bit 14) = reuse B | CDNA5 ISA §7.12.6.2 (p. 107); §14.3.7 Table 97 (p. 191); LLVM `AMDGPU/VOP3PInstructions.td:2672-2673` | Non-scaled F8F6F4 and non-scaled 32x16 F4 have no reuse in LLVM (`:2176,2236` HasMatrixReuse=0). This matches ISA "F8F6F4 has reuse bits in LD_SCALE part" (p. 97). |
| matrix_a_fmt / matrix_b_fmt values | 0 FP8 E4M3, 1 BF8 E5M2, 2 FP6 E2M3, 3 BF6 E3M2, 4 FP4 E2M1 | CDNA5 ISA §7.12.6 (p. 103); §15.10 (p. 441); LLVM `AMDGPU/SIDefines.h:1286-1292` | LLVM spelling: `matrix_a_fmt:MATRIX_FMT_BF8` etc. (`MC/gfx1250_asm_wmma_w32.s:906-948`). The register tuple size must match the format (`MC/gfx1250_asm_wmma_w32_err.s:391-453`). |
| Format bit placement | A fmt in OPSEL[2:0] (bits 13:11); B fmt in {OPSEL_HI[2]=bit 14, OPSEL_HI[1:0]=bits 60:59} | LLVM `AMDGPU/VOP3PInstructions.td:2571-2584`; VOP3PX2: `:2691-2693` (bits 77:75 and 78,124,123) | ISA §14.3.7 Table 98 (p. 191) names OPSEL = A type and OPSEL_HI2 = B type bit 2. Agrees. |
| F4 32x16 OPSEL | ISA: OPSEL and OPSEL_HI must be 0 | CDNA5 ISA §15.10 (p. 464), §15.11 (p. 465–468) | **Conflict:** LLVM encodes OPSEL_HI = 0b111 for these (`AMDGPU/VOP3PInstructions.td:2581-2584,2693`; `MC/gfx1250_asm_wmma_w32.s:1688` byte1 0x40 and byte7 0x1c; `:1728`). |
| index_key (SWMMAC) | LLVM: 1 bit in OPSEL[0] (bit 11), range 0..1 for every gfx1250 SWMMAC | LLVM `AMDGPU/VOP3PInstructions.td:2571-2575,1887-1920`; `MC/gfx1250_asm_wmma_w32.s:561-563,651-653,813`; `MC/gfx1250_asm_wmma_w32_err.s:150-151` | **Conflict / ISA silent:** ISA says OPSEL[1:0] must be 0 for SWMMAC (p. 97) and never defines an index-select field. Semantics unknown (no public source). |
| Scale select (A/B) | 0 = scale values from lanes 0–15, 1 = lanes 16–31; A at LD_SCALE bit 11, B at bit 59 | CDNA5 ISA §7.12.6.2 (p. 106); §14.3.7 Table 97 (p. 191); LLVM `AMDGPU/VOP3PInstructions.td:2670,2680` (`matrix_a_scale:MATRIX_SCALE_ROW0/ROW1`) | For 32x16 F4 the A select is unused (all 32 lanes hold A scales), p. 106. |
| Scale formats | 0 E8M0, 1 E5M3, 2 E4M3. A in LD_SCALE NEG[1:0] (bits 62:61), B in NEG_HI[1:0] (bits 9:8). | CDNA5 ISA §7.12.6 (p. 103–104); §14.3.7 Table 97 (p. 191); LLVM `AMDGPU/SIDefines.h:1299-1303`, `AMDGPU/VOP3PInstructions.td:2669,2682` | |
| Legal type × scale combinations | E8M0 works with any F8/F6/F4. E5M3/E4M3 only on an FP4 operand. When both operands use a non-E8M0 scale, both are FP4 with the same scale format. | CDNA5 ISA §7.12.6 (p. 104) (my reading of its combination table) | LLVM enforces it: `MC/gfx1250_asm_wmma_w32_err.s:526-532` ("invalid matrix and scale format combination"). |
| Scale source operands | SCALE: one 32-bit scale per matrix (VGPR, SGPR or inline). SCALE16: 64-bit (even-aligned VGPR pair or SGPR pair). SGPR → only bits [7:0], one scale for all. Inline 0 → 0x7F (×1.0). Scale VGPRs ignore VGPR-MSB, so only v0–v255. | CDNA5 ISA §7.12.6.2 (p. 106); §3.3.2 (p. 18); §15.11 (p. 465); LLVM `AMDGPU/VOP3PInstructions.td:2322-2323` (`VCSrc_b32_Lo256`, `VCSrc_b64_Lo256`); `AMDGPU/Utils/AMDGPUBaseInfo.cpp:3564-3569` | |
| E5M3 scale production | `v_cvt_pk_fp8_f32`, `v_cvt_f32_fp8`, `v_cvt_sr_fp8_f32` with CLAMP=1 convert to/from E5M3 | CDNA5 ISA §7.12.6 (p. 104) | Not on gfx1250-strict (LLVM `MC/gfx1250-strict_err.s:217-224`). |
| VOP3PX2 constant field | ISA: bits [58:50] of the LD_SCALE dword must be 0x100 | CDNA5 ISA §14.3.7 Table 97 (p. 191) | **Conflict:** LLVM emits 0x080 there (byte 7 = 0x02 → bit 57) in every test, e.g. `MC/gfx1250_asm_wmma_w32.s:1148,968`. Derived from the encodings. |

---

### 3. `atoms.matrix.layout`: VGPR lane layouts (wave32)

```rust
pub enum Layout {
    // Dense A (MxK); B (KxN) uses the same rule with M→N. C/D: one matrix row striped across lanes.
    A32  { /* 16x4  f32 */  k: |r,g| 2*g + r,                                   m: |i| i },               // r in 0..2
    A16  { /* 16x32 16b */  k: |r,h,g| 16*(r/4) + 8*g + 2*(r%4) + h,            m: |i| i },               // r in 0..8
    A8   { /* 16x64 8b  */  k: |r,b,g| 32*(r/4) + 16*((r%4)/2) + 8*g + 4*(r%2) + b, m: |i| i },           // r in 0..8; x128: r 8..16 adds 64
    A4   { /* 16x128 4b */  k: |r,n,g| 64*(r/4) + 32*g + 8*(r%4) + n,           m: |i| i },               // r in 0..8; 32x128: r 8..16 → M+16
    CD32 { /* 16x16 32b */  m: |r,g| r + 8*g,                                   n: |i| i },               // r in 0..8; 32x16: r 8..16 → M+16
    CD16 { /* 16x16 16b */  m: |r,h,g| 2*r + h + 8*g,                           n: |i| i },               // r in 0..4
    // Block-scale (F8F6F4 / F4) and SWMMAC-B patterns: the lane half owns a 16-wide K block inside each 32-K chunk
    F8x128 { k: |r,b,g| 64*(r/8) + 32*((r%8)/4) + 16*g + 4*(r%4) + b },       // 16 VGPRs
    F6x128 { k: |j,p,g| 64*j + 32*g + p },                                      // j = r/6, p = 6-bit slot in {V(6j+5)..V(6j)}
    F4x128 { k: |r,n,g| 64*(r/4) + 32*g + 8*(r%4) + n },                        // 8 VGPRs (= A4)
    SpB16  { k: |r,h,g| 32*(r/8) + 16*g + 2*(r%8) + h },                        // SWMMAC B 64x16 16-bit, 16 VGPRs
    SpB8   { k: |r,b,g| 64*(r/8) + 32*((r%8)/4) + 16*g + 4*(r%4) + b },         // SWMMAC B 128x16 8-bit, 16 VGPRs
}
```

| fact | value | source | notes |
|---|---|---|---|
| General rule | A: one row (fixed M) per lane, K spread over the lane's VGPRs and wrapping to the other lane half. B, C, D: one row spread across lanes within one VGPR. B is described by the A tables with M→N. Sparse: the rule applies to the packed (pre-expansion) data. | CDNA5 ISA §7.12.2 (p. 99) | |
| 32-bit C/D 16x16 | lane l → N = l%16. VGPR r (0..7) → M = r + 8·(l/16). 32x16 C/D: VGPRs 8–15 hold M+16. | CDNA5 ISA §7.12.2 (p. 100) | Rule `CD32`. |
| 16-bit C/D 16x16 | 4 VGPRs. VGPR r, half h → M = 2r + h + 8·(l/16), N = l%16. | CDNA5 ISA §7.12.2 (p. 100) | Rule `CD16`. |
| 32-bit A 16x4 | 2 VGPRs; lane → M = l%16; VGPR r → K = r + 2·(l/16) | CDNA5 ISA §7.12.2 (p. 100) | Rule `A32`. |
| 16-bit A 16x32 | 8 VGPRs; K = 16·(r/4) + 8·g + 2·(r%4) + h | CDNA5 ISA §7.12.2 (p. 100) | Rule `A16`; checked against all listed rows. |
| 8-bit A 16x64 | 8 VGPRs; K = 32·(r/4) + 16·((r%4)/2) + 8·g + 4·(r%2) + b. 16x128 = two 16x64 blocks in consecutive VGPRs (r 8–15: K+64). | CDNA5 ISA §7.12.2 (p. 100–101) | Rule `A8`. Applies to x64/x128 FP8/BF8 and IU8 WMMA, derived. |
| 4-bit A 16x128 | 8 VGPRs; K = 64·(r/4) + 32·g + 8·(r%4) + n. 32x128 = two 16x128 blocks, M 16–31 in the second 8 VGPRs. | CDNA5 ISA §7.12.2 (p. 101) | Rule `A4`. |
| Block-scale F8 (16x128) | 16 VGPRs; K = 64·(r/8) + 32·((r%8)/4) + 16·g + 4·(r%4) + b | CDNA5 ISA §7.12.6.1.1 (p. 105) | **Possible conflict:** for the same K this differs from the dense 8-bit rule (p. 100): dense puts K 8–11 in V0 of the upper half, block-scale puts K 16–19 there. See Open. |
| Block-scale F6 (16x128) | 12 VGPRs; the 6 VGPRs of a lane form a 192-bit string with K=0 in the LSBs. VGPRs 0–5: K 0–31 (g=0) / 32–63 (g=1). VGPRs 6–11: K 64–95 / 96–127. | CDNA5 ISA §7.12.6.1.2 (p. 105) | Rule `F6x128`. |
| Block-scale F4 (16x128) | 8 VGPRs; VGPRs 0–3: K 0–31 / 32–63; VGPRs 4–7: K 64–95 / 96–127 | CDNA5 ISA §7.12.6.1.3 (p. 105) | Same as dense `A4`. Nibble order within the 4-VGPR group is derived (sequential, as in p. 101). |
| B of block-scale | same as A with N in place of M | CDNA5 ISA §7.12.6.1 (p. 104) | |
| Scale values, SCALE (block 32) 16x128 | 1 VGPR. The selected lane half (SCL_OPSEL) holds row M = l%16. Byte b = scale for K 32b..32b+31. The other half is unused. | CDNA5 ISA §7.12.6.2 (p. 106) | |
| Scale values, SCALE16 16x128 | 2 VGPRs (even). Selected half: lane → M = l%16. VGPR v, byte b → K block 16·(4v+b)..+15. | CDNA5 ISA §7.12.6.2 (p. 106) | |
| Scale values, 32x16 F4 | SCALE: 1 VGPR, all lanes, lane l → M = l, byte b → K 32b..; SCALE16: 2 VGPRs, lane l → M = l, VGPR v, byte b → K 16·(4v+b).. | CDNA5 ISA §7.12.6.2 (p. 106–107) | The B scales (N=16) use SCL_OPSEL_HI[0] to pick the half (§15.11, p. 466). |
| Sparse 16-bit (SWMMAC x64) | A stored packed as 16x32 (8 VGPRs). The index VGPR holds 2-bit codes: bits [2c+1:2c] steer packed element c (VGPR c/2, half c%2). Each 4-wide group of expanded K gets 2 codes, idx0 < idx1. | CDNA5 ISA §7.12.3–7.12.4 (p. 101–102) | Lane rule of the pseudocode: lane = 16·(expanded_col / 32) + row, so lanes 16–31 cover expanded K 32–63. **Possible conflict** with the dense `A16` lane split (p. 100); see Open. |
| Sparse B 64x16 16-bit | 16 VGPRs; K = 32·(r/8) + 16·g + 2·(r%8) + h | CDNA5 ISA §7.12.4 (p. 102) | Rule `SpB16`. B must be dense and column-major (p. 101). |
| Sparse 8-bit (SWMMAC x128) | A packed 16x64 in 8 VGPRs. Index bits [2c+1:2c] → byte c of V0..V3 (c = 0..15). | CDNA5 ISA §7.12.5 (p. 102) | The codes for V4–V7 are not described. Second index VGPR: derived, and LLVM uses a 64-bit index (`AMDGPU/VOP3PInstructions.td:1993,2194-2200`). |
| Sparse B 128x16 8-bit | 16 VGPRs; K = 64·(r/8) + 32·((r%8)/4) + 16·g + 4·(r%4) + b | CDNA5 ISA §7.12.5 (p. 102) | Rule `SpB8`, which is the same pattern as block-scale F8 (derived). |
| Loading sparse data | same as dense 16x16 loads, but two load instructions into consecutive VGPR groups | CDNA5 ISA §7.12.3 (p. 101) | |
| LLVM element-count comments | wave32 16x16: 8 elements per lane; 16x32: 16 elements; 16x64: 32 elements | LLVM `AMDGPU/VOP3PInstructions.td:2080-2092` | Counts only, no lane maps. LLVM has no lane-layout tables. The ISA is the only public layout source. |

---

### 4. `atoms.copy`: transpose loads and other matrix copy atoms

```rust
pub const GFX1250_COPY_TR: &[CopyAtom] = &[
  // global (VGLOBAL encoding, LOADcnt), wave32 only, EXEC must be all-ones (or 0 → acts as s_nop)
  tr("global_load_tr16_b128", 87, elem_bits:16, tile:(16,16), vgprs:4, align:2),
  tr("global_load_tr8_b64",   88, elem_bits: 8, tile:(16,16), vgprs:2, align:2),
  tr("global_load_tr6_b96",  116, elem_bits: 6, tile:(16,32), vgprs:3, align:1),
  tr("global_load_tr4_b64",  115, elem_bits: 4, tile:(16,32), vgprs:2, align:2),
  // LDS (VDS encoding, DScnt), wave32 only, EXEC ignored (treated as all-ones)
  tr("ds_load_tr16_b128", 252, ..), tr("ds_load_tr8_b64", 253, ..), tr("ds_load_tr6_b96", 251, ..), tr("ds_load_tr4_b64", 250, ..),
];
```

| fact | value | source | notes |
|---|---|---|---|
| Purpose | load a 16x16 (8/16-bit) or 16x32 (4/6-bit) tile stored in the opposite major order and write it in WMMA VGPR order | CDNA5 ISA §10.9–10.9.2 (p. 137–138) | Choice rule: memory column-major + row-major VGPR layout → use TR. Same-order data → plain `global_load_b64/b128`. The rule is symmetric (p. 138). |
| Per-lane payload | TR16: 128 b per lane (4 VGPRs); TR8: 64 b (2); TR6: 96 b (3); TR4: 64 b (2) | CDNA5 ISA §10.9.2 (p. 138) | |
| Per-lane address map | example: one lane reads 64 contiguous bits that become K=0, M=0..7. The full map is in figures that are not in the text extraction. | CDNA5 ISA §10.9.2 (p. 137–138) | Exact lane → address rule: see Open (needs the PDF figures). |
| EXEC | global TR: EXEC==0 → acts as S_NOP, otherwise EXEC must be all ones. DS TR: EXEC ignored. | CDNA5 ISA §10.9.2 (p. 137); §11.2.4 (p. 154) | |
| Fields / counter | global TR: same fields as GLOBAL_LOAD_B64/B128, tracked by LOADcnt. DS TR: VDS fields, DScnt. | CDNA5 ISA §10.9.2 (p. 138); §11.2 (p. 149) | |
| Opcodes | VGLOBAL 87 tr16, 88 tr8, 115 tr4, 116 tr6; VDS 250 tr4, 251 tr6, 252 tr16, 253 tr8 | CDNA5 ISA §14.7.2 Table 116 (p. 206); §14.4 (p. 200); LLVM `AMDGPU/FLATInstructions.td:3683-3687`, `AMDGPU/DSInstructions.td:1549-1552` | LLVM 0x57/0x58/0x73/0x74 and 0xfa–0xfd. They agree. |
| LLVM availability | global/ds tr8/tr16: isGFX12Plus / isGFX1250Plus, wave32. tr4/tr6: `FeatureTransposeLoadF4F6Insts` (in the 12_50 common set, so also strict). | LLVM `AMDGPU/FLATInstructions.td:1328-1343`; `AMDGPU/DSInstructions.td:891-899`; `AMDGPU/GCNFeatures.td:1371-1373,2538` | |
| Aliases | `global_load_b128_tr_b16` etc. and `ds_load_b64_tr_b8` etc. map to the trN names; `ds_load_tr_b64/b128` → tr8/tr16 | LLVM `AMDGPU/FLATInstructions.td:3690-3693`; `AMDGPU/DSInstructions.td:1574-1581` | |
| Register alignment | tr6_b96 uses an unaligned 96-bit class; the others need an even VGPR | LLVM `AMDGPU/FLATInstructions.td:1342`; `AMDGPU/DSInstructions.td:896`; `MC/gfx1250_asm_load_tr.s:35,67` ("invalid register alignment") | |
| TR4/TR6 shape | §10.9.2: 16x32; Chapter 15 text: "16x32 or 32x16" | CDNA5 ISA §10.9.2 (p. 138); §15.15 (p. 749); §15.18.3 (p. 822–823) | Small ISA inconsistency. |
| Copy atom: DMA widths | async global↔LDS: B8, B32, B64, B128 per lane; cluster async: same; TDM: tile-based (see §7) | CDNA5 ISA §10.8 (p. 136); §10.7 (p. 135) | Replaces the gfx950 `buffer/global_load_lds_*` + M0 form. LLVM's legacy `global_load_lds_*` have no gfx1250 encoding (`AMDGPU/FLATInstructions.td:1217-1226`), derived. |

---

### 5. `mem.buffer`: buffer resource (V#) and buffer addressing

```rust
pub const VSHARP_GFX1250: Descriptor = Descriptor { bits: 128, sgprs: 4, sgpr_align: 4, fields: &[
    ("base",          0..=56),   // byte address, 57 bits
    ("num_records",  57..=101),  // 45 bits, bytes; 0 = unbound, all-ones = no OOB check
    ("reserved",    102..=107),
    ("stride",      108..=121),  // 14 bits, bytes, before scale
    ("stride_scale",122..=123),  // ×1, ×4, ×8, ×32
    ("swizzle_en",  124..=124),  // 1: AOS swizzle, index_stride 32, element 16 B
    ("oob_select",  125..=125),
    ("type",        126..=127),  // 0 = buffer
]};
pub const BUF_INST: BufInst = BufInst { ioffset_bits: 24, ioffset_max: 0x7F_FFFF /* non-negative */,
    soffset: "SGPR|M0|NULL", vaddr: "idx?, off?", addr_mask_bits: 45 };
```

| fact | value | source | notes |
|---|---|---|---|
| V# layout | as in the sketch | CDNA5 ISA §9.6 Table 57 (p. 123); §9.4 Table 54 (p. 119) | |
| num_records width | 45 bits (gfx1250). Earlier GFX12: 32 bits. | CDNA5 ISA §9.6 (p. 123); LLVM `AMDGPU/GCNFeatures.td:1647-1655,2574`; `docs/AMDGPUUsage.md:1980-1981` | |
| LLVM V# construction | low 64 bits = base (57) \| num_records[6:0]<<57. High 64 bits = num_records[44:7], 16-bit stride+scale at bit 108 (dword3 bit 12), 4-bit flags at bit 124. | LLVM `AMDGPU/SIISelLowering.cpp:13433-13471`; also `AMDGPU/AMDGPULegalizerInfo.cpp:6505-6508` | Agrees with the ISA, derived. |
| Stride scale | 0 ×1, 1 ×4 (max 65532), 2 ×8 (max 131064), 3 ×32 (max 524256) | CDNA5 ISA §9.6 (p. 123) | |
| OOB rule | swizzle 0, OOB_SELECT 0: OOB if total_offset + payload > num_records. OOB_SELECT 1: also if offset + payload > stride. Swizzle 1: no check. Loads return 0, stores dropped, checked per dword for B64–B128. | CDNA5 ISA §9.4.1 Table 56 (p. 120); §9.6 (p. 123) | Atomics are checked all-or-nothing (p. 121). |
| Address (linear) | base + soffset + ((ioffset + voff? + stride·vidx?) & (2^45−1)) | CDNA5 ISA §9.4 (p. 119) | |
| Swizzled address | index split by index_stride 32, element size 16 B | CDNA5 ISA §9.4.2 (p. 121–122) | |
| RSRC operand | 4 consecutive SGPRs, start a multiple of 4 in 0–120 | CDNA5 ISA §9.1 Table 48 (p. 115); §14.5.1 (p. 201) | |
| IOFFSET | 24-bit field, must be non-negative → 0..8388607 | CDNA5 ISA §9.1 Table 48 (p. 115); §14.5.1 (p. 201); LLVM `AMDGPU/SIInstrInfo.cpp:10751-10756`; `MC/gfx1250_asm_vbuffer_mubuf.s:4-5` (offset:8388607) | |
| VADDR usage | IDXEN/OFFEN select 0, 1 or 2 VGPRs (index first, then offset), unsigned | CDNA5 ISA §9.2 Table 50 (p. 117) | |
| VBUFFER cache bits | NV [7], SCOPE [51:50], TH [54:52]; no SCALE_OFFSET | CDNA5 ISA §14.5.1 Table 110 (p. 201); LLVM `MC/gfx1250_asm_vbuffer_mubuf.s:2308-2321` (nv) | LLVM's `supportsScaleOffset` returns false for buffer ops (`AMDGPU/Utils/AMDGPUBaseInfo.cpp:3613-3629`). Agrees. |
| Alignment modes | SH_MEM_CONFIG.alignment_mode: DWORD, DWORD_STRICT, STRICT, UNALIGNED; atomics always need natural alignment | CDNA5 ISA §9.5 (p. 122–123) | Kernel-mode config; the value used at runtime is unknown (no public source). |
| Data format field | ISA mentions "data-format INVALID = unbound", but Table 57 has no format field; VBUFFER instruction has FORMAT [61:55] | CDNA5 ISA §9.6 (p. 123); §14.5.1 (p. 201) | Small ISA inconsistency. Untyped loads take the format from the opcode (§9.3, p. 118). |
| SMEM buffer loads | use base, stride, num_records only; OOB if offset ≥ num_records | CDNA5 ISA §8.1 Table 45 (p. 108); §9.4.1.4 (p. 121) | |

---

### 6. `mem.addressing`: global / flat / scratch forms and offsets

```rust
pub enum VmemAddr {
    GV  { vaddr64 },                         // global/flat:  addr = V64 + ioff
    GVS { saddr64, voff32, scale_offset },   // global/flat:  addr = S64 + V32*scale + ioff
    GT  { saddr64 },                         // global only (addtid): addr = S64 + ioff + tid*4
    SV  { voff32, scale_offset },            // scratch: base + swz(V*scale + ioff)
    SS  { soff32 },                          // scratch: base + swz(S + ioff)
    SVS { soff32, voff32, scale_offset },    // scratch: base + swz(S + V*scale + ioff)
    ST  { },                                 // scratch: base + swz(ioff)   (ioff aligned to payload)
}
pub const IOFFSET_FLAT: Imm = Imm { bits: 24, signed: true };   // -8388608 ..= 8388607
pub const IOFFSET_DS:   Imm = Imm { bits: 16, signed: false };  // {offset1, offset0}
pub const IOFFSET_SMEM: Imm = Imm { bits: 24, signed: true };   // negative only for s_load (not s_buffer_load)
```

| fact | value | source | notes |
|---|---|---|---|
| Mode selection | Scratch: SVE + SADDR give SV, SS, SVS or ST. Flat/global: SADDR≠NULL → GVS, NULL → GV. GT: opcode-implied (ADDTID). | CDNA5 ISA §10.2 Table 60 (p. 130) | |
| Address equations | as in the sketch | CDNA5 ISA §10.2 (p. 130–131) | |
| IOFFSET | 24-bit signed for FLAT/GLOBAL/SCRATCH | CDNA5 ISA Ch. 10 fields (p. 126); §14.7 (p. 205); LLVM `AMDGPU/Utils/AMDGPUBaseInfo.cpp:3337-3343` and `AMDGPU/GCNFeatures.td:1905` (FeatureFlatOffsetBits24 in GFX12) | Test examples with negative offsets: `MC/gfx1250_asm_vflat.s:3074`. |
| SCALE_OFFSET (SO) | ScaleFactor = data size in bytes (1..16) when set. Applies to GVS (global/flat) and SV/SVS (scratch); ignored elsewhere. | CDNA5 ISA §10.2 (p. 130) | LLVM: SMEM non-buffer loads, scratch with vaddr, global with vaddr+saddr (`AMDGPU/Utils/AMDGPUBaseInfo.cpp:3613-3629`). Agrees. |
| GVS VGPR offset sign | ISA §10.2: signed 32-bit (VGPR_I32). §10.8 (async) and VADDR description: unsigned. | CDNA5 ISA §10.2 (p. 130); Ch. 10 fields (p. 126); §10.8 (p. 136) | LLVM treats it as signed on gfx1250 (`AMDGPU/GCNSubtarget.h:542` `hasSignedGVSOffset`). ISA inconsistency; see Open. |
| Scratch offsets | SGPR/VGPR offsets signed 32-bit; the sum inside SWIZZLE must be ≥ 0; ST mode: IOFFSET aligned to payload | CDNA5 ISA §10.2 (p. 130) | |
| Flat aperture | per-lane check: LDS (SHARED_BASE), scratch (PRIVATE_BASE[63:58]), hole → MEMVIOL, else global. LDS part: addr[31:0], range check < LDS size only. | CDNA5 ISA §10.2 (p. 131) | Flat ops count on LOADcnt/STOREcnt **and** DScnt (§10.1.1, p. 128). |
| Globally shared scratch | flat pointers into scratch allowed (4-byte swizzle, no 8-byte atomics). FLAT_SCRATCH_BASE readable as operand 230/231. No allocations in [−2^38, 2^38). | CDNA5 ISA §10.1.1 (p. 129); §10.2.1 (p. 131–132) | LLVM: `FeatureGloballyAddressableScratch` (`AMDGPU/GCNFeatures.td:2560`); `docs/AMDGPUUsage.md:1196` |
| VFLAT/VGLOBAL/VSCRATCH field bits | SADDR [6:0], NV [7], OP [20:13], ENC [31:24] (0xEC flat, 0xEE global, 0xED scratch), VDST [39:32], SO [48], SVE [49], SCOPE [51:50], TH [54:52], VSRC [62:55], VADDR [71:64], IOFFSET [95:72] | CDNA5 ISA §14.7 Table 114 (p. 204–205) | Confirmed by LLVM encodings, e.g. `MC/gfx1250_asm_vflat.s:2858-2886` (SO/SVE in byte 6). |
| Block load/store | `global/scratch_load/store_block`: up to 32 VGPRs, M0 bit-mask selects VGPRs; each step adds 4 to IOFFSET; one LOADcnt/STOREcnt per instruction | CDNA5 ISA §10.6 (p. 133–134); LLVM `AMDGPU/FLATInstructions.td:1318-1326` | Possible spill copy atom, derived. |
| Prefetch | `global/flat_prefetch_b8`; SCOPE picks the cache level (WGP vs GL2); no counter, no errors | CDNA5 ISA §10.5 (p. 133); TH meaning for prefetch §4.1.1 (p. 36) | |
| DS offsets | single-address ops: 16-bit unsigned {offset1,offset0}. 2ADDR: two 8-bit offsets ×4 (≤32-bit data) or ×8 (64-bit). | CDNA5 ISA §11.2 Table 68 (p. 149) | `Lds` field, cross-ref P1. |
| SMEM offsets | IOFFSET 24-bit signed (negative only for S_LOAD), SOFFSET SGPR/M0/NULL unsigned, SO only for non-buffer loads | CDNA5 ISA §8.1 Table 45 (p. 108); §14.2.1 Table 82 (p. 171) | |
| `global_atomic_ordered_add_b64` | ISA lists it as a GLOBAL op | CDNA5 ISA Ch. 10 Table 59 (p. 127) | **Conflict:** LLVM excludes gfx1250 (`AMDGPU/FLATInstructions.td:1228-1229`, `isGFX12PlusNot12_50`), and the ISA VGLOBAL opcode table (p. 205–206) has no opcode for it. |

---

### 7. `mem.async_lds`, `mem.cluster`, `mem.tdm`

```rust
pub const ASYNC_LDS: &[DmaForm] = &[
  // VGLOBAL encoding; addressing GV | GVS (+SO); LDS address = LDS_BASE + VGPR[VDST|VSRC][lane] + IOFFSET
  dma("global_load_async_to_lds_b{8,32,64,128}",   op: 95..=98,   dir: MemToLds, lds_addr: Vdst, counter: ASYNC, m0: None),
  dma("global_store_async_from_lds_b{8,32,64,128}",op: 99..=102,  dir: LdsToMem, lds_addr: Vsrc, counter: ASYNC, m0: None),
  dma("cluster_load_async_to_lds_b{8,32,64,128}",  op: 106..=109, dir: MemToLds, lds_addr: Vdst, counter: ASYNC,
      m0: Some(M0Use { wg_mask: 0..=15, early_timeout: 16 })),
];
pub const CLUSTER_LOAD: &[Load] = &[ load("cluster_load_b{32,64,128}", op: 103..=105, counter: LOAD, m0: wg_mask) ];
pub const TDM: Tdm = Tdm {
  ops: [("tensor_load_to_lds", 196), ("tensor_store_from_lds", 197)],  // VIMAGE, counter TENSOR, EXEC ignored, not in clauses
  groups: [G0 { sgprs: 4 }, G1 { sgprs: 8 }, G2 { sgprs: 4, optional }, G3 { sgprs: 4, optional }],
  max_dims: 5, data_size: [1, 2, 4, 8],
};
```

| fact | value | source | notes |
|---|---|---|---|
| Async load/store forms | global_load_async_to_lds_b8/32/64/128 and global_store_async_from_lds_b8/32/64/128; VGLOBAL encoding; GV or GVS; SVE unused (0); SO allowed | CDNA5 ISA §10.8 (p. 135–136); LLVM `AMDGPU/FLATInstructions.td:1252-1264,3674-3681`; `MC/gfx1250_asm_vflat.s:3286-3542` | Opcodes 95–102 agree (0x5f–0x66). |
| LDS address source | per-lane VGPR: VDST for loads, VSRC for stores. LDS addr = LDS_BASE + VGPR + IOFFSET (IOFFSET is added to both the LDS and the global address). M0 is not used. | CDNA5 ISA §10.8 (p. 136); §15.18.3 (p. 817–820); LLVM `AMDGPU/FLATInstructions.td:391-418,427-448` (comment: async loads do not use M0) | Asm: `global_load_async_to_lds_b32 vLDS, v[a:a+1], off` or `vLDS, vOff, s[b:b+1]`. Store: `vaddr, vLDS, saddr\|off`. |
| Async counter and order | ASYNCcnt. Async ops stay in order with each other toward global memory. LDS writes may be out of order (even load vs load). Loads complete in order; stores complete in order among stores; loads vs stores out of order. `ds_atomic_async_barrier_arrive_b64` also on ASYNCcnt, in order with async loads. | CDNA5 ISA §10.8 (p. 135–136) | `Counters.classes`: (Async, ASYNC, InOrder per direction), derived. LLVM does **not** insert `s_wait_asynccnt` / `s_wait_tensorcnt` automatically (`docs/AMDGPUUsage.md:14414-14416`). |
| Wave size | async, cluster and TDM ops: wave32 in LLVM | LLVM `AMDGPU/FLATInstructions.td:1241,1252,1310` | |
| Multicast (cluster) loads | cluster_load_b32/64/128 (LOADcnt, to VGPRs) and cluster_load_async_to_lds_b8..b128 (ASYNCcnt, to LDS). GV/GVS. M0[15:0] = workgroup mask within the cluster, M0[16] = early timeout. Mask 0 → plain load. Outside a cluster: plain global load. Always miss in WGP$. | CDNA5 ISA §10.7 (p. 134–135); LLVM `AMDGPU/FLATInstructions.td:1239-1246,1310-1316,2384-2392`; `MC/gfx1250_asm_vflat.s:3066-3134,3390-3490` | All participating waves should use the same M0. Only the requesting wave's counter is decremented (p. 135). The cluster topology itself is prose only (P0 design §4). |
| Cluster load opcodes | 103–105 (VGPR), 106–109 (async). 110–111 reserved. | CDNA5 ISA §14.7.2 Table 116 (p. 206); LLVM `AMDGPU/FLATInstructions.td:3666-3673` | Agree. |
| TDM instructions | tensor_load_to_lds (196), tensor_store_from_lds (197); VIMAGE encoding; operands are SGPR groups (VADDR0..3), no VGPRs, EXEC ignored, not allowed in a clause | CDNA5 ISA §10.11.1 Table 61 (p. 140); §14.6 Table 113 (p. 203); LLVM `AMDGPU/MIMGInstructions.td:2291-2362`; `MC/gfx1250_asm_vimage.s:5-43` | LLVM: 2-group form `s[0:3], s[4:11]` and 4-group form `+ s[12:15], s[16:19]`. LLVM sets VADDR4=NULL, dmask=1, dim=1, rsrc/vdata=0 (ISA: VADDR4 NULL, the other fields unspecified). |
| D# groups | G0 = 4 SGPRs (VADDR0); G1 = 8 SGPRs (VADDR1); G2 = 4 (VADDR2, NULL=0x7C to omit); G3 = 4 (VADDR3). ≤2D: G2/G3 NULL. Up to 5D: all four. VADDR2/3 must be both NULL or both set (VADDR3 may equal VADDR2). | CDNA5 ISA §10.11.1 (p. 140) | **Conflict:** LLVM accepts mixed `s[12:15], null` and `null, s[16:19]` (`MC/gfx1250_asm_vimage.s:33-39`). |
| G0 fields | [1:0] count (0 null tensor, 1 valid), [2] is_restore=0, [3] is_store=0, [4] nv=0 (the three restore-only bits must be 0 for user code), [6:5] scope, [9:7] th, [10] user_null, [30] gather index size (0 = 16-bit, 1 = 32-bit), [31] gather enable, [63:32] lds_addr (bytes), [120:64] global_addr (57-bit, tile start), [127:126] type = 2 | CDNA5 ISA §10.11.4 Table 62 (p. 144) | The scope/th inside G0 only apply on restore, derived. The instruction's own SCOPE/TH/NV fields apply to user ops (Table 61, p. 140). |
| G1 fields | [15:0] workgroup_mask (0 if not in a cluster), [17:16] data_size (1/2/4/8 B), [18] atomic_barrier_enable, [19] iterate_enable, [20] pad_enable, [21] early_timeout, [24:22] pad_interval (2^(v+1) DWORDs, 2..256), [31:25] pad_amount (v+1 DWORDs, 1..128), [47:32] atomic_barrier_address[18:3], [79:48] tensor_dim0, [111:80] tensor_dim1, [127:112] tile_dim0 (0 → NOP), [143:128] tile_dim1, [159:144] tile_dim2, [207:160] tensor_dim0_stride (48 b), [255:208] tensor_dim1_stride (48 b) | CDNA5 ISA §10.11.4 Table 63 (p. 144–145) | Units: elements of data_size. tile_dim1/2 = 0 means the dimension is unused. |
| G2 fields | [31:0] tensor_dim2; [63:32] tensor_dim3, or lds_addr_increment when iterating; [111:64] tensor_dim2_stride, or global_addr_increment when iterating; [127:112] tile_dim3, or iterate_count (v+1 iterations, up to 256) | CDNA5 ISA §10.11.4 Table 64 (p. 145–146) | |
| G3 fields | [47:0] tensor_dim3_stride, [79:48] tensor_dim4, [95:80] tile_dim4, [127:96] reserved | CDNA5 ISA §10.11.4 Table 65 (p. 146) | |
| Gather mode | 2D only. G2+G3 hold row indices: 16 × 16-bit or 8 × 32-bit. tile_dim1 = number of valid indices. Also works as scatter with tensor_store_from_lds. OOB is correct only with non-decreasing indices. | CDNA5 ISA §10.11.3.2 (p. 143); Tables 66–67 (p. 146–147) | |
| Iteration | 2D/3D only; iterate_count, lds/global increments; ignored in gather mode | CDNA5 ISA §10.11.3.1 (p. 143); Table 63 (p. 144) | |
| Padding | LDS-side only (load direction). pad_amount and pad_interval both 0 or both non-zero. tile_dim0 must be a multiple of 4 B for padding. Padding past the allocation is ignored. | CDNA5 ISA §10.11.2 (p. 142); §10.11.4 (p. 146) | |
| Address rule | global = global_addr + data_size·(x + y·stride0 + z·stride1 + …); LDS written sequentially from lds_addr (+ padding) | CDNA5 ISA §10.11.2 (p. 141–142) | |
| Completion | TENSORcnt, one decrement per instruction. In order with the same wave's tensor ops, unordered with other memory types. Optional `ds_atomic_async_barrier_arrive` to atomic_barrier_address after completion. | CDNA5 ISA §10.11.1 (p. 140); §10.11.3 (p. 142); §11.2.2.1 (p. 153) | LLVM `FeatureNeedsTDMDrain` "needs to keep tensor count low" in both gfx1250 and strict (`AMDGPU/GCNFeatures.td:1683-1686,2615,2654`); its numeric limit: see P2. |
| Multicast TDM | workgroup_mask ≠ 0 on a load → TDM uses CLUSTER_LOAD_ASYNC; ignored for stores; must be 0 outside a cluster (also G1[15:0]) | CDNA5 ISA §10.11.3 (p. 142); Table 61 (p. 140) | |
| Errors | OOB reads 0, OOB writes dropped. XNACK handled inside the TDM (errors imprecise). MEMVIOL for LDS out-of-range (if enabled) or a non-global address, reported once per op. VMID kill still decrements TENSORcnt. | CDNA5 ISA §10.11.5–10.11.6 (p. 147) | |
| `tensor_save` / `tensor_stop` | LLVM VFLAT ops 0x6e / 0x6f (HasMcastLoadInsts) | LLVM `AMDGPU/FLATInstructions.td:1248-1249,3652-3653` | Not in the public ISA. Context-switch helpers, derived. Not for codegen. |
| strict vs gfx1250 (memory) | strict adds `FeatureNeedsAligned2addrDS` and lacks `FeatureNoMTypeNC`. TDM, async, cluster and tr4/6 are common to both. | LLVM `AMDGPU/GCNFeatures.td:2494-2586,2588-2622,2624-2661,1688-1690` | |

---

### 8. `mem.cpol`: cache policy bits (SCOPE, TH, NV, SCALE_OFFSET)

```rust
pub const CPOL_GFX1250: CachePolicy = CachePolicy {
    scope: &[("WGP"/*LLVM SCOPE_CU*/,0), ("SE",1), ("DEV",2), ("SYS",3)],
    th_load:  &[("RT",0),("NT",1),("HT",2),("LU"/*LLVM BYPASS*/,3),("NT_RT",4),("RT_NT",5),("NT_HT",6)],     // 7 reserved
    th_store: &[("RT",0),("NT",1),("HT",2),("WB"/*LLVM BYPASS*/,3),("NT_RT",4),("RT_NT",5),("NT_HT",6),("NT_WB",7)],
    th_atomic: Bits { ret: 0, nt: 1, cascade: 2 },
    nv: true, scale_offset: true,
    bits: &[ (Vmem /*FLAT/GLOBAL/SCRATCH/BUFFER/IMAGE*/, Pos { nv: 7, scope: 50..=51, th: 52..=54, so: Some(48) /*not buffer*/ }),
             (Smem, Pos { nv: 20, scope: 21..=22, th: 23..=24 /*2 bits*/, so: Some(56) }) ],
    syncscope_map: &[("system", SYS), ("agent", DEV), ("cluster", SE), ("workgroup"|"wavefront"|"singlethread", WGP)],
};
```

| fact | value | source | notes |
|---|---|---|---|
| SCOPE values | 0 WGP, 1 SE, 2 DEV, 3 SYS; ops complete locally when ISA scope ≤ cache scope, else forwarded | CDNA5 ISA §4.1.1 (p. 35) | LLVM names: SCOPE_CU/SE/DEV/SYS = 0..3 (`AMDGPU/SIDefines.h:628-635`). |
| Release / acquire | WB / INV ops with a scope flush or invalidate caches whose scope is below it; GLOBAL_WB/WBINV/INV per scope | CDNA5 ISA §4.1.1 (p. 35–37, Table 13) | |
| TH for loads | 0 RT, 1 NT, 2 HT, 3 LU (last use), 4 NT_RT, 5 RT_NT, 6 NT_HT, 7 reserved | CDNA5 ISA §4.1.1 Table 10 (p. 36) | LLVM: value 3 printed as `TH_LOAD_BYPASS` when scope=SYS, else LU (`AMDGPU/SIDefines.h:613-619`; `MC/gfx1250_asm_vflat.s:2918,3066`). |
| TH for stores | 0 RT, 1 NT, 2 HT, 3 WB, 4 NT_RT, 5 RT_NT, 6 NT_HT, 7 NT_WB | CDNA5 ISA §4.1.1 Table 11 (p. 37) | LLVM `TH_STORE_BYPASS` = 3 with SYS (`MC/gfx1250_asm_vflat.s:3286`). |
| TH for atomics | bit0 return pre-op value, bit1 NT, bit2 cascade (deferred scope, non-returning only) | CDNA5 ISA §4.1.1 Table 12 (p. 37); LLVM `AMDGPU/SIDefines.h:623-626` | |
| Prefetch TH | TH codes 0–6 select speculative / non-speculative + temporal class for GL2 prefetch | CDNA5 ISA §4.1.1 (p. 36) | |
| SMEM SCOPE / TH | same 2-bit SCOPE; 2-bit TH (values 0–3, uniform across levels); scope WGP SMEM is not coherent with VMEM stores | CDNA5 ISA §4.1.1 (p. 37); §14.2.1 Table 82 (p. 171) | |
| NV bit | marks non-volatile data; WB/INV with NV=0 only touch NV=0 lines; a line seen as both is NV=0; flat adds MODE.FLAT_SCRATCH_IS_NV | CDNA5 ISA §4.1.2 (p. 38–39) | LLVM `CPol::NV` = 1<<5 (`AMDGPU/SIDefines.h:637`); asm `nv`, gfx1250+ only (`AMDGPU/AsmParser/AMDGPUAsmParser.cpp:5350-5362`); `MC/gfx1250_asm_vflat.s:2818-2854`, `MC/gfx1250_asm_smem.s:6-12`. LLVM does not yet document NV code sequences (`docs/AMDGPUUsage.md:14362-14367`). |
| SCALE_OFFSET bit | VMEM bit 48 (VFLAT/VGLOBAL/VSCRATCH), SMEM bit 56; not in VBUFFER | CDNA5 ISA §14.7 Table 114 (p. 204); §14.2.1 Table 82 (p. 171); §14.5.1 (p. 201) | LLVM `CPol::SCAL` = 1<<9, asm `scale_offset` (`AMDGPU/SIDefines.h:652`; `AMDGPU/AsmParser/AMDGPUAsmParser.cpp:5365-5369,7513-7517`). |
| Bit positions VMEM | NV [7], SCOPE [51:50], TH [54:52] in VBUFFER, VIMAGE, VFLAT/VGLOBAL/VSCRATCH | CDNA5 ISA §14.5.1 (p. 201); §14.6 (p. 203); §14.7 (p. 204) | Checked against LLVM encodings: `th:TH_LOAD_NT_HT scope:SCOPE_DEV` → byte 6 = 0x68 (`MC/gfx1250_asm_vflat.s:2950`). |
| syncscope → SCOPE (LLVM) | system/one-as → SYS, agent → DEV, cluster → SE, workgroup/wavefront/singlethread → CU (WGP, default, omitted in text) | LLVM `docs/AMDGPUUsage.md:14514-14537` | |
| LLVM cache-fill-size bits | `CPol::CFS` (bits 7–8) exist in LLVM | LLVM `AMDGPU/SIDefines.h:641-648`; `AMDGPU/AsmParser/AMDGPUAsmParser.cpp:5340-5347` | Not in the CDNA5 ISA. Whether gfx1250 has them: unknown from the read files (no gfx1250 test uses `cfs`). Do not emit. |

---

### Open

1. **Dense 8-bit vs block-scale F8 lane layout** (p. 100 vs p. 105) and **sparse lane split** (p. 101–102 pseudocode vs the dense
   16-bit rule on p. 100). In one the lane half owns an 8-wide K block, in the other a 16-wide block. Which one the hardware uses for
   `v_wmma_*_16x16x{64,128}_fp8*`, `_iu8` and the packed sparse A is unknown (no public source beyond the ISA). The ISA points to the AMD Matrix Instruction
   Calculator (p. 94–95), which would settle it but is outside the allowed sources. An llvm-mc probe cannot settle layouts.
2. **index_key semantics** for gfx1250 SWMMAC: LLVM accepts 1 bit (OPSEL[0]). The ISA requires OPSEL[1:0]=0 and defines no index selector. Also open: the
   content of the 2nd index VGPR for the 8-bit forms. Needs an ISA errata or LLVM intrinsic docs (`IntrinsicsAMDGPU.td`, outside the
   checkout).
3. **bf16f32 SWMMAC** accumulator storage (BF16 result in an F32-sized VDST?). The ISA does not say.
4. **Encoding conflicts** to probe with the rocm-10.1 llvm-mc (`-mcpu=gfx1250 -show-encoding`, and `--disassemble` of ISA-conformant
   bytes):
   - OPSEL_HI = 7 (LLVM) vs 0 (ISA) on the `v_wmma_*32x16x128_f4` forms.
   - VOP3PX2 constant field 0x080 (LLVM) vs 0x100 (ISA).
   - VOP3PX2 opcode 0x88 missing from ISA Table 97–98.
   - A/B neg accepted on 16-bit SWMMAC.
   - C inline constants other than 0.
   - Mixed NULL D# groups on TDM.
5. **Transpose-load per-lane address map**: the ISA figures (p. 137–138) are not in the text extraction. Read the PDF figures. The exact
   TR4/TR6 tile orientation (16x32 vs 32x16) also differs between §10.9 and Chapter 15.
6. **GVS VGPR offset sign** (ISA §10.2 signed vs §10.8 unsigned; LLVM signed): needs an ISA errata or a hardware test (not allowed here).
7. **`global_atomic_ordered_add_b64`**: listed in ISA Table 59 but has no ISA opcode, and LLVM excludes gfx1250. Treat as absent until an ISA errata.
8. **LLVM-only ops** (`v_wmma_ld_scale*_paired_*`, `tensor_save`, `tensor_stop`) and **CFS bits**: no public ISA description. Do not use
   them in codegen.
9. **TH value 3 naming** (ISA LU/WB vs LLVM BYPASS with SYS scope): same bits. Confirm that LLVM printing matches the intended semantics for the
   non-SYS scopes with an llvm-mc round trip.
10. **Runtime config** (SH_MEM_CONFIG.alignment_mode, LDS_CONFIG.ADDR_OUT_OF_RANGE_REPORTING, TDM_CONTROL, multicast timeout): unknown (no
    public source).

## 7. gfx1250 target description — P5: `emit`, `llvm`, `verify`, stepping, ISA ↔ gfx1250 correspondence

Scope: the `Emission`, `LlvmTail` and `VerifyLimits` fields of `Target` (see `target-design.md` §2), plus the public evidence on
variants (gfx1250 / gfx1250-strict / gfx1251), steppings and how the public CDNA5 ISA document relates to the LLVM processor.

Sources (only these):
- **ISA** = AMD "CDNA5" Instruction Set Architecture Reference Guide, 27-July-2026 (public PDF). Cited as `CDNA5 ISA §x (p. N)`, N = printed page "N of 823".
- **LLVM** = upstream llvm-project @274ab686 (sparse checkout). Cited as `LLVM <path>:<line>`; paths are relative to `llvm/` unless they start with `docs/` or `test/` (also under `llvm/`). `T/` abbreviates `lib/Target/AMDGPU/`.

Facts are paraphrased. "derived" marks my own inference. "ISA silent" means the ISA document does not address the point.

---

### 1. `emit: Emission`

#### 1.1 Target identity

```rust
emit: Emission {
    mcpu: "gfx1250",                       // alt. "gfx1250-strict" (separate ELF mach, separate subarch)
    triple: "amdgcn-amd-amdhsa",           // legacy spelling; new spelling "amdgpu12.50-amd-amdhsa"
    subarch: "amdgpu12.50",                // major subarch "amdgpu12.5" (shared with gfx1251 / gfx12-5-generic)
    isa_version: (12, 5, 0),
    elf: Elf { class: ELFCLASS64, machine: EM_AMDGPU /*224*/, osabi: ELFOSABI_AMDGPU_HSA /*64*/,
               mach: 0x049 /*EF_AMDGPU_MACH_AMDGCN_GFX1250*/, xnack: XnackSel::OnOffAny, sramecc: None /*hardwired*/ },
    code_object: CodeObject { default: V5, supported: &[V4, V5, V6] },
    ..
}
```

| fact | value | source | notes |
|---|---|---|---|
| Processor names | `gfx1250`, `gfx1250-strict`, `gfx1251`, generic `gfx12-5-generic` | LLVM `T/GCNProcessors.td:310-326` | gfx12-5-generic covers gfx1250 and gfx1251 only (`CoveredGPUs`, :325), **not** gfx1250-strict. |
| ISA version triple (major, minor, stepping) | gfx1250 = [12,5,0]; gfx1250-strict = [12,5,0]; gfx1251 = [12,5,1]; gfx12-5-generic = [12,5,0] | LLVM `T/GCNProcessors.td:311,316,320,324` | The third number is the LLVM ISA "stepping"; nothing public maps it to a silicon revision (see §4). |
| Scheduling model | gfx1250 and gfx1250-strict: `GFX1250SpeedModel`; gfx1251: `GFX1251SpeedModel` | LLVM `T/GCNProcessors.td:310,315,319` | strict-only latency variants are predicated inside the same model (`T/SISchedule.td:476-500`). |
| Triple subarch spelling | `amdgpu12.50` (gfx1250), `amdgpu12.51` (gfx1251), `amdgpu12.50s` (gfx1250-strict; explicit `SubArchSpelling = "12.50s"`) | LLVM `docs/AMDGPUUsage.md:105-113`; `T/GCNProcessors.td:312`; `T/AMDGPUTargetParser.td:39-41` | The `amdgcn` architecture name is a legacy alias of `amdgpu` (`docs/AMDGPUUsage.md:32-51`). |
| Major subarches | `amdgpu12.5` (default/generic processor gfx12-5-generic, covers 12.50 and 12.51); `amdgpu12.5s` (default gfx1250-strict, covers only 12.50s) | LLVM `docs/AMDGPUUsage.md:108-113` | derived: gfx1250-strict is its own family for linking; a 12.50s object is never compatible with 12.50/12.51 objects (compatibility rule `docs/AMDGPUUsage.md:178-191`). |
| CPU from triple | if `-mcpu` is empty the CPU is taken from the subarch; a CPU whose subarch (or its major) differs from the triple subarch is a fatal usage error | LLVM `T/MCTargetDesc/AMDGPUMCTargetDesc.cpp:82-87`; `T/AMDGPUTargetMachine.cpp:679-685,723-730` | derived: `-mtriple=amdgpu12.50 -mcpu=gfx1250-strict` is rejected; use `amdgpu12.50s` or the legacy `amdgcn` triple. MC tests drive gfx1250 with `-triple=amdgpu12.50` only (`test/MC/AMDGPU/gfx1250_asm_ds.s:1`) and strict with `-triple=amdgpu12.50s` (`test/MC/AMDGPU/gfx1250-strict_err.s:2`). |
| dGPU / APU column | gfx1250-strict: **dGPU**; gfx1250: **APU**; gfx1251: **APU** | LLVM `docs/AMDGPUUsage.md:1173-1220` | The only public text distinguishing the variants by device class. ISA silent. |
| Target features (target-id) | `xnack` only (no `cumode`, no `wavefrontsize64`, no `sramecc`) | LLVM `docs/AMDGPUUsage.md:1177,1193,1209` | Wave64 is rejected by the assembler (`test/MC/AMDGPU/gfx1250_asm_load_tr.s:3`, WAVESIZE-ERR). ISA: device runs wave32 only (CDNA5 ISA §1 (p. 3)). |
| Target properties | Architected flat scratch; Packed work-item IDs; Globally Accessible Scratch; Workgroup Clusters | LLVM `docs/AMDGPUUsage.md:1178-1181,1194-1197` | ISA: clusters §2.3 (p. 10); architected scratch base §3.5.2 (p. 29). |
| OS support column | empty for all three | LLVM `docs/AMDGPUUsage.md:1182,1198,1214` | derived: no published runtime support statement in the doc. |
| Product names | *TBA* (with a `todo: Add product names`) | LLVM `docs/AMDGPUUsage.md:1183-1188,1199-1204,1215-1220` | ISA names no product; it only says "CDNA5 devices" (CDNA5 ISA Preface (p. 1); §1 (p. 3)). |
| gfx12-5-generic | functionally equivalent to gfx1250; target feature `xnack`; requires code object V6 | LLVM `docs/AMDGPUUsage.md:1391-1398,1250-1251`; `T/GCNFeatures.td:2707-2726` (`FeatureRequiresCOV6` at :2717) | Its feature set lacks WMMA F4 / 16x16x128 FP8 / WMMAN16 (generic = intersection-like). |

#### 1.2 ELF header and `e_flags`

```rust
elf: Elf {
    machine: 224, osabi: 64,
    abiversion: |cov| match cov { V4 => 2, V5 => 3, V6 => 4 },
    e_flags: mach(0x049) | xnack_v4(any=0x100 | off=0x200 | on=0x300) /* sramecc field = 0 */,
}
```

| fact | value | source | notes |
|---|---|---|---|
| `EM_AMDGPU` | 224 | LLVM `docs/AMDGPUUsage.md:3795-3796` | |
| `ELFOSABI_AMDGPU_HSA` | 64 | LLVM `docs/AMDGPUUsage.md:3799-3800` | |
| `EI_ABIVERSION` | HSA V4 = 2, V5 = 3, V6 = 4 | LLVM `docs/AMDGPUUsage.md:3807-3814` | |
| `EF_AMDGPU_MACH` mask | 0x0ff | LLVM `docs/AMDGPUUsage.md:3993-3995` | |
| `EF_AMDGPU_MACH_AMDGCN_GFX1250` | **0x049** | LLVM `docs/AMDGPUUsage.md:4229-4231`; `T/MCTargetDesc/AMDGPUTargetStreamer.cpp:138` | |
| `EF_AMDGPU_MACH_AMDGCN_GFX1250_STRICT` | **0x0eb** | LLVM `docs/AMDGPUUsage.md:4313-4315`; `T/MCTargetDesc/AMDGPUTargetStreamer.cpp:137` | Far from the dense 0x04x-0x06x range; ISA silent. |
| `EF_AMDGPU_MACH_AMDGCN_GFX1251` | 0x05a | LLVM `docs/AMDGPUUsage.md:4280-4282` | |
| `EF_AMDGPU_MACH_AMDGCN_GFX12_5_GENERIC` | 0x05b | LLVM `docs/AMDGPUUsage.md:4283-4285` | Plus generic version in bits 31:24 (`docs/AMDGPUUsage.md:4026-4030`; `T/MCTargetDesc/AMDGPUTargetStreamer.cpp:910-928`). |
| XNACK field (V4+) | mode selectable: ANY 0x100 / OFF 0x200 / ON 0x300 (mask 0x300) | LLVM `docs/AMDGPUUsage.md:3996-4010`; `T/MCTargetDesc/AMDGPUTargetStreamer.cpp:841-859`; `T/GCNFeatures.td:2619-2620,2658-2659` (`FeatureXNACKOnOffModes` on both variants) | Conflict inside LLVM: a comment calls gfx1250 "hardwired xnack on" (`T/Disassembler/AMDGPUDisassembler.cpp:126-129`) but the same code only treats a target as hardwired when it lacks `XNACKOnOffModes`, which gfx1250 has. The feature list wins (derived). |
| SRAMECC field (V4+) | 0 (UNSUPPORTED_V4): SRAMECC supported but not mode-selectable, so it is implied by MACH | LLVM `T/GCNFeatures.td:2551` (`FeatureSupportsSRAMECC`, no `SRAMECCOnOffModes` in 12_50 lists); `T/MCTargetDesc/AMDGPUTargetStreamer.cpp:860-878`; `docs/AMDGPUUsage.md:4038-4042` | derived: zero does not mean SRAMECC off. ISA silent on SRAMECC modes. |

#### 1.3 Code object versions

| fact | value | source | notes |
|---|---|---|---|
| Emittable | V4, V5, V6 (V2/V3 can no longer be emitted) | LLVM `docs/AMDGPUUsage.md:3848-3861` | |
| Default | V5 | LLVM `docs/AMDGPUUsage.md:3855-3858` | |
| gfx1250-specific minimum | none found; only `gfx12-5-generic` requires V6 | LLVM `T/GCNFeatures.td:2717`; `docs/AMDGPUUsage.md:1250-1251` | derived: pick V5 (gfx950 parity) or V6 (needed for `.cluster_dims` metadata, below). |
| V6 metadata addition | `.cluster_dims` (3 integers) | LLVM `docs/AMDGPUUsage.md:7063-7076` | Clusters are prose-only in our Target (target-design §4). |
| Kernel descriptor | 64 bytes, 64-byte aligned; entry point 256-byte aligned | LLVM `docs/AMDGPUUsage.md:7283-7284,7324-7325,7434-7435` | Same layout as gfx950 (Code Object V3+ KD). ISA §2.4 (p. 11) separately requires shader padding — see P4/P1. |

#### 1.4 Kernel descriptor fields for GFX12.5

```rust
kd: KdRules {
    vgpr_blocks: |next_free_vgpr| max(0, ceil(next_free_vgpr.max(1) / 16) - 1),   // 6 bits, max_vgpr 1..=1024
    sgpr_blocks: |_| 0,                                                           // reserved on GFX10+ (fixed SGPRs)
    lds_units: 2048,                     // GRANULATED_LDS_SIZE unit (CP fills; KD value must be 0)
    user_sgpr_count: Field { lo: 1, width: 6, max: 32 },   // RSRC2[6:1] on GFX125*
    named_bar_cnt: Field { rsrc3_bits: 14..=16, unit: 4, max: 4 },               // 0..16 barriers
    wgp_mode: None,                      // not available on gfx1250
    accum_offset: None, tg_split: None, shared_vgpr_count: None,
    dynamic_vgpr: KdBit { rsrc3_bit: 17, directive: None },   // no .amdhsa_ directive
    wavefront_size32: true,              // set ENABLE_WAVEFRONT_SIZE32 (derived)
}
```

| fact | value | source | notes |
|---|---|---|---|
| `GRANULATED_WORKITEM_VGPR_COUNT` (RSRC1[5:0]) | GFX125X wave32: max_vgpr 1..1024, value = max(0, ceil(vgprs_used/16) − 1) | LLVM `docs/AMDGPUUsage.md:7474-7476`; `T/Utils/AMDGPUBaseInfo.cpp:1319-1332` (encoding granule 16 with `FeatureVGPRMSBs`) | ISA agrees on 16-VGPR blocks and 1024 max (CDNA5 ISA §3.3.2.1 (p. 16)). 6-bit field: 63 → 1024 VGPRs fits exactly (derived). |
| `GRANULATED_WAVEFRONT_SGPR_COUNT` (RSRC1[9:6]) | reserved, must be 0 (GFX10-GFX12: 128 SGPRs always allocated) | LLVM `docs/AMDGPUUsage.md:7496-7497` | ISA: fixed allocation of 106 SGPRs + VCC + 16 TTMPs (CDNA5 ISA §3.3.1.1 (p. 14)). The "128" is LLVM's wording; ISA count is 124 named (derived; no conflict for codegen). |
| RSRC1 bit 21 | `WG_RR_EN` on GFX12 (round-robin scheduling); `.amdhsa_round_robin_scheduling` | LLVM `docs/AMDGPUUsage.md:7570-7574,17899-17902`; `T/AsmParser/AMDGPUAsmParser.cpp:6483-6488` | ISA silent. |
| RSRC1 bit 23 | reserved on GFX12 (no IEEE mode); `.amdhsa_ieee_mode`/`.amdhsa_dx10_clamp` rejected | LLVM `docs/AMDGPUUsage.md:7599-7601`; `T/AsmParser/AMDGPUAsmParser.cpp:6388-6401` (needs `FeatureDX10ClampAndIEEEMode`, absent from the GFX12 generation `T/GCNFeatures.td:1881-1903`) | **Removed field** vs gfx950. |
| RSRC1 bit 27 | GFX125*: `FLAT_SCRATCH_IS_NV` (new) | LLVM `docs/AMDGPUUsage.md:7631-7644`; `T/Disassembler/AMDGPUDisassembler.cpp:2697-2704` (printed only as a comment; no directive) | ISA has a matching per-wave control of the same name (CDNA5 ISA §3.4 (p. 23); §4.1.2 (p. 38)). Agree. |
| RSRC1 bit 29 `WGP_MODE` | not available on gfx1250; `.amdhsa_workgroup_processor_mode` rejected | LLVM `T/Disassembler/AMDGPUDisassembler.cpp:2711-2715`; `T/Utils/AMDGPUBaseInfo.cpp:2521-2525`; `T/AsmParser/AMDGPUAsmParser.cpp:6413-6419` | Doc table still says "GFX10-GFX12" for this bit (`docs/AMDGPUUsage.md:7655-7661`) — doc imprecise, code is authoritative (derived). **Removed field.** |
| RSRC1 bits 30/31 | `MEM_ORDERED`, `FWD_PROGRESS` (directive defaults 1) | LLVM `docs/AMDGPUUsage.md:7662-7685,17915-17922` | ISA silent on KD. |
| RSRC2 `USER_SGPR_COUNT` | GFX125*: 6 bits at [6:1] (was 5 bits [5:1]); up to 32 user SGPRs | LLVM `docs/AMDGPUUsage.md:7733-7740`; `T/Utils/AMDGPUBaseInfo.cpp:2420-2424`; `T/AsmParser/AMDGPUAsmParser.cpp:6552-6564` | ISA agrees: CS loads 0..32 user data SGPRs (CDNA5 ISA §3.5.3.1 Table 9 (p. 30)). Consequence: RSRC2 bit 6 is no longer `ENABLE_DYNAMIC_VGPR`/trap bit on GFX125* (derived from `docs/AMDGPUUsage.md:7719-7740`). |
| RSRC2 `GRANULATED_LDS_SIZE` [23:15] | must be 0 in the KD (CP fills from the dispatch); unit on GFX125* = 512×4 = **2048 bytes** | LLVM `docs/AMDGPUUsage.md:7788-7812`; `T/GCNFeatures.td:2629` (`FeatureLDSEncodingGranularity2048`) | ISA: LDS allocated in 2048-byte blocks (CDNA5 ISA §3.3.4 (p. 19)). Agree. LDS size itself goes to `GROUP_SEGMENT_FIXED_SIZE` in bytes (`docs/AMDGPUUsage.md:7298-7302`). |
| RSRC2 private segment | `.amdhsa_enable_private_segment` (architected flat scratch); `.amdhsa_system_sgpr_private_segment_wavefront_offset`, `.amdhsa_user_sgpr_private_segment_buffer`, `.amdhsa_user_sgpr_flat_scratch_init`, `.amdhsa_reserve_flat_scratch` are errors | LLVM `T/AsmParser/AMDGPUAsmParser.cpp:6203-6213,6266-6276,6296-6312,6351-6360`; `T/GCNFeatures.td:2509` | **Removed directives** vs non-architected targets (gfx950 is architected too — same rule). |
| RSRC3 `INST_PREF_SIZE` [11:4] | 0..255 in 128-byte units; `.amdhsa_inst_pref_size` | LLVM `docs/AMDGPUUsage.md:7978-7982,17927-17930`; `T/AsmParser/AMDGPUAsmParser.cpp:6441-6452` | ISA silent. |
| RSRC3 `GLG_EN` [13] | group launch guarantee | LLVM `docs/AMDGPUUsage.md:7987-7990` | No directive; ISA silent. |
| RSRC3 `NAMED_BAR_CNT` [16:14] (new) | GFX125*: named barriers per workgroup in units of 4, range 0-4 (0..16 barriers); directive **`.amdhsa_named_barrier_count`** (default 0, "requires gfx1250+") | LLVM `docs/AMDGPUUsage.md:7991-8001,17808-17811`; `T/AsmParser/AMDGPUAsmParser.cpp:6343-6346,6602-6606`; `T/AMDGPUAsmPrinter.cpp:1550-1554` | ISA agrees: up to 16 extra barriers per work-group, from a pool of 64 per WGP (CDNA5 ISA §5.6.2 (p. 47)). Whether the directive value is the raw field (0-4) or a barrier count is not stated by the doc; the parser stores the expression directly into the field (derived: raw field value). Probe in Open. |
| RSRC3 `ENABLE_DYNAMIC_VGPR` [17] (new) | GFX125*: dynamic-VGPR mode bit | LLVM `docs/AMDGPUUsage.md:8002-8013`; `T/Disassembler/AMDGPUDisassembler.cpp:2846-2847` (comment only) | **No `.amdhsa_` directive** exists (`T/AsmParser/AMDGPUAsmParser.cpp:6186-6491` has none). LLVM attribute `amdgpu-dynamic-vgpr-block-size` ∈ {16, 32} (`docs/AMDGPUUsage.md:3474-3480`; `T/Utils/AMDGPUBaseInfo.cpp:2377-2385`), max 8 blocks (`T/Utils/AMDGPUBaseInfo.h:253`). `s_alloc_vgpr` assembles for gfx1250 (`test/MC/AMDGPU/gfx12_asm_sop1.s:3,6-7`; `T/SOPInstructions.td:438-444`). **ISA silent**: no dynamic-VGPR mode, no `S_ALLOC_VGPR` in the ISA text. derived: `dynamic_vgpr: false` always. |
| RSRC3 `TCP_SPLIT` [20:18] (new) | preferred LDS / vector-cache split of the TCP (0 = no preference; 1..7 = LDS 0…384 kB) | LLVM `docs/AMDGPUUsage.md:8014-8023`; `T/Disassembler/AMDGPUDisassembler.cpp:2848-2849` | No directive. ISA only says LDS is carved out of the WGP cache (CDNA5 ISA §3.3.4.1 (p. 19)). Values 7 (384 kB) exceed the 320 KiB LDS limit — unexplained (Open). |
| RSRC3 `ENABLE_DIDT_THROTTLE` [21] (new) | DIDT throttling for ACE pipes | LLVM `docs/AMDGPUUsage.md:8024-8033` | No directive; ISA silent. |
| RSRC3 `SHARED_VGPR_COUNT`, `ACCUM_OFFSET`, `TG_SPLIT` | not on gfx1250 (`.amdhsa_shared_vgpr_count` gfx10/11 only; `.amdhsa_accum_offset`/`.amdhsa_tg_split` require `GFX90AInsts`) | LLVM `T/AsmParser/AMDGPUAsmParser.cpp:6339-6342,6408-6412,6432-6440`; `T/Utils/AMDGPUBaseInfo.cpp:2553-2555` | Directive table says `.amdhsa_tg_split` is "GFX11-GFX12" (`docs/AMDGPUUsage.md:17907-17910`) but the parser rejects it without GFX90A insts — **doc/code conflict**; code wins (derived). **Removed vs gfx950** (`accum_offset`, `tg_split`). |
| `ENABLE_WAVEFRONT_SIZE32` (KD bit 458) | must be 1 for wave32 (`.amdhsa_wavefront_size32 1`) | LLVM `T/AsmParser/AMDGPUAsmParser.cpp:6284-6291`; `docs/AMDGPUUsage.md:17800-17803` | KD table only documents GFX10-GFX11 for this bit (`docs/AMDGPUUsage.md:7396-7404`) — doc silent for GFX12; derived: set it. |
| Kernarg preload (KD 470:464, 479:471) | supported (`FeatureKernargPreload`); length ≤ 32 user SGPRs, offset < 1024 dwords | LLVM `T/GCNFeatures.td:2561`; `T/AsmParser/AMDGPUAsmParser.cpp:6214-6237` | Doc lists only GFX90A/GFX942 (`docs/AMDGPUUsage.md:7414-7429,17959-17966`) — doc silent on GFX12.5. ISA silent. |
| Assembler-listing comment | `; NamedBarCnt: <n>` emitted for gfx1250 | LLVM `T/AMDGPUAsmPrinter.cpp:1070-1073` | |
| `.amdhsa_reserve_xnack_mask` | must equal target-id xnack setting | LLVM `T/AsmParser/AMDGPUAsmParser.cpp:6361-6371` | GFX10+ adds no extra SGPRs for VCC/XNACK/flat-scratch beyond VCC 2 (`T/Utils/AMDGPUBaseInfo.cpp:1261-1269`). |

---

### 2. `llvm: LlvmTail`

```rust
llvm: LlvmTail {
    mcpu: "gfx1250",
    triple: "amdgcn-amd-amdhsa",          // or "amdgpu12.50-amd-amdhsa"; never mix with gfx1250-strict
    mattr: "",                            // wavefront size 32 is the processor default
    dialect: Dialect::Llvm24,             // derived: needs an LLVM with 12.50 support (@274ab686 or a ROCm build of it)
    llc_args: &["-mtriple=amdgcn-amd-amdhsa", "-mcpu=gfx1250", "--amdhsa-code-object-version=5"],
}
```

#### 2.1 Processor definitions

| fact | value | source | notes |
|---|---|---|---|
| `gfx1250` | `FeatureISAVersion12_50` = Common (91 features) + 35 | LLVM `T/GCNProcessors.td:315-317`; `T/GCNFeatures.td:2624-2661` | |
| `gfx1250-strict` | `FeatureISAVersion12_50_STRICT` = Common + 32 | LLVM `T/GCNProcessors.td:310-313`; `T/GCNFeatures.td:2588-2622` | |
| shared base | `FeatureISAVersion12_50_Common` | LLVM `T/GCNFeatures.td:2494-2586` | Also base of gfx1251 and gfx12-5-generic (:2663-2726). |

#### 2.2 Common feature list (both variants)

| feature | name string | LLVM description (one line) | definition |
|---|---|---|---|
| `FeatureGFX12` | `gfx12` | GFX12 generation (wave32 support, VOPD, true16, 24-bit flat offsets, max 16 waves/EU, …) | `T/GCNFeatures.td:1881` |
| `FeatureGFX1250Insts` | `gfx1250-insts` | Additional instructions for GFX1250+ | `T/GCNFeatures.td:645` |
| `FeatureBackOffBarrier` | `back-off-barrier` | s_barrier can back off if an exception occurs | `T/GCNFeatures.td:1530` |
| `FeatureRequiresAlignedVGPRs` | `vgpr-align2` | VGPR tuple operands require even alignment | `T/GCNFeatures.td:483` |
| `FeatureCuMode` | `cumode` | CU wavefront execution mode | `T/GCNFeatures.td:258` |
| `FeatureVGPRMSBs` | `vgpr-msbs` | MSB registers address VGPRs beyond the encoding | `T/GCNFeatures.td:1392` |
| `Feature64BitLiterals` | `64-bit-literals` | 64-bit literals with single-DWORD instructions | `T/GCNFeatures.td:1388` |
| `FeatureDLInsts` | `dl-insts` | v_fmac_f32, v_xnor_b32 | `T/GCNFeatures.td:874` |
| `FeatureFmacF64Inst` | `fmacf64-inst` | v_fmac_f64 | `T/GCNFeatures.td:878` |
| `FeaturePackedFP32SingleSGPROps` | `packed-fp32-single-sgpr-ops` | packed fp32 ops with an SGPR operand propagate low→high | `T/GCNFeatures.td:807` |
| `FeatureDot7Insts` | `dot7-insts` | v_dot4_u32_u8, v_dot8_u32_u4 | `T/GCNFeatures.td:906` |
| `FeatureDot8Insts` | `dot8-insts` | v_dot4_i32_iu8, v_dot8_i32_iu4 | `T/GCNFeatures.td:910` |
| `FeatureWavefrontSize32` | `wavefrontsize32` | wave size 32 | `T/AMDGPUFeatures.td:102` |
| `FeatureShaderCyclesHiLoRegisters` | `shader-cycles-hi-lo-registers` | SHADER_CYCLES_HI/LO registers | `T/GCNFeatures.td:1256` |
| `FeatureArchitectedFlatScratch` | `architected-flat-scratch` | flat scratch is a read-only SPI-initialized register | `T/GCNFeatures.td:1558` |
| `FeatureArchitectedSGPRs` | `architected-sgprs` | architected SGPRs (workgroup IDs in TTMPs) | `T/GCNFeatures.td:1563` |
| `FeatureFlatGVSMode` | `flat-gvs-mode` | GVS addressing for flat_* | `T/GCNFeatures.td:144` |
| `FeatureAtomicFaddRtnInsts` | `atomic-fadd-rtn-insts` | buffer/global atomic add f32 with return | `T/GCNFeatures.td:1086` |
| `FeatureAtomicFaddNoRtnInsts` | `atomic-fadd-no-rtn-insts` | buffer/global atomic add f32 without return | `T/GCNFeatures.td:1116` |
| `FeatureAtomicDsPkAdd16Insts` | `atomic-ds-pk-add-16-insts` | ds_pk_add_{bf16,f16}[_rtn] | `T/GCNFeatures.td:1064` |
| `FeatureAtomicFlatPkAdd16Insts` | `atomic-flat-pk-add-16-insts` | flat_atomic_pk_add_{f16,bf16} | `T/GCNFeatures.td:1078` |
| `FeatureAtomicBufferGlobalPkAddF16Insts` | `atomic-buffer-global-pk-add-f16-insts` | buffer/global atomic pk_add_f16 (can return) | `T/GCNFeatures.td:1133` |
| `FeatureAtomicGlobalPkAddBF16Inst` | `atomic-global-pk-add-bf16-inst` | global_atomic_pk_add_bf16 | `T/GCNFeatures.td:1141` |
| `FeatureAtomicBufferPkAddBF16Inst` | `atomic-buffer-pk-add-bf16-inst` | buffer_atomic_pk_add_bf16 | `T/GCNFeatures.td:1148` |
| `FeatureFlatAtomicFaddF32Inst` | `flat-atomic-fadd-f32-inst` | flat_atomic_add_f32 | `T/GCNFeatures.td:1159` |
| `FeatureFP8ConversionInsts` | `fp8-conversion-insts` | fp8/bf8 conversions | `T/GCNFeatures.td:942` |
| `FeatureCvtFP8ByteSel` | `cvt-fp8-byte-sel` | fp8 conversions take a byte_sel operand | `T/GCNFeatures.td:981` |
| `FeatureOCPFP8ConversionInsts` | `ocp-fp8-conversion-insts` | OCP fp8 (E4M3FN, E5M2) conversions | `T/GCNFeatures.td:948` |
| `FeatureFP8F16ConversionInsts` | `fp8-f16-conversion-insts` | fp8/bf8 → f16 | `T/GCNFeatures.td:952` |
| `FeatureF16FP8ConversionInsts` | `f16-fp8-conversion-insts` | f16 → fp8/bf8 | `T/GCNFeatures.td:957` |
| `FeaturePackedTID` | `packed-tid` | work-item IDs packed into v0 | `T/GCNFeatures.td:1553` |
| `FeatureVcmpxPermlaneHazard` | `vcmpx-permlane-hazard` | (LLVM text: "TODO: describe me") v_cmpx → permlane hazard | `T/GCNFeatures.td:296` |
| `FeatureSALUFloatInsts` | `salu-float` | SALU floating point | `T/GCNFeatures.td:1329` |
| `FeaturePseudoScalarTrans` | `pseudo-scalar-trans` | pseudo-scalar transcendentals | `T/GCNFeatures.td:1333` |
| `FeatureRestrictedSOffset` | `restricted-soffset` | soffset cannot be an immediate | `T/GCNFeatures.td:1341` |
| `FeatureScalarDwordx3Loads` | `scalar-dwordx3-loads` | 96-bit scalar loads | `T/GCNFeatures.td:731` |
| `FeatureDPPSrc1SGPR` | `dpp-src1-sgpr` | SGPR allowed as DPP src1 | `T/GCNFeatures.td:790` |
| `FeatureBitOp3Insts` | `bitop3-insts` | v_bitop3_b32/b16 | `T/GCNFeatures.td:1355` |
| `FeatureTanhInsts` | `tanh-insts` | v_tanh_f32/f16 | `T/GCNFeatures.td:1359` |
| `FeatureTensorCvtLutInsts` | `tensor-cvt-lut-insts` | v_perm_pk16* | `T/GCNFeatures.td:1363` |
| `FeatureTDMInsts` | `tdm-insts` | tensor_load_to_lds / tensor_store_from_lds | `T/GCNFeatures.td:1367` |
| `FeatureVPermPk16Hazard` | `vperm-pk16-hazard` | V_PERM_PK16 must be followed by a safe VALU or V_NOP | `T/GCNFeatures.td:1313` |
| `FeatureLoadMonitorInsts` | `load-monitor-insts` | flat/global_load_monitor_b* | `T/GCNFeatures.td:575` |
| `FeatureTransposeLoadF4F6Insts` | `transpose-load-f4f6-insts` | ds/global_load_tr4/tr6 | `T/GCNFeatures.td:1371` |
| `FeatureBF16TransInsts` | `bf16-trans-insts` | bf16 transcendentals | `T/GCNFeatures.td:703` |
| `FeatureBF16ConversionInsts` | `bf16-cvt-insts` | bf16 conversions | `T/GCNFeatures.td:707` |
| `FeatureBF16PackedInsts` | `bf16-pk-insts` | packed bf16 fma/add/mul/max/min | `T/GCNFeatures.td:711` |
| `FeatureBF16InlineConstFromUpperFP32` | `bf16-from-upper-fp32` | bf16 inline constants = upper 16 bits of the fp32 constant | `T/GCNFeatures.td:715` |
| `FeatureCvtPkF16F32Inst` | `cvt-pk-f16-f32-inst` | cvt_pk_f16_f32 | `T/GCNFeatures.td:563` |
| `FeatureCvtSrPkBF16F32Inst` | `cvt-sr-pk-bf16-f32-inst` | cvt_sr_pk_bf16_f32 | `T/GCNFeatures.td:567` |
| `FeatureFmaMixBF16Insts` | `fma-mix-bf16-insts` | v_fma_mix*_bf16 | `T/GCNFeatures.td:200` |
| `FeatureMin3Max3PKF16` | `min3-max3-pkf16` | v_pk_{min3,max3}_num_f16 | `T/GCNFeatures.td:226` |
| `FeatureMinimum3Maximum3PKF16` | `minimum3-maximum3-pkf16` | v_pk_{minimum3,maximum3}_f16 | `T/GCNFeatures.td:230` |
| `FeaturePrngInst` | `prng-inst` | v_prng_b32 | `T/GCNFeatures.td:1375` |
| `FeaturePermlane16Swap` | `permlane16-swap` | v_permlane16_swap_b32 | `T/GCNFeatures.td:511` |
| `FeatureAshrPkInsts` | `ashr-pk-insts` | arithmetic shift-pack | `T/GCNFeatures.td:555` |
| `FeatureSupportsSRAMECC` | `sramecc-support` | hardware supports SRAMECC | `T/GCNFeatures.td:1211` |
| `FeatureMaxHardClauseLength63` | `max-hard-clause-length-63` | S_CLAUSE up to 63 instructions | `T/GCNFeatures.td:344-357` |
| `FeatureWaitXcnt` | `wait-xcnt` | s_wait_xcnt | `T/GCNFeatures.td:1401` |
| `FeatureAsynccnt` | `asynccnt` | separate asynccnt counter | `T/GCNFeatures.td:1235` |
| `FeatureAtomicFMinFMaxF64GlobalInsts` | `atomic-fmin-fmax-global-f64` | global/buffer f64 atomic fmin/fmax | `T/GCNFeatures.td:1098` |
| `FeatureAtomicFMinFMaxF64FlatInsts` | `atomic-fmin-fmax-flat-f64` | flat f64 atomic fmin/fmax | `T/GCNFeatures.td:1109` |
| `FeatureFlatBufferGlobalAtomicFaddF64Inst` | `flat-buffer-global-fadd-f64-inst` | flat/buffer/global f64 atomic fadd | `T/GCNFeatures.td:1174` |
| `FeatureMemoryAtomicFAddF32DenormalSupport` | `memory-atomic-fadd-f32-denormal-support` | f32 memory atomic fadd handles denormals | `T/GCNFeatures.td:1178` |
| `FeatureEmulatedSystemScopeAtomics` | `emulated-system-scope-atomics` | PCIe-unsupported system-scope atomics emulated in HW (CAS loop) | `T/GCNFeatures.td:1194` |
| `FeatureGloballyAddressableScratch` | `globally-addressable-scratch` | FLAT can reach any wave's scratch | `T/GCNFeatures.td:1593` |
| `FeatureKernargPreload` | `kernarg-preload` | kernarg preload into user SGPRs | `T/GCNFeatures.td:1540` |
| `FeatureVmemPrefInsts` | `vmem-pref-insts` | flat/global_prefetch_b8 | `T/GCNFeatures.td:316` |
| `FeatureLshlAddU64Inst` | `lshl-add-u64-inst` | v_lshl_add_u64 | `T/GCNFeatures.td:1606` |
| `FeatureAddSubU64Insts` | `add-sub-u64-insts` | v_add_u64, v_sub_u64 | `T/GCNFeatures.td:1610` |
| `FeatureMadU32Inst` | `mad-u32-inst` | v_mad_u32 | `T/GCNFeatures.td:1614` |
| `FeatureMadNC64_32Insts` | `mad-nc-64-32-insts` | v_mad_nc_{u64_u32, i64_i32} | `T/GCNFeatures.td:1618` |
| `FeatureAddMinMaxInsts` | `add-min-max-insts` | v_add_{min,max}_{i,u}32 | `T/GCNFeatures.td:1622` |
| `FeatureMinMaxI64Insts` | `min-max-i64-insts` | v_{min,max}_{i,u}64 | `T/GCNFeatures.td:1626` |
| `FeaturePkAddMinMaxInsts` | `pk-add-min-max-insts` | v_pk_add_{min,max}_{i,u}16 | `T/GCNFeatures.td:1630` |
| `FeatureLdsBarrierArriveAtomic` | `lds-barrier-arrive-atomic` | LDS barrier-arrive atomics | `T/GCNFeatures.td:1642` |
| `FeatureLdsAtomicAddF64` | `lds-atomic-add-f64` | ds_add_f64, ds_add_rtn_f64 | `T/GCNFeatures.td:1074` |
| `FeatureSetPrioIncWgInst` | `setprio-inc-wg-inst` | s_setprio_inc_wg | `T/GCNFeatures.td:1405` |
| `FeatureSWakeupBarrier` | `s-wakeup-barrier-inst` | s_wakeup_barrier | `T/GCNFeatures.td:1409` |
| `Feature45BitNumRecordsBufferResource` | `45-bit-num-records-buffer-resource` | V# num_records is 45 bits | `T/GCNFeatures.td:1647-1655` |
| `FeatureClusters` | `clusters` | workgroup clusters | `T/GCNFeatures.td:1657` |
| `FeatureD16Writes32BitVgpr` | `d16-write-vgpr32` | D16 ops may have 32-bit data dependencies | `T/GCNFeatures.td:699` |
| `FeatureMcastLoadInsts` | `mcast-load-insts` | multicast (cluster) loads | `T/GCNFeatures.td:571` |
| `FeatureAsyncLoadToLDSInsts` | `async-load-to-lds-insts` | global async load to LDS | `T/GCNFeatures.td:579` |
| `FeatureAsyncStoreFromLDSInsts` | `async-store-from-lds-insts` | global async store from LDS | `T/GCNFeatures.td:583` |
| `FeatureSmemPrefetchInsts` | `smem-prefetch-insts` | s_prefetch_inst/data[_pc_rel] | `T/GCNFeatures.td:320` |
| `FeatureNoF16PseudoScalarTransInlineConstants` | `no-f16-pseudo-scalar-trans-inline-constants` | no inline constants for f16 pseudo-scalar trans | `T/GCNFeatures.td:1337` |
| `FeatureRealTrue16Insts` | `real-true16` | true 16-bit registers | `T/GCNFeatures.td:693` |
| `FeatureVMovB64Inst` | `v-mov-b64-inst` | v_mov_b64 | `T/GCNFeatures.td:603` |
| `FeatureVMulU64Inst` | `v-mul-u64-inst` | v_mul_u64 | `T/GCNFeatures.td:607` |
| `FeatureSAddPCI64Inst` | `s-add-pc-i64-inst` | s_add_pc_i64 | `T/GCNFeatures.td:1432` |

#### 2.3 Variant-specific lists (both add these 30 on top of Common)

| feature | name string | LLVM description | definition |
|---|---|---|---|
| `FeatureVOPD3F64OPYSrcHazard` | `vopd3-f64-opy-src-hazard` | VOPD3 hazard: OPY VGPR source = last allocated VGPR with 64-bit OPX | `T/GCNFeatures.td:1323` |
| `FeatureAddressableLocalMemorySize327680` | `addressablelocalmemorysize327680` | LDS size 327680 bytes | `T/AMDGPUFeatures.td:36-47` |
| `FeatureLDSAllocGranularity2048` | `lds-alloc-granularity-2048` | LDS allocation granule 2048 bytes | `T/AMDGPUFeatures.td:51-61` |
| `FeatureLDSEncodingGranularity2048` | `lds-encoding-granularity-2048` | LDS size encoded in 2048-byte units | `T/AMDGPUFeatures.td:66-76` |
| `FeatureLDSBankCount64` | `ldsbankcount64` | 64 LDS banks | `T/GCNFeatures.td:421-430` |
| `FeatureWMMAN16Insts` | `wmma-n16-insts` | WMMA with N = 16 | `T/GCNFeatures.td:996` |
| `FeatureVOP3PX2IncrementsVaVdstTwice` | `vop3px2-increments-va-vdst-twice` | VOP3PX2 instructions bump VaVdst twice | `T/GCNFeatures.td:1679` |
| `FeatureSetregVGPRMSBFixup` | `setreg-vgpr-msb-fixup` | S_SETREG to MODE clobbers VGPR MSBs (needs fixup) | `T/GCNFeatures.td:1396` |
| `FeatureCubeInsts`, `FeatureLerpInst`, `FeatureSadInsts`, `FeatureQsadInsts`, `FeatureMsadInsts`, `FeatureMqsadPkInsts`, `FeatureMqsadInsts`, `FeatureCvtNormInsts`, `FeatureCvtPkNormVOP2Insts`, `FeatureCvtPkNormVOP3Insts` | `cube-insts` … `cvt-pknorm-vop3-insts` | legacy media ops: v_cube*, v_lerp_u8, v_sad*, v_qsad*, v_msad_u8, v_mqsad*, v_cvt_norm*, v_cvt_pk_norm_* | `T/GCNFeatures.td:1020-1062` |
| `FeatureSWMMACGfx1250Insts` | `swmmac-gfx1250-insts` | GFX1250 SWMMAC | `T/GCNFeatures.td:1012` |
| `FeatureTransCoexecutionHazard` | `trans-coexecution-hazard` | TRANS co-executing with VALU hazard | `T/GCNFeatures.td:1303` |
| `FeatureGFX1250VALUBlockingCycles` | `gfx1250-valu-blocking-cycles` | VALU occupancy modeled with GFX1250BlockingCycles | `T/GCNFeatures.td:1308` |
| `FeatureWMMACoexecutionHazards` | `wmma-coexecution-hazards` | WMMA co-execution hazards | `T/GCNFeatures.td:1318` |
| `FeatureDataCacheLineSize128` | `datacachelinesize128` | VMEM cache line 128 bytes | `T/GCNFeatures.td:454-461` |
| `FeatureRequiresInitialUnclausedVmem` | `requires-initial-unclause-vmem` | first VMEM at an entry point must not be in a clause | `T/GCNFeatures.td:1417` |
| `FeatureNeedsTDMDrain` | `needs-tdm-drain` | keep tensor count low | `T/GCNFeatures.td:1683` |
| `FeatureNoSleepForever` | `no-sleep-forever` | s_monitor_sleep must not use sleep_forever | `T/GCNFeatures.td:1693` |
| `FeatureSlowMaxMinMulI64Insts` | `slow-max-min-mul-i64-insts` | 64-bit int max/min/mul slower than expansion | `T/GCNFeatures.td:1422` |
| `FeatureXNACKOnOffModes` | `xnack-on-off-modes` | XNACK on/off modes | `T/GCNFeatures.td:241` |
| `FeatureSupportsXNACK` | `xnack-support` | hardware supports XNACK | `T/GCNFeatures.td:234` |
| `FeatureUnusableSAddPCI64` | `unusable-s-add-pc-i64` | s_add_pc_i64 unusable by codegen | `T/GCNFeatures.td:1669` |

#### 2.4 Set difference

| fact | value | source | notes |
|---|---|---|---|
| In **gfx1250 only** (5) | `FeatureWMMAF4Insts` (`wmma-f4-insts`: WMMA_*_F4), `FeatureWMMA16x16x128FP8Insts` (`wmma-16x16x128fp8-insts`), `FeatureBlock16ConversionScaleInsts` (`block16-cvt-scale-insts`), `FeatureFP8E5M3Insts` (`fp8e5m3-insts`), `FeatureNoMTypeNC` (`no-mtype-nc`: "Does not use MTYPE NC") | LLVM `T/GCNFeatures.td:2632,2633,2646,2647,2656` vs :2588-2622; definitions :1000,1004,551,961,1702 | derived by set comparison of the two lists. |
| In **gfx1250-strict only** (2) | `FeatureGFX1250_STRICT` (`gfx1250-strict`: "gfx1250 strict mode"), `FeatureNeedsAligned2addrDS` (`needs-aligned-2addr-lds`: 2addr LDS ops aligned to payload size) | LLVM `T/GCNFeatures.td:2591,2616`; definitions :1674,1688 | |
| Effect: instructions missing on strict | `v_wmma_f32_32x16x128_f4`, `v_wmma_scale{,16}_f32_32x16x128_f4`, all `v_wmma_{f16,f32}_16x16x128_{fp8,bf8}_{fp8,bf8}` → "not supported on this GPU (gfx1250-strict)" | LLVM `test/MC/AMDGPU/gfx1250-strict_err.s:2-36`; `T/VOP3PInstructions.td:2282-2298` | E5M3 forms of `v_cvt_pk_fp8_f32`/`v_cvt_sr_fp8_f32` need `FP8E5M3Insts` (`T/VOP3Instructions.td:832-848,935-963`). |
| Effect: block16 | `scale_sel` maximum doubles with Block16 (PK8 FP4: 4→8; PK8 FP8/BF8: 8→16) | LLVM `T/AsmParser/AMDGPUAsmParser.cpp:5605-5628`; `T/AMDGPULowerIntrinsics.cpp:280-305` | |
| Effect: `GFX1250_STRICT` codegen | (1) M0 forced to 0 before `cluster_load_*` / `cluster_load_async_to_lds_*`; (2) hard clauses split when memory scopes differ; (3) unscaled `wmma_f32_16x16x128_f8f6f4` intrinsic rewritten into the scaled form with zero scales; (4) longer WMMA latencies (16x16x64 FP8: 8 vs 4 cycles; f8f6f4 scaled: 16 if any f8 input) | LLVM `T/SIISelLowering.cpp:11948-11953,12736-12742`; `T/SIInsertHardClauses.cpp:187-201,259-260`; `T/AMDGPULowerIntrinsics.cpp:308-320`; `T/SISchedule.td:476-500,566-569` | Strict is the only variant that is not covered by gfx12-5-generic. |
| Effect: `NoMTypeNC` (gfx1250, gfx1251) | system/agent acquire invalidates only to SE scope; no `global_wb` at system/agent scope (HW keeps L2s coherent when NC is unused). Strict keeps SYS/DEV-scope invalidate + writeback | LLVM `T/SIMemoryLegalizer.cpp:2059-2072,2137-2157`; `docs/AMDGPUUsage.md:14354-14355` ("MTYPE NC not used on gfx1250 and gfx1251") | ISA silent on MTYPE (no occurrence). |
| Effect: `NeedsAligned2addrDS` (strict) | ds read2/write2 formed only when alignment ≥ payload; separate ISel patterns | LLVM `T/SILoadStoreOptimizer.cpp:1328-1333`; `T/DSInstructions.td:1163-1190`; `T/AMDGPU.td:562` | ISA describes 2ADDR ops without a stricter alignment rule (CDNA5 ISA §11.2 (pp. 149-150)) — matches gfx1250, not strict (derived, weak). |
| What the variants **mean** (public text) | Only: strict = "gfx1250 strict mode"; strict listed as dGPU, gfx1250 as APU; separate ELF mach 0x0eb and subarch `12.50s` | LLVM `T/GCNFeatures.td:1674-1677`; `docs/AMDGPUUsage.md:1173-1204,4313-4315` | No public document explains *why* strict exists or which hardware it targets: **unknown (no public source)**. derived reading: strict is a conservative subset (drops F4 / 128-K FP8 WMMA, E5M3, block16 scale, keeps MTYPE NC handling) plus extra workarounds. |
| gfx1251 vs gfx1250 (context) | adds FullRate64Ops, DPALU_DPP, packed FP64/U64 single-SGPR ops, GFX1251GEMMInsts, GFX125xLowestRateWMMA; drops SetregVGPRMSBFixup, GFX1250VALUBlockingCycles, NoSleepForever, SlowMaxMinMulI64Insts, UnusableSAddPCI64 | LLVM `T/GCNFeatures.td:2663-2701` | ISA §7.8: VOPD with a multi-cycle OPX (f64 ops) is multi-cycle "in certain products" (CDNA5 ISA §7.8 (p. 86)) — derived: the ISA hedges between full-rate and non-full-rate f64 parts. |

---

### 3. `verify: VerifyLimits`

```rust
verify: VerifyLimits {
    vgpr_limit: 1024,              // v0..v1023, allocated in blocks of 16
    sgpr_limit: 106,               // s0..s105 (+ vcc_lo/hi = s106/s107)
    user_sgpr_limit: 32,
    lds_limit: 327_680,            // bytes per work-group, 2048-byte granule
    named_barriers_max: 16,        // NAMED_BAR_CNT <= 4
    buffer: BufferRsrc { base_bits: 57, num_records_bits: 45, num_records_unit: Bytes,
                         stride_bits: 14, stride_scale: [1, 4, 8, 32], oob_disable: AllOnes, unbound: Zero },
}
```

| fact | value | source | notes |
|---|---|---|---|
| VGPR maximum | 1024 per wave (wave32), blocks of 16; ≥ 1 block | CDNA5 ISA §3.1 Table 4 (p. 12); §3.3.2.1 (p. 16). LLVM `T/Utils/AMDGPUBaseInfo.cpp:1336-1341,1297-1317`; `T/SIRegisterInfo.td:667-668` | Agree. Assembler rejects v256+ only on pre-1250 (`T/AsmParser/AMDGPUAsmParser.cpp:2775`). |
| SGPR maximum | 106 (s0..s105); VCC in s106/s107; 16 TTMPs | CDNA5 ISA §3.1 Table 4 (p. 12); §3.3.1.1 (p. 14). LLVM `T/SIRegisterInfo.td:405-406` (SGPR_32 = SGPR0..105 + VCC) | LLVM's non-addressable max for GFX10+ is 108 (`T/Utils/AMDGPUBaseInfo.cpp:1245-1246`). The addressable number comes from TargetParser (not in this checkout) — derived 106 from the register class. |
| User SGPRs | ≤ 32 | CDNA5 ISA §3.5.3.1 Table 9 (p. 30). LLVM `T/Utils/AMDGPUBaseInfo.cpp:2420-2424`; `T/AsmParser/AMDGPUAsmParser.cpp:6502-6506` | Agree. |
| LDS maximum | 320 KiB = 327680 bytes per work-group; 2048-byte allocation | CDNA5 ISA §3.1 Table 4 (p. 12); §3.3.4 (p. 19). LLVM `T/GCNFeatures.td:2627-2629`; `T/Utils/AMDGPUBaseInfo.cpp:1143-1164` | Agree. Not halved (gfx1250 always "full SIMD mode", `T/Utils/AMDGPUBaseInfo.cpp:2507-2509`). |
| Named barriers | ≤ 16 per work-group (pool of 64 per WGP) | CDNA5 ISA §5.6.2 (p. 47). LLVM `docs/AMDGPUUsage.md:7999-8001` | Agree. |
| Buffer base width | 57 bits (V# [56:0]) | CDNA5 ISA §9.6 Table 57 (p. 123); §9.4 Table 54 (p. 119). LLVM `docs/AMDGPUUsage.md:1980-1981`; `T/AMDGPULegalizerInfo.cpp:6505-6517` | Agree. |
| Buffer num_records | 45 bits at V# [101:57], in **bytes**; 0 = unbound; all-ones disables range checking | CDNA5 ISA §9.6 Table 57 (p. 123); §9.4 Table 54 (p. 119); SMEM uses the same fields (§8.1 (p. 110)). LLVM `T/GCNFeatures.td:1655,2574`; `T/AMDGPULegalizerInfo.cpp:6505-6521`; `T/SIISelLowering.cpp:13433` | Agree. gfx950 differs (32-bit field) — the verifier rule must be per target. |
| Buffer stride | 14 bits at [121:108] + 2-bit stride scale [123:122] (×1/×4/×8/×32) | CDNA5 ISA §9.6 Table 57 (p. 123) | LLVM packs a 16-bit "stride and scale" at bit 108 (`T/AMDGPULegalizerInfo.cpp:6519-6524`) — consistent. |
| Buffer offset arithmetic | the summed per-lane offset is masked to 45 bits before adding the base | CDNA5 ISA §9.4 (p. 119) | LLVM silent. derived verify rule: `offset_total < 2^45`. |
| Range check | per-DWORD for multi-dword untyped ops; OOB_SELECT bit picks the rule | CDNA5 ISA §9.4.1 (p. 120); §9.6 (p. 123) | |
| KD encodability | VGPR blocks ≤ 63 (6 bits); user SGPR count ≤ 63 field but ≤ 32 allowed; NAMED_BAR_CNT ≤ 7 field but 0-4 meaningful | LLVM `T/AsmParser/AMDGPUAsmParser.cpp:6525-6536`; `docs/AMDGPUUsage.md:7733-7740,7995-8001` | derived. |

---

### 4. Stepping / products

| fact | value | source | notes |
|---|---|---|---|
| ISA versions | 12.5.0 (gfx1250 and gfx1250-strict), 12.5.1 (gfx1251) | LLVM `T/GCNProcessors.td:311,316,320` | LLVM "stepping" digit only. |
| Product names | TBA in LLVM; none in the ISA | LLVM `docs/AMDGPUUsage.md:1183-1220`; CDNA5 ISA Preface (p. 1) | |
| CDNA5 ↔ gfx number | the ISA never names a gfx processor, LLVM never names CDNA5 (no match for "CDNA5"/"MI4xx" in the checkout) | CDNA5 ISA (whole text); LLVM `docs/AMDGPUUsage.md` (no occurrence) | The ISA's "Related Documents" points to AMDGPUUsage (CDNA5 ISA Preface (p. 2)) but no specific processor. |
| Silicon revisions | **nothing public**. A product or stepping label for gfx1250 cannot be supported from (a) or (b): **unknown (no public source)** | — | |
| Device class hint | gfx1250-strict "dGPU", gfx1250/gfx1251 "APU" | LLVM `docs/AMDGPUUsage.md:1176,1192,1208` | Only public product-shaped fact; could be a doc artifact (Open). |

---

### 5. Correspondence between the public CDNA5 ISA document and gfx1250 (derived)

Method: find instructions and features in the ISA text and in LLVM's gfx1250 definitions/tests; compare opcodes where both give them.
All conclusions in this section are **derived**.

| evidence | ISA | LLVM | match |
|---|---|---|---|
| `V_WMMA_F32_32X16X128_F4` opcode 136 | CDNA5 ISA §7.12 (p. 96); §15 VOP3P list (p. 464) | VOP3P 0x088, `T/VOP3PInstructions.td:2815`; test `test/MC/AMDGPU/gfx1250_asm_wmma_w32.s:1686`; **rejected on strict** `gfx1250-strict_err.s:5-6` | opcode equal; feature only on gfx1250 |
| `V_WMMA_SCALE_F32_32X16X128_F4` (opcode 136, scaled form) | CDNA5 ISA §7.12 (p. 96); p. 466 | `T/VOP3PInstructions.td:2825`; strict error `gfx1250-strict_err.s:8-9` | gfx1250 only |
| `V_WMMA_F32_16X16X128_FP8_FP8` opcode 128 | CDNA5 ISA §7.12 Table 43 (p. 95); p. 460 | VOP3P 0x080, `T/VOP3PInstructions.td:2807`; strict error `gfx1250-strict_err.s:26-27` | gfx1250 only |
| E5M3 via CLAMP=1 on fp8 conversions | CDNA5 ISA §7.6 (p. 82); scale formats §7.12.6 (p. 103) | `FP8E5M3Insts` patterns use DSTCLAMP.ENABLE for e5m3, `T/VOP3Instructions.td:935-947,961-963` | gfx1250 only |
| `scale_sel` = 3 OPSEL bits on `V_CVT_SCALE_PK8_*` (opcode 673 for PK8_F32_FP4) | CDNA5 ISA §15.14 (p. 599) | 0x2a1 `T/VOP3Instructions.td:2594`; limit 8 only with Block16 (`T/AsmParser/AMDGPUAsmParser.cpp:5605-5621`) | opcode equal; range fits gfx1250 (weak) |
| `S_WAIT_XCNT` opcode 69, `S_WAIT_TENSORCNT` 75 | CDNA5 ISA §5.7 (p. 53); pp. 293-294 | SOPP 0x045 / 0x04b, `T/SOPInstructions.td:3062,3070` | both variants |
| `S_SET_VGPR_MSB` opcode 6; S_SETREG MODE also writes MSBs | CDNA5 ISA §3.3.2.3 (p. 17); p. 281 | SOPP 0x006 `T/SOPInstructions.td:3061`; `FeatureSetregVGPRMSBFixup` `T/GCNFeatures.td:1396` | both |
| `V_TANH_F32` VOP1 30, `V_PRNG_B32` VOP1 75 | CDNA5 ISA §7.3 (pp. 77-78); pp. 335, 349 | 0x01e / 0x04b `T/VOP1Instructions.td:1343,1347` | both |
| `TENSOR_LOAD_TO_LDS` opcode 196 | CDNA5 ISA §10.11.1 (p. 140); p. 770 | 0xc4 `T/MIMGInstructions.td:2361`; `test/MC/AMDGPU/gfx1250_asm_vimage.s:5-7` | both |
| `GLOBAL_LOAD_ASYNC_TO_LDS_B128` opcode 98 | CDNA5 ISA §14.7 (p. 206); p. 818 | 0x62 `T/FLATInstructions.td:3677` | both |
| `DS_LOAD_TR4_B64` opcode 250 | CDNA5 ISA §11.2 (p. 155); p. 749 | 0x0fa `T/DSInstructions.td:1549` | both |
| `S_SETPRIO_INC_WG` opcode 62 | CDNA5 ISA §5.1 (p. 41); p. 292 | 0x03e `T/SOPInstructions.td:3068` | both |
| `V_CVT_PK_FP8_F32` opcode 873 | CDNA5 ISA p. 643 | 0x369 `T/VOP3Instructions.td:2563` (gfx1250 E5M3-capable form) | both |
| Cluster multicast loads, M0 mask (M0[15:0]=0 ⇒ plain load) | CDNA5 ISA §10.7 (p. 135) | `FeatureMcastLoadInsts`; strict zeroes M0 (`T/SIISelLowering.cpp:11951-11952`) | both (strict adds a workaround) |
| S_MONITOR_SLEEP "forever" is unsafe | CDNA5 ISA §10.10 (p. 139) | `FeatureNoSleepForever` on both variants `T/GCNFeatures.td:2617,2655` | both |
| Wave32 only; 1024 VGPRs; 106 SGPRs; 320 KiB LDS; 32 user SGPRs; 57/45-bit V# | CDNA5 ISA §1 (p. 3); §3.1 (p. 12); §3.5.3.1 (p. 30); §9.6 (p. 123) | §3 rows above | both |
| Absent in both: image ops, `BUFFER_LOAD_FORMAT_*`, `DS_BVH_*` | no occurrence in the ISA text | rejected: `test/MC/AMDGPU/gfx1250_asm_unsupported.s:203,293,364` | both |
| `S_ADD_PC_I64` | mentioned in prose only (CDNA5 ISA §3.2.1 (p. 13)); **no opcode entry** in the SOP1 list | assembles (0x04b, `T/SOPInstructions.td:2377`; `test/MC/AMDGPU/gfx1250_asm_sop1.s:6`) but `FeatureUnusableSAddPCI64` on both variants | consistent with "do not use" |
| `S_ALLOC_VGPR` / dynamic VGPRs | absent | assembles for GFX12+ (`test/MC/AMDGPU/gfx12_asm_sop1.s:3,6`) | **ISA silent** — do not emit |
| MTYPE NC | absent | gfx1250 assumes NC unused; strict does not | ISA cannot discriminate |

Conclusion (derived): the public CDNA5 ISA (27-July-2026) describes an ISA whose instruction set equals **`gfx1250` (non-strict)**: it contains the five
features that only gfx1250 has (F4 WMMA, 16x16x128 FP8 WMMA, E5M3, and the wider block16 `scale_sel`), with opcodes identical to LLVM's gfx1250
encodings, and it does not document anything that only gfx1250-strict has. It cannot be matched to gfx1251 (no evidence either way except the
"certain products" VOPD f64 hedge, §7.8 p. 86). Recommendation: `mcpu = "gfx1250"`; treat gfx1250-strict as a separate, more restrictive target.

---

### Open

1. **Strict vs non-strict hardware** — what gfx1250-strict is for, and which variant real silicon needs. Public sources only say "strict mode" and dGPU vs APU. Settle with: an AMD statement or ROCm release notes; no probe can answer it.
2. **dGPU/APU column** for gfx1250 (APU) vs gfx1250-strict (dGPU) — possible doc artifact. Settle with: a later AMDGPUUsage revision.
3. **`.amdhsa_named_barrier_count` units** (raw 0-4 field vs barrier count). Probe: `llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 -filetype=obj` with values 1, 4, 16, then `llvm-objdump -s` on the KD (bits 14-16 of RSRC3) — e.g. with the ROCm 10.1 assembler.
4. **Dynamic VGPR mode** (KD RSRC3[17], `s_alloc_vgpr`) is in LLVM but absent from the ISA. Treat as unsupported until the ISA documents it.
5. **TCP_SPLIT values above 320 KiB LDS** (value 7 = 384 kB) conflict with the 320 KiB LDS maximum. Settle with: an ISA/runtime document on the TCP split.
6. **SGPR addressable count**: the TargetParser source (`lib/TargetParser/AMDGPUTargetParser.cpp`) is outside this sparse checkout; 106 here comes from the ISA and `SIRegisterInfo.td`. Probe: assemble `s_mov_b32 s106, 0` with `-mcpu=gfx1250` (expect it to be rejected or parsed as vcc_lo).
7. **Doc vs code conflicts** to keep in mind: `.amdhsa_tg_split` ("GFX11-GFX12" in the doc, rejected by the parser); WGP_MODE (doc GFX10-12, code excludes gfx1250); kernarg preload and `ENABLE_WAVEFRONT_SIZE32` not documented for GFX12.5. Probe: `llvm-mc` with each directive for gfx1250.
8. **XNACK**: the comment says "hardwired on", but the feature lists include on/off modes. Probe: `llvm-mc -mcpu=gfx1250:xnack-` → check e_flags 0x200 vs an error.
9. **Code object version** for production (V5 default vs V6 for `.cluster_dims`). It depends on the runtime that loads the code object, and no public runtime document is in scope.
10. **Product / stepping mapping** (any product name or silicon-revision label): unknown (no public source).

## 8. Assembler smoke checks (`tests/gfx1250_smoke/`)

`bash tests/gfx1250_smoke/run.sh [-v]` assembles every `*.asm` snippet with the rocm-10.1 clang (`-mcpu=gfx1250`); `# EXPECT-FAIL` snippets must
be rejected, `-v` prints the disassembly. The instruction forms come from the public upstream MC tests (`llvm/test/MC/AMDGPU/gfx1250_asm_*.s`,
`gfx12_asm_*.s` @274ab686). Toolchain: AMD clang 24.0.0git (ROCm llvm-project `d6f6cb69`). Result 2026-11-10 (assembler only): **14 passed, 0 failed**.

| Snippet | Content | Result |
|---|---|---|
| `wmma` | 18 WMMA / SWMMAC forms (f16, bf16, bf16f32, f32, fp8 K=64/128, iu8 clamp, f8f6f4 (+FP4 format), scale, ld_scale, f4 32x16, SWMMAC with `index_key`) | assembles |
| `waits` | `s_wait_{load,store,ds,km,xcnt,async,tensor}cnt`, the fused `_dscnt` forms, `s_wait_alu` (named fields), `s_delay_alu`, `s_wait_idle` | assembles |
| `tdm` | `tensor_load_to_lds` / `tensor_store_from_lds` with 2 and 4 SGPR groups, `null` groups, `th:` / `scope:` | assembles |
| `vopd` | VOPD and VOPD3 pairs | assembles |
| `async` | `global_load_async_to_lds_*`, `global_store_async_from_lds_b128`, `cluster_load_async_to_lds_b128`, `cluster_load_b128`, `ds_atomic_async_barrier_arrive_b64`, `s_wait_asynccnt` | assembles |
| `sync` | `s_barrier_signal -1/-3/m0`, `s_barrier_wait`, `s_barrier_signal_isfirst -1`, `s_get_barrier_state`, `s_wakeup_barrier`, `s_barrier_leave` | assembles |
| `mem_scope` | global / buffer / scalar memory with `th:` / `scope:` / `nv` / `scale_offset`, `global_load_tr_b128`, `global_inv` / `wb` / `wbinv` | assembles |
| `nops_msb` | `s_nop`, `v_nop_e64`, `s_set_vgpr_msb` | assembles |
| `fail_s_waitcnt` | `s_waitcnt vmcnt(0) lgkmcnt(0)` (the gfx950 `WaitForm::Combined`) | rejected (not supported on gfx1250) |
| `fail_mfma` | `v_mfma_f32_32x32x16_bf16` | rejected |
| `fail_isfirst_cluster` | `s_barrier_signal_isfirst -3` | rejected ("does not support user_cluster_barrier_id (-3)", `AsmParser/AMDGPUAsmParser.cpp:5585`) |
| `probe_vnop_vop1` | plain `v_nop` (no gfx1250 RUN line upstream) | accepted (encoding `7E000000`) |
| `probe_open` | `cluster_load_async_to_lds_b8`, `s_set_vgpr_msb 0x100` (both accepted upstream) | accepted; hardware meaning of the extra immediate bits: unknown (no public source beyond P1 §2.1) |
| `probe_divergent_rocm` | `v_swmmac_f32_16x16x32_f16`, `v_dual_dot2acc_f32_f16` | **accepted by the rocm-10.1 assembler although upstream rejects both for gfx1250** (`gfx1250-strict_err.s:38-40`, `gfx1250_asm_unsupported.s`). This description follows upstream |

The smoke checks pin syntax only: not silicon behaviour, encodings' meaning, hazards or latencies.