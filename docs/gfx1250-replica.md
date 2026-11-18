# gfx1250 (MI450) 1:1 replicas of two vendor kernels through the safe DSL

Two aiter gfx1250 code objects are reproduced through the DSL's exact mode, word for word:

| | MXFP4 GEMM (192x256 tile, B pre-shuffled, 4x4 cluster, "NPS") | MLA decode (a8w8, 16 query heads, one split) |
|---|---|---|
| vendor instructions | 4567 | 18449 |
| replica instructions identical, in order (byte-identical build) | **4567 / 4567** | **18449 / 18449** |
| verbatim (`Raw`) instructions / ghost definitions | 0 / 0 | 0 / 0 |
| author-word annotations (`AuthorWords`, same instruction, encoder-filled fields differ) | 240 | 5509 |
| named deviations in the headline build | none | 79 × `v_nop` (§4) |
| launch / G5 facts (trusted items, §5) | `cluster_dims 16`, `g5:lds_above_64k` | `g5:lds_above_64k` |
| HW (MI450, same host, inputs, seed as the vendor) | output hash identical, 3 shapes | output hash identical, 4 shapes, both builds |

Nothing vendor-derived is in the repository: the lifted IR, the emitted assembly and the disassemblies stay outside it. This document gives
method, counts and classes only.

## 1. Method

```
vendor .co ──objdump──▶ instruction stream ──lift──▶ exact-mode IR ──amdc emit (exact, gfx1250)──▶ .s ──clang──▶ .co
                         (isa_parse_gfx1250)          (isa_lift_gfx1250 + isa_lift)   │                                │
                                                                                     ├ exact checker (waits, hazards, │
                                                                                     │  registers, MSB windows)        │
                                                                                     ├ G5 hang-safety rules           │
                                                                                     └ hang-scan of the assembly      │
                                                                        word compare vs vendor ◀── objdump ◀──────────┘
                                                                        HW A/B: output hash + paired timing
```

1. **Parse** (`tools/isa_parse_gfx1250.py`): the objdump stream with the physical VGPR of every operand. objdump forgets the
   `s_set_vgpr_msb` state at branch targets and prints bank-0 numbers there; `fix_msb()` rebuilds the physical registers by tracking the MSB
   state along the instruction order (per opcode, the operand → MSB-field map is learned from instructions objdump does annotate). True16
   halves (`v688.l` / `.h`) are parsed.
2. **Front end** (`tools/isa_lift_gfx1250.py`): renames to the shared lifter's vocabulary; fuses pointer-add pairs
   (`s_add_co_u32` + `s_add_co_ci_u32 hi, 0, hi`) into one 64-bit add; finds SCC values that live across instructions; attaches the
   author's words where the instruction's unused fields are filled differently by our assembler (`--words-ops`, `--words-vop3`);
   `--fix-trans-coexec` inserts the LLVM v_nop (§4) and lists every site (`--deviations`).
3. **Lift** (`tools/isa_lift.py`, gfx1250 handlers): every instruction becomes a DSL statement with its semantics. Control flow:
   `--dag --layout-keys` emits the top-level blocks in address order with forward `GotoIf` / `Goto` (no backward gotos, no gotos into an
   if arm or a loop body; loops stay structured), one `LayoutKey` per author block (key `2s + 1`, `s` = the block's first instruction) and
   a loop-entry key `2s` in front of each top-level loop. The author's dead blocks (unreachable code after a branch) are lifted too.
4. **Compile** (`amdc emit --from-ir … --target gfx1250 --backend own`): exact mode keeps every pinned register, the keyed block order
   (entry first), the author's branches (also a branch to the next block, `Stmt::Goto`), the author's latch shapes, and checks every
   `s_set_vgpr_msb` window against the registers used. A statement preceded by `AuthorWords` is emitted as `.long` words with our own
   spelling verified (MSB window) and kept as the comment.
5. **Verify**: backend's exact checker (waits / hazards / register clobbers, with the wave's scheduling mode), the G5 hang-safety rules, and
   `amdc hang-scan`. All three run inside `amdc emit`.
6. **Compare and run**: assemble with the node's clang, objdump, compare words in order; on HW run vendor and replica with the same host,
   inputs and seed: FNV-1a 64 of the whole output, plus paired alternating timing (`tests/gpu/g1250_timing.h`).

## 2. Numbers

### MXFP4 GEMM (f01-2)

| M × N × K | output hash (vendor = replica) | vendor µs | replica µs |
|---|---|---|---|
| 768 × 1024 × 256 | identical | 4.56 | 4.56 |
| 1536 × 2048 × 1536 | identical | 6.66 | 6.66 |
| 3072 × 4096 × 4096 | identical | 12.6 | 12.6 |

Launch: cluster 4 × 4, grid (N / 1024 · 4, M / 768 · 4), block 128, 80-byte preloaded kernarg.

### MLA decode (f01-2)

| B × ctx | output hash (vendor = replica = headline) | paired ratio byte-identical / vendor | paired ratio headline / vendor |
|---|---|---|---|
| 1 × 64 | identical | noise (5 µs kernel) | noise |
| 4 × 1000 | identical | 0.955 | 1.045 |
| 32 × 2048 | identical | 1.035 | 1.008 |
| 128 × 4096 | identical | 1.009 | 0.973 |

All MLA runs pass the fp32 CPU reference (`MLA_OK`). Launch: grid (1, B, 1), block 128, 120-byte preloaded kernarg, no cluster.

### Cluster barrier on HW (backend's `probe_g1250_barriers`)

Launched as a cluster (`hipDrvLaunchKernelEx` + `hipLaunchAttributeClusterDimension`; `g1250_probe_host barriers`, env `CLUSTER`, `GRID`):
clusters of 2 / 4 / 8 work-groups, grids up to 512, 20 runs each: correct, no hang.

## 3. What the DSL and the compiler needed

Each form has semantics the verifiers see (no opaque text); all are exact-mode / gfx1250 unless noted.

| class | forms |
|---|---|
| control flow | `LayoutKey`, `GotoIf`, `Goto` (forward, same structured region, checked by `check_gotos`); keyed layout with loop-entry keys; author branches; author latch shapes (`if !c goto exit; goto header`); dead author blocks kept |
| encoding | `AuthorWords` (the next statement's instructions, in order; all must be taken); `EncodeE64` with a literal operand |
| scalar ALU | `Salu { AddCoI32, SubCoI32, AddCoU32, AddCoCiU32, BfeU32, Bitset0, Bitset1, AddNcU64, MovB64 }` |
| system | `Sys { Version, SetRegImm, WaitIdle, PrefetchInst, PrefetchGlobal, VNop, Sleep }`, `SysVal { GetReg, GetPc }`, `SetSchedMode`, `WaitAlu` |
| EXEC (wave32) | `ExecSet`, `ExecAll`, `ExecFromMask`, `ExecSelect` |
| hardware inputs | `KernargPreload`, `ReadTtmp` (pinned at ttmp*n*, no instruction), `WorkitemIdPacked` (v0 as written: x / y / z ids) |
| vector ALU | `BinOp::AddNcI32`, `BinOp::SubrevU32`, `UnOp::RcpIflagF32`, `VAddU64`, `CvtPkFp8Half` (true16 half writes), `PkNegF32` (the `neg_lo` / `neg_hi` modifier of `PkFmaF32`) |
| lane exchange | `SwapA`, `Permlane16SwapA` (+ `Permlane32SwapB` for the second register) |
| memory | `LoadAsyncLds` (global → LDS, ASYNCcnt), `SBufferLoad`, TDM load / store with the author's descriptor words, `Wmma` with raw words |
| emitter | gfx1250 spellings / encoder samples for every replayed mnemonic; `code_end_pad`; `s_addk_co_i32`, `ds_load_tr8_b64`, `v_max3_num_f32` re-spellings |

## 4. Named deviations

| kernel | count | class | source |
|---|---|---|---|
| MLA | 79 | `v_nop` between a transcendental VALU instruction and the next VALU instruction that reads its result or overwrites its source | LLVM `GCNHazardRecognizer::fixVALUTransCoexecutionHazards` (gfx1250, FeatureTransCoexecutionHazard) inserts one; the vendor omits it. The byte-identical build keeps the vendor's code (diagnostic only: `AMDC_EXACT_WAITS_REPORT`); the headline build adds the 79 (`--fix-trans-coexec`). Cost on HW: within noise |

Not deviations (checked against the public LLVM source):

- **XCNT before overwriting a buffer store's resource SGPRs** (282 sites in MLA): LLVM tracks XCNT only with XNACK enabled; the vendor MLA
  object has XNACK off (e_flags 0x449). The checker models XCNT only under the kd fact `xnack`.
- **DScnt with interleaved DS stores**: every DS access is one in-order event class (`LDS_ACCESS`); the checker counts stores too.

## 5. Trust items

| item | kernel | why it is trusted, not proved |
|---|---|---|
| `cluster_dims 16` (launch fact) | GEMM | the cluster shape is a launch property; the host launches exactly the vendor's 4 × 4 clusters |
| `g5:lds_above_64k` | both | LDS above 64 KiB is a G5 trusted item; evidence: the vendor's identical bytes on MI450 |
| `AddCoCiU32` carry | MLA (36), GEMM | the carry-in is the SCC of the paired `AddCoU32`; the lifter pairs them only when no SCC writer lies between |
| `AuthorWords` | both | same instruction, same operands; only fields our encoder fills differently (unused VOP3 `src2`, a NULL `soffset`, VOP3P `op_sel_hi`, …). The spelled line is MSB-window checked and kept as the comment |

## 6. Findings along the way

- objdump drops the MSB state at branch targets: 13 WMMAs read wrong physical registers until `fix_msb()`.
- The code object's metadata must cover the whole kernarg segment (`.args` padded to `kernarg_size`): an 8-byte `.args` copied too little
  kernarg and the replica faulted with identical `.text`.
- The vendor GEMM signals the cluster barrier from wave 0 only (`waveIDinGroup` = TTMP8[29:25]); waves 0 / 2 and 1 / 3 run different
  loops, and the signal / wait pairing is across waves.
- **Open:** the scheduling mode. Until 2026-11-18 the lifter did not recognise the vendor's `SCHED_MODE = 2` setreg (the operand
  tokenizer split `hwreg(…)`), so the exact checker ran both replicas in mode 0. With mode 2 modelled, the checker reports VALU → DS
  `va_vdst` and VALU → buffer-store `vm_vsrc` findings at sites where the vendor has no `s_wait_alu` (GEMM: none anywhere) and runs
  correctly. Under review with backend: what mode 2 relaxes exactly (public LLVM / ISA), or a missing in-order distance bound in the model.

## 7. Reproducing

```
python tools/isa_lift_gfx1250.py <vendor.dis> --kernel --exact --raw --dag --layout-keys --words-vop3 \
    --words-ops <mnemonics> --preload <n> --name <name> --kernarg-size <bytes> --workgroup 128 --lds <bytes> --fp-mode 0,3,0 \
    --kd <kd directives and launch facts> --args <P/I list> [--fix-trans-coexec --deviations <file>] --out <file.ir>
amdc emit --from-ir <file.ir> --target gfx1250 --backend own --out <file.s>
```

Hosts: `tests/gpu/g1250_f4gemm_host.cpp`, `tests/gpu/g1250_mla_host.cpp` (both print `output hash`; `CO2` / `KERNEL2` = paired timing),
`tests/gpu/g1250_probe_host.cpp barriers` (cluster launch). Every HW run goes through the node's GPU lock with a hard timeout.
