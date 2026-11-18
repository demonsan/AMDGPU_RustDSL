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

## 8. First round: what the replica says about our own MXFP4 GEMM

`tools/isa_loop_profile.py` (instruction classes of a kernel and of its hot loops, counts only) on the vendor GEMM and on our
`gemm_g1250_tdm_stage_f4` (sched's TDM stage template, emitted from Main 2e68d3ef). Both use the same WMMA
(`v_wmma_scale_f32_32x16x128_f4`, 4 waves).

| per hot-loop body | vendor (192 x 256 tile, cluster 4 x 4) | ours (128 x 64 tile, no cluster) |
|---|---|---|
| K steps per body (unroll) | 12 | 1 |
| WMMA per body / per wave per K step | 288 / 24 | 4 / 4 |
| instructions per WMMA (all classes) | 4.3 | 57 |
| VALU per WMMA | **0** | 20 (16 of them `v_mov_b32`: accumulator copies) |
| SALU per WMMA | 0.19 | 23.8 (descriptor / ring-slot / address recomputation each step) |
| LDS reads per WMMA | 1.5 (b128 operands + b32 scales) | 4 |
| hazard `v_nop` per WMMA | 0 | 4 |
| memory waits per WMMA | 0.09 (19 `s_wait_dscnt N` + 6 `s_wait_tensorcnt N` per body, fine-grained) | 1.5 (load all, wait, compute) |
| TDM per K step | 1 (cluster multicast) | 4 (A, B, SA, SB per work-group) |
| barriers per K step | ~0.6 (incl. the cluster barrier) | 1 |
| `s_set_vgpr_msb` per WMMA | 1.2 (1024 VGPRs: 24 accumulators x 8 per wave) | 0 (≤ 256 VGPRs) |

Concrete compiler gaps, in the order they cost:

1. **Loop-carried accumulator copies** (wave / register structure): every WMMA result is copied with 8 `v_mov_b32` back into the
   loop variable, and the copies' read of the WMMA result costs the 4 `v_nop`. The vendor accumulates in place (D = C, same registers
   across iterations). Needed: tie the loop-carried `Frag` variable to the WMMA's D = C registers (coalesce the phi copy), so the loop
   body has no VALU at all.
2. **Per-step address and descriptor arithmetic in the loop** (SALU 95 per K step): the vendor keeps LDS operand addresses fixed per
   slot (offsets folded into `ds_load` immediates, slots unrolled) and advances one 64-bit global pointer per TDM (`s_add_nc_u64`).
   Needed: unroll the ring (slots × K steps) so slot addresses are constants, and strength-reduce the descriptor update to one pointer add.
3. **Tile / wave structure**: 24 WMMAs per wave per K step (vs 4) gives the vendor 1.5 LDS reads per WMMA (operand reuse across the
   2D WMMA grid of a wave) and needs > 256 VGPRs. Needed: larger per-wave WMMA grids in the stage template, and the allocator's use of
   the MSB windows (the replay proves the emitter and checker handle 1024 VGPRs; our allocator stays below 256).
4. **WMMA / LDS-read interleave and wait placement**: the vendor issues the next WMMA's operand reads between WMMAs and waits with
   `s_wait_dscnt N` (N > 0) right before the consumer; ours loads everything, waits to 0, then runs the WMMAs. Needed: the scheduler's
   in-loop software pipelining of DS reads against WMMAs with counted waits (sched's cross-block work).
5. **TDM issue split and cluster multicast**: one TDM per K step per cluster member, multicast to the 4 x 4 cluster (`workgroup_mask`),
   with a cluster barrier for slot reuse; ours issues 4 TDMs per work-group. Needed: multicast TDM in the stage template (the cluster
   barrier is now validated on HW, §2), and the A / B / scales packed into fewer descriptors.
6. **Wave roles**: the vendor runs two loop copies (waves 0 / 2 and 1 / 3), with only wave 0 signalling the cluster barrier; our
   template has one role. Needed when 5 is in: a producer / consumer split by `waveIDinGroup` (`Salu BfeU32` of TTMP8, already a DSL form).
7. **Epilogue**: the vendor converts to bf16 and writes through LDS with one `tensor_store_from_lds`; ours stores f32 with 64
   `global_store_b32` per wave. Needed for the bf16 output: a TDM store epilogue.
8. **Kernarg preload**: the vendor preloads 20 kernarg dwords into SGPRs (no `s_load` in the prologue); ours loads them (7 `s_load`).

An MLA decode compiled by us does not exist for gfx1250 yet; the vendor MLA's loop profile (per WMMA: 3.2 LDS reads incl.
`ds_load_tr8_b64`, 2.6 VALU incl. 0.5 transcendental, 2.2 `v_nop`, 0.12 async global → LDS loads, TDM 0.06) lists the forms such a kernel
needs, all of which the DSL now has (§3).

## 9. Our own MLA decode on gfx1250 (`mla_g1250_qh16`)

Same ABI and device layouts as the vendor qh16 kernel (120-byte kernarg, grid (1, B, 1), block 128), so `g1250_mla_host` runs both and
pairs them. Stage body `amdgpu_stage::attention::mla::MlaBody` on the `OneWindow` pipeline, one window = two KV pages (128 tokens):

- KV ring: 3 slots of two pages (64 tokens x 576 B each); every wave TDM-loads a quarter of each page; the LDS layout is padded by the TDM
  with the pad `amdgpu_dsl::tdm_pad::plan` chooses from the window's declared K and V read patterns (`PageLayout`); the sequence's page
  ids are staged once in LDS. The pipeline is drained (`s_wait_tensorcnt 0`) before the epilogue.
- S^T = K Q^T: wave w computes tokens 16 w .. +15 of each page against the 16 heads (9 fp8 WMMA 16x16x64 per page; inside a WMMA the K
  order is free as long as A and B agree, so each lane half reads 32 contiguous bytes: two `ds_load_b128`).
- base-2 online softmax; the tile maximum through LDS (8 partials per head); P to fp8 through LDS, laid out so that the bytes a lane reads
  back are the A fragment of P for the PV WMMA; lazy rescale (the O rescale only when some head's maximum grew: an any-lane branch).
- PV: wave w owns 128 of the 512 V dims, O accumulates in place (D = C = the loop variable); the B fragments come from `ds_load_tr8_b64`.
  Its lane mapping, measured by `probe_g1250_tr8`: fetch lane f supplies 8 contiguous bytes; byte b lands in lane
  `b + 8 ((f >> 2) & 1) + 16 (f >> 4)`, byte `(f & 3) + 4 ((f >> 3) & 1)`. So for a 16-dims x 16-tokens fragment, fetch lane f reads
  token `8 (f >> 4) + (f & 3) + 4 ((f >> 3) & 1)`, dims `+ 8 ((f >> 2) & 1)`.
- epilogue: the row sums through LDS; O / L * ks to bf16 through LDS (over the dead KV ring), 16-byte global stores; the natural-log LSE.
- bounds: every global access is proved; the paged KV pool and the page table use the trusted host contract (page id < P, the page table's
  entries < NPG), which the host validates.

Correct on MI450 against the fp32 reference on every shape tried (B x ctx up to 512 x 8192), with the same error as the vendor. Paired
against the vendor (a08-1):

| B x ctx | ours (µs) | ours / vendor |
|---|---|---|
| 4 x 1000 | 15.7 | 1.72 |
| 32 x 2048 | 25.1 | 1.98 |
| 128 x 4096 | 48.8 | 1.44 |
| 128 x 8192 | 87.5 | 1.70 |
| 512 x 8192 | 200 | 1.36 |

What moved it, which is also the compiler / template gap list for MLA:

1. **LDS layout vs the read pattern** (2.5x): token rows at a 512-byte stride put the 16 rows a K read covers, and the 8 rows of a
   transposing V read, in the same banks. Now planned (`tdm_pad`) from the declared read patterns.
2. **Template loads were per wave**: "every wave issues the same loads" is 4x the KV traffic; the slice per wave is in the body.
3. **Accumulator copies**: the rebuilt O fragments and the loop-carried copies (~155 `v_mov` per page) are gone with the in-place
   rescale and D = C accumulation.
4. **Barriers and exchanges per token**: two pages per window halve them (-18 %); lazy rescale skips most O rescales.
5. **Register reuse serialised the fragment reads** (read, wait 0, WMMA, nine times); issuing the reads first gives counted waits.
6. **Open**: per tile 569 instructions for 34 WMMA (16.7 per WMMA, the vendor's density is 15.6), so the remaining gap is latency, not
   instruction count. 43 `v_add_nc_u32` per tile are addresses the compiler does not fold into the DS offset field (the multi-address
   `tr8 x4` / `b128 x2` reads); no software pipelining across tiles (the vendor's loop body covers about 12 pages); a fixed prologue of
   dependent global loads (sequence bounds, page table) that costs about 4 µs at B = 1.

### 9.1 Wave-split variant (`mla_g1250_qh16_ws`)

The first kernel's limiter turned out to be the per-tile cross-wave synchronisation (two barriers, the max / P exchanges): a deeper ring
(`Pipeline::slots` 4) changed nothing, a cross-tile prefetch window gave -2 %, async per-lane KV loads were 30 % slower than the TDM. So the
wave-split variant removes the synchronisation instead (flash-decoding inside the work-group):

- wave w takes pages w, w + 4, .. (its own trip count `ceil((npages - w) / 4)`, in pairs; a wave without a page skips the loop) through its own
  private 2-slot ring (`lds.private`, one TDM per page): no barrier and no cross-wave LDS traffic in the loop;
- S for a page = 4 WMMA tiles of 16 tokens; their lane layout is the A fragment of P for the page's 64 tokens, so P never leaves the
  registers; the 4 S chains are interleaved chunk by chunk (independent accumulators: the WMMA and LDS latencies overlap);
- the row maximum across the two lane halves with `v_permlane16_swap`; the rescale factor of the O rows (heads 8 g + r) by `v_readlane`;
  O over all 512 dims per wave (32 accumulators, 560 VGPRs with the MSB windows); V fragments one group of 4 tiles ahead of their WMMAs;
- the token mask only on a page reaching past ctx (a uniform, VALU-only branch);
- one barrier and an LDS merge of the four (max, sum, O) partials over the dead rings at the end.

Paired on MI450 (a08-1), all MLA_OK with the same error as the vendor:

| B x ctx | ws (µs) | ws / first kernel | ws / vendor |
|---|---|---|---|
| 4 x 1000 | 18.0 | 1.17 | 1.98 |
| 32 x 2048 | 23.0 | 0.96 | 1.82 |
| 128 x 4096 | 38.7 | 0.83 | **1.14** |
| 128 x 8192 | 62.5 | 0.75 | **1.23** |
| 512 x 8192 | 172.9 | 0.84 | **1.14** |

Short sequences stay slower: a wave's pages run back to back with one wave per SIMD, so the per-page latency is exposed. The loop now
spends 3 `s_set_vgpr_msb` per WMMA (the accumulators are spread over the 1024-VGPR banks: an allocator item) and about 12 VALU per WMMA.

Later rounds (all paired on a08-1): a timestamp profile (`FnBuilder::shader_cycles`, the debug variant `mla_g1250_qh16_wsstd` writing
wave 0's stamps over the LSE) showed the cross-wave merge at 41 % of a one-page kernel: its 256 LDS stores per lane recomputed their
addresses with `v_mul_lo_u32`. With the lane base hoisted and the constants in the DS offset field (1551 -> 290 instructions), plus scalar
loads for the sequence bounds / page ids and an odd tail page instead of a masked pad page, the default `_ws` is at 2.67 / 1.34 / 1.35 /
**0.97** / 1.12 / 1.11 x the vendor for B x ctx = 1 x 64, 4 x 1000, 32 x 2048, 128 x 4096, 128 x 8192, 512 x 8192. Rejected on
measurement: the branch-free rescale (0 %), the cross-page and intra-page interleave forms (slower in authored order), a KV split across
work-groups with a combine kernel (slower: the fixed per-work-group cost is paid per split).
