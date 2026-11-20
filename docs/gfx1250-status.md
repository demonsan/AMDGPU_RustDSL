# gfx1250 (MI450) status

The state of the gfx1250 back end, the kernels written for it, how they compare with the vendor kernels on the same hardware, what the
optimisation rounds taught, and what is open. Measurements are paired runs on MI450 nodes (one GPU, both kernels in one process, same
inputs; see [`mi450-nodes.md`](mi450-nodes.md) for access, the GPU lock and the hang policy). The vendor kernels are only measured and
read; nothing of them is reproduced here.

Related documents: target description [`target-gfx1250.md`](target-gfx1250.md), hang-safety rules
[`gfx1250-hang-rules.md`](gfx1250-hang-rules.md), proven hazard relaxations [`gfx1250-hazards.md`](gfx1250-hazards.md), replica method and
the MLA kernel [`gfx1250-replica.md`](gfx1250-replica.md), MXFP4 GEMM plan [`mxfp4-gemm-gfx1250-plan.md`](mxfp4-gemm-gfx1250-plan.md).

## 1. What the back end supports

| area | state |
|---|---|
| **IR -> machine code** | own back end only (the LLVM-tail path is gfx950's): the shared lowering, scheduler and register allocator with a gfx1250 target; `amdc emit <kernel> --target gfx1250 --backend own` writes assembly; every gfx1250 kernel of the registry compiles in the tests (`gfx1250_emit`) |
| **Code generation** | kernarg preload (up to 16 dwords; the start sequence provably never clobbers the ABI inputs; scalar loads when SGPRs run out), in-place WMMA accumulation, block-local value numbering of address arithmetic, descriptor words written by their producers; opt-in experiments (`matorder`, WMMA operand reuse hints `kd wmma_reuse`, `AMDC_ACCWIN`) measured neutral or worse |
| **Spelling** | a gfx1250 re-spelling table (G1) maps the shared mnemonics to their gfx1250 forms (e.g. `_e32` / VOP3 forms, `v_max_num_f32`, `v_readlane_b32`, `v_cvt_pk_fp8_f32` in the 32-bit operand form with `op_sel` for the high half); a mnemonic without an entry is a compile error, never a guess |
| **Encoder** | an in-process gfx1250 encoder (tables generated from the node assembler's own encodings, `tools/gen_enc_tables_gfx1250.py`); every emitted line of every registry kernel is checked against `llvm-mc` (`gfx1250_encoder`), and whole probes against clang's code object; VOP3 `op_sel`, `s_get_pc_i64`, `s_prefetch_inst` are covered |
| **Waits** | split counters (LOADcnt, STOREcnt, DScnt, KMcnt, Xcnt, ASYNCcnt, TENSORcnt), counted waits over in-order returns. Wait immediates are range-checked in three layers (lowering, emitter, encoder) and by G5 `GV-1250-WAIT-RANGE` on the assembly (63 for all counters but KMcnt 31; TENSORcnt has no published width and is held to 63 like ASYNCcnt). An independent re-check of the final waits (`check::verify_planned`) models the counter widths and saturation per event class, so a dropped wait is a compile error. XCNT waits only when the kernel declares XNACK |
| **Hazards** | the gfx1250 hazard pass: TRANS co-execution `v_nop`s, VALU write -> read windows, `s_wait_alu` where required, the WMMA co-execution windows per WMMA kind, the tensor drain (at most 10 outstanding at an issue, G5 `GV-1250-TDM-DRAIN`). TRANS co-execution is relaxed only for the reader classes proven on hardware ([`gfx1250-hazards.md`](gfx1250-hazards.md): `v_add/v_fma/v_max/v_mul_f32`, `v_pk_add_f32`) |
| **Scheduling mode** | our own code runs in mode 0. Mode 2 without the extra `s_wait_alu` waits is wrong on hardware (measured: `tools/mi450/mode2.sh`), so exact replays that set mode 2 carry those waits as named deviations (the exact checker models mode 2: VA_VDST 15, VM_VSRC 7), not as relaxations |
| **VGPRs** | 1024 VGPRs per wave through `s_set_vgpr_msb` windows; a low-window register class for operands that must sit in v0..v255 (WMMA scale operands); as a last resort after the scan policies, a two-phase allocation (low-window values first, then tuples, then single dwords best-fit by start). Placing accumulators in the high window was tried and is archived (worse) |
| **Matrix** | fp8 / bf8 WMMA 16x16x64 and 16x16x128, scaled MXFP4 WMMA with `v_wmma_ld_scale_paired_b32`, `ds_load_tr8_b64` (lane mapping measured: [`gfx1250-replica.md`](gfx1250-replica.md) §9), in-place accumulation (D = C = the loop variable) and per-lane accumulator scaling |
| **Tensor data mover** | `tensor_load_to_lds` / `tensor_store_from_lds` with 2-D descriptors, LDS padding (planned from the read patterns: `amdgpu_dsl::tdm_pad`), cache policy, cluster multicast masks; completion on TENSORcnt (in order); a tensor store reads its LDS source unpadded, so a padded store descriptor is a compile error (G5 `GV-1250-TDM-STORE-PAD`; the pad planner is for loads); async per-lane copies (`global_load_async_to_lds_b128`, ASYNCcnt) |
| **Synchronisation** | split barriers (`s_barrier_signal` / `s_barrier_wait`), named barriers, cluster launch and the cluster barrier; workgroup ids in a cluster launch come from the TTMP registers (`TTMP6` positions, `TTMP9` / `TTMP7` cluster ids) |
| **Verification** | IR verifier, tile protocol (shared rings, wave-private regions, tensor and async producers), the hang rules G5, global / tensor bounds (provenance, declared extents, paged buffers with counted host contracts), the wait self-check; all run on every gfx1250 compile |
| **Pipeline templates** | `amdgpu_stage::Pipeline` with `Cadence::OneWindow` (ring depth `slots`, `stage_live` for the skip past the last tile, `finish` drain), used by the MXFP4 GEMM and the first MLA kernel |

## 2. Hang safety

The rules are in [`gfx1250-hang-rules.md`](gfx1250-hang-rules.md); in short, every behaviour that can leave a wave, a work-group or the
device waiting forever is a compile error unless a launch fact states why it is safe: barrier participation and pairing (split, named,
cluster), tensor operations in flight (at most 10 per path) and drained before `s_endpgm`, tensor descriptors with kernarg provenance and a
tile proven inside the declared extent, wait immediates inside the counter widths. Each rule has a mutation test and a positive test
(`crates/amdgpu-kernels/tests/hang_rules_1250.rs`, `tdm_bounds.rs`, `loop_ifs.rs`).

## 3. The two 1:1 replicas

Two vendor kernels were lifted to the DSL's exact-replay form and rebuilt byte for byte (method and deviations in
[`gfx1250-replica.md`](gfx1250-replica.md)):

- the MXFP4 GEMM (256 x 256 tile, 4 x 4 cluster, persistent): identical code object, identical output hashes;
- the MLA decode (qh16): identical output hashes on every tested shape.

They are the reference for what the compiler must eventually produce by itself; the compiler-gap lists of §8 / §9 of that document compare
our own kernels with them.

## 4. Our own kernels against the vendor (paired)

**MLA decode, qh16, fp8 Q / KV, page size 64** (same ABI as the vendor kernel, so the host runs both; all results checked against an fp32
reference, error equal to the vendor's):

| B x ctx (pages / sequence) | first kernel (`mla_g1250_qh16`) / vendor | best of ours / vendor (e08, 2026-11-19) | kernel |
|---|---|---|---|
| 1 x 64 (1) | - | ~1.3 (the vendor itself 2.9 - 3.8 us) | `_ws4` |
| 4 x 512 (8) | - | 1.09 - 1.13 | `_ws2` |
| 4 x 1000 (16) | 1.65 | 1.24 | `_ws` |
| 32 x 2048 (32) | 1.90 | 1.22 | `_ws` |
| 128 x 4096 (64) | 1.38 | **0.93 - 0.95** | `_ws` |
| 128 x 8192 (128) | 1.65 | **0.97 - 1.01** | `_ws` |
| 512 x 8192 (128) | 1.34 | 1.03 | `_ws` |

Dispatch rule: 1 page per sequence -> `mla_g1250_qh16_ws4` (all 4 waves take every page, 128 V dims each), 2 - 8 pages -> `_ws2` (wave
pairs share a page), more -> `_ws` (each wave its own pages, private 2-slot ring, final cross-wave merge). `_ws` today: scalar loads for
the sequence bounds / page ids, full page pairs + the sequence's last page computed apart (the only masked page; every other page
without mask code), the lean softmax (exponents / exps / sums batched: no TRANS-WAR `v_nop`), the GEMM compile policy. 128 x 4096 and
128 x 8192 are at or ahead of the vendor (a08: 0.93 / 0.97; e08: 0.95 / 1.01). Structure: [`gfx1250-replica.md`](gfx1250-replica.md) §9.

**MXFP4 GEMM** (all bit-exact against the CPU reference; ours f32 output with natural row-major A / B / scales, the vendor's bf16
output with preshuffled B, 4 x 4 cluster, persistent grid):

| kernel (e08, paired alternating with the vendor) | 4096^3 PF/s | 8192^3 PF/s | 8192 x 8192 x 16384 PF/s |
|---|---|---|---|
| `gemm_g1250_tdm_stage_f4_256x256_ks4_s2_epit_roles` (256 x 256, 4 waves, **K slots of 512 in a 2-slot ring**, TDM row padding, LDS-staged epilogue with a tensor store, wave roles) | 7.1 - 7.35 | **9.99 - 10.04** | 10.0 |
| the same in a 4 x 4 lockstep multicast cluster (`..._epit_roles_c4x4_lock`) | **7.90** | 9.99 | |
| previous best `..._ks2_epit_roles_relax` (K slots of 256, 3-slot ring) | 5.4 | 9.41 | 10.0 |
| vendor MXFP4 GEMM | 11.0 | 12.4 - 12.5 | |

That is about 80 % of the vendor at 8192^3 and 65 - 72 % at 4096^3. The 512-K kernel needs K a multiple of 512 (the host rejects other K; the 256-K kernels stay for K = 256 mod 512) and shows
no K-dependent dip (sweep at 4096 x 4096: K = 2048 5.5, 6144 7.7, 8192 9.1 PF/s; 8192 x 8192 x 4096 8.4). Progression at 8192^3: 0.98 (first stage-template kernel, 128 x 64) -> 1.37
(cooperative loads) -> 2.44 (two K sub-steps per slot) -> 3.3 (256 x 256 tile) -> 6.7 (TDM row padding: 8-way LDS bank conflicts removed) -> 8.1 (LDS-staged epilogue with a tensor store) ->
9.3 (wave roles + relaxed wait) -> **10.0 (K slots of 512 in a 2-slot ring: one barrier and one stage per 512 K)**. Per-K-step stamps (shader-cycle counter, `RingOpts::stamps`, host
`STAMPS=1`) of the 512-K kernel at 8192^3: window (fragment reads + WMMA issue) 81 %, barrier 12 %, stage issue 3 %, wait 3 %; per tile the prologue is 7 %, the K loop 78 % and the
epilogue 14 - 15 % (about 7,000 cycles: a fixed staged-tensor-store latency, the same for bf16 output). The K loop alone runs at the vendor's all-in cycles per WMMA; the remaining gap is
the prologue / epilogue per tile, which the vendor hides with a persistent loop (an overlap needs a staging buffer apart from the ring, which does not fit beside a 294 KiB ring).
Did not help (all measured, opt-in or dropped): cluster multicast on the 256-K kernels (-7 .. -22 %; now neutral at 8192^3 and +8 % at 4096^3 with 512-K slots), the preshuffled scale layout,
a 4-slot ring, a tile-order swizzle, a lazy prologue, instruction prefetch, a 256 x 128 tile, tile staggering (+5 % at 4096^3, -2 % at 8192^3), the persistent grid (+0.5 %), and the operand
swap with a bf16 epilogue (direct stores or a conflict-free staged tile): same or longer epilogue, 12 % slower window. Details: [`mxfp4-gemm-gfx1250-plan.md`](mxfp4-gemm-gfx1250-plan.md).

## 5. Optimisation log (MLA; what helped, what did not, and why)

The MXFP4 GEMM's log is in §4 above. A common lesson of both: LDS bank conflicts from row strides that are multiples of the bank width
were the largest single factor (2 - 2.5 x in each kernel); the TDM padding fixes them for free.


| step | effect | why |
|---|---|---|
| first correct kernel (one page per window, every wave loads the whole page) | 4.4 - 6.0 x the vendor | - |
| batch the K / V fragment reads before the WMMAs | small | the register allocator had reused one fragment's registers, serialising read -> wait -> WMMA |
| one quarter page per wave (not the whole page) | small | 4x redundant traffic, but traffic was not the limiter yet |
| **padded LDS page layout** (16 B after every 512 B) | **2.5 x** | token rows at a 512-byte stride put a K read's 16 rows and a transposing V read's 8 rows in one bank |
| in-place O rescale and accumulation | -3 % | ~155 `v_mov` per page of rebuilt fragments and loop-carried copies |
| two pages per window, lazy rescale | -18 % | barriers and exchanges per token halved; most rescales skipped |
| epilogue through LDS, 16-byte stores; mask only in the partial tile | -2 .. -5 % | 64 scalar stores per lane; one add + compare + select per element |
| deeper ring (4 slots) | 0 | the load latency was already hidden: not load bound |
| cross-tile prefetch window (S of tile t + 1 inside window t) | -2 % | the synchronisation, not the S latency, was the critical path |
| async per-lane KV loads instead of the TDM | +30 % (worse) | more instructions, no deeper prefetch than the TDM ring |
| instruction prefetch at entry | 0 | - |
| **wave-split restructure** (each wave its own pages, private ring, final merge) | -16 .. -25 % at large shapes | no barrier and no cross-wave exchange in the loop |
| interleaving the 4 S chains, V fragments one group ahead | -10 % | one wave per SIMD: only in-order ILP hides latency |
| dims split across wave pairs (`_wsd`) | +24 .. +39 % (worse) | S computed twice and twice as many pages per wave outweigh the smaller register footprint |
| branch-free rescale (after the I1 select fix in the back end) | 0 | the lazy branch was already cheap |
| cross-page form: page i's softmax with page i - 1's PV (`_wsp`) | +12 .. +31 % (worse) | with two private slots per wave the next page's load can only start after PV(i - 1): exposed load latency |
| intra-page interleave regions (`_wsi`, authored order) | +2 .. +10 % (worse) | the always-on rescale; kept as the back-end Weaver's (B4) baseline |
| KV split across work-groups + a combine kernel (`_wsk` + `_cmb`) | slower (0.52 - 0.84 x) | every split pays the fixed per-work-group cost again, plus a second launch |
| **timestamp profile (`shader_cycles`) -> the cross-wave merge's LDS addresses hoisted (DS offset field)** | **-10 .. -48 %** | the merge's 256 stores per lane recomputed their addresses with `v_mul_lo_u32`: 1551 -> 290 instructions; the merge was 41 % of a one-page kernel |
| scalar loads for the sequence bounds / scales / page ids (no LDS page table, no prologue barrier); full page pairs + an odd tail page | -3 .. -8 % | a shorter dependent prologue; no masked pad page |
| the GEMM compile policy (`CompileProfile::Gemm`) | -3 .. -5.5 % | its lowering passes and scheduler preset |
| lean softmax (no I1 loop variables / probability selects: a masked logit's exp2 underflows) | -2 .. -4 % | 64 VALU and the variable copies per page |
| exponents, exps and sums batched | -0.5 .. -1.2 % | 64 `v_nop` per pass from the TRANS write-after-read (the next sub overwrote the exp's source) |
| **the sequence's last page computed apart** (the only masked page) | **-2 .. -5 %** | no mask code on the full pages |
| page sharing for few pages: `_ws4` (1 page), `_ws2` (2 - 8 pages) | 1 x 64: 7.7 -> 4.3 us | one page on one wave is a full dependency chain; split the V dims over the waves |
| instruction prefetch with MODE.SCALAR_PREFETCH_EN set (before: a NOP), windows inside the code | 0 | not instruction-fetch bound; a window past the end of the code faulted (illegal access) |
| KV split with the last-arriver combine in the same launch (`_wskf`, g24 atomics) | 1.2 - 1.7 x slower | every split pays the fixed per-work-group cost again; the combine is a global-memory latency chain |
| the first page's tensor load right after the sequence bounds (`_wse`) | 0 (±1 %) | its latency was already overlapped by the Q loads |
| the final reduce's LDS loads hoisted before the max / sum arithmetic | +1 .. +6 % (worse) | register pressure / waits |
| cross-wave merge by LDS float atomics (`_wsa`: max exchange, 256 `ds_add_f32` per lane, 16 instead of 64 b128 final reads) | +2 .. +16 % (worse) | four waves adding into the same words cost more than the reads they replace; **nondeterministic**: 5 runs, 5 output hashes (cos_diff identical) |
| direct epilogue for `_ws4` (no cross-wave merge) | slower (5.1 vs 4.3 us) | - |
| the rope reads added to the page plan's read model | 0 (same plan) | no single TDM pad serves both the 512-byte K rows and the 64-byte rope rows; the rope chunk stays at 2x its ideal (~0.3 us of 52); a separate rope TDM with its own pad would be needed |

### Short sequences against the vendor (B = 4, e08, us; plain timed runs)

| ctx | vendor | `_ws` | `_ws2` | `_ws4` | first kernel (`mla_g1250_qh16`) |
|---|---|---|---|---|---|
| 64 | 2.95 | 8.05 | 5.77 | 4.51 | 8.07 |
| 128 | 3.01 | 8.12 | 5.89 | 5.36 | 7.95 |
| 256 | 3.69 | 8.23 | 6.70 | 6.98 | 9.00 |
| 512 | 7.78 | 9.37 | 8.47 | 10.17 | 11.10 |
| 1024 | 9.15 | 11.21 | 11.78 | 16.37 | 15.33 |
| 2048 | 12.16 | 14.73 | 18.21 | 28.04 | 23.50 |
| 4096 | 18.25 | 21.71 | 30.93 | 52.08 | 40.36 |

The vendor runs ~3 us flat up to 4 pages per sequence and steps to 7.8 us at 8 pages (presumably a second code path or its pipeline
fill); above that it fits ~6.3 us fixed + 0.187 us per page and work-group, `_ws` ~7.6 us + 0.22 us. At 4 x 1000 the 2 us gap is
~1.3 us fixed and ~0.5 us per-page rate. Structure (replica §8, the kernel name `16mx1_32nx4_np_3p`): 4 waves split each 2-page window's
tokens (32 each), P goes through LDS, each wave owns 128 V dims: no final cross-wave merge, a per-window exchange instead. That is our
first kernel's split, whose per-window cost is ~2.5 x the vendor's; `_ws` replaces the exchange by one final merge (~25 % of a short
kernel). Copying the vendor's quarter-page tensor loads into `_ws` (`_wsq`, 4 ops per page): 1.00 - 1.04 x (worse), dropped. Matching
the vendor on short sequences means bringing the first kernel's per-window exchange down to the vendor's cost (a redesign). The vendor's
KV path is per-lane async loads (`global_load_async_to_lds_b128`, no tensor ops); the first page of each `_ws` wave by async loads
(`_wsy`, 72 per lane): 1.06 - 1.12 x (worse), dropped. The prologue itself (bounds / scales / page-table scalar loads) is ~5 % of a
one-page kernel by the stamps, and issuing the first tensor load right after the bounds (`_wse`) measured 0.

### Profiling inside a kernel: shader-clock stamps

`FnBuilder::shader_cycles()` reads the 64-bit shader-clock counter (`s_get_shader_cycles_u64`; `s_getreg` of the older SHADER_CYCLES
register reads 0 on this architecture). A debug variant stores one wave's stamps at chosen checkpoints over an output it does not need
(the MLA's `mla_g1250_qh16_wsstd` writes wave 0's stamps over the LSE; `g1250_mla_host` prints them with `PRINT_LSE=1`). The stamps give
the phase split of one work-group (prologue, page loop, cross-wave merge, final reduce); at one page the merge was 41 % of the kernel,
which no instruction count of the loop had pointed to. The variant stays out of the registry (its timing instruction has no in-process
encoding yet) and is built on demand.

Two more debugging aids in the same host: `DUMP=<file>` writes the raw output (to diff two kernels element by element; it located the
I1-select miscompile to 4 rows of both lane halves) and `BADMAP=1` prints which (head, 16-dim tile) blocks exceed the tolerance.

### Per-phase microbenchmarks of the window kernel (e08, 2026-11-20)

`mla_g1250_qh16_ph_<mask>` (`probes/g1250_mla_phase.rs`, `tools/mi450/mla_phase.sh`) runs only selected blocks of the one-barrier window kernel's loop (`_w1` / `_w1i` structure, same ABI, extents, padded LDS ring
layout and fragment address expressions) on window 0 staged once into the ring, `nwin` times, and writes each wave's loop cycles to R. Blocks (mask bits): 1 = S (K fragments, 18 S WMMA), 2 = PV (P and V fragments, 16 PV WMMA),
4 = alpha (the O rescale, with PV), 8 = softmax (logits, per-wave maxima to RED, the exchange barrier, window maximum, exponents / sums, P to PBUF2), 16 = a work-group barrier. One work-group (B = 1), two sequence lengths
(16,384 / 32,768 tokens = 128 / 256 windows of 2 pages); the per-window cost is the difference of the two runs divided by 128 windows (fixed parts cancel). One window = 128 tokens:

| blocks | cycles per window (shader clock) | us per window | share of the compute-only loop |
|---|---:|---:|---:|
| loop skeleton (nothing) | 36 | 0.018 | |
| barrier | 52 (16 above the skeleton) | 0.025 | |
| **S** (K reads + 18 WMMA) | 298 | **0.125** | 18 % |
| **PV** (P + V reads + 16 WMMA) | 262 | **0.132** | 19 % |
| PV + alpha | 395 | 0.215 (alpha **0.083**) | 12 % |
| **softmax** (with its barrier) | 665 | **0.295** | 44 % |
| S + PV | 637 | 0.239 (sum of the parts 0.257: no overlap) | |
| S + PV + alpha | 647 | 0.308 | |
| S + PV + alpha + softmax | 1,524 | **0.677** (sum of the parts 0.635: nothing hides) | 100 % |
| the same + a barrier | 1,685 | 0.718 | |

In situ (B = 1, 128-token windows, MLA_OK): `_w1i` **0.958 us** per window ((251.2 - 128.5) / 128), the replica **0.379 us** ((102.9 - 54.3) / 128), so `_w1i` is 2.5 x the replica. Reading it: the compute-only
blocks already take 0.68 us, 1.8 x the replica's whole window (staging, waits and barriers included), the remaining 0.24 us in situ is the TDM staging, the tensor wait and the barriers. The MMA blocks (S + PV, 34 WMMA) cost
0.24 - 0.26 us = about 16 cycles per 16 x 16 x 64 fp8 WMMA, twice the WMMA's own issue time (LDS-read bound); the softmax block is 44 % of the compute: about 100 VALU / TRANS instructions at the 6 - 7 cycles per VALU that
the GEMM epilogue analysis found, and none of it overlaps the WMMAs (0.677 combined vs 0.635 as parts). With perfect hiding of the softmax and the alpha rescale under the MMA blocks the window would be about 0.26 - 0.3 us,
which is the replica's 0.38 us budget: the lever is overlap (and the lazy rescale), not the MMA blocks.

#### Does the interleave hide the softmax? (same microbenchmarks, e08, 2026-11-20)

More mask bits: 0x20 the w1i-style iteration (the softmax of window t + 1 emitted in 8 steps between the S / PV WMMAs, one barrier at the end), 0x40 lazy O rescale (a uniform `any_lane(alpha < 1)` branch),
0x80 an independent VALU block (8 chains, 128 instructions; 0x200: mul + add instead of mul + exp2), 0x100 the WMMA stream ordered round-robin over the independent chains (a dependent WMMA never directly follows its
producer), 0x400 lean softmax arithmetic (no per-token mask, which only the last window needs; the scale folded into one fma per exponent; the maximum on the raw scores): us per window, B = 1:

| experiment | us per window |
|---|---:|
| VALU block + barrier (128 instructions, mul + exp2 or mul + add) | 0.19 (0.165 without the barrier: about 3 - 3.7 cycles per VALU instruction, TRANS or not) |
| PV block + barrier | 0.19 (0.13 without the barrier: the barrier aligns the four waves' V reads and they collide in the LDS) |
| PV + VALU block, sequential | 0.358 (sum of the parts 0.36) |
| PV + VALU block, interleaved (adjacent WMMA pairs / spread over independent chains) | 0.330 / 0.327 (-8 %; the max of the parts would be 0.19) |
| the same with mul + add (no TRANS) | 0.337 / 0.328 |
| S + VALU: sequential / interleaved | 0.354 / 0.325 |
| all blocks, sequential blocks + barrier | 0.718 |
| all blocks, interleaved (w1i-style) | 0.813 (worse: the extra barrier phases and register pressure outweigh the 8 %) |
| interleaved + lean softmax | 0.718 |
| interleaved + lean + lazy rescale (steady state: the branch is not taken) | **0.673** |

So a single wave does NOT overlap VALU with WMMA on this part, with independent chains, TRANS or plain VALU, in any order I can express (at most 8 % of the VALU time); the combined cost is the sum, and the levers
are fewer instructions: the lean softmax arithmetic saves 0.095 us per window (about 90 of about 170 VALU), the lazy rescale 0.045 us (the steady state, where the maximum has stopped growing), together 0.74 -> 0.67 us
of the 0.72 us compute-only window. The replica's whole window is 0.379 us: its softmax must be below about 0.12 us (about 60 VALU) with WMMA issue the other 0.25, so the remaining gap is the MMA blocks (0.13 + 0.13
us for 34 WMMA, twice the WMMA time: LDS-read bound, and worse when the waves are barrier-aligned) and the VALU count, not the interleave order. Not measured here (it needs a different structure): two waves per SIMD, where
one wave's VALU runs under the other's WMMA (the occupancy result for the GEMM was negative, but that loop was WMMA-bound; here the VALU share is 44 %).

#### Two waves per SIMD (e08, 2026-11-20)

`probe_g1250_waves_<hex g1 g0>` (`probes/g1250_waves.rs`, host `tests/gpu/g1250_waves_host.cpp`, `tools/mi450/waves.sh`): one 256-work-item work-group = 8 waves, waves w and w + 4 share a SIMD; group 0 = waves 0 - 3, group 1 =
waves 4 - 7; per group a block set: 1 = a WMMA block (2 A + 16 B fragments from LDS, 16 fp8 16 x 16 x 64 WMMA into 8 accumulators), 2 = a VALU block (8 independent mul + exp2 chains, 128 instructions). Cycles per loop iteration per wave:

| group 0 | group 1 | group 0 waves | group 1 waves | |
|---|---|---:|---:|---|
| WMMA | idle | 576 | | one wave per SIMD (LDS-read bound: 36 b128 reads per iteration) |
| VALU | idle | 431 | | about 3.4 cycles per VALU instruction |
| WMMA + VALU in one wave | idle | 690 | | single-wave baseline: 31 % under the sum of the two (1,007), 20 % above the max |
| **WMMA** | **VALU** | **576** | **455** | **two waves per SIMD: the WMMA wave is unaffected, the VALU wave 5 % slower: combined = max** |
| VALU | VALU | 465 | 817 | the VALU pipe does not scale: the second wave takes 2 x |
| WMMA | WMMA | 1,152 | 1,152 | the LDS / matrix path does not scale either: 2 x |
| VALU | WMMA + VALU | 443 | 731 | |

So the SIMD does overlap a VALU-only wave with a WMMA / LDS wave (unlike the single-wave interleave of the section above): 576 instead of 690 for the same work, and 576 instead of 1,007 for strictly sequential work; two waves of the same
kind just serialize.

**MLA structure that uses it** (a dim-split, sketch): 8 waves per work-group, a SIMD hosts one MMA wave and one softmax wave. MMA waves (0 - 3, as today: each owns 128 V dims, 8 O accumulators = 64 VGPRs) run S(t + 2) and PV(t),
store the S tile (16 heads x 128 tokens of the window, f32, 8 KiB) to an LDS ring SBUF and read P(t) from PBUF; softmax waves (4 - 7: 32 tokens each) read S(t + 1) from SBUF, compute the logits (lean arithmetic), their partial
maxima (to RED), the exponents, the row sums and the P tile (fp8, 2 KiB) into PBUF, and the lazy O-rescale factor for the MMA waves. One work-group barrier per window publishes everything (the w1 software pipeline, one window
deeper because the maximum needs a second hop: iteration t reads S(t + 1), writes its partial maxima; iteration t + 1 reads the 4 partials and computes P(t + 1); PV(t + 1) runs in iteration t + 2). Budget: SBUF 3 slots x 8 KiB
and PBUF / RED a few KiB on top of a KV ring of 3 slots (3 x 2 pages x about 37 KiB = 221 KiB) fit the 320 KiB; each wave may use 512 VGPRs (O 64, S fragments, K / V fragments, 128 for the MMA wave; the softmax wave needs about
64). Expected per window: max(MMA about 0.25 us, softmax about 0.30 us lean about 0.2 us) + the barrier, i.e. about 0.3 us against 0.68 us compute-only today (the replica: 0.379 us in total). The S / P hand-off through the LDS costs
about 10 KiB per window and one more barrier phase of latency, not bandwidth.

**V-read stagger: rejected.** Rotating each wave's V tile order by 2 tiles per wave (wave w reads tile (nt + 2 w) % 8 at step nt; `PH_STAG`, mask bit 0x800) makes the PV block 0.13 -> 0.46 us (with the barrier 0.19 -> 0.46) and the
whole compute-only window 0.72 -> 0.96 us: the shifted addresses leave the padded layout's conflict-free pattern for `ds_load_tr8`. The aligned-waves collision seen with the barrier (PV 0.13 -> 0.19 us) needs a different remedy
(for example the two-role structure above, which removes the barrier alignment of the MMA waves' V reads from the softmax phase).

#### Two-role MLA prototype (e08, 2026-11-20)

`mla_g1250_qh16_roles` (`probes/g1250_mla_roles.rs`, run by `tools/mi450/mla_roles.sh`, which builds the MLA host with 256 work-items): 8 waves, waves 0 - 3 = MMA waves (K / V / P fragments, 18 S WMMA + 16 PV WMMA, the S tile to the LDS ring
`SBUF`), waves 4 - 7 = softmax waves (S from `SBUF`, partial maxima through `RED`, lean arithmetic, P to `PBUF`), one work-group barrier per window, the w1 pipeline one window deeper (iteration t: PV(t), S(t + 3) -> SBUF gen t + 3 /
softmax: S(t + 2), maxima gen t + 2, window maximum gen t + 1, P(t + 1)). Stub data (window 0 staged once), no KV streaming, no rescale. **The tile protocol, the roles / barrier-count rules and the bounds prover accept it with no new
trusted fact** (one prover quirk: `AndI32(const 2, 1)` is bounded as 3, so the ring slot of a constant generation is passed explicitly in the prologue; the split barrier form, `s_barrier_signal` early on one role and `s_barrier_wait`
late, is rejected by the protocol's phase model across roles, so both roles use the full barrier). 464 VGPRs (two waves per SIMD fit), LDS 182,272 B with a 2-slot KV ring (3 slots: about 256 KiB).

Per 128-token window, B = 1, compute-only, us (the same method as above):

| variant | us per window |
|---|---:|
| sequential blocks in one wave set (the phase kernels) | 0.718 |
| the same with lean softmax + lazy rescale | 0.673 |
| **two roles, as built** (hand-off, 8-wave barrier, lean softmax, no rescale) | **0.452** (0.496 before putting the softmax loads ahead of the maxima store) |
| two roles, MMA waves' work skipped (hand-off + barrier + the softmax waves) | 0.324 |
| two roles, softmax waves' math skipped (the MMA waves + hand-off + barrier) | 0.426 |
| two roles, both skipped (loops, hand-off, barrier only) | 0.205 |

So the roles do overlap the softmax with the MMA blocks (the softmax adds 0.03 us on top of the MMA waves), and the compute-only window drops 0.72 -> 0.45 (-37 %); but it is not the hoped-for 0.3: the MMA waves' path
(fragment loads -> 34 WMMA -> S store -> `wait_lds(0)` -> barrier) is 0.43 us and the hand-off / barrier skeleton 0.2 us (LDS latency chains of the softmax waves, an 8-wave barrier) adds to it almost serially. Candidate
next steps (not built): issue the next window's K / V fragment loads before the barrier (the slot is already published), two windows per iteration (one barrier per two windows needs two more ring generations), counted
`s_wait_dscnt` instead of `wait_lds(0)`. In situ the replica is 0.379 us and `_w1i` 0.958 us; with the roles the compute part falls by 0.27 us, so a real kernel (3-slot ring, KV streaming) would land around 0.7 us.

## 6. Incidents

1. **A device hang during GEMM development** (full account in [`gfx1250-hang-rules.md`](gfx1250-hang-rules.md)): a sweep left a process
   blocked in the driver and the device needed an administrator reset. Root cause of the preceding wrong results: the shared wait pass
   used the previous architecture's LDS counter range, so LDS loads were treated as complete too early. The hang itself was not
   reproduced. Added: the independent wait re-check with the gfx1250 counters, the counter-range rule, the one-work-group-first policy
   and the 8-wave regression kernel.
2. **An illegal memory access in the MLA kernel** (once, in back-to-back timed launches at the largest shape; not reproduced in 50+ runs).
   The kernel ended with a tensor load still in flight (the pipeline's last steady iteration stages one clamped tile nothing waits for),
   so a tensor write could land in LDS already owned by the next work-group on the CU, e.g. its page table. Added: the drain before
   `s_endpgm` in the kernel and in the pipeline driver (`PipeBody::finish`), and the G5 rule that TENSORcnt is 0 on every path to
   `s_endpgm`. A second "hang" observed during that stress was a foreign process occupying the GPU (it ran correctly on another GPU).
3. **A descriptor built from zeroed registers** (a code-generation bug in a GEMM): led to the tensor-bounds rule (provenance, extent,
   range) that is now mandatory for every non-replay kernel.

## 7. Open items

- **VGPR window churn**: the wave-split MLA flips `s_set_vgpr_msb` about 3 times per WMMA (accumulators spread over the banks);
  window-aware placement of accumulators vs streamed operands (back end).
- **Single-wave scheduling**: with one wave per SIMD, the remaining MLA gap is latency inside a wave (softmax VALU / TRANS against the
  WMMAs); in progress in the back end: an instruction-level software pipeline of page i's softmax under page i - 1's PV WMMAs.
- **Short sequences** (MLA ~1.2 x the vendor at 4 x 1000 and 32 x 2048): the fixed per-work-group cost (prologue chain, first pages,
  cross-wave merge, final). Measured and rejected so far: KV split (two launches or fused), early staging, instruction prefetch, load
  hoisting, an atomic merge (table above). The K reads are volume-bound (each wave reads the whole page; conflict-free per the planner).
  The vendor's advantage there is structural (no final merge, a cheap per-window exchange; §5 "Short sequences against the vendor").
- **Workgroup ids in a cluster launch**: the lowering of the workgroup-id ops reads TTMP9 / TTMP7, which hold the cluster ids in a
  cluster launch; cluster kernels compute their tile ids by hand (TTMP6 position + cluster id x dims). Planned: lower them that way when
  the kernel declares cluster dims, or reject the plain op there.
- **MXFP4 GEMM** (e08 round, 2026-11-19): the best kernel is `..._256x256_ks4_s2_epit_roles` (512-K slots in a 2-slot ring: 10.0 PF/s at 8192^3, 7.1 - 7.9 at 4096^3,
  80 % / 65 - 72 % of the reference library kernel). Per-K-step stamps (`RingOpts::stamps`, host `STAMPS=1`) put the K loop at the reference's cycles per WMMA; the remaining
  gap is the per-tile prologue (7 %) and the staged epilogue (14 - 15 %, a fixed ~7,000-cycle tensor-store latency whatever the output width). Tried and dropped, all measured:
  one tile per CU (4096^3: prologue 4,960 / loop 19,100 / epilogue 5,600 - 7,700 cycles; the 4 x 4 lockstep multicast removes the K = 4096 anomaly: 7.9 vs 6.3 PF/s, dispatch table in the plan doc),
  two 256 x 128 workgroups per CU (-19 % at 4096^3: the loop is WMMA-bound), a 192 x 256 tile (5 - 12 % slower at the same K step; with a separate 96 KiB bf16 staging region and the tensor stores draining behind the next tile the epilogue falls to
  2,250 cycles but the forced 256-K slot makes the loop 19 % costlier per WMMA: 8.2 vs 8.56 PF/s), epilogue stores from registers overlapped into the next tile's K loop (the
  stores contend with the TDM loads: loop +17 %, 7.8 - 8.4 PF/s), operand swap with bf16 output (direct or conflict-free staged: same epilogue time, 12 % slower window with the
  row permutation), tile staggering, wave-order permutations (the wave skew follows the wave id), lazy prologue, instruction prefetch, 256 x 128 tile, multicast (neutral at
  8192^3, +8 % at 4096^3 with 512-K slots). The code of the dropped variants stays as options (archive tags `archive/g1250-gemm-variants-2026-11-19{,b,c}`). Not done: partial M tiles
  (8192 is not a multiple of 192) need dynamic TDM dimensions and a relational footprint proof in the tensor-bounds prover.
