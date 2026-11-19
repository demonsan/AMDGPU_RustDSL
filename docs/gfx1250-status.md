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

| B x ctx | first kernel (`mla_g1250_qh16`) / vendor | wave-split (`mla_g1250_qh16_ws`) / vendor |
|---|---|---|
| 1 x 64 | - | 2.67 |
| 4 x 1000 | 1.65 | **1.34** |
| 32 x 2048 | 1.90 | **1.35** |
| 128 x 4096 | 1.38 | **0.97** |
| 128 x 8192 | 1.65 | **1.12** |
| 512 x 8192 | 1.34 | **1.11** |

`_ws` is the default. Structure of both: [`gfx1250-replica.md`](gfx1250-replica.md) §9.

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

### Profiling inside a kernel: shader-clock stamps

`FnBuilder::shader_cycles()` reads the 64-bit shader-clock counter (`s_get_shader_cycles_u64`; `s_getreg` of the older SHADER_CYCLES
register reads 0 on this architecture). A debug variant stores one wave's stamps at chosen checkpoints over an output it does not need
(the MLA's `mla_g1250_qh16_wsstd` writes wave 0's stamps over the LSE; `g1250_mla_host` prints them with `PRINT_LSE=1`). The stamps give
the phase split of one work-group (prologue, page loop, cross-wave merge, final reduce); at one page the merge was 41 % of the kernel,
which no instruction count of the loop had pointed to. The variant stays out of the registry (its timing instruction has no in-process
encoding yet) and is built on demand.

Two more debugging aids in the same host: `DUMP=<file>` writes the raw output (to diff two kernels element by element; it located the
I1-select miscompile to 4 rows of both lane halves) and `BADMAP=1` prints which (head, 16-dim tile) blocks exceed the tolerance.

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
- **Short sequences** (MLA 1.34 x the vendor at 4 x 1000, 2.7 x at 1 x 64): per the stamps, one page on one wave (~55 % of a one-page
  kernel) and the final cross-wave reduce (~25 %); next step: exchange the wave maxima first so each wave stores its O already scaled.
- **Workgroup ids in a cluster launch**: the lowering of the workgroup-id ops reads TTMP9 / TTMP7, which hold the cluster ids in a
  cluster launch; cluster kernels compute their tile ids by hand (TTMP6 position + cluster id x dims). Planned: lower them that way when
  the kernel declares cluster dims, or reject the plain op there.
- **MXFP4 GEMM** (e08 round, 2026-11-19): the best kernel is `..._256x256_ks4_s2_epit_roles` (512-K slots in a 2-slot ring: 10.0 PF/s at 8192^3, 7.1 - 7.9 at 4096^3,
  80 % / 65 - 72 % of the reference library kernel). Per-K-step stamps (`RingOpts::stamps`, host `STAMPS=1`) put the K loop at the reference's cycles per WMMA; the remaining
  gap is the per-tile prologue (7 %) and the staged epilogue (14 - 15 %, a fixed ~7,000-cycle tensor-store latency whatever the output width). Tried and dropped, all measured:
  a 192 x 256 tile (5 - 12 % slower at the same K step; with a separate 96 KiB bf16 staging region and the tensor stores draining behind the next tile the epilogue falls to
  2,250 cycles but the forced 256-K slot makes the loop 19 % costlier per WMMA: 8.2 vs 8.56 PF/s), epilogue stores from registers overlapped into the next tile's K loop (the
  stores contend with the TDM loads: loop +17 %, 7.8 - 8.4 PF/s), operand swap with bf16 output (direct or conflict-free staged: same epilogue time, 12 % slower window with the
  row permutation), tile staggering, wave-order permutations (the wave skew follows the wave id), lazy prologue, instruction prefetch, 256 x 128 tile, multicast (neutral at
  8192^3, +8 % at 4096^3 with 512-K slots). The code of the dropped variants stays as options (archive tags `archive/g1250-gemm-variants-2026-11-19{,b,c}`). Not done: partial M tiles
  (8192 is not a multiple of 192) need dynamic TDM dimensions and a relational footprint proof in the tensor-bounds prover.
