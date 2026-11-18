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

| kernel | 4096^3 PF/s | 8192^3 PF/s | 8192 x 8192 x 16384 PF/s |
|---|---|---|---|
| `gemm_g1250_tdm_stage_f4_256x256_ks2_epit_roles` (256 x 256, 4 waves, ks 2, TDM row padding, LDS-staged epilogue with a tensor store, wave roles) | **6.25** | 8.47 | 8.33 |
| `..._256x256_ks2_epit_roles_relax` (the same, tensor wait relaxed to the next tile only) | 5.34 | **9.27** | **10.0** |
| vendor MXFP4 GEMM | 11.1 | 12.9 | |

That is about 56 % of the vendor at 4096^3, 72 % at 8192^3 and 78 % of the vendor's 8192^3 rate at K = 16384. **Dispatch rule** (the
relaxed wait's benefit depends irregularly on K; measured table in [`mxfp4-gemm-gfx1250-plan.md`](mxfp4-gemm-gfx1250-plan.md)): the
relaxed kernel for K <= 3584 or K >= 5120, the plain `epit_roles` for 3840 <= K <= 4608. Progression at 8192^3: 0.98 (first stage-template
kernel, 128 x 64) -> 1.37 (cooperative loads) -> 2.44 (two K sub-steps per slot) -> 3.3 (256 x 256 tile) -> 6.7 (TDM row padding: 8-way
LDS bank conflicts removed) -> 8.1 (LDS-staged epilogue with a tensor store) -> 9.3 (wave roles + relaxed wait). Did not help (all
measured, opt-in or dropped): cluster multicast, free-running or in lockstep, 2 x 2 and 4 x 4, on the unpadded tile (-7 .. -15 %) and on
the padded LDS-epilogue tile (-11 .. -22 %); the preshuffled scale layout (alone and composed); a 4-slot ring; a tile-order swizzle; the
persistent grid (+0.5 % at best on the best kernel; a cross-tile overlap cannot fit beside the LDS-staged epilogue, which needs 264 of the
320 KiB). A deletion profile of the padded kernel before the epilogue work put the time in the tensor transfers and the work-group barriers,
not in the WMMAs. Details: [`mxfp4-gemm-gfx1250-plan.md`](mxfp4-gemm-gfx1250-plan.md).

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
- **Short sequences** (MLA at small B x ctx: 1.8 - 2 x the vendor): a wave's pages run back to back, plus a fixed prologue chain
  (sequence bounds -> page table -> first tensor load).
- **Workgroup ids in a cluster launch**: the lowering of the workgroup-id ops reads TTMP9 / TTMP7, which hold the cluster ids in a
  cluster launch; cluster kernels compute their tile ids by hand (TTMP6 position + cluster id x dims). Planned: lower them that way when
  the kernel declares cluster dims, or reject the plain op there.
- **MXFP4 GEMM**: the C epilogue is solved (LDS staging + one tensor store per wave) and the remaining gap to the vendor (about 22 - 44 %
  depending on the shape) is in the tensor transfers and the barrier coupling; the K dependence of the relaxed wait (above) is unexplained
  (not a power-of-two stride, not tile order). Tried and dropped: multicast, scale layout, deeper ring, persistence.
