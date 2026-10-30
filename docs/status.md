# Implementation status

Updated after the first end-to-end milestone on gfx950 (`mi355-gpu-53`).

## Verified working

| Step | Evidence |
|---|---|
| Rust toolchain only inside a container | `tools/container/Containerfile` (ubuntu 22.04 + rustup 1.83.0), built with rootless podman; host and shared home untouched |
| Workspace builds and tests | `tools/container/cargo.sh build` clean, `cargo test` → 8 backend tests pass |
| Typed IR + verifier | `amdgpu-dsl`: SSA values, loop variables, scope and type checking |
| Lowering + own register allocation | `amdgpu-compiler`: MIR, linear-scan RA, copy coalescing, wait-state insertion |
| Assembly emission | `amdc emit attention_fwd` → gfx950 asm accepted by the ROCm assembler |
| Code object loads and dispatches | HIP module load + launch from `tests/gpu/attn_host.cpp` |
| ABI + memory path | `probe_store` kernel: `o[tid] = q[tid]` plus a kernarg scalar, `PROBE_OK` |
| Hardware transcendental behaviour | `probe_math`: `exp(1) = 2.0`, `exp(2) = 4.0`, `rcp(3) = 0.333333313` |
| **Attention kernel numerics** | `attention_fwd` on gfx950 vs CPU reference: `H=1 S=64` max_abs_err `9.4e-05`; `H=2 S=128` max_abs_err `6.0e-05`; `bad = 0` in both |
| MFMA infrastructure | `probe_mfma`: emitted `v_mfma_f32_16x16x16_f16 v[12:15], v[4:5], v[6:7], v[8:11]` with 4/2/2/4-dword tuples; all 256 accumulator elements `16.0` → `PROBE_MFMA_OK` |
| LDS, barriers, `lgkmcnt` | `probe_lds`: 64 lanes write their id to LDS, barrier, read the neighbour's value → `PROBE_LDS_OK`; descriptor declares 256 B of LDS |
| Wide MFMA (16-dword accumulator) | `probe_mfma_wide`: `v_mfma_f32_32x32x8_f16` with a 16-dword zeroed accumulator; every element `8.0` → `PROBE_MFMA_WIDE_OK` |
| FP8 MFMA | `probe_mfma_fp8`: all-ones E4M3 operands; `v_mfma_f32_16x16x32_fp8_fp8` → `32.0` (K=32), `v_mfma_f32_32x32x16_fp8_fp8` → `16.0` (K=16) → `PROBE_MFMA_FP8_OK` |
| Block-scaled fp8 MFMA | `probe_mfma_scale`: `v_mfma_scale_f32_32x32x64_f8f6f4` with E8M0 scales; `(127,127)` → `64.0`, `(127,128)` → `128.0`, `(127,0x8000)` → `0.0`, `(127,0x7f00)` → `0.0` → `PROBE_MFMA_SCALE_OK` |
| Wave-role branching (8 waves) | `probe_roles`: waves 0-3 store `111.0`, waves 4-7 store `222.0` (`marker_bad = 0`) and exchange LDS data across waves (`bad = 0`) |
| FMHA control skeleton | `probe_roles_loop`: 8 waves, role A/B, 4 iterations, two paired barriers per iteration, cross-role LDS exchange → `PROBE_ROLES_LOOP_OK` |
| FMHA data path | `probe_fmha_datapath`: distinct A and B tiles staged in LDS (`ds_write_b128`), read back (`ds_read_b128`), concatenated into `Frag8` operands and multiplied with `v_mfma_scale_f32_32x32x64_f8f6f4` → every value is `64.0` → `PROBE_FMHA_DATAPATH_OK` |
| A/B paired loops | `probe_ab_loops`: 8 waves, role A/B, outer loop with two sequential inner phase loops, paired barriers, LDS read-modify-write accumulator → `PROBE_AB_LOOPS_OK` (closed form over 512 lanes) |
| Nested loops, register accumulator | `probe_nested_acc`: inner loop updates the outer loop variable, accumulator read after the loops → `PROBE_NESTED_ACC_OK` |
| MFMA accumulation loop | `probe_loop_mfma`: `Frag16` loop variable accumulated by `v_mfma_scale_f32_32x32x64_f8f6f4` over two K=64 iterations → every element `128.0` → `PROBE_LOOP_MFMA_OK` |
| FMHA core loop | `probe_fmha_core`: A/B phase staging with paired barriers, `ds_read_b128` → `Frag8` → in-place MFMA accumulation in a loop → every value `128.0` → `PROBE_FMHA_CORE_OK` |
| **prefetching staging + folded addresses (190 TFLOP/s)** | 16-byte global loads with the next K/V tile prefetched behind the MFMA work, LDS offsets folded into DS immediates with DCE, zero-C MFMA, folded x16 scale and `v_max3_f32`: **190 TFLOP/s on 1x8x2048x2048** (vendor 518), all 6 shapes `REF_KERNEL_OK` |
| **sub-register aliases + in-place accumulators (118 TFLOP/s)** | compiler removes ~75% of the `v_mov_b32` in `fmha_fwd_hd128_mxfp8_t`: 2859 asm lines instead of 4935, 176 VGPRs; **118 TFLOP/s on 1x8x2048x2048** (vendor 518); all 6 shapes `REF_KERNEL_OK` |
| **transposed register-softmax `fmha_fwd_hd128_mxfp8_t`** | same ABI and numerics (all 6 shapes `REF_KERNEL_OK`, mean error vs CPU 1.1-1.7e-3, vs vendor 2.5-2.8e-3) with `S^T = K Q^T`: scores stay in registers (lane = query), softmax max/sum are register reductions plus one LDS exchange with lane `l ^ 32`, the online-softmax correction is lane-local, no score tile goes through LDS; 192 VGPRs, 34.8 KB LDS; **88 TFLOP/s vs 15 for the first version (5.9x faster), vendor kernel 518 TFLOP/s** |
| **ABI-compatible `fmha_fwd_hd128_mxfp8`** | our kernel launched with the vendor kernel's 656-byte kernarg on BSHD fp8 Q/K/V, E8M0 1x32 descales and a BF16 output; `exp2` softmax; **every output element agrees with a double-precision CPU model (mean error 1.2-1.7e-3, better than the vendor kernel's own 1.8-2.8e-3) and with the vendor kernel (mean difference 2.5-2.8e-3, worst 0.10) on 6 shapes up to B=3 H=3 S=768 SK=1280 and 2M outputs**; bit-identical to the vendor kernel when Q = 0 (`REF_KERNEL_OK`) |
| **512-thread hd128 attention, LDS-staged K/V, A/B role loops** | `probe_fp8_fmha_hd128c`: reference tile shapes at reference scale — eight 32-row query tiles (256 rows) per block of 512, head dim 128, KV tile 128 processed as two 64-key online-softmax substeps, K (role A, waves 0-3) and V (role B, waves 4-7) copied through shared LDS (99,456 B), K fragments by `ds_read_b128`, V fragments by `ds_read_b64_tr_b8`, separate A/B steady loops with the same barrier schedule, four register-resident O accumulators; **224 VGPRs** (reference 246); all 8 waves match the host emulation (`1.8e-4`) -> `PROBE_FMHA_HD128C_OK` |
| **head dim 128, KV tile 128** | `probe_fp8_fmha_hd128`: reference tile shapes — Q 32x128, four 32-key score tiles per iteration each the sum of two chained K=64 scaled MFMAs, two-pass 128-wide softmax, eight PV MFMAs (two 64-key halves x four 32-wide output tiles) with V through `ds_read_b64_tr_b8` at pitch 128, four register-resident O accumulators; 240 VGPRs (reference 246); `worst_err = 4.3e-7` vs the host emulation -> `PROBE_FMHA_HD128_OK` |
| **512-thread tr + A/B role-split attention** | `probe_fp8_fmha_trab`: eight query tiles in one block of 512, V double-buffered in shared LDS and copied half by role A (waves 0-3) and half by role B (waves 4-7) with their own steady loops and identical barrier schedule, V consumed with `ds_read_b64_tr_b8`, 64 keys per iteration, register-resident O; all 8 waves match the host emulation (`1.8e-4`) -> `PROBE_FMHA_TRAB_OK` |
| **64-key online softmax, V through `ds_read_b64_tr_b8`** | `probe_fp8_fmha_tr`: two score tiles and a 64-wide online softmax per iteration, quantised P as the full K=64 A fragment (lane group g keeps keys 32g..32g+31), row-major V staged in LDS and turned into the B fragment by four transposing reads, register-resident O accumulator; `worst_err = 3.4e-7` vs the host emulation, `3.8e-2` vs exact softmax -> `PROBE_FMHA_TR_OK` |
| **ds_read_b64_tr_b8 B fragment** | `probe_fp8_matmul_tr`: row-major B staged in LDS, the 32x32x64 B fragment built with four `ds_read_b64_tr_b8` reads (no host-side transpose), random fp8 data and random scales, `worst_err = 0` vs the CPU reference -> `PROBE_MATMUL_TR_OK` |
| **separate A/B steady loops (block 512)** | `probe_fp8_fmha_abloops`: waves 0-3 and 4-7 each run their own steady loop with its own backedge (A copies K, B copies V, both compute their own query tile) and the same four collective barriers per iteration plus one in the epilogue (12 static `s_barrier`, 4 branches, 208 VGPRs); all 8 waves match the host emulation -> `PROBE_FMHA_MULTIWAVE_OK` |
| **role-split producer/consumer attention (block 512)** | `probe_fp8_fmha_pingpong`: waves 0-3 (role A) copy the next K tile and waves 4-7 (role B) copy the next V tile into a double-buffered LDS region; the two role branches (begin_if/else on wave id) each end in the same workgroup barrier (8 static `s_barrier`, 3 branches); every wave reads its fragments with `ds_read_b128` + `cat_frag8`; all 8 waves match the host emulation (`2.1e-4`) -> `PROBE_FMHA_MULTIWAVE_OK` |
| **eight-wave online-softmax attention (block 512)** | `probe_fp8_fmha_multiwave`: 512 threads, each wave owns one 32-row query tile and a private 8448 B LDS region, all waves share K/V tiles and run the same workgroup-barrier sequence; all 8 waves match the host emulation (`worst_err 2.1e-4`) -> `PROBE_FMHA_MULTIWAVE_OK` |
| **online-softmax attention, O accumulator in registers** | `probe_fp8_fmha_oreg`: same algorithm with O as a loop-carried `Frag16`; each iteration rebuilds the fragment from lane values times the LDS-published row correction (new `Frag16Build` op) and feeds it back as the MFMA C input; bit-identical results to the LDS-accumulator version (`worst_err 1.8e-4` vs emulation) -> `PROBE_FMHA_ONLINE_OK` |
| **online-softmax MXFP8 attention (4 tiles, 128 keys)** | `probe_fp8_fmha_online`: flash-style loop with loop-carried running max/sum (float bits in I32 loop vars), per-row correction `exp(m_old-m_new)` broadcast through LDS, O accumulator rescaled in LDS, final 1/l epilogue; `worst_err = 1.8e-4` vs a host emulation of the same online algorithm over the kernel's quantised P bytes, `5.3e-2` vs exact softmax -> `PROBE_FMHA_ONLINE_OK` |
| **two-tile MXFP8 attention with a tile loop** | `probe_fp8_fmha_2tile`: loop over two key tiles (loop-carried tile index, scores accumulated in LDS), 64-wide row softmax, lane-group split fp8 quantisation, one PV over 64 keys; `worst_err = 1.8e-4` vs the quantised-P reference, `8.3e-2` vs exact softmax (inherent fp8 P quantisation) -> `PROBE_FMHA_2TILE_OK` |
| **single-tile MXFP8 attention** | `probe_fp8_fmha_tile`: QK^T -> row softmax -> in-kernel fp8 quantisation (scaled `v_cvt_pk_fp8_f32`) -> PV through the same scaled MFMA, `worst_err = 0.0` vs a CPU reference built from the quantised P bytes -> `PROBE_FMHA_TILE_OK` |
| fp8 QK^T + softmax | `probe_fp8_softmax`: random fp8 Q/K with random E8M0 scales -> block-scaled MFMA -> LDS transpose -> per-row max/exp/sum/normalise, `worst_err=8.1e-08` vs a CPU double reference -> `PROBE_FP8_SOFTMAX_OK` |
| fp8 matmul (real data, general) | `probe_fp8_matmul`: random E4M3 data with random per-lane E8M0 scales through the real fragment/scale layout, matched element by element against a CPU double reference (`FP8_TEST=1,2,3,4,6,7`) -> `PROBE_FP8_MATMUL_OK` |
| fp8 load path | `probe_fp8_load`: address arithmetic and the ISA fragment mapping for the real-data fp8 matmul probe -> `PROBE_FP8_LOAD_OK` |
| Pack instructions | `probe_pack`: `v_cvt_pknorm_u16_f32(1.0, 0.5)` = `0x8000FFFF`; `v_cvt_scalef32_pk_fp8_bf16` low half = `0xAAAA3838`, high half = `0x3838AAAA` → `PROBE_PACK_OK` |
| LDS self-read | `probe_lds_self2`: own-slot read after a barrier → `SELF2_OK` |
| Two-wave LDS | `probe_waves2`: own + partner-wave slot → `PROBE_WAVES2_OK` |
| `v_max3_f32`, fp8 fragments | `v_max3_f32` in the ISA of the compiler, `Frag8`/`V8` register class for the `32x32x64` operands |

`attention_fwd` is a real dense attention kernel (one lane per query row, online
softmax over the key loop, f32 accumulation, f16 output) that is fully generated
by this compiler: typed DSL → internal MIR → own RA → gfx950 assembly → `.co` →
executed and numerically verified on hardware.

## Bugs found and fixed on the way

Compiler/IR:

1. `ValueId`/`VarId` were both `u32` → conflicting `From` impls; `VarId` is a newtype now.
2. Nested DSL builder calls had to be split into locals (Rust borrow rules).
3. The allocator did not mark assigned slots → every virtual register got the same physical register.
4. Constant moves carried the value in `imm` but the emitter only read `srcs` → **every constant was emitted as 0**.
5. `max_v`/`max_s` were computed from the occupancy bitmap *after* allocation, so reclaimed high slots were not declared; code used `v208` while metadata said 140 VGPRs.
6. `BreakIf` branch polarity was inverted.

Code object / ABI:

7. Missing `.amdhsa_user_sgpr_count` → no kernarg pointer was passed.
8. Kernel descriptor not 64-byte aligned → the runtime read every field shifted; surfaced as "illegal instruction" / "memory aperture violation".
9. `kernarg_size` not 8-byte aligned → the assembler rejected the metadata.
10. Single-letter argument names (`n`) parse as YAML booleans → argument names are quoted.

Hardware behaviour that the compiler now models:

11. **`v_exp_f32` computes `2^x`, not `e^x`.** Natural exp is lowered to
    `exp2(x * log2(e))`. This was invisible to the constant-input test (`exp(0) = 1`
    either way) and only showed up with random data.
12. **`v_mfma_scale_*_f8f6f4` scales are E8M0 exponents in the low byte of a
    VGPR**, one scalar per matrix: the result is
    `2^(a-127) * 2^(b-127) * (A @ B) + C`, and A/B bytes are decoded as E4M3.
    Passing the scales in SGPRs (as the syntax also accepts) silently produced
    zero, so the DSL/lowering now always materialises them as VGPRs.
13. **LDS read results need an `lgkmcnt` wait before use.** The lowerer originally
    only tracked VMEM and kernarg loads, so `ds_read_b32` results were consumed
    too early and read as zero. `pending_lds` now forces `s_waitcnt lgkmcnt(0)`.
14. **Transcendental results are not interlocked.** A consumer issued in the
    instruction slot right after `v_exp_f32`/`v_rcp_f32` reads the stale register
    value. `hazards::fix_value_hazards` models the latency and pads with `s_nop`;
    `attention_fwd` relies on it (it computes `l_new` immediately after `p` on
    purpose).

Host harness:

13. `hipModuleLaunchKernel` needs one pointer per argument, not one pointer to the whole kernarg struct.

## Regression tests

`crates/amdgpu-compiler/tests/backend.rs` (no GPU needed):

* every registered kernel compiles and has descriptor + metadata;
* the descriptor covers every physical register the instruction stream touches;
* the kernel descriptor is emitted with `.p2align 6`;
* natural exp lowers through `v_exp_f32` with the `log2(e)` constant;
* transcendental results get wait states;
* the IR verifier rejects ill-typed programs;
* the attention IR verifies and really contains a loop.

## Resolved: the "LDS self-read / multi-wave exchange" anomaly

Earlier rounds reported that reading a lane's own LDS slot, or exchanging LDS
data across waves, "read back zero". That was **not** a hardware or compiler
problem: the kernels store raw 32-bit values (`bitcast_f32` is an alias, so the
stored word is the integer bit pattern), while those probe checks reinterpreted
the buffer as `float` and compared against `(float)tid`. Small integers as float
bits are denormal values that print as `0.0`, so the checks failed while the
hardware behaved correctly. Checks now compare raw bit patterns.

Verified after the fix (all on gfx950):

| probe | meaning |
| --- | --- |
| `probe_lds` | neighbour-slot read after a barrier (`bad=0`) |
| `probe_lds_self2` | own-slot read after a barrier (`bad=0`) |
| `probe_waves2` | two waves: own slot + partner-wave slot (`own_bad=0 partner_bad=0`) |
| `probe_roles` | eight waves: role A/B ids exchanged across waves (`bad=0`) |
| `probe_roles_loop` | 8 waves, roles, 4 iterations, two paired barriers per iteration, cross-role LDS exchange (`bad=0`, 2048 values) |

`probe_roles_loop` is the control skeleton of the FMHA reference core loop, so
the structural pieces needed for the 1:1 reproduction are in place.

## Resolved: the 16 KB LDS probe failure

The 32 B/lane variant was writing its B tile at `8192 + tid*32`, which reaches
`24560` B — past the declared 16 KB and over the A tile. It was a probe bug, not
an LDS capacity or compiler problem. With both tiles inside the declaration the
A/B data path is exact.

## IR capabilities added for the loop (round 9)

1. `SetVar` now accepts **any enclosing loop's** variable, so an inner loop can
   update an outer loop accumulator.
2. Values and loop variables stay visible **after `EndLoop`**; a loop variable
   then holds its last value (undefined if the body never ran).
3. Loop variables can have fragment types (`Frag2/4/8/16`), so a `Frag16` MFMA
   accumulator can live in registers across iterations. Copies are emitted as
   `VReadSub`/`VWriteSub` pairs.
4. Every fragment operand materialisation now calls `touch()`, so `ds_read_*`
   results feeding `cat_frag8`/MFMA get their `lgkmcnt` wait.

Still unsupported: pointer-typed loop variables.

Risk to watch: the MFMA accumulation loop produced `64.0` instead of `128.0`
in an earlier register allocation and `128.0` after the probe gained a
diagnostic store (same IR semantics). That points at a register-aliasing hazard
for MFMA D/C versus A/B that the allocator does not model; the condition needs
to be found and enforced in MIR verification.

## Hardware facts confirmed from the CDNA4 ISA (kernel wiki)

Source: `MI_Kernel_Agent/skills/kernel-wiki/references/mi300-sources/cdna4-isa`
(chapter 7, MFMA).

* The matrix core has its own register file, the **AccVGPRs**. The `ACC` field of
  an MFMA instruction selects VGPRs (0) or AccVGPRs (1) for A (low bit) and B
  (high bit), and `ACC_CD` does the same for C and D. In the code object the
  boundary between the two files is `.amdhsa_accum_offset`.
* Operands must be **contiguous and the base register aligned to the operand
  width** (a 4-register A needs a base that is a multiple of 4). Our raw/linear
  allocator aligns tuples to their width, but this is not verified yet.
* For the F16/BF16/I8 MFMA group the sources and destination must additionally be
  **even aligned**.
* MFMA ignores the execution mask (forced to 1).
* `V_MFMA_SCALE_F32_32X32X64_F8F6F4` is a **4-dword instruction**: two back to
  back VOP3Ps, the first carrying `0xD3AC` plus the two scale sources (SRC0 = A
  scale, SRC1 = B scale). `OP_SEL[0]`/`OP_SEL[1]` pick **which byte** of the
  scale VGPR is used (00 -> `[7:0]`, 01 -> `[15:8]`, 10 -> `[23:16]`, 11 ->
  `[31:24]`). There is one scale byte per lane per 32 K-values and the scale
  applies to that instruction only.
* The F8F6F4 MFMA takes 32 cycles when either A or B is F8 and 16 otherwise
  (64/32 for the 32x32x64 form).

## MFMA accumulator is in place

`lower.rs` now materialises the incoming accumulator into the destination
registers and emits `C == D` (the form the aiter reference kernel uses), and
`regalloc` rejects any MFMA where C and D are not the same range or where A/B
overlap D. All MFMA probes still pass with this form: `PROBE_MFMA_SCALE_OK`,
`PROBE_MFMA_FP8_OK`, `PROBE_MFMA_WIDE_OK`, `PROBE_LOOP_MFMA_OK` (128.0),
`PROBE_FMHA_DATAPATH_OK`.

## Resolved: MFMA result reads need independent instructions

CDNA4 ISA section 7.6 ("Dependency Resolution: Required Independent
Instructions") states that an MFMA result written to a VGPR must not be read by a
VALU/VM/LDS instruction until 5/8/12/20 independent instructions have been
issued, depending on the opcode (there is no forwarding path). Chained in-place
MFMA accumulation of the same opcode needs 0. `hazards.rs` now models this: every
MFMA result gets a ready cycle and consumers are padded with `s_nop`.

That was the cause of the zeros from `probe_fmha_core` (the accumulator copy
right after the MFMA read a partially written VGPR). The same padding also makes
the other MFMA probes deterministic instead of timing-dependent.

Verified after the fix:

| probe | result |
| --- | --- |
| `probe_fmha_core` | A/B phase staging, three paired barriers per iteration, in-place MFMA accumulation over two K=64 iterations -> every value `128.0` -> `PROBE_FMHA_CORE_OK` |
| `probe_loop_mfma` | `PROBE_LOOP_MFMA_OK` (`128.0`) |
| `probe_mfma_scale` / `probe_mfma_fp8` / `probe_mfma_wide` | all still OK |
| `attention_fwd` | `bad=0/16384`, `max_abs_err=5.996e-05` |

Follow-ups: the wait counts are the conservative end of the ISA table (12 for the
16x16/32x32 F16/FP8 shapes, 20 for F8F6F4) and the pass is per basic block, so
cross-block producers are not covered yet. Both should be tightened/validated
once the FMHA core loop is wired to real data.

## Pack instructions and MFMA alignment checks (round 12)

* `v_cvt_pknorm_u16_f32` is wired through DSL/MIR/emit and verified on hardware:
  `(1.0, 0.5)` produces `0x8000FFFF` (low half = S0 as unorm16, high = S1).
* `v_cvt_scalef32_pk_fp8_bf16` is wired with the low-half form
  (`v_cvt_scalef32_pk_fp8_bf16 vD, vS0, vS1`), verified to replace only bits
  `[15:0]` of the destination and keep the rest (`0xAAAA3838`). Its OPSEL[3]
  high-half encoding could not be pinned down (`op_sel:[0,0,0,1]` and
  `op_sel:[0,0,1]` both produce wrong values), so the DSL builds the high half
  explicitly with `v_lshlrev_b32` + `v_and_b32` + `v_or_b32`; verified to give
  `0x3838AAAA`. The native high-half form should be found later to save the VALU
  ops.
* MFMA operand alignment is now enforced after allocation: each operand base must
  be aligned to its width (and even), as the CDNA4 ISA requires.

## Real-data block-scaled fp8 matmul is numerically verified

`probe_fp8_matmul` feeds a genuine 32x64 fp8 A and 64x32 fp8 B (stored
transposed) from global memory through `v_mfma_scale_f32_32x32x64_f8f6f4` and
compares against an analytic CPU reference:

* test 1 (identity): `A[i][k] = (k == i)`, `B[k][j] = (j == k)` -> `D` must be the
  32x32 identity,
* test 2 (scales): all-ones A and B with per-lane E8M0 scales
  `127 + (row & 1)` -> `D[i][j] = 64 * 2^((i & 1) + (j & 1))`.

Both pass on hardware (`PROBE_FP8_MATMUL_OK` for `FP8_TEST=1` and `FP8_TEST=2`),
using the fragment mapping from CDNA4 ISA 7.1.3 on both the kernel and the host
side: `A[i][k]` in lane `i + 32 * (k / 32)` at byte `k % 32`, `B[k][j]` in lane
`j + 32 * (k / 32)`, lane `l`'s scale byte covering row `l % 32` and K block
`l / 32`, and `D[i][j]` in lane `j + 32 * ((i / 4) % 2)` at register
`4 * (i / 8) + (i % 4)`.

### Root cause found by bisection

`probe_fp8_min` (constant `Frag8` operands) passed all 64.0, while
`probe_fp8_minload` (same MFMA with the operands loaded from global memory)
failed, so the MFMA itself and the store side were fine. The emitted assembly
showed no `s_waitcnt vmcnt(0)` between the scale loads and the MFMA: `as_v1` and
`as_v2` were the only operand helpers that did not call `touch`, so a V1/V2 value
produced by a memory load was consumed without waiting (the scale registers read
0, which multiplied the whole result to zero, and in other layouts produced
garbage addresses). Both helpers now touch their operand, which also covers the
V2 A/B operands of the F16 MFMA shapes.

## Scale semantics of the block-scaled fp8 MFMA (round 15 findings)

The unscaled data path is fully verified: `FP8_TEST=4` runs a real random fp8
matmul (mixed E4M3 values incl. negatives, all scales 127) through the
block-scaled MFMA and matches a CPU double-precision reference element by
element, so the E4M3 decode, the A/B fragment mapping and the K pairing are
correct.

The E8M0 scale *lane* mapping is partially pinned down:

* `FP8_TEST=11` (raise the scale of A lanes 0..3 to 2,4,8,16, everything else 1)
  gives `D[i][0] = 96, 160, 288, 544` for rows 0..3 and 64 for rows 4..7, i.e.
  lane `i`'s A scale multiplies **32 of row `i`'s 64 K values** by 2^(l-127).
* `FP8_TEST=9` (A scale 2 on lanes 32..63 only, all-ones data) gives 96 =
  32*1 + 32*2, so lanes 32..63 scale the other 32 K values of their row.
* `FP8_TEST=2`, `5`, `7`, `8` pass, but `FP8_TEST=6` (B restricted to the upper
  32 bytes of each B^T row, A scale varying as 2^(1 + (l % 4)) on lanes 32..63)
  gives 48 = 16*2 + 16*1 instead of 64, and the pattern repeats with period 4 in
  the row index. That was the clue for the real rule: **the scale lane is
  `row + 32 * ((k % 32) / 16)`**, i.e. the 32 bytes of a lane's fragment are
  split into two halves of 16 and each half is scaled by a different lane group
  (the upper half of the bytes uses the +32 scale lane). Encoding that in the
  host reference makes the general random-data test pass.



## MXFP8 score tile plus row softmax (round 16)

`probe_fp8_softmax` runs the whole score stage end to end: it loads a real fp8 Q
tile and a transposed K tile plus their per-lane E8M0 scales from global memory,
runs `v_mfma_scale_f32_32x32x64_f8f6f4`, writes the 32x32 score tile into LDS
using the D fragment layout, then every lane reloads one full row, computes
`exp(x - max) / sum(exp)` with plain VALU ops (a 32-step max chain, 32 `v_exp_f32`
calls, an add chain and one `v_rcp_f32`) and stores the normalised row.

Matched against a CPU double-precision reference over random fp8 data with random
per-lane scales: `worst_err = 8.1e-08`, so both the fragment/scale layout and the
transcendental/reciprocal path are numerically sound.

## Single-tile MXFP8 attention (round 17)

`probe_fp8_fmha_tile` is the first complete attention tile produced by the
compiler: QK^T with the block-scaled fp8 MFMA, a row-wise softmax through LDS,
**in-kernel fp8 quantisation** of the probabilities (`v_cvt_pk_fp8_f32`, two
values per instruction, combined into dwords with a shift and add), and a second
block-scaled MFMA for PV. The `p * 16` quantisation factor is folded into the PV
B-scale (`123 = 2^-4`), so no extra epilogue is needed.

Verified on random fp8 Q/K/V with random per-lane E8M0 scales: the kernel also
writes the quantised P bytes, the host rebuilds `O = (P_q @ V) / 16` from exactly
those bytes and compares element by element -> `worst_err = 0.0`,
`PROBE_FMHA_TILE_OK`.

New compiler pieces: the `CvtPkFp8F32` op (DSL/MIR/lowering/emit) with its
verified low-half write semantics.

## Two-tile MXFP8 attention (round 18)

`probe_fp8_fmha_2tile` adds a real loop: a loop-carried tile index selects the K
tile, its scales and the LDS slot of the score tile. After the loop each lane
computes a 64-wide softmax, quantises its lane group's half of the probabilities
(group 0: tile 0, group 1: tile 1, selected branch-free with xor/mul), and one
block-scaled MFMA contracts all 64 keys against the combined V^T. The
probabilities already carry 1/l, and the 1/16 quantisation factor is folded into
the B scale. Matches the quantised-P reference to 1.8e-4; the deviation from exact
double-precision softmax attention (8.3e-2 on values up to ~2) is the fp8 rounding
of P itself.

Not yet done: online softmax (running max/sum with rescaling of the O accumulator
across tiles), which needs per-row correction factors broadcast through LDS.

## Online-softmax attention (round 19)

`probe_fp8_fmha_online` is the flash-attention core loop on gfx950 through our
compiler, four 32-key tiles with fp8 data and block scales: loop-carried `m` and
`l` (float bits in I32 loop variables), per-iteration `corr = exp(m_old - m_new)`
published to LDS so every lane can rescale the O elements of the rows its
fragment covers, quantised `P * 16` fed to the PV block-scaled MFMA, the O
accumulator held in LDS, and a final `1/l` epilogue. Register use 160 VGPRs.
Matches a double-precision host emulation of the same online algorithm (built from
the kernel's own quantised P bytes) to 1.8e-4; the 5.3e-2 distance from exact
softmax attention is the inherent fp8 rounding of P.

`probe_fp8_fmha_oreg` (round 20) removes the LDS O accumulator: a new `Frag16Build`
DSL op assembles a fragment from sixteen scalars, so the rescale-then-accumulate
step becomes `O = MFMA(P, V, C = O * corr)` with O living in registers (176 VGPRs).

`probe_fp8_fmha_multiwave` (round 21) runs the same loop in a 512-thread block, 8 waves
with 8 query tiles (256 query rows, like the reference grid), per-wave LDS regions
(67,584 B) and shared K/V; 176 VGPRs.

`probe_fp8_fmha_pingpong` (round 22) adds the role split: K/V are now staged through a
double-buffered LDS region by a producer/consumer split (role A = waves 0-3 copy K, role B
= waves 4-7 copy V) and consumed with `ds_read_b128`; the barrier in each role branch is a
distinct instruction that lands on the same collective phase.

`probe_fp8_fmha_tr` (round 25) uses `ds_read_b64_tr_b8` for V inside the attention loop with
64 keys per iteration, so V no longer needs a host-side transpose and the K=64 B fragment is
fully populated (176 VGPRs).

`probe_fp8_fmha_trab` (round 26) combines the tr-based 64-key loop with the A/B role-split
loops in a 512-thread block (192 VGPRs, 70,656 B LDS vs the reference 246 VGPRs / 84,224 B).

Remaining distance to the reference kernel: see the round 28 section.

## Separate A/B loops and two lowering bugs (round 23)

Giving each role its own loop (`probe_fp8_fmha_abloops`) failed at first with ~200 wrong
elements per wave and run-to-run variation, which exposed two real compiler bugs in how
lowering treats control flow:

1. **Pending-wait state leaked across paths.** `pending_vmem/smem/lds` is a linear-lowering
   set; a wait emitted in the then-branch cleared it, so the else-branch consumed the same
   outstanding loads (the Q fragment) without any `s_waitcnt`. Lowering now drains all
   pending loads at every control-flow edge (loop entry, break, backedge, `if`, `else`,
   `endif`).
2. **Constant and conversion caches were not region scoped.** A constant first materialised
   inside role A's loop body was reused by role B's loop as if its `v_mov` had executed there.
   `consts` and `converted` are now snapshotted at loop/if entry and restored at `else`,
   `endif` and loop exit.

After both fixes all 8 waves match (`per-wave bad: 0 0 0 0 0 0 0 0`), and the shared-loop
variants still pass.

## ds_read_b64_tr_b8 (round 24)

The CDNA4 ISA only says the instruction transposes an 8-bit matrix on the way from LDS to
VGPRs, so the lane mapping was measured (`probe_lds_tr_b8`, LDS byte a = a & 0xFF, lane l
supplying address 8*l). For a group of 16 lanes g = l / 16 the result is

    result[lane j][byte i] = byte (j % 8) of the 8-byte chunk addressed by lane 16g + 2i + (j % 16) / 8

i.e. each lane contributes a row chunk and every lane receives one column of eight rows.
From that, the B fragment of the 32x32x64 block-scaled MFMA (lane L = j + 32 * gk, byte
b -> k = 32 * gk + b) is built from a row-major B (pitch P bytes) with four reads r = 0..3
at the per-lane address

    gk * 32P + (8r + i) * P + 8 * (2 * (g % 2) + h)     with i = (lane % 16) / 2, h = lane % 2

(`probe_fp8_matmul_tr`, P = 32). The DSL gained `load_lds_tr_b8` (`Frag2` result) and
`frag16_lane` now extracts a dword from any multi-dword fragment.

GPU-free regression tests for the round 23 lowering fixes were added
(`loads_are_drained_before_a_branch`, `constants_are_materialised_on_every_path`); the backend
suite is now 10 tests.

## Reference tile shapes (round 27)

`probe_fp8_fmha_hd128` runs the reference kernel's tile shapes (head dim 128, KV tile 128) for one
32-row query tile at 240 VGPRs (the reference uses 246). Bringing it up exposed three compiler /
codegen limits that are now documented:

1. **Register pressure from the constant cache.** Hundreds of distinct address constants
   (`4 * k` for every LDS word) were all live at once and exhausted the 256 VGPRs even though the
   real working set is small. Kernels now use running addresses (`addr += 4`); the allocator's
   out-of-registers error also reports how many vregs of each class were live at that point.
2. **13-bit global offsets.** `global_load/store` immediates are signed 13-bit (+-4 KiB); larger
   static offsets must be folded into the pointer.
3. **Transposing-read address for other pitches.** The K-block stride of the `ds_read_b64_tr_b8`
   address is `32 * pitch` (4096 for pitch 128), not a constant 1024: with the wrong stride the
   first query rows were right and the `p = 1` rows were wrong, which pinned it down.

Remaining distance to the reference kernel: combining the hd128 loop with the 512-thread A/B role
split (eight query tiles, K also staged through LDS), the `+0x80` KV progress updates and the
rollback exits.

## hd128 at reference scale and the register-allocator fixes (round 28)

`probe_fp8_fmha_hd128b` (single wave, K and V staged in LDS, two 64-key substeps per 128-key tile)
and `probe_fp8_fmha_hd128c` (the 512-thread, A/B role-split version) are verified on hardware. Getting
the 512-thread kernel to fit needed several allocator / lowering improvements, found by listing the
live wide vregs when allocation failed:

1. **Live ranges stretched to position 0 (real bug).** Blocks with no instructions bumped every value
   live across them at index 0, so any loop-carried accumulator that crossed an empty block (the
   structured `if`/loop skeleton creates many) was treated as live from the start of the function.
   Both role loops' four O accumulators (128 dwords) were therefore live simultaneously for the whole
   kernel. Empty blocks now use their true linear position; `hd128c` dropped from "out of VGPRs" to
   224.
2. **Constants are rematerialised per loop** instead of staying live across the whole loop body.
3. **Zero fragments are rematerialised** at use and MFMA destinations on a zero accumulator are
   zero-filled directly; single-use MFMA/`Frag16Build` accumulators are updated in place (no copy and
   no second sixteen-dword live range).
4. **Fallback allocation policy.** If compact allocation runs out of registers, retry with narrow
   operands packed from the top of the file so the low registers stay as dense sixteen-dword blocks.
5. The allocator failure message now lists live wide vregs and prints the register map.

Remaining distance to the reference kernel: the `+0x80` KV progress updates and the rollback exits,
plus replacing our row-per-lane LDS softmax with the reference's in-register `v_max3_f32` /
`v_cmp_nge_f32` rollback structure.

## The vendor reference kernel as ground truth (round 29)

`tests/gpu/ref_host.cpp` launches the vendor `fwd_hd128_mxfp8.co` (aiter commit 7061cbd, SHA256
83331484...4243438) on random data through its 656-byte kernarg ABI (layout taken from
`asm_mha_v4_fwd.cu`) and compares it with a double-precision CPU model. The binary is **not** stored
in the repository; copy it to the node (`scp`) and pass its path.

What the experiments established (all on hardware, `REF_KERNEL_OK`, 32768 outputs, 0 bad):

* ABI and layouts are as in the launcher: Q/K/V are BSHD e4m3 with a contiguous head dim of 128,
  Q/K descales are uint8 E8M0 `[B, S, H, 4]` (one byte per 1x32 block of the head dim, with zeroed slack
  rows behind them), V descale is a scalar float, output is BF16 `[B, S, H, 128]`; grid
  `(ceil(S/256), H, B)`, block 512.
* A zero Q gives exactly the mean of V, so the V path, output strides and key length are right.
* **The kernel computes `P = softmax_2(S)`, i.e. `exp2(S)` with `S = sum Qdeq * Kdeq`, and ignores the
  scalar passed at 0x50** (varying it, even to 0, changes nothing). A CPU model with an effective scale of
  `ln 2` matches the kernel best (error minimum at 0.693 in a sweep from 0.30 to 0.80); the caller is
  expected to fold `softmax_scale * log2(e)` into the Q/K data or descales.

Scale recipe check (`probe_fp8_matmul_blk`, exact): the scaled MFMA's per-lane scale byte applies to
the data byte `b` of either lane group through lane `row + 32 * (b / 16)`, so a 1x32 block of K
occupies bytes 0..15 (block 0) or 16..31 (block 1) of both lane groups. The fragment for lane group `g`
is therefore `K 16g..16g+15` followed by `K 32+16g..32+16g+15`, and lane group `g` supplies the scale
of block `g` as `descale_dword >> (8 g)` (op_sel 0). With that arrangement real 1x32 E8M0 scales are
reproduced exactly (`PROBE_MATMUL_BLK_OK`).

Next: an ABI-compatible kernel built on these layouts, compared element by element with the vendor
output on identical inputs.

## ABI-compatible kernel validated against the vendor kernel (round 30)

`fmha_fwd_hd128_mxfp8` (in `kernels.rs`) takes exactly the vendor kernel's 656-byte kernarg and
layouts and is run next to the vendor `fwd_hd128_mxfp8.co` on identical random inputs by
`tests/gpu/run_ref.sh` (`REF_CO=<vendor .co> bash tests/gpu/run_ref.sh`). New DSL pieces: `exp2_f32`
(the hardware `v_exp_f32`, no log2e scaling), `cvt_pk_bf16_f32` (`v_cvt_pk_bf16_f32`), and the
kernarg / workgroup-id plumbing for runtime strides. The scale recipe from round 29 is used as is:
Q/K fragments in the permuted K order and `descale_dword >> (16 c + 8 g)` as the per-lane scale.

Results on gfx950 (random fp8 data, random 126..128 E8M0 descales, per-tensor V descale 1.0; `REF_ROWS`
rows per head checked against the CPU model, all outputs compared against the vendor kernel):

| shape B,H,S,SK | ours vs CPU (mean / worst) | vendor vs CPU (mean / worst) | ours vs vendor (mean / worst, all outputs) |
|---|---|---|---|
| 1,1,256,256 | 1.3e-3 / 1.8e-2 | 2.2e-3 / 2.4e-2 | 2.5e-3 / 6.6e-2 |
| 1,2,512,512 | 1.1e-3 / 2.6e-2 | 1.8e-3 / 5.4e-2 | 2.8e-3 / 7.5e-2 |
| 2,4,1024,1024 | 1.7e-3 / 3.9e-2 | 2.4e-3 / 3.7e-2 | 2.7e-3 / 8.1e-2 |
| 1,8,2048,2048 | 1.7e-3 / 3.9e-2 | 2.8e-3 / 4.3e-2 | 2.8e-3 / 9.8e-2 |
| 1,2,512,2048 | 1.3e-3 / 3.7e-2 | 2.4e-3 / 3.7e-2 | 2.8e-3 / 7.5e-2 |
| 3,3,768,1280 | 1.2e-3 / 4.0e-2 | 2.3e-3 / 4.1e-2 | 2.7e-3 / 8.2e-2 |

With Q = 0 the two kernels' outputs are bit-identical on all 131072 outputs. The remaining
differences are the fp8 rounding of P (we quantise `P * 16`, the vendor kernel its own way), which is
why both kernels sit about 1-3e-3 from the exact model and ~3e-3 from each other.

Performance of the first version, honestly (`REF_ITERS=20`; see the round 31 section for the transposed kernel): the vendor kernel runs the 2048x2048, 8-head problem in 0.033 ms
(about 518 TFLOP/s); ours takes 1.14 ms (15 TFLOP/s), i.e. **~34x slower**. Our loop is not software
pipelined (K/V staging, S stores, softmax and PV are serialised by barriers), the softmax goes through
LDS one row per lane, and there is no overlap between the roles. Closing that gap is the remaining
work on the way to a 1:1 core loop.

Limits of this version: `S` a multiple of 256, `kv_seq_len` a multiple of 128, GQA ratio 1, no LSE.

## Register-resident softmax (round 31)

`fmha_fwd_hd128_mxfp8_t` computes the transposed product `S^T = K Q^T` (K fragment as the MFMA A
operand, Q as B), whose output layout gives every lane one query (`j = lane % 32`) and sixteen keys of
each 32-key tile. That removes almost all of the LDS traffic of the first version:

* **Softmax in registers.** The row max is a lane-local `max` chain over 32 values plus one exchange with
  the partner lane `l ^ 32` (a single LDS store and load of one dword, no barrier), the row sum stays a
  lane-local partial (the two partners are added once in the epilogue), and the online-softmax correction
  `exp2(m_old - m_new)` is a lane-local scalar. No score tile, no correction slot and no per-wave LDS
  region is needed any more.
* **`P^T` is the B operand of the next product.** `O^T = V^T P^T` takes the quantised `P^T` straight from
  the score registers (byte `b` of lane group `g` is key `16 r + 8 (i / 4) + 4 g + i % 4`) and the O
  accumulators come out with one query per lane, so the correction multiplies the accumulator registers
  directly and the epilogue stores four consecutive dims per 8 bytes.
* **The V fragment** is the same `ds_read_b64_tr_b8` read as before with a new lane address: for lane `M`,
  `key = 8 ((M & 15) >> 3) + 4 (M >> 5) + ((M >> 1) & 3)`, column `16 ((M >> 4) & 1) + 8 (M & 1)`, plus
  `2048 r + 32 e + 8192 hh` per instruction. (Derived from the measured transposing-read mapping and
  checked on hardware.)

Measured (`REF_ITERS=20`, 1x8x2048x2048): 0.194 ms = 88 TFLOP/s against 1.14 ms (15 TFLOP/s) for the first
version and 0.033 ms (518 TFLOP/s) for the vendor kernel, i.e. the gap closed from 34x to 5.9x. LDS now holds
only the K and V tiles (34.8 KB), so a second tile buffer (software pipelining) fits.

## Sub-register aliases: removing the register-to-register moves (round 32)

Profiling the transposed kernel's assembly showed that 2585 of its 4935 lines were `v_mov_b32`: every
`Frag16Lane` read, `Frag16Build` / `Frag8` write, fragment concatenation and loop-variable `set_var` of a
sixteen-dword accumulator was lowered to one or two moves per dword, which swamped the 16 MFMAs and ~130
`v_exp_f32` of a 128-key iteration. Fixes in the compiler:

* **Sub-register aliases** (`MFunction::alias`): an alias vreg owns no register; after allocation it is
  its tuple's physical register plus a dword offset. `frag16_lane` is now free, `copy_reg` is one move per
  dword instead of two, and liveness / hazards / allocation treat a use or definition of an alias as one of
  the tuple (hazards: an MFMA result tuple is waited on when any of its lanes is read).
* **Producer retargeting.** When a fragment is built from single-use ALU results (`Frag16Build`, `Frag8`),
  the producing instruction now writes the fragment's sub-register directly instead of a fresh register
  that is moved in afterwards.
* **Direct loads and in-place accumulators in the DSL**: `load_lds_b128x2` (two `ds_read_b128` into the halves
  of one `Frag8`), `load_lds_tr_b8x4` (four transposing reads into the quarters of one `Frag8`),
  `var_scale_f32` (16 in-place `v_mul_f32` on a loop-carried accumulator) and `var_mfma_scale` (in-place
  accumulation into a loop variable). The register allocator no longer reserves registers for vregs that are
  never referenced.

Result for `fmha_fwd_hd128_mxfp8_t` (all 6 shapes still `REF_KERNEL_OK`, numerics unchanged): assembly 4935 ->
2859 lines, `v_mov_b32` 2585 -> 633, VGPRs 192 -> 176, **88 -> 118 TFLOP/s** on 1x8x2048x2048 (vendor kernel:
518 TFLOP/s, a 4.4x gap). The other kernels also benefit (`probe_fp8_fmha_hd128c` 224 -> 208 VGPRs) and all
earlier probes still pass.

## Staging, address folding and instruction trimming (round 33)

The loop body of `fmha_fwd_hd128_mxfp8_t` was 945 instructions per 128-key iteration. Its histogram showed the
real culprit: the K/V staging issued sixteen 4-byte `global_load_dword`s per thread, each followed by an
`s_waitcnt vmcnt(0)` and a `ds_write_b32` -- sixteen serialised memory round trips per iteration. Changes:

* **16-byte global loads** (`load_b128`, `global_load_dwordx4`): four loads cover a thread's 64 bytes and all of
  them are in flight together.
* **Software-pipelined staging.** The next tile's four loads are issued at the top of the iteration (the last
  iteration re-reads its own tile so nothing is read out of range), the MFMA / softmax work of the current tile
  hides their latency, and after the end-of-iteration barrier the prefetched registers are written to LDS before
  the second barrier. Still two barriers per iteration.
* **LDS offset folding + DCE.** `x + const` chains in LDS addresses become the DS instruction's `offset:`
  immediate; a small dead-code pass removes the address adds that are no longer read (about 100 `v_add_u32` per
  iteration).
* **Zero-accumulator MFMA** (`D = A x B` with the inline constant 0 as C): the 64 `v_mov_b32` that zero-filled the
  S accumulators are gone.
* **Math trimming.** The `P * 16` quantisation scale is folded into the exp2 argument
  (`exp2(s - (m - 4)) = 16 exp2(s - m)`, so the row sum is corrected by 1/16 once per substep), the 32-value row
  max uses `v_max3_f32`.

Result (`REF_ITERS=20`, 1x8x2048x2048; all 6 shapes still `REF_KERNEL_OK`, numerics unchanged): **190 TFLOP/s**
(0.091 ms) against 118 before and 518 for the vendor kernel -- a 2.7x gap; assembly 2859 -> 2329 lines,
`v_mov_b32` 633 -> 447.

## Where the time goes, GQA and partial query tiles (round 34)

**Performance attribution (ablation).** `FMHA_ABLATE` (a debug switch read when the kernel is built; passed
into the build container by `tools/container/cargo.sh`) removes one component of `fmha_fwd_hd128_mxfp8_t` at a
time. Timing the 1x8x2048x2048 problem (0.090 ms = 190 TFLOP/s as is):

| ablated | time | change |
|---|---|---|
| `nostage` (no K/V prefetch + LDS writes) | 0.0769 ms | -15% |
| `nobar` (no in-loop barriers) | 0.0808 ms | -10% |
| `nopv` (no PV MFMAs) | 0.0815 ms | -10% |
| `noexp` (no `v_exp_f32`) | 0.0851 ms | -6% |
| `nostage,nobar` | 0.0692 ms | -23% |
| all four | 0.0533 ms | -41% |

No single component dominates: staging + barriers cost about a quarter, PV a tenth, the exponentials a
twentieth, and more than half of the time is the rest of the S path (the QK^T MFMAs with their hazard
padding, the max exchange, the `sub`/`cvt` work). Two experiments that did *not* help and were reverted:
issuing all four score tiles before the softmax (192 -> 224 VGPRs, 190 -> 179 TFLOP/s) and lazy rescaling of
the running max / O accumulators behind a wave-level branch (the reference's "rollback" idea; only 3% faster
even on data whose max never grows, and slightly noisier numerics), so the 128 `v_mul_f32` of the O
rescale are not the bottleneck.

**GQA and partial query tiles.** `fmha_fwd_hd128_mxfp8_t` now reads `s_seq_len` and `s_gqa`: the KV head of
query head `h` is `h >> log2(gqa)` (gqa a power of two up to 16), K/V/descale offsets use it, the descale
row strides are taken from the Q/K row strides (`stride >> 5`), and query rows past the end of a partial
tile are clamped to the last valid row, so they compute and store that row's identical result instead of
touching memory out of bounds. `tests/gpu/run_ref.sh` now covers 9 shapes including `S = 300`, `S = 700`
with `gqa = 2` and `gqa = 4`; all `REF_KERNEL_OK` against both the CPU model (mean error 1.1-1.8e-3) and the
vendor kernel (mean difference 2.4-2.8e-3, all outputs compared).

Still required at this point: `kv_seq_len` a multiple of 128 (lifted in round 35), no LSE output.

## Arbitrary key length (round 35)

`fmha_fwd_hd128_mxfp8_t` no longer requires `kv_seq_len` to be a multiple of 128 (the vendor kernel does not
either: it passed SK = 1, 100, 129, 1000, 1088, 1337 against the CPU model). The loop runs
`ceil(kv_len / 128)` iterations; K/V staging rows and K descale rows past the end are clamped to the last
valid key, and the last iteration (detected as `kv_len < 128 (it + 1)`, uniform across the block) takes a
second copy of the compute body in which every score value whose key index (`128 it + 32 s + 8 (r / 4) +
4 g + r % 4`) is `>= kv_len` is replaced by `-1e30` before the softmax, so those keys get probability 0.
Only the last iteration pays for the 64 compare/selects. Cost: 208 VGPRs and ~2x assembly size; throughput on the
main benchmark shape is 186 TFLOP/s (was 190).

`tests/gpu/run_ref.sh` now has 15 shapes (partial query tiles, GQA 2/4, key lengths 1..1337, batch 1..3); all
`REF_KERNEL_OK` for `fmha_fwd_hd128_mxfp8_t` against both the CPU model (mean error 1.1-1.8e-3) and the vendor
kernel on every output (mean difference 1.3-2.8e-3; SK = 1 is bit-identical to the vendor kernel).

Remaining differences from the vendor kernel's feature set: no LSE output.

## Partial `vmcnt` waits and a negative result (round 36)

Global loads complete in issue order, so the lowering now waits for the newest load that feeds an operand
with `s_waitcnt vmcnt(n)`, where `n` is the number of loads issued after it (instead of always `vmcnt(0)`),
and keeps the later loads pending. It also gained `var_load_b128`, a 16-byte global load written directly
into a `Frag4` loop variable (the loop back edge still drains everything). Neither changes the existing
kernels' output.

Hypothesis tested and rejected: that the K scale dword loads issued *after* the prefetch force a full wait for
the prefetch. Hoisting the four K scale loads in front of the prefetch (so their `vmcnt(4)` leaves the prefetch
in flight) made the kernel slower (224 VGPRs, 190 -> 167 TFLOP/s) and was reverted; the 15-shape comparison
stayed green throughout. Together with the ablation study of round 34 this says the loop is bound by the
dependency structure of the S path and by instruction issue rather than by memory latency, so the next
real step would be restructuring the schedule across the two waves of a SIMD (the reference kernel's
ping-pong), which is beyond what the unscheduled linear lowering can express.

## Final summary

The consolidated summary, the reference-kernel comparison table, the measured compile latency and the list of what
is not implemented are in [`final-report.md`](final-report.md).

1. Resolve the 16 KB LDS question above (needed before scaling the core loop).
2. Build the paired A/B loop skeleton: two loops with rollback exits, four
   paired barriers per outer iteration, exactly as in the reference kernel.
3. Wire the data path into the loop: LDS staging -> `ds_read_b64/b128` -> MFMA
   accumulation (accumulator lives in the fragment registers).
4. Add the softmax pack instructions
   (`v_cvt_pknorm_u16_f32`, `v_cvt_scalef32_pk_fp8_bf16`) with their
   low/high write-mask semantics.
5. Compare the emitted instruction stream against the reference witness
   section by section.

## Kernel optimisation (round 41+)

All numbers: B=1 H=8 S=SK=2048, GPU 0, 30 iterations, `tools/ab.sh` (build-time variants through env vars FMHA_PITCH / FMHA_OPT, forwarded by `tools/container/cargo.sh`). Vendor kernel in the same runs: 518 TFLOP/s.

| Experiment | Result |
|---|---|
| Baseline (pitch 128) | 186.4 TFLOP/s |
| LDS row pitch 144 (K/V tiles padded; bank conflicts) | **207.7-208.1** (+11%), kept |
| Pitch 160 / 176 / 208 / 272 | 206-208 (same plateau); 192: 197 |
| Pitch 132 / 136 / 152 (not 16-byte aligned) | wrong results (132) or 142 TFLOP/s: ds_read_b128 needs 16 B alignment |
| cvt4: P word = two `v_cvt_pk_fp8_f32` (second with op_sel:[0,0,1] keeping the low half) instead of mov+cvt+mov+cvt+shl+add (-64 instr/iter) | 207.5 -> 212.5-213.3 (+2.7%), numerics identical, kept |
| pksub: S - m via `v_pk_add_f32` on register pairs of the MFMA tuple, broadcast of -(m-4) in a duplicated pair (-32 instr/iter) | 207.5 -> 210-211 (+1.5%), kept |
| pkscale: O rescale with `v_pk_mul_f32` (-64 instr/iter) | 206.6 (no gain, -0.4%); NOT default (FMHA_OPT=pkscale) |
| cvt4+pksub (new default) | 213.1-213.3; all 15 run_ref shapes REF_KERNEL_OK, mean err unchanged (1.07e-3 vs CPU) |

New DSL ops: Frag16Pair, Frag2Dup, PkAddF32, PkMulF32, CvtPkFp8x4F32, Stmt::VarScalePk (lowering in lower.rs; MOps VPkAddF32, VPkMulF32, VCvtPkFp8F32Hi; hazards.rs only got the three MOps in the `movable` list; emit.rs three additive match arms).
Note: hazards track readiness per parent tuple, so two v_exp writing two halves of one pair would see a false WAW stall; exp results therefore stay in separate V1 registers.

Reading: removing 160 VALU instructions (~25%) gave only ~3-5%, so the kernel is not simply VALU-issue bound; barrier/dependency structure matters more.

### Vendor kernel schedule (llvm-objdump of ref_mxfp8.co)
- Two code paths per role (label_block_A_loop / label_block_B_loop), each loop covers two 128-key sub-steps with `s_setprio 1 ... s_barrier ... s_setprio 0 ... s_barrier` around the MFMA phases: wave ping-pong with priorities (idea 3).
- Softmax: v_max3 chains (34 per 32-key pair), `v_cvt_pknorm_u16_f32` + `v_cvt_scalef32_pk_fp8_bf16` (lo, then op_sel:[0,0,1] hi) to produce P, 256 v_pk_fma_f32 + 472 v_pk_mul_f32 + 526 v_exp (whole file), 46 v_permlane32_swap_b32 for the lane^32 exchange (idea 4), a rare overflow rollback branch (`v_cmp_nge_f32 vcc, s92, max` -> rescale only when the max grows by more than a threshold = lazy rescaling).
- PV MFMAs are unscaled `v_mfma_f32_32x32x64_f8f6f4` (no scale operands), and row sums come from `v_smfmac_f32_16x16x128_fp8_fp8` instead of VALU adds.
- V fragments via ds_read_b64_tr_b8 with `s_waitcnt lgkmcnt(4)` pipelining between MFMAs; K via ds_read_b128; LDS ring offsets wrap with s_cselect (multi-buffer ring, LDS 84 KB).
- Global loads via buffer_load_dwordx4 (8 per 2 steps).
Next candidates: unscaled PV MFMA + MFMA row sum, lazy rescale with threshold, permlane32_swap, setprio ping-pong, LDS ring double buffering.

### Negative / neutral experiments (all measured on top of cvt4+pksub = 213 TFLOP/s, run-to-run noise ~0.5)
| Experiment (FMHA_OPT flag) | Result |
|---|---|
| permlane: lane^32 max exchange with `v_permlane32_swap_b32` (2 movs + swap + max, new op PartnerMaxF32) instead of LDS store/wait/load/wait | correct, 212.3-213.4: no gain |
| dbuf: double-buffered K/V tiles in LDS (66 KB + exchange), one barrier per iteration | correct, 212.4-213.0: no gain |
| kdconst (ablation, wrong scales but data scales are all 127): K scale loads replaced by a constant | 212.5-213.0: the per-tile K-scale global loads are not on the critical path |
| prioA / prioB: `s_setprio 1` for role A or B waves for the whole loop (new Stmt::SetPrio) | 212.7-213.7: no gain |

Interpretation: five independent changes (barriers, exchange, scale loads, priorities, 25% fewer VALU) each move the time by at most 3%, while the 11 us launch/timing floor is included in every measurement (S=77,SK=1 takes 11.4 us; S=SK=256 20 us). Per iteration (128 keys, 8 waves) the kernel needs ~4.5 us (~11k cycles at 2.4 GHz) against ~1.5 us for the vendor kernel, although both execute the same 16 MFMAs per wave and iteration. The vendor loop (see above) differs structurally: unscaled PV MFMAs, MFMA row sums (smfmac), pknorm-based P conversion, lazy-rescale branch, 2x unrolled ping-pong with setprio/barrier phases. Untested hypotheses for the remaining gap: (1) the in-order MFMA->VALU dependency chain inside one wave (QK -> max -> exp -> PV never overlaps within a wave; the vendor issues the next tile's QK MFMAs before the current softmax), which needs software pipelining across iterations with 2 x 16 S registers; (2) the hazard pass pads MFMA/trans latencies with s_nop (grep s_nop in the emitted .s). Next step: count s_nop cycles in the loop and prototype the cross-iteration QK software pipeline.

### Round 42: VALU-count ablation and VALU reductions
Base cvt4+pksub, 215.5 TFLOP/s (0.0797 ms, incl. ~11 us launch floor). Saved cycles per iteration (1 us = 2400 cyc), measured vs predicted (2 waves/SIMD x ops x 4 cyc, exp/cvt 8 cyc):

| Removed (timing-only ablation) | measured | predicted | TFLOP/s |
|---|---|---|---|
| O rescale (128 v_mul) | 720 | 1024 | 229.6 |
| S-m (64 v_sub) | 480 | 512 | 221.8 (vs cvt4 base 212.5) |
| row-sum adds (62) | 390 | 496 | 222.7 |
| fp8 cvt (32) | 465 | 512 | 224.2 |
| max3 chain (~16) | 315 | 128 | 221.3 |
| v_exp (66) | 885 | 1056 | 232.9 |
| sub+sum+rescale+max (~270 instr) | 1700 | 2160 | 251 |
| all of the above + exp + cvt | 3030 | 3730 | 289 |

Savings are 80-100% of the VALU-slot prediction, so VALU issue slots are a major limiter. But the floor with all those removed is still ~7.3k cycles/iteration (MFMA 16 x 64 cyc x 2 waves = 2k, LDS, address/mov work, barriers, latency chains), and after the numerically valid reductions below the executed path has ~410 instructions (~3.6k VALU cycles per SIMD, the vendor's total) yet still takes ~9.8k cycles: the VALU count alone does not explain the vendor gap.

Numerically valid reductions (FMHA_OPT flags, now default together with cvt4,pksub):
| flag | change | TFLOP/s | mean err vs CPU |
|---|---|---|---|
| (base cvt4,pksub) | | 215.5 | 1.07e-3 |
| msum | row sums via MFMA with an all-ones E4M3 A operand and P^T as B (every lane gets its query's sum of the quantised P; 16 extra accumulator regs, VGPR 240), no v_add row sum, no l exchange | 218.5-219.3 | 1.08e-3 |
| lazy | rescale branch (any lane's tile max > running max + 4 -> new max, rescale O and l; P x 16 stays < 256 < 448) | 221.5 | 1.41e-3 |
| lazy+msum | | 222.8 | 1.25e-3 |
| lazy+msum+permlane | | 224.3-225 (default) | 1.25e-3 |
| lazyinf (ablation, tau=1e9, wrong numerics) | the branch never taken after tile 0 | 240 | - |
| over2/4/6/8 (new max = tile max + k on trigger) | fewer triggers in theory | 225 / 226 / 227 / 229 | 1.25e-3 / 1.41e-3 / 2.1e-3 / 4.7e-3: rejected (accuracy loss, gain <= 1%) |

On the benchmark's random data the lazy branch fires in most iterations (some lane of the wave exceeds the threshold), so lazy gets about half of the ablation upper bound (240). Final default: cvt4,pksub,lazy,msum,permlane = 224-225 TFLOP/s; 15/15 run_ref shapes REF_KERNEL_OK (mean diff vs vendor 2.3-2.6e-3, within the earlier 1.3-2.8e-3 pattern), cargo test and run.sh OK.
Not done: unscaled PV MFMA (no VALU effect), cross-iteration QK/softmax software pipelining (needs VGPRs beyond 240 with msum), wave ping-pong skew.

### Round 43: nondeterminism with lazy rescale on partial query tiles (fixed)
Reproduced with tests/gpu/determinism.sh: B=2 H=8 S=700 SK=640 gqa=2 gave three different output md5 in three runs (7b73481b / 9e558145 / acbaf99b); the other shapes (S=200, S=300 x2, 2048x2048) were bit-identical. Cause as diagnosed by sched: rows >= S are clamped to row S-1 and store to the same address; the lazy-rescale branch is a wave-level decision, so a wave holding only duplicate rows rounds differently from the wave holding the real row (last writer wins). Fix: the epilogue stores are guarded by the wave-uniform condition (first row of the wave < S); such waves still run the loop (barriers) but do not store. Duplicates inside the partially valid wave compute identically to row S-1 (same decisions), so their same-address stores agree. After the fix all five determinism shapes give identical md5 in 3 runs (S=700: cbc75e0a x3), 15/15 run_ref shapes REF_KERNEL_OK, cargo test and run.sh OK, no performance change (224.7 TFLOP/s).


## Round 44 (kernel-opt): vendor-gap work on `fmha_fwd_hd128_mxfp8_t` (B1 H8 S=SK=2048, vendor 517.5 TFLOP/s)

**Result: plain accurate kernel 227.5 -> 257.8 TFLOP/s (+13%, numerics bit-identical, mean err 1.248e-3); opt-in `fmha_fwd_hd128_mxfp8_sv` 285.5 TFLOP/s (mean err 2.186e-3 vs vendor 2.418e-3).** Gap to vendor: 2.0x (plain) / 1.81x (sv).

| variant (same run, 30 iters) | TFLOP/s | notes |
|---|---|---|
| vendor | 517.5 | |
| `_t` before this round (cvt4,pksub,lazy,msum,permlane) | 227.5 | |
| `_t` + `lacc1` | 229.3 -> 260.8 with kdlds,dbuf | opt-in, +1.2% only |
| `_t` default now (+kdlds,dbuf) | 257.8 | 15 shapes OK on both backends, determinism 5/5, cargo test green |
| `_sv` default (cvt4,svpk,kdlds,dbuf) | 285.5 | 11/15 shapes REF_KERNEL_OK (a few outlier elements, `bad` 1-7 per shape, exceed the tolerance; worst_err 7e-2 vs vendor 5.9e-2): stays opt-in |
| `_pp` phase-shifted ping-pong | 205 | negative (VGPR 256, 4 barriers/tile); kept only as an experiment |

**Root cause of the old "staging + barrier = 3.2k cycles/iter": the per-tile K-scale dword was loaded with `global_load_dword` right before its use, and the lowering/LLVM emitted `s_waitcnt vmcnt(0)` right after it (4x per 128-key iteration).** Loads complete in order, so every wait also waited for the 4 tile-prefetch loads: a full L2 round trip, four times per iteration. Fix (`FMHA_OPT=kdlds`, now default): role A stages the K-scale dword of each key row into LDS together with its K tile; all waves read it with `ds_read_b32`. +11.6% on the plain kernel, numerics unchanged. `dbuf` (two LDS buffers, one barrier per iteration, now default) +2%.

Other measured experiments (all on the benchmark shape):
* `kdhoist` (hoist kd loads) 211 (worse), `lazypk` 222 (worse), `kdpf` (kd software-prefetch through a loop var) 203 (VGPR 256), `stmid` (stage stores mid-iteration) +1.3% on sv, `permlane` neutral, `pp` 205.
* sv: per-64-key max3 + rollback branch + pknorm/scalef32 Schraudolph exp2 + row sums from the same quantised P. Rollback fires in ~70% of the substeps on this data (vendor ~86% of its 128-key checks, but its rollback block is cheaper). `svpk` (packed O rescale in the rollback) +2.7%. `svinf` (never roll back; wrong numerics, timing only) shows the rollback is worth ~13%.
* counters (rocprofv3, before kdlds): SQ_INSTS_VALU vendor 2.06M / sv 3.60M / plain 3.75M; CVT 569k vs 577k (sv equal to vendor), TRANS 536k plain -> 12k sv (vendor 8k), MUL_F32 959k plain / 859k sv / 265k vendor.
* ablations on sv+dbuf+kdlds (286): no staging 393, no barrier 335, both 443, nopv+svinf 401.

**Lowering change** (`lower.rs`): outstanding loads stay pending across `if/else` (snapshot per branch; drained at the join only when a branch changed the set). Neutral on speed, needed so the masked/unmasked diamonds do not drain the prefetch. GPU-free regression tests in `tests/backend.rs`: `loads_are_drained_on_every_path_of_a_branch` (replaces the old "drain before the branch" assertion: the invariant is that no path reads a load before its wait) and `untouched_loads_are_not_over_drained_across_a_branch`.

### Global -> LDS DMA staging (`FMHA_OPT=dma`, own backend only, NOT usable yet: multi-head shapes fail)
Implemented: DSL `load_global_lds` / `wait_vm`, MOps `GlobalLoadLdsB128/B32` (+emit, effects, model), LDS layout of 1040-byte runs like the vendor kernel (0x410), probes `probe_dma`, `probe_dma_tile`, `probe_dma_seq`, `probe_dma_kread`, `probe_dma_map` with `tests/gpu/dma_*_probe.sh`. Hardware findings (gfx950):
1. `global_load_lds_dwordx4`: lane i lands at `M0 + inst_offset + 16 i`, in lane order for ANY lane address pattern (reversed, strided, rows 8 KB apart): an XOR swizzle of the source chunks is possible but my first swizzle attempt was invalidated by a probe bug (empty `PD_*` env counted as set); the 1040-byte-run layout (rows {u + 8 s} per run) is conflict-free for the K fragment reads and was verified with `probe_dma_kread` (mode 5) for row strides 128..1024 and head offsets.
2. **Changing M0 between consecutive DMAs of one wave is unreliable** (the q-th DMA of 4 waves landed at the wrong LDS address, deterministic; even with `vmcnt(0)`+barrier between DMAs). Working scheme: ONE M0 per wave per group and the run offset through the signed instruction offset (`offset:-1040 k`, which applies to the global address AND the LDS address; the source address is compensated). `probe_dma_tile` with `PD_NEW=1` passes repeatedly.
3. The dword form (`global_load_lds_dword`) for the K scales needs its own M0 (same problem): the K scales are staged with a normal load + `ds_write` (`dmakd` = experimental dword DMA, flaky).
4. LLVM tail: `llc` aborts (SIGABRT) on the DMA MIR, so the DMA MOps have no LLVM table rows and the DMA probes are only reachable by name (not in the enumerated lists / tail-invariance); `BACKEND=own` only.
5. Status: with DMA the 1-head shapes (S up to 2048, 16 iterations) are REF_KERNEL_OK and the apparent speed is ~297-376 TFLOP/s; shapes with H >= 2 give a wrong second head (every dim of head 1; head 0 right; the old `ds_write` staging into the same layout is right). Not root-caused: the probe with the same scheme, row strides and head offsets passes. Next step would be a dump of the LDS K tile of a head-1 block. Do not enable by default.

### Round 44 addendum (kernel-opt, second session): verified state and re-profile

* **NaN blind spot fixed in `tests/gpu/ref_host.cpp`**: `err > tol` was false for NaN, so a kernel producing NaN printed `mean_err=nan` and `REF_KERNEL_OK`. NaN now counts as bad (`worst_err=inf`). Re-verified after the fix: plain `_t` 15/15 on llvm-tail and own, `_sv` (svcorr) 15/15, determinism 5/5, `cargo test` green.
* **Default plain kernel now 265.6 TFLOP/s** (vendor 519; was 257.8 at 37ecf7c): added `lacc1` (+1.2%) and `psh2` (P x 4 instead of x 16, rescale threshold 6 instead of 4: rarer lazy rescales, mean err 1.254e-3 vs 1.248e-3). `psh0` (P x 1, threshold 8) same speed, mean err 1.41e-3. Larger thresholds with the old P x 16 scale overflow E4M3 (NaN): `tau6/tau8` removed.
* **sv numerics (honest table)**: `svexact` (exact exp2 for P) 15/15 => the outliers come from the Schraudolph trick only. `svcorr` (v_perm LUT + v_pk_sub_u16 per bf16 pair) 15/15 at 254 TFLOP/s (mean err 1.75e-3); without it 285 TFLOP/s, 13/15, 2.19e-3. So sv with the same tolerance as the vendor is not faster than the plain accurate kernel.
* **rocprofv3, plain default vs vendor, same process, B1 H8 S2048** (per dispatch): SQ_INSTS_VALU 3.90M vs 2.06M; ACTIVE_INST_VALU 4.87M vs 2.41M; BUSY 4.79M vs 2.40M; MFMA 147k vs 155k; LDS 465k vs 430k; VMEM 65k vs 46k; ADD_F32 306k vs 448k; CVT 279k vs 568k; FMA 0 vs 33k; **MUL_F32 959k vs 265k**; **TRANS 536k vs 8.5k**; **INT32 563k vs 89k**; "other" (mov/cndmask/cmp/pk/max) ~1.26M vs ~0.64M. Top VALU classes to attack with accuracy-preserving changes: (1) TRANS (v_exp_f32 is quarter rate; 64 per 128 keys vs vendor's Schraudolph; a cheaper exact-enough exp2 would be a degree-3 polynomial but costs about the same issue slots), (2) MUL_F32 = the lazy rescale of O (64 muls per taken rescale; lazyinf = never rescale gives 285 TFLOP/s so rescales + their check cost ~7-10%; pk variant `lazypk` does not help), (3) INT32 address/mask arithmetic (6x the vendor) - mostly per-substep K scale / key-index / clamp arithmetic and loop-invariant constants re-materialised per branch copy.
* **DMA status (`FMHA_OPT=dma`)**: root cause of H>=2 failure NOT found. Excluded: M0 per wave (independent of the head), the signed instruction offset + compensated address (probe `probe_dma_kread` mode 5 passes for row strides 128..1024 and head offsets, 4 waves, 4 runs per wave), LDS pad / two blocks per CU, the old ds_write staging into the same layout (correct for all H), the K-scale path (old loader in the same layout correct). Only K-tile DMA for head >= 1 differs. A debug dump (`FMHA_OPT=...,dbgk`) crashes the kernel (illegal memory access, even with the stores removed), so no LDS dump yet. Kept opt-in, own backend only, documented as known-broken for H >= 2.

### Round 44 addendum 2 (kernel-opt): scheduler default, dead-code, re-ablation

* **Default plain kernel: 286.1 TFLOP/s** (vendor 519, gap 1.81x), mean err 1.254e-3 (vendor 2.42e-3). The gain over 265.6 comes from the scheduler's wait-recompute mode (`AMDC_SCHED_OPTS=rewait,xbar`, i.e. `reorder=false`, ALU across barriers): the lowering's waits are conservative and the pass re-derives partial `vmcnt`/`lgkmcnt` waits. It is now the DEFAULT for `fmha_fwd_hd128_mxfp8_t` and `_sv` (`compiler/src/lib.rs::lower_and_allocate`; `AMDC_SCHED=0` disables, explicit `AMDC_SCHED*` still wins). Bit-identical numerics. Verified: `_t` 15/15 on llvm-tail and own, `_sv` (svcorr) 15/15 on both (265.7 TFLOP/s), determinism 5/5, `cargo test` green (tail_invariance also with `AMDC_SCHED=1`). Other options measured on the plain kernel: `rewait` 285.9, `xbar` alone 268.4, `swp` 271.2, `taildup,swp` 273.4, plain `--sched` 271.9, `--sched=auto` 261.5 (no change).
* **Dead code**: with `kdlds` the plain kernel still built the K-scale global address (10 VALU per 32-key tile, results unused). Removing it alone is SLOWER without the scheduler (261.2 vs 265.6; the dead VALU filled MFMA-to-consumer latency), neutral/positive with it. Lesson: scheduling matters more than raw VALU count here.
* **Re-ablation of the final plain kernel** (base 0.0599 ms = 286.8 TFLOP/s = 7.3k cycles/iter): lazy rescale + check (`lazyinf`) 307.9 (-0.6k cycles), `nobar` 324.9 (-1.05k), `nostage` 331.6 (-1.2k), `nopv` 333.0 (-1.25k), `noexp` 317.7 (-0.87k), `nostage`+`nobar` 376.8 (-2.1k). The remaining gap to the vendor (1.81x) is therefore spread over: staging + barrier (about 2.1k cycles, the DMA path would remove most of it, parked), the exp/VALU work (0.9k + rescale 0.6k), and the PV/LDS part.
* **INT32 address-arithmetic evidence** (input to the compiler task): (1) dead K-scale address ~40 VALU/iter (removed), (2) clamp idiom through a materialised 0/1 (`v_cmp; v_cndmask 0,1; v_cmp_ne; v_cndmask`, 4 VALU instead of 2), (3) ~48 `v_mov imm` per iteration (fold into src0 / hoist), (4) double-buffer offsets via `v_mul_lo_u32` (use an xor induction variable), (5) 64-bit staging pointers rebuilt per iteration (strength-reduce to `ptr += tile stride`), (6) loop-invariant `g8+16` and `key0+g4` recomputed per tile.
* **Parked**: DMA (`FMHA_OPT=dma`) - known broken for H>=2, own backend only, opt-in; the DSL op and MOps stay as infrastructure and nothing in the default path uses them. Rollback-once-per-128-keys: needs 4 live S tiles (+48 VGPR), dropped.
* **Harness**: `tests/gpu/ref_host.cpp` used to treat NaN outputs as passing (`err > tol` is false for NaN); fixed, NaN is now bad. Always check for `mean_err=nan` in old logs.

## Round 45 (kernel-opt): LDS conflicts, peeled tail, role sensitivity - all validated negatives for speed

The default stays at 286 TFLOP/s (plain `_t`, scheduler `rewait,xbar`); everything below is opt-in and numerically bit-identical.

* **LDS bank conflicts (`FMHA_OPT=swz`)**: `SQ_LDS_BANK_CONFLICT` 524288 (exactly 2^19) -> 8704 (-98%), `SQ_LDS_IDX_ACTIVE` 1.93M -> 1.41M, TFLOP/s unchanged (286.1 vs 286.2): the kernel is not conflict-bound. Bit-exact vs the default build (REF_DUMP2 md5 equal on 5 shapes). All conflicts came from the `ds_read_b64_tr_b8` V reads (`nopv` -> 0); K reads, K-scale reads and staging writes are conflict-free at pitch 144; pitch 128 has 6.08M (160: 1.33M; 176/272: 524288). Layout: dense 128-byte rows, 16-byte chunk c of row r stored at chunk `c ^ s(r)`, `s2 = u2 ^ p, s1 = u0, s0 = u1 ^ p` with `u = (r >> 1) & 7`, `p = r & 1`. **How it was found (lesson)**: the LDS bank model matters and only hardware can confirm it. A swizzle derived for "64 banks, 16 lanes per b128 cycle" fixed V but left 262144 conflicts in the staging writes; one derived for "32 banks, 8 lanes per cycle" fixed the writes but broke the K reads (524288). A brute-force search over `s(r) = pi(u) ^ (d (r & 1))` (8! x 8 candidates, evaluated against BOTH models for the K fragment reads, the V tr reads and the staging writes) gave a function with zero conflicts under both; on hardware it removed the conflicts. Opt-in (costs 4 address registers + setup VALU for no speed).
* **Peeled tail iteration (`FMHA_OPT=peel`)**: the loop leaves through a head test when the iteration holds fewer than 128 keys and the masked iteration runs once after the loop (needs the DSL verifier to allow loop variables to be updated after their loop ended - added). 15/15 shapes on both backends. `SQ_IFETCH` 794k -> 788k (-0.7%), TFLOP/s unchanged: instruction fetch is not the limiter either.
* **Role sensitivity (which wave group is critical?)**: `dummyA` / `dummyB` add 192 dependent integer ops per iteration to role A (waves 0-3, stages K + K-scales) or role B (waves 4-7, stages V), kept alive by a scratch LDS store (dead code is removed by the compiler: my first run without the store showed no effect and was invalid). Results at 286.6 TFLOP/s base: `dummyA` 261.8 (-855 cycles/iter), `dummyB` 281.0 (-180), both 262.8; with `nobar`: A 295.1, B 302.0 (both sensitive when no barrier couples them, base 324.2). So role A is on the critical path and B has slack at the barrier. Moving the K-scale staging to role B (`kdB`) does not rebalance it (284.7, A still 5x more sensitive). The cause of A's extra latency is not identified. The most promising untested lever is a phase offset between the two wave groups that share a SIMD (wave w and w+4): all eight waves currently rendezvous at one barrier, so their MFMA and VALU phases coincide; the vendor kernel (and our slower `_pp`) offsets them.
* **VALU is not the limiter**: with live dummy VALU work removed from the picture, the dead-code removal and scheduler results (see round 44 addendum 2) agree: 14% fewer VALU instructions or +192 independent VALU per iteration in role B give no change.

### Round 45 close-out (kernel-opt): occupancy, phase-offset probe, final verified state

* **Occupancy (rocprofv3, same process, B1 H8 S2048)**: both kernels launch 64 workgroups x 8 waves = 512 waves (`SQ_WAVES` 512 each) on 256 CUs, so 64 CUs are busy and 192 idle for ours AND the vendor kernel; both use 512 threads and ~246-256 VGPR = 2 waves per SIMD. Resident waves per busy CU (`SQ_WAVE_CYCLES / SQ_BUSY_CU_CYCLES`): vendor 9.48M / 4.79M = 1.98, ours 16.96M / 8.73M = 1.94 - identical. `GRBM_GUI_ACTIVE` 768k vs 1272k, `SQ_BUSY_CYCLES` 2.39M vs 4.39M (ratio 1.8 = the speed ratio). **Occupancy is therefore ruled out as the difference**: the same two waves per SIMD run 1.8x longer; at the barrier the waves are present but stalled (dependency latency / role imbalance), not missing.
* **Phase-offset probe**: delaying role B by 8/24/64 dependent ops per iteration (`dBn8/24/64`) is neutral (285.4/286.6/286.4 vs 286.9) and delaying role A costs monotonically (282.7/281.3/276.6): a plain delay does not create useful MFMA/VALU interleaving; a real offset needs a software-pipelined role B (state carried across iterations), which does not fit the 256-register budget (a prior ping-pong attempt `_pp` hit 256 VGPR and 205 TFLOP/s; holding 4 S tiles for one rescale decision per 128 keys needs +48 VGPR).
* **Register budget is the structural wall**: 8 waves x 2 per SIMD = 256 registers per wave (arch + accumulator share the 512-entry file at 2 waves/SIMD). The O accumulators (4 x Frag16 = 64) and the two live S tiles (32) plus K/V fragments leave no room for software pipelining or a wider max/rescale window. A lower-occupancy-per-wave design (O split across two wave groups through LDS, 4 waves/SIMD at <=128 VGPR) would be a rewrite.
* **Final verified state (kernel-opt, fd549a2+)**: default plain `_t`: **286.5 TFLOP/s on llvm-tail, 281.3 on own** (vendor 518-519), mean err 1.254e-3 (vendor 2.42e-3); 15/15 shapes on both backends; `_sv` (svcorr) 15/15 on both, ~265; determinism 5/5; `cargo test` green. Reproduce: `OUR_KERNEL=fmha_fwd_hd128_mxfp8_t AB_VARIANTS="FMHA_OPT=" bash tools/ab.sh` (30 iters, same-run vendor), shapes: `REF_ROWS=16 OUR_KERNEL=fmha_fwd_hd128_mxfp8_t REF_CO=<vendor co> bash tests/gpu/run_ref.sh`, add `BACKEND=own` for the own backend, `REF_CO=<vendor co> bash tests/gpu/determinism.sh`.
* **Gap decomposition (1.81x, 7.3k cycles/iter)**: barrier 1.05k, staging 1.2k (together 2.1k), PV 1.25k, exp 0.87k, lazy rescale 0.6k; even with staging and barrier removed (`nostage,nobar` = 376.8 TFLOP/s) the kernel stays 1.4x from the vendor, so the residual is the MFMA/LDS-read/latency structure itself, not issue slots, conflicts, instruction fetch, occupancy or waits (all measured neutral). No kernel-only micro-change closes it; each element of the vendor structure was measured net-negative or blocked on its own here (Schraudolph: accuracy/speed trade, DMA: H>=2 bug, ping-pong: register budget).
* **Recommendation for a future round**: (1) fix DMA staging for H>=2 (a real LDS tile dump needs the dbgk crash understood) - it removes most of the 2.1k staging/barrier cycles; (2) only then redo the phase-offset ping-pong on top of DMA staging, which frees the staging VGPRs that block it today; (3) the O-split-across-wave-groups design for 4 waves/SIMD is the other structural option; (4) keep `AMDC_SCHED` (rewait,xbar) default and `ref_host`'s NaN check.

## Round 46 (kernel-opt): the DMA H>=2 bug was an own-backend register-allocator bug

* **Root cause**: `regalloc.rs` ended the scalar allocation with a loop that ASSIGNED `max_s = (highest still-live slot) + 1`, discarding the running maximum. Kernels whose last live scalar registers were low declared too few SGPRs: the DMA FMHA kernel declared `.amdhsa_next_free_sgpr 32` while using up to s43 (the workgroup-id copies and kernarg scalars), the non-DMA kernels declare 48. The hardware does not guarantee registers above the declared count; block 1 (head 1) got a corrupted head index and its whole output stayed at the memset zeros (32768 of 32768 head-1 elements were 0.0, head 0 matched the ds_write path bit-exactly). Single-head grids only ever launch workgroup id 0, which is why every probe and single-head kernel passed. The same bug explained the early-`return` debug-dump crash. Fix: delete the loop. New test `tests/reg_declaration.rs` checks for every registered kernel (default build and the `dma` FMHA build) that the declared SGPR/VGPR counts cover the highest register in the emitted instruction stream; it fails on the old regalloc ("SGPR s43 used but .amdhsa_next_free_sgpr = 32").
* **Verification**: own backend `_t` default 15/15 + determinism 5/5; own vs llvm-tail outputs bit-identical on 6 shapes for `_t` and `_sv`; DMA staging (`FMHA_OPT=...,dma`, own backend only) now correct on all H: `_sv`+svcorr+dma 15/15, determinism 5/5; plain `_t`+dma 15/15, determinism 5/5.
* **DMA performance (honest)**: with the default scheduler (rewait/xbar/ld) DMA is NOT faster for the plain kernel: 288.5 vs 287-288 (swz) TFLOP/s on B1 H8 S2048; `_sv`+svcorr 267.8 vs 260.0 (+3%). Without the scheduler (`AMDC_SCHED=0`) DMA gives +7% (278.2 vs 260.1, VGPR 256 -> 240): the scheduler already hides the staging cost, so the 1.2k cycles/iter of the `nostage` ablation was mostly the staging VALU/LDS-write issue that the scheduler now overlaps, not DMA-able latency; the barrier cost remains. DMA stays opt-in and own-backend only (llc SIGABRTs on the DMA MIR, so the LLVM tail cannot express it).
* **Separate bug, handed to the backend teammate**: `AMDC_LOWER_OPTS=dce` (also `all` and the default `dce,cmpfold`) makes the sv kernel built with `FMHA_OPT=cvt4,dma,nodmak,nodmav,kh0` (K/V base offset forced to the constant 0, a debug option) fail with an illegal memory access on `REF_B=1 REF_H=2 REF_HK=2 REF_S=256 REF_SK=128`; `cmpfold`, `imm` and `none` run clean, and the declared SGPR count is correct (40 for the used s39). Not reproducible with the default kernels, but dce removes something it must not for this pattern.
* **Phase-offset ping-pong on top of DMA**: not attempted: DMA only frees 16 VGPRs (240 used) and the scheduler already captures the staging gain, so the register budget that blocked ping-pong (2 waves/SIMD x 256) is unchanged in practice.
## Round 47 (kernel-opt): rewrite to the vendor's structure (`fmha_fwd_hd128_mxfp8_vs`) - correct, deterministic, NOT faster than `_t`
Spec: `docs/vendor-structure.md`. New kernel `fmha_fwd_hd128_mxfp8_vs` (own backend only: LLVM tail cannot export DMA; not in `kernel_names()`), 15/15 shapes, determinism 5/5, bit-identical to the sv default on 1 8 2048 2048.
- **Numbers** (B1H8S2048, 30 iters): `_vs` svcorr rollback 255-262 TFLOP/s (vs `_t` 292); `+svscale` 312 (14/15, see below).
- **Ablations** (FMHA_ABLATE, timing only): nobar 295, nodma 304, novmwait 262 (no effect), novalu 398, noqk 362, nopv 289, novalu+nodma+nobar 531. The ping-pong MFMA/LDS skeleton alone runs at 531 TFLOP/s (> vendor); the softmax VALU phase is the limiter. `svinf` (never roll back, wrong numerics): 262 -> 339: the rollback costs ~1.9k cycles per 128 keys; headroom knobs (sh2/sh4) do not change the rate because row maxima grow by tens of doublings on this data.
- **Bugs found**: (1) `Opnd::Var` read in the same iteration as `set_var` returns the iteration-start value (stale P); (2) a P fragment carried across the back edge in loop variables gave non-deterministic output in the waves that carried it (fixed: group B runs A's sequence one phase later; nothing carried); (3) DMA global-address compensation must be positive (a negative 32-bit offset is zero-extended: illegal access for SK<8); (4) `AMDC_LOWER_OPTS=...,sgpr,addrfold` miscompile `_vs`+svscale (all NaN; reproducer handed to backend); (5) `rv`/`rm`/`rp` are PowerShell aliases (tooling).
- **svscale** (`FMHA_OPT=...,svscale`, opt-in): instead of the common-case rollback, a per-lane power-of-two P scale (cvt scale operand, which DIVIDES) with the PV/row-sum MFMA scale_b per lane (123+Delta); rebase only when S' > 0.45. 312 TFLOP/s (+22% over `_vs`, +7% over `_t`), determinism 5/5. Accuracy is NOT vendor parity: mean error up to 1.63x the vendor's (REF_ROWS=16 sweep: 6 OK / 9 FAIL at the 1.05 strict parity gate) and 1 element of 1 shape (2 8 700 640 gqa2: want -0.1094 got -0.1680) fails the absolute tolerance. Cause: rollback aligns the reference to the row max, so the dominant key's P is exactly a power of two (no fp8 rounding error); with a fixed reference it lands at a random phase (+-6%). Lowering the rebase threshold restores accuracy but also the rollback cost (0.27: 250 TFLOP/s). Speed/accuracy trade-off, not a free win.
- **smfmac row sums: BLOCKED.** `Stmt::VarSmfmac`/`var_smfmac_fp8` added. With the constant-ones A of the recipe the result is D = 24 x (l_q + l_q+16) (two queries mixed: D column n sums lane groups n, n+16, n+32, n+48 while our P has query = lane%32); a lane-dependent A (ones iff (m/4 + kb) even) still mixes queries (D0/l = 16 x {1,2,~4}). The per-lane A/index construction of the vendor is not reproduced. `FMHA_OPT=smf` is wrong; `smfdbg` dumps both sums. fp8 `v_dot4` does not exist on gfx950 (assembler rejects it); bf16/f16 dot2 would sum the unquantised P (accuracy loss).
- **`fmha_fwd_hd128_mxfp8_vs128`** (128-key units, one rollback, two barriers; Q fragment b1 and the ones block in LDS): does not fit the register file even with `dce,cmpfold,scalarize,sgpr,addrfold,imm` (83-112 live vregs, 227-255 dwords at the failing V16 allocation). Needs ~20 more VGPRs (a working cheap row sum, or fewer loop-invariant V1).
- Parity gate: `tests/gpu/run_ref.sh` prints ours/vendor mean+worst and PARITY_OK/FAIL (mean-only, margin 1.10 recommended; worst-element on 16 rows is noise).

## Round 48 (kernel-opt): where the rollback cost goes, and why an exact "scale the new contribution" rollback does not exist
`_vs` (svcorr, own backend, B1H8S2048, 30 iters): 254.4 TFLOP/s; with the rollback body ablated (timing only, wrong numerics): no S shift (`rbnos`, 16 `v_pk_add_f32` per rollback) 271.1; no O rescale (`rbnoo`, 32 `v_pk_mul_f32`) 284.9; neither 304.3; no rollback at all (`svinf`) 339. So per rollback: S shift ~17 TFLOP/s, O rescale ~30, the remaining per-lane scalar chain + branch + `v_exp` ~35. Rollback fires in ~70% of units because the decision is wave-wide (32 query rows per wave, any record in any row) and the row maxima of this data jump by tens of doublings (headroom knobs sh2/sh4 and rebase thresholds down to 0.35 do not change the rate).
Packed ops are already used (O rescale = 4 fragments x 8 `v_pk_mul_f32`).
**Exact "rescale the contribution instead of the accumulator" (fold into scale_b) is not possible**: the MFMA scale operand is a power of two, so it can only absorb an INTEGER reference shift (that is `svscale`). Aligning the reference to the actual row max needs a real-valued shift; the accumulator then has to be multiplied by 2^delta (or the new contribution by a non-power-of-two, which must happen before fp8 rounding and breaks the alignment). Simulation (python, 2048 keys, 64-key units, e4m3 P, log2-domain logits N(0, sigma)): mean |O/l error| aligned vs fixed-reference-integer-shift: sigma 2: 1.10e-3 vs 1.16e-3; sigma 10: 3.75e-3 vs 5.22e-3 (+39%); sigma 30: 1.77e-3 vs 2.15e-3 (+21%) - the same direction and size as the kernel measurement of svscale vs rollback; the aligned max key is exactly representable in fp8 (P = 16 exactly), which reduces the error of the dominant keys. Hence svscale is a speed/accuracy trade, not an exact reformulation.
Remaining options (not done): cheaper per-lane scalar chain (f16 round trips of the bias word, `v_exp`), fold the S shift into the u16 exponent field (saturating `v_pk_sub_u16`), and one rollback decision per 128 keys (needs `vs128`, blocked on registers).

### Round 48 close-out (kernel-opt): scalar-chain trim = negative, verified state, bugs, open items
**Rollback trim (time-boxed).** `FMHA_OPT=...,rbtrim` removes the two redundant `& 0xffff` masks in the rollback's bias-word chain (`v_cvt_f32_f16` reads only the low 16 bits, `v_cvt_f16_f32` zero-extends): `_vs` (svcorr) 254.6 -> 257.8 TFLOP/s (3 reps: 254.67/254.56/254.54 vs 258.88/258.03/257.83; bit-identical output, mean err 1.7517e-3), i.e. +1.3%. The rest of the chain (f16 round trip that defines the quantised bias, `v_exp`, frag dups) is needed; folding the S shift into the u16 exponent field would add per-unit VALU on the fast path (and needs saturation). Total gain < 2% -> NOT installed as default, `_t` untouched. Rollback cost evidence is the round-48 table above (254.4 full / 271.1 no S shift / 284.9 no O rescale / 304.3 neither / 339 `svinf`) plus the accuracy simulation (aligned vs integer-shift reference: +39% mean error at sigma 10, +21% at sigma 30): the exact rollback cannot be removed, only made rarer (one decision per 128 keys = `vs128`, blocked on registers).

**Verified state** (B1H8S2048, own backend unless noted, vendor same run 515):
| kernel | TFLOP/s | accuracy / gates |
|---|---|---|
| `fmha_fwd_hd128_mxfp8_t` (default) | 292 llvm-tail / 286 own | mean err 1.25e-3 (vendor 2.42e-3), 15/15 both backends, determinism 5/5 |
| `fmha_fwd_hd128_mxfp8_vs` (svcorr rollback) | 255-262 | 15/15, determinism 5/5, 0 LDS bank conflicts, own only |
| `_vs` + `svscale` (opt-in) | 312 | 14/15 absolute (one element), mean err up to 1.63x vendor: speed/accuracy trade |
| `_vs` skeleton, VALU+DMA+barrier removed (`novalu,nodma,nobar`) | 531 (upper bound, wrong numerics) | the MFMA/LDS pipeline is not the limiter |
| vendor | 515 | |

**Bugs found and fixed along this line** (plus those from rounds 44-46): (1) own-backend regalloc under-declared SGPRs (DMA H>=2 failures); (2) dead async loads / loads not drained across branches; (3) `v_readfirstlane` VALU->SALU hazard in the DMA M0 setup; (4) smfmac's in-place accumulator C was invisible in the MIR (hazard padding; found by kernel-occ, my `VarSmfmac` lists C as a source); (5) `sgpr`/`addrfold` lowering opts miscompile `_vs`+svscale (all NaN; reproducer handed to backend, root-caused there); (6) non-determinism from a P fragment carried across the loop back edge in loop variables (fixed by running group B's sequence one phase later; `Opnd::Var` reads the iteration-start value - lint added by backend); (7) negative DMA global-address compensation wraps (zero-extended 32-bit pointer offset: illegal access for SK<8); (8) stale `ref_host` NaN counting (round 44).

**Open items.** `fmha_fwd_hd128_mxfp8_vs128` (128-key units, one rollback, two barriers) needs ~20 more VGPRs; the vendor's smfmac row-sum A/index construction (per-lane `v_cndmask`) is not reproduced (constant-ones A mixes queries q/q+16: D = 24 x (l_q + l_q+16)); gfx950 has no fp8 `v_dot4`; cheaper rollback remains the largest measured cost (5-15%); `_vs` is own-backend only (DMA).

### Round 48 addendum (kernel-opt): LDS layouts of the vendor-structure kernels and the spill base
`amdgpu_dsl::kernels::{VS_LDS_BYTES, VS128_LDS_BYTES, VS_SPILL_BASE}` (bytes). `_vs` declares 99840 = 3-slot ring (3 x 32768: K 16384 + V 16384 per slot, dense swizzled) + K-scale ring (3 x 512 at 98304); `_vs128` declares 102400 = 2-slot ring (0..65536) + K-scale ring (65536.., 2 x 512 in a 2048-byte block) + ones block (67584, 2048) + per-wave Q fragment b1 (69632, 16384) and b0 (86016, 16384). `VS_SPILL_BASE` = 102400 is the first LDS byte the compiler may use to spill lane-invariant values of `_vs128` (2048 bytes per dword = 8 waves x 64 lanes x 4 B; 61440 bytes = 30 dwords up to the 163840-byte limit). Both sides must use the constant, not the number. `_t` keeps its 69632-byte layout. The 3-slot -> 2-slot ring change of `_vs128` is unmeasured (it never builds); on `_vs` the 2-slot design measured the same speed as 3-slot (`novmwait` 262.6 vs 261.9).
`_vs128` register pressure at the allocation failure (own backend, `AMDC_SCHED=0`): `dce,cmpfold,scalarize` 90 live vregs / 227 dwords (V1 79, V16 9); `+sgpr,addrfold,imm` 119 / 255 (V1 109, V16 9). The count does not change with the P-conversion recipe (svcorr on/off), i.e. the pressure is the 9 live V16 (4 O, 4 S, row sum = 144 dwords) plus 79-109 live V1 (loop-invariant addresses, tables and constants), not the V-phase temporaries.

## Round 49 (kernel-opt): `fmha_fwd_hd128_mxfp8_vs128` - 128-key units, one rollback, two barriers: 299.3 TFLOP/s, accurate recipe, parity OK
After backend's pressure-model fix (ghost live-in values, merged from backend-tail) `_vs128` allocates (vgpr 244 with `dce,cmpfold,scalarize,imm`; 253 without `imm`). Own backend only (DMA). Defaults in `lib.rs`: lowering `dce,cmpfold,scalarize,imm`, full list scheduler (reordering).
- **Timing** (B1H8S2048, 30 iters, same-run vendor 517-519, GPU 0, 3 reps): `_vs128` **299.3 +- 0.3** TFLOP/s (298.9/299.2/299.7/299.3/299.6/299.5); with the `ld` scheduler preset 287.9 +- 0.3; without `imm` 281.0 (vgpr 256); `_vs` (64-key) 254-262; `_t` llvm-tail 291.2 +- 0.1, `_t` own 281.6 +- 0.3. So `_vs128` is +2.8% over `_t`'s best backend and +6% over `_t` on the same own backend, 58% of the vendor.
- **Accuracy** (svcorr rollback recipe, accurate): 15/15 vs the CPU reference at REF_ROWS=16 and 0 bad elements on all 15 shapes at REF_ROWS=64; vendor-parity gate (mean, margin 1.10, REF_ROWS=64) PARITY_OK on all 15 shapes (ours_mean 1.8e-3 .. 2.6e-3 vs vendor 1.7e-3 .. 2.9e-3; benchmark shape 1.77e-3 vs vendor 2.4e-3; `_t` has 1.25e-3). At 64 rows the vendor itself shows 4-5 bad elements on two shapes, ours 0.
- **Determinism**: 5/5, three runs (md5-identical).
- **Structure** (static asm counts, `v_*` per kernel): 15 `s_barrier` = per role A: P0,P1 + loop body 2 + peeled masked/unmasked 2+2; B: P0,P1,P2 + loop 2 + peeled 2+1 (last barrier omitted) -> exactly 2 barriers per 128-key tile in the loop; 6 `s_setprio 1` + 6 `s_setprio 0` = one pair per phase pair (loop + 2 peeled copies, per role); 6 `v_exp` = one rollback decision/site per 128 keys per code copy; 16 `global_load_lds` (DMA staging: 4 prologue + 4 loop, per role); no VGPR staging; row sums by the ones-MFMA (2 per tile, A and the Q b0/b1 fragments read from LDS).
- Caveats: own backend only (LLVM tail cannot export DMA); LDS 102400 bytes (ring 65536 + K scales + ones + Q b0/b1 in LDS, spill base `VS_SPILL_BASE` = 102400); mean error 1.77e-3 vs `_t` 1.25e-3 (svcorr Schraudolph recipe vs exact exp), better than the vendor's 2.4e-3. `_t` remains the default accurate kernel.

### Round 49 addendum (kernel-opt): `_vs128` LDS conflicts and code size
- **Bank conflicts fixed**: the Q b0/b1 and ones blocks were read with a 32-byte lane stride (two 16-byte halves per lane at lane*32), a 2-way conflict on every `ds_read_b128`. New layout: first 16 B of every lane contiguous (lane*16), second halves one KB later (1024 + lane*16). `SQ_LDS_BANK_CONFLICT` for `_vs128`: 0 (was ~82 per wave-iteration by kernel-occ's count); speed unchanged (299.0-299.6 TFLOP/s, 3 reps).
- **LDS instruction count** stays ~73 per wave-iteration (596480 total): +20 over `_vs` are the Q b0/b1 (8 ds_read_b128 each per tile) and ones-block (4) reads that replace 24 VGPRs. `FMHA_OPT=...,q0reg` (Q b0 back in registers) cuts it to 531k but runs slower (287.9; the scheduler's result changes, vgpr 243).
- **Code size**: `FMHA_OPT=...,peelone` (one always-masked peeled copy instead of masked + unmasked) shrinks the kernel from 5568 to 4713 instruction lines but needs 256 VGPRs and runs at 291 TFLOP/s (vs 299.3); the dynamic loop body is unchanged (peeled copies execute once), so it does not address the loop's instruction-fetch footprint. Default stays the two-copy form.

## Round 50 (kernel-opt): phase-imbalance ablations of `_vs128`, a merged-group variant (`_vsm`) that is slower, and `s_setprio` removed (+2.8%)
**Ablations of `_vs128`** (B1H8S2048, own backend, 30 iters; 0.0573 ms = ~6.9k cycles per 128-key tile per wave; timing only): none 299.8; `novalu` (softmax VALU phase removed) 437.4 (-2.7k cycles); `noqk,nopv` 359.9 (-1.4k); `noqk` 352.9; `nopv` 348.4; `nobar1` (V->M barrier) 308.9; `nobar2` 309.8; `nobar` 310.0 (barriers ~0.3k); `noprio` 308.5; `nodma` 339.7 (-1.0k); MFMA/LDS skeleton only (`novalu,nodma,nobar`) 531.7 (3.2k cycles); VALU-only (`noqk,nopv,nodma,nobar`) 604.6 (2.6k cycles). The parts SUM to the wall (3.2k + 2.6k + 1.0k DMA + 0.3k barriers = 7.1k vs 6.9k measured): the two waves of a SIMD alternating phases give no overlap, because each wave's own tile time is V + M (its own latency-bound sequence) - the partner only fills the other pipe if both pipes are idle-limited, which they are not (each phase is bound by dependent-instruction/LDS latency, not by pipe throughput).
**`s_setprio` pair hurts**: removing it gives 308.4 +- 0.5 vs 300.0 (3 reps) - now the default of `_vs128` (`FMHA_OPT=...,prio` restores it). Parity 15/15 (REF_ROWS=64), 0 bad, determinism 5/5 x3 unchanged.
**Merged group** (`fmha_fwd_hd128_mxfp8_vsm`, own backend): no phase alternation, no group offset, ONE barrier per tile; per wave: DMA of tile t+2 first, max + rollback of tile t, then P conversion tile by tile with QK(t+1) MFMAs issued into each S tile right after it is converted and PV of the first half issued while the second half is converted (3-slot ring, LDS 135168 B). Correct (15-shape subset, 0 bad, mean err as `_vs128`), but 277-284 TFLOP/s depending on scheduler/lowering opts (best 284.0 with `AMDC_SCHED=0`+`imm`) vs 308 for `_vs128`. Ablations: `novalu` 378, `noqk,nopv` 458, `nobar` 300, `nodma` 313, MFMA-only skeleton 449 (4.1k cycles vs 3.2k in `_vs128`), VALU-only 585 - the sum-of-parts persists inside one wave: in-order issue stalls on (a) the LDS reads issued right before each QK tile's MFMAs (K/Q fragments, K scales: lgkmcnt wait), (b) the second QK MFMA depending on the first (dependent chain, no VALU between them), (c) hazards of rollback VALU on O right after PV MFMAs; prefetching the fragments one tile ahead needs ~32 more VGPRs (the register file is full at 244-256). Conclusion: true intra-wave overlap needs the next-tile fragments (K/Q/scales, ~24 VGPR) resident one step ahead and VALU placed between dependent MFMAs - both blocked by registers; the vendor's VALU phase is ~100 instructions (ours ~400) so it hides under its partner's MFMA phase.

### Round 50 addendum (kernel-opt): software-pipelined LDS reads in the MFMA phase - no gain
Emitted `_vs128` asm (default full scheduler): the A-role MFMA phase has the strict pattern `dsTR x4 -> s_waitcnt lgkmcnt(0) -> MFMA` (x8 for PV) and `dsR x2-5 -> lgkmcnt(0) -> MFMA` for the QK tiles (LDS latency exposed per MFMA), while the B-role copy of the same phase got batched reads with partial waits (lgkmcnt(8)/(14)): the list scheduler's result differs per role. `FMHA_OPT=...,pvpipe` rewrites the PV loop in source order as read(i+1) -> MFMA(i) (+8 VGPR, 244 unchanged): default scheduler 303.3 vs 307.4 (base), `AMDC_SCHED=0` 289.4 vs 292.4, `ld` preset 287.4 vs 288.2, `rewait` 296.2 vs 295.1, `rewait,xbar` 296.8 vs 295.7; `ldshoist`/`hoist=64/128` variants 287-288. No scheduler configuration with the pipelined order beats the default reordering scheduler (307.4); the explicit order does not survive/help. The exposed read latency in the PV steps is therefore not the lever that the ablations suggest, or it is hidden behind the other waves' VALU already; the sum-of-parts wall needs the VALU instruction count cut (backend copy propagation) and the register headroom for deeper prefetch.

### Round 50 addendum 2 (kernel-opt): prefetch variants with the 12 free VGPRs - third negative
`_vs128` at 244 VGPR (12 free), default scheduler, own backend, B1H8S2048, TFLOP/s: base 307.6; `kdpre` (the 4 K-scale dwords read up front, +4 VGPR) 296.2; `qkfirst` (QK(i+1) before PV(i) in the MFMA phase) 280.0; `q0reg` (Q b0 resident, -8 reads/tile) 294.2; `q0reg,kdpre` 292.5; `kdpre,qkfirst` 273.0; `q0reg,kdpre,qkfirst` 297.4; `pvpipe,kdpre` 292.0; earlier: `pvpipe` 303.3, merged-group `_vsm` 277-284. Every manual reordering/prefetch is slower than the scheduler's own result (all variants 0 bad vs the CPU reference): the list scheduler's per-role output changes with the source order and register pressure, and the explicit prefetches lengthen live ranges without making the MFMA phase shorter. Together with the ablation sum-of-parts (3.2k MFMA/LDS + 2.6k VALU + 1.0k DMA) this closes the "intra-wave overlap" line within 256 VGPR and in-order issue: the achievable gain has to come from fewer VALU instructions (backend copy propagation) rather than from reordering. (`retarget2` was not yet on any remote branch when this was measured; re-measure on top of it.)

### Round 51 (kernel-opt): retarget2 default for the vs kernels; prefetch variants re-measured on top of it - still negative
Merged backend-tail (retarget2: cvt results written straight into the MFMA fragment lanes) and made `dce,cmpfold,scalarize,retarget2,imm,sgpr` the per-kernel lowering default for `fmha_fwd_hd128_mxfp8_vs*` (`addrfold`/`shl` stay off). Current default path, own backend, B1H8S2048, 3 reps: `_vs128` **313.1 +- 0.4** TFLOP/s (was 307.6 before retarget2: +1.8%), vgpr 244; `_vs` 263.3 +- 0.3 (was 256.3: +2.7%), vgpr 240; `_t` llvm-tail 291.2, vendor ~518. Gates: 15/15 at REF_ROWS=16 and 16, 0 bad vs the CPU reference on every shape at REF_ROWS=64, vendor-parity PARITY_OK 15/15, determinism 5/5 x3, for BOTH kernels.
Prefetch/overlap variants of `_vs128` on top of retarget2 (TFLOP/s): base 313.4; `kdpre` 304.2; `qkfirst` 306.3 (vgpr 256); `q0reg` 300.8; `pvpipe` 306.4; `q0reg,kdpre,qkfirst` 310.0 (vgpr 252); merged-group `_vsm` 279.1. All slower than the scheduler's own result; closing statement unchanged: within 256 VGPR and in-order issue the intra-wave overlap by manual reordering/prefetch is not achievable; the gains in this line came from fewer VALU instructions (retarget2) and from removing `s_setprio`.

### Round 51 addendum (kernel-opt): the two wave groups ARE in complementary phases; cross-wave MFMA/VALU co-issue works on the hardware
**Lock-step or offset?** `_vs128` is offset by design: group A (waves 0-3) runs `P0, QK(0), P1, loop [V, barrier, M, barrier]`; group B (waves 4-7) runs `P0, P1, QK(0), P2, loop [V, barrier, M, barrier]` - the same loop body, one phase later (A's first loop barrier is B's P2), so on each SIMD (waves w and w+4 share SIMD w mod 4) one wave is in its VALU phase while the other is in its MFMA phase. Evidence: `FMHA_OPT=...,split1` (role A = even waves) and `split2` (role A = waves with bit 1 clear) put the two waves of a SIMD in the SAME group (lock-step): 266.7 / 265.4 TFLOP/s vs 313.5 for the complementary default (-15%), at identical code and gates.
**Hardware co-issue probe** (`sched_coissue_*`, `tests/gpu/sched_host.cpp`, block 512, grid 256, 4000 iterations; waves 0-3 run 4 independent scaled MFMAs per iteration, waves 4-7 16 independent VALU chains): MFMA alone 368 cycles/iter, VALU (fma) alone 715, both 763 (= max + 48, not the sum 1083); with the softmax conversion recipe as the VALU work (pknorm pair, shift/mask/perm LUT, pk_sub, fp8 pack): alone 911, both 979 (max + 68). The SIMD overlaps cross-wave MFMA and VALU (including the conversion instructions) almost perfectly.
**Counters of `_vs128`** (rocprofv3, one dispatch totals): SQ_WAVE_CYCLES 15.45M, SQ_ACTIVE_INST_VALU 4.28M (28% of wave cycles; two waves per SIMD -> ~55% VALU duty), SQ_WAIT_ANY 6.15M (40%), SQ_WAIT_INST_ANY 4.10M (27%), SQ_INSTS_VALU 3.85M, SQ_INSTS_MFMA 180k, SQ_INSTS_LDS 596k, SQ_WAIT_INST_LDS 101k, SQ_INSTS_SALU 187k. So the pipes can overlap and the structure is complementary, yet each wave waits 40% of its cycles: the lost overlap is per-wave dependency/latency stalls inside each phase (in-order issue), not lock-step.

### Round 51 addendum 2 (kernel-opt): V-phase stall sources of `_vs128` (ablations) and three failed K-scale fixes
`_vs128` default (313.4 TFLOP/s = 0.0548 ms), timing-only ablations, own backend, B1H8S2048: no S shift (`rbnos`, 16 pk_add per rollback) 330.8; no O rescale (`rbnoo`, 32 pk_mul) 330.1; neither 358.4 (-1.0k cycles/tile); no rollback at all (`svinf`) 365.1 (-1.15k); K-scale LDS reads removed (`kdc`, constant scale) 337.9 (-0.6k, +8%); K-scale reads kept but the two lane-dependent shifts removed (`kdnoshift`) 312.1 (no effect: the shifts are free, the cost is the LDS read + its lgkm wait in front of every tile's MFMAs); no DMA 354.3 (-0.95k); no softmax VALU (`novalu`) 427.4 (-2.2k); no QK/PV 357.4 (-1.4k).
Fixes tried for the K-scale reads (all correct, 0 bad, same mean error, all SLOWER than the 313.0 base): `kdpre` (4 reads up front) 296.2; `kdv4` (lane-contiguous layout, one ds_read_b128 for the four scales, +4 VGPR) 299.0 (sched `0` 286.9 vs 292.1, `ld` 292.3 vs 294.9); `kdglob` (each wave loads its own K scales from global memory one tile ahead, no LDS staging) 291.4. The scheduler's list order changes with every extra live value, so the measured upper bound of the removal (+8%) is not reachable by moving the read. Remaining candidates with measured potential: rollback (~1.15k cycles; ~35 TFLOP/s of it is the per-lane chain/branch, the rest S shift + O rescale), DMA setup (~0.95k), K-scale latency (~0.6k).

### Round 51 addendum 3 (kernel-opt): rollback chain trim, DMA address hoist, early K-scale issue, scale-folded rollback
`_vs128`, own backend, B1H8S2048, TFLOP/s (base 313.2): (a) `rbtrim` (the two redundant `& 0xffff` masks in the rollback chain) **315.5**/315.6/315.5 (+0.8%, bit-identical output) - now the default (`norbtrim` disables); (b) `dmah` (tile-invariant per-q address parts precomputed: +4 VGPR) 307.0 (-2.0%); (c) `kdearly` (K-scale dwords of the next tile read at the start of the M phase, behind the PV MFMAs) 302.0 (-3.6%); `rbtrim,dmah` 308.6; `rbtrim,dmah,kdearly` 297.3. Gates on the new default: 3 reps 315.4/315.6/315.0 +- 0.3, parity 15/15 (REF_ROWS=64), 0 bad vs the CPU reference, 15/15 at 16 rows, determinism 5/5 x3.
(d) Scale-folded exact rollback: not built. Folding the O rescale into the scale operand is exact only for an INTEGER reference shift d (both factors powers of two) - that is `svscale`, whose accuracy loss is intrinsic to the integer shift (the unit's maximum key lands at a random fp8 phase instead of exactly 16; simulation: +21..39% mean error at sigma 10..30, kernel: mean error up to 1.63x the vendor). The exact (real-valued) shift to the true max needs a multiplication by 2^delta with a non-power-of-two factor, which no MFMA scale operand can express; clamping is not the issue.

### Round 51 addendum 4 (kernel-opt): where the remaining VALU of `_vs128` is (counters per wave-iteration, post retarget2/rbtrim)
rocprofv3, B1H8S2048, one dispatch, divided by 8192 wave-iterations (512 waves x 16 tiles; same normalisation as kernel-occ's tool: LDS 596480/8192 = 72.8). `_vs128` | `_t` (kernel-occ) | vendor (kernel-occ): SQ_INSTS_VALU **468.2** | 404.5 | 250.9; MUL_F32 37.2; ADD_F32 31.3; FMA 0; CVT 68.7; INT32 60.8; other (MOV/TRANS/CMP/PERM/PK/MAX) **270.2** | 203.1 | 79.6; SQ_INSTS 617.6; SALU 22.8; SQ_WAIT_ANY **741.6** | 739 | 415; SQ_WAIT_INST_ANY 495; SQ_ACTIVE_INST_VALU 520.2; SQ_WAVE_CYCLES 1870; LDS instr 72.8 (level 47.2, bank conflicts 0); VMEM 7.6; IFETCH 132.3; MFMA busy 1280/wave-iter.
Static composition of the A-role loop (466 instructions, 436 VALU incl. 22 MFMA; the rollback block included): packed (pk_sub_u16 32 + rollback pk_add/pk_mul 64) 96, CVT 67 (32 pknorm + 32 scalef32/pack + ...), INT32 shifts/masks 57 + `and` 32 + `v_add_u32` address adds 35, max3 33, perm 33, MOV 30, cmp/select 21, f32 8. **The svcorr correction (per packed pair: shift, mask, perm, pk_sub = 4 ops x 32 pairs = 128 VALU per tile) is the largest single item (27% of the VALU, and all of the INT32/PERM/pk_sub budget outside the rollback).** Test: `FMHA_OPT` without `svcorr`: **346.2 TFLOP/s (+9.7%)**, benchmark mean error 2.16e-3 (vendor 2.42e-3) - but vendor-parity FAIL on 14/15 shapes at REF_ROWS=64 (ours_mean 2.7e-3..3.7e-3 vs vendor 1.7e-3..2.9e-3) and 52 elements outside the absolute tolerance: the LUT correction is the price of accuracy parity and cannot be dropped; a cheaper formulation was not found (the selector needs (u>>4)&7 per half: perm selectors above 12 are constants, so the shift+mask cannot be fused).
Remaining VALU budget per tile (dynamic ~468 per wave-iteration): fast path ~320 (of which svcorr 128, pknorm/pack 64, max3 32 + partner max, address/ring/DMA ~80), rollback expected ~66 (70% x ~94), prologue/epilogue amortised ~70-80 (Q/ones staging, final scaling), plus the rest of "other". Vendor: 250.9 with a LUT-free conversion (32 pknorm + 32 cvt) and no per-tile bookkeeping.

### Round 51 addendum 5 (kernel-opt): can the svcorr correction be cheaper? - no (analysis; irreducible at 4 packed ops per pair)
The correction adds, per u16 half-word, Delta(m) = 128 (2^f - 1 - f) (f = m/128, m = the 7 mantissa bits of the pknorm result, max 11 in units of the mantissa LSB) back into u before the fp8 conversion: u' = u - LUT[(u >> 4) & 7] (shift, mask, `v_perm_b32` with the 8-byte table in two registers, `v_pk_sub_u16` = 4 packed ops per 2 values, 128 per 64 values). Alternatives costed (packed ops per pair of values): (1) integer quadratic Delta ~ m (127 - m) / 373: `and` (m), `xor 0x7f` (127-m), `v_pk_mul_lo_u16`, `v_pk_lshrrev_b16`, `v_pk_sub_u16` = 5 > 4; (2) LDS table (`ds_read` per pair, address from the same shift+mask): the same shift/mask/sub plus an LDS round trip, no saving; (3) fp32-domain polynomial before pknorm: needs `v_fract_f32` (not packed: 64 ops per tile) plus a packed parabola (>= 3 ops per two values): > 128; (4) a global scale/offset to hide the correction is exact only for uniform factors (a constant factor on P cancels in O / l, so a constant offset in the exponent field - free in the MFMA bias - only removes the bias, while the f-dependent SHAPE error (+-3% relative, up to 5.7% of 2^f) remains: that is exactly what the parity gate sees: 2.7e-3..3.7e-3 vs the vendor's 1.7e-3..2.9e-3 mean at 64 rows, 52 elements outside the absolute tolerance without svcorr); (5) merging the shifts of the two u words of a 4-value group with a 64-bit `v_lshrrev_b64` saves 1 of 8 ops per group (~16 per tile, ~1%) but needs the two u values in adjacent registers (retarget conflict) - not attempted. The selector of `v_perm_b32` cannot take (u & 0x70) directly (selector bytes above 12 select constants), so shift+mask cannot be fused. Conclusion: with the bit-trick exp2 the correction costs 128 VALU per tile = 9.7% of the wall (346.2 vs 315.6 TFLOP/s) and is the price of the vendor-parity gate. How the vendor reaches parity accuracy without a visible correction step in its fast path (same pknorm + scalef32 pair in the disassembly) is not established; its reference/scaling or rounding must differ in a way I have not found. This analysis was by op-count costing only: none of the alternatives was built or measured (nothing beat the 4-op LUT on paper).

### Round 52 (kernel-opt): overlap retry with 24 free VGPRs (zeroinit, vgpr 232) - closed, negative
Base `_vs128` 316.0 TFLOP/s (vgpr 232). Variants (own backend, B1H8S2048, all 0 bad): `qkfirst` (QK(i+1) MFMAs issued before PV(i)) 310.8 (vgpr 256); `qkfirst,kdpre` 309.4; `qkfirst,pvpipe` 297.9; `qkfirst,pvpipe,kdpre` 297.4; `q0reg` (Q b0 resident, +8 VGPR) 301.7; `q0reg,qkfirst` 310.5; `q0reg,qkfirst,pvpipe` 310.9; `q0reg,qkfirst,pvpipe,kdpre` 310.9; `kdearly` 304.3; `pvpipe,kdearly` 303.3; `q0reg,pvpipe,kdearly` 310.1; merged-group `_vsm` 287.5 (vgpr 256; was 279 before zeroinit; 281.7 with `AMDC_SCHED=0`). Nothing beats the base; the best variant is -1.6%. Closing statement: with the register file at 232-256, in-order issue and the list scheduler's order, intra-wave MFMA/VALU overlap by reordering or prefetch does not pay on this kernel: the complementary-phase pairing (worth 15%) is already in place, the hardware overlap is proven by the co-issue probe, and the remaining wall is per-wave dependency stalls plus the VALU count (468/wave-iteration, svcorr = 128 of it).

## Round 53 (kernel-opt): the exact-reference recipe ("vsx") - svcorr dropped for long sequences, 351.9 TFLOP/s, mean error 6.9e-4 (3.7x better than the vendor)
Prompted by kernel-occ's vendor-spec 4.7 (the vendor does NOT correct the Schraudolph conversion; its accuracy comes from the reference path). Findings on `_vs128` (own backend, B1H8S2048 unless noted):
- (a) **two-word bias** (`bias2w`: bias = 4 (hi + residual) with two f16 words over the 8 K slots; the reference shift delta = b_q - b_old that moves the scores and rescales O/l is exact to ~2^-22 instead of the single f16 grid of ~1.2e-4): with svcorr 307.4 TFLOP/s (-2.6%: more rollback ops) but the best accuracy so far (benchmark mean 1.68e-3, parity 15/15, ours_bad 0); WITHOUT svcorr 347.7 TFLOP/s but still parity FAIL on 14/15 shapes (ours_mean 2.8e-3..3.6e-3 vs vendor 1.7e-3..2.9e-3) - the quantised delta was NOT the source of the no-svcorr error.
- **The rollback threshold is the source.** With the old threshold (rebase only when the maximum exceeds 135/512 = 4 doublings above the aligned level 131/512) the dominant key keeps a random mantissa phase between rollbacks, and the Schraudolph shape error (0..-6%, exact only at f = 0) is no longer cancelled by an aligned maximum. Re-aligning EVERY unit (`thr120`: rebase whenever the unit maximum exceeds 120/512; the overflow bound is 2^11) puts the max key at f = 0 (exact) each time: no svcorr, 344-346 TFLOP/s, benchmark mean error 6.95e-4 (vendor 2.4e-3), parity OK on 13/15 shapes; headroom sweep (bias2w, no svcorr): thr135 3.30e-3, thr133 3.07e-3, thr132 2.78e-3, thr131 2.33e-3 (parity 11/15), thr130 1.78e-3 (12/15), thr126 8.9e-4 (13/15), thr120 8.4e-4 (13/15); `sh2/sh4/sh6` P_max placement had no effect. The rollback cost is unchanged (it already fired in ~70% of the units).
- The two failing shapes (SK=100, SK=129: single/2 tiles, ours_mean 2.8e-3..3.1e-3 vs vendor 1.7e-3..1.8e-3) have too few keys for the shape error to average out, and one element of 1 16 256 1024 (gqa 4) stays 0.0603 off (want -0.07145 got -0.01111, tolerance ~0.0555). Hence **sequences shorter than 9 tiles (nkv < 9 = SK <= 1024) use a second code copy with the LUT correction** (two complete kernel copies selected once at start: `vsx`, `cs9`; no branch in the loop). Long sequences (the benchmark and anything >= 9 tiles) run without svcorr.
- **Default `_vs128` recipe = vsx** (`novsx` restores svcorr everywhere / one-word bias / threshold 135): 3 reps **352.4 / 351.6 / 351.6 TFLOP/s**, vgpr 232, benchmark mean error **6.95e-4** (vendor 2.42e-3, `_t` 1.25e-3), 15/15 at 16 rows (REF_KERNEL_OK), 0 bad vs the CPU reference on every shape at both 16 and 64 rows, vendor-parity PARITY_OK 15/15 at 64 rows (ours_mean 0.8e-3..2.6e-3 vs vendor 1.7e-3..2.9e-3 per shape), determinism 5/5 x3. At 64 rows three shapes show REF_KERNEL_FAIL because the VENDOR has 1/4/5 bad elements there (ours 0).
Status line: `_vs128` 351.9 vs `_t` 291 (llvm-tail), vendor ~514: gap 1.46x.

### Round 53 addendum (kernel-opt): setprio window and single bias MFMA on the vsx recipe - both negative
`_vs128` vsx default 351.1-352.7 TFLOP/s (vgpr 232). (3) `s_setprio 1/0` pair over the MFMA phase (`FMHA_OPT=...,prio`): 349.7 (-0.9%): the balance after the VALU shrink still does not want it. (1b) MFMA count 22 per wave-iteration = 8 QK scaled + 4 f16 bias (one per S tile) + 8 PV + 2 row sums (the scaled 32x32x64 ones-MFMA, not f16): the vendor issues ONE bf16 bias MFMA per 128 keys. New `Stmt::VarMfmaScaleC` (destination variable and accumulator input are different registers; regalloc now accepts C != D when disjoint) lets the four first QK MFMAs take the single bias accumulator as their C input (`FMHA_OPT=...,biasonce`): correct (bit-identical output, mean err 6.95e-4) but 348.2 vs 351.1 and vgpr 232 -> 256: the 16-VGPR bias accumulator live across the QK group costs more through the scheduler than the three removed 64-cycle MFMAs save. (1a) smfmac row sums (-2 x 48 cycles) not attempted: needs the vendor's per-lane A/index construction (probe with kernel-occ's smfmac_host).

### Round 53 close-out (kernel-opt): the `_vs128` (vsx) line, end to end
**Final numbers** (own backend, B1H8S2048, GPU 0, vendor same-run ~514-517): `_vs128` vsx **351-352 TFLOP/s** (vgpr 229-232), benchmark mean error 6.95e-4 (vendor 2.42e-3, `_t` 1.25e-3), 15/15 vs the CPU reference at 16 rows, 0 bad elements on all 15 shapes at 64 rows (vendor 1/4/5 bad on three), vendor-parity 15/15 (REF_ROWS=64, mean margin 1.10), determinism 5/5 x3, cargo test green; gap to the vendor 1.47x (was 1.73x); `_t` 291 (llvm-tail). History of the line (TFLOP/s): `_t` 292 -> `_vs` 255 -> `_vs128` 299 -> 308 (no setprio) -> 313 (retarget2) -> 316 (rbtrim, zeroinit) -> **352** (vsx).
**The insight**: the vendor does not correct the Schraudolph conversion (kernel-occ, vendor-spec 4.7); its accuracy comes from an exact, carried reference. Ours rebased only when the maximum rose 4+ doublings (threshold 135/512 vs the aligned level 131/512), so the maximum key kept a random mantissa phase between rollbacks and the conversion's shape error (exact only at f = 0) was not cancelled; svcorr (128 VALU per tile) was compensating. Re-aligning EVERY unit (threshold 120/512) removes the need for svcorr on long sequences: -128 VALU, +10%, mean error 3.5x better than the vendor. Short sequences (< 9 tiles) keep the LUT correction in a second code copy chosen once at start; the two-word f16 bias (exact reference shift) is part of the recipe.
**Negatives of this line (all measured, all documented above)**: `s_setprio` pair (-0.9..-3%); single bias MFMA `biasonce` (-0.8%, vgpr 232 -> 256); K-scale variants `kdpre`/`kdv4`/`kdglob`/`kdearly` (-2..-8%); software-pipelined/prefetched reads `pvpipe`/`qkfirst`/`q0reg` and their combinations (-0.5..-5%); merged-group `_vsm` (279-287); DMA address hoist `dmah` (-2%); alternative svcorr formulations (not cheaper: 4 packed ops per pair is minimal); smfmac row sums (blocked: D mixes queries, the per-lane A/index mapping is kernel-occ's open probe); fp8 `v_dot4` does not exist on gfx950; the `sgpr`/`addrfold` lowering miscompile (fixed by backend). `s_setprio`-free lock-step test (`split1`/`split2`): -15%, so the complementary phase pairing is worth +15% and is in place.
**Last item (VMEM latency, INST_LEVEL_VMEM 119.7 vs vendor 52.6)**: `novmwait` (the vmcnt wait at the end of the V phase removed, wrong data, timing only) 346.7 vs 352.0 (no gain: the wait does not expose latency); `nodma` 378.3 (+7.5%): the cost of staging is the issue side (36 address VALU per tile, 4 VMEM issues, readfirstlane/M0 setup), not the latency; a deeper (3-slot) ring cannot hide what is not exposed, and spreading the four DMA loads across phases is blocked because tile t+1's K is needed by QK(t+1) in the same window. Not pursued.
**Open**: smfmac row sums (needs kernel-occ's mapping; ~2% = 96 XDL cycles per wave-iteration), backend's INT32/SGPR work (INT32 61 per wave-iteration), the remaining ~100 VALU per tile of bookkeeping vs the vendor's 250.9 total (now 347.9).

## Round 54 (kernel-opt): smfmac row sums (kernel-occ's hardware-verified recipe, vendor-spec 4.5.3) - `_vs128` 359.6 TFLOP/s
Wired in as `FMHA_OPT` default `smfv` (`nosmfv` restores the ones-MFMA row sums): per 128-key unit FOUR `v_smfmac_f32_16x16x128_fp8_fp8` (P half 0/1 x index patterns 0x44444444 / 0xeeeeeeee) with a per-lane constant A (`a_rs`, 4 VGPR built once from lane&3 / lane&4), accumulating IN PLACE into a Frag4 loop variable (`var_smfmac_fp8`); dword 2 = 16 x the row sum of the lane's own query (bit-identical output to the ones-MFMA path: benchmark mean error 6.9498e-4 both ways); the rollback rescales dword 2 (`var_scale_lane(la, 2, corr)`), the epilogue takes `frag4_lane(la, 2) * 0.0625`. The ones-MFMA row sums (2 x 64 XDL cycles per wave-iteration) and the LDS ones-fragment loads are gone: 4 x 16 cycles. Variants (same shape, own backend): per-unit zero + extract + scalar f32 loop sum (`smfrs`) 345.5 (-1.5%: zero-init copies, 16 shared P-fragment copies, vgpr 256); `smfrs,smf2` (two independent 2-chains) 350.2; `smfrs,smf2,smford` 351.3; the Frag4 loop-variable accumulator `smfv` **359.9** (+2.6%), `smfv,smford` 359.7. Default path (no env), 3 reps: **359.4 / 359.8 / 359.6 TFLOP/s**, vgpr 248, 15/15 vs the CPU reference at 16 rows, 0 bad on all 15 shapes at 16 and 64 rows (the three REF_KERNEL_FAIL verdicts at 64 rows are shapes where the vendor itself has 1/4/5 bad elements), vendor-parity 15/15 (REF_ROWS=64), determinism 5/5 x3. Gap to the vendor (~514-517): 1.43x.

## Round 55 (kernel-opt): structure items after the smfmac row sums - all inside 1%
Counters of the current default `_vs128` (smfv + vsx; rocprofv3, B1H8S2048, per wave-iteration = /8192): SQ_INSTS_VALU **352.1** (ADD_F32 38.0, MUL_F32 41.1, CVT 72.0, INT32 61.3, other 139.7), SQ_INSTS 494.3, SALU 22.8, SQ_WAIT_ANY **600.8** (741.6 before vsx, 655 vsx), WAIT_INST_ANY 517.8, ACTIVE_INST_VALU 402.1, SQ_WAVE_CYCLES **1629.7** (1870 before vsx), SQ_INSTS_LDS **68.8** (72.8 before: the ones loads are gone; level 46.0, bank conflicts 0), MFMA busy 1216 (1280 before: 2 x 64 ones-MFMA cycles replaced by 4 x 16 smfmac), IFETCH 105.9, VMEM 7.6 (level 119.1).
(1) `biasonce` retried with the freed registers (smfmac): 343.1 vs 359.4 (vgpr 248 -> 240 but slower); `q0reg` 355.2; `biasonce,q0reg` 346.7: still negative (the extra live 16-VGPR bias accumulator / resident Q fragment changes the scheduler's result more than the removed MFMAs/reads save). (2) LDS count 68.8 = K 16 + Q 16 + V tr 32 + K scales 4 (the vendor 52.5 keeps Q in registers); `qonce` (read the two Q fragments once per tile instead of once per S tile: -12 ds_read_b128 per wave-iteration, +16 live VGPR in the QK group): 360.6 vs 359.9 (+0.2%, vgpr 248 unchanged) - within noise, not installed; `qonce,q0reg` 357.0. (3) Two code copies vs one: `onecopy` (timing diagnostic, only the long-sequence copy emitted) halves the static code (9554 -> 4514 instruction lines) but changes nothing: 358.0 vs 359.1 TFLOP/s and SQ_IFETCH 105.8 vs 105.9 - the instruction-fetch counter is set by the loop body, not by the total code size; the short/long copies cost nothing at run time.

## Round 56 (kernel-opt): Q fragments resident in registers - `_vs128` 366.7 TFLOP/s, LDS instructions at the vendor's level
`q0reg` + `q1reg` (both Q fragments live in registers through the loop, 16 VGPR; the LDS Q staging is now unused) is the new default (`noqreg` restores the LDS reads): with the smfmac row sums having freed the ones accumulator and its loads, the register budget carries it (vgpr 248 -> 252). Default path, 3 reps: **366.5 / 367.0 / 366.7 TFLOP/s** (+2.0% over 359.5), benchmark mean error 6.95e-4 (bit-identical), 15/15 vs the CPU reference at 16 rows, 0 bad on all 15 shapes at 16 and 64 rows, parity 15/15 (REF_ROWS=64), determinism 5/5 x3. Counters per wave-iteration: SQ_INSTS_LDS **52.8** (68.8 before; vendor 52.5), SQ_WAIT_ANY **560.4** (600.8), SQ_WAVE_CYCLES 1598.9 (1629.7), SQ_INSTS_VALU 352.1, MFMA busy 1216, IFETCH 101.4, bank conflicts 0. Only half of it works alone: `q1reg` alone 354.4, `q0reg` alone 355.2 vs 359.4 - the two together 367.4.
The bias MFMAs (item 1 of the lead's list) cannot be folded into a constant: the bias accumulator is not the uniform s91 but the CARRIED REFERENCE (-r of each query, an f16 two-word value changed by every rollback; the vendor's single bf16 bias MFMA carries the same quantity) - folding it into the scores needs one `v_sub`/`v_add` per value (64 VALU per tile) instead of 4 MFMAs. Re-measured with the new base: `biasonce` 349.1 vs 367.2 (vgpr 256). Further source-order experiments on the new base are all neutral: `kdpre` 367.3, `pvpipe` 367.0, `kdearly` 366.2, `kdv4` 366.6, `qkfirst` 364.4.

### Round 56 addendum (kernel-opt): bias in VALU, wait structure and ablations on the resident-Q base
**Bias via packed VALU instead of the 4 f16 MFMAs** (`FMHA_OPT=...,biasv`: the S tiles are produced by zero-C scaled MFMAs and the carried reference b = 4 (hi + lo) is added with `v_pk_add_f32` (4 tiles x 8 pk ops = 32 per tile) at the start of the V phase; rollback code unchanged): correct (bit-identical output, mean 6.95e-4) but **324.7 vs 367.1 TFLOP/s (-12%)**: the kernel is VALU-issue bound, not XDL-pipe bound - 32 extra VALU slots cost far more than the 4 removed MFMAs (4 x 32 XDL cycles). Not installed.
**Wait structure of the loop (asm)**: with Q resident the MFMA block already has PARTIAL waits only - per role loop 16-19 `s_waitcnt`: 8 x `lgkmcnt(4)`, 5 x `lgkmcnt(2)` (reads one MFMA ahead), 2 x `lgkmcnt(0)` (one before the barrier, one before the last 5 PV/smfmac MFMAs of the chain), 1 x `vmcnt(0)` (DMA, end of V phase); 24 MFMA/smfmac and 52-53 ds instructions per iteration. **Ablations on the new base** (TFLOP/s, timing only): base 367.3; `novalu` 434.0 (-1.1k cycles per tile); `noqk,nopv` 428.2 (-1.0k); `nodma` 391.0 (-0.44k); `nobar` 354.6 (barriers HELP: -3.5% without); `notr` (V tr reads replaced by a constant fragment) 347.6 and `nokread` (K fragment reads constant) 339.8, `notr,nokread` 337.2 - slower, i.e. the LDS reads are not the exposed term (their removal changes registers/scheduling and loses); skeletons: VALU-only (`noqk,nopv,nodma,nobar`) 653.2 (2.3k cycles per tile), MFMA/LDS-only (`novalu,nodma,nobar`) 549.0 (3.0k): their SUM (5.3k) equals the wall (5.37k) again. The remaining ~24% non-overlapped time is therefore the phases themselves not overlapping (each wave's V + M), not LDS latency, barrier or DMA latency.
**Final counters per wave-iteration** (default, resident Q): SQ_INSTS_VALU 352.1; MFMA busy 1216; SQ_WAIT_ANY 560.4; SQ_WAIT_INST_ANY 543.4; SQ_INSTS_LDS 52.8; IFETCH 101.4; SQ_WAVE_CYCLES 1598.9; bank conflicts 0.

### Round 56 close-out (kernel-opt): phase-rebalance experiments - negative; item 3 closed
Measured V and M durations from the skeletons (per tile: VALU-only 2.3k cycles, MFMA/LDS-only 3.0k) and tried the two rebalances on the resident-Q base (367.0 TFLOP/s): (b) `convlate` - the P conversion of S tiles 2 and 3 moved from the V phase into the M phase (between PV half 0 and half 1, so the VALU of the second half runs while the first half's MFMAs are in flight; both phases then carry ~1.15k + 0.5k of VALU work): correct (bit-identical output) but **343.2 (-6.4%)**; (a) `midbar` - a third barrier per tile between PV and QK (finer phase granularity: the partner group may start its VALU earlier): **331.3 (-9.7%)**. Earlier in this line `nobar` was already -3.5% (barriers help) and `qkfirst`/`kdearly`/`pvpipe` were neutral-to-negative.
**Item 3 outcome**: resident Q fragments (+2.0%, 366.7 TFLOP/s, LDS instructions 52.8 = vendor parity, WAIT_ANY 560) was the last gain; everything else on the structure list was measured negative or neutral with reasons: bias in VALU (-12%, VALU-issue bound), single bias MFMA (-5%, registers), K-scale variants, DMA hoist/deeper prefetch (the DMA wait is not exposed; staging cost is issue-side), `s_setprio`, merged-group `_vsm`, lock-step layouts (-15%), phase-rebalance (convlate/midbar). The remaining gap to the vendor (1.40x) is the sum-of-parts wall (VALU 2.3k + MFMA/LDS 3.0k per tile = 5.3k vs 5.37k measured): the two phases of a wave and of its partner do not overlap even though the co-issue probe proves the hardware can overlap VALU and MFMA across waves. Closing it needs a design change beyond reordering - a register-budget / issue-model change that overlaps V and M WITHIN a wave (a next-generation kernel design), not another tuning item.

## Round 57 (kernel-opt): generation-2 experiment (`fmha_fwd_hd128_mxfp8_g2`) - closed; what it proved and what the next design needs
Design: `docs/kernel-gen2-design.md` (options A-D1 with VGPR/LDS budgets and a cycle model; B and C rejected by LDS bandwidth, D1 chosen). Prototype `fmha_fwd_hd128_mxfp8_g2` (own backend, registered by name only, NOT in `kernel_names()`): QK(t+1, s) is issued inside the V phase of tile t into the S registers freed by the conversion of tile s; M phase = PV + smfmac; 3-slot ring; same arithmetic.
Results (B1H8S2048): bit-identical output to `_vs128`; vgpr 252; g2 **333.4**, g2+`g2split` (conversion + QK of S tiles 2,3 moved into the M phase) **344.2** vs `_vs128` 367.2 (schedulers: `AMDC_SCHED=0` 293.7, `ld` 304.5). Counters per wave-iteration (g2 vs `_vs128`): VALU 385.6 vs 352.1, LDS 52.8 = 52.8, MFMA busy 1216 = 1216, **WAIT_INST_ANY 428.7 vs 543.4**, **WAIT_ANY 812.8 vs 560.4**, wave cycles 1760 vs 1599. Ablations of g2: `novalu` 361.4, `noqk` 346.1, `nopv` 311.2, `noqk,nopv` 348.3, `nobar` 318.5, `nodma` 337.4, skeletons 534.3 (MFMA/LDS) and 516.7 (VALU).
**Positive result 1 - intra-wave overlap works when the work is independent.** Removing QK from g2 saves only 0.3k cycles although it is ~0.77k of XDL work, and the dependency stalls fall by 115 cycles per wave-iteration: the QK MFMAs issued inside the V phase are genuinely hidden under the conversion VALU. This is the first direct evidence in this project that a kernel of ours co-issues VALU and MFMA inside one wave; together with the co-issue/ovl probes (`sched_coissue_*`, `sched_ovl_*`: independent MFMA + VALU interleaved overlap 66-73% of the smaller pipe) it refines the earlier conclusion: it is the DEPENDENCE structure of the tile (QK -> max/convert -> PV), not in-order issue, that prevented the overlap.
**Result 2 - why it did not pay: window balance and rendezvous.** V' = VALU 1.15k + MFMA issue + K reads versus M' = PV only; the complementary partner arrives early at the barrier and waits (SQ_WAIT_ANY +252, wave cycles +161); `g2split` recovers 3.3% (balance matters) but rebalancing alone does not close the gap.
**Prerequisites of a viable gen2 design**: (a) balanced V'/M' windows; (b) synchronisation finer than a block barrier every phase: per-group named barriers if gfx950 offers them for this purpose, double-buffered tile state so a group can run one phase ahead, or a fully merged single window per tile with only the ring-recycle barrier - the merged form `_vsm` (287) pre-dates the exact-reference recipe, smfmac row sums and resident Q and is untried with today's ingredients; (c) the loop authored explicitly (`Stmt::Asm`) if the list scheduler reorders the window.

### Round 57 addendum (kernel-opt): the merged single-window form (`fmha_fwd_hd128_mxfp8_g3`) - final negative of the line
`fmha_fwd_hd128_mxfp8_g3` (own backend, registered by name only, not in `kernel_names()`): ONE window per tile - max + rollback of tile t, then per S tile convert + QK(t+1, s) into the freed registers, then PV + smfmac row sums - with NO barrier between the VALU and MFMA parts and exactly ONE block barrier per tile (after the DMA wait, ring recycle; 3-slot ring; both wave groups run the same code, no prologue phase shift). Today's ingredients (exact-reference recipe, smfmac row sums, resident Q, zeroinit). Output **bit-identical** to `_vs128` (1 8 2048 2048, md5 identical). B1H8S2048: **343.3 TFLOP/s** (vgpr 254-256) vs `_vs128` 367.2 (-6.5%); schedulers: `AMDC_SCHED=0` 321.3, `ld` 316.1; `g3pv` (PV half 0 issued between the conversions of tiles 1 and 2) 322.2. Counters per wave-iteration (g3 | g2 | `_vs128`): SQ_WAIT_ANY **557.6** | 812.8 | 560.4 (the rendezvous tax of g2 is gone); SQ_WAIT_INST_ANY **627.3** | 428.7 | 543.4 (dependency stalls are back: +84 over `_vs128`, +199 over g2); SQ_WAVE_CYCLES 1702 | 1760 | 1599; VALU 385.6 | 385.6 | 352.1; LDS 52.8, MFMA busy 1216 (all equal). Ablations (TFLOP/s): `novalu` 363.7, `noqk` 344.1, `noqk,nopv` 410.3, `nobar` 315.5, `nodma` 338.5, skeletons 539.8 (MFMA/LDS), 515.0 (VALU).
Interpretation: removing the per-phase barriers removes the rendezvous waiting (WAIT_ANY back to the `_vs128` level), but without the complementary phase offset both groups' chains (convert -> QK -> PV) now stall on their own dependencies at the same time (WAIT_INST_ANY 429 -> 627), and the extra VALU of the per-tile conversion calls (+33) stays: the two effects add up to -6.5%. Same conclusion as g2: neither the dependence structure of the tile nor the barrier structure can be fixed independently; the complementary pairing (+15%) provides what the single window loses. Final negative of the generation-2 line: `_vs128` (367.2) stays the fast path.

### Round 58 (kernel-opt): gen3 (offset barrier placement) - negative
See docs/gen3-design.md section 6: g3o (one barrier per tile, A at the top of the window, B after its V part, 4-slot ring, bit-identical output) 334.6 TFLOP/s vs _vs128 367.2 and g3 343.3; free-running (no barrier) 302.7; WAIT_ANY 650 (+92 vs g3), WAIT_INST_ANY 587, wave cycles 1758. The rendezvous skew costs more than modelled and un-synchronised groups drift into unfavourable alignment, so R = 2 was not built. The hoist of the per-window qk values saves 30 static VALU but exceeds the register budget of g3 (scheduler fallback, 325.8).

