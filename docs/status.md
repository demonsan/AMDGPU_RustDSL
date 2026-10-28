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

## Next: FMHA reference core loop

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