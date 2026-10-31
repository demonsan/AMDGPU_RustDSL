# Occupancy hypothesis: would 4 waves/SIMD help `fmha_fwd_hd128_mxfp8_t`? (round 46, branch `kernel-occ`)

> Status: closed (negative) - 4 waves/SIMD does not fit our VGPR/LDS budget and the vendor runs the same 1 block / 8 waves per CU; see docs/current-status.md §5 and rounds 46 and 71-72 of docs/status.md.

Status: **measured negative -- the <=128-VGPR head-dim-split rewrite was not built.** This document records the proxy
measurement, the paper design and the decision; it does not change the default kernel.

## 1. Question and prior evidence

Round 45 close-out (`status.md`) measured that **both** our kernel and the vendor kernel launch 64 workgroups x 8 waves
= 512 waves on 256 CUs at ~2 waves/SIMD (240-256 VGPR), resident waves per busy CU 1.94 (ours) / 1.98 (vendor), while
the vendor is 1.78x faster. So *occupancy count* is ruled out as the difference to the vendor.

The remaining hypothesis tested here is the mechanism, not the count: our wave is **dependency-latency bound**
(`SQ_WAVE_CYCLES` 16.9M vs 9.5M, `SQ_BUSY_CYCLES` 4.8M vs 2.4M, i.e. the same ~26% issue-busy fraction), so a design
with <= 128 VGPR per wave would put **4 waves per SIMD** instead of 2 and let the extra waves hide the stalls.

Same-session baseline used for every statement in this document (B=1, H=8, S=SK=2048, 30 iters, GPU 5, LLVM tail):
ours **292.86 TFLOP/s**, vendor **517.11 TFLOP/s** (matches the README's 292); the kernel is 64 workgroups x 8 waves
on 64 of 256 CUs = 2 waves/SIMD.

## 2. Proxy measurement (Verified)

The real kernel cannot run at 4 waves/SIMD at 256 VGPR, so measure the *mechanism* on small-VGPR kernels with the same
dependency chain, at exactly 1, 2 and 4 waves per SIMD. Two independent ways to set waves/SIMD were used:

* block = 256 / 512 / 1024 threads, grid = 1 (4 / 8 / 16 waves on one CU; a 1024-thread block is 4 waves/SIMD);
* block = 256, grid = 256 / 512 / 1024 blocks on the 256-CU MI355X (1 / 2 / 4 resident blocks per CU = 1 / 2 / 4
  waves/SIMD) as a cross-check; the barrier-free kernels give the same answer both ways.

Harness: `tests/gpu/sched_host.cpp` (same as `tools/sched_hw.sh`), n = 4000 iterations, best of 10 reps, cycles per
iteration at the nominal 2.4 GHz, GPU 5, same session. Device: MI355X, 256 CUs, `maxThreadsPerBlock` 1024, 160 KB LDS,
512 VGPR per SIMD (prints verified from `hipDeviceProp_t`).

Kernels (all <= 128 VGPR, so 4 waves/SIMD fits the register file):

| kernel | per iteration | VGPR |
|---|---|---|
| `sched_valu4` | 1 MFMA + 4 dependent VALU (issue-bound at 2 waves) | 48 |
| `sched_mfma_valu16` | MFMA + 16-lane dependent sub/exp2/add chain (latency-bound at 2 waves) | 53 |
| `sched_mfma_chain` | in-place MFMA chain | 64 |
| `sched_occ_chain` (new) | FMHA-shaped: 2 dependent QK-like MFMA chains -> 32-lane exp2/serial-add softmax chain -> fp8 quantisation -> PV MFMA into a loop-carried O -> block-wide barrier | 88 |

`SCHED_WG` (new build-time env, forwarded by `tools/container/cargo.sh`) sets the block size that
`.max_flat_workgroup_size` advertises; without it the runtime rejects the 1024-thread dispatch (that was the initial
"unspecified launch failure", not a hardware limit).

One block on one CU (grid = 1), cycles/iteration and throughput relative to 1 wave/SIMD:

| kernel | 1 w/SIMD (256 th) | 2 w/SIMD (512 th) | 4 w/SIMD (1024 th) | rel. throughput 1/2/4 | 2 -> 4 waves |
|---|---|---|---|---|---|
| `sched_valu4` (issue-bound) | 250.5 | 259.0 | 474.9 | 1.00 / 1.93 / 2.11 | **+9.1%** |
| `sched_mfma_valu16` (latency-bound) | 478.4 | 693.8 | 1249.8 | 1.00 / 1.38 / 1.53 | **+11.0%** |
| `sched_mfma_chain` | 95.6 | 132.5 | 260.9 | 1.00 / 1.44 / 1.47 | +1.6% |
| `sched_occ_chain` (FMHA-shaped, +barrier) | 1147.1 | 1839.2 | 3222.9 | 1.00 / 1.25 / 1.42 | **+14.1%** |

Grid cross-check (256-thread blocks, grid 256/512/1024 = 1/2/4 blocks per CU), 2 -> 4 waves/SIMD gain:
`sched_valu4` +8.6% (259.5 / 277.2 / 510.3), `sched_mfma_valu16` +11.3% (482.5 / 703.6 / 1264.1), `sched_mfma_chain`
+5.0% (106.5 / 156.1 / 298.0), `sched_occ_chain` +8.4% (1157.1 / 1813.3 / 3346.1). The two methods agree within ~6%.

**Reading (Verified):** the second wave per SIMD already hides most of the dependency latency -- it buys 1.25-1.93x
throughput in these kernels -- and waves 3 and 4 add only **+2% to +14%** on top, even for the FMHA-shaped,
barrier-coupled chain (the most favourable synthetic). The hypothesis' best case is therefore a low-double-digit
percentage gain, not the factor needed to explain the remaining 1.78x gap. This is consistent with round 45's
independent findings (vendor fast at the same 2 waves/SIMD; independent dummy VALU in role B is neutral; 14% fewer VALU
instructions is neutral).

## 3. Paper design of the split (estimate only, not built)

Structure: keep the 512-thread/8-wave block and the 32-rows-per-wave mapping, split the 4 O fragments
(4 x Frag16 = 64 VGPR) over two wave groups: group A (waves 0-3) owns dim tiles 0-1 (dims 0-63), group B (waves 4-7)
owns tiles 2-3 (dims 64-127), so each wave holds 2 x Frag16 = 32 VGPR of O. Group A computes S^T, softmax and the
quantised P (unchanged code) and publishes P (32 fp8 per lane per substep = 8 dwords = 2 KB per substep per wave)
through LDS; group B reads it and runs its 2 PV MFMAs per substep. Register estimate: A ~ S(32) + O(32) + P(8) +
K fragments(8) + addressing(~25) ~= 105; B ~ O(32) + P(8) + V fragments(8) + addressing(~15) ~= 63 -> the <=128 VGPR
target is reachable in principle. Per-SIMD work per 128-key iteration is unchanged: 2 A x (8 QK + 4 PV) + 2 B x 4 PV =
32 MFMAs (today: 2 x 16), the softmax VALU count is unchanged, the V-tile reads are split.

Costs and why the proxy's headroom cannot be captured:

1. Option (a) "both groups recompute S^T+softmax+P" doubles the softmax VALU and adds 16 MFMAs per SIMD (+50% MFMA,
   +100% softmax) -- strictly worse; rejected on paper.
2. Option (b) "A computes, B consumes": the B waves are short (4 MFMAs) and **depend on A's P**, so each SIMD still
   holds only 2 long independent chains (the A waves), exactly as today. The third/fourth waves are not extra copies
   of the critical chain, so the +8-14% proxy figure (measured with 4 *identical independent* chains) is an upper
   bound the split cannot reach.
3. Unpipelined option (b) needs a barrier between the P publish and the P consume: 2 barriers per iteration instead of
   the current 1. The barrier/tile hand-off is already the largest measured single component, 1.05k of the 7.3k
   cycles/iteration (round 44 addendum 2, `nobar` ablation) = 14%, so this alone cancels the proxy gain.
4. Pipelined option (b) (lag group B by one iteration, P double-buffered so the existing one barrier/iteration is
   enough) additionally needs V triple buffering: the lagged B reads a V buffer that the double-buffer schedule
   overwrites. LDS: current layout is 69.6 KB; 2 blocks/CU would need <= 80 KB per block and does not fit with the P
   buffers, so this variant requires one 1024-thread/16-wave block per CU (LDS 101-134 KB, 16-wave barriers, barrier
   cost grows with the wave count). The arithmetic order per output dim is unchanged in this variant, so bit-exact
   output vs the current kernel is achievable -- bit-exactness is not the blocker; expected value is.

Expected net effect: between **-14% and +5%**, midpoint ~0, for a ~500-line rewrite plus a numerics/determinism
re-verification. Not built.

## 4. Decision

**STOP -- do not build the <=128-VGPR head-dim split.** The direct measurement of the mechanism says 4 waves/SIMD is
worth +8-14% on the closest synthetic, while the split spends at least that much on P exchange, extra tile hand-off
sync and A/B imbalance; and the split's 4 waves are 2 long chains + 2 dependent short chains, not the 4 independent
chains the proxy measured. Occupancy/waves-per-SIMD is not the lever for the remaining 1.78x. This extends round 45's
"occupancy is ruled out" from the resident-wave *count* to the latency-hiding *mechanism*.

## 5. What to do instead

1. **DMA staging for H >= 2** (kernel-opt): removes most of the 2.1k cycles/iteration staging + barrier component,
   the largest measured kernel-only lever (round 44/45 ablations: `nobar` 325, `nostage` 332 TFLOP/s vs 286 base).
2. **On top of DMA staging**: the vendor's 2x-unrolled `s_setprio` ping-pong / cross-iteration QK software pipeline at
   the current 2 waves/SIMD. The register wall is what blocked it; DMA staging frees the staging VGPRs, which is a
   better use of the register budget than halving O across wave groups.
3. Remaining VALU lever: an accuracy-preserving cheaper exp2 (the `_sv` Schraudolph variant traded accuracy for speed;
   `svcorr` gives it back at vendor accuracy and is no faster than the plain kernel).
4. No further occupancy experiments on this kernel.

## 6. Reproduce

```bash
# proxy: one block of 256/512/1024 threads on one CU (1/2/4 waves per SIMD)
SCHED_WG=1024 bash tools/container/cargo.sh run -q --bin amdc -- emit sched_occ_chain --out /work/build/occ.s
$ASM_ROC/llvm/bin/clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 \
    -o build/occ.co build/occ.s
bash tests/gpu/run_hw.sh   # or: ./build/sched_host build/occ.co sched_occ_chain <block> <grid> 4000 10
# grid cross-check: block=256, grid=256/512/1024 on a 256-CU MI355X
```

The `sched_occ_chain` kernel and the `SCHED_WG` env are test/proxy infrastructure only; the FMHA default kernels are
untouched. `cargo test` (including `tests/tail_invariance.rs`, which enumerates `kernel_names()`) is green with them.
