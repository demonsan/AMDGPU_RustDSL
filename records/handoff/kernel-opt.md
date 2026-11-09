# kernel-opt handoff (2026-11-09)

Branch `opcov-moe` (this commit = @5cec2137 + this brief). The code equals Main + the MoE stage-1 depth-3 change (5cec2137, with lead for merge).
Team: lead (merges), sched (ATT / rocprofv3 attribution, `tools/att_pa.sh`, profile docs on `origin/sched-mxfp4-prof`), backend-llvm-tail (lowering / RA / verifier).
Rules: adopt only measured same-run wins; no env knobs in shipped code (TEMP knobs must be reverted); rejected variants → tag `archive/...` + revert;
docs in Chinese in `docs/op-coverage-design.md` (relative links, `python tools\check_doc_links.py` = 0 findings).

## Workflow (Windows host → MI355 node)

- Local repo `C:\Users\blyu\Documents\Workspace\AMDGPU_RustDSL_fmha` (cwd of the session is `C:\Users\blyu\Documents\Workspace`, use absolute paths).
- Node runs: `C:\Users\blyu\Documents\Workspace\kopt_m.ps1 -Cmd "<bash>" [-Gpus 8] [-Step hh:mm:ss] | select -Skip 4` — it auto-commits, pushes the branch
  as a bundle and runs `<bash>` in a fresh checkout on mi355-gpu-53 (slurm job 9381). Use `export HIP_VISIBLE_DEVICES=5`. Steps over ~10 min move
  to a background job; host CPU references make the attention / MoE hosts slow at big shapes (use `CHECK=2`, smaller shapes, `timeout`).
- Build: `bash tools/container/cargo.sh build -q --release --bin amdc` → `/tmp/target-kfmha/release/amdc`.
- Emit + assemble: `$A emit <k> --backend own --out x.s && /shared/apps/ubuntu/opt/rocm-10.1.0/llvm/bin/clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o x.co x.s`.
- Hosts: `/shared/apps/ubuntu/opt/rocm-7.0.2/bin/hipcc -O2 -std=c++17 tests/gpu/<host>.cpp -o ...`.
- Vendor .co: `/shared/amdgpu/home/blyu_qle/sessions/aiter_src/hsa/gfx950/{fmha_v3_fwd,fmha_v3_bwd,pa,f4gemm,i8gemm,fp8gemm_blockscale,bf16gemm,fmoe_2stages,...}`;
  mangled names via `strings x.co | grep -m1 ^_ZN`. Upstream aiter sources: fetch from GitHub raw (the local tree lacks csrc).
- Gate before every handoff: `cargo test -q --release --no-fail-fast` (499/0 now), `amdc gpu-verify --shipped` (26/100/0), `amdc gpu-verify <k> --strict`
  (0 errors) for touched kernels, md5 of the shipped kernels unchanged unless explained, doc-link check. Then message lead.

## Next item: per-tensor FP8 attention prefill (not started)

Goal: `fmha_fwd_hd128_fp8pt_v3[_causal]` (hd128, causal + non-causal) vs aiter `fmha_v3_fwd/fwd_hd128_fp8.co` / `fwd_hd128_fp8_causal.co`, same run;
then a consolidated attention-family ratio table in §7.5 (like the GEMM table added in §7.x "GEMM 各族与厂商的当前比值汇总").

ABI (aiter `csrc/include/mha_fwd.h` `fmha_fwd_v3_args`, every field in a 16-byte slot; our MXFP8 v3 already uses it):
slot 0 ptr_o (bf16), 1 ptr_q, 2 ptr_k, 3 ptr_v (e4m3, BSHD), 4 ptr_lse, 5 scalar (softmax scale), 6 s_seq_len, 7 s_Seqs, 8 s_Ts, 9 s_Hs, 10 s_Bs,
11 s_gqa, 12 s_k_Seqs, 13 s_k_Hs, 14 s_k_Bs, 15 s_opt, 16 s_lse, 17 s_kv_seq_len, 18 s_qk_head_dim, 19 s_v_head_dim, 20 s_q_head_num, 21-23 v strides,
24-26 o strides, 27 ptr_qseq, 28 ptr_kseq, 29 s_lse_Hs, 30/31 seq padding, **32 ptr_q_descale (0x200), 33 ptr_k_descale (0x210), 34 ptr_v_descale
(0x220), 35/36 s_descale_q_Bs/Hs, 37/38 k, 39/40 v** (per-tensor = one fp32 per (b, h) via these strides; 0 strides = one scalar).
`tests/gpu/attn_host.cpp` already fills the bf16 part of this ABI (lines ~65-90) — add an FP8PT mode (fp8 data, descale floats, vendor run).

Plan: derive from the shipped `_vs128` MXFP8 body (`crates/amdgpu-kernels/src/fmha/mod.rs`: `fmha_fwd_hd128_mxfp8_v3_vs128` ~L536, `kscale_sel = true`
~L584, `out_scale` ~L399, `declare_io` ~L704 with the descale args ~L731-756; `fmha/io.rs`): unit E8M0 operands (0x7F7F7F7F) for the scaled
MFMAs, so the Q-scale load / K-scale ring + DMA + byte select disappear; q_ds * k_ds folded into the softmax scale (uniform scalar), v_ds into
`out_scale`. Keep the `_vs128` cadence, pairing, sched tokens. Verify strict, bad=0 vs the host CPU reference, and md5 of the shipped mxfp8 kernels
unchanged.

## Open items across my areas (current same-run ratios vs aiter; >1 = ours faster)

- **MXFP4 GEMM** 0.86-0.90× (8192³ 270 vs 242 µs). sched's attribution (docs/mxfp4-gemm-profile.md on origin/sched-mxfp4-prof): DMA-issue
  back-pressure ~52 % of the remaining gap, s_nop ~23 % (nop-work merged helped 2 %). Tried and rejected: B direct-to-register (archive/mxfp4-regb2,
  -3 %), A LDS swizzle (archive/mxfp4-a-swizzle, neutral), 256-B K tile (does not fit 2 LDS stages at 256x256), depth 3 (does not fit LDS at 256x256).
  Lead: leave at 0.90×.
- **MLA decode H16** 0.63× (from earlier sessions; head-parallel form wins only at H ≥ 64).
- **Attention bwd** 0.83× (2048²), 0.93-0.98× causal GQA 2048², 0.81× 4096² causal (LPT order), 0.68× 4096² non-causal. dkdv 1 wave/SIMD
  (236 VGPR + 128 AGPR); 2 waves/SIMD infeasible (arithmetic in §7.7); dq 245 VGPR at 2 waves. Rejected: fused dkdvq + dQ atomics (archive/bwd-fused),
  dkdv pairing (archive/bwd-dkdv-pair, superseded by sched's LPT order), dq AGPR pin (drops to 1 wave), dS streaming (VGPR up).
- **Prefill short sequences** (bf16 v3 at small grids, `_ks` key split exists; 1024² causal / small B still below vendor).
- **hd192/hd128 MLA prefill small grid** 0.51× at 1024², causal 0.73× (pairing disabled for VGPRs).
- **Paged decode** B16: split S2 27.2 µs vs vendor 18.5; backend's last-arriver `_c` (S2 23.8) is dispatched now (backend owns `decode16_plan`).
  Rejected: archive/decode16-{lookahead,prefetch,v2,f8,lastc}.
- **GEMM small-M**: done (3-slot ring `Pipeline::rs_depth = 3` / `GemmConfig::depth3`): int8 ≥ 1.0× everywhere, blockscale 0.91-1.45×, bf16 64 forms.
  **Deferred: Mx32 MoE / A16W4 / A8W4 depth 3** (needs per-row gathered E8M0 scale DMAs into the SC ring like the Block128 path in `Body::ab_stage`
  `sc_dma`; decode MoE is bandwidth-bound, expected gain single-digit %). MoE stage 2 / block_m 128 depth 3 measured mixed (not adopted).
- Numerics note: the f8f6f4 MFMA's internal summation precision causes ~2^-10 relative deviations under cancellation at small K (backend
  bisect; not a kernel bug) — the host tolerances at K=640 flag it on every blockscale form.

## Archive tags of rejected experiments (this and earlier sessions)

archive/bwd-dkdv-pair, archive/bwd-fused, archive/decode16-f8, archive/decode16-kr, archive/decode16-lastc, archive/decode16-lookahead,
archive/decode16-prefetch, archive/decode16-v2, archive/mxfp4-a-swizzle, archive/mxfp4-regb, archive/mxfp4-regb2, archive/sched-pa-lookup, plus older
fmha / scheduler tags (`git tag -l "archive/*"`).
