# kernel-opt handoff (2026-11-09, updated after the fp8pt work)

Branch `opcov-moe` = Main (merged 2026-11-09) + the MoE stage-1 depth-3 change (5cec2137) + per-tensor FP8 attention (see below).
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
- Gate before every handoff: `cargo test -q --release --no-fail-fast` (506/0 now), `amdc gpu-verify --shipped` (26/100/0), `amdc gpu-verify <k> --strict`
  (0 errors) for touched kernels, md5 of the shipped kernels unchanged unless explained, doc-link check. Then message lead.

## Done (2026-11-09): per-tensor FP8 attention prefill + attention ratio table

- `fmha_fwd_hd128_fp8pt_v3[_causal]` (`crates/amdgpu-kernels/src/fmha/fp8pt.rs`), host `tests/gpu/fp8pt_host.cpp` (vendor + ours, same process; use
  `SCALAR_DS=1`: the vendor .co ignores the descale Hs stride), same-run table script `tools/attn_family_table.sh` (`ONLY=fp8pt,mxfp8,bf16,hd192,bwd`).
- Ratios (vendor/ours): fp8pt 0.88-0.97 non-causal, 0.90-0.93 causal (0.77 at the 64-CTA B1H8 4096²). Full table in op-coverage-design §7.5.
- Gate state: cargo test 506/0 (+ the two new kernels in gpu_semantics' MIR-check list), shipped verify 26/100/0, strict 0 errors on both (causal 12/31/0),
  shipped md5 unchanged; new md5 fp8pt_v3 642420e0, causal 9dbd4b84.
- Reusable findings: (1) a second pipeline pass that reuses the first pass's `KvRing3` keeps its DMA address VGPRs live through pass 1's peeled tile —
  `KvRing3::rebuilt` (fresh lane partitions over the same LDS) fixed the paired causal form (OOM -> 252 VGPR); `amdc regplan <k> --json` live_at_peak
  gives the IR origin of every live vreg at the peak. (2) The verifier's tile-protocol simulation enumerates <= 64 paths over runtime ifs containing
  LDS / barriers; `store_out3_lds_x(.., stores_only = true)` keeps the epilogue guard out of that count.
- Next candidates: hd192 causal pairing with the `rebuilt` idea (bf16v3 has its own `Ring`; 0.74-0.77 now), causal small grids (0.75-0.77), fp8pt
  gap attribution (ATT, ask sched), bwd 0.72-0.86.
## Open items across my areas (current same-run ratios vs aiter; >1 = ours faster)

- **MXFP4 GEMM** 0.86-0.90× (8192³ 270 vs 242 µs). sched's attribution (docs/mxfp4-gemm-profile.md on origin/sched-mxfp4-prof): DMA-issue
  back-pressure ~52 % of the remaining gap, s_nop ~23 % (nop-work merged helped 2 %). Tried and rejected: B direct-to-register (archive/mxfp4-regb2,
  -3 %), A LDS swizzle (archive/mxfp4-a-swizzle, neutral), 256-B K tile (does not fit 2 LDS stages at 256x256), depth 3 (does not fit LDS at 256x256).
  Lead: leave at 0.90×.
- **MLA decode H16** 0.63× (from earlier sessions; head-parallel form wins only at H ≥ 64).
- **Attention bwd** (same run 2026-11-09, attn_family_table.sh, H16): 0.73-0.74× non-causal (2048² B4, 4096² B1, GQA4), causal 0.72-0.75× at 2048², 0.86× at 4096² (earlier records: 0.83× 2048², 0.93-0.98× causal GQA 2048² - other shapes / batches). dkdv 1 wave/SIMD
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
