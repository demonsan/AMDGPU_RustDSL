<!-- GENERATED on node 53 / GPU 1 (2026-11-12) by tools/compose_search.py check-moe / check-mla (5 - 7 interleaved repetitions, median): the sweeps behind the MoE block_m and MLA form rules in dispatch.rs. D 4096, I 256 (the host's sizes; the rule depends on the rows per expert and the pair count) -->

# MoE block_m and MLA form: search evidence

## MoE block_m (stage 1 + stage 2, us; bm64 = the 64-row tile, bm128 = the 128-row tile)

| E | M topk | r = M topk / E | fp8 bm64 | fp8 bm128 | bf16 bm64 | bf16 bm128 | winner |
|--:|--:|--:|--:|--:|--:|--:|---|
| 16 | 2048 | 128 | 54.6 | 67.8 | 67.2 | 85.1 | bm64 (+24 / +27 %) |
| 16 | 2560 | 160 | 58.0 | 72.6 | 71.3 | 89.1 | bm64 (+25 / +25 %) |
| 16 | 3072 | 192 | 66.0 | 74.0 | 76.2 | 91.2 | bm64 (+12 / +20 %) |
| 32 | 4096 | 128 | 81.5 | 87.7 | 99.3 | 107.8 | bm64 (+7.6 / +8.6 %) |
| 32 | 5120 | 160 | 93.8 | 105.8 | 117.3 | 122.5 | bm64 (+12.9 / +4.4 %) |
| 32 | 6144 | 192 | 107.1 | 110.6 | 137.1 | 127.3 | mixed (fp8 bm64 +3.3 %, bf16 bm128 +7.7 %) |
| 32 | 7168 | 224 | 122.6 | 114.7 | 160.4 | 132.6 | bm128 |
| 32 | 8192 | 256 | 147.3 | 125.5 | 197.3 | 147.6 | bm128 |
| 64 | 8192 | 128 | 157.4 | 139.6 | 225.8 | 188.9 | bm128 (+12.7 / +19 %) |
| 64 | 9216 | 144 | 170.2 | 163.3 | 243.4 | 217.9 | bm128 |
| 64 | 10240 | 160 | 182.2 | 181.3 | 258.2 | 231.2 | bm128 |
| 64 | 11264 | 176 | 192.8 | 191.2 | 270.8 | 239.1 | bm128 |
| 64 | 12288 | 192 | 218.9 | 199.5 | 297.4 | 245.7 | bm128 (+9.7 / +21 %) |

Rule (`dispatch::moe_block_m(m, topk, E, I, cus)`): bm64 when pairs < 128 E (the previous rule), or in the measured band pairs <= 192 E with at most 5120 pairs in total AND the stage-1 grid of the 128-row form (E x blocks per expert x ceil(2 I / 128))
fits the CUs (r = 128 exactly: half of them). The grid test is what the first attempt (without I) got wrong: perf_ab on E16 D2048 I768 bf16 M320 / M384 showed the 64-row form 35 % SLOWER there (I768: 12 N tiles x 32 blocks = 384 CTAs > 256, the 128-row form already fills the
CUs), so the rule now takes `inter` and `cus`; unknown `I` keeps the old rule. The previous rule (r < 128 only) left 4 - 25 % on the table for r = 128 .. 192 at E 16 / 32 with I256.

Gate (`perf_ab.py --families moe,mla`, old = the tree before, node 53 / GPU 1, 5 repetitions): MLA 13 cases geomean **1.230**, all OK, 7 IMPROVED (1.21 - 1.94x); MoE 26 stage rows geomean **1.031**: fp8 / fp4 band shapes IMPROVED (stage 1 +11 - 16 %, stage 2 +5 - 11 %; fp4 E16 M384 and E32 M512 included although the sweep
measured fp8 / bf16 only), bf16 E16 I768 unchanged. One row flags (`moe/bf16/M384D2048I768E16K8#stage1` 0.945 / 0.948 in two runs): a measurement artifact, not a code change: both sides run the byte-identical code objects (md5 of `gemm_moe1_bf16_silu.co` / `gemm_moe2_bf16.co` equal) and the same block_m 128,
the A/A control (old = new) gives 1.001 (allowed drop 11.3 %), the swapped control (old = new tree, new = old tree) gives 0.957 with the roles reversed (the first-listed side is the faster one), and the forward re-run gives 0.989 (OK).

## MLA decode: the form (per-wave p64 vs head-parallel hp_p64, best S each, us)

| H | B | ctx | per-wave | hp | winner |
|--:|--:|--:|--:|--:|---|
| 16 | 1 | 1024 | 35.8 | 18.0 | hp +99 % |
| 16 | 4 | 1024 | 39.2 | 18.9 | hp +107 % |
| 16 | 16 | 1024 | 43.2 | 22.2 | hp +95 % |
| 16 | 1 | 2048 | 36.1 | 23.9 | hp +51 % |
| 16 | 4 | 2048 | 39.4 | 24.7 | hp +59 % |
| 16 | 16 | 2048 | 41.7 | 27.2 | hp +53 % |
| 16 | 1 | 4096 | 40.4 | 29.4 | hp +37 % |
| 16 | 4 | 4096 | 43.4 | 35.6 | hp +22 % |
| 16 | 16 | 4096 | 47.2 | 38.6 | hp +22 % |
| 16 | 1 | 6144 | 44.7 | ~44.7 | tie |
| 16 | 16 | 6144 | 53.1 | 50.5 | hp +5 % |
| 16 | 1 / 4 / 16 / 64 | 8192 | 44.7 / 46.3 / 58.4 / 122.8 | worse | per-wave |
| 16 | 64 | 1024 .. 4096 | 36.7 / 47.0 / 68.4 | 36.4 / worse / worse | per-wave (tie at 1024) |
| 16 | 64 | 512 | 30.7 | 24.7 | hp +24 % |
| 32 | 1 / 16 | 2048 | 37.5 / 37.9 | 25.0 / 30.4 | hp +50 % / +25 % |
| 32 | 1 / 16 | 8192 | 46.1 / 67.9 | worse / 65.5 | per-wave / hp +3.6 % |
| 48 | 1 | 2048 | 38.0 | 26.4 | hp +44 % |
| 48 | 16 | 8192 | 109.9 | 71.2 | hp +54 % (not used by the rule: untested neighbours) |
| 64, 128 | 1 .. 64 | 2048 .. 8192 | hp is the best form in every measured shape (the recorded rule) |

Rule: head-parallel for H >= 64, or B <= 16 with ctx (the block table's capacity) <= 4096; per-wave otherwise (H16 at ctx >= 6144 at B1, every ctx at B64). The split S is unchanged (smallest power of two giving >= 256 CTAs, <= 16): S16 was the measured best at B <= 16.