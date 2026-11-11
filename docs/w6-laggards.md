# W6: back-end Weaver features on the laggard kernels

Work item W6 of [`two-weavers-plan.md`](two-weavers-plan.md). The candidates are the families where we trail the vendor or have no reference: MLA, small-batch paged decode, attention backward, hd192 small grids, and the MXFP4 GEMMs. Every change is measured with [`tools/perf_ab.sh`](../tools/perf_ab.sh) ([`perf-ab.md`](perf-ab.md)), old = Main, same session. Only wins are adopted.

## 1. Lowering tokens (`movb64`, `jthread`, `cmpneg`) on every laggard family — neutral

Tried: the W5 tokens added to every kernel of the families (21 kernels change their md5). `perf_ab --families mla,decode,fmha_bwd,fmha_hd192`, release `amdc`, 5 reps, old = Main `972408ea`:

| family | cases | geomean new / old | verdict |
|---|--:|--:|---|
| decode | 6 | 1.000 | OK |
| fmha_bwd | 6 | 1.001 | OK |
| fmha_hd192 | 8 | 0.997 | OK |
| mla | 4 | 1.001 | OK |

No case moves beyond noise. **Not adopted.** These kernels have no rare-branch / multi-exit / zeroing hot spots for the tokens to change.

Vendor ratios from the same run show where the real gaps are:
- **hd192**, small grids: full 1×8×8×4096² 0.81×, causal 1×16×16×4096² 0.77×;
- **bwd causal**: we are 1.33× the vendor; bwd full ≈ 1.00×;
- decode / MLA: no vendor side in the harness.

## 2. hd192 dispatch rule variants — negative

Tried two rule changes:
- non-causal: key split S = 4 already at U ≤ CUs / 2 with ≥ 32 key tiles;
- causal: the key split S = 2 also at U = CUs with ≥ 32 key tiles.

`perf_ab --families fmha_hd192`, 7 reps:

| case | new / old |
|---|--:|
| full 1×8×8×4096² | **0.872 FAIL** |
| causal 1×16×16×4096² | **0.826 FAIL** |
| family geomean | 0.961 FAIL |

**Not adopted.** The current rule is right at these points, and the gap is in the kernels, not the choice.

## 3. Plan

The back-end Weaver orders **units**. decode16, MLA and the backward dq have none (no `Units` / Weaver capture), so back-end ordering cannot act on them yet. The default scheduler measured +3–4 % on decode16 and +7 % on MLA at some shapes, but breaks MLA correctness at B32H16C2048 / B4H16C16384; backend-llvm-tail is fixing that first. Next:
1. Express the decode16 and MLA bodies as Weaver units in today's emission order. This is md5-neutral: `Stmt::UnitMark` brackets do not change codegen.
2. Then a per-family scheduler / ordering policy, measured with perf_ab.