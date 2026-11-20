def edit(path, fn):
    raw = open(path, encoding='utf-8', newline='').read(); crlf = '\r\n' in raw
    t = fn(raw.replace('\r\n', '\n'))
    open(path, 'w', encoding='utf-8', newline='').write(t.replace('\n', '\r\n') if crlf else t)


def st(t):
    marker = '#### Per-phase microbenchmarks of the window kernel'
    i = t.index(marker)
    add = '''### MLA structure study: summary (e08, 2026-11-20)

What was measured, in the order of the subsections below, and what it says about how to structure the decode kernel. All with stub data and loop shader cycles unless "in situ".

| question | answer | where |
|---|---|---|
| where does a window of the one-barrier kernel (`_w1i`) go? | per 128-token window, compute-only: S 0.125, PV 0.132, O rescale 0.083, softmax 0.295 (44 %), barrier 0.025 us; the blocks do not overlap (0.68 us combined, 0.96 us in situ; the replica is 0.379 us in situ) | per-phase microbenchmarks |
| does interleaving the softmax VALU between the WMMAs hide it? | no: one wave does not overlap VALU with WMMA in any order I can express (-8 % at best); the levers are fewer instructions: lean softmax arithmetic (-0.095 us), lazy rescale (-0.045 us) | "Does the interleave hide the softmax?" |
| can a second wave on the SIMD hide it? | yes, if the waves differ in kind: a WMMA / LDS wave and a VALU wave per SIMD run at max (576 / 455 cycles vs 690 in one wave); two waves of the same kind serialize | "Two waves per SIMD" |
| two roles (MMA waves + softmax waves, S / P through LDS), compute-only | 0.718 -> 0.447 us per window (-37 %); the barrier is not the cost (two windows per barrier: 0.446); with the fragments free 0.336 - 0.354; three roles (S / PV / softmax waves) 0.437; the hand-off skeleton alone is 0.205 | "Two-role prototype" and follow-ups |
| the two-role kernel in situ (`mla_g1250_qh16_w1r`, on kernel-opt's `W1Parts`) | correct (hash identical to `_w1p`), -1 % (4 x 512) to -12 % (128 x 8192) vs the same code in one role, not -37 %: the in-situ window also pays the stage, page ids, rescale, the second barrier (forced by the 4-slot ring) and the pad windows | `w1r` section |
| 8-wave `_ws` (pairs of an MMA wave and a softmax wave per SIMD; half-page waves) | pair -9 % per page step, half-page waves +20 %: the MMA wave's own chain (1,660 of 1,825 cycles) is the long pole, and a pair that splits a page across waves needs 3 live pages per pair (LDS holds 8) | "An 8-wave `_ws`" |

**Why `_ws` stays the best structure** (it wins every size from 4 x 1000 up, `_ws2` at 4 x 512): its waves own whole pages, so there is no cross-wave exchange, no barrier per window, and a private 2-slot ring per wave that fits the LDS with the TDM latency hidden (stage + TDM wait is 130 of ~2,230 cycles per page; per page: S ~610, lean softmax ~490, lazy rescale ~435 when taken / ~100 skipped, PV ~565). Every role-split variant pays for its
hand-off (an LDS round trip per hop, one or two barriers per window, extra live pages in the ring) about as much as the overlap gains, and the MMA waves' fragment-load chains (not the WMMA issue, 68 WMMA are ~370 cycles) stay on the critical path. What did help the family: the lean softmax arithmetic (packed ops, mask once, max on the
raw scores) and the lazy rescale, both already in `_ws` / `_w1v`; the remaining levers are VALU throughput (backend's opt-in VOPD dual issue: 1.7 - 1.9 x on VALU microbenchmarks, -1 .. -5 % on `_ws`) and fewer LDS fragment instructions per page. (kernel-opt's stamps of the out-of-loop tail pages first looked like a 24 - 33 % merge-barrier wait; a variant without
any out-of-loop code (`_wsnt`) is 1 - 8 % slower, so that was a stamping artefact and not a cost.) Tools: `mla_g1250_qh16_ph_<mask>` (the phase kernels), `mla_g1250_qh16_roles[_mask]`, `mla_g1250_qh16_w1r`, `mla_g1250_qh16_wsp_<mask>`, `probe_g1250_waves_<hex>`; runners `tools/mi450/mla_phase.sh`, `mla_roles.sh`, `mla_w1r.sh`, `mla_w1r_pair.sh`, `waves.sh`.

'''
    return t[:i] + add + t[i:]


edit('docs/gfx1250-status.md', st)
print('ok')
