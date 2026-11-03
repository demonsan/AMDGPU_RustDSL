#!/usr/bin/env python3
"""MFMA-stream gap analysis from a rocprofv3 --att wave timeline (se*_sm*_sl*_wv*.json + code.json of the same dispatch dir).
Each instruction event is [time, type, stall, duration, code_line]. For every pair of consecutive MFMAs the gap above the steady-state
interval (the mode) is attributed to the instruction in between with the largest stall+duration (waitcnt / barrier / DMA issue / nop / VALU / ...).
usage: att_gaps.py <ui_output_dir> [wave_json] [top=12]"""
import json, sys, glob, collections
d = sys.argv[1]
wf = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].isdigit() else glob.glob(d + "/se0_sm0_sl0_wv0.json")[0]
top = int(sys.argv[-1]) if sys.argv[-1].isdigit() else 12
w = json.load(open(wf))["wave"]; code = json.load(open(d + "/code.json"))["code"]
ev = w["instructions"]
def text(e): return code[e[4]][0]
def klass(t):
    m = t.split()[0]
    if m.startswith("v_mfma"): return "MFMA"
    if m.startswith("s_waitcnt"):
        return "wait vm+lgkm" if ("vmcnt" in t and "lgkm" in t) else ("wait vmcnt" if "vmcnt" in t else "wait lgkmcnt")
    if m == "s_barrier": return "barrier"
    if m == "s_nop": return "s_nop"
    if m.startswith("buffer_load") or m.startswith("global_load"): return "DMA/load issue"
    if m.startswith("ds_read") or m.startswith("ds_write"): return "ds op issue"
    if m.startswith("v_accvgpr"): return "accvgpr"
    if m.startswith("v_"): return "VALU"
    if m.startswith("s_cbranch") or m == "s_branch": return "branch"
    if m.startswith("s_"): return "SALU"
    return m
mf = [i for i, e in enumerate(ev) if text(e).startswith("v_mfma")]
gaps = [ev[b][0] - ev[a][0] for a, b in zip(mf, mf[1:])]
cnt = collections.Counter(gaps); mode = cnt.most_common(1)[0][0]
print(f"{wf.split('/')[-1]}: {len(mf)} MFMAs, wave begin..end {w['begin']}..{w['end']} ({w['end']-w['begin']} cycles), steady MFMA interval (mode) = {mode} cycles")
print("gap histogram (top):", cnt.most_common(8))
tot = sum(g - mode for g in gaps if g > mode); span = ev[mf[-1]][0] - ev[mf[0]][0]
print(f"MFMA stream span {span} cycles; ideal {mode*(len(mf)-1)}; excess over ideal {tot} ({100*tot/span:.1f}% of the span)")
by_class = collections.defaultdict(lambda: [0, 0]); by_site = collections.defaultdict(lambda: [0, 0, ""])
for a, b in zip(mf, mf[1:]):
    g = ev[b][0] - ev[a][0]
    if g <= mode: continue
    # blame: the in-between instruction with the largest (stall + duration); gap measured on the issue of the next MFMA
    cand = [(ev[k][2] + ev[k][3], k) for k in range(a + 1, b + 1)]
    _, k = max(cand)
    t = text(ev[k]); c = klass(t)
    by_class[c][0] += g - mode; by_class[c][1] += 1
    s = by_site[ev[k][4]]; s[0] += g - mode; s[1] += 1; s[2] = t[:64]
print(f"{'blamed class':16s} {'excess cycles':>14s} {'% of excess':>11s} {'gaps':>6s}")
for c, (x, n) in sorted(by_class.items(), key=lambda kv: -kv[1][0]):
    print(f"{c:16s} {x:14d} {100*x/max(tot,1):10.1f}% {n:6d}")
print("top sites:")
for l, (x, n, t) in sorted(by_site.items(), key=lambda kv: -kv[1][0])[:top]:
    print(f"  excess={x:7d} gaps={n:5d} line={l:5d} {t}")