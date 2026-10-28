#!/usr/bin/env python3
# analyze.py run.txt tests.json -> table: per kernel, per waves: cycles/instr (median over waves) for each rep.
import sys, json, statistics as st
N = json.load(open(sys.argv[2]))
rows = {}
for line in open(sys.argv[1]):
    if not line.startswith("R "): continue
    head, vals = line.split("|")
    _, name, w, rep = head.split()
    per = []
    for v in vals.split():
        a, b, c, d = v.split("/")
        per.append((int(a), int(b), int(c), d))
    rows.setdefault((name, int(w)), []).append(per)
print("kernel waves | memtime/instr per rep (median over waves) | min..max across waves rep0 | mark/instr | realtime ticks(rep0) | chk(rep0 wave0)")
for (name, w), reps in rows.items():
    n = N[name]
    med = [st.median(p[0] for p in per) / n for per in reps]
    mn = [min(p[0] for p in per) / n for per in reps]
    mx = [max(p[0] for p in per) / n for per in reps]
    mark = [st.median(p[2] for p in per) / n for per in reps]
    print("%-34s w=%d | %s | %.2f..%.2f | mark %s | real %d | %s" % (name, w, " ".join("%.3f" % m for m in med), mn[0], mx[0],
          " ".join("%.2f" % m for m in mark) if mark[0] else "-", reps[0][0][1], reps[0][0][3]))
