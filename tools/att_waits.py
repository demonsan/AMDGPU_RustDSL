#!/usr/bin/env python3
"""ATT wait-site report: every s_waitcnt / s_barrier with its stall, hits and the instructions around it (code order).
usage: att_waits.py stats.csv [min_stall_total=500]"""
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if not r["Instruction"].startswith(";")]
minst = int(sys.argv[2]) if len(sys.argv) > 2 else 500
def short(r):
    return f'{r["Instruction"][:58]:58s} hits={r["Hitcount"]:>5s} lat={r["Latency"]:>7s} stall={r["Stall"]:>7s}'
tot = {"lgkm": 0, "vm": 0, "both": 0, "barrier": 0}
for i, r in enumerate(rows):
    ins = r["Instruction"]; h = int(r["Hitcount"]); s = int(r["Stall"])
    if h == 0: continue
    if ins.startswith("s_waitcnt"):
        k = "both" if ("vmcnt" in ins and "lgkm" in ins) else ("vm" if "vmcnt" in ins else "lgkm")
    elif ins.startswith("s_barrier"): k = "barrier"
    else: continue
    tot[k] += s
    if s < minst: continue
    print(f"--- {k} wait: stall total {s}, per hit {s/h:.1f}")
    for j in range(max(0, i-4), min(len(rows), i+3)):
        mark = ">>" if j == i else "  "
        print(f"  {mark} {short(rows[j])}")
print("totals (stall cycles):", tot)
# ds_read / DMA instruction cost and the MFMA right after a wait
import collections
cls = collections.defaultdict(lambda: [0, 0, 0])
for r in rows:
    h = int(r["Hitcount"])
    if h == 0: continue
    m = r["Instruction"].split()[0]
    k = "ds_read" if m.startswith("ds_read") else ("dma" if (m.startswith("buffer_load") and "lds" in r["Instruction"]) else None)
    if k: c = cls[k]; c[0] += h; c[1] += int(r["Latency"]); c[2] += int(r["Stall"])
for k, c in cls.items(): print(f"{k}: hits={c[0]} latency={c[1]} stall={c[2]} lat/hit={c[1]/max(c[0],1):.1f}")