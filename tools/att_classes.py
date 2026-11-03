#!/usr/bin/env python3
"""Aggregate a rocprofv3 --att stats csv (per-instruction hitcount / latency / stall / idle) by instruction class.
usage: att_classes.py stats.csv [more.csv ...]"""
import csv, sys, collections, re

def cls(ins):
    m = ins.split()[0] if ins.split() else ""
    if m.startswith("v_mfma") or m.startswith("v_smfmac"): return "MFMA"
    if m.startswith("v_accvgpr"): return "accvgpr"
    if m.startswith("v_"): return "VALU"
    if m.startswith("s_waitcnt"):
        return "waitcnt(lgkm)" if "lgkm" in ins and "vmcnt" not in ins else ("waitcnt(vm)" if "vmcnt" in ins and "lgkm" not in ins else "waitcnt(both)")
    if m == "s_barrier": return "s_barrier"
    if m == "s_nop": return "s_nop"
    if m == "s_setprio": return "s_setprio"
    if m.startswith("s_cbranch") or m == "s_branch" or m == "s_endpgm": return "branch"
    if m.startswith("s_load") or m.startswith("s_buffer"): return "SMEM"
    if m.startswith("s_"): return "SALU"
    if m.startswith("ds_read") or m.startswith("ds_load"): return "ds_read"
    if m.startswith("ds_write") or m.startswith("ds_store"): return "ds_write"
    if m.startswith("buffer_load") or m.startswith("global_load"): return "vmem_load(DMA)"
    if m.startswith("buffer_store") or m.startswith("global_store"): return "vmem_store"
    return "other"

for f in sys.argv[1:]:
    agg = collections.defaultdict(lambda: [0, 0, 0, 0, 0]); rows = []
    for r in csv.DictReader(open(f)):
        ins = r["Instruction"]
        if ins.startswith(";") or int(r["Hitcount"]) == 0: continue
        h, l, s, i = int(r["Hitcount"]), int(r["Latency"]), int(r["Stall"]), int(r["Idle"])
        a = agg[cls(ins)]; a[0] += 1; a[1] += h; a[2] += l; a[3] += s; a[4] += i
        rows.append((l, h, s, i, ins))
    tl = sum(a[2] for a in agg.values()) or 1
    print(f"== {f}: instructions executed {sum(a[1] for a in agg.values())}, total latency {tl}")
    print(f"{'class':16s} {'static':>6s} {'hits':>8s} {'latency':>10s} {'%lat':>6s} {'stall':>9s} {'idle':>9s} {'lat/hit':>8s}")
    for k, a in sorted(agg.items(), key=lambda kv: -kv[1][2]):
        print(f"{k:16s} {a[0]:6d} {a[1]:8d} {a[2]:10d} {100*a[2]/tl:5.1f}% {a[3]:9d} {a[4]:9d} {a[2]/max(a[1],1):8.1f}")
    print("top instructions by latency:")
    for l, h, s, i, ins in sorted(rows, reverse=True)[:12]:
        print(f"  lat={l:8d} hits={h:6d} stall={s:7d} idle={i:7d}  {ins[:70]}")