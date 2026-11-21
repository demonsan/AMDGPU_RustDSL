#!/usr/bin/env python3
"""Summarise a pmc_run counter CSV: per kernel (name substring filter), the median over its dispatches of each counter (summed over the
counter's dimensions), with the dispatch count. `--all` lists every dispatch instead.
usage: pmc_sum.py <pass dir or csv> [kernel substring] [--all]"""
import csv, sys, os, collections, statistics
args = [a for a in sys.argv[1:] if a != "--all"]
every = "--all" in sys.argv
p = args[0]
if os.path.isdir(p):
    p = [os.path.join(dp, f) for dp, _, fs in os.walk(p) for f in fs if f.endswith("counter_collection.csv")][0]
pat = args[1] if len(args) > 1 else ""
d = collections.OrderedDict()
meta = {}
for r in csv.DictReader(open(p)):
    if pat not in r["Kernel_Name"]:
        continue
    k = int(r["Dispatch_Id"])
    meta[k] = (r["Kernel_Name"][:56], r["Grid_Size"], r["Workgroup_Size"], r.get("LDS_Block_Size", ""), r.get("VGPR_Count", ""))
    d.setdefault(k, collections.OrderedDict())
    d[k][r["Counter_Name"]] = d[k].get(r["Counter_Name"], 0.0) + float(r["Counter_Value"])
if every:
    for k, v in d.items():
        print(f"dispatch {k} {meta[k][0]} grid={meta[k][1]}: " + " ".join(f"{n}={x:.0f}" for n, x in v.items()))
    sys.exit(0)
by = collections.OrderedDict()
for k, v in d.items():
    by.setdefault(meta[k], []).append(v)
for m, vs in by.items():
    names = list(vs[0].keys())
    print(f"{m[0]} grid={m[1]} wg={m[2]} lds={m[3]} vgpr={m[4]}: {len(vs)} dispatch(es), median: " + " ".join(f"{n}={statistics.median(v.get(n, 0.0) for v in vs):.0f}" for n in names))