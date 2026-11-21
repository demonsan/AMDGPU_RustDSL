#!/usr/bin/env python3
"""Sum a pmc_run counter CSV per dispatch: kernel name, grid, counters (name = summed value over the counter's dimensions).
usage: pmc_sum.py <pass dir or csv> [kernel substring]"""
import csv, sys, os, collections
p = sys.argv[1]
if os.path.isdir(p):
    p = [os.path.join(dp, f) for dp, _, fs in os.walk(p) for f in fs if f.endswith("counter_collection.csv")][0]
pat = sys.argv[2] if len(sys.argv) > 2 else ""
d = collections.OrderedDict()
for r in csv.DictReader(open(p)):
    if pat not in r["Kernel_Name"]:
        continue
    k = (int(r["Dispatch_Id"]), r["Kernel_Name"][:48], r["Grid_Size"], r["Workgroup_Size"], r.get("LDS_Block_Size", ""), r.get("VGPR_Count", ""))
    d.setdefault(k, collections.OrderedDict())
    d[k][r["Counter_Name"]] = d[k].get(r["Counter_Name"], 0.0) + float(r["Counter_Value"])
for k, v in d.items():
    print(f"dispatch {k[0]} {k[1]} grid={k[2]} wg={k[3]} lds={k[4]} vgpr={k[5]}: " + " ".join(f"{n}={x:.0f}" for n, x in v.items()))