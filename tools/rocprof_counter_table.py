#!/usr/bin/env python3
"""Per-kernel median counter table from rocprofv3 CSV trees (one `pmc_N` pass per counter line).

usage: rocprof_counter_table.py KERNEL_REGEX CASE=DIR [CASE=DIR ...]
DIR holds `pmc/pmc_*/*/*_counter_collection.csv` (and optionally `trace/*/*_kernel_trace.csv` for durations / resources).
Prints, per case, the median over dispatches of every counter of the first kernel whose name matches KERNEL_REGEX, side by side.
Counters from different passes (pmc_N) are different replays: only ratios of counters from the SAME pass are meaningful; across passes compare the per-dispatch medians as a diagnostic.
"""
import csv, glob, re, statistics, sys, collections

def load(d, rx):
    vals = collections.defaultdict(list)
    for f in glob.glob(f'{d}/pmc/pmc_*/*/*_counter_collection.csv'):
        for r in csv.DictReader(open(f)):
            if re.search(rx, r['Kernel_Name']):
                vals[r['Counter_Name']].append(float(r['Counter_Value']))
    return {k: statistics.median(v) for k, v in vals.items()}

def trace(d, rx):
    out = {}
    for f in glob.glob(f'{d}/trace/*/*_kernel_trace.csv'):
        t = []
        for r in csv.DictReader(open(f)):
            if re.search(rx, r['Kernel_Name']):
                t.append((int(r['End_Timestamp']) - int(r['Start_Timestamp'])) / 1000.0)
                out['res'] = {k: r.get(k) for k in ('Grid_Size_X', 'Grid_Size_Y', 'Grid_Size_Z', 'Workgroup_Size_X', 'VGPR_Count', 'SGPR_Count', 'LDS_Block_Size', 'Scratch_Size')}
        if t:
            out['us'] = statistics.median(t)
    return out

def main():
    rx = sys.argv[1]
    cases = [a.split('=', 1) for a in sys.argv[2:]]
    data = {c: load(d, rx) for c, d in cases}
    for c, d in cases:
        t = trace(d, rx)
        if t:
            print(f"{c}: {t.get('us', float('nan')):.2f} us median; {t.get('res')}")
    names = sorted(set().union(*[set(v) for v in data.values()]))
    print('%-36s' % 'counter', *['%14s' % c for c, _ in cases])
    for n in names:
        print('%-36s' % n, *['%14s' % ('-' if n not in data[c] else '%.4g' % data[c][n]) for c, _ in cases])

main()