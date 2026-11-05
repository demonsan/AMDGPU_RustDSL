#!/usr/bin/env python3
"""Compare a kernel md5 list against a baseline: python3 tools/md5_compare.py <baseline.txt> <new.txt> [--expect name=md5 ...]

Lines are `<kernel> <md5-prefix>` (`FAILED` = the kernel no longer builds / no longer exists). Prints the kernels whose md5 differs from the
baseline (or from an --expect override), the removed ones, and a summary; exit 1 if an unexpected difference is found.
"""
import sys

def load(p):
    d = {}
    for line in open(p):
        f = line.split()
        if len(f) >= 2:
            d[f[0]] = f[1]
    return d

base, new = load(sys.argv[1]), load(sys.argv[2])
expect = dict(a.split("=", 1) for a in sys.argv[3:] if "=" in a)
n = len(min(base.values(), key=len)) if base else 12
kept = removed = same = 0
bad = []
for k, b in base.items():
    v = new.get(k, "FAILED")
    if v == "FAILED":
        removed += 1
        continue
    kept += 1
    want = expect.get(k, b)
    if v[:len(want)] == want[:len(v)]:
        same += 1
    else:
        bad.append(f"DIFF {k}: baseline {b} expected {want} now {v}")
for l in bad:
    print(l)
print(f"kept {kept} identical {same} differing {len(bad)} removed {removed}")
sys.exit(1 if bad else 0)
