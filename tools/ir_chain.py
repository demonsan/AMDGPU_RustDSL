#!/usr/bin/env python3
"""Print the definition chain of IR values in an irv1 file (debugging gpu_verify findings on lifted kernels).
usage: ir_chain.py <file.ir> <value id>... [--depth N]"""
import re, sys
args = sys.argv[1:]
depth = 7
if "--depth" in args:
    i = args.index("--depth"); depth = int(args[i + 1]); del args[i:i + 2]
path, vals = args[0], [int(x) for x in args[1:]]
defs = {}
for l in open(path):
    m = re.match(r"\s+%v(\d+): \S+ = (.*)", l)
    if m:
        defs[int(m.group(1))] = m.group(2)
seen = set()
def show(v, d):
    if v in seen or d > depth:
        return
    seen.add(v)
    print("  " * d + f"v{v} = {defs.get(v, '?')[:160]}")
    for x in re.findall(r"Val\((\d+)\)", defs.get(v, "")):
        show(int(x), d + 1)
for v in vals:
    show(v, 0)
