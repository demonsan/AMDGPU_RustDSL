#!/usr/bin/env python3
"""Read GV-TILE-STALE findings back onto the tile plan: per (input block, instruction, tile, slot rule, version rule) the set of
(latest published - expected) version differences the simulator saw. A group with ONE difference d means the rule's version is off by
a constant d at that place; several differences mean the place reads tiles of different relative age (a real protocol question).

usage: tile_fit.py IR GV_REPORT        (IR.markers from isa_lift.py --tile-plan)
"""
import collections, re, subprocess, sys, os

ir, rep = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ir_flat import flat
line_of = {k: ln for k, ln, _ in flat(ir)}
info = {}
for l in open(ir + '.markers'):
    ln, blk, op, tile, slot, ver = l.rstrip('\n').split('\t')
    info[int(ln)] = (blk, op, tile, slot, ver)
groups = collections.defaultdict(collections.Counter)
other = collections.Counter()
for l in open(rep):
    m = re.search(r'stmt (\d+): read of `(\w+)`\[(\d+)\] expects version (-?\d+) but the latest published version at phase \d+ is (-?\d+)', l)
    if m:
        st = int(m.group(1))
        key = info.get(line_of.get(st), ('?', '?', m.group(2), '?', '?'))
        groups[key][int(m.group(5)) - int(m.group(4))] += 1
        continue
    m = re.search(r'\[(GV-[A-Z-]+)[^\]]*\] Error: stmt (\d+)', l)
    if m:
        st = int(m.group(2))
        key = info.get(line_of.get(st), ('?',))
        other[(m.group(1),) + key[:3]] += 1
for k, c in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
    print('STALE', 'block', k[0], k[1], 'tile', k[2], 'slot', k[3], 'version', k[4], '-> latest-expected', dict(c))
for k, n in sorted(other.items(), key=lambda kv: -kv[1])[:30]:
    print('OTHER', n, k)
