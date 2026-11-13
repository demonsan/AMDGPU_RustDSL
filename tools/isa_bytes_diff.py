#!/usr/bin/env python3
"""Byte-level comparison of two disassemblies (llvm-objdump with encodings): instructions aligned in order (difflib on the text), then
every aligned pair whose ENCODING differs (same text, different bytes / length) is classed.

usage: isa_bytes_diff.py A.dis B.dis
"""
import difflib, re, sys, collections

def load(p):
    out = []
    for l in open(p):
        m = re.match(r'^\s+(\S.*?)\s*//\s*([0-9A-F]+):\s*([0-9A-F ]+)$', l.rstrip())
        if m:
            out.append((re.sub(r'\s+', ' ', m.group(1)), int(m.group(2), 16), m.group(3).split()))
    return out

a, b = load(sys.argv[1]), load(sys.argv[2])
ta = [re.sub(r'\b[A-Za-z_]*BB\d+|label_\w+', 'L', x[0]) for x in a]
tb = [re.sub(r'\b[A-Za-z_]*BB\d+|label_\w+', 'L', x[0]) for x in b]
sm = difflib.SequenceMatcher(a=ta, b=tb, autojunk=False)
c = collections.Counter()
ex = {}
size_a = size_b = 0
for blk in sm.get_matching_blocks():
    for k in range(blk.size):
        x, y = a[blk.a + k], b[blk.b + k]
        if x[2] != y[2]:
            key = (x[0].split()[0], len(x[2]), len(y[2]))
            c[key] += 1
            ex.setdefault(key, (x[0], ' '.join(x[2]), ' '.join(y[2])))
print(f'A {len(a)} instructions, {sum(4 * len(x[2]) for x in a)} bytes; B {len(b)}, {sum(4 * len(x[2]) for x in b)} bytes')
for (op, la, lb), n in c.most_common(20):
    print(f'{n:5d} {op:36s} words {la} -> {lb}   e.g. {ex[(op, la, lb)][0][:60]} | {ex[(op, la, lb)][1]} | {ex[(op, la, lb)][2]}')
