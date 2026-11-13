#!/usr/bin/env python3
"""Compare branch TARGETS of two aligned disassemblies: for each aligned branch pair, the instructions at the two targets (the next N,
labels ignored) must match; prints the mismatches.  usage: isa_branch_targets.py A.dis B.dis [N]"""
import difflib, re, sys

def load(p):
    ins, lab = [], {}
    for l in open(p):
        m = re.match(r'^([0-9a-f]+) <([^>]+)>:', l)
        if m:
            lab[m.group(2)] = int(m.group(1), 16)
            continue
        m = re.match(r'^\s+(\S.*?)\s*//\s*([0-9A-F]+):', l)
        if m:
            ins.append((re.sub(r'\s+', ' ', m.group(1)), int(m.group(2), 16)))
    return ins, lab

def tgt(ins, lab, i, n, depth=0):
    t = ins[i][0].split()
    if len(t) < 2:
        return None
    addr = lab.get(t[1])
    if addr is None:
        return None
    k = next((j for j, (_, a) in enumerate(ins) if a == addr), None)
    if k is None:
        return None
    # a target that only jumps on is the same edge as its destination (forwarding)
    if ins[k][0].startswith('s_branch ') and depth < 8:
        return tgt(ins, lab, k, n, depth + 1)
    # compare with branch operands normalised
    return [re.sub(r'(s_c?branch\w*) \S+', r'\1', x) for x, _ in ins[k:k + n]]

(a, la), (b, lb) = load(sys.argv[1]), load(sys.argv[2])
n = int(sys.argv[3]) if len(sys.argv) > 3 else 6
norm = lambda s: re.sub(r'(s_c?branch\w*) \S+', r'\1', s)
sm = difflib.SequenceMatcher(a=[norm(x) for x, _ in a], b=[norm(x) for x, _ in b], autojunk=False)
bad = 0
for blk in sm.get_matching_blocks():
    for k in range(blk.size):
        i, j = blk.a + k, blk.b + k
        if a[i][0].startswith(('s_cbranch', 's_branch')):
            ta, tb = tgt(a, la, i, n), tgt(b, lb, j, n)
            if ta != tb:
                bad += 1
                print(f'MISMATCH at A[{i}] {a[i][0]}  /  B[{j}] {b[j][0]}')
                print('   A ->', ta)
                print('   B ->', tb)
print('branch target mismatches (by content):', bad)
# by POSITION: the aligned image of the reference target must be the build target (after following unconditional jumps on both sides)
amap = {}
for blk in sm.get_matching_blocks():
    for k in range(blk.size):
        amap[blk.a + k] = blk.b + k
def tindex(ins, lab, i, depth=0):
    t = ins[i][0].split()
    addr = lab.get(t[1]) if len(t) > 1 else None
    if addr is None:
        return 'END'
    k = next((j for j, (_, x) in enumerate(ins) if x == addr), None)
    if k is not None and ins[k][0].startswith('s_branch ') and depth < 8:
        return tindex(ins, lab, k, depth + 1)
    return k
def fwd_b(k):
    # the build target, through the build's own forwarding jumps
    return k
pbad = 0
for i, j in sorted(amap.items()):
    if a[i][0].startswith(('s_cbranch', 's_branch')):
        ti, tj = tindex(a, la, i), tindex(b, lb, j)
        if ti == 'END' or tj == 'END':
            continue
        if amap.get(ti) != tj:
            pbad += 1
            print(f'POSITION MISMATCH A[{i}] {a[i][0]} -> A[{ti}] (image {amap.get(ti)})  /  B[{j}] {b[j][0]} -> B[{tj}]')
print('branch target mismatches (by position):', pbad)
