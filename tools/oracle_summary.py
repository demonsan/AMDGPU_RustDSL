#!/usr/bin/env python3
"""Summarise build/matrix.md (tests/hazard_oracle.rs) into hazard classes.
usage: oracle_summary.py matrix.md [--detail]"""
import re, sys, collections

def klass(name):
    n = name.split(':')[0]
    if n.startswith('VMfma'): return 'MFMA'
    if n in ('VExpF32', 'VRcpF32'): return 'VALU-trans'
    if n.startswith(('VCmp',)): return 'VALU-cmp(vcc)'
    if n in ('VAddCoU32', 'VAddcCoU32', 'VPtrAdd'): return 'VALU-carry(vcc)'
    if n == 'VCndmask': return 'VALU-cndmask(vcc)'
    if n == 'VReadFirstLane': return 'VALU-readfirstlane(->SGPR)'
    if n in ('VWriteSub', 'VReadSub'): return 'VALU-mov-sub'
    if n.startswith('VCvt'): return 'VALU-cvt'
    if n.startswith('V'): return 'VALU'
    if n.startswith('Lds'): return 'DS-' + ('read' if 'Read' in n else 'write')
    if n.startswith('GlobalLoad'): return 'VMEM-load'
    if n.startswith('GlobalStore'): return 'VMEM-store'
    if n.startswith('SLoad'): return 'SMEM-load'
    return 'SALU'

rows = []
for line in open(sys.argv[1]):
    if not line.startswith('| ') or line.startswith('| producer') or line.startswith('|---'):
        continue
    c = [x.strip() for x in line.strip().strip('|').split('|')]
    prod, cons, llvm, own = c[0], c[1], int(c[2]), int(c[3])
    slot = re.search(r'\(src(\d)\)', cons)
    cons_name = re.sub(r' \(src\d\)', '', cons)
    rows.append((prod, cons_name, slot.group(1) if slot else '-', llvm, own))

agg = collections.defaultdict(list)
for prod, cons, slot, llvm, own in rows:
    agg[(klass(prod), klass(cons))].append((prod, cons, slot, llvm, own))

print('class pair (producer -> consumer): LLVM nop-cycle range / own range, #pairs, #LLVM>own')
for k in sorted(agg):
    v = agg[k]
    ll = [x[3] for x in v]; ow = [x[4] for x in v]
    gt = sum(1 for x in v if x[3] > x[4])
    if max(ll) == 0 and max(ow) == 0:
        continue
    print(f'{k[0]:>28} -> {k[1]:<28} llvm {min(ll)}..{max(ll)}  own {min(ow)}..{max(ow)}  n={len(v)} llvm>own={gt}')
    if '--detail' in sys.argv:
        seen = set()
        for p, c, s, l, o in v:
            key = (p.split(':')[0], c.split(':')[0], s, l, o)
            if key in seen or l <= o: continue
            seen.add(key)
            print(f'      {p} -> {c} src{s}: llvm={l} own={o}')