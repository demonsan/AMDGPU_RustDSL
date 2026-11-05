#!/usr/bin/env python3
"""Static MFMA operand-reuse / forwarding analysis of a gfx950 assembly listing.

Usage: mfma_reuse.py <file.s|objdump.txt> [--label NAME] [--pairs N]

For every pair of MFMA-class instructions that are consecutive in program order
(v_mfma*, v_smfmac*), classify how the second one's operands relate to the first:

  C=D(prev)   srcC is exactly the previous MFMA's vDst  (back-to-back accumulate chain:
              the case the hardware's srcC forwarding / dependent-MFMA rules cover)
  C~D(prev)   srcC overlaps but differs from previous vDst (partial dependency)
  A same      srcA register range identical to the previous MFMA's srcA
  B same      srcB register range identical
  A/B RAW     srcA or srcB overlaps the previous vDst (read-after-write on a result)
  indep       none of the above

Also reports the instruction distance between the pair and, per accumulator, the
chain structure (how many MFMAs accumulate into the same vDst back-to-back vs
interleaved with other accumulators).
"""
import re
import sys
from collections import Counter, defaultdict

REG = re.compile(r'^([va])\[(\d+):(\d+)\]$|^([va])(\d+)$')
MNEM = re.compile(r'^\s*(?:[0-9a-fA-F]+:\s+)?(?:[0-9a-fA-F]{8}\s+)*\s*(v_mfma\S*|v_smfmac\S*|[a-z_][a-z0-9_]*)\b(.*)$')


def parse_reg(tok):
    tok = tok.strip()
    m = REG.match(tok)
    if not m:
        return None
    if m.group(1):
        f, lo, hi = m.group(1), int(m.group(2)), int(m.group(3))
    else:
        f, lo = m.group(4), int(m.group(5))
        hi = lo
    return (f, lo, hi)


def overlap(a, b):
    return a and b and a[0] == b[0] and not (a[2] < b[1] or b[2] < a[1])


def clean(line):
    line = line.split(';')[0].split('//')[0]
    return line.rstrip()


def load(path):
    insts = []
    for raw in open(path, encoding='utf-8', errors='replace'):
        line = clean(raw)
        s = line.strip()
        if not s or s.endswith(':') or s.startswith('.') or s.startswith('<'):
            continue
        m = MNEM.match(line)
        if not m:
            continue
        mn, rest = m.group(1), m.group(2)
        if not (mn.startswith('v_') or mn.startswith('s_') or mn.startswith('ds_') or mn.startswith('buffer_') or mn.startswith('global_')):
            continue
        insts.append((mn, rest))
    return insts


def mfma_ops(mn, rest):
    # operands: dst, A, B, C[, scaleA, scaleB | idx]; strip modifiers
    ops = [o.strip() for o in rest.split(',')]
    ops = [o.split(' ')[0] for o in ops if o]
    regs = [parse_reg(o) for o in ops]
    if mn.startswith('v_smfmac'):
        # v_smfmac dst(=C), A, B, idx
        d, a, b = regs[0], regs[1], regs[2]
        c = d
    else:
        d, a, b, c = regs[0], regs[1], regs[2], regs[3] if len(regs) > 3 else None
    return d, a, b, c


def fmt(r):
    if r is None:
        return 'imm'
    f, lo, hi = r
    return f'{f}{lo}' if lo == hi else f'{f}[{lo}:{hi}]'


def main():
    args = sys.argv[1:]
    path = args[0]
    show = 0
    if '--pairs' in args:
        show = int(args[args.index('--pairs') + 1])
    insts = load(path)
    mf = [(i, mn, *mfma_ops(mn, rest)) for i, (mn, rest) in enumerate(insts)
          if mn.startswith('v_mfma') or mn.startswith('v_smfmac')]
    print(f'file: {path}')
    print(f'instructions: {len(insts)}   MFMA-class: {len(mf)}')
    kinds = Counter(mn for _, mn, *_ in mf)
    for k, v in kinds.most_common():
        print(f'  {v:5d}  {k}')
    files = Counter()
    for _, mn, d, a, b, c in mf:
        files[f'D:{d[0] if d else "?"} A:{a[0] if a else "?"} B:{b[0] if b else "?"} C:{c[0] if c else "imm"}'] += 1
    print('operand files (D/A/B/C):')
    for k, v in files.most_common():
        print(f'  {v:5d}  {k}')

    cls = Counter()
    dist = Counter()
    examples = defaultdict(list)
    for (i0, m0, d0, a0, b0, c0), (i1, m1, d1, a1, b1, c1) in zip(mf, mf[1:]):
        gap = i1 - i0 - 1
        dist[min(gap, 9)] += 1
        tags = []
        if c1 is not None and d0 is not None and c1 == d0:
            tags.append('C=D(prev)')
        elif overlap(c1, d0):
            tags.append('C~D(prev)')
        if a1 == a0:
            tags.append('A same')
        if b1 == b0:
            tags.append('B same')
        if overlap(a1, d0) or overlap(b1, d0):
            tags.append('A/B RAW on prev D')
        key = ' + '.join(tags) if tags else 'indep'
        cls[key] += 1
        if len(examples[key]) < 2:
            examples[key].append(f'{m0} {fmt(d0)},{fmt(a0)},{fmt(b0)},{fmt(c0)}  ->  {m1} {fmt(d1)},{fmt(a1)},{fmt(b1)},{fmt(c1)}  (gap {gap})')
    n = max(1, len(mf) - 1)
    print(f'\nconsecutive MFMA pairs: {len(mf) - 1}')
    for k, v in cls.most_common():
        print(f'  {v:5d} ({100 * v / n:5.1f}%)  {k}')
        for e in examples[k]:
            print(f'           e.g. {e}')
    print('instructions between consecutive MFMAs (9 = 9+):')
    print('  ' + '  '.join(f'{g}:{dist[g]}' for g in range(10)))

    # accumulator chain structure: run lengths of identical vDst in program order
    runs = Counter()
    prev, run = None, 0
    for _, _, d, *_ in mf:
        if d == prev:
            run += 1
        else:
            if prev is not None:
                runs[run] += 1
            prev, run = d, 1
    if prev is not None:
        runs[run] += 1
    print('back-to-back runs into the same vDst (run length: count):')
    print('  ' + '  '.join(f'{k}:{v}' for k, v in sorted(runs.items())))
    # srcC == own vDst (in-place accumulate) vs separate C
    inplace = sum(1 for _, _, d, a, b, c in mf if c == d)
    print(f'in-place accumulate (srcC == vDst): {inplace}/{len(mf)}')


if __name__ == '__main__':
    main()
