#!/usr/bin/env python3
"""Instruction-class profile of a gfx1250 kernel and of its hot loops (the innermost backward-branch ranges), from llvm-objdump output (.dis,
read with the gfx1250 parser: TDM words joined, branch targets as labels) or from our emitted assembly (.s). Counts only (no instruction text
is printed), so it can be run on vendor code objects.

    python tools/isa_loop_profile.py K.dis|K.s [--loops N]
"""
import collections, os, re, sys

CLASSES = [
    ('wmma', r'^v_wmma'), ('tdm', r'^tensor_(load|store)'), ('async_lds', r'^global_load_async_to_lds|^global_load_lds'),
    ('gload', r'^(global|buffer)_load'), ('gstore', r'^(global|buffer)_store'), ('ds_read', r'^ds_(load|read)'),
    ('ds_write', r'^ds_(store|write)'), ('smem', r'^s_(load|buffer_load)'), ('barrier', r'^s_barrier'),
    ('wait_mem', r'^s_wait_(loadcnt|storecnt|dscnt|kmcnt|tensorcnt|asynccnt|xcnt|samplecnt|bvhcnt)|^s_waitcnt'),
    ('wait_alu', r'^s_wait_alu'), ('msb', r'^s_set_vgpr_msb'), ('v_nop', r'^v_nop'), ('s_nop', r'^s_nop'),
    ('branch', r'^s_(c)?branch'), ('prefetch', r'prefetch'), ('trans', r'^v_(exp|log|rcp|rsq|sqrt|sin|cos)_'),
    ('permlane', r'^v_permlane|^v_swap'), ('valu', r'^v_'), ('salu', r'^s_'),
]


def classify(op):
    for name, rx in CLASSES:
        if re.search(rx, op):
            return name
    return 'other'


def load_asm(path):
    """Our assembly: `label:` lines, instructions indented, branch targets are labels."""
    ins, pending = [], []
    for line in open(path, errors='replace'):
        line = line.rstrip('\n')
        m = re.match(r'^([.\w$]+):', line)
        if m:
            pending.append(m.group(1))
            continue
        m = re.match(r'^\s+([a-z_][a-z0-9_]*)\b(.*)$', line)
        if not m:
            continue
        op, rest = m.group(1), m.group(2)
        tgt = None
        if op.startswith(('s_branch', 's_cbranch')):
            t = re.match(r'\s*([.\w$]+)', rest)
            tgt = t.group(1) if t else None
        ins.append((op, pending, tgt))
        pending = []
    return ins


def load_dis(path):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import isa_lift_gfx1250 as P
    out, seen = [], None
    for sec, op, args in P.parse(path):
        labels = [sec] if sec != seen else []
        seen = sec
        tgt = args.split(' |||')[0].split(' words:')[0].strip() if op.startswith(('s_branch', 's_cbranch')) else None
        out.append((op, labels, tgt))
    return out


def main():
    path = sys.argv[1]
    nloops = int(sys.argv[sys.argv.index('--loops') + 1]) if '--loops' in sys.argv else 3
    ins = load_dis(path) if path.endswith('.dis') else load_asm(path)
    where = {}
    for k, (_, labels, _) in enumerate(ins):
        for l in labels:
            where[l] = k
    tot = collections.Counter(classify(op) for op, _, _ in ins)
    print(f'{os.path.basename(path)}: {len(ins)} instructions')
    print('  whole kernel:', ', '.join(f'{c} {n}' for c, n in tot.most_common()))
    loops = []
    for k, (op, _, tgt) in enumerate(ins):
        if tgt in where and where[tgt] <= k:
            loops.append((where[tgt], k))
    inner = [l for l in loops if not any(o != l and l[0] <= o[0] and o[1] <= l[1] for o in loops)]
    inner.sort(key=lambda l: -(l[1] - l[0]))
    for s, e in inner[:nloops]:
        c = collections.Counter(classify(ins[k][0]) for k in range(s, e + 1))
        w = max(c['wmma'], 1)
        print(f'  loop [{s}..{e}] {e - s + 1} instructions: ' + ', '.join(f'{x} {n}' for x, n in c.most_common()))
        print(f'     per WMMA: valu {c["valu"] / w:.2f}, salu {c["salu"] / w:.2f}, mem waits {c["wait_mem"] / w:.2f}, alu waits {c["wait_alu"] / w:.2f}, '
              f'msb {c["msb"] / w:.2f}, ds_read {c["ds_read"] / w:.2f}, nops {(c["v_nop"] + c["s_nop"]) / w:.2f}, tdm {c["tdm"] / w:.2f}, '
              f'barriers {c["barrier"] / w:.2f}')


if __name__ == '__main__':
    main()
