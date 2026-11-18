#!/usr/bin/env python3
"""TRANS co-execution probe shaping (tools/mi450/trans_coexec.sh): in the emitter's gfx1250 assembly, move the first reader of every TRANS
result right behind its `v_nop` (swapping it with independent instructions in between, register-checked), so the reader would be at
distance 0 without the v_nop. Then write one variant per reader class with only that class's v_nops removed.

usage: trans_shape.py ref.s outdir      -> outdir/shaped.s (reference, v_nops kept), outdir/no_<class>.s, prints the site histogram
"""
import re, sys, collections

TRANS = re.compile(r'^\s*v_(exp|log|rcp|rsq|sqrt|sin|cos)_f32')

def regs(op):
    out = set()
    for m in re.finditer(r'\bv\[(\d+):(\d+)\]|\bv(\d+)\b', op):
        if m.group(3) is not None:
            out.add(int(m.group(3)))
        else:
            out.update(range(int(m.group(1)), int(m.group(2)) + 1))
    return out

def split(line):
    t = line.strip()
    if ' ' not in t:
        return t, set(), set()
    mn, rest = t.split(' ', 1)
    ops = [o.strip() for o in rest.split(',')]
    if not mn.startswith('v_') or not ops:
        return mn, set(), set()
    d = regs(ops[0])
    s = set().union(*[regs(o) for o in ops[1:]]) if len(ops) > 1 else set()
    return mn, d, s

def klass(mn):
    mn = re.sub(r'_e(32|64)$', '', mn)
    if mn.startswith('v_pk_'): return 'vop3p'
    if mn in ('v_fma_f32', 'v_fmac_f32'): return 'fma'
    if mn.startswith('v_max') or mn.startswith('v_min'): return 'maxmin'
    if mn in ('v_add_f32', 'v_sub_f32', 'v_subrev_f32'): return 'add'
    if mn == 'v_mul_f32': return 'mul'
    if mn.startswith('v_mov'): return 'mov'
    return 'other'

def main():
    src, out = sys.argv[1], sys.argv[2]
    L = open(src).read().split('\n')
    sites = []   # (index of the v_nop, class)
    i = 0
    while i < len(L):
        if TRANS.match(L[i]) and i + 1 < len(L) and L[i + 1].strip() == 'v_nop':
            tmn, td, ts = split(L[i])
            # the first reader of the TRANS result within the next 14 instructions (same block)
            j = i + 2
            mid = []
            moved = False
            while j < len(L) and j < i + 16:
                s = L[j].strip()
                if not s.startswith('v_') or s.startswith('v_nop'):
                    break
                mn, d, sr = split(L[j])
                if sr & td:
                    # movable to right after the v_nop: no dependence on the instructions between
                    if all(not (sr & md) and not (d & (ms | md)) for (_, md, ms) in mid) and not (d & ts):
                        line = L.pop(j)
                        L.insert(i + 2, line)
                        sites.append((i + 1, klass(mn), tmn))
                        moved = True
                    break
                if d & td:
                    break
                mid.append((mn, d, sr))
                j += 1
            if not moved:
                nmn, nd, ns = split(L[i + 2]) if i + 2 < len(L) else ('', set(), set())
                kind = klass(nmn) + ('-raw' if ns & td else '-war')
                sites.append((i + 1, kind, tmn))
        i += 1
    open(f'{out}/shaped.s', 'w').write('\n'.join(L))
    hist = collections.Counter((k, re.sub(r'_e(32|64)$', '', t)) for _, k, t in sites)
    for (k, t), n in sorted(hist.items()):
        print(f'site {k:12s} {t:14s} {n}')
    for k in sorted(set(k for _, k, _ in sites)):
        drop = {idx for idx, kk, _ in sites if kk == k}
        open(f'{out}/no_{k}.s', 'w').write('\n'.join(l for n, l in enumerate(L) if n not in drop))
        print(f'class {k}: {len(drop)} v_nops removed')

main()