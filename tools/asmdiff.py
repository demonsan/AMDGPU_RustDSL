#!/usr/bin/env python3
"""Instruction-stream differ for OwnTail vs LlvmTail assembly.
usage: asmdiff.py own.s llvm.s
Prints (1) first mismatches of the non-wait instruction stream (should be none: LLVM tail never
reorders), (2) counts of s_nop cycles / s_waitcnt / s_delay_alu / s_set* per backend,
(3) every LLVM-inserted s_nop/s_waitcnt/delay_alu with the instruction pair around it."""
import re, sys, collections

def load(path):
    out = []
    for line in open(path):
        line = line.split(';')[0].split('//')[0].strip()
        if not line or line.startswith(('.', '-')) or line.endswith(':') or ': ' in line:
            continue
        out.append(re.sub(r'\s+', ' ', line))
    return out

def is_filler(l):
    # unconditional branches: we always emit them, LLVM drops fallthrough ones
    return l.startswith(('s_nop', 's_waitcnt', 's_delay_alu', 's_set_vgpr_msb', 's_branch'))

def norm(l):
    # LLVM may pick different (equivalent) mnemonics for the same op; normalise a few.
    l = l.replace('_e32', '').replace('_e64', '')
    l = re.sub(r'\b0x([0-9a-fA-F]+)\b', lambda m: str(int(m.group(1), 16)), l)
    l = re.sub(r'\bs_branch BB(\d+)', 's_branch L', l)
    l = re.sub(r'(s_cbranch_\w+) \S+', r'\1 L', l)
    return l

def split(insts):
    core, filler = [], []  # filler: (index into core where it sits, text)
    for l in insts:
        if is_filler(l):
            filler.append((len(core), l))
        else:
            core.append(l)
    return core, filler

def nop_cycles(filler):
    c = 0
    for _, l in filler:
        m = re.match(r's_nop (\d+)', l)
        if m: c += int(m.group(1)) + 1
    return c

def stats(filler):
    d = collections.Counter()
    for _, l in filler:
        d[l.split()[0]] += 1
    return d

own, llv = load(sys.argv[1]), load(sys.argv[2])
oc, of = split(own)
lc, lf = split(llv)
oc_n = [norm(x) for x in oc]
lc_n = [norm(x) for x in lc]
mism = [(i, a, b) for i, (a, b) in enumerate(zip(oc_n, lc_n)) if a != b]
print(f'core instructions: own={len(oc)} llvm={len(lc)} mismatches={len(mism)}')
for i, a, b in mism[:15]:
    print(f'  #{i}: own  : {a}')
    print(f'        llvm : {b}')
print(f'nop cycles : own={nop_cycles(of)} llvm={nop_cycles(lf)}')
print(f'filler ops : own={dict(stats(of))} llvm={dict(stats(lf))}')
if len(sys.argv) > 3 and sys.argv[3] == 'nops':
    print('--- LLVM-inserted filler with neighbours (core index: prev | filler | next)')
    for idx, l in lf:
        prev = lc[idx - 1] if idx > 0 else '<start>'
        nxt = lc[idx] if idx < len(lc) else '<end>'
        print(f'{idx:6d}: {prev}  |  {l}  |  {nxt}')
def waits_at(filler):
    d = collections.defaultdict(list)
    for idx, l in filler:
        if l.startswith('s_waitcnt'):
            d[idx].append(l[len('s_waitcnt '):])
    return d

if len(sys.argv) > 3 and sys.argv[3] == 'waits':
    ow, lw = waits_at(of), waits_at(lf)
    print('--- waitcnt differences at the same core position (own vs llvm)')
    n = 0
    for idx in sorted(set(ow) | set(lw)):
        if ow.get(idx) != lw.get(idx):
            n += 1
            prev = lc[idx - 1] if idx > 0 else '<start>'
            nxt = lc[idx] if idx < len(lc) else '<end>'
            print(f'{idx:6d}: own={ow.get(idx)} llvm={lw.get(idx)}   [{prev}] -> [{nxt}]')
    print(f'{n} positions differ')
def cls(l):
    m = l.split()[0]
    if m.startswith('v_mfma'): return 'MFMA'
    if m.startswith(('v_exp', 'v_rcp', 'v_log', 'v_rsq', 'v_sqrt', 'v_sin', 'v_cos')): return 'VALU-trans'
    if m.startswith('v_cmp') or m.startswith('v_cndmask'): return 'VALU-vcc'
    if m.startswith('v_readfirstlane') or m.startswith('v_readlane'): return 'VALU-readlane'
    if m.startswith('v_'): return 'VALU'
    if m.startswith('ds_'): return 'DS'
    if m.startswith(('global_', 'buffer_', 'flat_')): return 'VMEM'
    if m.startswith(('s_load', 's_buffer')): return 'SMEM'
    if m.startswith('s_cbranch'): return 'S-branch'
    if m.startswith('s_barrier'): return 'S-barrier'
    if m.startswith('s_'): return 'SALU'
    return m

def nop_amount(l):
    m = re.match(r's_nop (\d+)', l)
    return int(m.group(1)) + 1 if m else 0

if len(sys.argv) > 3 and sys.argv[3] == 'classes':
    # group consecutive filler at one core position; attribute the nops to (prev class -> next class)
    for name, fl, core in (('own', of, oc), ('llvm', lf, lc)):
        hist = collections.Counter(); cyc = collections.Counter(); ex = {}
        for idx, l in fl:
            n = nop_amount(l)
            if not n: continue
            prev = core[idx - 1] if idx > 0 else '<start>'
            nxt = core[idx] if idx < len(core) else '<end>'
            k = (cls(prev) if idx > 0 else 'start', cls(nxt) if idx < len(core) else 'end')
            hist[k] += 1; cyc[k] += n; ex.setdefault(k, (prev, l, nxt))
        print(f'--- {name}: s_nop cycles by (instruction before -> instruction after)')
        for k, c in cyc.most_common(14):
            p, nn, x = ex[k]
            print(f'  {k[0]:>14} -> {k[1]:<14} x{hist[k]:<4} cycles={c:<5} e.g. [{p}] {nn} [{x}]')