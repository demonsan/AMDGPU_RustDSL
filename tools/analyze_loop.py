#!/usr/bin/env python3
"""Static instruction-mix and VGPR-bank analysis of the hot loop of an assembly listing (own emitter or llvm-objdump).

usage: analyze_loop.py own|objdump FILE   -- prints, for every backward-branch loop that contains WMMAs, the instruction classes and the
operand-bank statistics (VGPR bank = register index % 8; the s_set_vgpr_msb windows are multiples of 256, so they do not change it).
Only counts are printed (no instruction text)."""
import re
import sys
from collections import Counter

fmt, path = sys.argv[1], sys.argv[2]
ins = []          # (mnemonic, operand string, addr or None)
labels = {}       # label -> instruction index
for ln in open(path, errors='replace'):
    ln = ln.rstrip('\n')
    if fmt == 'own':
        s = ln.strip()
        if not s or s.startswith(('.', ';', '#', '//')):
            continue
        m = re.match(r'^([A-Za-z_][A-Za-z_0-9.$]*):\s*$', s)
        if m:
            labels[m.group(1)] = len(ins)
            continue
        parts = s.split(None, 1)
        ins.append((parts[0], parts[1] if len(parts) > 1 else '', None))
    else:
        m = re.match(r'^\t([a-z][a-z_0-9]*)\s*([^/]*?)\s*(//\s*([0-9A-Fa-f]{12}):.*)?$', ln)
        if not m:
            continue
        ins.append((m.group(1), m.group(2), int(m.group(4), 16) if m.group(4) else None))

addr_to_idx = {a: i for i, (_, _, a) in enumerate(ins) if a is not None}


def targets():
    """(branch index, target index) of every branch"""
    out = []
    for i, (mn, op, a) in enumerate(ins):
        if not mn.startswith('s_branch') and not mn.startswith('s_cbranch'):
            continue
        if fmt == 'own':
            lab = op.split()[0] if op else ''
            if lab in labels:
                out.append((i, labels[lab]))
        else:
            try:
                v = int(op.split()[0], 0)
            except Exception:
                continue
            if v >= 0x8000:
                v -= 0x10000
            t = a + 4 + v * 4
            if t in addr_to_idx:
                out.append((i, addr_to_idx[t]))
    return out


def regs(op):
    """VGPR register operands of an operand string: list of (first, count)"""
    op = re.sub(r'/\*.*?\*/', '', op)
    res = []
    for m in re.finditer(r'\bv\[(\d+):(\d+)\]|\bv(\d+)\b', op):
        if m.group(1) is not None:
            a, b = int(m.group(1)), int(m.group(2))
            res.append((a, b - a + 1))
        else:
            res.append((int(m.group(3)), 1))
    return res


def classify(mn):
    if mn.startswith('v_wmma_ld_scale'):
        return 'wmma_ld_scale'
    if mn.startswith('v_wmma'):
        return 'wmma'
    if mn.startswith('ds_load'):
        return 'ds_load'
    if mn.startswith('ds_store'):
        return 'ds_store'
    if mn.startswith('tensor_'):
        return 'tensor'
    if mn.startswith('s_wait'):
        return mn
    if mn in ('s_nop', 'v_nop') or mn.startswith('s_delay'):
        return mn
    if mn == 's_set_vgpr_msb':
        return 'msb'
    if mn.startswith('s_barrier'):
        return mn
    if mn.startswith('v_'):
        return 'valu_other'
    if mn.startswith('s_'):
        return 'salu_other'
    return 'other'


loops = []
for br, tg in targets():
    if tg < br:
        n = sum(1 for k in range(tg, br + 1) if ins[k][0].startswith('v_wmma') and not ins[k][0].startswith('v_wmma_ld'))
        loops.append((tg, br, n))
print('loops with WMMAs: ' + ', '.join(f'[{t}..{b}] wmma={n} len={b - t + 1}' for t, b, n in sorted(set(loops)) if n))
if not any(n for _, _, n in loops):
    sys.exit(0)
cand = [l for l in loops if l[2] >= 200]
tg, br, _ = min(cand, key=lambda x: x[1] - x[0])
body = ins[tg:br + 1]
print(f'--- hot loop [{tg}..{br}]: {len(body)} instructions')
cl = Counter(classify(mn) for mn, _, _ in body)
for k, v in sorted(cl.items(), key=lambda kv: -kv[1]):
    print(f'  {k:24s} {v}')

# wmma operand banks
W = [(i, mn, op) for i, (mn, op, _) in enumerate(body) if mn.startswith('v_wmma') and not mn.startswith('v_wmma_ld')]
stat = Counter()
prevA = prevB = None
pa, pb = Counter(), Counter()
for i, mn, op in W:
    r = regs(op)
    if len(r) < 4:
        stat['wmma_unparsed'] += 1
        continue
    d, a, b_, c = r[0], r[1], r[2], r[3]
    sc = r[4:]
    banks = {'A': a[0] % 8, 'B': b_[0] % 8, 'C': c[0] % 8}
    stat['wmma'] += 1
    if banks['A'] == banks['B']:
        stat['A_B_same_start_bank'] += 1
    if banks['A'] == banks['C']:
        stat['A_C_same_start_bank'] += 1
    if banks['B'] == banks['C']:
        stat['B_C_same_start_bank'] += 1
    if len(sc) >= 2 and sc[0][0] % 8 == sc[1][0] % 8:
        stat['scaleA_scaleB_same_bank'] += 1
    for s in sc:
        for k, v in banks.items():
            if s[0] % 8 == v:
                stat[f'scale_same_bank_as_{k}'] += 1
    if prevA == a[0]:
        stat['A_same_as_prev_wmma'] += 1
    if prevB == b_[0]:
        stat['B_same_as_prev_wmma'] += 1
    prevA, prevB = a[0], b_[0]
    pa[banks['A']] += 1
    pb[banks['B']] += 1
for k, v in sorted(stat.items()):
    print(f'  wmma {k:30s} {v}')
print('  A start-bank histogram', dict(sorted(pa.items())), ' B', dict(sorted(pb.items())))

# ld_scale operands (vendor form)
L = [(mn, op) for mn, op, _ in body if mn.startswith('v_wmma_ld_scale')]
if L:
    s2 = Counter()
    for mn, op in L:
        r = regs(op)
        if len(r) >= 2 and r[0][0] % 8 == r[1][0] % 8:
            s2['ld_scale_same_bank_pair'] += 1
        s2['ld_scale'] += 1
        s2['reuse_a'] += 'matrix_a_reuse' in op
        s2['reuse_b'] += 'matrix_b_reuse' in op
    print('  ld_scale', dict(s2))

# ds loads: address / destination banks
D = [(mn, op) for mn, op, _ in body if mn.startswith('ds_load')]
ds = Counter()
for mn, op in D:
    r = regs(op)
    if len(r) >= 2:
        dst, ad = r[0], r[1]
        ds['ds_load'] += 1
        if ad[0] % 8 == dst[0] % 8:
            ds['addr_same_bank_as_dst_start'] += 1
ad_banks = Counter(regs(op)[1][0] % 8 for mn, op in D if len(regs(op)) >= 2)
print('  ds_load', dict(ds), 'addr banks', dict(sorted(ad_banks.items())))
# consecutive wmma count between waits etc.: longest wmma run without any wait / barrier / msb in between
run = best = 0
for mn, op, _ in body:
    if mn.startswith('v_wmma') and not mn.startswith('v_wmma_ld'):
        run += 1
        best = max(best, run)
    elif mn.startswith('s_wait') or mn.startswith('s_barrier') or mn == 's_set_vgpr_msb':
        run = 0
print('  longest WMMA run without wait/barrier/msb:', best)
