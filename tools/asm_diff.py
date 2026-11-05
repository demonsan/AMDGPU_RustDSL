#!/usr/bin/env python3
"""Static side-by-side of two gfx950 kernels' assembly (ours `amdc emit` text vs a vendor llvm-objdump disassembly).

usage: asm_diff.py ours.s vendor.dis [--units-ours N] [--units-vendor N] [--timeline]
  - descriptor fields (.vgpr_count / .agpr_count / .sgpr_count / accum_offset / LDS / scratch) from the text (ours) or the
    `llvm-readobj --notes` dump passed as vendor.dis.notes (if present)
  - loops = backward branches; per loop: instruction classes, MFMA forms, waits (+ partial counts), nops (wait states), barriers, setprio
  - the steady loop = the loop with the most matrix instructions; counts are divided by the number of 128-key / K units in one pass
    (--units-*, default: scaled 32x32x64 QK MFMAs / 8 for FMHA)
  - --timeline prints the steady loop as a compressed sequence of class runs
"""
import re, sys, collections

def parse(path):
    """-> list of (addr_or_index, label_or_None, text) for instructions; labels map name -> index."""
    ins, labels = [], {}
    for line in open(path, errors='replace'):
        # llvm-objdump (ROCm 7) form: "<tab>insn ops   // 000000001930: BF840004 <sym+0x144>" -> keep the address and the branch target
        cm = re.search(r'//\s*([0-9A-Fa-f]{8,}):\s+(?:[0-9A-Fa-f]{8}\s*)+(<[^>]+>)?', line)
        if cm and not re.match(r'^\s*[0-9a-f]+ <', line):
            body = line.split('//')[0].strip()
            if body and re.match(r'^[a-z_]', body):
                ins.append((int(cm.group(1), 16), body + (' ' + cm.group(2) if cm.group(2) and 'branch' in body else '')))
            continue
        s = line.split('//')[0].split(';')[0].rstrip()
        if not s.strip():
            continue
        m = re.match(r'^\s*([A-Za-z_.$][\w.$]*):\s*$', s)
        if m:
            labels[m.group(1)] = len(ins)
            continue
        m = re.match(r'^\s*([0-9a-f]+) <([^>]+)>:', s)          # objdump symbol header
        if m:
            labels[m.group(2)] = len(ins)
            continue
        t = s.strip()
        if t.startswith('.') or t.startswith('Disassembly') or ':' in t.split()[0]:
            # objdump lines look like "   2b68: d3ad0010 ...  v_mfma..." -> keep the mnemonic part
            m = re.match(r'^([0-9a-f]+):\s+(?:[0-9a-f]{8}\s+)+(\S.*)$', t)
            if m:
                ins.append((int(m.group(1), 16), m.group(2).strip()))
            continue
        if re.match(r'^[a-z_][a-z0-9_]*', t):
            ins.append((len(ins), t))
    return ins, labels

def target_index(ins, labels, txt):
    m = re.search(r'(s_cbranch_\w+|s_branch)\s+(\S+)', txt)
    if not m:
        return None
    tgt = m.group(2)
    if tgt in labels:
        return labels[tgt]
    m2 = re.search(r'<([^>+]+)(?:\+0x([0-9a-f]+))?>', txt)
    if m2:
        base = labels.get(m2.group(1))
        if base is not None and m2.group(2):
            addr = ins[base][0] + int(m2.group(2), 16)
            for k, (a, _) in enumerate(ins):
                if a == addr:
                    return k
        return base
    try:
        addr = int(tgt, 16)
        for k, (a, _) in enumerate(ins):
            if a == addr:
                return k
    except ValueError:
        pass
    return None

def cls(t):
    op = t.split()[0]
    if op.startswith('v_mfma_scale'):
        return 'mfma.scaled'
    if op.startswith('v_mfma_f32_32x32x64_f8f6f4') or op.startswith('v_mfma_f32_16x16x128_f8f6f4'):
        return 'mfma.unscaled'
    if op.startswith('v_smfmac'):
        return 'mfma.smfmac'
    if op.startswith('v_mfma'):
        return 'mfma.other'
    if op.startswith('ds_read_b64_tr') or op.startswith('ds_load_tr'):
        return 'ds.read_tr'
    if op.startswith('ds_read') or op.startswith('ds_load'):
        return 'ds.read'
    if op.startswith('ds_'):
        return 'ds.write/other'
    if (op.startswith('buffer_load') or op.startswith('global_load')) and ' lds' in t:
        return 'vmem.dma'
    if op.startswith('buffer_load') or op.startswith('global_load'):
        return 'vmem.load'
    if op.startswith('buffer_store') or op.startswith('global_store'):
        return 'vmem.store'
    if op.startswith('s_waitcnt'):
        return 'wait'
    if op == 's_barrier':
        return 'barrier'
    if op == 's_nop':
        return 'nop'
    if op == 's_setprio':
        return 'setprio'
    if op.startswith('s_cbranch') or op == 's_branch':
        return 'branch'
    if op.startswith('v_accvgpr'):
        return 'valu.accvgpr'
    if op.startswith('v_'):
        return 'valu'
    if op.startswith('s_'):
        return 'salu'
    return 'other'

def loops(ins, labels):
    out = []
    for k, (a, t) in enumerate(ins):
        if cls(t) == 'branch':
            j = target_index(ins, labels, t)
            if j is not None and j < k:
                out.append((j, k))
    return out

def summarize(body):
    c = collections.Counter()
    ops = collections.Counter()
    waits = collections.Counter()
    nop_states = 0
    for t in body:
        k = cls(t)
        c[k] += 1
        op = t.split()[0]
        if k.startswith('mfma') or k.startswith('ds') or k.startswith('vmem') or k in ('valu', 'salu'):
            ops[op] += 1
        if k == 'wait':
            for m in re.finditer(r'(vmcnt|lgkmcnt|expcnt)\((\d+)\)', t):
                waits['%s(%s)' % (m.group(1), '0' if m.group(2) == '0' else 'n>0')] += 1
        if k == 'nop':
            m = re.search(r's_nop\s+(\S+)', t)
            nop_states += (int(m.group(1), 0) + 1) if m else 1
    return c, ops, waits, nop_states

def timeline(body):
    runs = []
    for t in body:
        k = cls(t)
        if k == 'wait':
            k = 'wait[' + ','.join(re.findall(r'(vmcnt\(\d+\)|lgkmcnt\(\d+\))', t)) + ']'
        elif k == 'setprio':
            k = 'setprio ' + t.split()[1]
        if runs and runs[-1][0] == k:
            runs[-1][1] += 1
        else:
            runs.append([k, 1])
    return ' | '.join('%s x%d' % (k, n) if n > 1 else k for k, n in runs)

def main():
    a = sys.argv[1:]
    paths = [x for x in a if not x.startswith('--')]
    opts = {x.split('=')[0]: (x.split('=')[1] if '=' in x else '1') for x in a if x.startswith('--')}
    res = {}
    for name, p in zip(('ours', 'vendor'), paths):
        ins, labels = parse(p)
        ls = loops(ins, labels)
        best = None
        if '--list' in opts:
            for i, (j, k) in enumerate(ls):
                body = [t for _, t in ins[j:k + 1]]
                print('%s loop %d: [%x..%x] %d instr, mfma %d, scaled %d, perm %d, barrier %d' % (name, i, ins[j][0], ins[k][0], k - j + 1,
                      sum(1 for t in body if cls(t).startswith('mfma')), sum(1 for t in body if cls(t) == 'mfma.scaled'),
                      sum(1 for t in body if t.startswith('v_perm_b32')), sum(1 for t in body if cls(t) == 'barrier')))
        sel = opts.get('--loop-' + name)
        for i, (j, k) in enumerate(ls):
            body = [t for _, t in ins[j:k + 1]]
            nm = sum(1 for t in body if cls(t).startswith('mfma'))
            if sel is not None:
                if i == int(sel):
                    best = (nm, j, k, body)
            elif best is None or nm > best[0]:
                best = (nm, j, k, body)
        units = float(opts.get('--units-' + name, 0)) or None
        body = best[3]
        if units is None:
            q = sum(1 for t in body if cls(t) == 'mfma.scaled')
            units = max(1.0, q / float(opts.get('--qk-per-unit', 8)))
        c, ops, waits, nops = summarize(body)
        res[name] = (c, ops, waits, nops, units, len(body), body, len(ins), len(ls))
    print('loops: ours %d (static %d instr), vendor %d (static %d instr)' % (res['ours'][8], res['ours'][7], res['vendor'][8], res['vendor'][7]))
    print('steady loop: ours %d instr / %.0f unit(s), vendor %d instr / %.0f unit(s)' % (res['ours'][5], res['ours'][4], res['vendor'][5], res['vendor'][4]))
    print('\n%-26s %10s %10s   (per unit)' % ('class', 'ours', 'vendor'))
    keys = sorted(set(res['ours'][0]) | set(res['vendor'][0]))
    for k in keys:
        print('%-26s %10.1f %10.1f' % (k, res['ours'][0][k] / res['ours'][4], res['vendor'][0][k] / res['vendor'][4]))
    print('%-26s %10.1f %10.1f' % ('nop wait states', res['ours'][3] / res['ours'][4], res['vendor'][3] / res['vendor'][4]))
    print('\nwaits (per unit):')
    for k in sorted(set(res['ours'][2]) | set(res['vendor'][2])):
        print('  %-24s %10.1f %10.1f' % (k, res['ours'][2][k] / res['ours'][4], res['vendor'][2][k] / res['vendor'][4]))
    print('\nopcodes (per unit, |diff| >= 0.5):')
    allk = set(res['ours'][1]) | set(res['vendor'][1])
    rows = sorted(allk, key=lambda k: -abs(res['ours'][1][k] / res['ours'][4] - res['vendor'][1][k] / res['vendor'][4]))
    for k in rows:
        o, v = res['ours'][1][k] / res['ours'][4], res['vendor'][1][k] / res['vendor'][4]
        if abs(o - v) >= 0.5:
            print('  %-40s %8.1f %8.1f' % (k, o, v))
    if '--timeline' in opts:
        for name in ('ours', 'vendor'):
            print('\n== timeline %s (one loop pass = %.0f unit(s)):\n%s' % (name, res[name][4], timeline(res[name][6])))

if __name__ == '__main__':
    main()
