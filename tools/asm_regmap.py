#!/usr/bin/env python3
"""Register-role map of an FMHA steady loop (ours `amdc emit` text or vendor objdump), from operand positions only.

usage: asm_regmap.py file loop_index   (loop indices as listed by asm_diff.py --list)
Roles: O acc (MFMA dst not read by v_max3), S tiles (MFMA dst read by v_max3), K frag (A of the scaled MFMA writing S), Q frag (B of it),
V frag (A of the MFMA writing O / ds_read_b64_tr dst), P (B of the MFMA writing O), MFMA scale words, rowsum acc (smfmac dst), smfmac A/idx,
DMA voffset (buffer/global ... lds address VGPRs), other VGPRs (the rest touched in the loop), SGPRs touched, AGPRs touched.
"""
import re, sys, collections
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from asm_diff import parse, loops, cls

def regs(tok):
    m = re.match(r'^([vsa])\[(\d+):(\d+)\]$', tok)
    if m:
        return [(m.group(1), i) for i in range(int(m.group(2)), int(m.group(3)) + 1)]
    m = re.match(r'^([vsa])(\d+)$', tok)
    if m:
        return [(m.group(1), int(m.group(2)))]
    return []

def operands(t):
    parts = t.split(None, 1)
    if len(parts) < 2:
        return []
    return [regs(x.strip()) for x in re.split(r',\s*', parts[1].split(' offset')[0].split(' op_sel')[0])]

def rng(s):
    s = sorted(s)
    if not s:
        return '-'
    out, a, b = [], s[0], s[0]
    for x in s[1:]:
        if x == b + 1:
            b = x
        else:
            out.append((a, b)); a = b = x
    out.append((a, b))
    return ' '.join('%d' % a if a == b else '%d-%d' % (a, b) for a, b in out)

def main():
    path, li = sys.argv[1], int(sys.argv[2])
    ins, labels = parse(path)
    j, k = loops(ins, labels)[li]
    body = [t for _, t in ins[j:k + 1]]
    max3_src = set()
    for t in body:
        if t.startswith('v_max3_f32') or t.startswith('v_max_f32'):
            for o in operands(t)[1:]:
                max3_src |= {r for (c, r) in o if c == 'v'}
    role = collections.defaultdict(set)
    tr_dst = set()
    for t in body:
        if t.startswith('ds_read_b64_tr'):
            tr_dst |= {r for (c, r) in operands(t)[0] if c == 'v'}
    for t in body:
        k_ = cls(t)
        ops = operands(t)
        if k_.startswith('mfma') and k_ != 'mfma.smfmac' and len(ops) >= 4:
            dst = {r for (c, r) in ops[0] if c in 'v'} | {r + 1000 for (c, r) in ops[0] if c == 'a'}
            A = {r for (c, r) in ops[1] if c == 'v'}
            B = {r for (c, r) in ops[2] if c == 'v'}
            if dst & max3_src or 'f16' in t.split()[0] or 'bf16' in t.split()[0] and dst & max3_src:
                role['S tiles (QK acc)'] |= dst
                if 'scale' in t.split()[0]:
                    role['K frag (QK A)'] |= A
                    role['Q frag (QK B)'] |= B
                else:
                    role['bias/helper MFMA operands'] |= A | B
            elif 'f16' in t.split()[0] or 'bf16' in t.split()[0]:
                role['bias/helper MFMA (dst+ops)'] |= dst | A | B
            else:
                role['O acc (PV acc)'] |= dst
                role['V frag (PV A)'] |= A
                role['P (PV B)'] |= B
            if 'scale' in t.split()[0] and len(ops) >= 6:
                role['MFMA scale words'] |= {r for o in ops[4:6] for (c, r) in o if c == 'v'}
        elif k_ == 'mfma.smfmac' and len(ops) >= 4:
            role['row-sum acc (smfmac D)'] |= {r for (c, r) in ops[0] if c == 'v'}
            role['smfmac A (ones) + idx'] |= {r for (c, r) in ops[1] if c == 'v'} | {r for (c, r) in ops[3] if c == 'v'}
            role['P (PV B)'] |= {r for (c, r) in ops[2] if c == 'v'}
        elif k_ == 'vmem.dma':
            role['DMA voffset/address'] |= {r for o in ops[:1] for (c, r) in o if c == 'v'}
            role['SGPR: DMA rsrc/soffset'] |= {r for o in ops[1:] for (c, r) in o if c == 's'}
    if tr_dst:
        role['V frag (PV A)'] |= tr_dst
    allv, alls, alla = set(), set(), set()
    for t in body:
        for o in operands(t):
            for (c, r) in o:
                (allv if c == 'v' else alls if c == 's' else alla).add(r)
    named = set().union(*[v for kk, v in role.items() if not kk.startswith('SGPR')])
    role['other VGPRs (softmax state, addresses, constants, temps)'] = allv - named
    for kk in sorted(role):
        print('%-58s %4d  %s' % (kk, len(role[kk]), rng(role[kk])))
    print('%-58s %4d  max v%d' % ('VGPRs touched in the loop', len(allv), max(allv) if allv else -1))
    print('%-58s %4d  %s' % ('SGPRs touched in the loop', len(alls), rng(alls)))
    print('%-58s %4d  %s' % ('AGPRs touched in the loop', len(alla), rng(alla)))

if __name__ == '__main__':
    main()
