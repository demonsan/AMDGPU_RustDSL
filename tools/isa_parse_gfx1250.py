#!/usr/bin/env python3
"""gfx1250 disassembly -> normalized instruction stream (the first stage of the gfx1250 exact lifter; docs/exact-mode.md).

Input: `llvm-objdump -d --mcpu=gfx1250` text of one kernel (node toolchain). Output: JSON lines, one instruction each:
    {"addr", "words": [hex dwords], "op", "args": [operands, physical register names], "mods": [modifiers], "cls", "raw": bool, ...}

What the objdump text does not give directly is made explicit here:
- VGPR banking: gfx1250 addresses 1024 VGPRs through `s_set_vgpr_msb`; objdump prints the encoded register and the physical one in a
  comment (`v[64:67] /*v[832:835]*/`). The stream carries the PHYSICAL register; the encoded one stays in `enc_args`.
- Words the disassembler cannot print (`.long`): instructions are rebuilt from the raw dwords. A tensor-data-mover instruction (VIMAGE
  encoding, op field 196 / 197) is 3 dwords; its third dword is usually mis-decoded as a 4-byte VOPC by objdump, so it is re-absorbed here.
  Decoded fields: op, cpol (th = dword1[22:20], scope = dword1[19:18]), SGPR groups g0..g3 (dword2 bytes, 0x7C = none). Any other `.long`
  run is kept as an opaque RAW instruction with the words of the following mis-decoded line absorbed when its encoding says so.

    python3 tools/isa_parse_gfx1250.py kernel.dis [--out kernel.jsonl] [--summary]
"""
import json, re, sys, collections

LINE = re.compile(r'^\t(?P<op>[a-z_0-9.]+)(?P<rest>.*?)\s*// (?P<addr>[0-9A-F]+): (?P<words>[0-9A-F ]+?)(?:\s*<.*>)?$')
# TDM ops of the VIMAGE encoding (bits [31:26] = 0b110100), op field [21:14]
TDM_OPS = {196: 'tensor_load_to_lds', 197: 'tensor_store_from_lds'}
NULL_SGPR = 0x7C


def split_args(s):
    """Top-level comma split that keeps `/*..*/` comments and brackets with their operand."""
    out, cur, depth, i = [], '', 0, 0
    while i < len(s):
        if s.startswith('/*', i):
            j = s.find('*/', i)
            cur += s[i:j + 2]
            i = j + 2
            continue
        c = s[i]
        if c in '[(':
            depth += 1
        elif c in '])':
            depth -= 1
        if c == ',' and depth == 0:
            out.append(cur.strip())
            cur = ''
        else:
            cur += c
        i += 1
    if cur.strip():
        out.append(cur.strip())
    return out


def phys(tok):
    """(physical, encoded) register spelling of one operand token."""
    m = re.match(r'^(?P<enc>\S+?)\s*/\*(?P<phys>[vsa]\[?[0-9:]+\]?)\*/$', tok)
    if m:
        return m.group('phys'), m.group('enc')
    m = re.match(r'^(?P<enc>\S+)/\*Invalid register.*\*/$', tok)
    if m:
        return m.group('enc'), m.group('enc')
    return tok, tok


def classify(op):
    for pre, c in [('v_wmma', 'wmma'), ('v_swmmac', 'wmma'), ('tensor_', 'tdm'), ('s_wait_', 'wait'), ('s_barrier', 'barrier'), ('ds_', 'lds'),
                   ('global_load_async', 'async'), ('global_', 'global'), ('buffer_', 'buffer'), ('s_load', 'smem'), ('s_buffer_load', 'smem'),
                   ('s_prefetch', 'prefetch'), ('s_cbranch', 'branch'), ('s_branch', 'branch'), ('s_set_vgpr_msb', 'msb'), ('s_setreg', 'hwreg'),
                   ('s_getreg', 'hwreg'), ('s_nop', 'nop'), ('v_nop', 'nop'), ('s_endpgm', 'end'), ('v_', 'valu'), ('s_', 'salu')]:
        if op.startswith(pre):
            return c
    return 'other'


def decode_tdm(w0, w1, w2):
    if (w0 >> 26) != 0b110100:
        return None
    opn = (w0 >> 14) & 0xFF
    if opn not in TDM_OPS:
        return None
    g = [(w2 >> (8 * k)) & 0xFF for k in range(4)]
    return dict(op=TDM_OPS[opn], cpol=dict(th=(w1 >> 20) & 7, scope=(w1 >> 18) & 3), g0=f's{g[0]}', g1=f's{g[1]}',
                g2=None if g[2] == NULL_SGPR else f's{g[2]}', g3=None if g[3] == NULL_SGPR else f's{g[3]}')


def parse(text):
    lines = [l for l in text.splitlines() if l.startswith('\t') and '//' in l]
    recs = []
    for l in lines:
        m = LINE.match(l)
        if not m:
            continue
        recs.append(dict(addr=int(m.group('addr'), 16), op=m.group('op'), rest=m.group('rest').strip(), words=[int(w, 16) for w in m.group('words').split()]))
    out = []
    i = 0
    while i < len(recs):
        r = recs[i]
        if r['op'] == '.long':
            w0 = r['words'][0]
            nxt = recs[i + 1] if i + 1 < len(recs) else None
            nn = recs[i + 2] if i + 2 < len(recs) else None
            # a 3-dword TDM: .long w0, .long w1, then a mis-decoded 4-byte line holding w2
            # (dword1 may itself decode as a 4-byte instruction, e.g. 0x00000000)
            if nxt and len(nxt['words']) == 1 and nn and len(nn['words']) == 1:
                t = decode_tdm(w0, nxt['words'][0], nn['words'][0])
                if t:
                    out.append(dict(addr=r['addr'], words=[w0, nxt['words'][0], nn['words'][0]], op=t['op'], args=[a for a in (t['g0'], t['g1'], t['g2'], t['g3']) if a],
                                    mods=[f"th:{t['cpol']['th']}", f"scope:{t['cpol']['scope']}"], cls='tdm', raw=True, absorbed=[nxt['op'], nn['op']]))
                    i += 3
                    continue
            # any other undecodable word: kept opaque; a following 1-dword line is absorbed when the run is an 8-byte (VOP3 / VOP3P / VIMAGE...)
            # encoding (bits [31:30] = 0b11 and not SOP*)
            words = [w0]
            if (w0 >> 30) == 0b11 and (w0 >> 28) != 0b1011 and nxt and len(nxt['words']) == 1 and nxt['op'] != '.long':
                words.append(nxt['words'][0])
                out.append(dict(addr=r['addr'], words=words, op='raw', args=[], mods=[], cls='raw', raw=True, absorbed=nxt['op']))
                i += 2
                continue
            out.append(dict(addr=r['addr'], words=words, op='raw', args=[], mods=[], cls='raw', raw=True))
            i += 1
            continue
        toks = split_args(r['rest'])
        args, enc, mods = [], [], []
        for t in toks:
            # modifiers: key:value or bare flags after the operands
            if re.match(r'^[a-z_]+(:[^\s]+)?$', t) and not re.match(r'^(v|s|a|ttmp|vcc|exec|m0|null|scc|off)', t):
                mods.append(t)
                continue
            # `matrix_a_reuse` style trailing flags glued after the last operand
            parts = t.split(' ')
            head = ' '.join(p for p in parts if not re.match(r'^[a-z_]+(:[^\s]+)?$', p) or p.startswith(('v', 's', 'a', 'ttmp', 'vcc', 'exec', 'm0', 'null')) or '/*' in p)
            mods += [p for p in parts if re.match(r'^[a-z_]+:[^\s]+$', p) or p in ('matrix_a_reuse', 'matrix_b_reuse', 'neg_lo:[1,0]', 'offen', 'idxen', 'nt', 'glc', 'slc', 'dlc')]
            p, e = phys(head.strip() if head.strip() else t)
            args.append(p)
            enc.append(e)
        out.append(dict(addr=r['addr'], words=r['words'], op=r['op'], args=args, enc_args=enc, mods=mods, cls=classify(r['op']),
                        text=(r['op'] + ' ' + re.sub(r'\s*/\*.*?\*/', '', r['rest'])).strip(), raw=False))
        i += 1
    return out


def selftest():
    """Synthetic lines built from the public CDNA5 VIMAGE encoding (no vendor content): a TDM load whose dword1 decodes as a 4-byte instruction,
    a banked VGPR operand, an opaque 8-byte run."""
    w0 = (0b110100 << 26) | (196 << 14)                  # DMASK = 0, DIM = 0
    w1 = (2 << 20)                                       # TH = 2, SCOPE = 0, VADDR4 = 0
    w2 = 0x7C7C2420                                      # VADDR0 = s32, VADDR1 = s36, VADDR2 / 3 = NULL
    text = '\n'.join([
        f'\t.long 0x{w0:08x}                       // 000000000100: {w0:08X}',
        f'\tv_cndmask_b32_e32 v0, s0, v0             // 000000000104: {w1:08X}',
        f'\tv_cmp_ge_u16_e32 vcc_lo, s32, v18.l      // 000000000108: {w2:08X}',
        '\tds_load_b128 v[64:67] /*v[832:835]*/, v76 offset:7168 // 00000000010C: DBFC1C00 4000004C',
        '\t.long 0xcc880064                       // 000000000114: CC880064',
        '\tv_min_num_f64_e32 v[1:2], v[8:9], v[8:9] // 000000000118: 1A021108',
        '\ts_endpgm                               // 00000000011C: BFB00000',
    ])
    ins = parse(text)
    assert [x['op'] for x in ins] == ['tensor_load_to_lds', 'ds_load_b128', 'raw', 's_endpgm'], [x['op'] for x in ins]
    t0 = ins[0]
    assert t0['args'] == ['s32', 's36'] and t0['mods'] == ['th:2', 'scope:0'] and t0['words'] == [w0, w1, w2], t0
    assert ins[1]['args'][0] == 'v[832:835]' and ins[1]['enc_args'][0] == 'v[64:67]', ins[1]
    assert ins[2]['words'] == [0xCC880064, 0x1A021108], ins[2]
    print('selftest ok')


def main():
    if '--selftest' in sys.argv:
        return selftest()
    src = sys.argv[1]
    o = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else None
    ins = parse(open(src, encoding='utf-8', errors='replace').read())
    if o:
        with open(o, 'w') as f:
            for x in ins:
                f.write(json.dumps(x) + '\n')
    if '--summary' in sys.argv or not o:
        c = collections.Counter(x['cls'] for x in ins)
        print(f'{len(ins)} instructions; ' + ', '.join(f'{k} {n}' for k, n in c.most_common()))
        raw = collections.Counter((x['op'], tuple(x['mods'])) for x in ins if x['raw'])
        for k, n in raw.most_common(8):
            print(f'  raw {k[0]} {" ".join(k[1])}: {n}')


if __name__ == '__main__':
    main()
