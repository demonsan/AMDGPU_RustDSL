#!/usr/bin/env python3
"""Instruction-level diff of an exact-mode build (AMDC_EXACT_DUMP=all output) against a reference disassembly (llvm-objdump).

Both sides are normalised (comments / encodings dropped, numeric literals to integers, _e32 / _e64 suffixes kept, labels dropped) and
aligned with difflib; prints the counts (identical / differing / only-in-one) and every differing run. Branch targets are compared by
opcode only (label names differ by construction).

usage: exact_isa_diff.py REF.dis DUMP.txt [--context N]
The inputs are session artifacts (the reference is a vendor binary): never commit them.
"""
import difflib, re, sys


FLOATS = {'0.5': 0x3f000000, '-0.5': 0xbf000000, '1.0': 0x3f800000, '-1.0': 0xbf800000, '2.0': 0x40000000, '-2.0': 0xc0000000,
          '4.0': 0x40800000, '-4.0': 0xc0800000}


def num(tok):
    if tok in FLOATS:
        return str(FLOATS[tok])
    try:
        if re.fullmatch(r'-?0x[0-9a-fA-F]+', tok):
            v = int(tok, 16)
        elif re.fullmatch(r'-?\d+', tok):
            v = int(tok)
        else:
            return tok
        return str(v & 0xffffffff) if v < 0 or v > 0x7fffffff else str(v)
    except ValueError:
        return tok


def norm(s):
    s = re.sub(r'\s*//.*$', '', s).strip()
    if not s:
        return None
    op, _, rest = s.partition(' ')
    # printer aliases / defaults that are the same encoding
    op = {'v_mov_b64': 'v_mov_b64_e32'}.get(op, op)
    rest = re.sub(r'\s*op_sel_hi:\[0,\s*0,\s*0\]', '', rest)
    if op.startswith(('s_cbranch', 's_branch')):
        return op
    toks = [t.strip() for t in re.split(r'(,\s*|\s+)', rest) if t.strip() and t.strip() != ',']
    out = []
    for t in toks:
        if ':' in t and not t.startswith(('v[', 's[', 'a[')):
            k, _, v = t.partition(':')
            out.append(f'{k}:{num(v)}')
        else:
            out.append(num(t))
    return op + ' ' + ' '.join(out)


def ref_lines(path):
    out = []
    for line in open(path):
        if re.match(r'^[0-9a-f]+ <', line) or line.startswith('Disassembly') or not line.startswith(('\t', ' ')):
            continue
        n = norm(line)
        if n and not n.startswith('s_code_end'):
            out.append(n)
    return out


def dump_lines(path):
    out = []
    for line in open(path):
        m = re.match(r'^DUMP BB\d+ \d+ stmt\S* (.*)$', line.rstrip())
        if m:
            n = norm(m.group(1))
            if n:
                out.append(n)
    return out


def main():
    ref, mine = ref_lines(sys.argv[1]), dump_lines(sys.argv[2])
    ctx = int(sys.argv[sys.argv.index('--context') + 1]) if '--context' in sys.argv else 2
    if '--no-branches' in sys.argv:
        ref = [x for x in ref if not x.startswith(('s_cbranch', 's_branch'))]
        mine = [x for x in mine if not x.startswith(('s_cbranch', 's_branch'))]
    sm = difflib.SequenceMatcher(a=ref, b=mine, autojunk=False)
    same = sum(b.size for b in sm.get_matching_blocks())
    print(f'reference {len(ref)} instructions, exact build {len(mine)}; identical in order {same} '
          f'({100.0 * same / max(len(ref), 1):.2f} % of the reference)')
    kinds = {}
    cls = {}
    def classify(x, y):
        ox, _, rx = x.partition(' ')
        oy, _, ry = y.partition(' ')
        if ox == oy and sorted(rx.split()) == sorted(ry.split()):
            return 'operand order (commuted)'
        if re.sub(r'_e(32|64)$', '', ox) == re.sub(r'_e(32|64)$', '', oy) and sorted(rx.split()) == sorted(ry.split()):
            return 'encoding size (e32 / e64) + operand order'
        if ox == oy:
            return f'operands differ ({ox})'
        return f'opcode differs ({ox} -> {oy})'
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            continue
        if tag == 'replace' and i2 - i1 == j2 - j1:
            for x, y in zip(ref[i1:i2], mine[j1:j2]):
                c = classify(x, y)
                cls[c] = cls.get(c, 0) + 1
        else:
            for x in ref[i1:i2]:
                c = f'only in reference: {x.split()[0]}'
                cls[c] = cls.get(c, 0) + 1
            for y in mine[j1:j2]:
                c = f'only in build: {y.split()[0]}'
                cls[c] = cls.get(c, 0) + 1
        kinds[tag] = kinds.get(tag, 0) + 1
        if '--quiet' in sys.argv:
            continue
        print(f'--- {tag} ref[{i1}:{i2}] build[{j1}:{j2}]')
        for x in ref[i1:i2][:12]:
            print(f'  - {x}')
        for x in mine[j1:j2][:12]:
            print(f'  + {x}')
    print('runs:', kinds)
    for c, n in sorted(cls.items(), key=lambda kv: -kv[1]):
        print(f'  {n:6d}  {c}')


if __name__ == '__main__':
    main()
