#!/usr/bin/env python3
"""Static VGPR RAM-bank model of a gfx1250 kernel's hot loop (counts only, so it runs on vendor code objects too).

Model (lead / ISA notes): the VGPR file is 8 RAMs, RAM = register % 8; a VALU instruction reading two or more DIFFERENT registers that sit in
the same RAM and are not in the 4-entry operand source cache pays one extra cycle per extra read of that RAM. The source cache holds the
last 4 distinct VGPRs read (LRU); v_readlane / v_readfirstlane / v_permlane* and EXEC writes invalidate it. WMMA operands (A / B / C
tiles) are not modelled. s_set_vgpr_msb moves operands to v256+ (identity only: 256 % 8 = 0, the RAM is unchanged).

Also counts the scheduling-overhead instructions (s_wait_alu, memory waits, v_nop, s_nop, s_set_vgpr_msb) per loop.

    python tools/vgpr_banks.py K.s|K.dis [--loops N] [--worst N]
"""
import collections, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load(path):
    """[(op, operand text, labels, branch target)]"""
    out = []
    if path.endswith('.dis'):
        import isa_lift_gfx1250 as P
        seen = None
        for sec, op, args in P.parse(path):
            labels = [sec] if sec != seen else []
            seen = sec
            a = args.split(' |||')[0].split(' words:')[0]
            tgt = a.strip() if op.startswith(('s_branch', 's_cbranch')) else None
            out.append((op, a, labels, tgt))
        return out
    pending = []
    for line in open(path, errors='replace'):
        line = line.rstrip('\n')
        m = re.match(r'^([.\w$]+):', line)
        if m:
            pending.append(m.group(1))
            continue
        m = re.match(r'^\s+([a-z_][a-z0-9_]*)\b(.*)$', line)
        if not m or m.group(1).startswith('.'):
            continue
        op, rest = m.group(1), m.group(2)
        tgt = None
        if op.startswith(('s_branch', 's_cbranch')):
            t = re.match(r'\s*([.\w$]+)', rest)
            tgt = t.group(1) if t else None
        out.append((op, rest, pending, tgt))
        pending = []
    return out


VREG = re.compile(r'\bv(\d+)\b|\bv\[(\d+):(\d+)\]')


def regs(text):
    out = []
    for m in VREG.finditer(text):
        if m.group(1) is not None:
            out.append([int(m.group(1))])
        else:
            out.append(list(range(int(m.group(2)), int(m.group(3)) + 1)))
    return out


def split_ops(text):
    # top-level comma split (ignore commas inside brackets / parentheses)
    parts, depth, cur = [], 0, ''
    for ch in text:
        if ch in '[(':
            depth += 1
        elif ch in '])':
            depth -= 1
        if ch == ',' and depth == 0:
            parts.append(cur)
            cur = ''
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    return [p.strip() for p in parts]


def valu_sources(op, text):
    """(sources: list of register lists per operand, dst registers) of one VALU instruction (VOPD: both halves)"""
    if '::' in text or op.startswith('v_dual_'):
        halves = text.split('::')
        srcs, dsts = [], []
        first = True
        for h in halves:
            h = h.strip()
            if not first:
                m = re.match(r'([a-z_0-9]+)\s+(.*)$', h)
                h = m.group(2) if m else h
            first = False
            ops = split_ops(h)
            if ops:
                dsts += sum(regs(ops[0]), [])
                for o in ops[1:]:
                    srcs += regs(o)
        return srcs, dsts
    ops = split_ops(text.split(' ')[0] if False else text)
    if not ops:
        return [], []
    # v_cmp* write vcc / an sgpr: every v-register is a source
    if op.startswith(('v_cmp', 'v_readlane', 'v_readfirstlane')):
        return sum([regs(o) for o in ops], []), []
    dst = sum(regs(ops[0]), [])
    srcs = []
    for o in ops[1:]:
        srcs += regs(o)
    return srcs, dst


def analyse(ins, s, e, worst=0):
    cache = collections.OrderedDict()
    msb = 0
    st = collections.Counter()
    worst_list = []
    for k in range(s, e + 1):
        op, text, _, _ = ins[k]
        if op == 's_set_vgpr_msb':
            m = re.search(r'(0x[0-9a-fA-F]+|\d+)', text)
            msb = int(m.group(1), 0) if m else 0
            st['msb'] += 1
            continue
        if op.startswith('s_wait_alu'):
            st['wait_alu'] += 1
        elif re.match(r'^s_wait_(loadcnt|storecnt|dscnt|kmcnt|tensorcnt|asynccnt|xcnt)|^s_waitcnt', op):
            st['wait_mem'] += 1
        elif op == 'v_nop':
            st['v_nop'] += 1
        elif op == 's_nop':
            st['s_nop'] += 1
        if 'exec' in text and op.startswith('s_') and not op.startswith(('s_wait', 's_cbranch')):
            cache.clear()
        if not op.startswith('v_') or op == 'v_nop' or op.startswith('v_wmma') or op.startswith('v_swmmac'):
            continue
        st['valu'] += 1
        srcs, dsts = valu_sources(op, text)
        # identity with the msb window (the source slots' high bits; approximate: one window for all sources)
        hi = ((msb >> 0) & 3) * 256
        flat = sorted(set(r + hi for o in srcs for r in o))
        misses = [r for r in flat if r not in cache]
        st['src_reads'] += len(flat)
        st['cache_hits'] += len(flat) - len(misses)
        ram = collections.Counter(r % 8 for r in misses)
        extra = sum(n - 1 for n in ram.values() if n > 1)
        if extra:
            st['conflict_instrs'] += 1
            st['extra_cycles'] += extra
            if worst:
                worst_list.append((extra, k, op))
        for r in flat:
            cache.pop(r, None)
            cache[r] = True
            while len(cache) > 4:
                cache.popitem(last=False)
        if op.startswith(('v_readlane', 'v_readfirstlane', 'v_permlane', 'v_writelane')):
            cache.clear()
    worst_list.sort(reverse=True)
    return st, worst_list[:worst]


def main():
    path = sys.argv[1]
    nloops = int(sys.argv[sys.argv.index('--loops') + 1]) if '--loops' in sys.argv else 1
    worst = int(sys.argv[sys.argv.index('--worst') + 1]) if '--worst' in sys.argv else 0
    ins = load(path)
    where = {}
    for k, (_, _, labels, _) in enumerate(ins):
        for l in labels:
            where[l] = k
    loops = [(where[t], k) for k, (_, _, _, t) in enumerate(ins) if t in where and where[t] <= k]
    inner = [l for l in loops if not any(o != l and l[0] <= o[0] and o[1] <= l[1] for o in loops)]
    inner.sort(key=lambda l: -(l[1] - l[0]))
    wm = lambda s, e: sum(1 for k in range(s, e + 1) if ins[k][0].startswith('v_wmma'))
    print(f'{os.path.basename(path)}: {len(ins)} instructions')
    st, _ = analyse(ins, 0, len(ins) - 1)
    print('  whole kernel: ' + ', '.join(f'{k} {v}' for k, v in sorted(st.items())))
    for s, e in inner[:nloops]:
        st, wl = analyse(ins, s, e, worst)
        w = max(wm(s, e), 1)
        print(f'  loop [{s}..{e}] {e - s + 1} instr, wmma {wm(s, e)}: ' + ', '.join(f'{k} {v}' for k, v in sorted(st.items())))
        print(f'     per WMMA: valu {st["valu"] / w:.2f}, bank-conflict instr {st["conflict_instrs"] / w:.2f}, extra cycles {st["extra_cycles"] / w:.2f}, '
              f'cache hit rate {st["cache_hits"] / max(st["src_reads"], 1):.2f}, wait_alu {st["wait_alu"] / w:.2f}, v_nop {st["v_nop"] / w:.2f}, msb {st["msb"] / w:.2f}')
        for x, k, op in wl:
            print(f'       +{x} at [{k}] {op}' if not path.endswith('.dis') else f'       +{x} at [{k}]')


if __name__ == '__main__':
    main()
