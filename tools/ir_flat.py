#!/usr/bin/env python3
"""Flat statement indices of an irv1 file, as `Function::stmts()` linearizes the structure tree (the indices gpu_verify reports):
`if` = BeginIf .. [Else ..] EndIf, `loop` = BeginLoop .. EndLoop, `region` = TileProduce/Consume .. TileEnd, `stage` = StageBegin .. StageEnd.

usage: ir_flat.py IR [stmt ...]       prints the given statements (with their text line), or every statement with its index
"""
import sys


def flat(path):
    L = open(path).read().split('\n')
    bs = next(i for i, l in enumerate(L) if l.strip() == 'body {')
    out = []          # (flat index, line number, text)
    stack = []
    for i in range(bs + 1, len(L)):
        s = L[i].strip()
        if not s or s.startswith('#'):
            continue
        if s == '}':
            if not stack:
                break
            kind = stack.pop()
            out.append((len(out), i + 1, {'if': 'EndIf', 'loop': 'EndLoop', 'region': 'TileEnd', 'stage': 'StageEnd'}[kind]))
            continue
        if s.startswith('} else {'):
            out.append((len(out), i + 1, 'Else'))
            continue
        head = s.split()[0]
        if s.endswith('{') and head in ('if', 'loop', 'region', 'stage'):
            stack.append(head)
            out.append((len(out), i + 1, s))
            continue
        out.append((len(out), i + 1, s))
    return out


def main():
    fl = flat(sys.argv[1])
    want = [int(x) for x in sys.argv[2:]]
    for k, ln, s in fl:
        if not want or k in want:
            print(f'{k}\tline {ln}\t{s[:160]}')


if __name__ == '__main__':
    main()
