#!/usr/bin/env python3
"""Control-flow shape of a kernel's assembly (vendor objdump or `amdc emit`): basic blocks, back edges, natural loops with their exits, and the
blocks laid out after the code that branches back into a loop (out-of-line cold paths). Answers whether the CFG maps onto the DSL's
structured loops / ifs (docs/fmha-replica.md R2).

usage: isa_cfg.py file.dis
"""
import collections
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asm_diff import parse, target_index  # noqa: E402


def blocks(ins, labels):
    starts = {0} | set(labels.values())
    for k, (_, t) in enumerate(ins):
        op = t.split()[0]
        if op.startswith("s_cbranch") or op in ("s_branch", "s_endpgm"):
            starts.add(k + 1)
            ti = target_index(ins, labels, t)
            if ti is not None:
                starts.add(ti)
    starts = sorted(s for s in starts if s < len(ins))
    name = {v: k for k, v in labels.items()}
    bl = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(ins)
        bl.append((s, e, name.get(s, f"@{s}")))
    idx = {s: i for i, (s, _, _) in enumerate(bl)}
    succ = collections.defaultdict(list)
    for i, (s, e, _) in enumerate(bl):
        last = ins[e - 1][1]
        op = last.split()[0]
        ti = target_index(ins, labels, last) if (op.startswith("s_cbranch") or op == "s_branch") else None
        if ti is not None and ti in idx:
            succ[i].append(idx[ti])
        if op != "s_branch" and op != "s_endpgm" and i + 1 < len(bl):
            succ[i].append(i + 1)
    return bl, succ


def main():
    ins, labels = parse(sys.argv[1])
    bl, succ = blocks(ins, labels)
    pred = collections.defaultdict(list)
    for a, ss in succ.items():
        for b in ss:
            pred[b].append(a)
    back = [(a, b) for a, ss in succ.items() for b in ss if b <= a]
    print(f"== {len(bl)} basic blocks, {sum(len(v) for v in succ.values())} edges, {len(back)} back edges")
    for a, h in back:
        body = {h}
        stack = [a]
        while stack:
            x = stack.pop()
            if x not in body:
                body.add(x)
                stack.extend(pred[x])
        exits = sorted({(x, y) for x in body for y in succ[x] if y not in body})
        side = sorted({x for x in body if x != h for p in pred[x] if p not in body})
        cold = sorted(x for x in body if x > a)
        n = sum(bl[x][1] - bl[x][0] for x in body)
        print(f"   loop header {bl[h][2]} (latch {bl[a][2]}): {len(body)} blocks, {n} instructions")
        print(f"      exits ({len(exits)}): " + ", ".join(f"{bl[x][2]} -> {bl[y][2]}" for x, y in exits))
        if side:
            print("      entries other than the header: " + ", ".join(bl[x][2] for x in side))
        if cold:
            print("      loop blocks laid out after the latch (out-of-line): " + ", ".join(bl[x][2] for x in cold))
    multi = [b for b in range(len(bl)) if len(pred[b]) > 1]
    print(f"== join blocks (> 1 predecessor): {len(multi)}: " + ", ".join(f"{bl[b][2]}({len(pred[b])})" for b in multi[:30]))


if __name__ == "__main__":
    main()
