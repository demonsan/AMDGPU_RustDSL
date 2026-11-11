#!/usr/bin/env python3
"""Instruction-level diff of two gfx950 kernels: ours (`amdc emit` text) vs a vendor `llvm-objdump -d` dump (or any two of either).

The streams are normalized and compared on three separate axes (docs/fmha-replica.md):
  1. opcode stream - the instructions without the fillers (s_waitcnt, s_nop), aligned with a longest-matching-blocks diff;
     mnemonic suffixes (_e32 / _e64) are dropped, branch targets abstracted;
  2. registers - over the aligned equal-opcode pairs: exact operand equality, and equality under the best register renaming (per
     register file, the majority A -> B map), so a pure renaming shows up as such;
  3. waits / nops - the fillers attached in front of each aligned pair: s_waitcnt (vmcnt / lgkmcnt / expcnt) and s_nop wait states,
     compared pair by pair (only in A, only in B, different counts).

usage: isa_diff.py A B [--range-a L1:L2] [--range-b L1:L2] [--show N] [--json out.json]
  --range-*: restrict a side to the instructions from label L1 up to label L2 (exclusive; empty = to the end)
"""
import argparse
import collections
import difflib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asm_diff import cls, parse  # noqa: E402

FILLERS = ("s_waitcnt", "s_nop")


def norm(text):
    """-> (mnemonic, operands list) with the encoding suffix dropped, hex immediates as decimal, branch targets abstracted."""
    t = re.sub(r"\s*<[^>]*>", "", text).strip()
    parts = t.split(None, 1)
    op = re.sub(r"_e(32|64)$", "", parts[0])
    ops = []
    if len(parts) > 1:
        body = parts[1]
        if op.startswith("s_cbranch") or op == "s_branch":
            body = "<target>"
        body = re.sub(r"\b0x([0-9a-fA-F]+)\b", lambda m: str(int(m.group(1), 16)), body)
        ops = [x.strip() for x in re.split(r",(?![^\[]*\])", body) if x.strip()]
    return op, ops


def region(ins, labels, rng):
    if not rng:
        return ins
    a, _, b = rng.partition(":")
    lo = labels[a] if a else 0
    hi = labels[b] if b else len(ins)
    return ins[lo:hi]


def split(ins):
    """-> list of (op, operands, text, fillers) for the non-filler instructions; fillers = the (op, operands) in front of each."""
    out, pend = [], []
    for _, t in ins:
        op, ops = norm(t)
        if op in FILLERS:
            pend.append((op, ops))
            continue
        out.append((op, ops, t, pend))
        pend = []
    if pend:
        out.append(("<end>", [], "<end>", pend))
    return out


REG = re.compile(r"^([vsa])(?:\[(\d+):(\d+)\]|(\d+))$")


def regs_of(tok):
    m = REG.match(tok)
    if not m:
        return None
    f = m.group(1)
    lo = int(m.group(2) if m.group(2) is not None else m.group(4))
    hi = int(m.group(3)) if m.group(3) is not None else lo
    return f, lo, hi


def wait_fields(ops):
    d = {}
    for o in ops:
        for m in re.finditer(r"(vmcnt|lgkmcnt|expcnt)\((\d+)\)", o):
            d[m.group(1)] = int(m.group(2))
        if re.fullmatch(r"\d+", o):  # raw immediate form
            d["raw"] = int(o)
    return d


def filler_summary(fl):
    waits = [wait_fields(o) for op, o in fl if op == "s_waitcnt"]
    nops = sum(int(o[0]) + 1 if o and o[0].isdigit() else 1 for op, o in fl if op == "s_nop")
    return waits, nops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--range-a", default="")
    ap.add_argument("--range-b", default="")
    ap.add_argument("--show", type=int, default=20)
    ap.add_argument("--json")
    args = ap.parse_args()
    ia, la = parse(args.a)
    ib, lb = parse(args.b)
    A, B = split(region(ia, la, args.range_a)), split(region(ib, lb, args.range_b))
    rep = {"a": args.a, "b": args.b}

    # 1. classes and opcodes
    ca, cb = collections.Counter(cls(x[2]) for x in A), collections.Counter(cls(x[2]) for x in B)
    fa = collections.Counter(op for x in A for op, _ in x[3])
    fb = collections.Counter(op for x in B for op, _ in x[3])
    print(f"== instructions (fillers excluded): A {len(A)}  B {len(B)};  s_waitcnt A {fa['s_waitcnt']} B {fb['s_waitcnt']};  s_nop A {fa['s_nop']} B {fb['s_nop']}")
    print("== classes (A / B):")
    for c in sorted(set(ca) | set(cb), key=lambda c: -(ca[c] + cb[c])):
        print(f"   {c:18s} {ca[c]:6d} {cb[c]:6d}  {cb[c] - ca[c]:+d}")
    oa, ob = collections.Counter(x[0] for x in A), collections.Counter(x[0] for x in B)
    only = sorted(((op, oa[op], ob[op]) for op in set(oa) | set(ob) if oa[op] != ob[op]), key=lambda r: -abs(r[1] - r[2]))
    print("== opcode count differences (A / B), largest first:")
    for op, x, y in only[: args.show]:
        print(f"   {op:40s} {x:6d} {y:6d}")
    rep["classes"] = {c: [ca[c], cb[c]] for c in set(ca) | set(cb)}

    # 2. alignment of the opcode streams
    sa, sb = [x[0] for x in A], [x[0] for x in B]
    sm = difflib.SequenceMatcher(None, sa, sb, autojunk=False)
    pairs = []
    blocks = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            pairs.extend(zip(range(i1, i2), range(j1, j2)))
        else:
            blocks.append((tag, i1, i2, j1, j2))
    n_eq = len(pairs)
    print(f"== opcode alignment: {n_eq} equal pairs = {n_eq / max(1, len(A)):.1%} of A, {n_eq / max(1, len(B)):.1%} of B; {len(blocks)} differing blocks")
    for tag, i1, i2, j1, j2 in sorted(blocks, key=lambda b: -max(b[2] - b[1], b[4] - b[3]))[: args.show]:
        da = collections.Counter(sa[i1:i2])
        db = collections.Counter(sb[j1:j2])
        fmt = lambda c: ", ".join(f"{k}x{v}" for k, v in c.most_common(4))
        print(f"   {tag:7s} A[{i1}:{i2}] ({i2 - i1}) {fmt(da)}  |  B[{j1}:{j2}] ({j2 - j1}) {fmt(db)}")
    rep["alignment"] = {"equal_pairs": n_eq, "a": len(A), "b": len(B), "blocks": len(blocks)}

    # 3. registers over the equal pairs
    exact = 0
    votes = {f: collections.defaultdict(collections.Counter) for f in "vsa"}
    for i, j in pairs:
        if A[i][1] == B[j][1]:
            exact += 1
        for ta, tb in zip(A[i][1], B[j][1]):
            ra, rb = regs_of(ta), regs_of(tb)
            if ra and rb and ra[0] == rb[0]:
                votes[ra[0]][(ra[1], ra[2] - ra[1])][rb[1]] += 1
    rmap = {f: {k: c.most_common(1)[0][0] for k, c in v.items()} for f, v in votes.items()}
    renamed = 0
    for i, j in pairs:
        oka = len(A[i][1]) == len(B[j][1])
        for ta, tb in zip(A[i][1], B[j][1]):
            ra, rb = regs_of(ta), regs_of(tb)
            if ra and rb:
                oka &= ra[0] == rb[0] and rmap[ra[0]].get((ra[1], ra[2] - ra[1])) == rb[1]
            else:
                oka &= ta == tb
        renamed += oka
    conflicts = {f: sum(1 for c in v.values() if len(c) > 1) for f, v in votes.items()}
    print(f"== registers over the equal pairs: identical operands {exact / max(1, n_eq):.1%}; identical up to one consistent renaming {renamed / max(1, n_eq):.1%}")
    print(f"   register tuples used by more than one counterpart (renaming not a bijection): v {conflicts['v']}  s {conflicts['s']}  a {conflicts['a']}")
    rep["registers"] = {"identical": exact, "renamed": renamed, "conflicts": conflicts}

    # 4. waits and nops in front of the equal pairs
    stats = collections.Counter()
    samples = collections.defaultdict(list)
    for i, j in pairs:
        wa, na = filler_summary(A[i][3])
        wb, nb = filler_summary(B[j][3])
        key = None
        if wa and not wb:
            key = "wait only in A"
        elif wb and not wa:
            key = "wait only in B"
        elif wa and wb and wa != wb:
            key = "wait counts differ"
        if key:
            stats[key] += 1
            samples[key].append((A[i][2], wa, wb))
        if na != nb:
            k2 = "more nop states in A" if na > nb else "more nop states in B"
            stats[k2] += 1
            stats[k2 + " (states)"] += abs(na - nb)
            samples[k2].append((A[i][2], na, nb))
    print("== waits / nops in front of the equal pairs:")
    for k in ["wait only in A", "wait only in B", "wait counts differ", "more nop states in A", "more nop states in B"]:
        extra = f" ({stats[k + ' (states)']} states)" if "nop" in k else ""
        print(f"   {k:24s} {stats[k]:6d}{extra}")
        for s in samples[k][:3]:
            print(f"        before `{s[0]}`: A {s[1]}  B {s[2]}")
    rep["fillers"] = dict(stats)
    if args.json:
        json.dump(rep, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
