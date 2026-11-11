#!/usr/bin/env python3
"""Instruction-form coverage: which instruction forms of a kernel (e.g. the vendor disassembly) our backend already emits somewhere.

A form is the mnemonic (encoding suffix dropped) plus the operand kinds: register file and width (v1, v8, s2, a16, vcc, m0, exec ...),
inline constant / literal, and the modifiers (op_sel:[..], offen, lds, offset:, glc ...). The corpus is any set of our emitted kernels
(`amdc emit` text), e.g. every kernel of the md5 manifest. Prints, per vendor form, whether the same form, only the mnemonic, or nothing
occurs in the corpus, with instruction counts (docs/fmha-replica.md R2).

usage: isa_forms.py kernel.dis corpus1.s [corpus2.s ...] [--show N]
"""
import collections
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asm_diff import parse  # noqa: E402

SPECIAL = {"vcc", "vcc_lo", "vcc_hi", "m0", "exec", "exec_lo", "exec_hi", "scc", "off", "null"}


def kind(tok):
    t = tok.strip()
    m = re.match(r"^([vsa])\[(\d+):(\d+)\]$", t)
    if m:
        return f"{m.group(1)}{int(m.group(3)) - int(m.group(2)) + 1}"
    if re.match(r"^[vsa]\d+$", t):
        return t[0] + "1"
    if t in SPECIAL:
        return t
    if re.match(r"^-?(0x[0-9a-fA-F]+|\d+(\.\d+)?)$", t):
        v = int(t, 16) if t.lower().startswith(("0x", "-0x")) else float(t)
        return "ic" if -16 <= v <= 64 and float(v).is_integer() else "lit"
    m = re.match(r"^([a-z_]+)(:|\()", t)
    if m:
        return m.group(1) + ("()" if m.group(2) == "(" else ":")
    m = re.match(r"^-?\|?([vs])", t)
    if m and re.match(r"^-?\|?[vs]\d+\|?$", t):
        return "mod." + m.group(1) + "1"
    return t if re.match(r"^[a-z_]+$", t) else "?"


def form(text):
    t = re.sub(r"\s*<[^>]*>", "", text).strip()
    parts = t.split(None, 1)
    op = re.sub(r"_e(32|64)$", "", parts[0])
    if op.startswith("s_cbranch") or op == "s_branch":
        return op, op
    toks = []
    if len(parts) > 1:
        for x in re.split(r",(?![^\[]*\])|\s+(?=[a-z_]+(?::|\b))", parts[1]):
            x = x.strip().rstrip(",")
            if x:
                toks.append(kind(x))
    return op, op + " " + " ".join(toks)


def forms(path):
    ins, _ = parse(path)
    return collections.Counter(form(t) for _, t in ins)


def main():
    argv = sys.argv[1:]
    show = 40
    if "--show" in argv:
        i = argv.index("--show")
        show = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    args = argv
    target = forms(args[0])
    corpus = collections.Counter()
    for p in args[1:]:
        corpus.update(forms(p))
    have_form = {f for (_, f) in corpus}
    have_op = {o for (o, _) in corpus}
    rows = {"same form": [], "mnemonic only": [], "absent": []}
    for (op, f), n in target.items():
        k = "same form" if f in have_form else "mnemonic only" if op in have_op else "absent"
        rows[k].append((n, f))
    tot = sum(target.values())
    for k, v in rows.items():
        n = sum(x for x, _ in v)
        print(f"== {k}: {len(v)} forms, {n} instructions ({n / max(1, tot):.1%})")
        if k != "same form":
            for x, f in sorted(v, reverse=True)[:show]:
                print(f"   {x:6d}  {f}")


if __name__ == "__main__":
    main()
