#!/usr/bin/env python3
"""Attribute the s_nop wait states of the steady loop of an emitted .s (the largest loop: a back edge to an earlier label).

usage: python3 tools/nop_attrib.py <kernel.s> [--all]
Each `s_nop N` (N+1 wait states) is classified by its neighbours:
  m0->lds-dma   : `s_mov_b32 m0, ...` before, an LDS-DMA (global_load_lds / buffer_load ... lds) after
  valu->mfma    : a VALU write before, an MFMA / smfmac after (VALU write -> matrix operand)
  valu->rfl/perm/dpp : a VALU write before, v_readfirstlane / v_permlane / DPP after
  mfma->*       : an MFMA earlier in the run of instructions before the nop, a reader after (result latency / SrcC WAR)
  trans->*      : v_exp / v_log / v_rcp before
  vcc           : a v_cmp / carry producer before, a vcc consumer / branch after
  store->valu   : a wide VMEM store before (data WAR)
  block-end     : the nop is the last instruction before a branch / label
  other
"""
import re, sys

lines = open(sys.argv[1]).read().split("\n")
labels = {m.group(1): i for i, l in enumerate(lines) if (m := re.match(r"^(BB\d+):", l))}
loops = []
for i, l in enumerate(lines):
    m = re.match(r"^\s+s_(?:cbranch_\w+|branch)\s+(BB\d+)", l)
    if m and labels.get(m.group(1), 10**9) < i:
        a = labels[m.group(1)]
        loops.append((i - a, a, i))
lo, hi = (0, len(lines)) if "--all" in sys.argv or not loops else max(loops)[1:]

def ins(i):
    s = lines[i].strip()
    return s if re.match(r"^(v_|s_|ds_|buffer_|global_)", s) else None

def cls(s):
    if s is None:
        return "label"
    op = s.split()[0]
    if op.startswith(("v_mfma", "v_smfmac")):
        return "mfma"
    if op.startswith(("v_exp", "v_log", "v_rcp")):
        return "trans"
    if op.startswith("v_readfirstlane") or op.startswith("v_permlane") or "_dpp" in op:
        return "rfl"
    if op.startswith(("v_cmp", "v_add_co", "v_addc", "v_cndmask")):
        return "vcc" if "vcc" in s or op.startswith(("v_cmp_", "v_add_co")) else "valu"
    if op.startswith("v_"):
        return "valu"
    if op.startswith(("global_load_lds",)) or (op.startswith("buffer_load") and s.endswith(" lds")):
        return "dma"
    if op.startswith(("global_store", "buffer_store")) and ("x4" in op or "x3" in op):
        return "wstore"
    if op.startswith(("s_cbranch", "s_branch")):
        return "branch"
    if op == "s_mov_b32" and s.split()[1].startswith("m0"):
        return "m0"
    if op == "s_nop":
        return "nop"
    return op

cat = {}
total = 0
for i in range(lo, hi + 1):
    s = ins(i)
    if not s or not s.startswith("s_nop"):
        continue
    n = int(s.split()[1]) + 1
    total += n
    j = i - 1
    while j >= lo and (ins(j) is None and not re.match(r"^BB\d+:", lines[j]) or (ins(j) or "").startswith("s_nop")):
        j -= 1
    k = i + 1
    while k <= hi and (ins(k) is None and not re.match(r"^BB\d+:", lines[k]) or (ins(k) or "").startswith("s_nop")):
        k += 1
    prev, nxt = cls(ins(j) if j >= lo else None), cls(ins(k) if k <= hi else None)
    # an MFMA among the 6 instructions before the nop (latency / WAR windows span several instructions)
    recent_mfma = any(cls(ins(x)) == "mfma" for x in range(max(lo, i - 6), i) if ins(x))
    if prev == "m0" and nxt == "dma":
        c = "m0->lds-dma"
    elif nxt in ("branch", "label"):
        c = "block-end"
    elif prev == "trans":
        c = "trans->" + nxt
    elif prev == "wstore":
        c = "store->" + nxt
    elif prev == "vcc" or nxt == "vcc":
        c = "vcc"
    elif prev == "valu" and nxt == "mfma":
        c = "valu->mfma"
    elif prev == "valu" and nxt == "rfl":
        c = "valu->rfl/perm/dpp"
    elif recent_mfma:
        c = "mfma->" + nxt
    else:
        c = f"other ({prev}->{nxt})"
    cat.setdefault(c, [0, 0])
    cat[c][0] += 1
    cat[c][1] += n
print(f"range lines {lo}..{hi}: {total} wait states in s_nop")
for c, (k, n) in sorted(cat.items(), key=lambda x: -x[1][1]):
    print(f"  {c:28s} {k:4d} nops {n:4d} states")
