#!/usr/bin/env python3
"""Which instruction pairs does each s_nop sit between? usage: nop_pairs.py kernel.s   (loop region = blocks with MFMAs)"""
import re, sys, collections
def cls(l):
    m = l.split()[0]
    if m.startswith("v_mfma"): return "MFMA"
    if m.startswith("v_accvgpr"): return "accvgpr"
    if m.startswith("v_readfirstlane"): return "readfirstlane"
    if m.startswith("v_"): return "VALU"
    if m.startswith("s_waitcnt"): return "waitcnt"
    if m.startswith("s_mov_b32") and "m0" in l: return "s_mov m0"
    if m.startswith("s_"): return "SALU"
    if m.startswith("ds_read"): return "ds_read"
    if m.startswith("ds_write"): return "ds_write"
    if m.startswith("buffer_load") or m.startswith("global_load"): return "DMA/load"
    if m.startswith("buffer_store") or m.startswith("global_store"): return "store"
    return m
lines = [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.strip().startswith((".", "//", ";")) and not l.strip().endswith(":")]
pairs = collections.Counter(); cyc = collections.Counter(); ex = {}
total_nop = 0
for i, l in enumerate(lines):
    if not l.startswith("s_nop"): continue
    n = int(l.split()[1]) + 1; total_nop += n
    p = lines[i-1] if i else ""; q = lines[i+1] if i+1 < len(lines) else ""
    # skip over directly following nops for the consumer
    j = i+1
    while j < len(lines) and lines[j].startswith("s_nop"): j += 1
    q = lines[j] if j < len(lines) else ""
    k = (cls(p) if p and not p.startswith("s_nop") else "s_nop", cls(q) if q else "end")
    pairs[k] += 1; cyc[k] += n
    ex.setdefault(k, (p[:60], q[:60]))
print(f"{sys.argv[1]}: {sum(pairs.values())} s_nop instructions, {total_nop} wait-state cycles")
for k, c in sorted(cyc.items(), key=lambda kv: -kv[1]):
    print(f"  {k[0]:>14s} -> {k[1]:<14s} count={pairs[k]:4d} cycles={c:5d}   e.g. [{ex[k][0]}] ... [{ex[k][1]}]")