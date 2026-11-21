#!/usr/bin/env python3
"""Count write-after-read register reuses between WMMAs and later loads in gfx1250 asm: a load (ds_load / global_load / buffer_load)
whose destination VGPRs a WMMA issued at most WINDOW instructions earlier still reads (A / B / C operands). In mode 0 the hardware
interlocks such a write until the WMMA has read its sources; in mode 2 it costs an s_wait_alu. VGPR windows (s_set_vgpr_msb) are tracked,
so registers compare as absolute numbers. Reports per kernel: total sites, sites inside loops (between a label and a backward branch to it),
and the distance histogram.
usage: war_reuse.py file.s [WINDOW]"""
import re, sys
from collections import Counter

def regs(tok, w):
    t = tok.strip()
    m = re.fullmatch(r"v\[(\d+):(\d+)\]", t)
    if m:
        return [x + 256 * w for x in range(int(m.group(1)), int(m.group(2)) + 1)]
    m = re.fullmatch(r"v(\d+)", t)
    return [int(m.group(1)) + 256 * w] if m else []

def main():
    path = sys.argv[1]
    win = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    lines = [l.split("//")[0].strip() for l in open(path)]
    labels = {}
    insts = []
    for l in lines:
        if not l or l.startswith("."):
            continue
        if l.endswith(":"):
            labels[l[:-1]] = len(insts)
            continue
        insts.append(l)
    # loop spans: a backward branch to a label
    loops = []
    for i, l in enumerate(insts):
        mn = l.split()[0]
        if mn.startswith("s_cbranch") or mn == "s_branch":
            tgt = l.split()[1].split(",")[0] if len(l.split()) > 1 else ""
            if tgt in labels and labels[tgt] <= i:
                loops.append((labels[tgt], i))
    in_loop = lambda i: any(a <= i <= b for a, b in loops)
    msb = [0, 0, 0, 0]
    recent = []  # (index, set of regs read by a WMMA)
    sites = []
    for i, l in enumerate(insts):
        mn = l.split()[0]
        ops = [o.strip() for o in l[len(mn):].split(",")] if " " in l else []
        if mn == "s_set_vgpr_msb":
            v = int(ops[0], 0)
            msb = [(v >> 6) & 3, v & 3, (v >> 2) & 3, (v >> 4) & 3]
            continue
        if mn.startswith("v_wmma") or mn.startswith("v_swmmac"):
            rd = set()
            for k, o in enumerate(ops[1:4]):
                rd |= set(regs(o.split()[0], msb[k + 1]))
            recent.append((i, rd))
        if (mn.startswith("ds_load") or mn.startswith("ds_read") or mn.startswith("global_load") or mn.startswith("buffer_load")) and ops:
            dst = set(regs(ops[0], msb[0]))
            for (j, rd) in reversed(recent):
                if i - j > win:
                    break
                if dst & rd:
                    sites.append((i, i - j, in_loop(i)))
                    break
        recent = [x for x in recent if i - x[0] <= win]
    loop_sites = [s for s in sites if s[2]]
    print(f"{path}: {len(sites)} WAR-near reuse site(s) within {win} instructions ({len(loop_sites)} inside loops); distance histogram {sorted(Counter(d for _, d, _ in sites).items())}")

main()
