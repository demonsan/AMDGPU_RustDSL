#!/usr/bin/env python3
"""Group the failing tests of a differential lit log by cause.
usage: conformance_classify.py lit-diff-<level>.log"""
import re, sys, collections

log = open(sys.argv[1]).read()
blocks = re.split(r"\*{20} TEST '", log)[1:]
groups = collections.defaultdict(list)
for b in blocks:
    name = b.split("'")[0].split("/")[-1]
    errs = [l[4:].strip() for l in b.split("\n") if l.startswith("# | ") and re.search(r"error:|Bad machine code|LLVM ERROR|Assertion|Stack dump", l)]
    # llc error text may also appear as "# | " lines before the DIFFERENTIAL MISMATCH marker
    if errs:
        e = errs[0]
        e = re.sub(r"\S*/llvm/test/\S*", "<T>", e)
        e = re.sub(r"[0-9]+", "N", e)[:100]
        groups["llc rejected the transformed MIR: " + e].append(name)
        continue
    m = re.search(r"# \| @@.*?\n((?:# \| .*\n){0,40})", b)
    ch = [l[4:].rstrip() for l in (m.group(1).split("\n") if m else []) if l.startswith("# | ") and l[4:5] in "-+" and not l[4:].startswith(("---", "+++"))]
    removed = [c for c in ch if c.startswith("-")]
    added = [c for c in ch if c.startswith("+")]
    if removed and not added and any("source_filename" in c or "target triple" in c for c in removed):
        groups["empty output (llc failed silently / no output)"].append(name)
    elif any("S_WAITCNT" in c or "S_WAIT_" in c or "s_waitcnt" in c for c in ch):
        groups["different wait insertion (conservative waits)"].append(name)
    elif any("S_NOP" in c or "s_nop" in c for c in ch):
        groups["different s_nop insertion"].append(name)
    else:
        groups["other instruction difference: " + (ch[0][:70] if ch else "?")].append(name)
for g, ns in sorted(groups.items(), key=lambda x: -len(x[1])):
    print("%3d  %s" % (len(ns), g))
    print("       " + ", ".join(sorted(ns))[:400])