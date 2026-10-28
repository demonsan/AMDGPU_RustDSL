#!/usr/bin/env python3
"""Census of the .mir tests of test/CodeGen/AMDGPU: pre-RA (virtual registers in a body) vs post-RA
(physical registers only), and how many RUN lines use a tail pass.  usage: mir_census.py <test dir>"""
import os, re, sys

d = sys.argv[1]
TAIL = re.compile(r"post-RA-hazard-rec|si-insert-waitcnts|amdgpu-insert-delay-alu|amdgpu-wait-sgpr-hazards|si-memory-legalizer|si-pre-emit-peephole|si-insert-hard-clauses|si-mode-register|amdgpu-lower-vgpr-encoding|gcn-create-vopd|branch-relaxation|si-late-branch-lowering")
VREG = re.compile(r"(?<![\w.])%(?:\d+|[A-Za-z_]\w*)(?::|\b(?!\.))")
tot = pre = post = nobody = tail_pre = tail_post = 0
for f in sorted(os.listdir(d)):
    if not f.endswith(".mir"):
        continue
    t = open(os.path.join(d, f), errors="replace").read()
    tot += 1
    runs = [l for l in t.split("\n") if l.startswith("# RUN:")]
    istail = any(TAIL.search(l) for l in runs)
    bodies = re.findall(r"^body:\s*\|\n((?:(?: .*|)\n)+)", t, re.M)
    if not bodies:
        nobody += 1
        continue
    has_v = False
    for b in bodies:
        for l in b.split("\n"):
            l2 = re.sub(r"%(bb|stack|ir-block|const|fixed-stack|jump-table|ir\.|subreg)[.\w-]*", "", l)
            if re.search(r"%\d+\b|%[A-Za-z_]\w*:", l2):
                has_v = True
                break
        if has_v:
            break
    if has_v:
        pre += 1
        tail_pre += istail
    else:
        post += 1
        tail_post += istail
print("mir tests: %d  (no body section: %d)" % (tot, nobody))
print("pre-RA (virtual registers): %d   post-RA (physical only): %d" % (pre, post))
print("with a tail-pass RUN line: pre-RA %d, post-RA %d" % (tail_pre, tail_post))