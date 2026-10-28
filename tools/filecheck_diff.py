#!/usr/bin/env python3
"""FileCheck stand-in for differential conformance.
FC_MODE=record : save the checked stdin under $GOLD_DIR, then run the real FileCheck (test still has to pass).
FC_MODE=compare: normalise stdin (drop register/instruction flags, memory operands, debug locations,
                 liveins lines, comments) and require equality with a recorded golden of the same test+RUN."""
import sys, os, re, hashlib, subprocess, difflib

real, gold, mode = os.environ["REAL_FILECHECK"], os.environ["GOLD_DIR"], os.environ["FC_MODE"]
args = sys.argv[1:]
if "--input-file" in " ".join(args) or "-input-file" in " ".join(args) or sys.stdin.isatty():
    sys.exit(subprocess.run([real] + args).returncode)
data = sys.stdin.buffer.read()
files = [a for a in args if not a.startswith("-")]
path = files[-1] if files else "nofile"
rest = [a for a in args if a != path]
key = os.path.basename(path) + "." + hashlib.sha1(" ".join(rest).encode()).hexdigest()[:8]

IFLAGS = r"(?:nofpexcept|nuw|nsw|exact|nnan|ninf|nsz|arcp|contract|afn|reassoc|nneg|disjoint|samesign|frame-setup|frame-destroy|unpredictable) "

def norm(b):
    out = []
    for l in b.decode(errors="replace").split("\n"):
        t = l.strip()
        if not t or t.startswith(";") or t.startswith("#") or t.startswith("liveins:"):
            continue
        t = re.sub(r"\b(killed|dead|undef|renamable|internal|early-clobber) ", "", t)
        t = re.sub(r" :: \(.*$", "", t)
        t = re.sub(r",? ?debug-location !\d+.*$", "", t)
        t = re.sub(r"= (?:" + IFLAGS + ")+", "= ", t)
        t = re.sub(r"^(?:" + IFLAGS + ")+", "", t)
        out.append(t)
    return out

if mode == "record":
    n = 0
    while os.path.exists("%s/%s.%d" % (gold, key, n)):
        n += 1
    open("%s/%s.%d" % (gold, key, n), "wb").write(data)
    sys.exit(subprocess.run([real] + args, input=data).returncode)

cands = sorted(f for f in os.listdir(gold) if f.startswith(key + "."))
new = norm(data)
best = None
for c in cands:
    g = norm(open(os.path.join(gold, c), "rb").read())
    if g == new:
        sys.exit(0)
    d = list(difflib.unified_diff(g, new, "golden", "ours", lineterm="", n=1))
    if best is None or len(d) < len(best):
        best = d
sys.stderr.write("DIFFERENTIAL MISMATCH (%s, %d goldens)\n" % (key, len(cands)))
sys.stderr.write("\n".join((best or ["no golden recorded"])[:40]) + "\n")
sys.exit(1)