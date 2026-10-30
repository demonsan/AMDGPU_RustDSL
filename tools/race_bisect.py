#!/usr/bin/env python3
"""ddmin over the set of blocks the full list scheduler is allowed to reorder (AMDC_SCHED_OPTS=only=a+b+..): finds a minimal block set that still
makes the biasacc,vpre build produce an output different from the reference md5. Run on the GPU node from the repo root."""
import os, re, subprocess, sys

REF = os.environ.get("REF_MD5", "c8b7a893e41c")
RUNS = os.environ.get("RUNS", "3")
OPT = "cvt4,pksub,lazy,msum,permlane,kdlds,dbuf,lacc1,psh2,svpk,svcorr,swz,biasacc,vpre"
cache = {}

def race(blocks):
    key = tuple(sorted(blocks))
    if key in cache:
        return cache[key]
    env = dict(os.environ, FMHA_OPT=OPT, AMDC_SCHED="1", AMDC_SCHED_OPTS="only=" + "+".join(map(str, key)), TAG="bis", RUNS=RUNS)
    out = subprocess.run(["bash", "tools/race_repro.sh"], env=env, capture_output=True, text=True).stdout
    md5s = re.findall(r"md5=(\w+)", out)
    bad = any(m != REF for m in md5s) or len(md5s) == 0
    print(f"test {len(key)} blocks {key[:12]}{'...' if len(key) > 12 else ''}: {md5s} -> {'RACE' if bad else 'ok'}", flush=True)
    cache[key] = bad
    return bad

n_blocks = int(sys.argv[1])
cur = list(range(n_blocks))
assert race(cur), "the full set must reproduce"
chunk = len(cur) // 2
while chunk >= 1:
    i = 0
    while i < len(cur):
        trial = cur[:i] + cur[i + chunk:]
        if trial and race(trial):
            cur = trial      # this chunk is not needed
        else:
            i += chunk
    chunk //= 2
print("MINIMAL BLOCK SET:", cur, flush=True)