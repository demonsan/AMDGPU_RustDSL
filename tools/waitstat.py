#!/usr/bin/env python3
"""Wait census per basic block: python3 waitstat.py file.s  -> s_waitcnt forms per block (block size, vmcnt/lgkmcnt values)."""
import re,sys,collections
blocks=[];cur=None
for l in open(sys.argv[1]):
    s=l.split(';')[0].strip()
    if not s: continue
    if s.endswith(':') and (s.startswith('.LBB') or s.startswith('BB')):
        cur=[s[:-1],[]];blocks.append(cur);continue
    if s.startswith('.') or cur is None: continue
    cur[1].append(s)
tot=collections.Counter()
for b in blocks:
    ws=[i for i in b[1] if i.startswith('s_waitcnt')]
    if not ws: continue
    forms=collections.Counter(re.sub(r'^s_waitcnt\s*','',w) for w in ws)
    bar=sum(1 for i in b[1] if i=='s_barrier')
    print('%-9s n=%3d waits=%2d barrier=%d  %s' % (b[0],len(b[1]),len(ws),bar,dict(forms)))
    tot.update(forms)
print('TOTAL static', sum(tot.values()), dict(tot))