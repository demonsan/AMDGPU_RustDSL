import re,sys,collections
blocks=[];cur=None
for l in open(sys.argv[1]):
    s=l.split(';')[0].strip()
    if not s: continue
    if s.endswith(':') and (s.startswith('.LBB') or s.startswith('BB')):
        cur=[s,[]];blocks.append(cur);continue
    if s.startswith('.') or cur is None: continue
    cur[1].append(s)
for b in blocks:
    n=len(b[1])
    if n<30: continue
    c=collections.Counter(i.split()[0].replace('_e32','').replace('_e64','') for i in b[1])
    valu=sum(v for k,v in c.items() if k.startswith('v_') and not k.startswith('v_mfma'))
    top=', '.join('%s:%d'%(k,v) for k,v in c.most_common(14))
    print('%s n=%d valu=%d mfma=%d | %s'%(b[0],n,valu,sum(v for k,v in c.items() if k.startswith('v_mfma')),top))