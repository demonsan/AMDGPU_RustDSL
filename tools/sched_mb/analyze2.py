import json,collections,sys
n=json.load(open('tests2.json'))
raw=sys.argv[1] if len(sys.argv)>1 else 'run2_raw.txt'
rec=collections.defaultdict(dict)
for l in open(raw):
    if l.startswith('ERR') or l.startswith('MISSING'): print(l.strip())
    if not l.startswith('R '): continue
    h,t=l.split('|'); _,nm,w,r=h.split(); w=int(w); r=int(r)
    if r==0: continue
    rec[(nm,w)][r]=[tuple(int(x,16) if i==3 else int(x) for i,x in enumerate(s.split('/'))) for s in t.split()]
def per(nm,w,r,ovh):
    xs=rec[(nm,w)][r]
    if nm.startswith('t5_') or nm.startswith('u_'):
        v=[(a-m-ovh[w])/n[nm] for a,_,m,c in xs]
    else:
        v=[(a-48)/n[nm] for a,_,m,c in xs]
    return sum(v)/len(v), max(v), min(v)
ovh={}
for w in (1,4,8):
    if ('t5_markempty',w) in rec:
        ovh[w]=sum(sum(a-m for a,_,m,c in rec[('t5_markempty',w)][r])/len(rec[('t5_markempty',w)][r]) for r in rec[('t5_markempty',w)])/3
print('mark overhead',ovh)
for (nm,w),rs in rec.items():
    if nm=='t5_markempty' or len(rs)<3: continue
    m=[per(nm,w,r,ovh)[0] for r in rs]
    wv=[per(nm,w,r,ovh) for r in rs]
    spread=(max(m)-min(m))/max(min(m),1e-9)
    chk=sorted(set(c for r in rs for *_,c in rec[(nm,w)][r]))
    print("%-26s w=%d mean=%8.2f wavemin=%8.2f wavemax=%8.2f spread=%5.1f%% %s chk=%s"%(nm,w,sum(m)/3,min(x[2] for x in wv),max(x[1] for x in wv),spread*100,"V" if spread<0.05 else "E",[hex(c) for c in chk][:2]))
