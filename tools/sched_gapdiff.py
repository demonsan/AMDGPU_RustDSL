import re,sys
def load(p):
    seq=[];gap=[0,[]]
    started=False
    for l in open(p):
        l=l.split(';')[0].strip()
        if not l or l.startswith('.') and not l.startswith('.LBB'): continue
        if l.endswith(':'):
            continue
        m=l.split()[0]
        if not (m.startswith('v_') or m.startswith('s_') or m.startswith('ds_') or m.startswith('global_')): continue
        if m in ('s_branch','s_cbranch_vccnz','s_cbranch_vccz','s_endpgm'): continue
        if m=='s_nop':
            gap[0]+=int(l.split()[1])+1;continue
        if m=='s_waitcnt':
            gap[1].append(l.split(None,1)[1]);continue
        seq.append((l,gap[0],gap[1]));gap=[0,[]]
    return seq
a=load(sys.argv[1]);b=load(sys.argv[2])
print(len(a),len(b))
n=0
for i,(x,y) in enumerate(zip(a,b)):
    nm=lambda s: re.sub(r'_e(32|64)$','',s.split()[0])
    if nm(x[0])!=nm(y[0]):
        print("MISMATCH at",i,x[0],y[0]);break
    if y[1]<x[1] or (set(x[2])-set(y[2])):
        n+=1
        if y[1]<x[1] and not re.match(r'v_(cndmask|addc)',x[0]): print(i,x[0][:60],'| llvm nop',x[1],'waits',x[2],'| own nop',y[1],'waits',y[2],'| prev:',a[i-1][0][:50])
print('total diffs',n)