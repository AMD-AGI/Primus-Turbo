import json,glob,sys,collections
d=sys.argv[1]; lo,hi=int(sys.argv[2]),int(sys.argv[3]); waves=sys.argv[4].split(',')
code=json.load(open(d+'/code.json'))['code']
va={i:c[5] for i,c in enumerate(code)}; txt={i:c[0] for i,c in enumerate(code)}
def cls(ins):
    op=ins.split()[0]
    if 'wmma' in op: return 'W'
    if op.startswith('ds_load'): return 'DL'
    if op.startswith('ds_store'): return 'DS'
    if op.startswith(('buffer_','global_')): return 'VM'
    if op.startswith('s_wait'): return 'wait'
    if op.startswith('s_'): return 's'
    if op.startswith('v_exp'): return 'X'
    if op=='v_nop': return 'nop'
    return 'v'
site=collections.Counter(); trips=0
for wv in waves:
    ins=json.load(open(f'{d}/se0_sm3_sl0_{wv}.json'))['wave']['instructions']
    for a,b in zip(ins,ins[1:]):
        v=va.get(a[4])
        if v is None or not(lo<=v<=hi): continue
        if v==lo: trips+=1
        site[v]+=b[0]-a[0]
addrs=sorted(site); t={c[5]:c[0] for c in code if lo<=c[5]<=hi and c[6]}
# runs
runs=[];cur=None
for v in addrs:
    c=cls(t[v]); 
    if c.startswith('wait'): c='wait:'+t[v].split()[0][7:]+' '+t[v].split()[1]
    if cur and cur[0]==c: cur[1]+=1; cur[2]+=site[v]/trips; cur[4]=v
    else:
        cur=[c,1,site[v]/trips,v,v]; runs.append(cur)
cum=0
for c,n,w,a,b in runs:
    cum+=w
    print(f"{a:>6}-{b:<6} {c:18s} x{n:<4d} wall {w:7.1f} cum {cum:7.0f}")
