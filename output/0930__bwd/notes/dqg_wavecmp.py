import json,glob,sys,collections
d=sys.argv[1]; lo,hi=int(sys.argv[2]),int(sys.argv[3])
code=json.load(open(d+'/code.json'))['code']
va={i:c[5] for i,c in enumerate(code)}; txt={i:c[0] for i,c in enumerate(code)}
def cls(ins):
    op=ins.split()[0]
    if 'wmma' in op: return 'wmma'
    if op.startswith('ds_'): return op[:8]
    if op.startswith(('buffer_','global_')): return 'vmem'
    if op.startswith('s_wait'): return op+' '+ins.split()[1]
    if op.startswith('s_'): return 'salu'
    if op.startswith('v_exp'): return 'v_exp'
    if op.startswith('v_pk'): return 'v_pk'
    return 'valu'
res={}
for f in sorted(glob.glob(d+'/se*_wv*.json'),key=lambda s:int(s.split('wv')[1][:-5])):
    ins=json.load(open(f))['wave']['instructions']
    acc=collections.Counter(); trips=0; site=collections.Counter()
    for a,b in zip(ins,ins[1:]):
        v=va.get(a[4])
        if v is None or not(lo<=v<=hi): continue
        if v==lo: trips+=1
        dt=b[0]-a[0]; acc[cls(txt[a[4]])]+=dt; site[(v,txt[a[4]][:50])]+=dt
    res[f.split('_')[-1][:-5]]=(trips,{k:v/trips for k,v in acc.items()},{k:v/trips for k,v in site.items()})
keys=sorted({k for r in res.values() for k in r[1]}, key=lambda k:-res['wv1'][1].get(k,0))
print('%-26s'%'class', ' '.join('%7s'%w for w in res))
for k in keys: print('%-26s'%k[:26], ' '.join('%7.0f'%r[1].get(k,0) for r in res.values()))
print('%-26s'%'TOTAL', ' '.join('%7.0f'%sum(r[1].values()) for r in res.values()))
if len(sys.argv)>4:
  a,b=sys.argv[4],sys.argv[5]
  sa,sb=res[a][2],res[b][2]
  df=sorted(((sa.get(k,0)-sb.get(k,0)),k) for k in set(sa)|set(sb))
  for x,k in df[-15:][::-1]: print('%7.1f'%x, '%7.1f %7.1f'%(sa.get(k,0),sb.get(k,0)), k)
