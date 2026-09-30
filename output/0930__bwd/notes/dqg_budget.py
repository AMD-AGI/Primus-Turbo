import json,glob,sys,collections
d=sys.argv[1]; lo,hi=int(sys.argv[2]),int(sys.argv[3])
code=json.load(open(d+'/code.json'))['code']
va={i:c[5] for i,c in enumerate(code)}; txt={i:c[0] for i,c in enumerate(code)}
def cls(ins,nxt):
    op=ins.split()[0]
    if 'wmma' in op: return 'wmma->ds (switch)' if nxt.startswith('ds_') else 'wmma'
    if op.startswith('ds_load'): return 'ds_load'
    if op.startswith('ds_store'): return 'ds_store'
    if op.startswith(('buffer_','global_')): return 'vmem issue'
    if op.startswith('s_wait'): return op
    if op.startswith('s_'): return 'salu'
    if op.startswith('v_exp'): return 'v_exp'
    if op.startswith('v_pk'): return 'v_pk'
    if op=='v_nop': return 'v_nop'
    if op.startswith('v_cvt'): return 'v_cvt'
    return 'valu other'
groups={'all':None,'fast':'1,2,3,5,6,7,9,10,11,13,14,15','slow':'0,4,8,12'}
out={}
for g,sel in groups.items():
    acc=collections.Counter(); trips=0; big=collections.Counter()
    for f in glob.glob(d+'/se*_wv*.json'):
        wv=f.split('wv')[1][:-5]
        if sel and wv not in sel.split(','): continue
        ins=json.load(open(f))['wave']['instructions']
        for a,b in zip(ins,ins[1:]):
            v=va.get(a[4])
            if v is None or not(lo<=v<=hi): continue
            if v==lo: trips+=1
            c=cls(txt[a[4]],txt[b[4]]); dt=b[0]-a[0]; acc[c]+=dt
            if c=='wmma' and dt>=12: big['wmma gap>=12']+=dt-8
    out[g]={k:v/trips for k,v in acc.items()}; out[g]['_trips']=trips; out[g]['_big']=big['wmma gap>=12']/trips
keys=sorted(out['all'],key=lambda k:-out['all'][k])
print('%-22s %8s %8s %8s'%('class','all','fast','slow'))
for k in keys: print('%-22s %8.1f %8.1f %8.1f'%(k,out['all'].get(k,0),out['fast'].get(k,0),out['slow'].get(k,0)))
print('%-22s %8.1f %8.1f %8.1f'%('TOTAL',*[sum(v for k,v in out[g].items() if not k.startswith('_')) for g in groups]))
