"""Max outstanding LDS (dscnt) ops at every s_barrier_signal, dataflow over the ISA CFG."""
import sys,re
L=[l.split(';')[0].strip() for l in open(sys.argv[1]).read().split('\n')]
st=[i for i,l in enumerate(L) if l.startswith('kn_') and l.endswith(':')][0]
en=[i for i,l in enumerate(L) if l.startswith('.Lfunc_end')][0]
body=[(i,L[i]) for i in range(st+1,en) if L[i] and not L[i].startswith('.p2align') and not L[i].startswith(';')]
# blocks
blocks=[];cur=None;labels={}
for i,t in body:
    if re.match(r'^\.LBB\S+:$',t):
        cur={'label':t[:-1],'ins':[]};blocks.append(cur);labels[t[:-1]]=len(blocks)-1;continue
    if cur is None: cur={'label':'entry','ins':[]};blocks.append(cur)
    if t.startswith('.'): continue
    cur['ins'].append((i,t))
    if t.startswith('s_branch') or t.startswith('s_cbranch') or t.startswith('s_endpgm'):
        cur={'label':f'fall{i}','ins':[]};blocks.append(cur)
succ={}
for b,bl in enumerate(blocks):
    s=[];last=bl['ins'][-1][1] if bl['ins'] else ''
    if last.startswith('s_branch'): s=[labels[last.split()[1]]]
    elif last.startswith('s_cbranch'): s=[labels[last.split()[1]],b+1]
    elif last.startswith('s_endpgm'): s=[]
    else: s=[b+1] if b+1<len(blocks) else []
    succ[b]=s
inn={0:0};work=[0];res={}
while work:
    b=work.pop();o=inn[b]
    for i,t in blocks[b]['ins']:
        if t.startswith('s_barrier_signal'): res[i]=max(res.get(i,0),o)
        if t.startswith('ds_'): o+=1
        m=re.match(r's_wait_dscnt (0x[0-9a-f]+|\d+)',t)
        if m: o=min(o,int(m.group(1),0))
        o=min(o,64)
    for s in succ[b]:
        if s<len(blocks) and (s not in inn or inn[s]<o): inn[s]=o;work.append(s)
for i in sorted(res): print(f"line {i+1}: max outstanding ds at s_barrier_signal = {res[i]}")
