import csv,sys,collections
f=sys.argv[1]; H=int(sys.argv[2])
rows=list(csv.DictReader(open(f)))
L=[r for r in rows if r['Hitcount']==str(H)]
def cls(ins):
    op=ins.split()[0]
    if 'wmma' in op: return 'wmma'
    if op.startswith('ds_load'): return 'ds_load'
    if op.startswith('ds_store'): return 'ds_store'
    if op.startswith('ds_'): return 'ds_other'
    if op.startswith(('buffer_load','global_load','tensor_')): return 'vmem_ld'
    if op.startswith(('buffer_','global_')): return 'vmem_other'
    if op.startswith('s_wait'): return op
    if op.startswith('s_barrier'): return 'barrier'
    if op.startswith('s_'): return 'salu'
    if op in('v_exp_f32','v_exp_f32_e32') or op.startswith('v_exp'): return 'v_exp'
    if op.startswith('v_pk_'): return 'v_pk'
    if op.startswith('v_dual'): return 'v_dual'
    if op.startswith('v_nop'): return 'v_nop'
    if op.startswith(('v_mov','v_cndmask')): return 'v_mov/cnd'
    if op.startswith('v_cvt') : return 'v_cvt'
    if op.startswith('v_'): return 'valu_other'
    return 'other'
t=collections.defaultdict(lambda:[0,0,0,0])
for r in L:
    c=cls(r['Instruction']); x=t[c]; x[0]+=1; x[1]+=int(r['Latency']); x[2]+=int(r['Stall']); x[3]+=int(r['Idle'])
tot=sum(x[1] for x in t.values())
print('static instrs in loop',len(L),'addr',L[0]['Vaddr'],L[-1]['Vaddr'],'per-iter latency',tot/H)
for c,x in sorted(t.items(),key=lambda kv:-kv[1][1]):
    print(f"{c:18s} n={x[0]:4d} lat/it={x[1]/H:8.1f} ({100*x[1]/tot:4.1f}%) stall/it={x[2]/H:7.1f} idle/it={x[3]/H:7.1f}")
