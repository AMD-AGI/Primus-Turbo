import sys, re, collections
sys.path.insert(0,'/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-flydsl-attn-bwd-20260917-115934/rounds/015/_scratch/work')
from chain import bodies, real

def cls(op):
    if op.startswith('v_wmma'): return 'WMMA'
    if op.startswith('ds_store') or op.startswith('ds_write'): return 'LDS_store'
    if op.startswith('ds_load') or op.startswith('ds_read'): return 'LDS_load'
    if op.startswith(('buffer_load','global_load','flat_load','scratch_load')): return 'VMEM_load'
    if op.startswith(('buffer_store','global_store','flat_store','scratch_store')): return 'VMEM_store'
    if op.startswith('s_wait'): return 'WAIT'
    if op == 'v_nop': return 'v_nop'
    if op.startswith('s_set_vgpr_msb') or op.startswith('s_set'): return 's_set_prefix'
    if op.startswith('v_cvt') or op.startswith('v_pk_'): return 'VALU_cvt/pk'
    if op.startswith('v_exp') or op.startswith('v_rcp') or op.startswith('v_log'): return 'VALU_trans'
    if op.startswith('v_cmp') or op.startswith('v_cndmask'): return 'VALU_mask'
    if op.startswith('v_mov') or op.startswith('v_accvgpr'): return 'VALU_mov'
    if op.startswith('v_'): return 'VALU_other'
    if op.startswith('s_'): return 'SALU/ctrl'
    return 'other:'+op

for p in sys.argv[1:]:
    kname=p.split('/')[-2]
    lines, bs = bodies(p)
    for t,a,b in bs:
        body=[l.strip() for l in lines[a:b+1] if real(l)]
        if len(body)<200: continue
        c=collections.Counter(cls(l.split()[0]) for l in body)
        print(f'\n### {kname} {t} slots={len(body)}')
        for k,v in c.most_common():
            print(f'   {k:14s} {v:5d}  ({100*v/len(body):4.1f}%)')
        # sub-census of VALU_other
        sub=collections.Counter(l.split()[0] for l in body if cls(l.split()[0]) in ('VALU_other','SALU/ctrl','VALU_mov','VALU_cvt/pk'))
        print('   -- top opcodes in VALU_other/SALU/mov/cvt:')
        for k,v in sub.most_common(14): print(f'      {k:26s} {v:5d}')
