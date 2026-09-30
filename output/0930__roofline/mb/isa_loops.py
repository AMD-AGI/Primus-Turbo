import re,sys,collections
s=open(sys.argv[1]).read().split('\n')
func=None; lines=[]
funcs=collections.OrderedDict()
for l in s:
    m=re.match(r'^(\S+):\s*;\s*@',l)
    if m: func=m.group(1); funcs[func]=[]; continue
    if func: funcs[func].append(l)
for f,ls in funcs.items():
    # find backward branches: s_cbranch_* .LBBx_y where label defined earlier
    labels={}
    for i,l in enumerate(ls):
        m=re.match(r'^(\.LBB\d+_\d+):',l)
        if m: labels[m.group(1)]=i
    for i,l in enumerate(ls):
        m=re.search(r's_cbranch_\w+\s+(\.LBB\d+_\d+)',l)
        if m and m.group(1) in labels and labels[m.group(1)]<i:
            body=[x.strip().split()[0] for x in ls[labels[m.group(1)]+1:i+1] if x.strip() and not x.strip().startswith(('.',';'))]
            c=collections.Counter(body)
            key=lambda p: sum(v for k,v in c.items() if k.startswith(p))
            print(f"{f[:14]:14s} n={len(body):4d} wmma={key('v_wmma'):3d} exp={key('v_exp'):3d} fma={key('v_fma')+key('v_pk_fma'):3d} nop={key('v_nop'):3d} delay={key('s_delay_alu'):3d} other={len(body)-key('v_wmma')-key('v_exp')-key('v_fma')-key('v_pk_fma')-key('v_nop')-key('s_delay_alu')}")
