import re,sys
def probe(path,tag):
    t=open(path).read()
    parts=re.split(r"^(\.LBB\S+):",t,flags=re.M)
    out=[]
    for i in range(1,len(parts),2):
        b=[l.strip() for l in parts[i+1].split("\n") if l.strip() and not l.strip().startswith((".",";"))]
        w=sum(1 for l in b if l.startswith("v_wmma"))
        if w==0: continue
        N=len(b)
        loads=[k for k,l in enumerate(b) if l.startswith("buffer_load_b128")]
        fence=[k for k,l in enumerate(b) if "s_wait_loadcnt_dscnt" in l]
        barr=[k for k,l in enumerate(b) if l.startswith("s_barrier")]
        last=max(loads) if loads else None
        cov=None
        if last is not None:
            if fence:
                f=fence[0]; cov=(f-last) if f>last else (N-last)+f
            else: cov=f"跨迭代(N-{last}+?)"
        out.append((parts[i],N,w,len(loads),barr,fence,last,cov))
    print(f"  === {tag} ===")
    for n,N,w,nl,barr,fence,last,cov in out:
        print(f"    {n:12s} N={N:4d} wmma={w:3d} b128={nl:3d} barrier@{barr} fence@{fence} lastload=rel-{last} cover={cov}")
for tag,p in (("4wave基线",sys.argv[1]),("S1",sys.argv[2]),("冠军",sys.argv[3])): probe(p,tag)
