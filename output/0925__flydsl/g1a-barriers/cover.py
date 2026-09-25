"""Cover distance: instructions from each prefetch load to the next FULL drain wait."""
import re,sys
def analyse(path,tag):
    lines=[l.strip() for l in open(path) if l.strip() and not l.strip().startswith((".",";","//"))]
    ins=[l for l in lines if re.match(r"^[a-z_0-9]+\s", l) or re.match(r"^s_\w+$", l)]
    # 找热体：最大的 .LBB 块
    txt=open(path).read()
    blocks=re.split(r"^(\.LBB\S+):", txt, flags=re.M)
    best=None
    for i in range(1,len(blocks),2):
        name,body=blocks[i],blocks[i+1]
        n=len([l for l in body.split("\n") if l.strip() and not l.strip().startswith((".",";"))])
        if best is None or n>best[1]: best=(name,n,body)
    name,n,body=best
    bl=[l.strip() for l in body.split("\n") if l.strip() and not l.strip().startswith((".",";"))]
    loads=[i for i,l in enumerate(bl) if l.startswith(("buffer_load","global_load"))]
    fulls=[i for i,l in enumerate(bl) if re.match(r"s_wait_loadcnt\s+0x0\b", l) or "s_wait_loadcnt_dscnt 0x0" in l]
    partial=[i for i,l in enumerate(bl) if l.startswith("s_wait_loadcnt") and not re.match(r"s_wait_loadcnt\s+0x0\b",l)]
    covers=[]
    for li in loads:
        nxt=[f for f in fulls if f>li]
        if nxt: covers.append(nxt[0]-li)
    print(f"  [{tag}] 热体 {name} 共 {n} 条指令")
    print(f"        load={len(loads)}  全排空 s_wait_loadcnt 0x0={len(fulls)}  部分 wait={len(partial)}")
    if covers:
        print(f"        cover 距离: 最小 {min(covers)}  中位 {sorted(covers)[len(covers)//2]}  最大 {max(covers)}")
    else:
        print(f"        热体内无全排空 -> cover 跨迭代（未被截断）")
for tag,p in (("cur",sys.argv[1]),("g1a",sys.argv[2])): analyse(p,tag)
