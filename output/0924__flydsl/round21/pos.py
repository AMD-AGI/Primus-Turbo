import re,sys
p,lab,nxt=sys.argv[1],sys.argv[2],sys.argv[3]
src=open(p).read().splitlines()
st=None;en=len(src)
for i,l in enumerate(src):
    if re.match(r'^'+re.escape(lab)+r':',l): st=i
    elif st is not None and re.match(r'^'+re.escape(nxt)+r':',l): en=i;break
b=src[st:en]
ev=[]
for i,l in enumerate(b):
    s=l.strip()
    if re.match(r'^(s_wait_|buffer_load|s_barrier|v_mov_b64|global_load)',s): ev.append((i,s[:55]))
n=len(b)
print(f"{lab}: {n} lines")
# summarize buffer_load positions
bl=[i for i,l in enumerate(b) if 'buffer_load' in l]
mv=[i for i,l in enumerate(b) if 'v_mov_b64' in l]
print("buffer_load pos: first",bl[0] if bl else None,"last",bl[-1] if bl else None,"count",len(bl))
print("v_mov_b64 pos: first",mv[0] if mv else None,"last",mv[-1] if mv else None,"count",len(mv))
for i,s in ev:
    if s.startswith('s_wait') or s.startswith('s_barrier'): print(f"  {i:5d} ({100*i//n:3d}%) {s}")
