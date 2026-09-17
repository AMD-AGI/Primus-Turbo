import sys,statistics as st,collections
f=sys.argv[1]
rows=[]
for ln in open(f):
    p=ln.split()
    if len(p)<6: continue
    rows.append((float(p[0]),int(p[1])/1e6,int(p[3]),int(p[5])/1000))
if not rows: print("no samples"); sys.exit()
t0=rows[0][0]
# find the longest contiguous stretch of busy>=85 -- the timed phase is the tail
busy=[r for r in rows if r[2]>=85]
# split by clock plateau: take last 60% of busy samples (timing phase follows correctness phase)
tail=busy[int(len(busy)*0.45):]
def s(n,v):
    v=sorted(v); k=len(v)
    print(f"  {n}: n={k} min={v[0]:.0f} p5={v[int(.05*k)]:.0f} med={st.median(v):.0f} mean={st.mean(v):.1f} p95={v[int(.95*k)]:.0f} max={v[-1]:.0f}")
print(f"== {f}  total={len(rows)} busy={len(busy)}")
s("sclk ALL busy",[r[1] for r in busy])
s("sclk timing-phase tail",[r[1] for r in tail])
s("temp tail",[r[3] for r in tail])
b=collections.defaultdict(list)
for r in rows: b[int((r[0]-t0)//5)].append(r)
print("  trace(5s): "+" ".join(f"{k*5}s:{st.mean(x[1] for x in v):.0f}/{st.mean(x[2] for x in v):.0f}%" for k,v in sorted(b.items())))
