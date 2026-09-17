import sys,json
from collections import Counter
d=json.JSONDecoder()
rows=[]
for l in sys.stdin:
    l=l.strip()
    if not l: continue
    try: rows.append(d.raw_decode(l)[0])
    except Exception as e: print('SKIP',e,l[:80],file=sys.stderr)
hdr=('shape','impl','n','fwd_ms','bwd_ms','bwd_TF','bwd_GBs','spr%')
print('%-16s %-9s %-2s %8s %9s %8s %8s %7s'%hdr)
c=Counter()
for r in rows:
    k=(r['shape'],r['impl']); c[k]+=1
    print('%-16s %-9s %-2d %8.4f %9.4f %8.2f %8.1f %7.2f'%(
      r['shape'],r['impl'],c[k],r['fwd_ms'],r['bwd_ms'],r['bwd_tflops'],r['bwd_gbps'],r['bwd_spread_pct']))
