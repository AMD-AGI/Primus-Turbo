import os,sys,json,statistics,importlib,glob,re
JC="/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-attn-llama31-8b-bwd-20260914-015415/job_context"
sys.path.insert(0,os.path.join(JC,"op")); sys.path.insert(0,os.path.join(JC,"op","ut"))
import torch, benchmark
from shapes import ALL_SHAPES
OP=os.environ["OPDIR"]
shape=[s for s in ALL_SHAPES if s["name"]=="b4_s8192_hq32_hkv8_d128"][0]
mod=benchmark.load_impl(OP)
q,k,v,do=benchmark.make_inputs(shape)
out=mod.attention(q,k,v,causal=True)
torch.cuda.synchronize()
def bwd(): torch.autograd.grad(out,(q,k,v),do,retain_graph=True)

CONFIGS=json.loads(os.environ["SWEEP"])
res=[]
for spec in CONFIGS:
    os.environ["PRIMUS_TURBO_FUSED_MHA_BWD_TUNE"]=spec if spec else "off"
    try:
        bwd(); torch.cuda.synchronize()
    except Exception as e:
        print(f"{spec or 'DEFAULT':<60} BUILD-FAIL {type(e).__name__}: {str(e)[:120]}"); continue
    benchmark._warm(bwd,2.0)
    ms=benchmark._time_continuous(bwd,20)
    m=statistics.median(ms)
    ys=sorted(ms); sp=(ys[int(0.9*len(ys))]-ys[max(0,int(0.1*len(ys))-1)])/m*100
    res.append((m,spec))
    print(f"{spec or 'DEFAULT':<60} bwd_ms={m:.4f} spread={sp:.2f}% min={min(ms):.4f}",flush=True)
print("--- sorted ---")
for m,s in sorted(res): print(f"{m:.4f}  {s or 'DEFAULT'}")
