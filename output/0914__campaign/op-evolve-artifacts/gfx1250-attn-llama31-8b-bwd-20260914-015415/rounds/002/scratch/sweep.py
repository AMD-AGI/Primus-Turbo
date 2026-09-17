import os,sys,json,statistics
JC="/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-attn-llama31-8b-bwd-20260914-015415/job_context"
sys.path.insert(0,os.path.join(JC,"op")); sys.path.insert(0,os.path.join(JC,"op","ut"))
import torch, benchmark
from shapes import ALL_SHAPES
OP=os.environ["OPDIR"]
shape=[s for s in ALL_SHAPES if s["name"]=="b4_s8192_hq32_hkv8_d128"][0]
mod=benchmark.load_impl(OP)
import sys as _s
FB=[m for n,m in _s.modules.items() if n.endswith("attention_fused_bwd_impl")][0]
KM=[m for n,m in _s.modules.items() if n.endswith("fused_mha_bwd_kernel")][0]
KM._check_block_invariant=lambda cfg: None
q,k,v,do=benchmark.make_inputs(shape)
out=mod.attention(q,k,v,causal=True); torch.cuda.synchronize()
import torch as T
ref=None
def bwd(): torch.autograd.grad(out,(q,k,v),do,retain_graph=True)
res=[]
for tile,spec in json.loads(os.environ["SWEEP"]):
    FB.fused_backward_tile=lambda n,_t=tile:_t
    os.environ["PRIMUS_TURBO_FUSED_MHA_BWD_TUNE"]=spec if spec else "off"
    label=f"tile={tile} {spec or 'DEFAULT'}"
    try:
        g=torch.autograd.grad(out,(q,k,v),do,retain_graph=True); torch.cuda.synchronize()
    except Exception as e:
        print(f"{label:<60} FAIL {type(e).__name__}: {str(e)[:160]}",flush=True); continue
    if ref is None: ref=[x.double().clone() for x in g]
    sq=min(float(10*torch.log10((r**2).sum()/(((x.double()-r)**2).sum()+1e-30))) for x,r in zip(g,ref))
    benchmark._warm(bwd,2.0); ms=benchmark._time_continuous(bwd,20)
    m=statistics.median(ms); res.append((m,label))
    print(f"{label:<60} bwd_ms={m:.4f} min={min(ms):.4f} sqnr_vs_default={sq:.1f}dB",flush=True)
print("--- sorted ---")
for m,s in sorted(res): print(f"{m:.4f}  {s}")
