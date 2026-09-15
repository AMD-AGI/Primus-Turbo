import os,sys,json,statistics
JC="/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-attn-llama31-8b-bwd-20260914-015415/job_context"
sys.path.insert(0,os.path.join(JC,"op")); sys.path.insert(0,os.path.join(JC,"op","ut"))
import torch, benchmark
from shapes import ALL_SHAPES
OP=os.environ["OPDIR"]
shape=[s for s in ALL_SHAPES if s["name"]=="b4_s8192_hq32_hkv8_d128"][0]
mod=benchmark.load_impl(OP)
import primus_turbo.pytorch.kernels.attention.attention_fused_bwd_impl as FB
BASE=FB.get_fused_bwd_config()
q,k,v,do=benchmark.make_inputs(shape)
out=mod.attention(q,k,v,causal=True); torch.cuda.synchronize()
def bwd(): torch.autograd.grad(out,(q,k,v),do,retain_graph=True)
res=[]
for extra in json.loads(os.environ["SWEEP"]):
    def cfg(_e=extra):
        c={kk:dict(vv) for kk,vv in FB.get_fused_bwd_config.__wrapped__().items()} if False else None
        import primus_turbo.triton.attention.fused_mha_bwd_kernel as K
        d=K.get_fused_bwd_config(); d["onekernel"].update(_e); return d
    FB.get_fused_bwd_config=cfg
    label=json.dumps(extra,sort_keys=True)
    try:
        bwd(); torch.cuda.synchronize()
    except Exception as e:
        print(f"{label:<64} FAIL {type(e).__name__}: {str(e)[:160]}",flush=True); continue
    benchmark._warm(bwd,2.0); ms=benchmark._time_continuous(bwd,20)
    m=statistics.median(ms); res.append((m,label))
    print(f"{label:<64} bwd_ms={m:.4f} min={min(ms):.4f}",flush=True)
print("--- sorted ---")
for m,s in sorted(res): print(f"{m:.4f}  {s}")
