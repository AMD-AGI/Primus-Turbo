import os,sys
JC="/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-attn-llama31-8b-bwd-20260914-015415/job_context"
sys.path.insert(0,os.path.join(JC,"op")); sys.path.insert(0,os.path.join(JC,"op","ut"))
import torch, benchmark
from shapes import ALL_SHAPES
from torch.profiler import profile, ProfilerActivity
OP=os.environ["OPDIR"]
shape=[s for s in ALL_SHAPES if s["name"]=="b4_s8192_hq32_hkv8_d128"][0]
mod=benchmark.load_impl(OP)
q,k,v,do=benchmark.make_inputs(shape)
out=mod.attention(q,k,v,causal=True); torch.cuda.synchronize()
def bwd(): torch.autograd.grad(out,(q,k,v),do,retain_graph=True)
benchmark._warm(bwd,2.0)
with profile(activities=[ProfilerActivity.CUDA]) as p:
    for _ in range(5): bwd()
    torch.cuda.synchronize()
print(p.key_averages().table(sort_by="self_device_time_total",row_limit=15))
