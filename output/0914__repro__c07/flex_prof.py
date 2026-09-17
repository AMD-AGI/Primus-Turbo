import os, sys
if os.environ.get("GPU"): os.environ["HIP_VISIBLE_DEVICES"] = os.environ["GPU"]
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
B,S,HQ,HKV,D = 4,8192,32,8,128
torch.manual_seed(0)
dev,dt="cuda",torch.bfloat16
q=torch.randn(B,HQ,S,D,device=dev,dtype=dt,requires_grad=True)
k=torch.randn(B,HKV,S,D,device=dev,dtype=dt,requires_grad=True)
v=torch.randn(B,HKV,S,D,device=dev,dtype=dt,requires_grad=True)
do=torch.randn(B,HQ,S,D,device=dev,dtype=dt)
bm=create_block_mask(lambda b,h,i,j: i>=j, B=None,H=None,Q_LEN=S,KV_LEN=S,device=dev)
fa=torch.compile(flex_attention,dynamic=False)
out=fa(q,k,v,block_mask=bm,enable_gqa=True)
for _ in range(3):
    q.grad=k.grad=v.grad=None
    out.backward(do,retain_graph=True)
torch.cuda.synchronize()
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as pr:
    for _ in range(3):
        q.grad=k.grad=v.grad=None
        out.backward(do,retain_graph=True)
    torch.cuda.synchronize()
ev=[e for e in pr.key_averages() if e.device_time_total>0]
ev.sort(key=lambda e:-e.device_time_total)
tot=sum(e.device_time_total for e in ev if e.key.startswith(("triton","void","Cijk","at::","elementwise","_"))) 
print(f"{'kernel':70s} {'calls':>6s} {'ms/iter':>9s}")
s=0
for e in ev[:15]:
    ms=e.device_time_total/1e3/3
    s+=ms
    print(f"{e.key[:70]:70s} {e.count:6d} {ms:9.3f}")
print(f"SUM(top15) {s:.3f} ms/iter")
