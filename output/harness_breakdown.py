import math, time
import torch
from primus_turbo.flydsl.attention.kernels.csa_pool_sparse_fwd_kernel import build_csa_pool_sparse_fwd_module
from primus_turbo.flydsl.attention.kernels.sla_fwd_kernel import build_swa_fwd_module

def bench(fn, it=30):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / it * 1e3

B,H,S,D,K,P,W = 1,64,2048,512,512,512,128
dev="cuda"; scale=D**-0.5
q = torch.randn(B,H,S,D,device=dev,dtype=torch.bfloat16)
k_local = torch.randn(B,1,S,D,device=dev,dtype=torch.bfloat16)
v_local = torch.randn(B,1,S,D,device=dev,dtype=torch.bfloat16)
pool = torch.randn(B,P,D,device=dev,dtype=torch.bfloat16)
topk = torch.randint(0,P,(B,S,K),device=dev,dtype=torch.int32)

swa = build_swa_fwd_module(num_heads=H,head_dim=D,swa_window=W,dtype_str="bf16",layout_bhld=True,mqa_kv=True,block_m=128,block_n=32)
sp = build_csa_pool_sparse_fwd_module(num_heads=H,head_dim=D,dtype_str="bf16")
o_local=torch.empty_like(q); lse_local=torch.zeros(B,H,S,device=dev,dtype=torch.float32)
o_sparse=torch.empty_like(q); lse_sparse=torch.zeros(B,H,S,device=dev,dtype=torch.float32)
qf=q.contiguous().view(-1); klf=k_local.contiguous().view(-1); vlf=v_local.contiguous().view(-1)
poolf=pool.contiguous().view(-1); tkf=topk.contiguous().view(-1)

t_swa = bench(lambda: swa(qf,klf,vlf,o_local.view(-1),lse_local.view(-1),B,S))
t_sp = bench(lambda: sp(qf,poolf,tkf,o_sparse.view(-1),lse_sparse.view(-1),B,S,int(K),int(P)))

def merge():
    lse_stack = torch.stack([lse_local, lse_sparse],0)
    lse = torch.logsumexp(lse_stack,dim=0)
    wl = torch.exp(lse_local-lse).unsqueeze(-1); ws = torch.exp(lse_sparse-lse).unsqueeze(-1)
    return (o_local.float()*wl + o_sparse.float()*ws).to(q.dtype)
t_merge = bench(merge)

def merge_bf16():
    m = torch.maximum(lse_local, lse_sparse)
    wl = torch.exp(lse_local - m); ws = torch.exp(lse_sparse - m)
    denom = wl + ws
    wl = (wl / denom).unsqueeze(-1).to(q.dtype); ws = (ws / denom).unsqueeze(-1).to(q.dtype)
    return o_local * wl + o_sparse * ws
t_merge2 = bench(merge_bf16)
print(f"merge_bf16: {t_merge2:.3f} ms")

try:
    cmerge = torch.compile(merge_bf16)
    t_merge3 = bench(cmerge)
    print(f"merge_compiled: {t_merge3:.3f} ms")
except Exception as e:
    print("compile failed", e)
print(f"local SWA: {t_swa:.3f} ms")
print(f"sparse   : {t_sp:.3f} ms")
print(f"merge    : {t_merge:.3f} ms")
print(f"sum      : {t_swa+t_sp+t_merge:.3f} ms  (Triton 0.70)")
