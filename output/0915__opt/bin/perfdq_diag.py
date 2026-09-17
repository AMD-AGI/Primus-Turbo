import os, sys, math
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")
R="/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo"
sys.path.insert(0, R); sys.path.insert(0, R+"/tools/gfx1250")
import torch, asm_bwd_launcher as L

b,s,hq,hk,d = 1,2048,32,8,128
torch.manual_seed(0)
q=torch.randn(b,s,hq,d,device="cuda",dtype=torch.bfloat16)
k=torch.randn(b,s,hk,d,device="cuda",dtype=torch.bfloat16)
v=torch.randn(b,s,hk,d,device="cuda",dtype=torch.bfloat16)
do=torch.randn(b,s,hq,d,device="cuda",dtype=torch.bfloat16)
from primus_turbo.pytorch.ops.attention.flash_attn_interface import triton_dense_forward
scale=1.0/math.sqrt(d)
o,lse=triton_dense_forward(q,k,v,softmax_scale=scale,causal=True); torch.cuda.synchronize()

def get(variant):
    dq,dk,dv=L.asm_backward(q,k,v,o,do,lse,scale,dkdv_heads="q",co_variant=variant)
    return dq.float()
a=get(""); p=get("_perf")
def sqnr(r,g):
    n=(r*r).sum(); e=((r-g)**2).sum()
    return float(10*torch.log10(n/e)) if e>0 else float("inf")
print("raw SQNR(_perf vs shipped): %.2f dB"%sqnr(a,p))
c=float((a*p).sum()/(p*p).sum())
print("best-fit scale _perf->shipped: %.6f   SQNR after scaling: %.2f dB"%(c, sqnr(a,c*p)))
# is only part of the tensor wrong?
err=((a-p)**2).sum(-1).flatten()
tot=(a*a).sum(-1).flatten()
bad=(err > 0.1*tot)
print("rows with >10%% error: %d / %d (%.1f%%)"%(int(bad.sum()), bad.numel(), 100*float(bad.float().mean())))
# where are they -- by sequence position
idx=bad.view(b,hq,s) if a.shape[1]==s else None
pos=bad.view(b,s,hq).any(-1) if True else None
first=int(pos[0].float().argmax()) if pos is not None and pos.any() else -1
print("first bad seq position:", first, " last:", int(s-1-pos[0].flip(0).float().argmax()) if pos.any() else -1)
frac=[float(pos[0][i*s//8:(i+1)*s//8].float().mean()) for i in range(8)]
print("bad fraction by seq octile:", " ".join("%.2f"%f for f in frac))
