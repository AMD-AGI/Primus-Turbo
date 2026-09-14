import os, json, sys
if os.environ.get("GPU"): os.environ["HIP_VISIBLE_DEVICES"] = os.environ["GPU"]
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")
import torch, math
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
B,S,HQ,HKV,D = 4,8192,32,8,128
NW = int(sys.argv[1]) if len(sys.argv)>1 else 4
torch.manual_seed(0)
dev,dt="cuda",torch.bfloat16
q=torch.randn(B,HQ,S,D,device=dev,dtype=dt,requires_grad=True)
k=torch.randn(B,HKV,S,D,device=dev,dtype=dt,requires_grad=True)
v=torch.randn(B,HKV,S,D,device=dev,dtype=dt,requires_grad=True)
do=torch.randn(B,HQ,S,D,device=dev,dtype=dt)
bm=create_block_mask(lambda b,h,i,j: i>=j, B=None,H=None,Q_LEN=S,KV_LEN=S,device=dev)
fa=torch.compile(flex_attention,dynamic=False)
out=fa(q,k,v,block_mask=bm,enable_gqa=True,kernel_options={"num_warps":NW})
out.backward(do)
got={"out":out.detach().float(),"dq":q.grad.float(),"dk":k.grad.float(),"dv":v.grad.float()}
# fp32 reference, per (b, hq)
ref={n:torch.zeros_like(t) for n,t in got.items()}
qf,kf,vf,dof=q.detach().float(),k.detach().float(),v.detach().float(),do.float()
scale=1/math.sqrt(D)
mask=torch.full((S,S),float("-inf"),device=dev).triu(1)
for b in range(B):
    for h in range(HQ):
        g=h//(HQ//HKV)
        qq=qf[b,h].clone().requires_grad_(); kk=kf[b,g].clone().requires_grad_(); vv=vf[b,g].clone().requires_grad_()
        s_=(qq@kk.T)*scale+mask
        p=torch.softmax(s_,dim=-1)
        o=p@vv
        o.backward(dof[b,h])
        ref["out"][b,h]=o.detach(); ref["dq"][b,h]=qq.grad
        ref["dk"][b,g]+=kk.grad; ref["dv"][b,g]+=vv.grad
        del s_,p,o
res={}
for n in got:
    num=(ref[n]**2).sum().item(); den=((ref[n]-got[n])**2).sum().item()
    res[n]=round(10*math.log10(num/max(den,1e-30)),2)
print(json.dumps({"num_warps":NW,"sqnr_db":res,"correct":all(x>=50 for x in res.values())}))
