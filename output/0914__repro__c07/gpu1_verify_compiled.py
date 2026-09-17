import os, sys
spec = sys.argv[1] if len(sys.argv)>1 else ""
os.environ["TORCH_BLAS_PREFER_HIPBLASLT"]="0"
os.environ["HIP_VISIBLE_DEVICES"]=os.environ.get("GPU","1")
if spec: os.environ["PRIMUS_TURBO_ATTN_TRITON_TUNE"]=spec
sys.path.insert(0,"/home/lihuzhan/code/2026_0903__turbo/wt-bakeoff")
import torch, primus_turbo.triton.attention.attention_kernel as ak
from primus_turbo.pytorch.ops import flash_attn_func
b,s,hq,hkv,d = 1,1024,8,2,128
q=torch.randn(b,s,hq,d,device="cuda",dtype=torch.bfloat16,requires_grad=True)
k=torch.randn(b,s,hkv,d,device="cuda",dtype=torch.bfloat16,requires_grad=True)
v=torch.randn(b,s,hkv,d,device="cuda",dtype=torch.bfloat16,requires_grad=True)
o=flash_attn_func(q,k,v,causal=True)
o.backward(torch.randn_like(o)); torch.cuda.synchronize()
print("SPEC=",repr(spec))
def walk(obj, out, depth=0):
    if depth>3: return
    if isinstance(obj, dict):
        for vv in obj.values(): walk(vv,out,depth+1)
    elif isinstance(obj,(list,tuple)):
        for vv in obj: walk(vv,out,depth+1)
    else:
        md=getattr(obj,"metadata",None)
        if md is not None and hasattr(md,"num_warps"):
            out.append((md.num_warps, md.num_stages, getattr(md,"name","?")))
for name in ("attn_fwd","_bwd_kernel_dkdv","_bwd_kernel_dq"):
    fn=getattr(ak,name).fn
    out=[]
    walk(getattr(fn,"device_caches",None) or getattr(fn,"cache",{}), out)
    print(f"  {name}: compiled (num_warps,num_stages,name) = {sorted(set(out))}")
