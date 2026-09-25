"""Calibrate the dq run-to-run tolerance from a SHIPPING atomic implementation.
aiter's backward accumulates dQ with 514 buffer_atomic_add_f32, so its run-to-run dq spread
is an empirical bound on what fp32 atomic reordering costs on this card."""
import os, sys, importlib.util, math
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")
sys.path.insert(0, "/home/lihuzhan/.local/flydsl032"); sys.path.insert(0, "/home/lihuzhan/code/aiter-src")
import torch
from aiter.ops.mha import fmha_fwd_with_sink_asm as ASM_FWD
B="/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-flydsl-attn-bwd-20260917-115934/job_context/op/beat/impl.py"
sp=importlib.util.spec_from_file_location("beat_impl",B); beat=importlib.util.module_from_spec(sp); sp.loader.exec_module(beat)

def sqnr_db(ref, got):
    ref=ref.float(); got=got.float()
    num=(ref**2).sum(); den=((ref-got)**2).sum()
    return float('inf') if den==0 else 10*math.log10(float(num/den))

for name,(b,sq,skv,hq,hkv,d) in (("fast",(1,1024,1024,8,2,128)),("prod",(4,8192,8192,32,8,128))):
    dt=torch.bfloat16; scale=d**-0.5
    q=torch.randn(b,sq,hq,d,device="cuda",dtype=dt); k=torch.randn(b,skv,hkv,d,device="cuda",dtype=dt)
    v=torch.randn(b,skv,hkv,d,device="cuda",dtype=dt)
    o,lse=ASM_FWD(q,k,v,scale,True,True); do=torch.randn_like(o)
    runs=[]
    for i in range(6):
        dq,dk,dv = beat.attn_bwd(do,q,k,v,o,lse,softmax_scale=scale,causal=True)
        torch.cuda.synchronize(); runs.append((dq.clone(),dk.clone(),dv.clone()))
    r0=runs[0]
    for tag,idx in (("dq",0),("dk",1),("dv",2)):
        bits=all(torch.equal(r0[idx],r[idx]) for r in runs[1:])
        s=[sqnr_db(r0[idx],r[idx]) for r in runs[1:]]
        rel=[float((r0[idx].float()-r[idx].float()).abs().max()/r0[idx].float().abs().max()) for r in runs[1:]]
        lo = "inf" if all(x==float('inf') for x in s) else f"{min(s):.1f}"
        print(f"  [{name}] {tag}: 逐位相同={bits}  跑间 SQNR 最低={lo} dB  最大相对偏差={max(rel):.3e}")
    del q,k,v,o,lse,do,runs,r0; beat._scratch.clear(); torch.cuda.empty_cache()
