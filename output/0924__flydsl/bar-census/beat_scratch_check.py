"""Does aiter's pssk ASM fully write every q-head slice of its dk/dv scratch?
beat/impl.py:66-69 allocates with torch.empty and never zeroes. If pssk leaves any slice
unwritten, the bar has been reading the previous call's stale gradients.
Decisive test: poison the scratch with NaN, call once, look for surviving NaN."""
import os, sys, importlib.util, math
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")
sys.path.insert(0, "/home/lihuzhan/.local/flydsl032")
sys.path.insert(0, "/home/lihuzhan/code/aiter-src")
import torch
from aiter.ops.mha import fmha_fwd_with_sink_asm as ASM_FWD
B="/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-flydsl-attn-bwd-20260917-115934/job_context/op/beat/impl.py"
spec=importlib.util.spec_from_file_location("beat_impl",B); beat=importlib.util.module_from_spec(spec); spec.loader.exec_module(beat)

for name,(b,sq,skv,hq,hkv,d) in (("fast",(1,1024,1024,8,2,128)),("prod",(4,8192,8192,32,8,128))):
    dt=torch.bfloat16; scale=d**-0.5
    q=torch.randn(b,sq,hq,d,device="cuda",dtype=dt)
    k=torch.randn(b,skv,hkv,d,device="cuda",dtype=dt)
    v=torch.randn(b,skv,hkv,d,device="cuda",dtype=dt)
    o,lse=ASM_FWD(q,k,v,scale,True,True); do=torch.randn_like(o)
    # first call populates beat._scratch
    dq0,dk0,dv0 = beat.attn_bwd(do,q,k,v,o,lse,softmax_scale=scale,causal=True)
    torch.cuda.synchronize()
    key=list(beat._scratch.keys())[0]; s=beat._scratch[key]
    # POISON every scratch buffer with NaN
    for kk in ("dk","dv","dq_acc"): s[kk].fill_(float("nan"))
    torch.cuda.synchronize()
    dq1,dk1,dv1 = beat.attn_bwd(do,q,k,v,o,lse,softmax_scale=scale,causal=True)
    torch.cuda.synchronize()
    nan_dk=torch.isnan(dk1).sum().item(); nan_dv=torch.isnan(dv1).sum().item(); nan_dq=torch.isnan(dq1).sum().item()
    tot=dk1.numel()
    same_dk=torch.equal(dk0,dk1); same_dv=torch.equal(dv0,dv1); same_dq=torch.equal(dq0,dq1)
    print(f"  [{name}] NaN after poison:  dk {nan_dk}/{tot} ({100*nan_dk/tot:.4f}%)  dv {nan_dv}/{tot}  dq {nan_dq}/{dq1.numel()}")
    print(f"           bitwise identical to un-poisoned call:  dk {same_dk}  dv {same_dv}  dq {same_dq}")
    verdict = "CLEAN (pssk writes every slice)" if (nan_dk==0 and nan_dv==0 and nan_dq==0) else "*** STALE-READ CONFIRMED ***"
    print(f"           VERDICT: {verdict}")
    del q,k,v,o,lse,do,dq0,dk0,dv0,dq1,dk1,dv1; beat._scratch.clear(); torch.cuda.empty_cache()
