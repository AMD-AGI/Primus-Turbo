#!/usr/bin/env python3
"""Per-kernel GPU-time split of the three-kernel backward, by CUDA events.

rocprofv3 on gfx1250 exits 0 and writes no CSV here (the documented trap), so the split
is taken with events around each launch inside a copy of impl.attn_bwd's body.
Synthetic o/lse -- timing only, no rocBLAS anywhere.
"""
import sys, math, torch
J = "/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-flydsl-attn-bwd-20260917-115934/job_context/op"
sys.path.insert(0, J + "/ut")
from common import SHAPES
import importlib.util as ilu, pathlib as pl

path = pl.Path(sys.argv[1]); shapes = sys.argv[2].split(","); iters = int(sys.argv[3])
def sib(stem):
    n = f"{stem}__{abs(hash(str(path)))}"
    sp = ilu.spec_from_file_location(n, path / f"{stem}.py")
    m = ilu.module_from_spec(sp); sys.modules[n] = m; sp.loader.exec_module(m); return m
sib("_env"); _k = sib("kernels")

for shape in shapes:
    b, sq, skv, hq, hkv, d = SHAPES[shape]
    g = torch.Generator(device="cuda").manual_seed(0)
    r = lambda *s: torch.randn(*s, generator=g, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    q, k, v, do = r(b,sq,hq,d), r(b,skv,hkv,d), r(b,skv,hkv,d), r(b,sq,hq,d)
    o = r(b,sq,hq,d)
    lse = (torch.randn(b,hq,sq, generator=g, device="cuda", dtype=torch.float32).abs()+8.0)
    n_rows = b*sq*hq; gg = hq//hkv; sc = 1.0/math.sqrt(d); st = torch.cuda.current_stream()
    delta = torch.empty((b,hq,sq), device="cuda", dtype=torch.float32)
    dk32 = torch.empty((b,skv,hkv,d), device="cuda", dtype=torch.float32); dv32 = torch.empty_like(dk32)
    dq32 = torch.empty((b,sq,hq,d), device="cuda", dtype=torch.float32)
    def f_delta(): _k.launch_delta(do,o,delta,sq,hq,n_rows,n_rows//_k.ROWS_DELTA,st)
    def f_dkdv():  _k.launch_dkdv(q,k,v,do,lse,delta,dv32,dk32,sc,sq,skv,hq,hkv,gg,sq//16,skv-sq,1,skv//16,hkv,b,st)
    def f_dq():    _k.launch_dq(q,k,v,do,lse,delta,dq32,sc,sq,skv,hq,hkv,gg,skv//_k.KV_STEP,skv-sq,1,sq//16,hq,b,st)
    stages = [("delta",f_delta),("dkdv",f_dkdv),("dq",f_dq)]
    for _ in range(3):
        for _n,f in stages: f()
    torch.cuda.synchronize()
    out = {}
    for name, f in stages:
        ts = []
        e0,e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        for _ in range(iters):
            e0.record(); f(); e1.record(); e1.synchronize(); ts.append(e0.elapsed_time(e1))
        ts.sort(); out[name] = ts[len(ts)//2]
    tot = sum(out.values())
    print(f"SPLIT shape={shape} " + " ".join(f"{n}={v:.4f}ms({100*v/tot:.1f}%)" for n,v in out.items())
          + f" total={tot:.4f}ms", flush=True)
