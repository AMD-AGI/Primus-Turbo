"""Minimal prod-shape driver for rocprofv3 --pmc / --kernel-trace: asm and fly (r6 fwd + r20 bwd), real
training inputs of layer PM_SET (default L16 from the step-43 dump; 'randn' for N(0,1)), PM_N fwd and bwd
calls per arm, back-to-back, one sync per arm. No profiler inside; kernels are told apart by name.
"""
import os, sys, json
from pathlib import Path
E2E = Path("/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/e2e")
sys.path.insert(0, str(E2E / "attn_backends")); sys.path.insert(0, "/home/lihuzhan/.local/flydsl0341")
os.environ["E2E_ATTN"] = "asm"
BLAS = {k: os.environ.get(k, "") for k in ("TORCH_BLAS_PREFER_HIPBLASLT", "HIPBLASLT_TENSILE_LIBPATH")}
import torch  # noqa
from e2e_attn import arms  # noqa
os.environ.update(BLAS)
b, s, hq, hkv, d = 4, 8192, 32, 8, 128; sc = d ** -0.5; bf = torch.bfloat16
SET = os.environ.get("PM_SET", "L16"); N = int(os.environ.get("PM_N", "5"))
if SET == "randn":
    torch.manual_seed(0)
    q = torch.randn(b, s, hq, d, device="cuda", dtype=bf); k = torch.randn(b, s, hkv, d, device="cuda", dtype=bf); v = torch.randn_like(k)
else:
    call = {"L00": 672, "L01": 673, "L02": 674, "L08": 680, "L16": 688, "L31": 703}[SET]
    t = torch.load(f"/home/lihuzhan/_prof_dump/qkv_call{call:04d}.pt")
    q, k, v = (t[n].cuda() for n in ("q", "k", "v"))
do = torch.randn(b, s, hq, d, device="cuda", dtype=bf) * 1e-3
for a in os.environ.get("PM_ARMS", "asm,fly").split(","):
    fwd, bwd = arms.get_fwd(a), arms.get_bwd(a)
    for _ in range(N):
        o, lse = fwd(q, k, v, sc)
    torch.cuda.synchronize()
    for _ in range(N):
        bwd(do, q, k, v, o, lse, sc)
    torch.cuda.synchronize()
print("PMC_DONE", SET, N, flush=True)
