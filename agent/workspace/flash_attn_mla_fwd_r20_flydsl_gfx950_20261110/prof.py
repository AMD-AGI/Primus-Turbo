"""Run one MLA forward kernel N times for profilers (rocprofv3 kernel trace / PMC / ATT).

Usage: python prof.py DIR|aiter [shape] [iters]
"""

import sys

import torch

import ab_bench as ab

kernel = sys.argv[1]
label, B, S, H = next(s for s in ab.FULL if s[0] == (sys.argv[2] if len(sys.argv) > 2 else "dsv3"))
iters = int(sys.argv[3]) if len(sys.argv) > 3 else 5
g = torch.Generator(device="cuda").manual_seed(0)
q, k, v = (torch.randn(S, B, H, d, generator=g, device="cuda", dtype=torch.bfloat16) for d in (ab.DQK, ab.DQK, ab.DV))
if kernel == "aiter":
    ab.GlobalBackendManager.set_attn_backend(ab.BackendType.AITER, ab.PrecisionType.BF16_FP16_FP32)
    leaves = [t.permute(1, 0, 2, 3).detach().requires_grad_() for t in (q, k, v)]
    fn = lambda: ab.pt.ops.flash_attn_func(*leaves, softmax_scale=ab.DQK**-0.5, causal=True)  # noqa: E731
else:
    m = ab.load_pkg(kernel, "mla_prof")
    fn = lambda: m.mla_fwd_sbhd(q, k, v, causal=True, return_lse=True)  # noqa: E731
for _ in range(iters):
    fn()
torch.cuda.synchronize()
print("done", kernel, label, iters)
