"""Minimal backward harness for rocprof-compute occupancy/tail analysis of dkdv.

Forward once, a couple warmup bwd, then a few timed bwd. Kept tiny so the
multi-pass rocprof-compute profile stays fast (app import dominates per pass).
"""
import torch

from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    attention_flydsl_backward_impl,
    attention_flydsl_forward_impl,
)

torch.manual_seed(0)
B, S, Hq, Hkv, D = 1, 8192, 64, 8, 64
dev, dt = "cuda", torch.bfloat16
scale = 1.0 / (D**0.5)

q = torch.randn(B, S, Hq, D, device=dev, dtype=dt)
k = torch.randn(B, S, Hkv, D, device=dev, dtype=dt)
v = torch.randn(B, S, Hkv, D, device=dev, dtype=dt)
dout = torch.randn(B, S, Hq, D, device=dev, dtype=dt)

out, lse = attention_flydsl_forward_impl(q, k, v, scale, True)

for _ in range(2):
    attention_flydsl_backward_impl(dout, q, k, v, out, lse, scale, True)
torch.cuda.synchronize()

N = 4
for _ in range(N):
    attention_flydsl_backward_impl(dout, q, k, v, out, lse, scale, True)
torch.cuda.synchronize()
print(f"done {N} bwd iters")
