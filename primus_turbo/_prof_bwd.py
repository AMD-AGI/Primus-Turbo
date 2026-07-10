"""rocprofv3 timeline harness: FlyDSL attention BACKWARD only (GPT-OSS shape).

Runs the forward once to produce (out, lse), then loops the backward N times so the
per-kernel stats are dominated by the backward kernels + its torch ops (delta kernel,
dq kernel, dkdv kernel, the split-K ws.sum reduction, the LSE prescale muls).
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

for _ in range(5):
    attention_flydsl_backward_impl(dout, q, k, v, out, lse, scale, True)
torch.cuda.synchronize()

N = 50
for _ in range(N):
    attention_flydsl_backward_impl(dout, q, k, v, out, lse, scale, True)
torch.cuda.synchronize()
print(f"done {N} bwd iters, B={B} S={S} Hq={Hq} Hkv={Hkv} D={D}")
