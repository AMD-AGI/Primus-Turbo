"""rocprofv3 timeline/PMC harness: FlyDSL attention FORWARD only (GPT-OSS shape).

Loops the forward N times so the per-kernel stats are dominated by the single
forward flash-attn kernel (flash_attn_generic_kernel).
"""
import torch

from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    attention_flydsl_forward_impl,
)

torch.manual_seed(0)
B, S, Hq, Hkv, D = 1, 8192, 64, 8, 64
dev, dt = "cuda", torch.bfloat16
scale = 1.0 / (D**0.5)

q = torch.randn(B, S, Hq, D, device=dev, dtype=dt)
k = torch.randn(B, S, Hkv, D, device=dev, dtype=dt)
v = torch.randn(B, S, Hkv, D, device=dev, dtype=dt)

for _ in range(5):
    attention_flydsl_forward_impl(q, k, v, scale, True)
torch.cuda.synchronize()

N = 50
for _ in range(N):
    attention_flydsl_forward_impl(q, k, v, scale, True)
torch.cuda.synchronize()
print(f"done {N} fwd iters, B={B} S={S} Hq={Hq} Hkv={Hkv} D={D}")
