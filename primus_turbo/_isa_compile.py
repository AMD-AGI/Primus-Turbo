"""Minimal compile-only driver to trigger dkdv/fused/odo ISA emission (fast)."""
import torch

from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    attention_flydsl_backward_impl,
    attention_flydsl_forward_impl,
)

torch.manual_seed(0)
B, S, Hq, Hkv, D = 1, 512, 64, 8, 64
dev, dt = "cuda", torch.bfloat16
sc = 1.0 / (D**0.5)
q = torch.randn(B, S, Hq, D, device=dev, dtype=dt)
k = torch.randn(B, S, Hkv, D, device=dev, dtype=dt)
v = torch.randn(B, S, Hkv, D, device=dev, dtype=dt)
do = torch.randn(B, S, Hq, D, device=dev, dtype=dt)
o, l = attention_flydsl_forward_impl(q, k, v, sc, True)
attention_flydsl_backward_impl(do, q, k, v, o, l, sc, True)
torch.cuda.synchronize()
print("compiled")
