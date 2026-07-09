import math
import torch
from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    attention_flydsl_forward_impl as fwd,
    attention_flydsl_backward_impl as bwd,
)

B, S, Hq, Hkv, D = 1, 8192, 64, 8, 64
scale = 1.0 / math.sqrt(D)
torch.manual_seed(0)
q = torch.randn(B, S, Hq, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.bfloat16)
do = torch.randn(B, S, Hq, D, device="cuda", dtype=torch.bfloat16)

out, lse = fwd(q, k, v, scale, True)
for _ in range(2):
    bwd(do, q, k, v, out, lse, scale, True)
torch.cuda.synchronize()
for _ in range(6):
    bwd(do, q, k, v, out, lse, scale, True)
torch.cuda.synchronize()
