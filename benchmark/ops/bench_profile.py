import torch

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
)

M, N, K = 8192, 8192, 8192
dtype = torch.bfloat16
device = "cuda"

config = Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.TENSORWISE)
a = torch.randn((M, K), dtype=dtype, device=device, requires_grad=True)
b = torch.randn((N, K), dtype=dtype, device=device, requires_grad=True)
grad_out = torch.randn((M, N), dtype=dtype, device=device)

# Warmup (not profiled)
for _ in range(3):
    out = turbo.ops.gemm_fp8(a, b, trans_b=True, config=config)
    out.backward(grad_out, retain_graph=True)
torch.cuda.synchronize()


# Profile - 10 iterations
for i in range(10):

    out = turbo.ops.gemm_fp8(a, b, trans_b=True, config=config)
    out.backward(grad_out, retain_graph=True)

    torch.cuda.synchronize()
