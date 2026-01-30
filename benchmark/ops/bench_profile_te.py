import torch
import transformer_engine as te
from transformer_engine.common.recipe import Float8CurrentScaling, Format

M, N, K = 8192, 8192, 8192
dtype = torch.bfloat16
device = "cuda"

fp8_recipe = Float8CurrentScaling(fp8_format=Format.E4M3)
x = torch.randn((M, K), dtype=dtype, device=device, requires_grad=True)
layer = te.pytorch.Linear(K, N, bias=False, params_dtype=dtype).to(device)
grad_out = torch.randn((M, N), dtype=dtype, device=device)

# Warmup (not profiled)
for _ in range(3):
    with te.pytorch.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        out = layer(x)
    out.backward(grad_out, retain_graph=True)
torch.cuda.synchronize()

# Start profiling marker
torch.cuda.nvtx.range_push("profile_region")

# Profile - 10 iterations
for i in range(10):
    torch.cuda.nvtx.range_push(f"iter_{i}")
    with te.pytorch.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        out = layer(x)
    out.backward(grad_out, retain_graph=True)
    torch.cuda.nvtx.range_pop()

torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
