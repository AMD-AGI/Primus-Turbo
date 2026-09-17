#!/bin/bash
# Written by tools/check_runner.py and left here so the check can be repeated by hand.
set -euo pipefail
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-1}
python3 - <<'PY'
import torch, time
assert torch.cuda.is_available(), "no GPU visible to torch"
n, iters = 4096, 20
a = torch.randn(n, n, dtype=torch.bfloat16, device="cuda")
b = torch.randn(n, n, dtype=torch.bfloat16, device="cuda")
for _ in range(3):
    torch.matmul(a, b)
torch.cuda.synchronize()
t = time.perf_counter()
for _ in range(iters):
    torch.matmul(a, b)
torch.cuda.synchronize()
secs = (time.perf_counter() - t) / iters
print(f"device  {torch.cuda.get_device_name(0)}")
print(f"arch    {torch.cuda.get_device_properties(0).gcnArchName}")
print(f"torch   {torch.__version__}")
print(f"gemm    {n}^3 bf16  {secs * 1e3:.3f} ms  {2 * n ** 3 / secs / 1e12:.1f} TFLOP/s")
PY
