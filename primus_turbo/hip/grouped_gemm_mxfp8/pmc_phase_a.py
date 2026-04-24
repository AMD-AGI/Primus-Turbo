###############################################################################
# PMC harness for HIP MX-FP8 grouped GEMM Phase A.
# Runs the kernel N times (warm + measured) for rocprofv3 to sample counters.
# Writes nothing — the parent rocprofv3 invocation captures PMC output.
###############################################################################
from __future__ import annotations

import os
import sys
import torch

sys.path.insert(0, "/work/Primus-Turbo")

from primus_turbo.hip.grouped_gemm_mxfp8 import grouped_gemm_mxfp8_hip_fwd
from primus_turbo.triton.grouped_gemm.grouped_gemm_mxfp8_kernel import (
    grouped_gemm_mxfp8_triton_kernel,
)
from primus_turbo.triton.quantization.mxfp8_quant_kernels import quant_mxfp8_rowwise


def main():
    mode = os.environ.get("KERNEL", "hip")  # "hip" or "triton"
    m, k, n, g = 65536, 3072, 5760, 32
    assert m % g == 0
    m_per = m // g

    torch.manual_seed(0)
    device = "cuda"
    a_bf = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    b_bf = torch.randn(g, n, k, device=device, dtype=torch.bfloat16)
    a_fp8, a_scale = quant_mxfp8_rowwise(a_bf)
    b_fp8_flat, b_scale_flat = quant_mxfp8_rowwise(b_bf.reshape(g * n, k))
    b_fp8 = b_fp8_flat.view(g, n, k)
    b_scale = b_scale_flat.view(g, n, k // 32)
    group_offs = torch.arange(0, m + 1, m_per, dtype=torch.int64, device=device)

    # Warm up JIT build + kernel cache
    for _ in range(5):
        if mode == "hip":
            _ = grouped_gemm_mxfp8_hip_fwd(a_fp8, b_fp8, a_scale, b_scale, group_offs)
        else:
            _ = grouped_gemm_mxfp8_triton_kernel(
                a_fp8, b_fp8, a_scale, b_scale, group_offs,
                trans_b=True, out_dtype=torch.bfloat16
            )
    torch.cuda.synchronize()

    # PMC-measured iterations
    iters = int(os.environ.get("ITERS", "20"))
    for _ in range(iters):
        if mode == "hip":
            _ = grouped_gemm_mxfp8_hip_fwd(a_fp8, b_fp8, a_scale, b_scale, group_offs)
        else:
            _ = grouped_gemm_mxfp8_triton_kernel(
                a_fp8, b_fp8, a_scale, b_scale, group_offs,
                trans_b=True, out_dtype=torch.bfloat16
            )
    torch.cuda.synchronize()
    print(f"PMC run done: mode={mode}, iters={iters}")


if __name__ == "__main__":
    main()
