"""PMC harness: fwd-kernel-as-wgrad invocation. Measures HBM BW + stalls to
confirm memory-bound vs compute-bound classification for the wgrad-shape
fwd-kernel call."""
from __future__ import annotations
import os, sys
import torch

sys.path.insert(0, "/work/Primus-Turbo")

from primus_turbo.hip.grouped_gemm_mxfp8 import grouped_gemm_mxfp8_hip_fwd
from primus_turbo.triton.quantization.mxfp8_quant_kernels import quant_mxfp8_rowwise


def main():
    m, k, n, g = 65536, 2880, 5760, 32
    m_g = m // g
    device = "cuda"
    torch.manual_seed(0)

    # Reproduce the v1 wgrad fwd-kernel call shape:
    #   A_k shape [G*N, M_g],  B_k shape [G, K, M_g],  output [G*N, K]
    go_bf = torch.randn(m, n, device=device, dtype=torch.bfloat16)
    a_bf = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    go_T = go_bf.view(g, m_g, n).permute(0, 2, 1).contiguous()  # [G, N, M_g]
    a_T = a_bf.view(g, m_g, k).permute(0, 2, 1).contiguous()    # [G, K, M_g]

    go_flat_fp8, go_flat_scale = quant_mxfp8_rowwise(go_T.view(g * n, m_g))
    a_flat_fp8, a_flat_scale = quant_mxfp8_rowwise(a_T.view(g * k, m_g))
    a_fp8_3d = a_flat_fp8.view(g, k, m_g)
    a_scale_3d = a_flat_scale.view(g, k, m_g // 32)
    group_offs_stacked = torch.arange(0, g * n + 1, n, dtype=torch.int64, device=device)

    # Warm
    for _ in range(5):
        grouped_gemm_mxfp8_hip_fwd(
            go_flat_fp8, a_fp8_3d, go_flat_scale, a_scale_3d,
            group_offs_stacked, out_dtype=torch.bfloat16,
        )
    torch.cuda.synchronize()

    iters = int(os.environ.get("ITERS", "20"))
    for _ in range(iters):
        grouped_gemm_mxfp8_hip_fwd(
            go_flat_fp8, a_fp8_3d, go_flat_scale, a_scale_3d,
            group_offs_stacked, out_dtype=torch.bfloat16,
        )
    torch.cuda.synchronize()
    print(f"PMC run done: wgrad-shape fwd kernel, iters={iters}")


if __name__ == "__main__":
    main()
