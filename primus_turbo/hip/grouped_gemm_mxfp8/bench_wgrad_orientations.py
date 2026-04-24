"""Bench HIP wgrad with both trans_c orientations to find the better tile-fit.

trans_c=False: M_stack=G*K=92160,  N_kern=N=5760, K_red=M_g
trans_c=True : M_stack=G*N=184320, N_kern=K=2880, K_red=M_g
"""
from __future__ import annotations
import math, torch
import torch.utils.benchmark as tbench

from primus_turbo.hip.grouped_gemm_mxfp8 import grouped_gemm_mxfp8_hip_variable_k
from primus_turbo.triton.grouped_gemm.grouped_gemm_mxfp8_variable_k_kernel import (
    grouped_gemm_mxfp8_variable_k_triton_kernel,
)
from primus_turbo.triton.quantization.mxfp8_quant_kernels import quant_mxfp8_dual_jagged


def main():
    m, k, n, g = 65536, 2880, 5760, 32
    m_per = m // g
    device = "cuda"
    torch.manual_seed(0)
    a_bf  = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    go_bf = torch.randn(m, n, device=device, dtype=torch.bfloat16)
    group_offs = torch.arange(0, m + 1, m_per, dtype=torch.int64, device=device)
    group_lens = torch.full((g,), m_per, dtype=torch.int64, device=device)
    _, _, a_col,  a_scale_col,  a_sc_offs  = quant_mxfp8_dual_jagged(a_bf,  group_offs, group_lens)
    _, _, go_col, go_scale_col, _          = quant_mxfp8_dual_jagged(go_bf, group_offs, group_lens)

    flops = 2.0 * m * k * n

    def hip_F():  # trans_c=False: lhs=a(K), rhs=go(N) -> [G, K, N]
        return grouped_gemm_mxfp8_hip_variable_k(
            a_col, go_col, a_scale_col, go_scale_col,
            group_offs, a_sc_offs, out_dtype=torch.bfloat16, trans_c=False)

    def hip_T():  # trans_c=True : lhs=a(K), rhs=go(N), output [G, N, K] (swap inside)
        return grouped_gemm_mxfp8_hip_variable_k(
            a_col, go_col, a_scale_col, go_scale_col,
            group_offs, a_sc_offs, out_dtype=torch.bfloat16, trans_c=True)

    def hip_F_swap():  # trans_c=False with swapped lhs/rhs: lhs=go(N), rhs=a(K) -> [G, N, K]
        return grouped_gemm_mxfp8_hip_variable_k(
            go_col, a_col, go_scale_col, a_scale_col,
            group_offs, a_sc_offs, out_dtype=torch.bfloat16, trans_c=False)

    def tri_F():
        return grouped_gemm_mxfp8_variable_k_triton_kernel(
            a_col, go_col, a_scale_col, go_scale_col,
            group_offs, a_sc_offs, out_dtype=torch.bfloat16)

    for _ in range(3):
        hip_F(); hip_T(); hip_F_swap(); tri_F()
    torch.cuda.synchronize()

    print(f"{'config':40s} {'wall_ms':>9s} {'TFLOPS':>9s} {'vs_Triton':>10s}")
    print("-" * 75)
    t_tri = tbench.Timer(stmt="f()", globals={"f": tri_F}).timeit(50).mean * 1e3
    print(f"{'Triton variable-K (baseline)':40s} {t_tri:9.3f} {flops/t_tri/1e9:9.1f} {1.0:10.3f}")
    for label, fn in [
        ("HIP trans_c=False (M_stack=G*K=92160)",        hip_F),
        ("HIP trans_c=True  (M_stack=G*N=184320)",       hip_T),
        ("HIP trans_c=False swap-lhs/rhs (M=G*N)",       hip_F_swap),
    ]:
        t = tbench.Timer(stmt="f()", globals={"f": fn}).timeit(50).mean * 1e3
        print(f"{label:40s} {t:9.3f} {flops/t/1e9:9.1f} {t_tri/t:10.3f}")


if __name__ == "__main__":
    main()
