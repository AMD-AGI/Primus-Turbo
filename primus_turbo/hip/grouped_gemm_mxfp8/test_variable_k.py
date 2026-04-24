###############################################################################
# Test the VariableK HIP backend drop-in via the variable-K entry point.
# Mirrors the Triton variable-K kernel's signature + API, so both compute
# the same dB output.
###############################################################################
from __future__ import annotations

import argparse
import math

import torch
import torch.utils.benchmark as tbench

from primus_turbo.hip.grouped_gemm_mxfp8 import grouped_gemm_mxfp8_hip_variable_k
from primus_turbo.triton.grouped_gemm.grouped_gemm_mxfp8_variable_k_kernel import (
    grouped_gemm_mxfp8_variable_k_triton_kernel,
)
from primus_turbo.triton.quantization.mxfp8_quant_kernels import quant_mxfp8_dual_jagged


def snr_db(ref, out):
    ref_f = ref.float(); out_f = out.float()
    n = (ref_f - out_f).pow(2).mean().item()
    s = ref_f.pow(2).mean().item()
    return 10.0 * math.log10(s / max(n, 1e-30)) if n else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=65536)
    ap.add_argument("--k", type=int, default=2880)
    ap.add_argument("--n", type=int, default=5760)
    ap.add_argument("--g", type=int, default=32)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    m, k, n, g = args.m, args.k, args.n, args.g
    m_per = m // g
    assert m % g == 0
    torch.manual_seed(0)
    device = "cuda"

    # Inputs
    a_bf  = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    go_bf = torch.randn(m, n, device=device, dtype=torch.bfloat16)
    group_offs = torch.arange(0, m + 1, m_per, dtype=torch.int64, device=device)
    group_lens = torch.full((g,), m_per, dtype=torch.int64, device=device)

    # Col-quant (same pre-quant as Triton wgrad path)
    _, _, a_col,  a_scale_col,  a_scale_offs  = quant_mxfp8_dual_jagged(a_bf,  group_offs, group_lens)
    _, _, go_col, go_scale_col, go_scale_offs = quant_mxfp8_dual_jagged(go_bf, group_offs, group_lens)

    # Triton variable-K: output [G, K, N] with lhs=a (cols become output rows=K),
    # rhs=go (cols become output cols=N).
    dB_tri_KN = grouped_gemm_mxfp8_variable_k_triton_kernel(
        a_col, go_col, a_scale_col, go_scale_col,
        group_offs, a_scale_offs, out_dtype=torch.bfloat16,
    )  # [G, K, N]

    # HIP variable-K: same API → output [G, K, N]
    dB_hip_KN = grouped_gemm_mxfp8_hip_variable_k(
        a_col, go_col, a_scale_col, go_scale_col,
        group_offs, a_scale_offs, out_dtype=torch.bfloat16, trans_c=False,
    )  # [G, K, N]

    # bf16 reference
    dB_bf_KN = torch.zeros(g, k, n, device=device, dtype=torch.bfloat16)
    for gi in range(g):
        s = int(group_offs[gi]); e = int(group_offs[gi + 1])
        dB_bf_KN[gi] = (a_bf[s:e].float().T @ go_bf[s:e].float()).to(torch.bfloat16)

    print("\n─── Correctness (output shape [G, K, N]) ───")
    print(f"  Triton vs bf16: {snr_db(dB_bf_KN, dB_tri_KN):.2f} dB")
    print(f"  HIP    vs bf16: {snr_db(dB_bf_KN, dB_hip_KN):.2f} dB")
    print(f"  HIP    vs Triton: {snr_db(dB_tri_KN, dB_hip_KN):.2f} dB")

    # Also test trans_c=True path (output [G, N, K])
    dB_hip_NK = grouped_gemm_mxfp8_hip_variable_k(
        a_col, go_col, a_scale_col, go_scale_col,
        group_offs, a_scale_offs, out_dtype=torch.bfloat16, trans_c=True,
    )  # [G, N, K]
    dB_bf_NK = dB_bf_KN.transpose(1, 2).contiguous()
    print(f"\n  (trans_c=True) HIP vs bf16 [G,N,K]: {snr_db(dB_bf_NK, dB_hip_NK):.2f} dB")

    print("\n─── Bench (kernel-only, excludes quant which is shared) ───")
    def run_hip():
        return grouped_gemm_mxfp8_hip_variable_k(
            a_col, go_col, a_scale_col, go_scale_col,
            group_offs, a_scale_offs, out_dtype=torch.bfloat16, trans_c=False,
        )

    def run_triton():
        return grouped_gemm_mxfp8_variable_k_triton_kernel(
            a_col, go_col, a_scale_col, go_scale_col,
            group_offs, a_scale_offs, out_dtype=torch.bfloat16,
        )

    for _ in range(3):
        run_hip(); run_triton()
    torch.cuda.synchronize()
    t_h = tbench.Timer(stmt="f()", globals={"f": run_hip}).timeit(args.iters).mean
    t_t = tbench.Timer(stmt="f()", globals={"f": run_triton}).timeit(args.iters).mean
    flops = 2.0 * m * k * n
    print(f"  Triton (variable-K native) : {t_t*1e3:7.3f} ms  {flops/t_t/1e12:7.1f} TFLOPS")
    print(f"  HIP    (fwd-reuse w/ pre-quant): {t_h*1e3:7.3f} ms  {flops/t_h/1e12:7.1f} TFLOPS  ({t_t/t_h:.3f}x)")


if __name__ == "__main__":
    main()
