###############################################################################
# Phase C correctness + bench for HIP MX-FP8 grouped GEMM wgrad (v0 path).
#
# dB[g, n, k] = sum_{m in [offs[g], offs[g+1])} grad_out[m, n] * a[m, k]
# v0 uses per-expert fwd-kernel calls with global transpose+quant.
###############################################################################

from __future__ import annotations

import argparse
import math

import torch
import torch.utils.benchmark as tbench

from primus_turbo.hip.grouped_gemm_mxfp8 import grouped_gemm_mxfp8_hip_wgrad
from primus_turbo.triton.grouped_gemm.grouped_gemm_mxfp8_variable_k_kernel import (
    grouped_gemm_mxfp8_variable_k_triton_kernel,
)
from primus_turbo.triton.quantization.mxfp8_quant_kernels import (
    quant_mxfp8_dual_jagged,
)


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
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    if args.small:
        m, k, n, g = 2048, 512, 512, 4   # M_g=512 >= 384, mult of 128
    else:
        m, k, n, g = args.m, args.k, args.n, args.g

    torch.manual_seed(0)
    device = "cuda"
    assert m % g == 0
    m_per = m // g
    assert m_per >= 384 and m_per % 128 == 0, (
        f"wgrad v0 needs M_g >= 384 and M_g % 128 == 0; got M_g={m_per}"
    )
    print(f"Shape: grad_out=[M={m}, N={n}], a=[M={m}, K={k}]  =>  dB=[G={g}, N={n}, K={k}]")
    print(f"       balanced M_g={m_per}")

    grad_out_bf = torch.randn(m, n, device=device, dtype=torch.bfloat16)
    a_bf        = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    group_offs  = torch.arange(0, m + 1, m_per, dtype=torch.int64, device=device)
    group_lens  = torch.full((g,), m_per, dtype=torch.int64, device=device)

    # ── Reference: bf16 per-expert wgrad
    dB_bf = torch.zeros(g, n, k, device=device, dtype=torch.bfloat16)
    for gi in range(g):
        s = int(group_offs[gi]); e = int(group_offs[gi + 1])
        dB_bf[gi] = (grad_out_bf[s:e].float().T @ a_bf[s:e].float()).to(torch.bfloat16)

    # ── HIP wgrad v0
    dB_hip = grouped_gemm_mxfp8_hip_wgrad(grad_out_bf, a_bf, group_offs, out_dtype=torch.bfloat16)

    # ── Triton variable-K wgrad reference
    (go_row, go_scale_row, go_col, go_scale_col, go_scale_offs) = quant_mxfp8_dual_jagged(
        grad_out_bf, group_offs, group_lens
    )
    (a_row, a_scale_row, a_col, a_scale_col, a_scale_offs) = quant_mxfp8_dual_jagged(
        a_bf, group_offs, group_lens
    )
    # Triton wgrad output layout with lhs/rhs swap: lhs=grad_out_col (rows=N),
    # rhs=a_col (cols=K) → output [G, N, K] (matches HIP convention).
    dB_tri = grouped_gemm_mxfp8_variable_k_triton_kernel(
        go_col, a_col, go_scale_col, a_scale_col,
        group_offs, go_scale_offs, out_dtype=torch.bfloat16,
    )  # [G, N, K]

    print("\n─── Correctness ───")
    print(f"  HIP    wgrad vs bf16 ref   SNR: {snr_db(dB_bf, dB_hip):.2f} dB")
    print(f"  Triton wgrad vs bf16 ref   SNR: {snr_db(dB_bf, dB_tri):.2f} dB")
    print(f"  HIP    vs Triton           SNR: {snr_db(dB_tri, dB_hip):.2f} dB")
    hip_ok = snr_db(dB_bf, dB_hip) >= 25.0
    print(f"  Gate (HIP >=25 dB vs bf16): {'PASS' if hip_ok else 'FAIL'}")

    if not hip_ok:
        return

    # ── Bench
    print("\n─── Bench (kernel-only, includes per-expert transpose+quant in HIP path) ───")
    def run_hip():
        return grouped_gemm_mxfp8_hip_wgrad(grad_out_bf, a_bf, group_offs, out_dtype=torch.bfloat16)

    def run_triton():
        return grouped_gemm_mxfp8_variable_k_triton_kernel(
            go_col, a_col, go_scale_col, a_scale_col,
            group_offs, go_scale_offs, out_dtype=torch.bfloat16,
        )

    for _ in range(3):
        run_hip(); run_triton()
    torch.cuda.synchronize()
    t_h = tbench.Timer(stmt="f()", globals={"f": run_hip}).timeit(args.iters).mean
    t_t = tbench.Timer(stmt="f()", globals={"f": run_triton}).timeit(args.iters).mean
    flops = 2.0 * m * n * k  # sum over G experts of 2*M_g*N*K = 2*M_total*N*K total
    print(f"  Triton  : {t_t*1e3:7.3f} ms  {flops/t_t/1e12:7.1f} TFLOPS")
    print(f"  HIP v1  : {t_h*1e3:7.3f} ms  {flops/t_h/1e12:7.1f} TFLOPS  ({t_t/t_h:.3f}x)")
    print()
    print("  NOTE: v1 HIP wgrad is single-launch fwd-kernel reuse (one global")
    print("        permute + one batched quant + one fwd call). Remaining gap to")
    print("        Triton is the fwd-kernel-at-wgrad-shape cost (256x256x128 tile")
    print("        is suboptimal for (M=G*N, K_red=M_g)). v2 purpose-built")
    print("        variable-K kernel is future work — see WGRAD_DESIGN.md.")


if __name__ == "__main__":
    main()
