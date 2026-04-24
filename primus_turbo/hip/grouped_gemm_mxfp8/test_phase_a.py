###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Phase A correctness + bench harness for HIP MX-FP8 grouped GEMM.
#
# Compares HIP grouped GEMM vs Triton tl.dot_scaled reference on the
# gpt_oss_20B MoE gate_up shape (M_total=65536, K=2880, N=5760, G=32, balanced).
# Reports SNR and kernel-only TFLOPS.
###############################################################################

from __future__ import annotations

import argparse
import math

import torch
import torch.utils.benchmark as tbench

from primus_turbo.hip.grouped_gemm_mxfp8 import grouped_gemm_mxfp8_hip_fwd
from primus_turbo.triton.grouped_gemm.grouped_gemm_mxfp8_kernel import (
    grouped_gemm_mxfp8_triton_kernel,
)
from primus_turbo.triton.quantization.mxfp8_quant_kernels import (
    quant_mxfp8_rowwise,
    quant_mxfp8_weight_fwd,
)


def snr_db(ref: torch.Tensor, out: torch.Tensor) -> float:
    ref_f = ref.float()
    out_f = out.float()
    noise = (ref_f - out_f).pow(2).mean().item()
    signal = ref_f.pow(2).mean().item()
    if noise == 0.0:
        return float("inf")
    return 10.0 * math.log10(signal / max(noise, 1e-30))


def make_balanced_offs(m_total: int, g: int, device) -> torch.Tensor:
    """Balanced prefix-sum such that each M_g = m_total / g."""
    assert m_total % g == 0
    m_per = m_total // g
    offs = torch.arange(0, m_total + 1, m_per, dtype=torch.int64, device=device)
    return offs


def gen_inputs(m_total: int, k: int, n: int, g: int, device="cuda"):
    a_bf = torch.randn(m_total, k, device=device, dtype=torch.bfloat16)
    b_bf = torch.randn(g, n, k, device=device, dtype=torch.bfloat16)  # [G, N, K] NT

    # Quant A rowwise MX-FP8 — layout [M_total, K//32] e8m0
    a_fp8, a_scale = quant_mxfp8_rowwise(a_bf)
    # Quant B per-expert fwd — Triton's weight_fwd returns [G, K, N] FP8 + [G, N, K//32] scale
    # but we need [G, N, K] layout for HIP NT kernel. Swap.
    # Easier: call quant_mxfp8_rowwise on b reshaped to [G*N, K]
    b_flat_fp8, b_flat_scale = quant_mxfp8_rowwise(b_bf.reshape(g * n, k))
    b_fp8 = b_flat_fp8.view(g, n, k)
    b_scale = b_flat_scale.view(g, n, k // 32)

    group_offs = make_balanced_offs(m_total, g, device=device)
    return a_fp8, b_fp8, a_scale, b_scale, group_offs, a_bf, b_bf


def run_triton_ref(a_fp8, b_fp8, a_scale, b_scale, group_offs):
    """Triton reference expects B in [G, K, N] layout with trans_b=False OR
    [G, N, K] with trans_b=True. We have [G, N, K], so use trans_b=True.

    IMPORTANT: b_scale [G, N, K//32] matches tl.dot_scaled's required N-first
    layout for both trans_b paths."""
    return grouped_gemm_mxfp8_triton_kernel(
        a_fp8, b_fp8, a_scale, b_scale, group_offs,
        trans_b=True, out_dtype=torch.bfloat16
    )


def run_hip(a_fp8, b_fp8, a_scale, b_scale, group_offs):
    return grouped_gemm_mxfp8_hip_fwd(
        a_fp8, b_fp8, a_scale, b_scale, group_offs,
        out_dtype=torch.bfloat16,
    )


def bench_kernel(fn, *args, n_iters: int = 100) -> float:
    torch.cuda.synchronize()
    t = tbench.Timer(
        stmt="fn(*args)",
        globals={"fn": fn, "args": args},
    )
    # Warm
    for _ in range(3):
        fn(*args)
    torch.cuda.synchronize()
    measurement = t.timeit(n_iters)
    return measurement.mean  # seconds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=65536)
    ap.add_argument("--k", type=int, default=2880)
    ap.add_argument("--n", type=int, default=5760)
    ap.add_argument("--g", type=int, default=32)
    ap.add_argument("--small", action="store_true",
                    help="Small sanity shape: M=1024, K=512, N=512, G=4")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--skip-bench", action="store_true")
    args = ap.parse_args()

    if args.small:
        m, k, n, g = 1024, 512, 512, 4
    else:
        m, k, n, g = args.m, args.k, args.n, args.g

    torch.manual_seed(0)
    device = "cuda"
    print(f"Shape: M={m}, K={k}, N={n}, G={g}, balanced (M_g={m//g})")

    a_fp8, b_fp8, a_scale, b_scale, group_offs, a_bf, b_bf = gen_inputs(m, k, n, g, device=device)

    # Correctness
    print("\n─── Correctness ───")
    out_triton = run_triton_ref(a_fp8, b_fp8, a_scale, b_scale, group_offs)
    out_hip = run_hip(a_fp8, b_fp8, a_scale, b_scale, group_offs)

    # Reference: bf16 grouped gemm
    # For per-expert: C[m_start:m_end] = A[m_start:m_end] @ B[g].T  (since B is [G, N, K] NT)
    out_bf = torch.zeros_like(out_triton)
    for gi in range(g):
        s = group_offs[gi].item()
        e = group_offs[gi + 1].item()
        out_bf[s:e] = (a_bf[s:e].float() @ b_bf[gi].float().T).to(torch.bfloat16)

    snr_tri_vs_bf = snr_db(out_bf, out_triton)
    snr_hip_vs_bf = snr_db(out_bf, out_hip)
    snr_hip_vs_tri = snr_db(out_triton, out_hip)
    print(f"  Triton vs bf16 ref   SNR: {snr_tri_vs_bf:.2f} dB")
    print(f"  HIP    vs bf16 ref   SNR: {snr_hip_vs_bf:.2f} dB")
    print(f"  HIP    vs Triton     SNR: {snr_hip_vs_tri:.2f} dB  (should be > 40 dB)")

    gate_ok = snr_hip_vs_bf >= 25.0
    print(f"  Gate (>=25 dB vs bf16): {'PASS' if gate_ok else 'FAIL'}")

    if args.skip_bench:
        return

    print("\n─── Bench (kernel-only) ───")
    flops = 2.0 * m * k * n
    t_hip = bench_kernel(run_hip, a_fp8, b_fp8, a_scale, b_scale, group_offs, n_iters=args.iters)
    t_tri = bench_kernel(run_triton_ref, a_fp8, b_fp8, a_scale, b_scale, group_offs, n_iters=args.iters)
    print(f"  Triton : {t_tri*1e3:7.3f} ms  {flops/t_tri/1e12:7.1f} TFLOPS")
    print(f"  HIP    : {t_hip*1e3:7.3f} ms  {flops/t_hip/1e12:7.1f} TFLOPS")
    print(f"  HIP / Triton: {t_tri/t_hip:.3f}x")


if __name__ == "__main__":
    main()
