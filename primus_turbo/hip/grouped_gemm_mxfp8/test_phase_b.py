###############################################################################
# Phase B correctness + bench for HIP MX-FP8 grouped GEMM dgrad.
#
# dA = dC @ B where dC [M, N], B [G, N, K]  =>  dA [M, K]
# Reuses the fwd kernel by pre-transposing B to dgrad layout [G, K, N].
###############################################################################

from __future__ import annotations

import argparse
import math

import torch
import torch.utils.benchmark as tbench

from primus_turbo.hip.grouped_gemm_mxfp8 import grouped_gemm_mxfp8_hip_dgrad
from primus_turbo.triton.grouped_gemm.grouped_gemm_mxfp8_kernel import (
    grouped_gemm_mxfp8_triton_kernel,
)
from primus_turbo.triton.quantization.mxfp8_quant_kernels import (
    quant_mxfp8_rowwise,
    quant_mxfp8_weight_dgrad,
)


def snr_db(ref: torch.Tensor, out: torch.Tensor) -> float:
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
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    if args.small:
        m, k, n, g = 1024, 512, 512, 4
    else:
        m, k, n, g = args.m, args.k, args.n, args.g

    torch.manual_seed(0)
    device = "cuda"
    print(f"Shape: dC=[M={m}, N={n}], B=[G={g}, N={n}, K={k}]  =>  dA=[M={m}, K={k}]")

    # Gen bf16 inputs
    dc_bf = torch.randn(m, n, device=device, dtype=torch.bfloat16)
    b_bf  = torch.randn(g, n, k, device=device, dtype=torch.bfloat16)  # fwd layout
    assert m % g == 0, "balanced MoE expected in Phase A/B"
    m_per = m // g
    group_offs = torch.arange(0, m + 1, m_per, dtype=torch.int64, device=device)

    # Quant dC rowwise MX-FP8 (scales along N axis of dC — which is dgrad's K-reduction axis)
    dc_fp8, dc_scale = quant_mxfp8_rowwise(dc_bf)

    # For dgrad B: start from fwd-layout b_bf [G, N, K], transpose to [G, K, N],
    # and run dgrad-layout weight quant (scales along N).
    b_dgrad_in  = b_bf.transpose(1, 2).contiguous()  # [G, K, N]
    b_dgrad_fp8, b_dgrad_scale = quant_mxfp8_weight_dgrad(b_dgrad_in)

    # ── Triton reference: run fwd with trans_b-like interpretation
    # Triton grouped_gemm_mxfp8_triton_kernel(a, b, a_s, b_s, group_offs, trans_b)
    #   trans_b=False: b is [G, K, N], b_scale is [G, K, N//32]-N-first? Actually
    #   the kernel expects b_scale [G, N, K//32] N-first regardless — but the
    #   dgrad quant produces [G, K, N//32]. We need to verify by running both.
    #
    # For the reference, compute bf16 dgrad directly.
    out_bf = torch.zeros(m, k, device=device, dtype=torch.bfloat16)
    for gi in range(g):
        s = int(group_offs[gi]); e = int(group_offs[gi + 1])
        out_bf[s:e] = (dc_bf[s:e].float() @ b_bf[gi].float()).to(torch.bfloat16)  # [M_g, N] @ [N, K] -> [M_g, K]

    # HIP dgrad
    out_hip = grouped_gemm_mxfp8_hip_dgrad(
        dc_fp8, b_dgrad_fp8, dc_scale, b_dgrad_scale, group_offs,
        out_dtype=torch.bfloat16,
    )

    print("\n─── Correctness ───")
    print(f"  HIP dgrad vs bf16 ref   SNR: {snr_db(out_bf, out_hip):.2f} dB")
    gate_ok = snr_db(out_bf, out_hip) >= 25.0
    print(f"  Gate (>=25 dB): {'PASS' if gate_ok else 'FAIL'}")

    if not gate_ok:
        return

    # Bench
    print("\n─── Bench (kernel-only) ───")
    def run_hip():
        return grouped_gemm_mxfp8_hip_dgrad(
            dc_fp8, b_dgrad_fp8, dc_scale, b_dgrad_scale, group_offs,
            out_dtype=torch.bfloat16,
        )

    # Warm + time
    for _ in range(3): run_hip()
    torch.cuda.synchronize()
    t = tbench.Timer(stmt="run_hip()", globals={"run_hip": run_hip}).timeit(args.iters)
    flops = 2.0 * m * k * n
    print(f"  HIP dgrad : {t.mean*1e3:7.3f} ms  {flops/t.mean/1e12:7.1f} TFLOPS")


if __name__ == "__main__":
    main()
