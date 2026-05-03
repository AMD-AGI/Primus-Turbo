"""HK vs Triton kernel-only TFLOPS for forward + dA + dB var-K.

Why this script exists (R25): the wall metric
``scripts/_metric_grouped_fused_wall.py`` measures full fwd + bwd wall
including ``quantize_fp8(grad_out)`` (~85 µs / call delayed-cache hit).
That cost is paid equally by HK and Triton, but it dilutes the kernel-
only HK lead in the wall ratio. To know where the actual lever lives
(GEMM kernel vs Python overhead vs quantize), we need the kernel-only
breakdown.

Usage:
    python3 scripts/_probe_kernel_only_hk_vs_triton.py [--shapes BOTTOM|FULL]

Output: per-shape (HK_TFLOPS, Triton_TFLOPS, ratio) for fwd / dA / var-K
plus geomean. Decision rules:

    * fwd ratio < 1.0 → HK forward kernel IS the wedge → MFMA cell port
      worthwhile.
    * fwd ratio >= 1.05 → HK forward kernel already wins → don't bother.
    * dB var-K ratio < 1.0 → HK var-K behind Triton → BK=64 templating
      worthwhile.
    * dB var-K ratio >= 1.0 → HK var-K already wins → don't bother.

R25 baseline measurement (HEAD a418818 Primus-Turbo, a7683112 HK):

    fwd kernel-only geomean      = 1.139    (HK +14% over Triton)
    dA  kernel-only ~= fwd ratio (same kernel template)
    dB  var-K kernel-only geomean = 1.840   (HK +84% over Triton)

Conclusion (R25 doc): HK kernels already dominate Triton; the wall
metric gap is in non-kernel overhead (quantize_fp8 ~85 µs per call).
The lever is **Path A backward** (fuse BF16->FP8 cvt into dA / dB
kernels, skipping the standalone quantize launch).
"""
import argparse
import os
os.environ.setdefault("PRIMUS_TURBO_HIPKITTEN_PATH", "/workspace/code/HipKittens")

import sys
import math
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
import _metric_hk_ratio as hk_ratio  # noqa: E402

import primus_turbo  # noqa: F401  E402
from primus_turbo.pytorch.core.backend import BackendType, PrecisionType  # noqa: E402
from primus_turbo.pytorch.core.low_precision import (  # noqa: E402
    Float8QuantConfig, Format, ScalingGranularity,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (  # noqa: E402
    grouped_gemm_compute_offs,
    grouped_gemm_fp8_impl,
    grouped_gemm_fp8_variable_k_impl,
)
from primus_turbo.pytorch.ops.quantization import quantize_fp8  # noqa: E402

torch.manual_seed(0)


def _bench_fwd(B, M, N, K, backend: BackendType) -> float:
    cfg = Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.TENSORWISE)
    a = torch.randn((B * M, K), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((B, N, K), dtype=torch.bfloat16, device="cuda")
    g_lens = torch.full((B,), M, dtype=torch.int64, device="cuda")
    g_offs = grouped_gemm_compute_offs(g_lens)
    a_fp8, a_s = quantize_fp8(a, torch.float8_e4m3fn, cfg.granularity)
    b_fp8, b_s = quantize_fp8(b, torch.float8_e4m3fn, cfg.granularity)

    def _call():
        return grouped_gemm_fp8_impl(
            a_fp8, b_fp8, a_s, b_s, g_lens, g_offs,
            trans_a=False, trans_b=True, out_dtype=torch.bfloat16,
            granularity=cfg.granularity.value, num_cu=None,
            default_backend=BackendType.HIPKITTEN.value,
        )

    with hk_ratio.force_grouped_gemm_backend(backend, PrecisionType.FP8):
        for _ in range(3): _call()
        torch.cuda.synchronize()
        ms = hk_ratio._time_op(_call)
    return 2.0 * (B * M) * N * K / (ms * 1e9)


def _bench_var_k(B, M, N, K, backend: BackendType) -> float:
    cfg = Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.TENSORWISE)
    a = torch.randn((B * M, K), dtype=torch.bfloat16, device="cuda")
    grad = torch.randn((B * M, N), dtype=torch.bfloat16, device="cuda")
    g_lens = torch.full((B,), M, dtype=torch.int64, device="cuda")
    g_offs = grouped_gemm_compute_offs(g_lens)
    a_col, a_s_col = quantize_fp8(a, torch.float8_e4m3fn, cfg.granularity, axis=-2)
    g_col, g_s_col = quantize_fp8(grad, torch.float8_e4m3fn, cfg.granularity, axis=-2)

    def _call():
        return grouped_gemm_fp8_variable_k_impl(
            a_col, g_col, a_s_col, g_s_col, g_lens, g_offs,
            trans_a=True, trans_b=False, trans_c=True,
            out_dtype=torch.bfloat16, granularity=cfg.granularity.value, num_cu=None,
            default_backend=BackendType.HIPKITTEN.value,
        )

    with hk_ratio.force_grouped_gemm_backend(backend, PrecisionType.FP8):
        for _ in range(3): _call()
        torch.cuda.synchronize()
        ms = hk_ratio._time_op(_call)
    return 2.0 * (B * M) * N * K / (ms * 1e9)


SHAPES_FULL = [
    (16, 2048, 7168, 4096, "DSV3-GateUP-B16-M2048"),
    (16, 4096, 7168, 4096, "DSV3-GateUP-B16-M4096"),
    (32, 2048, 7168, 4096, "DSV3-GateUP-B32-M2048"),
    (32, 4096, 7168, 4096, "DSV3-GateUP-B32-M4096"),
    (16, 2048, 4096, 7168, "DSV3-Down-B16-M2048"),
    (16, 4096, 4096, 7168, "DSV3-Down-B16-M4096"),
    (32, 2048, 4096, 7168, "DSV3-Down-B32-M2048"),
    (32, 4096, 4096, 7168, "DSV3-Down-B32-M4096"),
    (4,  2048, 5760, 2880, "gpt_oss-GateUP-B4-M2048"),
    (4,  4096, 5760, 2880, "gpt_oss-GateUP-B4-M4096"),
    (32, 2048, 5760, 2880, "gpt_oss-GateUP-B32-M2048"),
    (32, 4096, 5760, 2880, "gpt_oss-GateUP-B32-M4096"),
    (4,  2048, 2880, 2880, "gpt_oss-Down-B4-M2048"),
    (4,  4096, 2880, 2880, "gpt_oss-Down-B4-M4096"),
    (32, 2048, 2880, 2880, "gpt_oss-Down-B32-M2048"),
    (32, 4096, 2880, 2880, "gpt_oss-Down-B32-M4096"),
    (16, 2048, 1536, 4096, "Qwen3-GateUP-B16-M2048"),
    (16, 4096, 1536, 4096, "Qwen3-GateUP-B16-M4096"),
    (32, 2048, 1536, 4096, "Qwen3-GateUP-B32-M2048"),
    (32, 4096, 1536, 4096, "Qwen3-GateUP-B32-M4096"),
    (16, 2048, 4096, 1536, "Qwen3-Down-B16-M2048"),
    (16, 4096, 4096, 1536, "Qwen3-Down-B16-M4096"),
    (32, 2048, 4096, 1536, "Qwen3-Down-B32-M2048"),
    (32, 4096, 4096, 1536, "Qwen3-Down-B32-M4096"),
]

SHAPES_BOTTOM = [
    (16, 2048, 4096, 1536, "Qwen3-Down-B16-M2048"),
    (16, 2048, 1536, 4096, "Qwen3-GateUP-B16-M2048"),
    (16, 4096, 4096, 1536, "Qwen3-Down-B16-M4096"),
    (32, 2048, 2880, 2880, "gpt_oss-Down-B32-M2048"),
    (16, 2048, 7168, 4096, "DSV3-GateUP-B16-M2048"),
]


def _run(shapes):
    print(f"{'shape':<30} {'M_total':>8} {'N':>5} {'K':>5} | "
          f"{'fwd HK':>8} {'fwd Trt':>8} {'fwd_r':>6} | "
          f"{'dB HK':>8} {'dB Trt':>8} {'dB_r':>6}")
    print("-" * 110)
    fwd_rs, dB_rs = [], []
    for B, M, N, K, name in shapes:
        fhk = _bench_fwd(B, M, N, K, BackendType.HIPKITTEN)
        ftr = _bench_fwd(B, M, N, K, BackendType.TRITON)
        bhk = _bench_var_k(B, M, N, K, BackendType.HIPKITTEN)
        btr = _bench_var_k(B, M, N, K, BackendType.TRITON)
        if fhk and ftr and bhk and btr:
            fr, br = fhk / ftr, bhk / btr
            fwd_rs.append(fr); dB_rs.append(br)
            print(f"  {name:<28} {B*M:>8} {N:>5} {K:>5} | "
                  f"{fhk:>8.0f} {ftr:>8.0f} {fr:>6.3f} | "
                  f"{bhk:>8.0f} {btr:>8.0f} {br:>6.3f}")
    if fwd_rs:
        gm_fwd = math.exp(sum(math.log(r) for r in fwd_rs) / len(fwd_rs))
        gm_dB  = math.exp(sum(math.log(r) for r in dB_rs)  / len(dB_rs))
        print()
        print(f"fwd kernel-only geomean (HK / Triton) = {gm_fwd:.3f}")
        print(f"dB  kernel-only geomean (HK / Triton) = {gm_dB:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", choices=["BOTTOM", "FULL"], default="FULL",
                   help="BOTTOM = 5 worst metric shapes; FULL = all 24 metric shapes.")
    args = p.parse_args()
    _run(SHAPES_BOTTOM if args.shapes == "BOTTOM" else SHAPES_FULL)
