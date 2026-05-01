#!/usr/bin/env python3
"""Why does the metric report 38 TF for Triton FP8 on this shape?

Reproduces what the metric does, including the correctness check first."""
import os, sys, time, torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))
sys.path.insert(0, _REPO_ROOT)

import _metric_hk_ratio as hk_ratio
import _metric_grouped_only as mgo
import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.core.backend import BackendType, PrecisionType
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig, Format, ScalingGranularity,
)


for B, M, N, K in [(4, 2048, 2880, 2880)]:
    print(f"=== B={B} M={M} N={N} K={K} ===")

    # 1. JUST the bench (skip correctness check) — Triton
    torch.cuda.synchronize()
    trt = hk_ratio._bench_grouped_fp8(B, M, N, K, backend=BackendType.TRITON)
    print(f"  [no-correctness-check]  Triton FP8 = {trt:7.1f} TF")

    # 2. Bench Triton AGAIN on a fresh import — see if there's state leak
    torch.cuda.synchronize()
    trt2 = hk_ratio._bench_grouped_fp8(B, M, N, K, backend=BackendType.TRITON)
    print(f"  [no-correctness-check]  Triton FP8 (rerun) = {trt2:7.1f} TF")

    # 3. Run correctness check first (mimics metric ordering), then Triton
    torch.cuda.synchronize()
    ok, reason = mgo._check_grouped_fp8_correctness(B, M, N, K)
    print(f"  correctness: ok={ok} reason={reason}")
    torch.cuda.synchronize()
    trt3 = hk_ratio._bench_grouped_fp8(B, M, N, K, backend=BackendType.TRITON)
    print(f"  [POST-correctness]      Triton FP8 = {trt3:7.1f} TF")

    # 4. HIPKITTEN bench
    hk = hk_ratio._bench_grouped_fp8(B, M, N, K, backend=BackendType.HIPKITTEN)
    print(f"  [POST-correctness]      HK     FP8 = {hk:7.1f} TF")
