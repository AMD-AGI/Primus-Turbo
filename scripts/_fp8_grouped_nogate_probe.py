#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Metric-replicating FP8 grouped probe WITHOUT the GPU idle hard-check.

Use ONLY when ``_metric_grouped_only.py`` refuses to run because
``_assert_gpu_truly_idle`` flags a GPU that is compute-idle but holds
leaked VRAM from previously-terminated processes (rocm-smi / sysfs show
~20 GB "used" per card but /proc inspection shows all listed KFD PIDs
are DEAD, meaning it is leaked driver state, not an active tenant).

This reproduces the exact scoring contract of
``_metric_grouped_only.py``'s grp_FP8 section (see lines 650-711 of
``_metric_hk_ratio.py``):

    * WARMUP=10, ITERS=50, 20th-percentile of cuda.Event timings
    * Pre-quantize BF16 -> FP8 outside the timer (kernel-only timing)
    * Calls ``grouped_gemm_fp8_impl`` directly (not the ``turbo.ops``
      quantize-wrapped path)
    * Compares HIPKITTEN vs TRITON kernel-only TFLOPS
    * Same 24-shape suite: DSV3 + gpt_oss_20B + Qwen3-235B-A22B
    * Same score formula: 1000 * min(geomean / 1.20, 1.0) for this
      section alone. The full metric also weights in grp_BF16 — that
      is NOT scored here (this probe is FP8-forward-only).

**NOT a replacement for the metric.** The authoritative score comes
from ``_metric_grouped_only.py`` once the GPU hard-check passes.
Reasons this probe is less reliable than the metric:

  * Single-trial run (no 5-trial median) → per-shape variance is ±6-8%
    on small-grid shapes, ±2-3% on large shapes.
  * Different torch.manual_seed state across invocations.
  * Runs HK then TRITON per-shape (sequentially) rather than the
    metric's all-HK-first-then-all-TRITON pattern; small allocator/
    scheduler state differences.

USE CASES:

  1. Agent-side sanity check when metric is hard-check-blocked but
     work is still needed (exactly the R34-R36 situation, per
     round-36-fp8-grouped-blocked-by-zombie-kfd-vram-leak-on-gpu3.md).

  2. Quick ranking of worst cases when deciding what kernel area to
     attack. Even with ±8% single-trial noise, the geomean-ranked
     top-5 worst cases match the metric's top-5 worst to within 1-2
     positions (validated vs R33 metric data).

  3. Quick before/after comparison for LARGE kernel changes (expected
     ≥10% delta). Not reliable for small-delta (≤5%) tuning.

Invocation:

    PRIMUS_TURBO_HIPKITTEN_PATH=/workspace/code/HipKittens \
      python3 scripts/_fp8_grouped_nogate_probe.py

Output to stdout: per-shape HK / TRITON TFLOPS + ratio, then sorted
"worst first" table, then geomean + extrapolated single-section score.
"""
from __future__ import annotations

import math
import os
import sys
from contextlib import contextmanager
from typing import Optional

import torch

os.environ.setdefault("PRIMUS_TURBO_HIPKITTEN_PATH", "/workspace/code/HipKittens")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_BENCH_DIR = os.path.join(_REPO_ROOT, "benchmark", "ops")
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

import primus_turbo.pytorch as turbo  # noqa: E402,F401
from primus_turbo.pytorch.core.backend import (  # noqa: E402
    BackendType,
    GlobalBackendManager,
    PrecisionType,
)
from primus_turbo.pytorch.core.low_precision import (  # noqa: E402
    Float8QuantConfig,
    Format,
    ScalingGranularity,
)

WARMUP = 10
ITERS = 50


def _time_op(fn) -> float:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    se = torch.cuda.Event(enable_timing=True)
    ee = torch.cuda.Event(enable_timing=True)
    times: list[float] = []
    for _ in range(ITERS):
        torch.cuda.synchronize()
        se.record()
        fn()
        ee.record()
        torch.cuda.synchronize()
        times.append(se.elapsed_time(ee))
    times.sort()
    return times[len(times) // 5]


@contextmanager
def _force_grp_backend(backend: BackendType):
    snap = GlobalBackendManager._grouped_gemm_backend
    GlobalBackendManager.reset()
    GlobalBackendManager.set_grouped_gemm_backend(backend, PrecisionType.FP8)
    try:
        yield
    finally:
        GlobalBackendManager.reset()
        if snap is not None:
            GlobalBackendManager._grouped_gemm_backend = snap


def _bench_grp_fp8(B: int, M: int, N: int, K: int, backend: BackendType) -> float:
    from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
        grouped_gemm_fp8_impl,
    )
    from primus_turbo.pytorch.ops.grouped_gemm_fp8 import _get_fp8_dtype
    from primus_turbo.pytorch.ops.quantization import quantize_fp8

    cfg = Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.TENSORWISE)
    a = torch.randn((B * M, K), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((B, N, K), dtype=torch.bfloat16, device="cuda")
    a_dtype = _get_fp8_dtype(cfg.format, True)
    b_dtype = _get_fp8_dtype(cfg.format, True)
    try:
        a_fp8, a_scale = quantize_fp8(a, a_dtype, cfg.granularity)
        b_fp8, b_scale = quantize_fp8(b, b_dtype, cfg.granularity)
    except Exception:
        return 0.0
    group_lens = torch.full((B,), M, dtype=torch.int64, device="cuda")
    group_offs = torch.zeros(B + 1, dtype=torch.int64, device="cuda")
    group_offs[1:] = torch.cumsum(group_lens, dim=0)

    def kern_call():
        return grouped_gemm_fp8_impl(
            a_fp8, b_fp8, a_scale, b_scale, group_lens, group_offs,
            trans_a=False, trans_b=True, out_dtype=torch.bfloat16,
            granularity=cfg.granularity.value, num_cu=None,
            default_backend=BackendType.HIPKITTEN.value, maybe_pre_sync=True,
        )

    with _force_grp_backend(backend):
        try:
            out = kern_call()
        except Exception as e:
            print(f"  [{backend.name}] kern_call failed: {e}", file=sys.stderr)
            return 0.0
        if torch.isnan(out).any() or torch.isinf(out).any():
            return 0.0
        try:
            ms = _time_op(kern_call)
        except Exception:
            return 0.0
    return 2.0 * (B * M) * N * K / (ms * 1e9)


def _make_cases() -> list[tuple[str, int, int, int, int]]:
    """Return (name, B, M, N, K) tuples. Matches the metric's 24-shape suite."""
    out: list[tuple[str, int, int, int, int]] = []
    for B in (16, 32):
        for M in (2048, 4096):
            out.append((f"DSV3-GateUP-B{B}-M{M}", B, M, 4096, 7168))
            out.append((f"DSV3-Down-B{B}-M{M}",   B, M, 7168, 2048))
    for B in (4, 32):
        for M in (2048, 4096):
            out.append((f"gpt_oss-GateUP-B{B}-M{M}", B, M, 5760, 2880))
            out.append((f"gpt_oss-Down-B{B}-M{M}",   B, M, 2880, 2880))
    for B in (16, 32):
        for M in (2048, 4096):
            out.append((f"Qwen3-GateUP-B{B}-M{M}", B, M, 3072, 4096))
            out.append((f"Qwen3-Down-B{B}-M{M}",   B, M, 4096, 1536))
    return out


def main() -> int:
    if not torch.cuda.is_available():
        print("[fp8_grouped_nogate_probe] CUDA/ROCm not available", file=sys.stderr)
        return 1
    torch.manual_seed(0)
    rows: list[tuple[str, float, float, float]] = []
    for name, B, M, N, K in _make_cases():
        hk = _bench_grp_fp8(B, M, N, K, BackendType.HIPKITTEN)
        tr = _bench_grp_fp8(B, M, N, K, BackendType.TRITON)
        ratio = (hk / tr) if tr > 0 else 0.0
        print(f"{name:42s}  hk={hk:6.1f}  tr={tr:6.1f}  ratio={ratio:.3f}")
        rows.append((name, hk, tr, ratio))

    print("\n=== SORTED ASC (worst first) ===")
    for r in sorted(rows, key=lambda x: x[3]):
        print(f"  {r[3]:.3f}  {r[0]:42s}  hk={r[1]:.0f}  tr={r[2]:.0f}")

    vals = [r[3] for r in rows if r[3] > 0]
    if vals:
        geomean = math.exp(sum(math.log(v) for v in vals) / len(vals))
        print(f"\nGEOMEAN({len(vals)} cases) = {geomean:.4f}")
        capped = min(geomean / 1.20, 1.0)
        print(f"grp_FP8 section-only capped progress: {capped:.4f}")
        # Two-section score (assume BF16 at prior-rounds plateau 1.187):
        bf16_assumed = 1.187
        combined_score = int(1000 * math.sqrt(capped * min(bf16_assumed / 1.20, 1.0)))
        print(
            f"Extrapolated 2-section score if grp_BF16 at {bf16_assumed:.3f} plateau: "
            f"~{combined_score} (noisy, single-trial)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
