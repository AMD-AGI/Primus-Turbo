#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""HipKittens / default-backend TFLOPS ratio metric for auto_optimize.

For each shape in a fixed LLM-typical suite (BF16 dense + FP8 tensorwise dense,
forward-only), this script:

  1. Times the op with HIPKITTEN forced as the GEMM backend.
  2. Times the op with no override (Primus's default dispatch — HIPBLASLT for
     BF16 / FP8-tensorwise, which is "Turbo's current best" on these shapes).
  3. Computes ratio = hk_tflops / default_tflops.
  4. If HIPKITTEN.can_handle rejects the shape (raises ValueError) or produces
     NaN/Inf, the ratio is clipped to 0.01 — which sinks geomean and forces the
     optimizer to actually extend HipKittens coverage instead of routing
     around hard cases.

Score (single integer printed on the last line of stdout, consumed by
auto_optimize.py):

    score = int(geomean(ratios) * 1000)

A score of 900 means HIPKITTEN is at 90% of the default backend across the
suite — which is the explicit DoD set by the user. 1000 means parity, >1000
means HIPKITTEN is faster on average.

What this metric will NOT reward:
  * Adding shapes to the suite without making HIPKITTEN actually competitive
    on them — geomean of 14 ratios is robust to cherry-picking.
  * "Fixing" rejection by skipping the shape — every entry in SUITE is
    counted, no exceptions.
  * Editing this script — auto_optimize.py forbids commits that only touch
    scripts/_metric_*.py.

Wall: ~10-15 seconds on an idle MI350. Auto-picks an idle GPU from
HIPKITTEN_GPU_POOL.
"""

from __future__ import annotations

import math
import os
import sys
import time
from contextlib import contextmanager
from typing import Optional

# Hint Primus where HipKittens lives so the BF16 / FP8 .so modules + caches
# are auto-discovered.
os.environ.setdefault("PRIMUS_TURBO_HIPKITTEN_PATH", "/workspace/code/HipKittens")


def _gpu_pool() -> Optional[set[int]]:
    raw = os.environ.get("HIPKITTEN_GPU_POOL", "").strip()
    if not raw:
        return None
    pool: set[int] = set()
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            pool.add(int(tok))
        except ValueError:
            pass
    return pool or None


def _pick_idle_gpu() -> Optional[str]:
    """Return the smallest idle GPU id within HIPKITTEN_GPU_POOL.

    Falls back to None when rocm-smi is unavailable so the caller can let the
    runtime choose the default device.
    """
    import re
    import subprocess
    THR = 100 * 1024 * 1024
    pool = _gpu_pool()
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showuse", "--showpids"],
            stderr=subprocess.DEVNULL, text=True, timeout=10,
        )
    except Exception:
        if pool:
            return str(min(pool))
        return None
    all_gpus = sorted({int(m) for m in re.findall(r"^GPU\[(\d+)\]", out, flags=re.M)})
    if pool is not None:
        all_gpus = [g for g in all_gpus if g in pool]
    busy: set[int] = set()
    in_kfd = False
    for line in out.splitlines():
        if "KFD process information" in line:
            in_kfd = True
            continue
        if not in_kfd:
            continue
        if line.startswith("=") or "PROCESS NAME" in line:
            continue
        cols = line.split()
        if len(cols) < 4 or not cols[0].isdigit():
            continue
        try:
            vram = int(cols[3])
        except ValueError:
            continue
        if vram <= THR:
            continue
        for gid in re.findall(r"\d+", cols[2]):
            busy.add(int(gid))
    idle = [g for g in all_gpus if g not in busy]
    if idle:
        return str(idle[0])
    return str(all_gpus[0]) if all_gpus else None


if "HIP_VISIBLE_DEVICES" not in os.environ:
    pick = _pick_idle_gpu()
    if pick is not None:
        os.environ["HIP_VISIBLE_DEVICES"] = pick
        print(f"[metric_hk_ratio] auto-picked HIP_VISIBLE_DEVICES={pick}", file=sys.stderr)


import torch  # noqa: E402

import primus_turbo.pytorch as turbo  # noqa: E402
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


# ----------------------------------------------------------------------------
# Suite — one row per (op, dtype, granularity, M, N, K, layout).
#
# Layout convention matches turbo.ops.gemm{_fp8}: trans_b=True is the
# canonical "RCR" / NT layout (a row-major, b row-major-then-transposed).
# Shapes are MBS=1 GEMMs from the canonical LLM benchmark suite
# (benchmark/ops/config.py::DenseModelConfigs) — so we measure exactly the
# shapes Megatron / DeepSeek-V3 / gpt_oss training pipelines hit, NOT
# arbitrary cache-friendly cherry picks.
# ----------------------------------------------------------------------------

# Llama-2-7B, MBS=1, seqlen=4096
_LLAMA2_7B = [
    (4096, 12288, 4096),  # attn_qkv
    (4096, 4096, 4096),   # attn_out
    (4096, 22016, 4096),  # mlp_gate_up
    (4096, 4096, 11008),  # mlp_down
]
# Llama-3.1-8B, MBS=1, seqlen=8192
_LLAMA31_8B = [
    (8192, 6144, 4096),   # attn_qkv
    (8192, 4096, 4096),   # attn_out
    (8192, 28672, 4096),  # mlp_gate_up
    (8192, 4096, 14336),  # mlp_down
]

BF16_SHAPES: list[tuple[int, int, int]] = _LLAMA2_7B + _LLAMA31_8B
FP8_SHAPES: list[tuple[int, int, int]] = _LLAMA2_7B + _LLAMA31_8B


# ----------------------------------------------------------------------------
# Backend forcing helpers
# ----------------------------------------------------------------------------

@contextmanager
def force_gemm_backend(backend: Optional[BackendType], precision: PrecisionType):
    """Temporarily pin the GEMM backend for one precision."""
    snapshot = GlobalBackendManager._gemm_backend  # type: ignore[attr-defined]
    GlobalBackendManager.reset()
    if backend is not None:
        GlobalBackendManager.set_gemm_backend(backend, precision)
    try:
        yield
    finally:
        GlobalBackendManager.reset()
        if snapshot is not None:
            GlobalBackendManager._gemm_backend = snapshot  # type: ignore[attr-defined]


# ----------------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------------

WARMUP = 10
ITERS = 50


def _time_op(fn) -> float:
    """Return min wall (ms) over ITERS launches after WARMUP."""
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
    # 20th percentile to reject outlier tails (matches HK bench convention).
    return times[len(times) // 5]


def _bench_bf16(M: int, N: int, K: int, backend: Optional[BackendType]) -> float:
    """Return TFLOPS, or 0.0 if backend rejects / NaNs out."""
    a = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")
    fn = lambda: turbo.ops.gemm(a, b, trans_b=True)  # noqa: E731
    with force_gemm_backend(backend, PrecisionType.BF16_FP16_FP32):
        try:
            out = fn()
        except Exception:
            return 0.0
        if torch.isnan(out).any() or torch.isinf(out).any():
            return 0.0
        try:
            ms = _time_op(fn)
        except Exception:
            return 0.0
    return 2.0 * M * N * K / (ms * 1e9)


def _bench_fp8(M: int, N: int, K: int, backend: Optional[BackendType]) -> float:
    """Return TFLOPS for FP8 tensorwise NT GEMM, or 0.0 on failure."""
    cfg = Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.TENSORWISE)
    a = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")
    fn = lambda: turbo.ops.gemm_fp8(a, b, trans_b=True, config=cfg)  # noqa: E731
    with force_gemm_backend(backend, PrecisionType.FP8):
        try:
            out = fn()
        except Exception:
            return 0.0
        if torch.isnan(out).any() or torch.isinf(out).any():
            return 0.0
        try:
            ms = _time_op(fn)
        except Exception:
            return 0.0
    return 2.0 * M * N * K / (ms * 1e9)


# ----------------------------------------------------------------------------
# Suite runner
# ----------------------------------------------------------------------------

def _run() -> int:
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        print("[metric_hk_ratio] CUDA/ROCm not available", file=sys.stderr)
        print(0)
        return 1

    rows: list[tuple[str, float, float, float]] = []
    t0 = time.monotonic()

    for (M, N, K) in BF16_SHAPES:
        ref = _bench_bf16(M, N, K, backend=None)
        hk = _bench_bf16(M, N, K, backend=BackendType.HIPKITTEN)
        ratio = (hk / ref) if ref > 0 else 0.0
        rows.append((f"BF16_{M}x{N}x{K}", hk, ref, ratio))

    for (M, N, K) in FP8_SHAPES:
        ref = _bench_fp8(M, N, K, backend=None)
        hk = _bench_fp8(M, N, K, backend=BackendType.HIPKITTEN)
        ratio = (hk / ref) if ref > 0 else 0.0
        rows.append((f"FP8_{M}x{N}x{K}", hk, ref, ratio))

    wall = time.monotonic() - t0

    # Print per-shape table to stderr so the auto_optimize log captures it
    # without polluting stdout (which carries the score).
    print(f"\n[metric_hk_ratio] Shape suite ({len(rows)} cases, {wall:.1f}s wall):", file=sys.stderr)
    print(
        f"  {'name':28s}  {'hk_tflops':>10s}  {'ref_tflops':>10s}  {'ratio':>6s}",
        file=sys.stderr,
    )
    for name, hk, ref, r in rows:
        flag = "" if r >= 0.9 else ("  *reject" if r == 0 else "  <90%")
        print(
            f"  {name:28s}  {hk:>10.1f}  {ref:>10.1f}  {r:>6.3f}{flag}",
            file=sys.stderr,
        )

    # Geomean over clipped ratios. Clipping at 0.01 keeps the geomean defined
    # but punishes rejection with a 100x drop on that entry.
    ratios = [max(r, 0.01) for _, _, _, r in rows]
    log_sum = sum(math.log(r) for r in ratios)
    geomean = math.exp(log_sum / len(ratios))
    score = int(round(geomean * 1000))

    n_reject = sum(1 for _, _, _, r in rows if r == 0)
    n_below_90 = sum(1 for _, _, _, r in rows if 0 < r < 0.9)

    print(
        f"[metric_hk_ratio] geomean={geomean:.4f}  "
        f"reject={n_reject}/{len(rows)}  below_90={n_below_90}/{len(rows)}  "
        f"score={score}",
        file=sys.stderr,
    )

    print(score)  # stdout: only the integer, parsed by auto_optimize.py
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
