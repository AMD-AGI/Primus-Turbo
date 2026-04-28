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


# Make benchmark/ops/config.py importable so we share the canonical MoE shape
# table with the rest of the repo (DeepSeek-V3 / gpt_oss_20B grouped GEMM
# suite is generated there, NOT duplicated here).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

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
from benchmark.ops.config import (  # noqa: E402
    GROUPED_GEMM_M_SIZE_LIST,
    MoEModelConfigs,
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


def _grouped_suite() -> list[tuple[str, int, int, int, int]]:
    """Return [(name, B, M_per_group, N, K)] for DeepSeek-V3 + gpt_oss_20B.

    Mirrors benchmark/ops/config.py::gen_grouped_gemm_test_cases():
      * batch_sizes per model (DeepSeek-V3=[16,32], gpt_oss_20B=[4,32])
      * M_per_group ∈ GROUPED_GEMM_M_SIZE_LIST = [2048, 4096]
      * layer ∈ {GateUP=(2*moe_int, hidden), Down=(hidden, moe_int)}
      * skip when n_routed_experts %% B != 0

    DeepSeek-V3 (n_routed=256, moe_int=2048, hidden=7168):
      * 2 batch × 2 M × 2 layer = 8 cases
    gpt_oss_20B (n_routed=32, moe_int=2880, hidden=2880):
      * 2 batch × 2 M × 2 layer = 8 cases
    Total: 16 cases.
    """
    rows: list[tuple[str, int, int, int, int]] = []
    for model_name, cfg in MoEModelConfigs.items():
        if "moe_intermediate_size" not in cfg or "hidden_size" not in cfg:
            continue
        n_routed = cfg["n_routed_experts"]
        moe_int = cfg["moe_intermediate_size"]
        hidden = cfg["hidden_size"]
        batch_sizes = cfg.get("grouped_gemm_batch_sizes", [4, 16, 32])
        layers = {
            "GateUP": (2 * moe_int, hidden),
            "Down": (hidden, moe_int),
        }
        for B in batch_sizes:
            if n_routed % B != 0:
                continue
            for M in GROUPED_GEMM_M_SIZE_LIST:
                for layer, (N, K) in layers.items():
                    name = f"{model_name}-{layer}-B{B}-M{M}"
                    rows.append((name, B, M, N, K))
    return rows


GROUPED_BF16_SHAPES: list[tuple[str, int, int, int, int]] = _grouped_suite()


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


@contextmanager
def force_grouped_gemm_backend(backend: Optional[BackendType], precision: PrecisionType):
    """Temporarily pin the grouped-GEMM backend for one precision."""
    snapshot = GlobalBackendManager._grouped_gemm_backend  # type: ignore[attr-defined]
    GlobalBackendManager.reset()
    if backend is not None:
        GlobalBackendManager.set_grouped_gemm_backend(backend, precision)
    try:
        yield
    finally:
        GlobalBackendManager.reset()
        if snapshot is not None:
            GlobalBackendManager._grouped_gemm_backend = snapshot  # type: ignore[attr-defined]


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
    """BF16 NT GEMM forward TFLOPS for the given backend, 0.0 on reject/NaN."""
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


def _bench_bf16_bwd(M: int, N: int, K: int, backend: Optional[BackendType]) -> float:
    """BF16 NT GEMM **backward only** TFLOPS; same time-subtraction approach
    as :func:`_bench_fp8_bwd`. Backward FLOPs = 2 * forward = 4 * M * N * K
    (a_grad + b_grad GEMMs).

    NOTE: ``out.sum().backward()`` would feed an expanded scalar gradient
    (stride=0) into the backward GEMM, which hipBLASLt rejects with
    ``A must be contiguous``. We construct a fresh contiguous ``dy``
    once and pass it via ``out.backward(dy)`` instead.
    """
    a_ng = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    b_ng = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")
    a_g = torch.randn((M, K), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    b_g = torch.randn((N, K), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    dy = torch.randn((M, N), dtype=torch.bfloat16, device="cuda")
    fwd_only = lambda: turbo.ops.gemm(a_ng, b_ng, trans_b=True)  # noqa: E731

    def fwd_bwd():
        a_g.grad = None
        b_g.grad = None
        out = turbo.ops.gemm(a_g, b_g, trans_b=True)
        out.backward(dy)

    with force_gemm_backend(backend, PrecisionType.BF16_FP16_FP32):
        try:
            out = fwd_only()
            fwd_bwd()
        except Exception:
            return 0.0
        if torch.isnan(out).any() or torch.isinf(out).any():
            return 0.0
        if a_g.grad is None or b_g.grad is None:
            return 0.0
        if torch.isnan(a_g.grad).any() or torch.isnan(b_g.grad).any():
            return 0.0
        try:
            t_fwd = _time_op(fwd_only)
            t_total = _time_op(fwd_bwd)
        except Exception:
            return 0.0
    t_bwd = max(t_total - t_fwd, 1e-3)
    return 4.0 * M * N * K / (t_bwd * 1e9)


def _bench_fp8(M: int, N: int, K: int, backend: Optional[BackendType]) -> float:
    """Return TFLOPS for FP8 tensorwise NT GEMM forward, or 0.0 on failure."""
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


def _bench_fp8_bwd(M: int, N: int, K: int, backend: Optional[BackendType]) -> float:
    """Return TFLOPS for FP8 tensorwise NT GEMM **backward only**, or 0.0 on failure.

    Backward issues two GEMMs of the same size as forward (one for grad_a,
    one for grad_b), so the work is 2x the forward GEMM:
        forward FLOPs  = 2 * M * N * K
        backward FLOPs = 2 * forward FLOPs = 4 * M * N * K

    To isolate backward time we time the forward and the (forward+backward)
    paths separately and subtract — ``retain_graph=True`` tricks would let
    us time backward in a tighter loop, but they share saved tensors across
    iterations which can hide quantization overhead. The subtraction
    approach is noisier per-sample but uses 20th-percentile selection
    inside :func:`_time_op`, which is robust enough.
    """
    cfg = Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.TENSORWISE)
    a_ng = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    b_ng = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")
    a_g = torch.randn((M, K), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    b_g = torch.randn((N, K), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    # Contiguous dy: ``out.sum().backward()`` would feed a stride-0 expanded
    # scalar that hipBLASLt rejects.
    dy = torch.randn((M, N), dtype=torch.bfloat16, device="cuda")
    fwd_only = lambda: turbo.ops.gemm_fp8(a_ng, b_ng, trans_b=True, config=cfg)  # noqa: E731

    def fwd_bwd():
        a_g.grad = None
        b_g.grad = None
        out = turbo.ops.gemm_fp8(a_g, b_g, trans_b=True, config=cfg)
        out.backward(dy)

    with force_gemm_backend(backend, PrecisionType.FP8):
        try:
            out = fwd_only()
            fwd_bwd()
        except Exception:
            return 0.0
        if torch.isnan(out).any() or torch.isinf(out).any():
            return 0.0
        if a_g.grad is None or b_g.grad is None:
            return 0.0
        if torch.isnan(a_g.grad).any() or torch.isnan(b_g.grad).any():
            return 0.0
        try:
            t_fwd = _time_op(fwd_only)
            t_total = _time_op(fwd_bwd)
        except Exception:
            return 0.0
    t_bwd = max(t_total - t_fwd, 1e-3)  # guard against measurement noise
    return 4.0 * M * N * K / (t_bwd * 1e9)


def _bench_grouped_bf16(
    B: int, M: int, N: int, K: int, backend: Optional[BackendType]
) -> float:
    """Return TFLOPS for BF16 grouped GEMM (NT, balanced groups), or 0.0 on failure.

    Layout matches turbo.ops.grouped_gemm with trans_b=True:
        a: [B*M, K]   bf16
        b: [B, N, K]  bf16
        out: [B*M, N] bf16
    Total FLOPs = 2 * (B*M) * N * K.
    """
    a = torch.randn((B * M, K), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((B, N, K), dtype=torch.bfloat16, device="cuda")
    group_lens = torch.full((B,), M, dtype=torch.int64, device="cuda")
    fn = lambda: turbo.ops.grouped_gemm(a, b, group_lens, trans_b=True)  # noqa: E731
    with force_grouped_gemm_backend(backend, PrecisionType.BF16_FP16_FP32):
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
    return 2.0 * (B * M) * N * K / (ms * 1e9)


# ----------------------------------------------------------------------------
# Suite runner
# ----------------------------------------------------------------------------

def _run() -> int:
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        print("[metric_hk_ratio] CUDA/ROCm not available", file=sys.stderr)
        print(0)
        return 1

    # Each row carries its own reference backend so different sections can
    # compare against different baselines, per project policy:
    #   - dense GEMM forward / backward (BF16 + FP8 fwd) compete with HIPBLASLT
    #   - FP8 dense backward competes with TRITON (TRITON is the canonical
    #     reference for FP8 backward since hipBLASLt's FP8 backward path
    #     differs in quantization recipe)
    #   - grouped BF16 competes with TRITON
    rows: list[tuple[str, float, float, float, str, str]] = []
    # row tuple: (name, hk_tflops, ref_tflops, ratio, ref_backend_name, section)
    t0 = time.monotonic()
    HIPB = BackendType.HIPBLASLT
    TRT = BackendType.TRITON
    HK = BackendType.HIPKITTEN

    # ── (1) Dense BF16 forward (vs HIPBLASLT) ──
    for (M, N, K) in BF16_SHAPES:
        ref = _bench_bf16(M, N, K, backend=HIPB)
        hk = _bench_bf16(M, N, K, backend=HK)
        ratio = (hk / ref) if ref > 0 else 0.0
        rows.append((f"BF16fwd_{M}x{N}x{K}", hk, ref, ratio, "hipblaslt", "bf16_fwd"))

    # ── (2) Dense BF16 backward (vs HIPBLASLT) ──
    for (M, N, K) in BF16_SHAPES:
        ref = _bench_bf16_bwd(M, N, K, backend=HIPB)
        hk = _bench_bf16_bwd(M, N, K, backend=HK)
        ratio = (hk / ref) if ref > 0 else 0.0
        rows.append((f"BF16bwd_{M}x{N}x{K}", hk, ref, ratio, "hipblaslt", "bf16_bwd"))

    # ── (3) Dense FP8 tensorwise forward (vs HIPBLASLT) ──
    for (M, N, K) in FP8_SHAPES:
        ref = _bench_fp8(M, N, K, backend=HIPB)
        hk = _bench_fp8(M, N, K, backend=HK)
        ratio = (hk / ref) if ref > 0 else 0.0
        rows.append((f"FP8fwd_{M}x{N}x{K}", hk, ref, ratio, "hipblaslt", "fp8_fwd"))

    # ── (4) Dense FP8 tensorwise backward (vs TRITON) ──
    for (M, N, K) in FP8_SHAPES:
        ref = _bench_fp8_bwd(M, N, K, backend=TRT)
        hk = _bench_fp8_bwd(M, N, K, backend=HK)
        ratio = (hk / ref) if ref > 0 else 0.0
        rows.append((f"FP8bwd_{M}x{N}x{K}", hk, ref, ratio, "triton", "fp8_bwd"))

    # ── (5) Grouped BF16 (vs TRITON) ──
    for (name, B, M, N, K) in GROUPED_BF16_SHAPES:
        ref = _bench_grouped_bf16(B, M, N, K, backend=TRT)
        hk = _bench_grouped_bf16(B, M, N, K, backend=HK)
        ratio = (hk / ref) if ref > 0 else 0.0
        rows.append((f"grpBF16_{name}", hk, ref, ratio, "triton", "grp_bf16"))

    wall = time.monotonic() - t0

    # Print per-shape table to stderr so the auto_optimize log captures it
    # without polluting stdout (which carries the score).
    print(f"\n[metric_hk_ratio] Shape suite ({len(rows)} cases, {wall:.1f}s wall):", file=sys.stderr)
    print(
        f"  {'name':40s}  {'hk_tflops':>10s}  {'ref_tflops':>10s}  {'vs':>9s}  {'ratio':>6s}",
        file=sys.stderr,
    )
    for name, hk, ref, r, ref_name, _section in rows:
        if r == 0:
            flag = "  *reject"
        elif name.startswith("FP8bwd_") or name.startswith("grpBF16_"):
            flag = "" if r >= 1.20 else ("  <120%")
        else:
            flag = "" if r >= 0.97 else ("  <97%")
        print(
            f"  {name:40s}  {hk:>10.1f}  {ref:>10.1f}  {ref_name:>9s}  {r:>6.3f}{flag}",
            file=sys.stderr,
        )

    # Per-section geomean breakdown.
    def _section_geomean(section: str) -> tuple[float, int]:
        sub = [max(r, 0.01) for _n, _, _, r, _ref, s in rows if s == section]
        if not sub:
            return (float("nan"), 0)
        return (math.exp(sum(math.log(r) for r in sub) / len(sub)), len(sub))

    g_bf16_fwd, n_bf16_fwd = _section_geomean("bf16_fwd")
    g_bf16_bwd, n_bf16_bwd = _section_geomean("bf16_bwd")
    g_fp8_fwd, n_fp8_fwd = _section_geomean("fp8_fwd")
    g_fp8_bwd, n_fp8_bwd = _section_geomean("fp8_bwd")
    g_grp_bf16, n_grp_bf16 = _section_geomean("grp_bf16")

    for label, g, n in [
        ("BF16_fwd  vs hipblaslt", g_bf16_fwd, n_bf16_fwd),
        ("BF16_bwd  vs hipblaslt", g_bf16_bwd, n_bf16_bwd),
        ("FP8_fwd   vs hipblaslt", g_fp8_fwd, n_fp8_fwd),
        ("FP8_bwd   vs triton   ", g_fp8_bwd, n_fp8_bwd),
        ("grp_BF16  vs triton   ", g_grp_bf16, n_grp_bf16),
    ]:
        if n:
            print(
                f"[metric_hk_ratio]   {label} geomean={g:.4f} (n={n})",
                file=sys.stderr,
            )

    # Five-goal acceptance per project policy:
    #   (1) BF16_fwd  vs hipblaslt >= 0.97
    #   (2) BF16_bwd  vs hipblaslt >= 0.97
    #   (3) FP8_fwd   vs hipblaslt >= 0.97
    #   (4) FP8_bwd   vs triton    >= 1.20
    #   (5) grp_BF16  vs triton    >= 1.20
    print("[metric_hk_ratio] Goals:", file=sys.stderr)
    goals = [
        ("(1) BF16_fwd  vs hipblaslt >= 0.97", g_bf16_fwd, 0.97, n_bf16_fwd),
        ("(2) BF16_bwd  vs hipblaslt >= 0.97", g_bf16_bwd, 0.97, n_bf16_bwd),
        ("(3) FP8_fwd   vs hipblaslt >= 0.97", g_fp8_fwd, 0.97, n_fp8_fwd),
        ("(4) FP8_bwd   vs triton    >= 1.20", g_fp8_bwd, 1.20, n_fp8_bwd),
        ("(5) grp_BF16  vs triton    >= 1.20", g_grp_bf16, 1.20, n_grp_bf16),
    ]
    n_pass = 0
    for label, g, target, n in goals:
        passed = (g >= target) if n else False
        if passed:
            n_pass += 1
        print(
            f"[metric_hk_ratio]   {label}  : {g:.4f}  "
            f"{'PASS' if passed else 'FAIL'}",
            file=sys.stderr,
        )

    # Score is the geomean over all 5 section-geomeans, scaled to int
    # (one per goal so each section weighs equally regardless of n).
    # This is just an auto_optimize ranking signal — acceptance is the
    # 5-goal PASS/FAIL above, not score.
    section_geomeans = [g for _, g, _, n in goals if n]
    if section_geomeans:
        section_geo = math.exp(
            sum(math.log(max(g, 0.01)) for g in section_geomeans) / len(section_geomeans)
        )
    else:
        section_geo = 0.0
    score = int(round(section_geo * 1000))

    n_reject = sum(1 for _n, _, _, r, _ref, _s in rows if r == 0)
    n_below = sum(
        1 for n, _, _, r, _ref, _s in rows
        if 0 < r < (1.20 if (n.startswith("FP8bwd_") or n.startswith("grpBF16_")) else 0.97)
    )

    print(
        f"[metric_hk_ratio] section_geomean={section_geo:.4f}  "
        f"reject={n_reject}/{len(rows)}  below_target={n_below}/{len(rows)}  "
        f"goals={n_pass}/5  score={score}",
        file=sys.stderr,
    )

    print(score)  # stdout: only the integer, parsed by auto_optimize.py
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
