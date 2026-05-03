#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Weighted wall-time metric for HipKittens BF16 grouped GEMM across the
**full 24-shape MoE suite** (DeepSeek-V3 + gpt_oss_20B + Qwen3-235B-A22B).

Why weighted, not raw geomean:
  * gpt_oss_20B (8 / 24 shapes) is currently HK / Triton ≈ 0.88 — the
    binding bottleneck. DSV3 + Qwen3 (16 / 24 shapes) are ≈ 1.12 — already
    +12 % vs Triton.
  * Equal-weight geomean across 24 dilutes a +20 % gpt_oss win to a +6 %
    full-suite move, removing the agent's signal during K-tail kernel
    surgery.
  * gpt_oss-only scoring lets the agent silently tank DSV3 / Qwen3 and
    still hit 1000.
  * Solution: score every shape, but **weight gpt_oss 3x** (per-shape
    weight 3, vs 1 for DSV3 / Qwen3). gpt_oss owns ~60 % of the score
    headroom, but DSV3 / Qwen3 still own the last ~30-40 % — agent can't
    plateau without lifting all three families.

Per-shape metric:
  HK     : ``turbo.ops.grouped_gemm(a, b, ..., backend=HIPKITTEN)``
           fwd + ``out.backward(grad_out)``.
  TRT    : same op pinned to BackendType.TRITON.
  ratio  : hk_tflops / trt_tflops (higher = HK faster on full step).
  flops  : 6 * (B*M) * N * K = 2 fwd + 2 dA + 2 dB.

Per-shape progress (capped — keeps each shape from dominating):
  progress_i = min(ratio_i / TARGET_RATIO, 1.0)
where TARGET_RATIO defaults to 1.25 (overridable via
``METRIC_BF16_WEIGHTED_TARGET`` env). A shape with ratio >= 1.25 cannot
contribute > 1.0 — no over-shooting one shape to mask another.

Per-shape weight (overridable via env, default fixed):
  gpt_oss_20B    : 3
  DeepSeek-V3    : 1
  Qwen3-235B-A22B: 1
Total weight: 8*3 + 8*1 + 8*1 = 40.

Score:
  score = int(weighted_avg(progress_i) * 1000)
score = 1000 ⇔ every shape has ratio >= 1.25 (no exceptions). Reaching
1000 is genuinely "全部都要优化".

Suite execution:
  * Iteration order = ``_metric_hk_ratio.GROUPED_BF16_SHAPES`` =
    DSV3 (8) → gpt_oss (8) → Qwen3 (8). DSV3 first incidentally warms
    the HK runtime around the K%128==0 path before the K-tail
    (K%128==64) gpt_oss launches — workaround for the HK BF16 K-tail
    cold-start sync-fault bug. **Don't reorder.**
  * Correctness on every shape: HK fwd+bwd cross-checked vs Triton on
    a downsized version of the shape (B' = min(B, 4), M' = min(M, 256));
    bfloat16 ``check_allclose`` must agree on out, dA, dB.
    Downsizing avoids the fault that triggers when full-shape
    correctness checks compete with bench tensors for VRAM. Full-shape
    correctness is enforced separately by DoD smoke.

Wall: ~30-40 s on idle MI355X (24 shapes × {downsized correctness +
2-backend wall timings}).

Score is consumed by ``auto_optimize.py`` via ``--metric``; emit integer
to stdout, all diagnostics to stderr.
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Optional

os.environ.setdefault("PRIMUS_TURBO_HIPKITTEN_PATH", "/workspace/code/HipKittens")

_SCRIPTS_DIR = os.path.abspath(os.path.dirname(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import _metric_hk_ratio as hk_ratio  # noqa: E402

import torch  # noqa: E402

from primus_turbo.pytorch.core.backend import BackendType, PrecisionType  # noqa: E402
import primus_turbo.pytorch as turbo  # noqa: E402

from benchmark.ops.config import check_allclose  # noqa: E402


_TARGET_RATIO = float(os.environ.get("METRIC_BF16_WEIGHTED_TARGET", "1.25"))

# Per-(model_family) weights. Higher weight = larger share of the
# weighted average. gpt_oss 3 vs DSV3/Qwen3 1 means gpt_oss owns
# 8*3/40 = 60 % of the score; DSV3 + Qwen3 each own 8*1/40 = 20 %.
# This split is hand-tuned so that:
#   * Agent can't ignore DSV3/Qwen3 (40 % of headroom is theirs).
#   * Agent can't game by static-DSV3/Qwen3 + all-gpt_oss-to-1.25
#     and still hit 1000 (would cap at 957 in that scenario).
# Override via env ``METRIC_BF16_WEIGHT_<FAMILY>``.
_FAMILY_WEIGHTS: dict[str, float] = {
    "gpt_oss_20B": float(os.environ.get("METRIC_BF16_WEIGHT_GPT_OSS", "3.0")),
    "DeepSeek-V3": float(os.environ.get("METRIC_BF16_WEIGHT_DSV3", "1.0")),
    "Qwen3-235B-A22B": float(os.environ.get("METRIC_BF16_WEIGHT_QWEN3", "1.0")),
}


def _full_suite() -> list[tuple[str, int, int, int, int]]:
    """All 24 MoE BF16 shapes (DSV3 + gpt_oss_20B + Qwen3-235B-A22B)."""
    return list(hk_ratio.GROUPED_BF16_SHAPES)


def _shape_weight(name: str) -> float:
    """Lookup the per-shape weight by matching the leading model-family
    prefix in the shape name. Falls back to 0.0 (= shape excluded from
    score) so an unknown family doesn't silently inflate the average.
    """
    for family, weight in _FAMILY_WEIGHTS.items():
        if name.startswith(family):
            return weight
    return 0.0


def _check_hk_vs_triton_small(B: int, M: int, N: int, K: int) -> tuple[bool, str]:
    """Cheap correctness sanity check on a *downsized* version of the
    shape (M' = min(M, 256), B' = min(B, 4)). Runs HipKittens BF16
    grouped fwd+bwd and Triton on the same inputs and demands per-element
    bfloat16 ``check_allclose`` agreement on out, dA, dB. Sized small to
    avoid the GPU memory pressure that triggers HK fault on big gpt_oss
    (B=32, M=4096) — the *kernel correctness* is shape-independent at
    this granularity (same K-tail / RCR / CRR dispatch). Full-suite
    correctness is enforced separately by DoD smoke tests.
    """
    torch.manual_seed(0)
    Bs = min(B, 4)
    Ms = min(M, 256)
    a_data = torch.randn((Bs * Ms, K), dtype=torch.bfloat16, device="cuda")
    b_data = torch.randn((Bs, N, K), dtype=torch.bfloat16, device="cuda")
    group_lens = torch.full((Bs,), Ms, dtype=torch.int64, device="cuda")
    grad_data = torch.randn((Bs * Ms, N), dtype=torch.bfloat16, device="cuda")

    a_hk = a_data.clone().requires_grad_()
    b_hk = b_data.clone().requires_grad_()
    with hk_ratio.force_grouped_gemm_backend(
        BackendType.HIPKITTEN, PrecisionType.BF16_FP16_FP32
    ):
        try:
            out_hk = turbo.ops.grouped_gemm(a_hk, b_hk, group_lens, trans_b=True)
        except Exception as e:
            return False, f"hk-fwd-exception:{type(e).__name__}"
        if torch.isnan(out_hk).any() or torch.isinf(out_hk).any():
            return False, "hk-fwd-nan"
        try:
            out_hk.backward(grad_data)
        except Exception as e:
            return False, f"hk-bwd-exception:{type(e).__name__}"

    a_tr = a_data.clone().requires_grad_()
    b_tr = b_data.clone().requires_grad_()
    with hk_ratio.force_grouped_gemm_backend(
        BackendType.TRITON, PrecisionType.BF16_FP16_FP32
    ):
        try:
            out_tr = turbo.ops.grouped_gemm(a_tr, b_tr, group_lens, trans_b=True)
            out_tr.backward(grad_data)
        except Exception as e:
            return False, f"triton-ref-exception:{type(e).__name__}"

    if not check_allclose(out_hk.detach(), out_tr.detach(), torch.bfloat16):
        return False, "fwd-allclose"
    if a_hk.grad is None or a_tr.grad is None:
        return False, "dA-none"
    if not check_allclose(a_hk.grad, a_tr.grad, torch.bfloat16):
        return False, "dA-allclose"
    if b_hk.grad is None or b_tr.grad is None:
        return False, "dB-none"
    if not check_allclose(b_hk.grad, b_tr.grad, torch.bfloat16):
        return False, "dB-allclose"
    return True, ""


def _bench_grouped_bf16_wall(
    B: int, M: int, N: int, K: int, backend: Optional[BackendType]
) -> float:
    """Wall-time TFLOPS of ``turbo.ops.grouped_gemm`` (fwd + bwd) for the
    given backend.

    Total FLOPs = 6 * (B*M) * N * K = 2*fwd + 2*dA + 2*dB. Returns 0.0
    if the call rejects / NaN / Inf.
    """
    a = torch.randn((B * M, K), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    b = torch.randn((B, N, K), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    group_lens = torch.full((B,), M, dtype=torch.int64, device="cuda")

    with hk_ratio.force_grouped_gemm_backend(backend, PrecisionType.BF16_FP16_FP32):
        try:
            out_warm = turbo.ops.grouped_gemm(a, b, group_lens, trans_b=True)
        except Exception:
            return 0.0
        if torch.isnan(out_warm).any() or torch.isinf(out_warm).any():
            return 0.0
        grad_out = torch.randn_like(out_warm)
        a.grad = None
        b.grad = None
        try:
            out_warm.backward(grad_out)
        except Exception:
            return 0.0

    def fwd_bwd():
        a.grad = None
        b.grad = None
        out = turbo.ops.grouped_gemm(a, b, group_lens, trans_b=True)
        out.backward(grad_out)

    with hk_ratio.force_grouped_gemm_backend(backend, PrecisionType.BF16_FP16_FP32):
        try:
            ms = hk_ratio._time_op(fwd_bwd)
        except Exception:
            return 0.0
    return 6.0 * (B * M) * N * K / (ms * 1e9)


def _run() -> int:
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        print("[metric_bf16_weighted] CUDA/ROCm not available", file=sys.stderr)
        print(0)
        return 1

    rows: list[
        tuple[str, float, float, float, bool, float, str]
    ] = []  # (name, hk_tflops, trt_tflops, ratio, ok, weight, fail_reason)
    t0 = time.monotonic()
    TRT = BackendType.TRITON
    HK = BackendType.HIPKITTEN

    suite = _full_suite()
    if not suite:
        print(
            "[metric_bf16_weighted] empty MoE suite (config mismatch?)",
            file=sys.stderr,
        )
        print(0)
        return 1

    weight_summary = ", ".join(
        f"{family.split('-')[0]}={int(w)}x" if w == int(w) else f"{family.split('-')[0]}={w:g}x"
        for family, w in _FAMILY_WEIGHTS.items()
    )
    print(
        f"[metric_bf16_weighted] running {len(suite)} BF16 wall cases | "
        f"target ratio = {_TARGET_RATIO:.2f} per shape | weights: {weight_summary} | "
        f"score = int(weighted_avg(min(ratio_i / target, 1.0)) * 1000)",
        file=sys.stderr,
    )

    for (name, B, M, N, K) in suite:
        weight = _shape_weight(name)
        print(
            f"[metric_bf16_weighted] running {name} B={B} M={M} N={N} K={K} "
            f"(weight={weight:g}) ...",
            file=sys.stderr,
            flush=True,
        )
        if weight > 0.0:
            ok, reason = _check_hk_vs_triton_small(B, M, N, K)
        else:
            ok, reason = True, ""
        ref = _bench_grouped_bf16_wall(B, M, N, K, backend=TRT)
        hk = _bench_grouped_bf16_wall(B, M, N, K, backend=HK)
        if ok:
            ratio = (hk / ref) if ref > 0 else 0.0
        else:
            ratio = 0.0
        rows.append((f"BF16_{name}", hk, ref, ratio, ok, weight, reason))
        progress = min(ratio / _TARGET_RATIO, 1.0) if _TARGET_RATIO > 0 else 0.0
        print(
            f"[metric_bf16_weighted]   {name}  ok={ok}  hk={hk:.1f}  "
            f"trt={ref:.1f}  ratio={ratio:.3f}  progress={progress:.3f}",
            file=sys.stderr,
            flush=True,
        )

    wall = time.monotonic() - t0

    print(
        f"\n[metric_bf16_weighted] Shape suite ({len(rows)} cases, "
        f"{wall:.1f}s wall):",
        file=sys.stderr,
    )
    print(
        f"  {'name':55s}  {'hk_tflops':>10s}  {'trt_tflops':>10s}  "
        f"{'ratio':>6s}  {'progress':>8s}  {'weight':>6s}  status",
        file=sys.stderr,
    )

    family_stats: dict[str, list[float]] = {family: [] for family in _FAMILY_WEIGHTS}

    for name, hk, ref, r, ok, weight, reason in rows:
        progress = min(r / _TARGET_RATIO, 1.0) if _TARGET_RATIO > 0 else 0.0
        if not ok:
            flag = f"  *FAIL[{reason}]"
        elif r == 0:
            flag = "  *reject"
        elif r >= _TARGET_RATIO:
            flag = "  PASS"
        else:
            flag = f"  <{int(_TARGET_RATIO*100)}%"
        print(
            f"  {name:55s}  {hk:>10.1f}  {ref:>10.1f}  {r:>6.3f}  "
            f"{progress:>8.3f}  {weight:>6.2f}{flag}",
            file=sys.stderr,
        )
        family = next(
            (f for f in _FAMILY_WEIGHTS if name.removeprefix("BF16_").startswith(f)),
            None,
        )
        if family is not None and ok and r > 0:
            family_stats[family].append(r)

    weighted_progress_sum = 0.0
    weight_total = 0.0
    for _name, _hk, _ref, r, ok, weight, _why in rows:
        if weight <= 0.0:
            continue
        if not ok or r == 0:
            progress = 0.0
        else:
            progress = min(r / _TARGET_RATIO, 1.0)
        weighted_progress_sum += weight * progress
        weight_total += weight

    weighted_progress = (
        weighted_progress_sum / weight_total if weight_total > 0 else 0.0
    )
    score = int(round(max(weighted_progress, 0.0) * 1000))

    n_pass = sum(
        1 for _n, _h, _r, r, ok, w, _why in rows
        if w > 0 and ok and r >= _TARGET_RATIO
    )
    n_fail = sum(1 for _n, _h, _r, _r2, ok, w, _why in rows if w > 0 and not ok)
    n_reject = sum(
        1 for _n, _h, _r, r, ok, w, _why in rows if w > 0 and ok and r == 0
    )
    n_below = sum(
        1 for _n, _h, _r, r, ok, w, _why in rows
        if w > 0 and ok and 0 < r < _TARGET_RATIO
    )
    n_scored = sum(1 for _n, _h, _r, _r2, _ok, w, _why in rows if w > 0)

    print(
        f"\n[metric_bf16_weighted] Per-family geomean (un-weighted):",
        file=sys.stderr,
    )
    for family, vals in family_stats.items():
        if vals:
            geo = math.exp(sum(math.log(max(r, 0.01)) for r in vals) / len(vals))
            print(
                f"  {family:25s}  n={len(vals):2d}  geomean_ratio={geo:.4f}  "
                f"(target {_TARGET_RATIO:.2f}, weight {_FAMILY_WEIGHTS[family]:g}x)",
                file=sys.stderr,
            )
        else:
            print(
                f"  {family:25s}  n=0   (no PASS rows — all failed correctness?)",
                file=sys.stderr,
            )

    print(
        f"\n[metric_bf16_weighted] Goals: per-shape ratio >= "
        f"{_TARGET_RATIO:.2f}  weighted_progress={weighted_progress:.4f}  "
        f"score={score}/1000  "
        f"{'PASS-ALL' if n_pass == n_scored else 'PARTIAL'}",
        file=sys.stderr,
    )
    print(
        f"[metric_bf16_weighted] correct_fail={n_fail}/{n_scored}  "
        f"reject={n_reject}/{n_scored}  "
        f"below_target={n_below}/{n_scored}  "
        f"goals={n_pass}/{n_scored}",
        file=sys.stderr,
    )
    if n_fail:
        bad = [
            f"{name}({why})"
            for name, _hk, _ref, _r, ok, w, why in rows
            if w > 0 and not ok
        ]
        print(
            f"[metric_bf16_weighted] correctness FAIL shapes: {', '.join(bad)}",
            file=sys.stderr,
        )

    print(score)
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
