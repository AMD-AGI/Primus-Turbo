#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Wall-time metric for HipKittens FP8 grouped GEMM with Path A fused-act
(activation BF16->FP8 cvt fused into forward, dA, AND dB var-K kernels).

Single section (24 cases): full **forward + backward** wall time of one
grouped FP8 GEMM call. HK runs with ``Float8QuantConfig.fuse_act_quant=
True`` (Phase 0 fallback today, fully fused once agent ships the kernels);
Triton runs with ``fuse_act_quant=False`` (Triton can't fuse).

Why time fwd + bwd together (not fwd-only):
  - Backward is ~2x forward FLOPs (dA + dB) and pays ``quantize_fp8(grad_out)``
    which is the same HBM traffic as ``quantize_fp8(a)`` in fwd. A fwd-only
    metric misses 2/3 of the optimization surface.
  - Path A's full benefit (no FP8 staging buffer, no max_abs 2nd pass)
    needs BOTH fwd-fuse AND bwd-fuse to land — gating the metric on fwd
    only would let the agent stop at ~3% wall saving instead of pushing
    through the harder bwd kernels for the full ~10%.
  - VRAM win (Path A's unique advantage over NVIDIA TE): eliminating the
    M*K FP8 ``a_fp8`` staging buffer + the M*N FP8 ``grad_out_fp8`` staging
    buffer over the entire fwd→bwd window.

Why a separate metric file vs ``_metric_grouped_only.py``:
  - ``_metric_grouped_only.py`` measures kernel-only TFLOPS under the rule
    "BF16->FP8 quantize is a tax both backends pay equally" — that rule
    fails the moment HK fuses. Wall ratio (this metric) surfaces the
    asymmetric saving Path A gives us.
  - The "no quantize fuse" hard rule from the prior task is EXPLICITLY
    suspended here. Fusion IS the optimization.

Comparison per shape (FLOPs = 6 * M_total * N * K = 2 fwd + 2 dA + 2 dB):
  HK    : ``turbo.ops.grouped_gemm_fp8(a, b, ..., fuse_act_quant=True,
          backend=HIPKITTEN)`` followed by ``out.backward(grad_out)``.
  TRT   : same op with ``fuse_act_quant=False``, backend pinned to TRITON.
  ratio : hk_tflops / trt_tflops  (higher = HK faster on the full step).

Score:
    score = int(min(geomean(ratio_i) / TARGET, 1.0) * 1000)
with TARGET configurable via ``METRIC_FUSED_WALL_TARGET`` env (default
1.25; analytical: kernel-only ratio plateau ~1.10, fusion saves ~10% wall
on top → ~1.21 ratio. Set 1.20 for "everyone wins fast", 1.30 for
"only Phase 3 fully landed wins").

Score landscape (24-shape geomean, target=1.25):
    Phase 0 (all 3 helpers raise NotImplementedError, fwd+bwd fully
            un-fused, bit-identical to FP8GroupedGemmTensorFunc) : ~870-900
    Phase 1 (forward fused, dA + dB un-fused)                    : ~920-940
    Phase 2 (+ dA fused, dB still un-fused)                      : ~960-980
    Phase 3 (+ dB var-K fused, no FP8 staging anywhere)          : 1000

Correctness gate: every shape runs fwd+bwd vs torch-native ref. SNR > 25
dB on fwd output, dA, dB. FAIL clips that shape's ratio to 0.0 (geomean
uses 0.01 floor) — same penalty as a runtime reject. Mirrors
``_metric_grouped_only.py``.

Wall: ~60-90s on idle MI355X (24 FP8 grouped × {correctness check + 2
backend fwd+bwd wall timings}; ~2x slower than _metric_grouped_only
because we time bwd too).

Score is consumed by ``auto_optimize.py`` via ``--metric``; emit integer
to stdout, all diagnostics to stderr.
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Optional

# Hint Primus where HipKittens lives so the BF16 / FP8 .so modules are auto-discovered.
os.environ.setdefault("PRIMUS_TURBO_HIPKITTEN_PATH", "/workspace/code/HipKittens")

# Make scripts/ importable so we can reuse helpers from _metric_hk_ratio.py.
_SCRIPTS_DIR = os.path.abspath(os.path.dirname(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

# Importing _metric_hk_ratio runs its module-level GPU-pinning side-effect
# (sets HIP_VISIBLE_DEVICES if unset) AND imports primus_turbo + benchmark
# config. Both are exactly what we want — we share the same GPU-pinning
# convention and the canonical MoE shape suite.
import _metric_hk_ratio as hk_ratio  # noqa: E402

import torch  # noqa: E402

from primus_turbo.pytorch.core.backend import BackendType, PrecisionType  # noqa: E402
from primus_turbo.pytorch.core.low_precision import (  # noqa: E402
    Float8QuantConfig,
    Format,
    ScalingGranularity,
)
import primus_turbo.pytorch as turbo  # noqa: E402

from benchmark.ops.config import (  # noqa: E402
    compute_snr,
    grouped_gemm_ref,
)


# Matches benchmark/ops/bench_grouped_gemm_turbo.py for E4M3 tensorwise.
_FP8_SNR_THRESHOLD_DB = 25.0

# Wall-ratio target. Default 1.35 — gives Path A's full ~12% wall saving
# (Phase 0 fwd+bwd baseline geomean ~1.246 → target 1.35 means score
# 1.246/1.35 = 0.923 → ~923 baseline, ~80 points of headroom for Phase
# 1+2+3 to climb. Override via env when re-tuning the score landscape).
_TARGET_RATIO = float(os.environ.get("METRIC_FUSED_WALL_TARGET", "1.35"))


def _check_fused_grouped_fp8_correctness(B: int, M: int, N: int, K: int) -> tuple[bool, str]:
    """Run HIPKITTEN FP8 grouped fwd+bwd with fuse_act_quant=True, SNR vs ref.

    The reference is the same un-fused torch-native ``grouped_gemm_ref``
    that ``_metric_grouped_only.py`` uses — fusion changes the *path*
    inside the kernel (BF16->FP8 cvt happens inside load_a_tile vs in a
    standalone quantize op) but the *semantics* must match within the
    SNR > 25 dB E4M3 noise floor on fwd output AND dA AND dB.

    Returns (ok, fail_reason). On PASS, fail_reason is "". On FAIL, a
    short code: "fwd-exception" / "fwd-nan" / "fwd-snr<XX.X" /
    "bwd-exception" / "dA-snr<XX.X" / "dB-snr<XX.X".
    """
    torch.manual_seed(0)
    cfg = Float8QuantConfig(
        format=Format.E4M3,
        granularity=ScalingGranularity.TENSORWISE,
        fuse_act_quant=True,
    )
    a = torch.randn((B * M, K), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    b = torch.randn((B, N, K), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    group_lens = torch.full((B,), M, dtype=torch.int64, device="cuda")

    with hk_ratio.force_grouped_gemm_backend(BackendType.HIPKITTEN, PrecisionType.FP8):
        try:
            out = turbo.ops.grouped_gemm_fp8(a, b, group_lens, trans_b=True, config=cfg)
        except Exception as e:
            return False, f"fwd-exception:{type(e).__name__}"
        if torch.isnan(out).any() or torch.isinf(out).any():
            return False, "fwd-nan"
        grad_out = torch.randn_like(out)
        try:
            out.backward(grad_out, retain_graph=True)
        except Exception as e:
            return False, f"bwd-exception:{type(e).__name__}"

    a_ref = a.detach().clone().requires_grad_()
    b_ref = b.detach().clone().requires_grad_()
    out_ref = grouped_gemm_ref(a_ref, b_ref, group_lens, trans_b=True)
    out_ref.backward(grad_out)

    out_snr = compute_snr(out_ref.detach(), out.detach())
    if out_snr <= _FP8_SNR_THRESHOLD_DB:
        return False, f"fwd-snr<{out_snr:.1f}"
    if a.grad is None:
        return False, "dA-none"
    da_snr = compute_snr(a_ref.grad, a.grad)
    if da_snr <= _FP8_SNR_THRESHOLD_DB:
        return False, f"dA-snr<{da_snr:.1f}"
    if b.grad is None:
        return False, "dB-none"
    db_snr = compute_snr(b_ref.grad, b.grad)
    if db_snr <= _FP8_SNR_THRESHOLD_DB:
        return False, f"dB-snr<{db_snr:.1f}"
    return True, ""


def _bench_grouped_fp8_fused_wall(
    B: int, M: int, N: int, K: int, backend: Optional[BackendType], fuse_act_quant: bool
) -> float:
    """Wall-time TFLOPS of ``turbo.ops.grouped_gemm_fp8`` (fwd + bwd) with
    the given backend + fuse_act_quant flag.

    Total FLOPs (numerator) = 6 * (B*M) * N * K = 2*fwd + 2*dA + 2*dB.
    Times the FULL training-step grouped GEMM call: forward op
    (config -> quantize -> kernel -> output) **plus** ``out.backward(grad_out)``
    which dispatches dA + dB in HK or Triton according to the backend
    pin and the autograd Function.

    Why fwd + bwd: see module docstring. Path A's optimization surface
    is fwd + bwd; fwd-only metric misses 2/3 of the FLOPs and lets the
    agent stop short.

    Mirrors :func:`_metric_hk_ratio._bench_grouped_fp8` for the forward
    half; backward half mirrors the timing pattern from
    :func:`_metric_hk_ratio._bench_grouped_fp8_bwd`.
    """
    cfg = Float8QuantConfig(
        format=Format.E4M3,
        granularity=ScalingGranularity.TENSORWISE,
        fuse_act_quant=fuse_act_quant,
    )
    a = torch.randn((B * M, K), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    b = torch.randn((B, N, K), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    group_lens = torch.full((B,), M, dtype=torch.int64, device="cuda")

    # Pre-compute grad_out shape with one warm-up forward (outside timer)
    with hk_ratio.force_grouped_gemm_backend(backend, PrecisionType.FP8):
        try:
            out_warm = turbo.ops.grouped_gemm_fp8(a, b, group_lens, trans_b=True, config=cfg)
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
        # Zero accumulated grads each iter so the autograd graph has a
        # fresh root; .backward without retain_graph frees the graph each
        # iter so requires_grad tensors don't leak.
        a.grad = None
        b.grad = None
        out = turbo.ops.grouped_gemm_fp8(a, b, group_lens, trans_b=True, config=cfg)
        out.backward(grad_out)

    with hk_ratio.force_grouped_gemm_backend(backend, PrecisionType.FP8):
        try:
            ms = hk_ratio._time_op(fwd_bwd)
        except Exception:
            return 0.0
    # 6 * M * N * K FLOPs (fwd 2x + dA 2x + dB 2x).
    return 6.0 * (B * M) * N * K / (ms * 1e9)


def _run() -> int:
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        print("[metric_fused_wall] CUDA/ROCm not available", file=sys.stderr)
        print(0)
        return 1

    rows: list[tuple[str, float, float, float, bool, str]] = []
    t0 = time.monotonic()
    TRT = BackendType.TRITON
    HK = BackendType.HIPKITTEN

    grouped_shapes = hk_ratio.GROUPED_BF16_SHAPES

    print(
        f"[metric_fused_wall] suite: {len(grouped_shapes)} FP8 wall cases | "
        f"target ratio = {_TARGET_RATIO:.2f} | "
        f"HK uses fuse_act_quant=True; Triton uses standard un-fused path "
        f"(it cannot fuse).",
        file=sys.stderr,
    )

    for (name, B, M, N, K) in grouped_shapes:
        ok, reason = _check_fused_grouped_fp8_correctness(B, M, N, K)
        ref = _bench_grouped_fp8_fused_wall(B, M, N, K, backend=TRT, fuse_act_quant=False)
        hk = _bench_grouped_fp8_fused_wall(B, M, N, K, backend=HK, fuse_act_quant=True)
        if ok:
            ratio = (hk / ref) if ref > 0 else 0.0
        else:
            ratio = 0.0
        rows.append((f"fusedFP8_{name}", hk, ref, ratio, ok, reason))

    wall = time.monotonic() - t0

    print(
        f"\n[metric_fused_wall] Shape suite ({len(rows)} cases, {wall:.1f}s wall):",
        file=sys.stderr,
    )
    print(
        f"  {'name':50s}  {'hk_tflops':>10s}  {'trt_tflops':>10s}  {'ratio':>6s}  status",
        file=sys.stderr,
    )
    for name, hk, ref, r, ok, reason in rows:
        if not ok:
            flag = f"  *FAIL[{reason}]"
        elif r == 0:
            flag = "  *reject"
        else:
            flag = "" if r >= _TARGET_RATIO else f"  <{int(_TARGET_RATIO*100)}%"
        print(
            f"  {name:50s}  {hk:>10.1f}  {ref:>10.1f}  {r:>6.3f}{flag}",
            file=sys.stderr,
        )

    sub = [max(r, 0.01) for _n, _, _, r, _ok, _why in rows]
    if not sub:
        geomean = float("nan")
    else:
        geomean = math.exp(sum(math.log(r) for r in sub) / len(sub))

    n_pass = sum(1 for _, _, _, r, ok, _ in rows if ok and r >= _TARGET_RATIO)
    n_correct_fail = sum(1 for _, _, _, _, ok, _ in rows if not ok)
    n_reject = sum(1 for _, _, _, r, ok, _ in rows if ok and r == 0)
    n_below = sum(1 for _, _, _, r, ok, _ in rows if ok and 0 < r < _TARGET_RATIO)

    progress = min(geomean / _TARGET_RATIO, 1.0) if _TARGET_RATIO > 0 else 0.0
    progress = max(progress, 0.001)
    score = int(round(progress * 1000))

    print(
        f"\n[metric_fused_wall] Goals: HK_fused / TRT_baseline >= "
        f"{_TARGET_RATIO:.2f}  geomean={geomean:.4f}  "
        f"progress={progress:.3f}  {'PASS' if geomean >= _TARGET_RATIO else 'FAIL'}",
        file=sys.stderr,
    )
    print(
        f"[metric_fused_wall] correct_fail={n_correct_fail}/{len(rows)}  "
        f"reject={n_reject}/{len(rows)}  below_target={n_below}/{len(rows)}  "
        f"goals={n_pass}/{len(rows)}  score={score}",
        file=sys.stderr,
    )
    if n_correct_fail:
        bad = [
            f"{name}({why})"
            for name, _hk, _ref, _r, ok, why in rows
            if not ok
        ]
        print(
            f"[metric_fused_wall] correctness FAIL shapes: {', '.join(bad)}",
            file=sys.stderr,
        )

    print(score)  # stdout: integer only, parsed by auto_optimize.py
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
