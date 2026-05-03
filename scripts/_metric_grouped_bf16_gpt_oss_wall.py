#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Wall-time metric for HipKittens BF16 grouped GEMM on the **gpt_oss_20B
sub-suite** (8 shapes, K=2880 K%128=64).

Why a focused metric (not the full 24-shape MoE suite):

  * Empirically (HEAD 2026-05-03), HK BF16 grouped GEMM is +10-13 % vs
    Triton on DeepSeek-V3 + Qwen3-235B-A22B (16/24 shapes), but **-15 to
    -25 % SLOWER on every gpt_oss_20B shape (8/24)** because gpt_oss
    has K=2880 / K%128=64 which routes through the multi-launch RRR
    K-tail kernel path.

  * A 24-shape geomean dilutes the gpt_oss signal; a +20 % win on
    gpt_oss only moves the full-suite geomean by ~+6 %. To attack the
    hard kernel surgery on the K-tail path, the metric must surface
    gpt_oss progress directly.

  * gpt_oss BF16 baseline: geomean ratio ≈ 0.866 (Triton +13.4 % vs HK).
    Target 1.25 (matching FP8 wall ratio) requires +44 % swing — pure
    HK kernel surgery, no Path-A-style cvt fusion (no quantize step
    in BF16).

Per-shape metric:
  HK    : ``turbo.ops.grouped_gemm(a, b, ..., backend=HIPKITTEN)``
          fwd + ``out.backward(grad_out)``.
  TRT   : same op pinned to BackendType.TRITON.
  ratio : hk_tflops / trt_tflops (higher = HK faster on full step).
  flops : 6 * (B*M) * N * K = 2 fwd + 2 dA + 2 dB.

Score:
    score = int(min(geomean(ratio_i) / TARGET, 1.0) * 1000)
TARGET=1.25 by default (override via METRIC_BF16_GPT_OSS_TARGET env).

Suite execution: runs the full 24-shape MoE suite (DSV3 + gpt_oss +
Qwen3) but **only** scores the 8 gpt_oss_20B rows. The DSV3 / Qwen3
rows are run BEFORE gpt_oss to warm the HipKittens runtime — without
them, the very first BF16 K-tail (K%128≠0) launch in a fresh process
trips a known cold-start memory-fault bug in HipKittens BF16 grouped
that corrupts later ``cudaDeviceSynchronize``. Their HK/Triton ratios
are still printed (status = ``warmup-only``) so DSV3 / Qwen3 silent
regressions surface in stderr — they just don't move the score.

Correctness gate: every gpt_oss shape runs HK fwd+bwd cross-checked
against Triton on a *downsized* version of the shape (B' = min(B, 4),
M' = min(M, 256)); ``check_allclose`` for bfloat16 must agree on out,
dA, dB. The downsize is required because the full shape's correctness
check competes with the bench tensors for VRAM and triggers HK fault
on big gpt_oss (B=32 M=4096) after a few shapes. Full-shape correctness
is enforced separately by DoD smoke tests.

Wall: ~16 s on idle MI355X (24 BF16 grouped wall timings × 2 backends +
8 small correctness checks).

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
import primus_turbo.pytorch as turbo  # noqa: E402

from benchmark.ops.config import check_allclose  # noqa: E402


# Wall-ratio target. Default 1.25 — matches the FP8 wall metric ceiling
# (HK FP8 currently at ~1.25 vs Triton). Reaching this on BF16 means
# closing the K-tail kernel gap on gpt_oss + extracting the same
# kernel-quality margin HK has on DSV3/Qwen3 BF16.
_TARGET_RATIO = float(os.environ.get("METRIC_BF16_GPT_OSS_TARGET", "1.25"))


def _full_suite() -> list[tuple[str, int, int, int, int]]:
    """All 24 MoE BF16 shapes (DSV3 + gpt_oss_20B + Qwen3-235B-A22B)."""
    return list(hk_ratio.GROUPED_BF16_SHAPES)


def _is_scored(name: str) -> bool:
    """Score only the gpt_oss_20B sub-family (8/24). The other 16 shapes
    are still executed (DSV3 / Qwen3) — they keep the HipKittens runtime
    warm so K-tail (gpt_oss K=2880) launches don't trip the cold-start
    sync-fault — but only the 8 gpt_oss_20B ratios feed the geomean. This
    is critical because:

      * cold-start of HK BF16 K-tail kernels (K%128!=0) corrupts the
        next ``cudaDeviceSynchronize`` if the K-tail launch is the
        first op in the process. Running DSV3 (K%128==0) first
        prevents this.
      * scoring only gpt_oss surfaces the K-tail bottleneck directly;
        a 24-shape geomean would dilute a +20 % gpt_oss win to +6 %.
    """
    return name.startswith("gpt_oss_20B")


def _check_hk_vs_triton_small(B: int, M: int, N: int, K: int) -> tuple[bool, str]:
    """Cheap correctness sanity check on a *downsized* version of the
    shape (M' = min(M, 256), B' = min(B, 4)). Runs HipKittens BF16
    grouped fwd+bwd and Triton on the same inputs and demands per-element
    bfloat16 ``check_allclose`` agreement on out, dA, dB. Sized small to
    avoid the GPU memory pressure that triggers HK fault on big gpt_oss
    (B=32, M=4096) — the *kernel correctness* is shape-independent at
    this granularity (same K-tail dispatch). Full-suite correctness is
    enforced separately by DoD smoke tests.
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
        print("[metric_bf16_gpt_oss] CUDA/ROCm not available", file=sys.stderr)
        print(0)
        return 1

    rows: list[tuple[str, float, float, float, bool, bool, str]] = []
    t0 = time.monotonic()
    TRT = BackendType.TRITON
    HK = BackendType.HIPKITTEN

    suite = _full_suite()
    if not suite:
        print(
            "[metric_bf16_gpt_oss] empty MoE suite (config mismatch?)",
            file=sys.stderr,
        )
        print(0)
        return 1

    n_scored = sum(1 for (n, *_rest) in suite if _is_scored(n))
    print(
        f"[metric_bf16_gpt_oss] running {len(suite)} BF16 wall cases "
        f"(scoring on {n_scored} gpt_oss_20B shapes; DSV3 + Qwen3 run for "
        f"runtime stability only) | target ratio = {_TARGET_RATIO:.2f} | "
        f"HK fwd+bwd vs Triton fwd+bwd, no quantize involved.",
        file=sys.stderr,
    )

    for (name, B, M, N, K) in suite:
        scored = _is_scored(name)
        tag = "SCORED" if scored else "warmup"
        print(
            f"[metric_bf16_gpt_oss] [{tag}] running {name} B={B} M={M} N={N} K={K} ...",
            file=sys.stderr,
            flush=True,
        )
        # Skip correctness on warmup-only shapes — they only need to
        # stabilize the HK runtime; saves ~30 % of the wall.
        if scored:
            ok, reason = _check_hk_vs_triton_small(B, M, N, K)
        else:
            ok, reason = True, ""
        ref = _bench_grouped_bf16_wall(B, M, N, K, backend=TRT)
        hk = _bench_grouped_bf16_wall(B, M, N, K, backend=HK)
        if ok:
            ratio = (hk / ref) if ref > 0 else 0.0
        else:
            ratio = 0.0
        rows.append((f"BF16_{name}", hk, ref, ratio, ok, scored, reason))
        print(
            f"[metric_bf16_gpt_oss]   [{tag}] {name}  ok={ok}  hk={hk:.1f}  "
            f"trt={ref:.1f}  ratio={ratio:.3f}",
            file=sys.stderr,
            flush=True,
        )

    wall = time.monotonic() - t0

    print(
        f"\n[metric_bf16_gpt_oss] Shape suite ({len(rows)} cases, "
        f"{wall:.1f}s wall):",
        file=sys.stderr,
    )
    print(
        f"  {'name':55s}  {'hk_tflops':>10s}  {'trt_tflops':>10s}  "
        f"{'ratio':>6s}  scored  status",
        file=sys.stderr,
    )
    for name, hk, ref, r, ok, scored, reason in rows:
        if not ok:
            flag = f"  *FAIL[{reason}]"
        elif r == 0:
            flag = "  *reject"
        elif not scored:
            flag = "  warmup-only"
        else:
            flag = "" if r >= _TARGET_RATIO else f"  <{int(_TARGET_RATIO*100)}%"
        sm = "yes" if scored else "no"
        print(
            f"  {name:55s}  {hk:>10.1f}  {ref:>10.1f}  {r:>6.3f}  "
            f"{sm:>6s}{flag}",
            file=sys.stderr,
        )

    scored_rows = [
        (n, hk, ref, r, ok, why)
        for (n, hk, ref, r, ok, scored, why) in rows
        if scored
    ]
    if not scored_rows:
        geomean = float("nan")
    else:
        sub = [max(r, 0.01) for _n, _, _, r, _ok, _why in scored_rows]
        geomean = math.exp(sum(math.log(r) for r in sub) / len(sub))

    n_pass = sum(
        1 for _, _, _, r, ok, _ in scored_rows if ok and r >= _TARGET_RATIO
    )
    n_correct_fail = sum(1 for _, _, _, _, ok, _ in scored_rows if not ok)
    n_reject = sum(1 for _, _, _, r, ok, _ in scored_rows if ok and r == 0)
    n_below = sum(
        1 for _, _, _, r, ok, _ in scored_rows if ok and 0 < r < _TARGET_RATIO
    )

    progress = (
        min(geomean / _TARGET_RATIO, 1.0)
        if (_TARGET_RATIO > 0 and not math.isnan(geomean))
        else 0.0
    )
    progress = max(progress, 0.001)
    score = int(round(progress * 1000))

    print(
        f"\n[metric_bf16_gpt_oss] Goals (gpt_oss_20B only): HK / TRT >= "
        f"{_TARGET_RATIO:.2f}  geomean={geomean:.4f}  progress={progress:.3f}  "
        f"{'PASS' if geomean >= _TARGET_RATIO else 'FAIL'}",
        file=sys.stderr,
    )
    print(
        f"[metric_bf16_gpt_oss] correct_fail={n_correct_fail}/{len(scored_rows)}  "
        f"reject={n_reject}/{len(scored_rows)}  "
        f"below_target={n_below}/{len(scored_rows)}  "
        f"goals={n_pass}/{len(scored_rows)}  score={score}",
        file=sys.stderr,
    )
    if n_correct_fail:
        bad = [
            f"{name}({why})"
            for name, _hk, _ref, _r, ok, why in scored_rows
            if not ok
        ]
        print(
            f"[metric_bf16_gpt_oss] correctness FAIL shapes: {', '.join(bad)}",
            file=sys.stderr,
        )

    print(score)  # stdout: integer only, parsed by auto_optimize.py
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
