#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Grouped-only HipKittens TFLOPS ratio metric for auto_optimize.

Two independent sections (32 cases total), each compared against TRITON:

    (1) grp_BF16  vs TRITON  —  16 MoE shapes (DeepSeek-V3 + gpt_oss_20B)
    (2) grp_FP8   vs TRITON  —  same 16 MoE shapes, FP8 tensorwise

Acceptance: BOTH section geomeans >= 1.20 (printed via Goals: PASS/FAIL block).

**Correctness gate (mandatory)**: every HIPKITTEN row first runs a fwd+bwd
correctness check against the torch-native reference (``grouped_gemm_ref``):

  * BF16: ``torch.allclose(rtol=1e-2, atol=1e-2)`` on fwd output, dA, dB.
  * FP8 : ``compute_snr(ref, actual) > 25 dB`` on fwd output, dA, dB.

Any failure (fwd / dA / dB) clips that shape's ratio to 0.0 (geomean uses
0.01 floor) — same penalty as a runtime reject. Rationale: the metric
**must not reward a fast-but-wrong kernel**. A correctness regression that
ships fwd-OK but bwd-broken (e.g. K-misaligned RRR layout dx kernel) would
otherwise be invisible to the loop — agent has no incentive to fix it
because score keeps climbing on fwd-only TFLOPS.

Score (single integer printed on stdout, consumed by auto_optimize.py):

    score = int(weighted_geomean(min(g_i / 1.20, 1.0)) * 1000)

with both sections weighted equally (1, 1). Capped progress means
overshooting one section to 1.5x does NOT bank extra score — the agent
must spend cycles on the section still below target. Correctness FAIL
(or HIPKITTEN raise / NaN / Inf) clips that shape's ratio to 0.0.

When BOTH sections are at >= 1.20 vs Triton AND all shapes correct,
score = 1000. Anything below target ⇒ score < 1000 (this is the score
= ranking signal; the GOAL is the 2-line PASS/FAIL block in stderr —
both must read PASS).

**Focused scoring (env var ``METRIC_MODEL_FILTER``)**: when set to
``gpt_oss`` or ``deepseek``, only shapes whose name starts with that
model are counted in the geomean / Goals / score (call them
"in-focus"). All shapes are still benchmarked + correctness-checked;
non-focused shapes show in the per-shape table tagged ``[watch]`` and
their correctness FAILs surface in a separate ``watch`` line so the
user sees silent regressions on the un-focused model. Default
(unset / ``all``) keeps the original behavior — both models count.

Use the focus knob to spend optimization rounds on the model whose
gap-to-target dominates: e.g. ``METRIC_MODEL_FILTER=gpt_oss`` makes
the score insensitive to DSV3 perf wiggle while still flagging DSV3
correctness regressions, so the agent can attack gpt_oss-specific
bottlenecks (K%128/N%128 misalignment, B=4 grid under-utilization)
without distractor cycles on the well-tuned DSV3 path.

Wall: ~25-40s on idle MI355X (16 BF16 + 16 FP8 grouped × {correctness
check + 2 backend timings}; correctness adds ~10-15s overhead vs the
old fwd-only pass).

This script imports helper functions / shape suite from
_metric_hk_ratio.py to avoid duplicating the suite ground truth and the
GPU pinning / timing logic. _metric_hk_ratio.py remains the canonical
6-segment metric; this script is the focused 2-segment slice.
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

# benchmark/ops/config.py is on sys.path because importing _metric_hk_ratio
# above ran its module-level `_REPO_ROOT` insertion. Reuse the same correctness
# helpers and torch-native reference that bench_grouped_gemm_turbo.py uses, so
# this metric's PASS/FAIL semantics line up with the canonical bench script
# (rtol=atol=1e-2 for BF16, SNR > 25 dB for FP8 E4M3).
from benchmark.ops.config import (  # noqa: E402
    check_allclose,
    compute_snr,
    grouped_gemm_ref,
)


# 25 dB matches benchmark/ops/bench_grouped_gemm_turbo.py for E4M3 tensorwise.
_FP8_SNR_THRESHOLD_DB = 25.0


def _check_grouped_bf16_correctness(B: int, M: int, N: int, K: int) -> tuple[bool, str]:
    """Run HIPKITTEN BF16 grouped fwd+bwd, allclose vs torch-native ref.

    Returns (ok, fail_reason). fail_reason is empty on PASS, otherwise one of:
      "fwd-exception", "fwd-nan", "fwd", "bwd-exception", "dA-none", "dA",
      "dB-none", "dB". The first failure short-circuits.

    Mirrors bench_grouped_gemm_turbo.profile_grouped_gemm correctness path.
    """
    torch.manual_seed(0)
    x = torch.randn((B * M, K), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    w = torch.randn((B, N, K), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    group_lens = torch.full((B,), M, dtype=torch.int64, device="cuda")

    with hk_ratio.force_grouped_gemm_backend(BackendType.HIPKITTEN, PrecisionType.BF16_FP16_FP32):
        try:
            out = turbo.ops.grouped_gemm(x, w, group_lens, trans_b=True)
        except Exception:
            return False, "fwd-exception"
        if torch.isnan(out).any() or torch.isinf(out).any():
            return False, "fwd-nan"
        grad_out = torch.randn_like(out)
        try:
            out.backward(grad_out, retain_graph=True)
        except Exception:
            return False, "bwd-exception"

    x_ref = x.detach().clone().requires_grad_()
    w_ref = w.detach().clone().requires_grad_()
    out_ref = grouped_gemm_ref(x_ref, w_ref, group_lens, trans_b=True)
    out_ref.backward(grad_out)

    if not check_allclose(out.detach(), out_ref.detach(), torch.bfloat16):
        return False, "fwd"
    if x.grad is None:
        return False, "dA-none"
    if not check_allclose(x.grad, x_ref.grad, torch.bfloat16):
        return False, "dA"
    if w.grad is None:
        return False, "dB-none"
    if not check_allclose(w.grad, w_ref.grad, torch.bfloat16):
        return False, "dB"
    return True, ""


def _check_grouped_fp8_correctness(B: int, M: int, N: int, K: int) -> tuple[bool, str]:
    """Run HIPKITTEN FP8 grouped fwd+bwd, SNR vs torch-native ref.

    Returns (ok, fail_reason). On PASS, fail_reason is "". On FAIL, a short
    code: "fwd-exception" / "fwd-nan" / "fwd-snr<XX.X" / "bwd-exception" /
    "dA-snr<XX.X" / "dB-snr<XX.X". SNR threshold is 25 dB (E4M3 tensorwise),
    matching bench_grouped_gemm_turbo.check_grouped_gemm_fp8_correctness.
    """
    torch.manual_seed(0)
    cfg = Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.TENSORWISE)
    a = torch.randn((B * M, K), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    b = torch.randn((B, N, K), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    group_lens = torch.full((B,), M, dtype=torch.int64, device="cuda")

    with hk_ratio.force_grouped_gemm_backend(BackendType.HIPKITTEN, PrecisionType.FP8):
        try:
            out = turbo.ops.grouped_gemm_fp8(a, b, group_lens, trans_b=True, config=cfg)
        except Exception:
            return False, "fwd-exception"
        if torch.isnan(out).any() or torch.isinf(out).any():
            return False, "fwd-nan"
        grad_out = torch.randn_like(out)
        try:
            out.backward(grad_out, retain_graph=True)
        except Exception:
            return False, "bwd-exception"

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


_FOCUS_PREFIXES = {
    "gpt_oss": ("gpt_oss",),
    "deepseek": ("DeepSeek",),
    "dsv3": ("DeepSeek",),
    "all": (),
    "": (),
}


_SEGMENT_FILTERS = {
    "grp_bf16": {"grp_bf16"},
    "grp_fp8": {"grp_fp8"},
    "bf16": {"grp_bf16"},
    "fp8": {"grp_fp8"},
    "all": {"grp_bf16", "grp_fp8"},
    "": {"grp_bf16", "grp_fp8"},
}


def _resolve_focus() -> tuple[str, tuple[str, ...]]:
    """Resolve METRIC_MODEL_FILTER env var to (label, prefixes).

    label is the canonical name printed to stderr; prefixes are the model
    name prefixes whose shapes count toward score/Goals (other shapes
    still get benchmarked + correctness-checked but are tagged [watch]
    in the per-shape table and excluded from the geomean / Goals / score).
    """
    raw = os.environ.get("METRIC_MODEL_FILTER", "").strip().lower()
    if raw not in _FOCUS_PREFIXES:
        # Unknown value — treat as "all" but warn so misconfig is visible.
        print(
            f"[metric_grouped_only] WARN: unknown METRIC_MODEL_FILTER={raw!r}, "
            "falling back to all-shapes scoring",
            file=sys.stderr,
        )
        return ("all", ())
    return (raw or "all", _FOCUS_PREFIXES[raw])


def _resolve_segment_filter() -> tuple[str, set[str]]:
    """Resolve METRIC_SEGMENT_FILTER env var to (label, allowed_sections).

    Independent from _resolve_focus (model). When the user wants to attack
    only one section (e.g. ``grp_fp8`` because BF16 already passes target),
    set ``METRIC_SEGMENT_FILTER=grp_fp8``: BF16 rows are still benchmarked
    and correctness-checked (FAILs are still hard errors) but tagged
    ``[watch]`` and excluded from score / Goals geomean. Section orthogonal
    to model focus — the two intersect to determine in_focus.
    """
    raw = os.environ.get("METRIC_SEGMENT_FILTER", "").strip().lower()
    if raw not in _SEGMENT_FILTERS:
        print(
            f"[metric_grouped_only] WARN: unknown METRIC_SEGMENT_FILTER={raw!r}, "
            "falling back to all-segments scoring",
            file=sys.stderr,
        )
        return ("all", _SEGMENT_FILTERS["all"])
    label = raw or "all"
    return (label, _SEGMENT_FILTERS[raw])


def _is_in_focus(
    shape_name: str,
    section: str,
    focus_prefixes: tuple[str, ...],
    segment_filter: set[str],
) -> bool:
    """Whether a shape counts toward score: must satisfy BOTH model + segment focus."""
    if section not in segment_filter:
        return False
    if not focus_prefixes:
        return True
    return any(shape_name.startswith(p) for p in focus_prefixes)


def _run() -> int:
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        print("[metric_grouped_only] CUDA/ROCm not available", file=sys.stderr)
        print(0)
        return 1

    focus_label, focus_prefixes = _resolve_focus()
    segment_label, segment_filter = _resolve_segment_filter()

    # row tuple: (name, hk_tflops, ref_tflops, ratio, section, ok, fail_reason, in_focus)
    rows: list[tuple[str, float, float, float, str, bool, str, bool]] = []
    t0 = time.monotonic()
    TRT = BackendType.TRITON
    HK = BackendType.HIPKITTEN

    grouped_shapes = hk_ratio.GROUPED_BF16_SHAPES  # 16 cases (DeepSeek-V3 + gpt_oss_20B)

    n_focus_bf16 = sum(
        1 for s in grouped_shapes if _is_in_focus(s[0], "grp_bf16", focus_prefixes, segment_filter)
    )
    n_focus_fp8 = sum(
        1 for s in grouped_shapes if _is_in_focus(s[0], "grp_fp8", focus_prefixes, segment_filter)
    )
    n_focus_total = n_focus_bf16 + n_focus_fp8
    print(
        f"[metric_grouped_only] suite: {len(grouped_shapes)} BF16 + "
        f"{len(grouped_shapes)} FP8 = {2 * len(grouped_shapes)} grouped cases "
        f"(each gated on fwd+dA+dB correctness vs torch ref) | "
        f"model_focus={focus_label} segment_focus={segment_label} "
        f"({n_focus_total}/{2 * len(grouped_shapes)} shapes count toward score; "
        f"others tagged [watch])",
        file=sys.stderr,
    )

    # ── (1) Grouped BF16 vs TRITON ──
    for (name, B, M, N, K) in grouped_shapes:
        ok, reason = _check_grouped_bf16_correctness(B, M, N, K)
        ref = hk_ratio._bench_grouped_bf16(B, M, N, K, backend=TRT)
        hk = hk_ratio._bench_grouped_bf16(B, M, N, K, backend=HK)
        if ok:
            ratio = (hk / ref) if ref > 0 else 0.0
        else:
            # Fast-but-wrong kernel must NOT bank score: clip ratio to 0
            # regardless of measured TFLOPS. We still record hk/ref TFLOPS
            # in the table for the agent to see (visibility into "where
            # the kernel is broken vs where it's slow").
            ratio = 0.0
        in_focus = _is_in_focus(name, "grp_bf16", focus_prefixes, segment_filter)
        rows.append((f"grpBF16_{name}", hk, ref, ratio, "grp_bf16", ok, reason, in_focus))

    # ── (2) Grouped FP8 tensorwise vs TRITON ──
    for (name, B, M, N, K) in grouped_shapes:
        ok, reason = _check_grouped_fp8_correctness(B, M, N, K)
        ref = hk_ratio._bench_grouped_fp8(B, M, N, K, backend=TRT)
        hk = hk_ratio._bench_grouped_fp8(B, M, N, K, backend=HK)
        if ok:
            ratio = (hk / ref) if ref > 0 else 0.0
        else:
            ratio = 0.0
        in_focus = _is_in_focus(name, "grp_fp8", focus_prefixes, segment_filter)
        rows.append((f"grpFP8_{name}", hk, ref, ratio, "grp_fp8", ok, reason, in_focus))

    wall = time.monotonic() - t0

    # Per-shape table to stderr.
    print(
        f"\n[metric_grouped_only] Shape suite ({len(rows)} cases, {wall:.1f}s wall):",
        file=sys.stderr,
    )
    print(
        f"  {'name':50s}  {'hk_tflops':>10s}  {'trt_tflops':>10s}  {'ratio':>6s}  status",
        file=sys.stderr,
    )
    for name, hk, ref, r, _section, ok, reason, in_focus in rows:
        focus_tag = "" if in_focus else "  [watch]"
        if not ok:
            flag = f"  *FAIL[{reason}]"
        elif r == 0:
            flag = "  *reject"
        else:
            flag = "" if r >= 1.20 else "  <120%"
        print(
            f"  {name:50s}  {hk:>10.1f}  {ref:>10.1f}  {r:>6.3f}{flag}{focus_tag}",
            file=sys.stderr,
        )

    # Per-section geomean breakdown (clip rejects + correctness FAIL to 0.01
    # floor for log-safety). Only in-focus shapes count toward the geomean
    # / Goals / score; [watch] shapes are excluded so the agent's score is
    # responsive to the focused model's perf, but still see the watch ratios
    # in the table for regression visibility. Correctness FAIL on a [watch]
    # shape still surfaces in the FAIL line below.
    def _section_geomean(section: str) -> tuple[float, int]:
        sub = [
            max(r, 0.01)
            for _n, _, _, r, s, _ok, _why, in_f in rows
            if s == section and in_f
        ]
        if not sub:
            return (float("nan"), 0)
        return (math.exp(sum(math.log(r) for r in sub) / len(sub)), len(sub))

    g_grp_bf16, n_grp_bf16 = _section_geomean("grp_bf16")
    g_grp_fp8, n_grp_fp8 = _section_geomean("grp_fp8")

    for label, g, n in [
        ("grp_BF16  vs triton", g_grp_bf16, n_grp_bf16),
        ("grp_FP8   vs triton", g_grp_fp8, n_grp_fp8),
    ]:
        if n:
            print(
                f"[metric_grouped_only]   {label} geomean={g:.4f} (n={n})",
                file=sys.stderr,
            )

    # Two-goal acceptance: both grouped sections >= 1.20 vs TRITON.
    print("[metric_grouped_only] Goals:", file=sys.stderr)
    goals = [
        ("(1) grp_BF16  vs triton  >= 1.20", g_grp_bf16, 1.20, n_grp_bf16),
        ("(2) grp_FP8   vs triton  >= 1.20", g_grp_fp8, 1.20, n_grp_fp8),
    ]

    # Equal weight 1:1 — both sections matter equally; agent picks by which
    # has the bigger gap to target.
    SECTION_WEIGHTS: dict[str, float] = {
        "grp_bf16": 1.0,
        "grp_fp8": 1.0,
    }
    section_keys = ["grp_bf16", "grp_fp8"]

    n_pass = 0
    n_active_goals = 0  # only segments with n > 0 (i.e. in segment_filter)
    progresses: list[tuple[str, float, float]] = []  # (key, progress, weight)
    for (label, g, target, n), key in zip(goals, section_keys):
        if n == 0:
            # Segment excluded by METRIC_SEGMENT_FILTER — skip from goals
            # and weighted geomean entirely. Don't drag score down with a
            # log-floor 0.001 progress on a section the user explicitly
            # said "I'm not attacking this right now".
            print(
                f"[metric_grouped_only]   {label}  : (segment excluded by "
                f"METRIC_SEGMENT_FILTER, skipped)",
                file=sys.stderr,
            )
            continue
        n_active_goals += 1
        passed = (g >= target)
        if passed:
            n_pass += 1
        # Capped progress: overshooting one section does NOT bank extra
        # score — agent must close the gap on the section still below.
        progress = min(g / target, 1.0) if target > 0 else 0.0
        progress = max(progress, 0.001)  # log-safe floor
        weight = SECTION_WEIGHTS.get(key, 1.0)
        progresses.append((key, progress, weight))
        print(
            f"[metric_grouped_only]   {label}  : {g:.4f}  "
            f"progress={progress:.3f}  w={weight:.0f}  "
            f"{'PASS' if passed else 'FAIL'}",
            file=sys.stderr,
        )

    # Weighted geomean of capped progresses across in-focus segments only.
    # When METRIC_SEGMENT_FILTER restricts to one segment, score == that
    # segment's progress * 1000 (target 1.20 → score=1000 iff geomean>=1.20).
    total_w = sum(w for _, _, w in progresses)
    if total_w > 0:
        weighted_log = sum(w * math.log(p) for _, p, w in progresses) / total_w
        weighted_progress = math.exp(weighted_log)
    else:
        weighted_progress = 0.0
    score = int(round(weighted_progress * 1000))

    # Counts are over the focused subset (the part that drives score). Watch
    # shapes get separate counters so the user can see at-a-glance whether a
    # silent regression is happening on the un-focused model.
    n_focus_total = sum(1 for r in rows if r[7])
    n_correct_fail = sum(1 for _n, _, _, _r, _s, ok, _why, in_f in rows if in_f and not ok)
    n_reject = sum(1 for _n, _, _, r, _s, ok, _why, in_f in rows if in_f and ok and r == 0)
    n_below = sum(1 for _n, _, _, r, _s, ok, _why, in_f in rows if in_f and ok and 0 < r < 1.20)

    n_watch_fail = sum(1 for _n, _, _, _r, _s, ok, _why, in_f in rows if (not in_f) and not ok)
    n_watch_below = sum(
        1 for _n, _, _, r, _s, ok, _why, in_f in rows if (not in_f) and ok and 0 < r < 1.0
    )
    n_watch_total = len(rows) - n_focus_total

    print(
        f"[metric_grouped_only] model_focus={focus_label} segment_focus={segment_label}  "
        f"weighted_progress={weighted_progress:.4f}  "
        f"correct_fail={n_correct_fail}/{n_focus_total}  "
        f"reject={n_reject}/{n_focus_total}  below_target={n_below}/{n_focus_total}  "
        f"goals={n_pass}/{n_active_goals}  score={score}  weights=grpBF16:1 grpFP8:1",
        file=sys.stderr,
    )
    if n_watch_total:
        print(
            f"[metric_grouped_only] watch (not-counted): "
            f"correct_fail={n_watch_fail}/{n_watch_total}  "
            f"below_1.0={n_watch_below}/{n_watch_total}",
            file=sys.stderr,
        )
    if n_correct_fail or n_watch_fail:
        # Surface the broken shapes in a single grep-friendly line so the
        # agent (and the auto_optimize round prompt) can see them at a glance
        # without scrolling the per-shape table. Tag [watch] entries so it's
        # clear those don't drive score (but still need to not regress).
        bad = []
        for name, _hk, _ref, _r, _s, ok, why, in_f in rows:
            if not ok:
                tag = "" if in_f else "[watch]"
                bad.append(f"{name}{tag}({why})")
        print(
            f"[metric_grouped_only] correctness FAIL shapes: {', '.join(bad)}",
            file=sys.stderr,
        )

    print(score)  # stdout: integer only, parsed by auto_optimize.py
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
