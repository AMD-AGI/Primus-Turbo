###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused Triton kernel for the delayed FP8 scale update vs a Python-loop reference.

Covers correctness for both amax algorithms, zero / NaN / tiny amaxes, the
circular history index and a 50-step run against the reference.
"""

import pytest
import torch

from primus_turbo.triton.quantization.fp8_delayed_scaling_kernel import (
    fused_delayed_scale_update,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")

DEVICE = "cuda:0"
FP8_FWD_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
FP8_BWD_MAX = torch.finfo(torch.float8_e5m2).max  # 57344.0
FP32_MAX = torch.finfo(torch.float32).max


def _reference_update(amax_history, staged_amaxes, scales, fp8_maxes, history_idx, use_max_algo):
    """Pure-PyTorch reference matching the fused delayed-scale update logic."""
    T, N, H = amax_history.shape
    for t in range(T):
        for m in range(N):
            new_amax = staged_amaxes[t, m].item()
            amax_history[t, m, history_idx] = new_amax

            if use_max_algo:
                amax = amax_history[t, m, :].max().item()
            else:
                amax = new_amax

            fp8_max = fp8_maxes[t].item()
            old_scale = scales[t, m].item()

            if amax > 0.0 and amax == amax:  # positive and not NaN
                sf = fp8_max / max(amax, 1e-12)
                sf = min(sf, FP32_MAX)
            else:
                sf = old_scale

            scales[t, m] = sf


def _run_fused_kernel(amax_history, staged_amaxes, scales, fp8_maxes, history_idx, use_max_algo):
    fused_delayed_scale_update(amax_history, staged_amaxes, scales, fp8_maxes, history_idx, use_max_algo)


@pytest.mark.parametrize(
    "N,H",
    [
        (1, 1),
        (4, 16),
        (57, 1024),
        (10, 128),
    ],
)
@pytest.mark.parametrize("algo", ["max", "most_recent"])
def test_correctness(N, H, algo):
    """Fused kernel must produce identical scales and history as reference."""
    use_max_algo = algo == "max"
    fp8_maxes = torch.tensor([FP8_FWD_MAX, FP8_FWD_MAX, FP8_BWD_MAX], dtype=torch.float32, device=DEVICE)

    torch.manual_seed(42)
    amax_history_ref = torch.rand(3, N, H, dtype=torch.float32, device=DEVICE) * 10.0
    staged_amaxes = torch.rand(3, N, dtype=torch.float32, device=DEVICE) * 5.0
    scales_ref = torch.ones(3, N, dtype=torch.float32, device=DEVICE)
    history_idx = H // 3

    amax_history_fused = amax_history_ref.clone()
    scales_fused = scales_ref.clone()

    _reference_update(amax_history_ref, staged_amaxes, scales_ref, fp8_maxes, history_idx, use_max_algo)
    _run_fused_kernel(amax_history_fused, staged_amaxes, scales_fused, fp8_maxes, history_idx, use_max_algo)

    torch.testing.assert_close(
        amax_history_fused,
        amax_history_ref,
        atol=0,
        rtol=0,
        msg=f"amax_history mismatch for N={N}, H={H}, algo={algo}",
    )
    torch.testing.assert_close(
        scales_fused, scales_ref, atol=1e-5, rtol=1e-5, msg=f"scales mismatch for N={N}, H={H}, algo={algo}"
    )


def test_zero_amaxes():
    """When all staged amaxes are zero and history is zero, scales should stay at old_scale."""
    N, H = 8, 16
    fp8_maxes = torch.tensor([FP8_FWD_MAX, FP8_FWD_MAX, FP8_BWD_MAX], dtype=torch.float32, device=DEVICE)
    amax_history = torch.zeros(3, N, H, dtype=torch.float32, device=DEVICE)
    staged_amaxes = torch.zeros(3, N, dtype=torch.float32, device=DEVICE)
    old_scales = torch.full((3, N), 42.0, dtype=torch.float32, device=DEVICE)
    scales = old_scales.clone()

    _run_fused_kernel(amax_history, staged_amaxes, scales, fp8_maxes, history_idx=0, use_max_algo=True)

    torch.testing.assert_close(
        scales, old_scales, atol=0, rtol=0, msg="Scales should be unchanged when all amaxes are zero"
    )


def test_nan_amaxes():
    """NaN amaxes should leave scales unchanged (NaN guard)."""
    N, H = 4, 8
    fp8_maxes = torch.tensor([FP8_FWD_MAX, FP8_FWD_MAX, FP8_BWD_MAX], dtype=torch.float32, device=DEVICE)
    amax_history = torch.zeros(3, N, H, dtype=torch.float32, device=DEVICE)
    staged_amaxes = torch.full((3, N), float("nan"), dtype=torch.float32, device=DEVICE)
    old_scales = torch.full((3, N), 7.0, dtype=torch.float32, device=DEVICE)
    scales = old_scales.clone()

    _run_fused_kernel(amax_history, staged_amaxes, scales, fp8_maxes, history_idx=0, use_max_algo=True)

    torch.testing.assert_close(
        scales, old_scales, atol=0, rtol=0, msg="Scales should be unchanged when amaxes are NaN"
    )


def test_very_small_amaxes():
    """Very small amaxes should produce large scales clamped to FP32_MAX."""
    N, H = 4, 8
    fp8_maxes = torch.tensor([FP8_FWD_MAX, FP8_FWD_MAX, FP8_BWD_MAX], dtype=torch.float32, device=DEVICE)
    amax_history = torch.zeros(3, N, H, dtype=torch.float32, device=DEVICE)
    staged_amaxes = torch.full((3, N), 1e-40, dtype=torch.float32, device=DEVICE)
    scales = torch.ones(3, N, dtype=torch.float32, device=DEVICE)

    _run_fused_kernel(amax_history, staged_amaxes, scales, fp8_maxes, history_idx=0, use_max_algo=False)

    assert not scales.isinf().any(), "Scales should not be inf (should be clamped)"
    assert not scales.isnan().any(), "Scales should not be NaN"
    assert (scales <= FP32_MAX).all(), "Scales should be <= FP32_MAX"


@pytest.mark.parametrize("H", [4, 16, 64])
def test_history_rollover(H):
    """Circular index should write to correct position across multiple steps."""
    N = 3
    fp8_maxes = torch.tensor([FP8_FWD_MAX, FP8_FWD_MAX, FP8_BWD_MAX], dtype=torch.float32, device=DEVICE)
    amax_history = torch.zeros(3, N, H, dtype=torch.float32, device=DEVICE)
    scales = torch.ones(3, N, dtype=torch.float32, device=DEVICE)

    for step in range(H + 5):
        idx = step % H
        staged = torch.full((3, N), float(step + 1), dtype=torch.float32, device=DEVICE)
        _run_fused_kernel(amax_history, staged, scales, fp8_maxes, history_idx=idx, use_max_algo=True)

        assert amax_history[0, 0, idx].item() == float(step + 1), (
            f"History not written at idx={idx} on step={step}"
        )


def test_multi_step_vs_reference():
    """Run 50 steps of both fused and reference, verify they stay in sync."""
    N, H = 10, 32
    use_max_algo = True
    fp8_maxes = torch.tensor([FP8_FWD_MAX, FP8_FWD_MAX, FP8_BWD_MAX], dtype=torch.float32, device=DEVICE)

    torch.manual_seed(123)
    amax_history_ref = torch.zeros(3, N, H, dtype=torch.float32, device=DEVICE)
    amax_history_fused = torch.zeros(3, N, H, dtype=torch.float32, device=DEVICE)
    scales_ref = torch.ones(3, N, dtype=torch.float32, device=DEVICE)
    scales_fused = torch.ones(3, N, dtype=torch.float32, device=DEVICE)

    for step in range(50):
        idx = step % H
        staged = torch.rand(3, N, dtype=torch.float32, device=DEVICE) * (10.0 + step)

        _reference_update(amax_history_ref, staged, scales_ref, fp8_maxes, idx, use_max_algo)
        _run_fused_kernel(amax_history_fused, staged, scales_fused, fp8_maxes, idx, use_max_algo)

    torch.testing.assert_close(
        amax_history_fused, amax_history_ref, atol=0, rtol=0, msg="amax_history diverged over 50 steps"
    )
    torch.testing.assert_close(
        scales_fused, scales_ref, atol=1e-5, rtol=1e-5, msg="scales diverged over 50 steps"
    )
