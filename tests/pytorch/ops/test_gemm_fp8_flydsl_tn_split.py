###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Split-K tile assignment of the FlyDSL fp8 TN (wgrad) whole-loop kernel.

Every workgroup of the split-K window derives its (tile, slice) pair, and every slice its
k range, through a compile-time reciprocal divide. If that divide is inexact for any dividend a
workgroup can hold, a slice is computed twice and another never, the reduce folds an unwritten
workspace band into C, and a misassigned workgroup stores outside it.
"""

import pytest
import torch

from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
)
from primus_turbo.pytorch.core.utils import get_device_compute_capability
from primus_turbo.pytorch.ops import gemm_fp8
from tests.pytorch.test_utils import compute_snr

kern = pytest.importorskip("primus_turbo.flydsl.gemm.gemm_fp8_kernel")

torch.manual_seed(42)

# Window tile counts reach the CU count; ids below the window (as low as -lo) only need to stay
# in int32, and lo is at most the grid's whole tiles.
_MAX_WINDOW = 1024
_MAX_LO = 1 << 16


def _assert_exact(s, x_max, x_min=0):
    m, k = kern._slice_div_magic(s, x_max, x_min)
    assert max(x_max, -x_min) * m < 1 << 31, (s, x_max, x_min, m, k)
    x = torch.arange(0, x_max + 1, dtype=torch.int64)
    assert torch.equal((x * m) >> k, x // s), (s, x_max, m, k)


def test_slice_div_window_exact():
    """win_off // nwin over every window the planner can build: nwin tiles, s slices each."""
    for nwin in range(3, _MAX_WINDOW + 1):
        if nwin & (nwin - 1) == 0:
            continue  # a power of two takes the shift
        for s in kern._TN4_SPLIT_S:
            _assert_exact(nwin, nwin * s - 1, -_MAX_LO)


def test_slice_div_k_range_exact():
    """(K_ITERS * slice) // s for every K_ITERS the planner admits."""
    for s in kern._TN4_SPLIT_S:
        if s & (s - 1) == 0:
            continue
        k_iters = (kern._TN4_RCP_MAX - 1) // (s - 1)
        _assert_exact(s, k_iters * s)


def test_slice_div_window_192():
    """The window of an 8960 x 1536 wgrad at 384 x 192 tiles on 256 CUs: 192 tiles x 4 slices.
    (x * ceil(2^16 / 192)) >> 16 gives slices 3 and 4 at x = 575 and 767."""
    m, k = kern._slice_div_magic(192, 767)
    assert [(x * m) >> k for x in (575, 767)] == [2, 3]
    assert [(x * -(-(1 << 16) // 192)) >> 16 for x in (575, 767)] == [3, 4]


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_device_compute_capability() < (9, 5),
    reason="FlyDSL fp8 GEMM is gfx950-only",
)
@pytest.mark.parametrize("format", [Format.E4M3, Format.HYBRID])
def test_gemm_fp8_flydsl_tn_split_window(monkeypatch, format):
    """wgrad (TN, M x N = 8960 x 1536 over 65536 rows) with the 384 x 192 geometry alone in the
    first-call race, so the split-K window it builds is the one that runs."""
    m, k, n = 65536, 8960, 1536
    ncu = kern._dense_num_cus()
    split = kern._tn4_geom_split(k, n, m, ncu, kern._TN4_RECT)
    if split is None or split[1] & (split[1] - 1) == 0:
        pytest.skip(f"no non-power-of-two split-K window for this shape on {ncu} CUs")
    monkeypatch.setattr(kern, "_TN4_GEOMS", (kern._TN4_RECT,))
    monkeypatch.setattr(kern, "_TN4_PLAN_CACHE", {})
    monkeypatch.setattr(kern, "_TN_WAVE4_CACHE", {})
    GlobalBackendManager.set_gemm_backend(BackendType.FLYDSL)
    try:
        device = "cuda:0"
        # NN forward: on gfx950 an NT forward runs its backward in NT layout, which skips TN.
        a = torch.randn((m, k), dtype=torch.bfloat16, device=device, requires_grad=True)
        b = torch.randn((k, n), dtype=torch.bfloat16, device=device, requires_grad=True)
        a_ref = a.detach().clone().requires_grad_()
        b_ref = b.detach().clone().requires_grad_()
        c_ref = a_ref @ b_ref
        grad_c = torch.randn_like(c_ref)
        c_ref.backward(grad_c)

        config = Float8QuantConfig(granularity=ScalingGranularity.TENSORWISE, format=format)
        c = gemm_fp8(a, b, False, False, torch.bfloat16, config)
        c.backward(grad_c)
        torch.cuda.synchronize()

        assert torch.isfinite(b.grad).all(), "wgrad has non-finite values"
        threshold = 25 if format == Format.E4M3 else 20
        b_grad_snr = compute_snr(b_ref.grad, b.grad)
        print(f"split {split}, BGrad-SNR: {b_grad_snr:.2f} dB")
        assert b_grad_snr > threshold, "b_grad_snr too low"
    finally:
        GlobalBackendManager.reset()
