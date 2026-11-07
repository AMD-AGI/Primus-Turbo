###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Numerical coverage for tensorwise FP8 backward without the NT fast path."""

import pytest
import torch

from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
)
from primus_turbo.pytorch.ops import gemm_fp8
from tests.pytorch.test_utils import compute_snr


@pytest.fixture(autouse=True)
def hipblaslt_backend():
    GlobalBackendManager.set_gemm_backend(BackendType.HIPBLASLT)
    GlobalBackendManager.set_auto_tune(False)
    yield
    GlobalBackendManager.reset()


@pytest.mark.parametrize("layout", ["NN", "TN"])
@pytest.mark.parametrize("fmt", [Format.E4M3, Format.HYBRID])
@pytest.mark.parametrize("strided_grad", [False, True], ids=["contiguous", "strided"])
@pytest.mark.parametrize("fused", [False, True], ids=["autograd", "main_grad"])
@pytest.mark.parametrize("native_threads", [True, False], ids=["native_threads", "caller_thread"])
def test_tensorwise_non_nt_backward(layout, fmt, strided_grad, fused, native_threads):
    # Unequal dimensions expose accidental operand/output transposes. Both
    # layouts naturally select the general backward path, including on gfx950.
    # hipBLASLt does not support a TT forward GEMM.
    torch.manual_seed(42)
    m, n, k = 128, 256, 384
    trans_a, trans_b = layout[0] == "T", layout[1] == "T"
    a = torch.randn((k, m) if trans_a else (m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k) if trans_b else (k, n), device="cuda", dtype=torch.bfloat16)
    a.requires_grad_()
    b.requires_grad_()
    a_ref, b_ref = a.detach().float().requires_grad_(), b.detach().float().requires_grad_()
    ref = (a_ref.T if trans_a else a_ref) @ (b_ref.T if trans_b else b_ref)
    if strided_grad:
        grad = torch.randn((n, m), device="cuda", dtype=torch.bfloat16).T
        assert not grad.is_contiguous()
    else:
        grad = torch.randn((m, n), device="cuda", dtype=torch.bfloat16)
    ref.backward(grad.float())

    if fused:
        previous = torch.randn_like(b, dtype=torch.float32)
        b.main_grad = previous.clone()
        b.grad_added_to_main_grad = False
        main_grad_ptr = b.main_grad.data_ptr()

    out = gemm_fp8(
        a,
        b,
        trans_a=trans_a,
        trans_b=trans_b,
        config=Float8QuantConfig(format=fmt, granularity=ScalingGranularity.TENSORWISE),
        fuse_bgrad_accum_pattern="megatron" if fused else None,
    )
    # coverage.py cannot trace autograd's native worker threads. Also execute on
    # the caller thread to measure Python backward, retaining the normal thread
    # mode as a numerical regression check. Both modes launch real GPU kernels.
    with torch.autograd.set_multithreading_enabled(native_threads):
        out.backward(grad)
    if fused:
        assert b.grad_added_to_main_grad is True
        assert b.main_grad.data_ptr() == main_grad_ptr
        assert b.grad.shape == b.shape and b.grad.dtype == b.dtype
        b_grad = b.main_grad - previous
    else:
        b_grad = b.grad

    threshold = 25 if fmt == Format.E4M3 else 20
    for name, expected, actual in [
        ("out", ref, out),
        ("a_grad", a_ref.grad, a.grad),
        ("b_grad", b_ref.grad, b_grad),
    ]:
        assert actual.shape == expected.shape
        assert torch.isfinite(actual).all()
        snr = compute_snr(expected, actual)
        assert snr > threshold, f"{name}: {snr:.2f} dB <= {threshold} dB"
