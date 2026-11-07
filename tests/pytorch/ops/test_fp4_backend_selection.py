###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shape-time backend selection and Triton variable-K numerical gradients."""

import pytest
import torch

from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager
from primus_turbo.pytorch.core.low_precision import Float4QuantConfig
from primus_turbo.pytorch.core.utils import is_gfx950
from primus_turbo.pytorch.kernels.gemm import gemm_fp4_impl as impl
from primus_turbo.pytorch.ops import gemm_fp4, grouped_gemm_fp4
from tests.pytorch.ops.gradient_contract_helpers import caller_thread_backward  # noqa: F401
from tests.pytorch.test_utils import compute_snr

pytestmark = [
    pytest.mark.usefixtures("caller_thread_backward"),
    pytest.mark.skipif(not is_gfx950(), reason="This regression suite requires gfx950"),
]


@pytest.mark.parametrize("backend", [BackendType.AITER, BackendType.FLYDSL, None])
def test_shape_resolution_matches_actual_kernel(backend, monkeypatch):
    monkeypatch.setattr(impl, "_FP4_BACKEND_FOR_SHAPE", {})
    GlobalBackendManager.set_gemm_backend(backend)
    GlobalBackendManager.set_auto_tune(backend is None)
    selected = impl.resolve_fp4_backend(128, 128, 256, out_dtype=torch.bfloat16, device=torch.device("cuda"))
    assert selected in (BackendType.AITER, BackendType.FLYDSL, BackendType.HIPBLASLT)
    if backend is not None:
        assert selected is backend
    assert (
        impl.resolve_fp4_backend(128, 128, 256, out_dtype=torch.bfloat16, device=torch.device("cuda"))
        is selected
    )
    torch.manual_seed(808)
    a = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    out = gemm_fp4(a, b, trans_b=True, config=Float4QuantConfig())
    assert compute_snr(a.float() @ b.float().T, out.float()) > 10


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_triton_grouped_fp4_backward(dtype):
    GlobalBackendManager.set_grouped_gemm_backend(BackendType.TRITON)
    torch.manual_seed(809)
    a = torch.randn(384, 256, device="cuda", dtype=dtype).requires_grad_()
    b = torch.randn(2, 128, 256, device="cuda", dtype=dtype).requires_grad_()
    lens = torch.tensor([128, 256], device="cuda", dtype=torch.int64)
    ar, br = a.detach().float().requires_grad_(), b.detach().float().requires_grad_()
    out = grouped_gemm_fp4(a, b, lens, trans_b=True, config=Float4QuantConfig())
    ref = torch.cat([x @ w.T for x, w in zip(ar.split([128, 256]), br)])
    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad.float())
    for actual, expected in [(out, ref), (a.grad, ar.grad), (b.grad, br.grad)]:
        assert actual.shape == expected.shape
        assert compute_snr(expected, actual.float()) > 8


@pytest.mark.parametrize("precision", ["bf16", "fp8", "fp4"])
def test_pinned_accumulation_with_global_autotune_enabled(precision):
    from tests.pytorch.ops.test_gemm_accumulation_gradients import (
        test_existing_main_grad_accumulates as run_accum,
    )

    GlobalBackendManager.set_auto_tune(True)
    run_accum(precision, grouped=True, activation_grad=True)
