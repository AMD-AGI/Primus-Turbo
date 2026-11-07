###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Training-style microbatch accumulation into a pre-existing gradient buffer."""

import pytest
import torch

from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager
from primus_turbo.pytorch.core.low_precision import Float4QuantConfig, Float8QuantConfig, check_mxfp4_support
from primus_turbo.pytorch.ops import (
    gemm,
    gemm_fp4,
    gemm_fp8,
    grouped_gemm,
    grouped_gemm_fp4,
    grouped_gemm_fp8,
)
from tests.pytorch.ops.gradient_contract_helpers import caller_thread_backward  # noqa: F401
from tests.pytorch.test_utils import compute_snr

pytestmark = pytest.mark.usefixtures("caller_thread_backward")


@pytest.mark.parametrize("precision", ["bf16", "fp8", "fp4"])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("activation_grad", [False, True])
def test_existing_main_grad_accumulates(precision, grouped, activation_grad):
    if precision == "fp4":
        supported, reason = check_mxfp4_support()
        if not supported:
            pytest.skip(reason)
    torch.manual_seed(720)
    backend = BackendType.TRITON if grouped else BackendType.HIPBLASLT
    if precision == "fp4":
        backend = BackendType.FLYDSL
    (GlobalBackendManager.set_grouped_gemm_backend if grouped else GlobalBackendManager.set_gemm_backend)(
        backend
    )
    funcs = {
        "bf16": (gemm, grouped_gemm),
        "fp8": (gemm_fp8, grouped_gemm_fp8),
        "fp4": (gemm_fp4, grouped_gemm_fp4),
    }
    func = funcs[precision][int(grouped)]
    options = {"trans_b": True, "fuse_bgrad_accum_pattern": "megatron"}
    if precision != "bf16":
        options["config"] = Float4QuantConfig() if precision == "fp4" else Float8QuantConfig()
    lens = torch.tensor([128, 256], device="cuda", dtype=torch.int64)
    b = torch.randn(
        (2, 128, 256) if grouped else (128, 256), device="cuda", dtype=torch.bfloat16
    ).requires_grad_()
    # FlyDSL FP4 accumulation requires the output dtype (BF16); BF16/FP8
    # implementations also accept a higher-precision FP32 main_grad.
    previous = torch.randn_like(b, dtype=torch.bfloat16 if precision == "fp4" else torch.float32)
    b.main_grad = previous.clone()
    b.grad_added_to_main_grad = False
    expected_delta = torch.zeros_like(previous, dtype=torch.float32)
    snr_floor = {"bf16": 40, "fp8": 25, "fp4": 8}[precision]
    for _ in range(2):
        a = torch.randn(384, 256, device="cuda", dtype=torch.bfloat16).requires_grad_(activation_grad)
        ar, br = a.detach().float().requires_grad_(activation_grad), b.detach().float().requires_grad_()
        args = (a, b, lens) if grouped else (a, b)
        out = func(*args, **options)
        assert b.grad_added_to_main_grad is True
        ref = torch.cat([x @ w.T for x, w in zip(ar.split([128, 256]), br)]) if grouped else ar @ br.T
        grad = torch.randn_like(out)
        out.backward(grad)
        ref.backward(grad.float())
        expected_delta += br.grad
        assert compute_snr(ref, out.float()) > snr_floor
        if activation_grad:
            assert compute_snr(ar.grad, a.grad.float()) > snr_floor
        else:
            assert a.grad is None
        # Check after EACH microbatch, so overwrite/lost-update errors cannot hide.
        assert compute_snr(expected_delta, b.main_grad.float() - previous.float()) > snr_floor
        assert torch.isfinite(b.main_grad).all()
        assert b.grad.shape == b.shape and b.grad.dtype == b.dtype
        # Fused wgrad returns an intentionally uninitialized dummy. Its values are
        # not a numerical result and must not be consumed by an optimizer.
        b.grad = None
