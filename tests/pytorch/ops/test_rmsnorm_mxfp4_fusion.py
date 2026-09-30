###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch


def _require_gfx950():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    arch = str(torch.cuda.get_device_properties(0).gcnArchName).split(":", 1)[0]
    if arch != "gfx950":
        pytest.skip("fused RMSNorm MXFP4 requires gfx950")


def _inputs(rows=256, hidden=2880, seed=123):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def tensor(*shape):
        return (
            torch.randn(*shape, device="cuda", dtype=torch.float32, generator=generator)
            .bfloat16()
            .requires_grad_()
        )

    return tensor(rows, hidden), tensor(rows, hidden), tensor(hidden)


def test_rmsnorm_mxfp4_skip_y_forward_backward():
    _require_gfx950()
    from primus_turbo.flydsl.quantization.rmsnorm_mxfp4_fusion import (
        rmsnorm_residual_mxfp4_fused,
    )

    x_ref, residual_ref, gamma_ref = _inputs()
    x = x_ref.detach().clone().requires_grad_()
    residual = residual_ref.detach().clone().requires_grad_()
    gamma = gamma_ref.detach().clone().requires_grad_()

    reference = rmsnorm_residual_mxfp4_fused(x_ref, residual_ref, gamma_ref, skip_y_store=False)
    fused = rmsnorm_residual_mxfp4_fused(x, residual, gamma, skip_y_store=True)

    torch.testing.assert_close(fused[1], reference[1], rtol=0, atol=0)
    for actual, expected in zip(fused[2:], reference[2:]):
        mismatch = (actual != expected).float().mean()
        assert mismatch <= 0.05

    grad_y = torch.randn_like(x)
    grad_xpr = torch.randn_like(x)
    torch.autograd.backward(reference[:2], (grad_y, grad_xpr))
    torch.autograd.backward(fused[:2], (grad_y, grad_xpr))
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(residual.grad, residual_ref.grad, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(gamma.grad, gamma_ref.grad, rtol=3e-2, atol=3e-2)


def test_rmsnorm_mxfp4_unsupported_rows_use_reference():
    _require_gfx950()
    from primus_turbo.flydsl.quantization.rmsnorm_mxfp4_fusion import (
        rmsnorm_residual_mxfp4_fused,
    )

    x, residual, gamma = _inputs(rows=32)
    reference = rmsnorm_residual_mxfp4_fused(x, residual, gamma, skip_y_store=False)
    actual = rmsnorm_residual_mxfp4_fused(x, residual, gamma, skip_y_store=True)
    for got, want in zip(actual, reference):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
