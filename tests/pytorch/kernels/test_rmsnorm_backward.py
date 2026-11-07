###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

import pytest
import torch

from primus_turbo.pytorch.kernels.normalization import rmsnorm_impl as impl


@pytest.mark.parametrize("hidden", [128, 256, 1024, 8192, 16384])
@pytest.mark.parametrize("zero_centered", [False, True])
def test_rmsnorm_backward_against_autograd(hidden, zero_centered):
    torch.manual_seed(10)
    x = torch.randn(8, hidden, device="cuda", requires_grad=True)
    gamma = torch.randn(hidden, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    y, saved, rstd, *config = impl.rmsnorm_fwd_impl(x, gamma, 1e-6, zero_centered=zero_centered)
    dx, dg = impl.rmsnorm_bwd_impl(dy, saved, gamma, rstd, *config, zero_centered=zero_centered)
    ref = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * (gamma + int(zero_centered))
    ref_dx, ref_dg = torch.autograd.grad(ref, (x, gamma), dy)
    torch.testing.assert_close(y, ref, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(dx, ref_dx, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(dg, ref_dg, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("hidden", [128, 1024])
@pytest.mark.parametrize("has_residual_grad", [False, True])
def test_residual_backward_includes_both_output_gradients(hidden, has_residual_grad):
    torch.manual_seed(11)
    x = torch.randn(8, hidden, device="cuda", requires_grad=True)
    residual = torch.randn_like(x, requires_grad=True)
    gamma = torch.randn(hidden, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    dxpr = torch.randn_like(x) if has_residual_grad else None
    _, saved, rstd, *config = impl.rmsnorm_fwd_residual_impl(x, residual, gamma, 1e-6)
    dx, dg = impl.rmsnorm_bwd_residual_impl(dy, dxpr, saved, gamma, rstd, *config)
    combined = x + residual
    ref = combined * torch.rsqrt(combined.square().mean(-1, keepdim=True) + 1e-6) * gamma
    loss = (ref * dy).sum()
    if dxpr is not None:
        loss = loss + (combined * dxpr).sum()
    ref_dx, ref_dr, ref_dg = torch.autograd.grad(loss, (x, residual, gamma))
    torch.testing.assert_close(dx, ref_dx, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(dx, ref_dr, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(dg, ref_dg, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("parts, width", [(1, 17), (64, 128), (513, 128)])
def test_dgamma_reduction_small_and_large_partials(parts, width):
    torch.manual_seed(12)
    partial = torch.randn(parts, width, device="cuda")
    actual = impl._finalize_dgamma(partial, torch.float32)
    torch.testing.assert_close(actual, partial.sum(0), rtol=2e-5, atol=2e-5)


def test_hidden_dimension_mismatch_is_reported():
    with pytest.raises(ValueError, match="last dim mismatch"):
        impl._reshape_batch_hidden(torch.empty(2, 7), 8)
