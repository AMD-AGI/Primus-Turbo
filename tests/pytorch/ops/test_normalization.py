###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch
import torch.nn.functional as F

from primus_turbo.pytorch.ops.normalization import rmsnorm, rmsnorm_residual
from tests.pytorch.test_utils import get_tolerances


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("outer_shape", [(1,), (511,), (4096,), (8192,)])
@pytest.mark.parametrize("inner_shape", [33, 128, 513, 4096, 5120, 7168, 8192])
def test_rmsnorm_ops(dtype, outer_shape, inner_shape):
    torch.manual_seed(1)
    device = "cuda:0"
    eps = 1e-6

    shape = outer_shape + (inner_shape,)
    x = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    gamma = torch.randn(inner_shape, dtype=dtype, device=device, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()
    gamma_ref = gamma.detach().clone().requires_grad_()

    # Forward
    y_ref = F.rms_norm(x_ref, [inner_shape], gamma_ref, eps)
    y = rmsnorm(x, gamma, eps)

    torch.testing.assert_close(y_ref, y, **get_tolerances(dtype))

    # Backward
    grad_out = torch.randn_like(y)
    y.backward(grad_out)
    y_ref.backward(grad_out)

    torch.testing.assert_close(x.grad, x_ref.grad, **get_tolerances(dtype))
    torch.testing.assert_close(gamma.grad, gamma_ref.grad, **get_tolerances(dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("outer_shape", [(1,), (511,), (4096,)])
@pytest.mark.parametrize("inner_shape", [128, 4096, 8192])
def test_rmsnorm_zero_centered_ops(dtype, outer_shape, inner_shape):
    # zero_centered=True applies a (1 + gamma) gain. Reference folds the +1 into the weight.
    torch.manual_seed(3)
    device = "cuda:0"
    eps = 1e-6

    shape = outer_shape + (inner_shape,)
    x = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    gamma = torch.randn(inner_shape, dtype=dtype, device=device, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()
    gamma_ref = gamma.detach().clone().requires_grad_()

    # FWD — reference uses (1 + gamma) as the gain.
    y_ref = F.rms_norm(x_ref, [inner_shape], 1.0 + gamma_ref, eps)
    y = rmsnorm(x, gamma, eps, zero_centered=True)

    torch.testing.assert_close(y_ref, y, **get_tolerances(dtype))

    # BWD — d(1 + gamma)/d(gamma) = 1, so the gamma grad is unchanged and
    # flows through (1 + gamma_ref) to the gamma_ref leaf.
    grad_out = torch.randn_like(y)
    y.backward(grad_out)
    y_ref.backward(grad_out)

    torch.testing.assert_close(x.grad, x_ref.grad, **get_tolerances(dtype))
    torch.testing.assert_close(gamma.grad, gamma_ref.grad, **get_tolerances(dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("outer_shape", [(1,), (511,), (4096,)])
@pytest.mark.parametrize("inner_shape", [128, 4096, 8192])
def test_rmsnorm_residual_ops(dtype, outer_shape, inner_shape):
    torch.manual_seed(2)
    device = "cuda:0"
    eps = 1e-6

    shape = outer_shape + (inner_shape,)
    x = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    residual = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    gamma = torch.randn(inner_shape, dtype=dtype, device=device, requires_grad=True)

    x_ref = x.detach().clone().requires_grad_()
    r_ref = residual.detach().clone().requires_grad_()
    gamma_ref = gamma.detach().clone().requires_grad_()

    # Forward
    h_ref = x_ref + r_ref
    y_ref = F.rms_norm(h_ref, [inner_shape], gamma_ref, eps)

    y, x_plus_r = rmsnorm_residual(x, residual, gamma, eps)

    torch.testing.assert_close(x_plus_r, h_ref.detach(), **get_tolerances(dtype))
    torch.testing.assert_close(y, y_ref, **get_tolerances(dtype))

    # Backward — only flow gradient through y so dxpr = 0.
    grad_out = torch.randn_like(y)
    y.backward(grad_out)
    y_ref.backward(grad_out)

    torch.testing.assert_close(x.grad, x_ref.grad, **get_tolerances(dtype))
    torch.testing.assert_close(residual.grad, r_ref.grad, **get_tolerances(dtype))
    # ``dgamma`` is a reduction over ``B`` rows; the residual variant doubles the
    # input magnitude entering the sum, so for low-precision dtypes we use a
    # slightly looser tolerance to absorb the extra bf16/fp16 reduction noise.
    dg_tol = get_tolerances(dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dg_tol = dict(rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(gamma.grad, gamma_ref.grad, **dg_tol)
