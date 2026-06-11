###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch
import torch.nn.functional as F

import primus_turbo.pytorch as turbo
from tests.pytorch.test_utils import get_tolerances


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rmsnorm(dtype):
    torch.manual_seed(42)
    device = "cuda"
    eps = 1e-6
    seq = 128
    hidden = 512

    x_torch = torch.randn(seq, hidden, device=device, dtype=dtype, requires_grad=True)
    x_turbo = x_torch.clone().detach().requires_grad_()

    torch_rmsnorm = torch.nn.RMSNorm(normalized_shape=hidden, eps=eps, device=device, dtype=dtype)
    turbo_rmsnorm = turbo.modules.RMSNorm(normalized_shape=hidden, eps=eps, device=device, dtype=dtype)
    turbo_rmsnorm.weight.data.copy_(torch_rmsnorm.weight.data)

    # FWD
    out_torch = torch_rmsnorm(x_torch)
    out_turbo = turbo_rmsnorm(x_turbo)

    # print(out_torch)
    # print(out_turbo)
    torch.testing.assert_close(out_torch, out_turbo, **get_tolerances(dtype))

    # BWD
    grad_out = torch.randn_like(out_torch)
    out_torch.backward(grad_out)
    out_turbo.backward(grad_out)

    torch.testing.assert_close(x_torch.grad, x_turbo.grad, **get_tolerances(dtype))
    torch.testing.assert_close(turbo_rmsnorm.weight.grad, torch_rmsnorm.weight.grad, **get_tolerances(dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rmsnorm_zero_centered(dtype):
    torch.manual_seed(42)
    device = "cuda"
    eps = 1e-6
    seq = 128
    hidden = 512

    turbo_rmsnorm = turbo.modules.RMSNorm(
        normalized_shape=hidden, eps=eps, zero_centered_gamma=True, device=device, dtype=dtype
    )
    # zero_centered_gamma initializes the weight to zeros (effective gain == 1).
    assert torch.count_nonzero(turbo_rmsnorm.weight) == 0

    # Non-trivial weight so the (1 + gamma) gain actually exercises the kernel.
    turbo_rmsnorm.weight.data.copy_(torch.randn(hidden, device=device, dtype=dtype) * 0.1)

    x_turbo = torch.randn(seq, hidden, device=device, dtype=dtype, requires_grad=True)
    x_ref = x_turbo.clone().detach().requires_grad_()
    weight_ref = turbo_rmsnorm.weight.detach().clone().requires_grad_()

    # FWD — reference folds the +1 into the gain.
    out_turbo = turbo_rmsnorm(x_turbo)
    out_ref = F.rms_norm(x_ref, [hidden], 1.0 + weight_ref, eps)

    torch.testing.assert_close(out_ref, out_turbo, **get_tolerances(dtype))

    # BWD
    grad_out = torch.randn_like(out_turbo)
    out_turbo.backward(grad_out)
    out_ref.backward(grad_out)

    torch.testing.assert_close(x_turbo.grad, x_ref.grad, **get_tolerances(dtype))
    torch.testing.assert_close(turbo_rmsnorm.weight.grad, weight_ref.grad, **get_tolerances(dtype))
