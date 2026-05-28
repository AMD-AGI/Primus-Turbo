###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch
import torch.nn.functional as F

from primus_turbo.pytorch.ops.normalization import rmsnorm
from tests.pytorch.test_utils import get_tolerances


# inner_shape spans:
#   - tiny aligned (64/128/256/512) → warp-per-row fast path
#     (fp32 ≤256, fp16/bf16 ≤512)
#   - 33/513 → unaligned fallback (UNROLL=1)
#   - 4096..8192 → block-per-row at LDGS=1..2 + iter-3 per-shape UNROLL/2
#   - 12288/16384 → block-per-row at LDGS=2..4, beyond the iter-3 range; also
#     exercises the LDGS-cap guard added to launch_{fwd,bwd}_stage0. Without
#     this coverage a regression at hidden ≥ 12K would slip past unit tests
#     and only surface in bench/training.
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("outer_shape", [(1,), (511,), (4096,), (8192,), (16384,)])
@pytest.mark.parametrize("inner_shape", [64, 128, 256, 512, 33, 513, 4096, 5120, 7168, 8192, 12288, 16384])
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

    # print(y_ref, y_ref.shape)
    # print(y, y.shape)
    torch.testing.assert_close(y_ref, y, **get_tolerances(dtype))

    # Backward
    grad_out = torch.randn_like(y)
    y.backward(grad_out)
    y_ref.backward(grad_out)

    # print(x.grad)
    # print(x_ref.grad)

    # print(gamma.grad)
    # print(gamma_ref.grad)

    torch.testing.assert_close(x.grad, x_ref.grad, **get_tolerances(dtype))
    torch.testing.assert_close(gamma.grad, gamma_ref.grad, **get_tolerances(dtype))
