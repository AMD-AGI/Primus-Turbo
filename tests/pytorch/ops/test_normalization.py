import pytest
import torch
import torch.nn.functional as F

from primus_turbo.pytorch.ops.normalization import rmsnorm
from tests.test_utils import get_tolerances


# @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("outer_shape", [(1,), (511,), (4096,), (8192,), (16384,)])
@pytest.mark.parametrize("inner_shape", [33, 513, 4096, 5120, 7168, 8192])
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
