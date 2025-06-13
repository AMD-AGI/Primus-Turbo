import pytest
import torch

import primus_turbo.pytorch as pt
from tests.utils.metric_utils import get_tolerances


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gemm(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"
    m, n, k = 512, 2048, 1024
    a = torch.randn((m, k), dtype=dtype, device=device, requires_grad=True)
    b = torch.randn((k, n), dtype=dtype, device=device, requires_grad=True)
    a_ref = a.detach().clone().requires_grad_()
    b_ref = b.detach().clone().requires_grad_()

    # PyTorch baseline
    c_ref = torch.matmul(a_ref, b_ref)
    loss_ref = c_ref.sum()
    loss_ref.backward()

    # primus_turbo GEMM
    c = pt.ops.gemm(a, b)
    loss = c.sum()
    loss.backward()

    # Log
    print("Forward diff (max abs):", (c - c_ref).detach().abs().max().item())
    print("Grad A diff (max abs):", (a.grad - a_ref.grad).abs().max().item())
    print("Grad B diff (max abs):", (b.grad - b_ref.grad).abs().max().item())

    # Check forward close
    torch.testing.assert_close(c, c_ref, **get_tolerances(dtype))

    # Check gradients
    torch.testing.assert_close(a.grad, a_ref.grad, **get_tolerances(dtype))
    torch.testing.assert_close(b.grad, b_ref.grad, **get_tolerances(dtype))
