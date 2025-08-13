###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch

import primus_turbo.pytorch as turbo
from tests.test_utils import get_tolerances


@pytest.mark.parametrize("m", [1, 16, 128, 256, 512, 1024, 2048])
@pytest.mark.parametrize("n", [1, 16, 129, 512, 1024, 2048, 4096])
@pytest.mark.parametrize("k", [1, 16, 127, 255, 512, 1024, 2048])
@pytest.mark.parametrize("layout", ["TN", "NN", "NT"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_gemm(m, n, k, layout, dtype):
    transA = layout[0] == "T"
    transB = layout[1] == "T"

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = "cuda"
    torch.manual_seed(42)

    print(f"\nM={m}, N={n}, K={k}, TransA={transA}, TransB={transB}, dtype={dtype}")

    a_shape = (m, k) if not transA else (k, m)
    b_shape = (k, n) if not transB else (n, k)

    a = torch.randn(a_shape, dtype=dtype, device=device)
    b = torch.randn(b_shape, dtype=dtype, device=device)
    a = a / a.abs().max()
    b = b / b.abs().max()
    a.requires_grad_()
    b.requires_grad_()
    a.grad = None
    b.grad = None
    a_ref = a.detach().clone().requires_grad_()
    b_ref = b.detach().clone().requires_grad_()
    torch.cuda.synchronize()

    # Reference output
    a_mat = a_ref.T if transA else a_ref
    b_mat = b_ref.T if transB else b_ref
    c_ref = a_mat @ b_mat

    # Turbo
    c = turbo.ops.gemm(a, b, transA, transB, dtype)

    # print("a:", a.shape)
    # print("b:", b.shape)
    # print("c: ", c, c.shape)
    # print("c_ref: ", c_ref, c_ref.shape)

    # Check fwd
    torch.testing.assert_close(c, c_ref, **get_tolerances(dtype))

    # Backward
    grad_c = torch.randn_like(c)
    c_ref.backward(grad_c)
    c.backward(grad_c)
    torch.testing.assert_close(a.grad, a_ref.grad, **get_tolerances(dtype))
    torch.testing.assert_close(b.grad, b_ref.grad, **get_tolerances(dtype))
