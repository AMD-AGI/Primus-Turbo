###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch

from primus_turbo.pytorch.ops import grouped_gemm
from tests.pytorch.ref.gemm_ref import (
    generate_grouped_gemm_group_lens,
    grouped_gemm_ref,
)
from tests.test_utils import get_tolerances


# TODO: "N_K" = [2048, 1408] is bad case
@pytest.mark.parametrize("B", [1, 2, 3, 4, 8, 16])
@pytest.mark.parametrize("M", [128, 256, 512, 1024, 2048])
@pytest.mark.parametrize("N_K", [(2816, 2048), (3072, 5120), (5120, 1536), (4096, 7168)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("balance", [True, False])
@pytest.mark.parametrize("trans_b", [True, False])
def test_grouped_gemm_func(B, M, N_K, dtype, balance, trans_b):
    device = "cuda"
    N, K = N_K
    group_lens = generate_grouped_gemm_group_lens(B, M, balance=balance).to(device)
    print(B, M, N, K, dtype, balance, trans_b)

    b_shape = (B, N, K) if trans_b else (B, K, N)

    a = torch.randn((B * M, K), dtype=dtype, device=device, requires_grad=True)
    b = torch.randn(b_shape, dtype=dtype, device=device, requires_grad=True)
    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)

    # FWD
    out = grouped_gemm(a, b, group_lens, trans_b=trans_b)
    out_ref = grouped_gemm_ref(a_ref, b_ref, group_lens.clone(), trans_b)
    torch.testing.assert_close(out_ref, out, **get_tolerances(dtype))

    # BWD
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)
    out.backward(grad_out)
    torch.testing.assert_close(a_ref.grad, a.grad, **get_tolerances(dtype))
    torch.testing.assert_close(b_ref.grad, b.grad, **get_tolerances(dtype))
