###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch

from primus_turbo.pytorch.core.backend import GlobalBackendManager
from primus_turbo.pytorch.modules import GroupedLinear
from tests.pytorch.ref.gemm_ref import (
    GroupedLinearRef,
    generate_grouped_gemm_group_lens,
)
from tests.pytorch.test_utils import get_tolerances


@pytest.mark.parametrize("B", [4, 8])
@pytest.mark.parametrize("M", [256, 512, 2048])
@pytest.mark.parametrize("N_K", [(1024, 2048), (512, 1024)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("balance", [True, False])
@pytest.mark.parametrize("auto_tune", [False, True])
@pytest.mark.parametrize("enable_torch_compile", [True, False])
def test_grouped_linear(B, M, N_K, dtype, balance, auto_tune, enable_torch_compile):
    GlobalBackendManager.set_auto_tune(auto_tune)
    torch._dynamo.reset()
    device = "cuda"
    N, K = N_K
    group_lens = generate_grouped_gemm_group_lens(B, M, balance=balance).to(device)

    primus_linear = GroupedLinear(B, K, N, device, dtype=dtype)
    torch_linear = GroupedLinearRef(B, K, N, device, dtype=dtype)
    with torch.no_grad():
        torch_linear.weight.copy_(primus_linear.weight)
    if enable_torch_compile:
        primus_linear = torch.compile(primus_linear, fullgraph=True, mode="max-autotune")

    x1 = torch.randn((B * M, K), device=device, dtype=dtype, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_()

    # FWD
    out1 = primus_linear(x1, group_lens)
    out2 = torch_linear(x2, group_lens)
    torch.testing.assert_close(out1, out2, **get_tolerances(dtype))

    # BWD
    grad_output = torch.randn_like(out1)
    out1.backward(grad_output)
    out2.backward(grad_output)
    torch.testing.assert_close(x1.grad, x2.grad, **get_tolerances(dtype))
    torch.testing.assert_close(
        primus_linear.weight.grad,
        torch_linear.weight.grad,
        **get_tolerances(dtype),
    )

    GlobalBackendManager.set_auto_tune(None)
