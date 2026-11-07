###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Ragged grouped GEMM with frozen inputs and noncontiguous output gradients."""

import pytest
import torch

from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager
from primus_turbo.pytorch.ops import grouped_gemm
from tests.pytorch.ops.gradient_contract_helpers import (
    assert_bf16_close,
    caller_thread_backward,  # noqa: F401
)

pytestmark = pytest.mark.usefixtures("caller_thread_backward")


@pytest.mark.parametrize("backend", [BackendType.CK, BackendType.HIPBLASLT, BackendType.TRITON])
@pytest.mark.parametrize("grad_inputs", ["a", "b", "ab"])
@pytest.mark.parametrize("trans_b", [True, False])
def test_ragged_partial_gradients(backend, grad_inputs, trans_b):
    torch.manual_seed(715)
    GlobalBackendManager.set_grouped_gemm_backend(backend)
    lengths = [17, 129, 238]
    lens = torch.tensor(lengths, device="cuda", dtype=torch.int64)
    a = torch.randn(sum(lengths), 128, device="cuda", dtype=torch.bfloat16).requires_grad_("a" in grad_inputs)
    b = torch.randn(
        (3, 64, 128) if trans_b else (3, 128, 64), device="cuda", dtype=torch.bfloat16
    ).requires_grad_("b" in grad_inputs)
    ar, br = [x.detach().float().requires_grad_(x.requires_grad) for x in (a, b)]
    ref = torch.cat([x @ (w.T if trans_b else w) for x, w in zip(ar.split(lengths), br)])
    out = grouped_gemm(a, b, lens, trans_b=trans_b)
    assert_bf16_close(out, ref, 45)
    grad = torch.randn(64, sum(lengths), device="cuda", dtype=torch.bfloat16).T
    out.backward(grad)
    ref.backward(grad.float())
    for actual, expected in ((a, ar), (b, br)):
        if actual.requires_grad:
            assert_bf16_close(actual.grad, expected.grad, 45)
        else:
            assert actual.grad is None
