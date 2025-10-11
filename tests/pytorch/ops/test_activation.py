###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import random

import pytest
import torch
import torch.nn.functional as F

from primus_turbo.pytorch.ops.activation import geglu_with_probs, swiglu_with_probs
from tests.test_utils import get_tolerances

torch.manual_seed(42)


# NOTE: Align precision with torch.compile
@torch.compile
def swiglu_with_probs_ref(x: torch.Tensor, probs: torch.Tensor):
    dtype = x.dtype
    x = torch.chunk(x, 2, dim=-1)
    res = F.silu(x[0]) * x[1]
    return (res * probs).to(dtype)


def generate_tokens_per_expert_list(num_experts: int, num_tokens: int):
    random.seed(42)

    if num_experts == 1:
        return [num_tokens]

    parts = []
    remaining = num_tokens
    for _ in range(num_experts - 1):
        val = random.randint(0, remaining)
        parts.append(val)
        remaining -= val

    parts.append(remaining)
    return parts


@pytest.mark.parametrize(
    "num_tokens",
    [
        1,
        128,
        2048,
        2025,
        8192 * 8,
    ],
)
@pytest.mark.parametrize(
    "hidden_size",
    [
        128,
        256,
        2048,
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("with_tokens_per_expert", [False, True])
def test_swiglu_with_probs(num_tokens, hidden_size, dtype, with_tokens_per_expert):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"

    probs_dtype = torch.float32

    x = torch.randn(num_tokens, hidden_size * 2, device=device, dtype=dtype, requires_grad=True)
    probs = torch.rand(num_tokens, device=device, dtype=probs_dtype, requires_grad=True)

    x_ref = x.clone().detach()
    x_ref.requires_grad_()
    probs_ref = probs.clone().detach()
    probs_ref.requires_grad_()

    if with_tokens_per_expert:
        num_experts = 64
        tokens_per_expert = torch.tensor(
            generate_tokens_per_expert_list(num_experts, num_tokens), device=device, requires_grad=False
        )
        row_mask = torch.zeros(num_tokens, device=device, dtype=torch.int64, requires_grad=False)
        row_mask[: torch.sum(tokens_per_expert)] = 1
    else:
        row_mask = None

    out = swiglu_with_probs(x, probs, row_mask)
    out_ref = swiglu_with_probs_ref(x_ref, probs_ref.unsqueeze(-1))
    torch.testing.assert_close(out, out_ref, **get_tolerances(dtype))

    out.backward(torch.ones_like(out))
    grad_x = x.grad.clone()
    grad_probs = probs.grad.clone()

    out_ref.backward(torch.ones_like(out_ref))
    grad_x_ref = x_ref.grad.clone()
    grad_probs_ref = probs_ref.grad.clone()

    torch.testing.assert_close(grad_x, grad_x_ref, **get_tolerances(dtype))
    torch.testing.assert_close(grad_probs, grad_probs_ref, **get_tolerances(probs_dtype))


# NOTE: Align precision with torch.compile
@torch.compile
def geglu_with_probs_ref(x: torch.Tensor, probs: torch.Tensor):
    dtype = x.dtype
    x = torch.chunk(x, 2, dim=-1)
    res = F.gelu(x[0]) * x[1]
    return (res * probs).to(dtype)


@pytest.mark.parametrize(
    "num_tokens",
    [
        1,
        128,
        2048,
        2025,
        8192 * 8,
    ],
)
@pytest.mark.parametrize(
    "hidden_size",
    [
        128,
        256,
        2048,
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("with_tokens_per_expert", [False, True])
def test_geglu_with_probs(num_tokens, hidden_size, dtype, with_tokens_per_expert):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"

    probs_dtype = torch.float32

    x = torch.randn(num_tokens, hidden_size * 2, device=device, dtype=dtype, requires_grad=True)
    probs = torch.rand(num_tokens, device=device, dtype=probs_dtype, requires_grad=True)

    x_ref = x.clone().detach()
    x_ref.requires_grad_()
    probs_ref = probs.clone().detach()
    probs_ref.requires_grad_()

    if with_tokens_per_expert:
        num_experts = 64
        tokens_per_expert = torch.tensor(
            generate_tokens_per_expert_list(num_experts, num_tokens), device=device, requires_grad=False
        )
        row_mask = torch.zeros(num_tokens, device=device, dtype=torch.bool, requires_grad=False)
        row_mask[: torch.sum(tokens_per_expert)] = 1
    else:
        row_mask = None

    out = geglu_with_probs(x, probs, row_mask)
    out_ref = geglu_with_probs_ref(x_ref, probs_ref.unsqueeze(-1))
    torch.testing.assert_close(out, out_ref, **get_tolerances(dtype))

    out.backward(torch.ones_like(out))
    grad_x = x.grad.clone()
    grad_probs = probs.grad.clone()

    out_ref.backward(torch.ones_like(out_ref))
    grad_x_ref = x_ref.grad.clone()
    grad_probs_ref = probs_ref.grad.clone()

    torch.testing.assert_close(grad_x, grad_x_ref, **get_tolerances(dtype))
    torch.testing.assert_close(grad_probs, grad_probs_ref, **get_tolerances(probs_dtype))
