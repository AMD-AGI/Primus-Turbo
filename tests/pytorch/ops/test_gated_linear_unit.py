import random
from typing import Union

import pytest
import torch
import torch.nn.functional as F

from primus_turbo.pytorch.ops.gated_linear_unit import swiglu
from tests.test_utils import get_tolerances

torch.manual_seed(42)


def swiglu_ref(x: torch.Tensor, probs: torch.Tensor, tokens_per_expert: Union[torch.Tensor, None]):

    dtype = x.dtype
    if tokens_per_expert is None:
        x = torch.chunk(x, 2, dim=-1)
        return (F.silu(x[0]) * x[1] * probs).to(dtype)
    else:
        num_tokens = torch.sum(tokens_per_expert).item()
        probs = probs[0:num_tokens]
        x = x[0:num_tokens]
        x = torch.chunk(x, 2, dim=-1)
        return (F.silu(x[0]) * x[1] * probs).to(dtype)


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
    ],
)
@pytest.mark.parametrize(
    "hidden_size",
    [
        128,
        256,
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("with_tokens_per_expert", [True, False])
def test_swiglu(num_tokens, hidden_size, dtype, with_tokens_per_expert):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = "cuda"

    x = torch.randn(num_tokens, hidden_size * 2, device=device, dtype=dtype, requires_grad=True)
    probs = torch.rand(num_tokens, device=device, dtype=torch.float32, requires_grad=True)
    probs = probs.unsqueeze(-1)

    x_ref = x.clone().requires_grad_(True)
    probs_ref = probs.clone().requires_grad_(True)

    if with_tokens_per_expert:
        num_experts = 64
        tokens_per_expert = torch.tensor(
            generate_tokens_per_expert_list(num_experts, num_tokens), device=device
        )
    else:
        tokens_per_expert = None

    out = swiglu(x, probs, tokens_per_expert)
    out_ref = swiglu_ref(x_ref, probs_ref, tokens_per_expert)
    torch.testing.assert_close(out, out_ref, **get_tolerances(dtype))

    grad_out = torch.ones_like(out)
    out.backward(grad_out)
    grad_x = x.grad
    grad_probs = probs.grad

    x.grad = None
    probs.grad = None

    out_ref.backward(grad_out)
    grad_x_ref = x.grad
    grad_probs_ref = probs.grad

    torch.testing.assert_close(grad_x, grad_x_ref, **get_tolerances(dtype))
    torch.testing.assert_close(grad_probs, grad_probs_ref, **get_tolerances(dtype))
