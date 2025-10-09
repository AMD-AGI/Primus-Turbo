from typing import Union

import torch

from primus_turbo.pytorch.kernels.gated_linear_unit.geglu import (
    geglu_bwd_with_tokens_per_expert,
    geglu_fwd_with_tokens_per_expert,
)
from primus_turbo.pytorch.kernels.gated_linear_unit.swiglu import (
    swiglu_bwd_with_tokens_per_expert,
    swiglu_fwd_with_tokens_per_expert,
)


class GLU(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor, probs: torch.Tensor, tokens_per_expert: Union[torch.Tensor, None], act_type: str
    ):
        assert x.size(0) == probs.size(0), "first dimension of x and probs must be the same"
        assert x.ndim == 2, "x must be 2D tensor"
        assert probs.ndim == 1, "probs must be 1D tensor"
        assert probs.dtype == torch.float32, "probs must be float32"

        SUPPORTED_ACT_TYPES = ["silu", "gelu"]
        assert (
            act_type in SUPPORTED_ACT_TYPES
        ), f"Unsupported act_type: {act_type}. Supported types: {SUPPORTED_ACT_TYPES}"

        if tokens_per_expert is not None:
            assert tokens_per_expert.is_cuda, "tokens_per_expert must be a CUDA tensor"

        if act_type == "silu":
            out = swiglu_fwd_with_tokens_per_expert(x, probs, tokens_per_expert)
        elif act_type == "gelu":
            out = geglu_fwd_with_tokens_per_expert(x, probs, tokens_per_expert)

        ctx.save_for_backward(x, probs, tokens_per_expert)
        ctx.act_type = act_type

        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        assert grad_output.ndim == 2

        x, probs, tokens_per_expert = ctx.saved_tensors

        if ctx.act_type == "silu":
            grad_x, grad_probs = swiglu_bwd_with_tokens_per_expert(grad_output, x, probs, tokens_per_expert)
        elif ctx.act_type == "gelu":
            grad_x, grad_probs = geglu_bwd_with_tokens_per_expert(grad_output, x, probs, tokens_per_expert)

        return grad_x, grad_probs, None, None


def swiglu(
    x: torch.Tensor, probs: torch.Tensor, tokens_per_expert: Union[torch.Tensor, None]
) -> torch.Tensor:
    return GLU.apply(x, probs, tokens_per_expert, "silu")


def geglu(x: torch.Tensor, probs: torch.Tensor, tokens_per_expert: Union[torch.Tensor, None]) -> torch.Tensor:
    return GLU.apply(x, probs, tokens_per_expert, "gelu")
