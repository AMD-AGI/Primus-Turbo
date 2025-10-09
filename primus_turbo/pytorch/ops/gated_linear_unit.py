from typing import Union

import torch

from primus_turbo.pytorch.kernels.gated_linear_unit.swiglu import (
    swiglu_bwd,
    swiglu_bwd_with_tokens_per_expert,
    swiglu_fwd,
    swiglu_fwd_with_tokens_per_expert,
)


class SwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, probs: torch.Tensor, tokens_per_expert: Union[torch.Tensor, None]):
        assert x.size(0) == probs.size(0), "first dimension of x and probs must be the same"
        assert x.ndim == 2, "x must be 2D tensor"
        assert probs.ndim == 1, "probs must be 1D tensor"
        assert probs.dtype == torch.float32, "probs must be float32"

        if tokens_per_expert is not None:
            assert tokens_per_expert.is_cuda, "tokens_per_expert must be a CUDA tensor"
            out = swiglu_fwd_with_tokens_per_expert(x, probs, tokens_per_expert)
        else:
            out = swiglu_fwd(
                x,
                probs,
            )

        ctx.save_for_backward(x, probs, tokens_per_expert)

        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        assert grad_output.ndim == 2

        x, probs, tokens_per_expert = ctx.saved_tensors

        if tokens_per_expert is not None:
            grad_x, grad_probs = swiglu_bwd_with_tokens_per_expert(grad_output, x, probs, tokens_per_expert)
        else:
            grad_x, grad_probs = swiglu_bwd(grad_output, x, probs)

        return grad_x, grad_probs, None


def swiglu(
    x: torch.Tensor, probs: torch.Tensor, tokens_per_expert: Union[torch.Tensor, None]
) -> torch.Tensor:
    return SwiGLU.apply(x, probs, tokens_per_expert)
