import torch

from primus_turbo.triton.moe.fused_activation_with_probs import (
    fused_gelu_with_probs_fwd,
    fused_swiglu_with_probs_bwd,
    fused_swiglu_with_probs_fwd,
)


class FuseSwiGLUWithProbs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, probs: torch.Tensor, tokens_per_expert: torch.Tensor):
        assert tokens_per_expert.is_cuda

        out = fused_swiglu_with_probs_fwd(x, probs, tokens_per_expert)

        ctx.save_for_backward(x, probs, tokens_per_expert)

        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, probs, tokens_per_expert = ctx.saved_tensors
        grad_x, grad_probs = fused_swiglu_with_probs_bwd(grad_output, x, probs, tokens_per_expert)

        return grad_x, grad_probs, None


class FuseGeGLUWithProbs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, probs: torch.Tensor, tokens_per_expert: torch.Tensor):
        assert tokens_per_expert.is_cuda

        out = fused_gelu_with_probs_fwd(x, probs, tokens_per_expert)

        ctx.save_for_backward(x, probs, tokens_per_expert)

        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        pass
        # x, probs, tokens_per_expert = ctx.saved_tensors
        # grad_x, grad_probs = fused_gelu_with_probs_bwd(
        #     grad_output, x, probs, tokens_per_expert)

        # return grad_x, grad_probs, None


def fused_activation_with_probs(
    x: torch.Tensor, probs: torch.Tensor, tokens_per_expert: torch.Tensor, act_type: str
) -> torch.Tensor:
    if act_type == "swiglu":
        return FuseSwiGLUWithProbs.apply(x, probs, tokens_per_expert)
    elif act_type == "geglu":
        return FuseGeGLUWithProbs.apply(x, probs, tokens_per_expert)
    else:
        raise ValueError(f"Unsupported activation type: {act_type}")
