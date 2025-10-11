import torch

from primus_turbo.pytorch.kernels.moe.tokens_per_expert_to_mask_impl import (
    tokens_per_expert_to_mask_impl,
)

__all__ = ["tokens_per_expert_to_mask"]


class TokensPerExpertToMask(torch.autograd.Function):

    @classmethod
    def forward(ctx, tokens_per_expert: torch.Tensor, num_tokens: int):
        assert tokens_per_expert.is_cuda, "tokens_per_expert must be a CUDA tensor."
        assert tokens_per_expert.ndim == 1, "tokens_per_expert must be a 1D tensor."

        out = tokens_per_expert_to_mask_impl(tokens_per_expert, num_tokens)

        return out


def tokens_per_expert_to_mask(tokens_per_expert: torch.Tensor, num_tokens: int):
    return TokensPerExpertToMask.forward(tokens_per_expert, num_tokens)
