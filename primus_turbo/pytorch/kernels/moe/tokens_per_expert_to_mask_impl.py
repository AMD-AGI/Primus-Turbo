import torch
import triton

from primus_turbo.triton.moe.tokens_per_expert_to_mask_kernel import (
    tokens_per_expert_to_mask_kernel,
)


def tokens_per_expert_to_mask_impl(tokens_per_expert: torch.Tensor, num_tokens: int) -> torch.Tensor:
    num_expert = tokens_per_expert.size(0)

    mask = torch.zeros(num_tokens, dtype=torch.int64, device=tokens_per_expert.device)

    grid = lambda meta: (triton.cdiv(num_tokens, meta["BLOCK_SIZE"]),)
    tokens_per_expert_to_mask_kernel[grid](
        tokens_per_expert,
        mask,
        num_expert=num_expert,
        TOKENS_PER_EXPERT_LOAD_WIDTH=triton.next_power_of_2(num_expert),
        BLOCK_SIZE=256,
    )

    return mask
