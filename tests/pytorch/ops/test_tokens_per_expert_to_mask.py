###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import random

import pytest
import torch

from primus_turbo.pytorch.ops.tokens_per_expert_to_mask import tokens_per_expert_to_mask

torch.manual_seed(42)
random.seed(42)


def tokens_per_expert_to_mask_ref(tokens_per_expert, num_tokens):
    row_mask = torch.zeros(
        num_tokens, device=tokens_per_expert.device, dtype=torch.int64, requires_grad=False
    )
    row_mask[: torch.sum(tokens_per_expert)] = 1

    return row_mask


def generate_tokens_per_expert_list(num_experts: int, num_tokens: int):
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


@pytest.mark.parametrize("num_experts", [1, 2, 8, 64])
@pytest.mark.parametrize("num_tokens", [1, 128, 2048, 2025])
def test_tokens_per_expert_to_mask(num_experts, num_tokens):
    device = "cuda"

    tokens_per_expert = torch.tensor(
        generate_tokens_per_expert_list(num_experts, num_tokens), device=device, requires_grad=False
    )

    random_padding = random.randint(0, num_tokens)
    out = tokens_per_expert_to_mask(tokens_per_expert, num_tokens + random_padding)
    out_ref = tokens_per_expert_to_mask_ref(tokens_per_expert, num_tokens + random_padding)

    torch.testing.assert_close(out, out_ref, rtol=0, atol=0)
