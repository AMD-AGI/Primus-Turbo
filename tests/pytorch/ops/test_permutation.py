###############################################################################
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


import random

import pytest
import torch

import primus_turbo.pytorch as turbo
from tests.pytorch.ref.permuatation_ref import (
    pytorch_permute_mask_map,
    pytorch_unpermute_mask_map,
)
from tests.test_utils import get_tolerances

# TODO: add FP8 test


def manual_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed = seed


def _test_permutation_mask_map(
    dtype: torch.dtype,
    num_tokens,
    num_expert,
    hidden_size,
    topK,
    num_out_tokens,
    with_probs,
):
    if topK > num_expert:
        pytest.skip("topK should be smaller than the number of experts.")

    if num_out_tokens == None:
        num_out_tokens = num_tokens * topK

    manual_seed(1234)
    print(
        "mask map:" f" token:{num_tokens} hidden_size:{hidden_size} expert:{num_expert} topK:{topK} {dtype}"
    )

    pytorch_permute_fwd_input = torch.rand((num_tokens, hidden_size), dtype=dtype).cuda()
    pytorch_permute_bwd_input = torch.rand((num_out_tokens, hidden_size), dtype=dtype).cuda()
    pytorch_unpermute_bwd_input = torch.rand((num_tokens, hidden_size), dtype=dtype).cuda()

    pytorch_permute_fwd_input.requires_grad_(True)

    restore_shape = pytorch_permute_fwd_input.shape

    _tmp_tensor = torch.zeros((num_tokens * num_expert,))
    _tmp_tensor[: int(num_out_tokens)] = 1.0
    _tmp_idx = torch.randperm(num_tokens * num_expert)
    routing_map = torch.reshape(_tmp_tensor[_tmp_idx], (num_tokens, num_expert)).bool().cuda()

    probs = None
    if with_probs:
        probs = torch.rand(num_tokens, num_expert).cuda() * routing_map
        row_sums = probs.sum(dim=1, keepdim=True)
        probs = probs / row_sums
        probs = probs.to(dtype)
        probs.requires_grad_(True)

    ###################################################################################################################################
    #
    # PyTorch Permutation
    #
    ###################################################################################################################################
    pytorch_permute_output, sorted_indices = pytorch_permute_mask_map(pytorch_permute_fwd_input, routing_map)
    pytorch_permute_output.backward(pytorch_permute_bwd_input, retain_graph=True)

    pytorch_unpermute_fwd_input = pytorch_permute_output.detach()
    pytorch_unpermute_fwd_input.requires_grad_(True)

    pytorch_unpermute_output = pytorch_unpermute_mask_map(
        pytorch_unpermute_fwd_input, sorted_indices, restore_shape, probs, routing_map
    )
    pytorch_unpermute_output.backward(pytorch_unpermute_bwd_input, retain_graph=True)

    ###################################################################################################################################
    #
    # Turbo Permutation
    #
    ###################################################################################################################################
    turbo_permute_fwd_input = pytorch_permute_fwd_input.detach()
    turbo_permute_fwd_input.requires_grad_(True)
    turbo_permute_bwd_input = pytorch_permute_bwd_input.detach()

    turbo_permute_output, _, row_id_map = turbo.ops.token_permute(
        turbo_permute_fwd_input, routing_map=routing_map, num_out_tokens=num_out_tokens
    )
    turbo_permute_output.backward(turbo_permute_bwd_input, retain_graph=True)

    turbo_probs = None
    if with_probs:
        turbo_probs = probs.detach()
        turbo_probs.requires_grad_(True)
    turbo_unpermute_fwd_input = turbo_permute_output.detach()
    turbo_unpermute_fwd_input.requires_grad_(True)
    turbo_unpermute_bwd_input = pytorch_unpermute_bwd_input.detach()

    turbo_unpermute_output = turbo.ops.token_unpermute(
        turbo_unpermute_fwd_input, row_id_map, turbo_probs, restore_shape
    )
    turbo_unpermute_output.backward(turbo_unpermute_bwd_input, retain_graph=True)

    ###################################################################################################################################
    #
    # Results Check
    #
    ###################################################################################################################################
    # TODO: can't pass turbo tolerances
    tol = get_tolerances(dtype)
    torch.testing.assert_close(
        pytorch_permute_output,
        turbo_permute_output,
        msg=f"Mismatch in turbo_permute fwd",
        **tol,
    )
    torch.testing.assert_close(
        pytorch_permute_fwd_input.grad,
        turbo_permute_fwd_input.grad,
        **tol,
    )
    torch.testing.assert_close(
        pytorch_unpermute_output,
        turbo_unpermute_output,
        msg=f"Mismatch in turbo_unpermute fwd",
        **tol,
    )
    torch.testing.assert_close(
        pytorch_unpermute_fwd_input.grad,
        turbo_unpermute_fwd_input.grad,
        msg=f"Mismatch in turbo_unpermute bwd",
        **tol,
    )
    if with_probs:
        torch.testing.assert_close(
            probs.grad,
            turbo_probs.grad,
            msg=f"Mismatch in turbo_unpermute bwd",
            **tol,
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("num_tokens", [4096])
@pytest.mark.parametrize("num_expert", [7, 16])
@pytest.mark.parametrize("hidden_size", [4096])
@pytest.mark.parametrize("topK", [1, 2, 5])
@pytest.mark.parametrize("num_out_tokens", [None, 2039])
def test_permutation_mask_map(
    dtype,
    num_tokens,
    num_expert,
    hidden_size,
    topK,
    num_out_tokens,
):
    with_probs = True

    _test_permutation_mask_map(
        dtype=dtype,
        num_tokens=num_tokens,
        num_expert=num_expert,
        hidden_size=hidden_size,
        topK=topK,
        num_out_tokens=num_out_tokens,
        with_probs=with_probs,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_permutation_mask_map_empty_input(dtype):
    with_probs = True
    _test_permutation_mask_map(
        dtype=dtype,
        num_tokens=0,
        num_expert=8,
        hidden_size=4096,
        topK=2,
        num_out_tokens=0,
        with_probs=with_probs,
    )
