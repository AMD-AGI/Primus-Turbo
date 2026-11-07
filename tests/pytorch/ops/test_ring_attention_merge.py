###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Stable attention partition merging, including selective query updates."""

import pytest
import torch

from primus_turbo.pytorch.ops.attention.usp.attention_ring import update_out_and_lse


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("offset", [-1000.0, 0.0, 1000.0])
def test_partition_merge_matches_joint_softmax(partial, offset):
    torch.manual_seed(821)
    scores = torch.randn(2, 3, 5, 16, device="cuda")
    scores[..., 8:] += offset
    values = torch.randn(2, 3, 16, 7, device="cuda")
    oa = (scores[..., :8].softmax(-1) @ values[..., :8, :]).transpose(1, 2)
    ob = (scores[..., 8:].softmax(-1) @ values[..., 8:, :]).transpose(1, 2)
    la, lb = scores[..., :8].logsumexp(-1), scores[..., 8:].logsumexp(-1)
    out, lse = update_out_and_lse(None, None, oa, la)
    original = out.clone()
    sl = (slice(None), slice(1, 4)) if partial else None
    out, lse = update_out_and_lse(
        out, lse, ob[sl] if partial else ob, lb[:, :, 1:4] if partial else lb, slice_=sl
    )
    expected = (scores.softmax(-1) @ values).transpose(1, 2)
    expected_lse = scores.logsumexp(-1).transpose(1, 2).unsqueeze(-1)
    if partial:
        torch.testing.assert_close(out[:, [0, 4]], original[:, [0, 4]])
        out, expected = out[sl], expected[sl]
        lse, expected_lse = lse[sl], expected_lse[sl]
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(lse, expected_lse)


def test_first_partition_cannot_update_a_slice():
    out = torch.zeros(1, 2, 3, 4)
    with pytest.raises(RuntimeError, match="first update"):
        update_out_and_lse(None, None, out, torch.zeros(1, 3, 2), slice_=slice(1))
