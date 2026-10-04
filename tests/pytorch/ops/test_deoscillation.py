###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch

from primus_turbo.pytorch.ops.deoscillation import weight_deosc_close, weight_deosc_update


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP required")


def test_weight_deosc_update_matches_pytorch():
    generator = torch.Generator(device="cuda").manual_seed(17)
    current = torch.randn(8193, generator=generator, device="cuda", dtype=torch.bfloat16)
    current_qdq = torch.randn(8193, generator=generator, device="cuda", dtype=torch.bfloat16)
    previous = torch.randn(8193, generator=generator, device="cuda", dtype=torch.bfloat16)
    previous_qdq = torch.randn(8193, generator=generator, device="cuda", dtype=torch.bfloat16)
    dist = torch.rand(8193, generator=generator, device="cuda", dtype=torch.float32)
    dist_qdq = torch.rand(8193, generator=generator, device="cuda", dtype=torch.float32)

    expected_dist = dist + (current - previous).abs()
    expected_dist_qdq = dist_qdq + (current_qdq - previous_qdq).abs()

    weight_deosc_update(current, current_qdq, previous, previous_qdq, dist, dist_qdq)

    torch.testing.assert_close(dist, expected_dist, rtol=0, atol=0)
    torch.testing.assert_close(dist_qdq, expected_dist_qdq, rtol=0, atol=0)


@pytest.mark.parametrize("collect_count", [False, True])
def test_weight_deosc_close_matches_pytorch(collect_count):
    master = torch.tensor([0.2, 1.2, -0.7, 9.0, 4.0], device="cuda", dtype=torch.float32)
    previous = master.to(torch.bfloat16)
    current_qdq = torch.tensor([0.5, 1.0, -1.0, 8.0, 3.0], device="cuda", dtype=torch.bfloat16)
    dist = torch.tensor([0.1, 0.0, 0.5, 1.0, float("nan")], device="cuda")
    dist_qdq = torch.tensor([0.5, 9.0, 1.5, 4.0, 8.0], device="cuda")
    ratio_threshold = 4.0
    eps = 1.0e-12

    mask = (dist > 0) & (dist_qdq / dist.clamp(min=eps) >= ratio_threshold)
    expected_master = torch.where(mask, current_qdq, master)
    expected_previous = torch.where(mask, current_qdq, previous)
    expected_count = mask.sum()

    count = torch.zeros((), device="cuda", dtype=torch.int64) if collect_count else None
    weight_deosc_close(
        master,
        previous,
        current_qdq,
        dist,
        dist_qdq,
        ratio_threshold,
        eps,
        count,
    )

    torch.testing.assert_close(master, expected_master, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(previous, expected_previous, rtol=0, atol=0, equal_nan=True)
    assert torch.count_nonzero(dist).item() == 0
    assert torch.count_nonzero(dist_qdq).item() == 0
    if collect_count:
        assert count.dtype == torch.int64 and count.shape == torch.Size([])
        assert count.item() == expected_count.item()


def test_weight_deosc_rejects_non_contiguous_input():
    current = torch.zeros((4, 4), device="cuda", dtype=torch.bfloat16).t()
    contiguous = torch.zeros(16, device="cuda", dtype=torch.bfloat16)
    dist = torch.zeros(16, device="cuda")
    with pytest.raises(RuntimeError, match="current must be contiguous"):
        weight_deosc_update(current, contiguous, contiguous, contiguous, dist, dist.clone())
