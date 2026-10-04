###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch

from primus_turbo.pytorch.ops.deoscillation import (
    weight_deosc_close,
    weight_deosc_qdq,
    weight_deosc_update,
)


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


def _reference_qdq(master, rows, cols, start, mode):
    """Independent staging + existing forward quantizer, including split tiles."""
    from primus_turbo.pytorch.core import QuantizedTensor
    from primus_turbo.pytorch.core.low_precision import (
        ScalingGranularity,
        ScalingRecipe,
        float4_e2m1fn_x2,
    )

    result = torch.empty_like(master, dtype=torch.bfloat16)
    end = start + master.numel()
    matrix_size = rows * cols
    for matrix in range(start // matrix_size, (end - 1) // matrix_size + 1):
        begin = max(start, matrix * matrix_size)
        stop = min(end, (matrix + 1) * matrix_size)
        tile_begin = ((begin - matrix * matrix_size) // cols // 32) * 32 * cols
        tile_end = (((stop - matrix * matrix_size - 1) // cols // 32) + 1) * 32 * cols
        tile = torch.zeros(
            ((tile_end - tile_begin) // cols, cols), dtype=torch.bfloat16, device=master.device
        )
        left = begin - matrix * matrix_size - tile_begin
        right = stop - matrix * matrix_size - tile_begin
        tile.view(-1)[left:right].copy_(master[begin - start : stop - start])
        qt = QuantizedTensor.quantize(
            tile,
            dest_dtype=float4_e2m1fn_x2,
            granularity=ScalingGranularity.MX_BLOCKWISE,
            block_size=32,
            scaling_recipe=ScalingRecipe(use_2d_block=True),
            axis=-1,
            scale_rounding_mode=mode,
        )
        qdq = qt.dequantize()[: tile.shape[0], :cols].contiguous().view(-1)
        result[begin - start : stop - start].copy_(qdq[left:right])
    return result


def _state_like(master):
    return (
        torch.empty_like(master, dtype=torch.bfloat16),
        torch.empty_like(master, dtype=torch.bfloat16),
        torch.empty_like(master),
        torch.empty_like(master),
    )


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize(
    "rows,cols,start,n",
    [
        (64, 64, 0, 4096),
        (96, 160, 17, 19013),  # Odd start, split matrix and end.
        (64, 64, 4093, 9),  # Straddles experts with almost entirely missing tiles.
        (64, 64, 37, 1),
        (37, 64, 17, 37 * 64 * 2 - 20),
        (64, 64, 2, 8192),
        (2880, 2880, 65537, 2880 * 64 + 17),
        (5760, 2880, 5760 * 2880 - 17, 2880 * 96 + 31),
    ],
)
def test_direct_qdq_seed_matches_forward(rows, cols, start, n, mode):
    gen = torch.Generator(device="cuda").manual_seed(101)
    master = torch.randn(n, device="cuda", generator=gen) * 0.037
    if n > 1024:
        master[:1024] = 0
    previous, previous_qdq, dist, dist_qdq = _state_like(master)
    expected = _reference_qdq(master, rows, cols, start, mode)
    weight_deosc_qdq(
        master, previous, previous_qdq, dist, dist_qdq, rows, cols, start, scale_rounding_mode=mode, seed=True
    )
    torch.testing.assert_close(previous, master.bfloat16(), rtol=0, atol=0)
    torch.testing.assert_close(previous_qdq, expected, rtol=0, atol=0)
    assert torch.count_nonzero(dist).item() == 0
    assert torch.count_nonzero(dist_qdq).item() == 0


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("count_enabled", [False, True])
@pytest.mark.parametrize("start,n", [(31, 8193), (0, 8192), (2, 8192)])
def test_direct_qdq_multiple_windows_match_reference(mode, count_enabled, start, n):
    gen = torch.Generator(device="cuda").manual_seed(202)
    master = torch.randn(n, device="cuda", generator=gen) * 0.01
    rows, cols = 64, 64
    state = _state_like(master)
    reference_master = master.clone()
    reference_prev = master.bfloat16()
    reference_q = _reference_qdq(master, rows, cols, start, mode)
    distance = torch.zeros_like(master)
    distance_q = torch.zeros_like(master)
    weight_deosc_qdq(master, *state, rows, cols, start, scale_rounding_mode=mode, seed=True)
    count = torch.zeros((), dtype=torch.int64, device="cuda") if count_enabled else None
    expected_count = torch.zeros((), dtype=torch.int64, device="cuda")
    for step in range(1, 10):
        delta = torch.randn(master.shape, device="cuda", generator=gen) * 0.0001
        master.add_(delta)
        reference_master.add_(delta)
        current = reference_master.bfloat16()
        qdq = _reference_qdq(reference_master, rows, cols, start, mode)
        distance.add_((current - reference_prev).abs())
        distance_q.add_((qdq - reference_q).abs())
        close = step % 3 == 0
        reference_prev = current
        reference_q = qdq
        if close:
            mask = (distance > 0) & (distance_q / distance.clamp(min=1e-12) >= 4.0)
            expected_count.add_(mask.sum())
            reference_master = torch.where(mask, qdq, reference_master)
            reference_prev = torch.where(mask, qdq, current)
            distance.zero_()
            distance_q.zero_()
        weight_deosc_qdq(
            master, *state, rows, cols, start, scale_rounding_mode=mode, close=close, reset_count=count
        )
        for actual, expected in zip(
            (master, *state),
            (
                reference_master,
                reference_prev,
                reference_q,
                distance,
                distance_q,
            ),
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if count_enabled:
        assert count.item() == expected_count.item()


def test_direct_qdq_empty_and_non_default_stream():
    master = torch.empty(0, device="cuda")
    weight_deosc_qdq(master, *_state_like(master), 32, 32, 0, seed=True)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        master = torch.ones(1025, device="cuda")
        state = _state_like(master)
        weight_deosc_qdq(master, *state, 64, 64, 7, seed=True)
    stream.synchronize()
    torch.testing.assert_close(state[0], master.bfloat16(), rtol=0, atol=0)
    torch.testing.assert_close(state[1], master.bfloat16(), rtol=0, atol=0)


def test_direct_qdq_rejects_overlapping_state():
    master = torch.ones(1024, device="cuda")
    state = _state_like(master)
    with pytest.raises(RuntimeError):
        weight_deosc_qdq(master, state[0], state[0], state[2], state[3], 32, 32, 0, seed=True)


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("value", [0.0, 1e-35, 1e-20, 1e20, float("inf"), float("nan")])
def test_direct_qdq_special_scales(mode, value):
    master = torch.full((1024,), value, device="cuda")
    master[1::2].neg_()
    state = _state_like(master)
    expected = _reference_qdq(master, 32, 32, 0, mode)
    weight_deosc_qdq(master, *state, 32, 32, 0, seed=True, scale_rounding_mode=mode)
    torch.testing.assert_close(state[0], master.bfloat16(), rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(state[1], expected, rtol=0, atol=0, equal_nan=True)


def test_direct_qdq_unaligned_contiguous_views():
    # Tensor contiguity does not imply vector-load alignment.
    master = torch.randn(8194, device="cuda")[1:-1]
    previous = torch.empty(8194, device="cuda", dtype=torch.bfloat16)[1:-1]
    previous_q = torch.empty_like(previous)
    distance = torch.empty(8194, device="cuda")[1:-1]
    distance_q = torch.empty_like(distance)
    expected = _reference_qdq(master, 64, 64, 0, 2)
    weight_deosc_qdq(
        master, previous, previous_q, distance, distance_q, 64, 64, 0, seed=True, scale_rounding_mode=2
    )
    torch.testing.assert_close(previous_q, expected, rtol=0, atol=0)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two GPUs required")
def test_direct_qdq_uses_tensor_device():
    with torch.cuda.device(0):
        master = torch.ones(1024, device="cuda:1")
        state = _state_like(master)
        weight_deosc_qdq(master, *state, 32, 32, 0, seed=True)
        torch.testing.assert_close(state[1], master.bfloat16(), rtol=0, atol=0)


def test_direct_qdq_rejects_unaligned_columns():
    master = torch.ones(1024, device="cuda")
    with pytest.raises(RuntimeError, match="cols must be divisible by 32"):
        weight_deosc_qdq(master, *_state_like(master), 32, 61, 0, seed=True)
