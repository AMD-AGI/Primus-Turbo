###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional

import torch

__all__ = ["weight_deosc_update", "weight_deosc_close", "weight_deosc_qdq"]


def weight_deosc_qdq(
    master: torch.Tensor,
    previous: torch.Tensor,
    previous_qdq: torch.Tensor,
    dist: torch.Tensor,
    dist_qdq: torch.Tensor,
    rows: int,
    cols: int,
    start: int,
    *,
    scale_rounding_mode: int = 0,
    seed: bool = False,
    close: bool = False,
    ratio_threshold: float = 4.0,
    eps: float = 1e-12,
    reset_count: Optional[torch.Tensor] = None,
    grouped: bool = False,
) -> None:
    """Fuse local MXFP4 QDQ, tracking, snapshots and optional closure (gfx950).

    ``master`` is a contiguous flattened FP32 shard starting at ``start`` in
    one matrix or a flattened batch of ``[rows, cols]`` matrices. Each 32x32
    tile uses BF16-rounded input and zeroes for elements outside the shard.
    Persistent state tensors have the same number of elements as ``master``.
    ``seed=True`` initializes them without observing movement. No packed FP4,
    scale, transpose, BF16 input or dequantized temporary is materialized.
    ``cols`` must be divisible by 32. ``grouped=True`` reproduces the batched
    forward quantizer's row-then-tile NaN reduction; finite inputs are identical.
    """
    torch.ops.primus_turbo_cpp_extension.weight_deosc_qdq(
        master,
        previous,
        previous_qdq,
        dist,
        dist_qdq,
        rows,
        cols,
        start,
        scale_rounding_mode,
        seed,
        close,
        ratio_threshold,
        eps,
        reset_count,
        grouped,
    )


def weight_deosc_update(
    current: torch.Tensor,
    current_qdq: torch.Tensor,
    previous: torch.Tensor,
    previous_qdq: torch.Tensor,
    dist: torch.Tensor,
    dist_qdq: torch.Tensor,
) -> None:
    """Accumulate BF16 and QDQ movement into two FP32 distance tensors."""
    torch.ops.primus_turbo_cpp_extension.weight_deosc_update(
        current, current_qdq, previous, previous_qdq, dist, dist_qdq
    )


def weight_deosc_close(
    master: torch.Tensor,
    previous: torch.Tensor,
    current_qdq: torch.Tensor,
    dist: torch.Tensor,
    dist_qdq: torch.Tensor,
    ratio_threshold: float,
    eps: float,
    reset_count: Optional[torch.Tensor] = None,
) -> None:
    """Snap oscillating weights and clear the period accumulators.

    When supplied, ``reset_count`` is a device int64 scalar shared by every
    shard in a period and is incremented in place.
    """
    torch.ops.primus_turbo_cpp_extension.weight_deosc_close(
        master,
        previous,
        current_qdq,
        dist,
        dist_qdq,
        ratio_threshold,
        eps,
        reset_count,
    )
