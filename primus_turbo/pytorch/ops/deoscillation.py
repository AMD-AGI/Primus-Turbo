###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional

import torch

__all__ = ["weight_deosc_update", "weight_deosc_close"]


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
