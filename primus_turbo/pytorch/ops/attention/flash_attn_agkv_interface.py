###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional

import torch
import torch.distributed as dist

from primus_turbo.pytorch.ops.attention.usp.attention_agkv import attention_agkv

__all__ = ["flash_attn_agkv_func"]


def flash_attn_agkv_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cp_group: dist.ProcessGroup,
    softmax_scale: Optional[float] = None,
    mode: str = "overlap",
    deterministic: bool = False,
) -> torch.Tensor:
    """Causal context-parallel attention that all-gathers K / V (the Llama 3 scheme).

    Each rank holds two zigzag chunks of the sequence, chunk r then chunk 2n-1-r of 2n (the order of
    Megatron's ``get_batch_on_this_cp_rank``), and all heads. q [s_local, b, hq, d], k / v
    [s_local, b, hkv, d], sbhd, bf16, d in (64, 128), gfx950 FlyDSL kernels.

    ``mode``: ``"overlap"`` (default) overlaps the K / V exchange with attention, ``"serial"`` uses the
    same copy-free layout without overlap, ``"naive"`` is the plain gather / reorder / scatter baseline.
    """
    return attention_agkv(
        q, k, v, cp_group, mode=mode, softmax_scale=softmax_scale, deterministic=deterministic
    )
