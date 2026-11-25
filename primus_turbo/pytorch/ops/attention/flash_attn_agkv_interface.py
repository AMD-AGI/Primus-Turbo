###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional

import torch
import torch.distributed as dist

from primus_turbo.pytorch.ops.attention.agkv import attention_agkv

__all__ = ["flash_attn_agkv_func"]


def flash_attn_agkv_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cp_group: dist.ProcessGroup,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Causal context-parallel attention that all-gathers K / V (the Llama 3 scheme), fused.

    Each rank holds two zigzag chunks of the sequence, chunk r then chunk 2n-1-r of 2n (the order of
    Megatron's ``get_batch_on_this_cp_rank``), and all heads. q [s_local, b, hq, d], k / v
    [s_local, b, hkv, d], sbhd, bf16, d in (64, 128), an even number of q heads per kv head, chunks of a
    multiple of 128 tokens (at least 256), the default softmax scale; gfx950 FlyDSL kernels, single node.

    The forward and the backward are one persistent kernel each with the K / V exchange (XGMI stores into
    symmetric buffers) inside it. Run with ``GPU_MAX_HW_QUEUES>=8`` beside overlapped collectives: a kernel
    waiting on its peers' kernels deadlocks when an RCCL kernel shares its hardware queue.
    """
    return attention_agkv(q, k, v, cp_group, softmax_scale=softmax_scale)
