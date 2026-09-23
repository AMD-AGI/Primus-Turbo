###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FlyDSL-backed fused QKV rotary position embedding.

Public API:
    - ``fused_qkv_rope(qkv, q_freqs, k_freqs, qkv_split_arg_list) -> (q, k, v)``
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch

from primus_turbo.pytorch.kernels.rope.rope_impl import (
    rope_bwd_impl,
    rope_fwd_impl,
    rope_shape_error,
)

__all__ = ["fused_qkv_rope"]


class _FusedQKVRoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, q_freqs, k_freqs, qkv_split_arg_list):
        # The kernels index the tables as fp32 and stride them contiguously.
        q_freqs = q_freqs.float().contiguous()
        k_freqs = k_freqs.float().contiguous()
        q, k, v = rope_fwd_impl(qkv, q_freqs, k_freqs, qkv_split_arg_list)
        ctx.save_for_backward(q_freqs, k_freqs)
        ctx.qkv_split_arg_list = qkv_split_arg_list
        return q, k, v

    @staticmethod
    def backward(ctx, dq, dk, dv):
        q_freqs, k_freqs = ctx.saved_tensors
        dqkv = rope_bwd_impl(dq, dk, dv, q_freqs, k_freqs, ctx.qkv_split_arg_list)
        return dqkv, None, None, None


def fused_qkv_rope(
    qkv: torch.Tensor,
    q_freqs: torch.Tensor,
    k_freqs: torch.Tensor,
    qkv_split_arg_list: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rotary position embedding over a packed QKV tensor, split on the way out.

    Args:
        qkv: ``[S, B, H, q + k + v]`` bfloat16, contiguous. ``S * B`` must be a
            multiple of the kernels' row group.
        q_freqs: ``[S, 1, 1, D]`` rotation angles for the q heads.
        k_freqs: ``[S, 1, 1, D]`` rotation angles for the k heads.
        qkv_split_arg_list: ``[q, k, v]`` widths of the last dim; k and v are one
            head each, q a whole multiple of it.

    Returns:
        ``(q, k, v)``, q and k rotated and v a plain copy.

    Raises:
        ValueError: if the tensors are not a shape the kernels accept.
    """
    why = rope_shape_error(qkv, q_freqs, k_freqs, qkv_split_arg_list)
    if why is not None:
        raise ValueError(f"fused_qkv_rope: unsupported input ({why})")
    return _FusedQKVRoPEFunction.apply(qkv, q_freqs, k_freqs, qkv_split_arg_list)
