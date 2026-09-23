###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Python-side wrappers that launch the FlyDSL fused QKV RoPE kernels."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch

from primus_turbo.flydsl.rope.rope_kernel import (
    ROPE_HEAD_DIM,
    ROPE_ROW_GROUP,
    flydsl_qkv_rope_backward,
    flydsl_qkv_rope_forward,
)


def rope_shape_error(qkv, q_freqs, k_freqs, qkv_split_arg_list) -> str | None:
    """The reason this shape is unsupported, or None when the kernels accept it."""
    if qkv.ndim != 4:
        return f"qkv must be 4-D [S, B, H, q+k+v], got {tuple(qkv.shape)}"
    if qkv.dtype != torch.bfloat16:
        return f"qkv must be bfloat16, got {qkv.dtype}"
    if not qkv.is_contiguous():
        return "qkv must be contiguous"
    if len(qkv_split_arg_list) != 3:
        return f"qkv_split_arg_list must be [q, k, v], got {list(qkv_split_arg_list)}"
    q_size, k_size, v_size = qkv_split_arg_list
    if k_size != ROPE_HEAD_DIM or v_size != ROPE_HEAD_DIM or q_size % ROPE_HEAD_DIM:
        return f"k and v must be {ROPE_HEAD_DIM} and q a multiple of it, got {list(qkv_split_arg_list)}"
    if qkv.shape[-1] != q_size + k_size + v_size:
        return f"qkv last dim {qkv.shape[-1]} != q+k+v {q_size + k_size + v_size}"
    if q_freqs.shape[-1] != ROPE_HEAD_DIM or k_freqs.shape[-1] != ROPE_HEAD_DIM:
        return f"freqs last dim must be {ROPE_HEAD_DIM}"
    # No per-row predicate, and the descriptors span all of memory: a partial last
    # row group would read and write out of bounds rather than be clamped.
    rows = qkv.shape[0] * qkv.shape[1]
    if rows % ROPE_ROW_GROUP:
        return f"S*B must be a multiple of {ROPE_ROW_GROUP}, got {rows}"
    return None


def rope_fwd_impl(
    qkv: torch.Tensor,
    q_freqs: torch.Tensor,
    k_freqs: torch.Tensor,
    qkv_split_arg_list: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rotate the q and k halves of a packed QKV tensor; v is copied through."""
    why = rope_shape_error(qkv, q_freqs, k_freqs, qkv_split_arg_list)
    if why is not None:
        raise ValueError(f"fused_qkv_rope: unsupported input ({why})")
    return flydsl_qkv_rope_forward(qkv, q_freqs, k_freqs, qkv_split_arg_list)


def rope_bwd_impl(
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    q_freqs: torch.Tensor,
    k_freqs: torch.Tensor,
    qkv_split_arg_list: Sequence[int],
) -> torch.Tensor:
    """Packed dQKV for :func:`rope_fwd_impl`: the forward rotation transposed."""
    return flydsl_qkv_rope_backward(
        dq.contiguous(), dk.contiguous(), dv.contiguous(), q_freqs, k_freqs, qkv_split_arg_list
    )
