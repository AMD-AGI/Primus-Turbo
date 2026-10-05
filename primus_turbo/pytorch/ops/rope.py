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

from primus_turbo.pytorch.kernels.rope.qk_rmsnorm_rope_impl import (
    qk_rmsnorm_rope_bwd_impl,
    qk_rmsnorm_rope_fwd_impl,
    qk_rmsnorm_rope_shape_error,
)
from primus_turbo.pytorch.kernels.rope.rope_impl import (
    rope_bwd_impl,
    rope_fwd_impl,
    rope_shape_error,
)

__all__ = ["fused_qkv_rope", "fused_qkv_rmsnorm_rope"]


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


class _FusedQKVRMSNormRoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, q_gamma, k_gamma, freqs, qkv_split_arg_list, eps, return_scaled_q=False):
        freqs = freqs.float().contiguous()
        outputs = qk_rmsnorm_rope_fwd_impl(
            qkv, q_gamma, k_gamma, freqs, qkv_split_arg_list, eps, return_scaled_q=return_scaled_q
        )
        q, k, v, q_rstd, k_rstd = outputs[:5]
        ctx.save_for_backward(qkv, q_gamma, k_gamma, freqs, q_rstd, k_rstd)
        ctx.qkv_split_arg_list = tuple(qkv_split_arg_list)
        if return_scaled_q:
            # Auxiliary cache only: all gradients flow through original Q/K/V.
            # Avoid materializing a tensor of zeros for its unused gradient.
            ctx.set_materialize_grads(False)
            ctx.mark_non_differentiable(outputs[5])
            return q, k, v, outputs[5]
        return q, k, v

    @staticmethod
    def backward(ctx, dq, dk, dv, dscaled_q=None):
        qkv, q_gamma, k_gamma, freqs, q_rstd, k_rstd = ctx.saved_tensors
        S, B, NG, _ = qkv.shape
        D = ctx.qkv_split_arg_list[1]
        if dq is None:
            dq = qkv.new_zeros((S, B, NG * (ctx.qkv_split_arg_list[0] // D), D))
        if dk is None:
            dk = qkv.new_zeros((S, B, NG, D))
        if dv is None:
            dv = qkv.new_zeros((S, B, NG, D))
        dqkv, dq_gamma, dk_gamma = qk_rmsnorm_rope_bwd_impl(
            dq,
            dk,
            dv,
            qkv,
            q_gamma,
            k_gamma,
            freqs,
            q_rstd,
            k_rstd,
            ctx.qkv_split_arg_list,
        )
        gradients = (dqkv, dq_gamma, dk_gamma, None, None, None, None)
        return gradients[: len(ctx.needs_input_grad)]


def fused_qkv_rmsnorm_rope(
    qkv: torch.Tensor,
    q_gamma: torch.Tensor,
    k_gamma: torch.Tensor,
    freqs: torch.Tensor,
    qkv_split_arg_list: Sequence[int],
    eps: float,
    return_scaled_q: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """Split packed GPT-OSS QKV while fusing per-head Q/K RMSNorm and RoPE.

    The normalization result and inverse-RoPE gradient are rounded to BF16 in
    registers, preserving the numerical boundaries of the unfused training path.
    ``return_scaled_q`` additionally emits a nondifferentiable BF16 attention Q
    cache after the original RoPE rounding; Q/K/V and their gradients are unchanged.
    """
    why = qk_rmsnorm_rope_shape_error(qkv, q_gamma, k_gamma, freqs, qkv_split_arg_list)
    if why is not None:
        raise ValueError(f"fused_qkv_rmsnorm_rope: unsupported input ({why})")
    return _FusedQKVRMSNormRoPEFunction.apply(
        qkv, q_gamma, k_gamma, freqs, qkv_split_arg_list, float(eps), bool(return_scaled_q)
    )
