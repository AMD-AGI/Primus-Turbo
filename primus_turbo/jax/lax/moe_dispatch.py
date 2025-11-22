###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


from functools import partial
from typing import Any, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from primus_turbo.jax.lax.moe_combine import _moe_combine_fwd
from primus_turbo.jax.primitive.moe_dispatch import (
    moe_cached_dispatch_p,
    moe_dispatch_p,
)

from .moe_utils import Config

__all__ = ["get_dispatch_config", "moe_dispatch"]


_default_num_sms = 32
num_ranks = 8


def get_dispatch_config(num_ranks: int = 8) -> Config:
    """
    Get a recommended dispatch config.

    Argument:
        num_ranks: the number of ranks.

    Returns:
        config: the recommended config.
    """
    global _default_num_sms
    return Config(_default_num_sms, 36, 288, 20, 128)


def moe_dispatch(
    x: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]],
    handle: Optional[Tuple] = None,
    topk_idx: Optional[jnp.ndarray] = None,
    topk_weights: Optional[jnp.ndarray] = None,
    expert_alignment: int = 1,
    num_experts: Optional[int] = None,
    config: Optional[Config] = None,
) -> Tuple[
    Union[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray], Optional[jnp.ndarray], Optional[jnp.ndarray], Tuple
]:
    return _moe_dispatch(x, handle, topk_idx, topk_weights, expert_alignment, num_experts, config)


@partial[Any](jax.custom_vjp, nondiff_argnums=(1, 2, 4, 5, 6))
def _moe_dispatch(
    x: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]],
    handle: Optional[Tuple] = None,
    topk_idx: Optional[jnp.ndarray] = None,
    topk_weights: Optional[jnp.ndarray] = None,
    expert_alignment: int = 1,
    num_experts: Optional[int] = None,
    config: Optional[Config] = None,
) -> Tuple[
    Union[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray], Optional[jnp.ndarray], Optional[jnp.ndarray], Tuple
]:
    out, _ = _moe_dispatch_fwd(x, handle, topk_idx, topk_weights, expert_alignment, num_experts, config)
    return out


def _moe_dispatch_fwd(
    x: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]],
    handle: Optional[Tuple] = None,
    topk_idx: Optional[jnp.ndarray] = None,
    topk_weights: Optional[jnp.ndarray] = None,
    expert_alignment: int = 1,
    num_experts: Optional[int] = None,
    config: Optional[Config] = None,
) -> Tuple[
    Union[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray], Optional[jnp.ndarray], Optional[jnp.ndarray], Tuple
]:
    if isinstance(x, tuple):
        x, x_scales = x
    else:
        x_scales = jnp.array([], dtype=jnp.float32)

    x.ndim == 2, "x must be a 2D array, but got {}".format(x.ndim)
    num_tokens, _ = x.shape
    num_worst_tokens = num_tokens * num_ranks

    # default config
    config = get_dispatch_config(num_ranks) if config is None else config

    if handle is not None:
        assert topk_idx is None and topk_weights is None
        (
            rank_prefix_matrix,
            channel_prefix_matrix,
            recv_channel_prefix_matrix,
            recv_src_idx,
            is_token_in_rank,
            send_head,
        ) = handle
        num_recv_tokens = recv_src_idx.size(0)
        recv_x, recv_x_scales, _, _, _ = moe_cached_dispatch_p.bind(
            x,
            x_scales,
            is_token_in_rank,
            rank_prefix_matrix,
            channel_prefix_matrix,
            num_recv_tokens,
            expert_alignment=expert_alignment,
            num_worst_tokens=num_worst_tokens,
            num_sms=config.num_sms,
            num_max_nvl_chunked_send_tokens=config.num_max_nvl_chunked_send_tokens,
            num_max_nvl_chunked_recv_tokens=config.num_max_nvl_chunked_recv_tokens,
            num_max_rdma_chunked_send_tokens=config.num_max_rdma_chunked_send_tokens,
            num_max_rdma_chunked_recv_tokens=config.num_max_rdma_chunked_recv_tokens,
        )
        return (
            (recv_x, recv_x_scales) if x_scales.size > 0 else recv_x,
            None,
            None,
            None,
        )
    else:
        assert topk_idx is not None and topk_weights is not None
        assert num_experts is not None

        (
            recv_x,
            recv_x_scales,
            recv_topk_idx,
            recv_topk_weights,
            is_token_in_rank,
            num_tokens_per_rank,
            num_tokens_per_expert,
            rank_prefix_matrix,
            channel_prefix_matrix,
            recv_channel_prefix_matrix,
            recv_src_idx,
            send_head,
        ) = moe_dispatch_p.bind(
            x,
            x_scales,
            topk_idx,
            topk_weights,
            num_experts=num_experts,
            expert_alignment=expert_alignment,
            num_worst_tokens=num_worst_tokens,
            num_sms=config.num_sms,
            num_max_nvl_chunked_send_tokens=config.num_max_nvl_chunked_send_tokens,
            num_max_nvl_chunked_recv_tokens=config.num_max_nvl_chunked_recv_tokens,
            num_max_rdma_chunked_send_tokens=config.num_max_rdma_chunked_send_tokens,
            num_max_rdma_chunked_recv_tokens=config.num_max_rdma_chunked_recv_tokens,
        )

        handle = (
            rank_prefix_matrix,
            channel_prefix_matrix,
            recv_channel_prefix_matrix,
            recv_src_idx,
            is_token_in_rank,
            send_head,
        )
        return (
            (recv_x, recv_x_scales) if x_scales.size > 0 else recv_x,
            recv_topk_idx,
            recv_topk_weights,
            handle,
        ), handle


# input: nondiff_argnums, ctx, grad
# output: input grad
def _moe_dispatch_bwd(
    handle, topk_idx, experts_alignment, num_experts, config, ctx, grad_x, grad_topk_weights
):
    (recv_x, recv_topk_weights), _ = _moe_combine_fwd(grad_x, ctx, grad_topk_weights)

    return recv_x, recv_topk_weights


_moe_dispatch.defvjp(_moe_dispatch_fwd, _moe_dispatch_bwd)
