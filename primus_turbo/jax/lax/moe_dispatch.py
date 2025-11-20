###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


import jax.numpy as jnp
import jax
from primus_turbo.jax.primitive.moe_dispatch import moe_dispatch_p, moe_cached_dispatch_p
from functools import partial
from typing import Optional, Tuple, Union, Any
from typing import NamedTuple


__all__ = ["Config", "get_dispatch_config", "moe_dispatch"]


class Config(NamedTuple):
    num_sms: int
    num_max_nvl_chunked_send_tokens: int
    num_max_nvl_chunked_recv_tokens: int
    num_max_rdma_chunked_send_tokens: int
    num_max_rdma_chunked_recv_tokens: int


_default_num_sms = 32


def get_dispatch_config(num_ranks: int) -> Config:
    """
    Get a recommended dispatch config.

    Argument:
        num_ranks: the number of ranks.

    Returns:
        config: the recommended config.
    """
    global _default_num_sms

    # TODO: automatically tune
    config_map = {
        2: Config(_default_num_sms, 24, 256, 6, 128),
        4: Config(_default_num_sms, 6, 256, 6, 128),
        8: Config(_default_num_sms, 6, 256, 6, 128),
        16: Config(_default_num_sms, 36, 288, 20, 128),
        24: Config(_default_num_sms, 8, 288, 32, 128),
        32: Config(_default_num_sms, 32, 288, 32, 128),
        64: Config(_default_num_sms, 20, 288, 28, 128),
        128: Config(_default_num_sms, 20, 560, 32, 128),
        144: Config(_default_num_sms, 32, 720, 12, 128),
        160: Config(_default_num_sms, 28, 720, 12, 128),
    }
    assert num_ranks in config_map, f"Unsupported number of EP ranks: {num_ranks}"
    return config_map[num_ranks]


@partial[Any](jax.custom_vjp, nondiff_argnums=(7, ))
def moe_dispatch(x: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]],
                 num_experts: int,
                 handle: Optional[Tuple] = None,
                 topk_idx: Optional[jnp.ndarray] = None,
                 topk_weights: Optional[jnp.ndarray] = None,
                 expert_alignment: int = 1,
                 num_worst_tokens: int = 0,
                 rank: int = 0,
                 num_ranks: int = 1,
                 config: Optional[Config] = None,
                 ) -> Tuple[Union[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                            Optional[jnp.ndarray],
                            Optional[jnp.ndarray],
                            Tuple]:
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
        recv_x, recv_x_scales, _, _, _ = moe_cached_dispatch_p.bind(x,
                                                                    x_scales,
                                                                    is_token_in_rank,
                                                                    rank_prefix_matrix,
                                                                    channel_prefix_matrix,
                                                                    num_recv_tokens,
                                                                    expert_alignment=expert_alignment,
                                                                    num_worst_tokens=num_worst_tokens,
                                                                    rank=rank,
                                                                    num_ranks=num_ranks,
                                                                    num_sms=config.num_sms,
                                                                    num_max_nvl_chunked_send_tokens=config.num_max_nvl_chunked_send_tokens,
                                                                    num_max_nvl_chunked_recv_tokens=config.num_max_nvl_chunked_recv_tokens,
                                                                    num_max_rdma_chunked_send_tokens=config.num_max_rdma_chunked_send_tokens,
                                                                    num_max_rdma_chunked_recv_tokens=config.num_max_rdma_chunked_recv_tokens)
        return ((recv_x, recv_x_scales) if x_scales.size > 0 else recv_x,
                None,
                None,
                None,
                )
    else:
        assert topk_idx is not None and topk_weights is not None

        (recv_x,
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
         send_head) = moe_dispatch_p.bind(x,
                                          x_scales,
                                          topk_idx,
                                          topk_weights,
                                          num_experts=num_experts,
                                          expert_alignment=expert_alignment,
                                          num_worst_tokens=num_worst_tokens,
                                          rank=rank,
                                          num_ranks=num_ranks,
                                          num_sms=config.num_sms,
                                          num_max_nvl_chunked_send_tokens=config.num_max_nvl_chunked_send_tokens,
                                          num_max_nvl_chunked_recv_tokens=config.num_max_nvl_chunked_recv_tokens,
                                          num_max_rdma_chunked_send_tokens=config.num_max_rdma_chunked_send_tokens,
                                          num_max_rdma_chunked_recv_tokens=config.num_max_rdma_chunked_recv_tokens)

        handle = (
            rank_prefix_matrix,
            channel_prefix_matrix,
            recv_channel_prefix_matrix,
            recv_src_idx,
            is_token_in_rank,
            send_head,
        )
        return ((recv_x, recv_x_scales) if x_scales.size > 0 else recv_x,
                recv_topk_idx,
                recv_topk_weights,
                handle)


# Ref: https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.defvjp.html
# Input : same input signature as the underlying primal function
# Output: out, ctx
def _moe_dispatch_fwd(x: jnp.ndarray,
                      num_experts: int,
                      expert_alignment: int,
                      num_worst_tokens: int,
                      rank: int,
                      num_ranks: int,
                      num_sms: int,
                      num_max_nvl_chunked_send_tokens: int,
                      num_max_nvl_chunked_recv_tokens: int,
                      num_max_rdma_chunked_send_tokens: int,
                      num_max_rdma_chunked_recv_tokens: int,
                      x_scales: Optional[jnp.ndarray] = None,
                      topk_idx: Optional[jnp.ndarray] = None,
                      topk_weights: Optional[jnp.ndarray] = None,
                      cached_rank_prefix_matrix: Optional[jnp.ndarray] = None,
                      cached_channel_prefix_matrix: Optional[jnp.ndarray] = None,
                      cached_recv_channel_prefix_matrix: Optional[jnp.ndarray] = None,
                      cached_recv_src_idx: Optional[jnp.ndarray] = None,
                      cached_send_head: Optional[jnp.ndarray] = None,
                      num_tokens_per_rank: Optional[jnp.ndarray] = None,
                      num_tokens_per_expert: Optional[jnp.ndarray] = None):
    (is_token_in_rank, rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, send_head, recv_x, recv_x_scales, recv_topk_idx, recv_topk_weights) = moe_dispatch_p.bind(x,
                                                                                                                                                                                                      x_scales,
                                                                                                                                                                                                      topk_idx,
                                                                                                                                                                                                      topk_weights,
                                                                                                                                                                                                      cached_rank_prefix_matrix,
                                                                                                                                                                                                      cached_channel_prefix_matrix,
                                                                                                                                                                                                      cached_recv_channel_prefix_matrix,
                                                                                                                                                                                                      cached_recv_src_idx,
                                                                                                                                                                                                      cached_send_head,
                                                                                                                                                                                                      num_tokens_per_rank,
                                                                                                                                                                                                      num_tokens_per_expert,
                                                                                                                                                                                                      num_experts=num_experts,
                                                                                                                                                                                                      expert_alignment=expert_alignment,
                                                                                                                                                                                                      num_worst_tokens=num_worst_tokens,
                                                                                                                                                                                                      rank=rank,
                                                                                                                                                                                                      num_ranks=num_ranks,
                                                                                                                                                                                                      num_sms=num_sms,
                                                                                                                                                                                                      num_max_nvl_chunked_send_tokens=num_max_nvl_chunked_send_tokens,
                                                                                                                                                                                                      num_max_nvl_chunked_recv_tokens=num_max_nvl_chunked_recv_tokens,
                                                                                                                                                                                                      num_max_rdma_chunked_send_tokens=num_max_rdma_chunked_send_tokens,
                                                                                                                                                                                                      num_max_rdma_chunked_recv_tokens=num_max_rdma_chunked_recv_tokens)
    ctx = (
        rank_prefix_matrix,
        channel_prefix_matrix,
        recv_channel_prefix_matrix,
        recv_src_idx,
        is_token_in_rank,
        send_head,
    )
    return (recv_x, recv_x_scales, recv_topk_idx, recv_topk_weights), ctx


# input: nondiff_argnums, ctx, grad
# output: input grad
def _moe_dispatch_bwd(eps, ctx, dy):
    pass


# moe_dispatch.defvjp(
#     _moe_dispatch_fwd, _moe_dispatch_bwd)
