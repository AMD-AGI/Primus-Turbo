###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from functools import partial

import jax
import jax.numpy as jnp
from jax.core import ShapedArray
from jax.extend.core import Primitive
from jax.interpreters import xla

from primus_turbo.jax.primitive import ABSTRACT_EVAL_TABLE, IMPL_TABLE, LOWERING_TABLE

# ----------------------------------------
# Step-1: Primitive Define
# ----------------------------------------
moe_dispatch_p = Primitive("moe_dispatch")
moe_cached_dispatch_p = Primitive("moe_cached_dispatch")
moe_cached_dispatch_p.multiple_results = True
moe_dispatch_p.multiple_results = True


# ----------------------------------------
# Step-2: Impl
# ----------------------------------------
IMPL_TABLE[moe_dispatch_p] = partial(xla.apply_primitive, moe_dispatch_p)
IMPL_TABLE[moe_cached_dispatch_p] = partial(xla.apply_primitive, moe_cached_dispatch_p)
# ----------------------------------------
# Step-3: Abstract eval
# ----------------------------------------
num_ranks = 8


def _get_recv_x_scale_shape(x_scales: jnp.ndarray, num_worst_tokens: int) -> ShapedArray:
    if x_scales.size > 0:
        if x_scales.ndim == 1:
            return ShapedArray((num_worst_tokens,), jnp.float32)
        else:
            return ShapedArray((num_worst_tokens, x_scales.shape[1]), jnp.float32)
    else:
        return ShapedArray(x_scales.shape, jnp.float32)


def _moe_dispatch_abstract_eval(
    x: jnp.ndarray,
    x_scales,
    topk_idx,
    topk_weights,
    num_experts: int,
    expert_alignment: int,
    num_worst_tokens: int,
    num_sms: int,
    num_max_nvl_chunked_send_tokens: int,
    num_max_nvl_chunked_recv_tokens: int,
    num_max_rdma_chunked_send_tokens: int,
    num_max_rdma_chunked_recv_tokens: int,
):

    assert x.ndim == 2, "x must be a 2D array, but got {}".format(x.ndim)
    assert topk_idx.ndim == 2, "topk_idx must be a 2D array, but got {}".format(topk_idx.ndim)

    num_tokens, hidden_size = x.shape
    num_topk = topk_idx.shape[1]
    num_channels = num_sms // 2
    recv_x_scales = _get_recv_x_scale_shape(x_scales, num_worst_tokens)
    recv_x = ShapedArray((num_worst_tokens, hidden_size), x.dtype)
    recv_topk_idx = ShapedArray((num_worst_tokens, num_topk), jnp.int64)
    recv_topk_weights = ShapedArray((num_worst_tokens, num_topk), jnp.float32)
    is_token_in_rank = ShapedArray((num_tokens, num_ranks), jnp.bool_)
    num_tokens_per_rank = ShapedArray((num_ranks,), jnp.int32)
    num_tokens_per_expert = ShapedArray((num_experts,), jnp.int32)
    rank_prefix_matrix = ShapedArray((num_ranks, num_ranks), jnp.int32)
    channel_prefix_matrix = ShapedArray((num_ranks, num_channels), jnp.int32)
    recv_channel_prefix_matrix = ShapedArray((num_ranks, num_channels), jnp.int32)
    recv_src_idx = ShapedArray((num_worst_tokens,), jnp.int32)
    send_head = ShapedArray((num_tokens, num_ranks), jnp.int32)

    return (
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
    )


def _moe_cached_dispatch_abstract_eval(
    x: jnp.ndarray,
    x_scales,
    is_token_in_rank,
    rank_prefix_matrix,
    channel_prefix_matrix,
    num_recv_tokens: int,
    expert_alignment: int,
    num_worst_tokens: int,
    num_sms: int,
    num_max_nvl_chunked_send_tokens: int,
    num_max_nvl_chunked_recv_tokens: int,
    num_max_rdma_chunked_send_tokens: int,
    num_max_rdma_chunked_recv_tokens: int,
):
    assert x.ndim == 2, "x must be a 2D array, but got {}".format(x.ndim)
    num_tokens, hidden_size = x.shape
    num_channels = num_sms // 2

    recv_x = ShapedArray((num_recv_tokens, hidden_size), x.dtype)
    recv_x_scales = _get_recv_x_scale_shape(x_scales, num_recv_tokens)
    recv_channel_prefix_matrix = ShapedArray((num_ranks, num_channels), jnp.int32)
    recv_src_idx = ShapedArray((num_ranks, num_channels), jnp.int32)
    send_head = ShapedArray((num_ranks, num_channels), jnp.int32)
    return recv_x, recv_x_scales, recv_channel_prefix_matrix, recv_src_idx, send_head


ABSTRACT_EVAL_TABLE[moe_dispatch_p] = _moe_dispatch_abstract_eval
ABSTRACT_EVAL_TABLE[moe_cached_dispatch_p] = _moe_cached_dispatch_abstract_eval

# ----------------------------------------
# Step-4: JIT Lowering
# ----------------------------------------
LOWERING_TABLE[moe_dispatch_p] = jax.ffi.ffi_lowering("moe_dispatch")
LOWERING_TABLE[moe_cached_dispatch_p] = jax.ffi.ffi_lowering("moe_cached_dispatch")
# ----------------------------------------
# Step-5: batching
# ----------------------------------------
# TODO


__all__ = ["moe_dispatch_p", "moe_cached_dispatch_p"]
